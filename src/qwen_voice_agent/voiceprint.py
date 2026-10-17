"""Speaker (voiceprint) embedding and identification.

Three embedding backends behind one protocol:

* :class:`MelEmbedding` — a dependency-free log-mel statistics pool.  This is a
  *deterministic placeholder*, not a real speaker model: it discriminates audio
  by its spectral envelope, which is enough to make the enrolment/identification
  plumbing testable without shipping a checkpoint, but it is **not** robust to a
  speaker saying different words.  Production deployments should set
  ``voiceprint.backend`` to ``onnx`` and point ``model_path`` at an ECAPA /
  WeSpeaker / 3D-Speaker checkpoint.
* :class:`OnnxEmbedding` — any ONNX speaker-embedding model via onnxruntime.
* :class:`ResemblyzerEmbedding` — the ``resemblyzer`` package, if installed.

:class:`VoiceprintStore` keeps enrolled speakers in a JSON file and identifies a
new utterance by cosine similarity, returning the *full ranking* rather than only
the verdict so a caller can see how close a decision was.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

from .audio import AudioFrame, resample
from .config import VoiceprintConfig

__all__ = [
    "EmbeddingBackend",
    "MelEmbedding",
    "OnnxEmbedding",
    "ResemblyzerEmbedding",
    "build_embedder",
    "VoiceprintStore",
    "IdentificationResult",
]

_EMBED_SR = 16000


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int, f_min: float = 20.0, f_max: Optional[float] = None) -> np.ndarray:
    """Triangular mel filterbank of shape ``(n_mels, n_fft//2 + 1)``."""
    f_max = f_max or sample_rate / 2.0
    mels = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center = left + 1
        if right == center:
            right = center + 1
        right = min(right, n_fft // 2)
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(1, right - center)
    return fb


def _log_mel(x: np.ndarray, sample_rate: int, n_mels: int = 40, n_fft: int = 512, hop: int = 160) -> np.ndarray:
    """Log-mel spectrogram, shape ``(n_mels, n_frames)``."""
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + max(0, (x.size - n_fft) // hop)
    frames = np.stack([x[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = np.abs(spec) ** 2
    fb = _mel_filterbank(n_mels, n_fft, sample_rate)
    mel = power @ fb.T
    return np.log(mel + 1e-6).T  # (n_mels, n_frames)


class EmbeddingBackend(Protocol):
    """Maps an audio frame to a fixed-length speaker embedding."""

    dim: int

    def embed(self, frame: AudioFrame) -> np.ndarray:  # pragma: no cover - protocol
        ...


class MelEmbedding:
    """Deterministic log-mel statistics-pool embedding (placeholder).

    The embedding is ``[mean(mel) ; std(mel)]`` across time, L2-normalised.  See
    the module docstring: this is a plumbing placeholder, not a speaker model.
    """

    def __init__(self, n_mels: int = 40) -> None:
        self.n_mels = n_mels
        self.dim = n_mels * 2

    def embed(self, frame: AudioFrame) -> np.ndarray:
        x = frame.samples
        if frame.sample_rate != _EMBED_SR:
            x = resample(x, frame.sample_rate, _EMBED_SR)
        mel = _log_mel(x, _EMBED_SR, n_mels=self.n_mels)
        vec = np.concatenate([mel.mean(axis=1), mel.std(axis=1)]).astype(np.float32)
        return _l2norm(vec)


class OnnxEmbedding:
    """Speaker embedding from any ONNX model (ECAPA / WeSpeaker / 3D-Speaker)."""

    def __init__(self, model_path: str) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OnnxEmbedding needs onnxruntime; install the 'onnx' extra") from exc
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._sess = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._in = self._sess.get_inputs()[0].name
        self._out = self._sess.get_outputs()[0].name
        self.dim = 0  # learned on first call

    def embed(self, frame: AudioFrame) -> np.ndarray:
        x = frame.samples
        if frame.sample_rate != _EMBED_SR:
            x = resample(x, frame.sample_rate, _EMBED_SR)
        feat = _log_mel(x, _EMBED_SR, n_mels=80).T[None, :, :].astype(np.float32)
        out = self._sess.run([self._out], {self._in: feat})[0]
        vec = np.asarray(out).reshape(-1).astype(np.float32)
        self.dim = vec.size
        return _l2norm(vec)


class ResemblyzerEmbedding:
    """Embedding via the ``resemblyzer`` package (optional)."""

    def __init__(self) -> None:
        try:
            from resemblyzer import VoiceEncoder  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ResemblyzerEmbedding needs the 'resemblyzer' package") from exc
        self._enc = VoiceEncoder()
        self.dim = 256

    def embed(self, frame: AudioFrame) -> np.ndarray:
        x = frame.samples
        if frame.sample_rate != _EMBED_SR:
            x = resample(x, frame.sample_rate, _EMBED_SR)
        vec = self._enc.embed_utterance(x).astype(np.float32)
        return _l2norm(vec)


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-12 else v.astype(np.float32)


def build_embedder(cfg: VoiceprintConfig) -> EmbeddingBackend:
    """Instantiate the embedding backend named by ``cfg.backend``."""
    if cfg.backend == "onnx":
        if not cfg.model_path:
            raise ValueError("voiceprint.backend='onnx' requires model_path")
        return OnnxEmbedding(cfg.model_path)
    if cfg.backend == "resemblyzer":
        return ResemblyzerEmbedding()
    return MelEmbedding()


@dataclass
class IdentificationResult:
    """Result of identifying an utterance against the enrolled speakers."""

    person_id: Optional[str]
    name: Optional[str]
    score: float
    accepted: bool
    ranking: List[Tuple[str, float]]  # [(person_id, cosine), ...] best first
    embedding: Optional[np.ndarray] = None


class VoiceprintStore:
    """Persistent registry of enrolled speakers and their embeddings.

    Storage is a single JSON file (embeddings base64-encoded float32 arrays).
    Enrolment keeps a *running mean* of a speaker's embeddings so several
    enrolment clips sharpen one profile instead of overwriting each other.
    """

    def __init__(self, cfg: VoiceprintConfig, embedder: Optional[EmbeddingBackend] = None) -> None:
        self.cfg = cfg
        self.embedder = embedder or build_embedder(cfg)
        self._speakers: Dict[str, Dict[str, object]] = {}
        self.path = cfg.store_path
        if self.path and os.path.exists(self.path):
            self.load(self.path)

    # ---- enrolment -----------------------------------------------------
    def enroll(self, person_id: str, name: str, frame: AudioFrame) -> np.ndarray:
        """Enrol (or update) a speaker; returns the stored embedding."""
        emb = self.embedder.embed(frame)
        entry = self._speakers.get(person_id)
        if entry is None:
            self._speakers[person_id] = {
                "name": name,
                "dim": int(emb.size),
                "count": 1,
                "embedding": emb,
            }
        else:
            count = int(entry["count"])
            prev = np.asarray(entry["embedding"], dtype=np.float32)
            if prev.size != emb.size:
                raise ValueError(
                    f"embedding dim changed for {person_id}: {prev.size} -> {emb.size}"
                )
            merged = (prev * count + emb) / (count + 1)
            entry["name"] = name
            entry["count"] = count + 1
            entry["embedding"] = _l2norm(merged)
        return np.asarray(self._speakers[person_id]["embedding"], dtype=np.float32)

    def remove(self, person_id: str) -> bool:
        return self._speakers.pop(person_id, None) is not None

    def speakers(self) -> List[str]:
        return list(self._speakers)

    # ---- identification ------------------------------------------------
    def identify(self, frame: AudioFrame) -> IdentificationResult:
        """Identify a speaker; returns the full ranking and the accept verdict."""
        emb = self.embedder.embed(frame)
        ranking: List[Tuple[str, float]] = []
        for pid, entry in self._speakers.items():
            ref = np.asarray(entry["embedding"], dtype=np.float32)
            if ref.size != emb.size:
                continue
            ranking.append((pid, float(np.dot(emb, ref))))  # both L2-normalised
        ranking.sort(key=lambda kv: kv[1], reverse=True)
        if not ranking:
            return IdentificationResult(None, None, 0.0, False, [], embedding=emb)
        best_id, best_score = ranking[0]
        accepted = best_score >= self.cfg.threshold
        name = str(self._speakers[best_id]["name"]) if accepted else None
        return IdentificationResult(
            person_id=best_id if accepted else None,
            name=name,
            score=best_score,
            accepted=accepted,
            ranking=ranking,
            embedding=emb,
        )

    # ---- persistence ---------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for pid, entry in self._speakers.items():
            emb = np.asarray(entry["embedding"], dtype=np.float32)
            out[pid] = {
                "name": entry["name"],
                "count": entry["count"],
                "embedding_b64": base64.b64encode(emb.tobytes()).decode("ascii"),
                "dim": int(emb.size),
            }
        return {"version": 1, "speakers": out}

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path:
            raise ValueError("no store_path configured")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        self.path = path

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._speakers = {}
        for pid, entry in data.get("speakers", {}).items():
            raw = base64.b64decode(entry["embedding_b64"])
            emb = np.frombuffer(raw, dtype=np.float32).copy()
            self._speakers[pid] = {
                "name": entry.get("name", pid),
                "count": int(entry.get("count", 1)),
                "embedding": emb,
            }
        self.path = path
