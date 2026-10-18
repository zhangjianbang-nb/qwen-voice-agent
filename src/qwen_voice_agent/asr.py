"""ASR backends.

One protocol, five implementations, so the same pipeline runs against a **local**
FunASR checkpoint, a real OpenAI-compatible ``/v1/audio/transcriptions`` server, a
``transformers`` checkpoint, a fixed demo script, or a deterministic mock.

The ``funasr`` backend is what this deployment runs.  It is preferred over
``http`` here for two reasons that matter to a 24/7 agent: it cannot fail because
the network did, and it is faster than any round trip (SenseVoice transcribes
3.7 s of speech in ~0.1 s on this CPU).

Every backend raises :class:`AsrError` with a human-readable message rather than
hanging, because a 24/7 agent that silently blocks on a dead ASR server is worse
than one that fails loudly.
"""

from __future__ import annotations

import hashlib
import io
import re
import wave
from typing import List, Optional, Protocol

import numpy as np

from .audio import AudioFrame, resample, write_wav
from .config import ASRConfig

__all__ = [
    "AsrBackend",
    "AsrError",
    "FunAsrAsr",
    "HttpAsr",
    "TransformersAsr",
    "ScriptedAsr",
    "MockAsr",
    "build_asr",
]

# SenseVoice returns its annotations inline, e.g.
#   "<|zh|><|NEUTRAL|><|Speech|><|withitn|>其实我觉得人是一个多重的社会属性的动物。"
# The agent speaks the text, never the tags, so they are stripped rather than
# rendered (FunASR's own `rich_transcription_postprocess` turns them into emoji,
# which is worse for a voice agent: it would be read out loud).
_SENSEVOICE_TAG = re.compile(r"<\|[^|>]*\|>")


def clean_sensevoice_text(raw: str) -> str:
    """Strip SenseVoice's inline annotation tags, leaving only the transcript."""
    return _SENSEVOICE_TAG.sub("", raw).strip()


class AsrError(RuntimeError):
    """Raised when an ASR backend cannot produce a transcript."""


class AsrBackend(Protocol):
    def transcribe(self, audio: AudioFrame) -> str:  # pragma: no cover - protocol
        ...


def _wav_bytes(frame: AudioFrame) -> bytes:
    """Serialise a frame to 16-bit PCM WAV bytes for multipart upload."""
    clipped = np.clip(frame.samples, -1.0, 1.0)
    ints = np.round(clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(frame.sample_rate)
        wf.writeframes(ints.tobytes())
    return buf.getvalue()


class FunAsrAsr:
    """Local ASR through a FunASR checkpoint (SenseVoice), no server, no network.

    The model is loaded once and cached in :mod:`qwen_voice_agent.local_models`,
    so the ~1 s load is paid at the first utterance rather than at every one.  The
    audio is handed to the model as a numpy array, which avoids writing a temp WAV
    per turn — on a 24/7 process that is a disk write every few seconds that buys
    nothing.
    """

    def __init__(self, cfg: ASRConfig) -> None:
        self.cfg = cfg
        self._model = None

    def _ensure(self):
        if self._model is None:
            from .local_models import load_funasr, sensevoice_dir

            path = self.cfg.model_path or sensevoice_dir()
            if not path:
                raise AsrError(
                    "no local FunASR ASR checkpoint: set asr.model_path, or "
                    "download 'iic/SenseVoiceSmall' into the ModelScope cache"
                )
            try:
                self._model = load_funasr(path, device=self.cfg.device)
            except Exception as exc:
                raise AsrError(f"could not load the FunASR checkpoint at {path}: {exc}") from exc
        return self._model

    def warmup(self) -> None:
        """Load the checkpoint now instead of inside the first utterance."""
        self._ensure()

    def transcribe(self, audio: AudioFrame) -> str:
        model = self._ensure()
        x = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
        if audio.sample_rate != 16000:
            x = resample(x, audio.sample_rate, 16000)
        if x.size == 0:
            return ""
        try:
            out = model.generate(
                input=x,
                fs=16000,
                cache={},
                language=self.cfg.language or "auto",
                use_itn=self.cfg.use_itn,
                batch_size_s=60,
            )
        except Exception as exc:
            raise AsrError(f"FunASR transcription failed: {exc}") from exc
        if not out:
            return ""
        return clean_sensevoice_text(str(out[0].get("text", "")))


class HttpAsr:
    """OpenAI-compatible ``POST {base_url}/audio/transcriptions``."""

    def __init__(self, cfg: ASRConfig) -> None:
        self.cfg = cfg
        self._client = None

    def _http(self):
        """One client for the process: a per-turn client pays a TCP handshake
        every turn, which is pure latency in a loop that runs every few seconds."""
        if self._client is None:
            try:
                import httpx  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise AsrError("HttpAsr needs httpx installed") from exc
            self._client = httpx.Client(timeout=self.cfg.timeout_s)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def transcribe(self, audio: AudioFrame) -> str:
        client = self._http()
        url = self.cfg.base_url.rstrip("/") + "/audio/transcriptions"
        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        data = {"model": self.cfg.model}
        if self.cfg.language:
            data["language"] = self.cfg.language
        files = {"file": ("audio.wav", _wav_bytes(audio), "audio/wav")}
        try:
            resp = client.post(url, headers=headers, data=data, files=files)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise AsrError(f"ASR request to {url} failed: {exc}") from exc
        text = payload.get("text")
        if not isinstance(text, str):
            raise AsrError(f"ASR response missing 'text': {payload!r}")
        return text


class TransformersAsr:
    """Local ASR through a ``transformers`` automatic-speech-recognition pipeline."""

    def __init__(self, cfg: ASRConfig) -> None:
        self.cfg = cfg
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            try:
                from transformers import pipeline  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise AsrError("TransformersAsr needs the 'transformers' package") from exc
            self._pipe = pipeline("automatic-speech-recognition", model=self.cfg.model)
        return self._pipe

    def transcribe(self, audio: AudioFrame) -> str:
        pipe = self._ensure()
        out = pipe({"array": audio.samples.astype(np.float32), "sampling_rate": audio.sample_rate})
        return str(out.get("text", "")).strip()


class ScriptedAsr:
    """Pops the next line from a fixed script — for demos and deterministic tests."""

    def __init__(self, cfg: ASRConfig) -> None:
        self._script: List[str] = list(cfg.script) or ["hello", "are you there", "goodbye"]
        self._i = 0

    def transcribe(self, audio: AudioFrame) -> str:
        line = self._script[self._i % len(self._script)]
        self._i += 1
        return line


class MockAsr:
    """Deterministic transcript derived from the audio content itself.

    The same audio always yields the same text, which lets tests assert on
    transcripts without a model while still varying with the input.
    """

    def transcribe(self, audio: AudioFrame) -> str:
        digest = hashlib.sha1(audio.samples.astype(np.float32).tobytes()).hexdigest()[:8]
        return f"<mock:{digest}:{audio.duration_s:.2f}s>"


def build_asr(cfg: ASRConfig) -> AsrBackend:
    """Instantiate the ASR backend named by ``cfg.backend``."""
    if cfg.backend == "funasr":
        return FunAsrAsr(cfg)
    if cfg.backend == "http":
        return HttpAsr(cfg)
    if cfg.backend == "transformers":
        return TransformersAsr(cfg)
    if cfg.backend == "scripted":
        return ScriptedAsr(cfg)
    return MockAsr()
