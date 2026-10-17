"""ASR backends.

One protocol, four implementations, so the same pipeline runs against a real
OpenAI-compatible ``/v1/audio/transcriptions`` server, a local ``transformers``
checkpoint, a fixed demo script, or a deterministic mock.  The ``http`` backend
is the production path (this is how Qwen ASR is normally reached); the others
exist so the whole agent is exercisable without a model or a GPU.

Every backend raises :class:`AsrError` with a human-readable message rather than
hanging, because a 24/7 agent that silently blocks on a dead ASR server is worse
than one that fails loudly.
"""

from __future__ import annotations

import hashlib
import io
import wave
from typing import List, Optional, Protocol

import numpy as np

from .audio import AudioFrame, write_wav
from .config import ASRConfig

__all__ = ["AsrBackend", "AsrError", "HttpAsr", "TransformersAsr", "ScriptedAsr", "MockAsr", "build_asr"]


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


class HttpAsr:
    """OpenAI-compatible ``POST {base_url}/audio/transcriptions``."""

    def __init__(self, cfg: ASRConfig) -> None:
        self.cfg = cfg

    def transcribe(self, audio: AudioFrame) -> str:
        try:
            import httpx  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise AsrError("HttpAsr needs httpx installed") from exc
        url = self.cfg.base_url.rstrip("/") + "/audio/transcriptions"
        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        data = {"model": self.cfg.model}
        if self.cfg.language:
            data["language"] = self.cfg.language
        files = {"file": ("audio.wav", _wav_bytes(audio), "audio/wav")}
        try:
            with httpx.Client(timeout=self.cfg.timeout_s) as client:
                resp = client.post(url, headers=headers, data=data, files=files)
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
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
    if cfg.backend == "http":
        return HttpAsr(cfg)
    if cfg.backend == "transformers":
        return TransformersAsr(cfg)
    if cfg.backend == "scripted":
        return ScriptedAsr(cfg)
    return MockAsr()
