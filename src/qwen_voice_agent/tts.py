"""TTS backends.

Four implementations behind one protocol:

* :class:`HttpTts` — OpenAI-compatible ``POST {base_url}/audio/speech``.
* :class:`CommandTts` — run an external CLI (piper / edge-tts + ffmpeg / a local
  Qwen TTS wrapper) that writes a WAV file, then read it back.  This wires a real
  TTS without adding a Python dependency.
* :class:`MockTts` — a deterministic tone derived from the text, so pipeline and
  barge-in tests have real audio to play.
* :class:`NullTts` — silence; the "headless, no speaker" default.

A 24/7 agent must be able to stop speaking the instant the user barges in, so
``synthesize`` is always a *bounded* call (``timeout_s``) that returns audio the
caller can discard.
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tempfile
import wave
from typing import Optional, Protocol

import numpy as np

from .audio import AudioFrame, read_wav
from .config import TTSConfig

__all__ = ["TtsBackend", "TtsError", "HttpTts", "CommandTts", "MockTts", "NullTts", "build_tts"]


class TtsError(RuntimeError):
    """Raised when a TTS backend cannot produce audio."""


class TtsBackend(Protocol):
    def synthesize(self, text: str) -> AudioFrame:  # pragma: no cover - protocol
        ...


def _parse_wav_bytes(data: bytes, fallback_sr: int) -> AudioFrame:
    """Decode WAV bytes into an AudioFrame; fall back to raw int16 PCM."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            n_ch, width, sr, n = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
            raw = wf.readframes(n)
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width, np.int16)
        arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if width == 1:
            arr = (arr - 128.0) / 128.0
        else:
            arr = arr / float(2 ** (8 * width - 1))
        if n_ch > 1:
            arr = arr.reshape(-1, n_ch).mean(axis=1)
        return AudioFrame(arr, sr)
    except Exception:
        # Not a WAV container: assume little-endian int16 PCM at the configured rate.
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return AudioFrame(arr, fallback_sr)


class HttpTts:
    """OpenAI-compatible ``POST {base_url}/audio/speech`` returning WAV bytes."""

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg

    def synthesize(self, text: str) -> AudioFrame:
        try:
            import httpx  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TtsError("HttpTts needs httpx installed") from exc
        url = self.cfg.base_url.rstrip("/") + "/audio/speech"
        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        payload = {
            "model": self.cfg.model,
            "voice": self.cfg.voice,
            "input": text,
            "response_format": "wav",
        }
        try:
            with httpx.Client(timeout=self.cfg.timeout_s) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.content
        except httpx.HTTPError as exc:
            raise TtsError(f"TTS request to {url} failed: {exc}") from exc
        return _parse_wav_bytes(data, self.cfg.sample_rate)


class CommandTts:
    """Run an external CLI that writes a WAV file, then read it back.

    ``cfg.command`` is a template containing ``{out}`` (and optionally
    ``{text}``), e.g. ``piper --model x.onnx -f {out}`` with the text on stdin.
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg

    def synthesize(self, text: str) -> AudioFrame:
        if not self.cfg.command:
            raise TtsError("CommandTts requires a 'command' template")
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            cmd = self.cfg.command.format(out=out_path, text=text)
            proc = subprocess.run(
                cmd,
                shell=True,
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=self.cfg.timeout_s,
            )
            if proc.returncode != 0:
                raise TtsError(
                    f"TTS command failed ({proc.returncode}): {proc.stderr.decode('utf-8', 'ignore')[:400]}"
                )
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                raise TtsError("TTS command produced no output file")
            samples, sr = read_wav(out_path)
            return AudioFrame(samples, sr)
        except subprocess.TimeoutExpired as exc:
            raise TtsError(f"TTS command timed out after {self.cfg.timeout_s}s") from exc
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass


class MockTts:
    """Deterministic tone whose length scales with the text.

    Two different strings give two different tones, and the same string always
    gives the same tone, so tests can assert on playback timing and content
    without a model.
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg

    def synthesize(self, text: str) -> AudioFrame:
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        freq = 180.0 + (digest[0] % 40) * 8.0          # 180..492 Hz
        dur = min(4.0, 0.3 + 0.045 * len(text))        # grows with the sentence
        sr = self.cfg.sample_rate
        n = int(round(dur * sr))
        t = np.arange(n, dtype=np.float32) / sr
        wave_ = 0.25 * np.sin(2 * np.pi * freq * t)
        # Gentle fade in/out so concatenated utterances do not click.
        fade = min(n // 20, int(0.02 * sr))
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            wave_[:fade] *= ramp
            wave_[-fade:] *= ramp[::-1]
        return AudioFrame(wave_.astype(np.float32), sr)


class NullTts:
    """Silence of a plausible length — the headless default."""

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg

    def synthesize(self, text: str) -> AudioFrame:
        dur = min(4.0, 0.3 + 0.045 * len(text))
        n = int(round(dur * self.cfg.sample_rate))
        return AudioFrame(np.zeros(n, dtype=np.float32), self.cfg.sample_rate)


def build_tts(cfg: TTSConfig) -> TtsBackend:
    """Instantiate the TTS backend named by ``cfg.backend``."""
    if cfg.backend == "http":
        return HttpTts(cfg)
    if cfg.backend == "command":
        return CommandTts(cfg)
    if cfg.backend == "null":
        return NullTts(cfg)
    return MockTts(cfg)
