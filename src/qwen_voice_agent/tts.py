"""TTS backends.

Five implementations behind one protocol:

* :class:`EdgeTts` — Microsoft Edge's online neural voices through the
  ``edge-tts`` package.  Synthesised in-process (no subprocess, no temp file) and
  decoded to float32 with ``soundfile``; the best Chinese quality available here.
* :class:`HttpTts` — OpenAI-compatible ``POST {base_url}/audio/speech``.
* :class:`CommandTts` — run an external CLI (piper / a local Qwen TTS wrapper)
  that writes a WAV file, then read it back.  This wires a real TTS without
  adding a Python dependency.
* :class:`MockTts` — a deterministic tone derived from the text, so pipeline and
  barge-in tests have real audio to play.
* :class:`NullTts` — silence; the "headless, no speaker" default.

A 24/7 agent must be able to stop speaking the instant the user barges in, so
``synthesize`` is always a *bounded* call (``timeout_s``) that returns audio the
caller can discard.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import subprocess
import tempfile
import wave
from typing import Optional, Protocol

import numpy as np

from .audio import AudioFrame, _decode_pcm, read_wav, resample
from .config import TTSConfig

__all__ = [
    "TtsBackend",
    "TtsError",
    "EdgeTts",
    "PiperTts",
    "HttpTts",
    "CommandTts",
    "MockTts",
    "NullTts",
    "build_tts",
]


class TtsError(RuntimeError):
    """Raised when a TTS backend cannot produce audio."""


class TtsBackend(Protocol):
    def synthesize(self, text: str) -> AudioFrame:  # pragma: no cover - protocol
        ...


def _bytes_audio_format(data: bytes) -> Optional[int]:
    """WAV ``audioFormat`` code from raw bytes (None if not a RIFF/WAVE header)."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    i = 12
    while i + 8 <= len(data):
        cid = data[i : i + 4]
        size = int.from_bytes(data[i + 4 : i + 8], "little")
        if cid == b"fmt ":
            return int.from_bytes(data[i + 8 : i + 10], "little")
        i += 8 + size + (size & 1)
    return None


def _parse_wav_bytes(data: bytes, fallback_sr: int) -> AudioFrame:
    """Decode WAV bytes into an AudioFrame; fall back to raw int16 PCM.

    Uses the RIFF ``audioFormat`` code to tell float32 from int32, so a float32
    body from a TTS server is not decoded as int32 garbage.
    """
    fmt = _bytes_audio_format(data)
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            n_ch = wf.getnchannels()
            width = wf.getsampwidth()
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        arr = _decode_pcm(raw, width, is_float=(fmt == 3))
        if n_ch > 1:
            arr = arr.reshape(-1, n_ch).mean(axis=1)
        return AudioFrame(arr, sr)
    except Exception:
        # Not a WAV container: assume little-endian int16 PCM at the configured rate.
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return AudioFrame(arr, fallback_sr)


class EdgeTts:
    """Speech from Microsoft Edge's online voices, synthesised in-process.

    ``edge-tts`` streams MP3 over websockets; the stream is collected and decoded
    with ``soundfile`` (libsndfile handles MP3), then resampled to the agent's
    rate so the sink sees one consistent format.  Doing this in-process rather
    than through ``CommandTts`` avoids a subprocess and a temp file per utterance.

    The trade-off is honest: it needs the network at speak time.  A deployment
    that must speak with the network down should use ``command`` with a local
    engine, or a served Qwen TTS through ``http``.
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg

    def synthesize(self, text: str) -> AudioFrame:
        text = (text or "").strip()
        if not text:
            return AudioFrame(np.zeros(0, dtype=np.float32), self.cfg.sample_rate)
        try:
            import edge_tts  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TtsError("EdgeTts needs the 'edge-tts' package") from exc

        async def collect() -> bytes:
            buf = bytearray()
            async for chunk in edge_tts.Communicate(text, self.cfg.voice).stream():
                if chunk.get("type") == "audio":
                    buf += chunk["data"]
            return bytes(buf)

        try:
            # The worker thread has no running loop, so asyncio.run is safe here.
            mp3 = asyncio.run(asyncio.wait_for(collect(), timeout=self.cfg.timeout_s))
        except asyncio.TimeoutError as exc:
            raise TtsError(f"edge-tts timed out after {self.cfg.timeout_s}s") from exc
        except Exception as exc:
            raise TtsError(f"edge-tts synthesis failed: {exc}") from exc
        if not mp3:
            raise TtsError("edge-tts returned no audio")

        try:
            import soundfile as sf  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TtsError("EdgeTts needs 'soundfile' to decode the MP3 stream") from exc
        try:
            data, sr = sf.read(io.BytesIO(mp3), dtype="float32")
        except Exception as exc:
            raise TtsError(f"could not decode edge-tts audio: {exc}") from exc
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != self.cfg.sample_rate:
            data = resample(data, sr, self.cfg.sample_rate)
        return AudioFrame(data, self.cfg.sample_rate)


class PiperTts:
    """Speech from a local ``piper`` voice (ONNX), synthesised in-process.

    Piper runs a small VITS model through ``onnxruntime``: one short Chinese
    sentence costs ~25-70 ms on this CPU, against ~1.5 s for edge-tts's network
    round trip.  That difference *is* the perceived latency of the agent — the
    user hears the reply roughly two seconds sooner — so piper is the default
    when a voice is installed, and ``edge`` stays available for the best
    naturalness.

    The voice is loaded once and kept: ``PiperVoice.load`` costs ~1 s, which must
    not land inside a turn.  Output is int16 PCM at the voice's own rate (22050
    for ``zh_CN-huayan-medium``); the sink plays whatever rate the frame carries.
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._voice: object = None

    def _load(self):
        if self._voice is not None:
            return self._voice
        try:
            from piper import PiperVoice  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TtsError("PiperTts needs the 'piper-tts' package") from exc
        path = self.cfg.model_path
        if not path:
            raise TtsError("PiperTts needs tts.model_path (a piper voice .onnx)")
        if not os.path.isfile(path):
            # Without this the loader downloads the voice from the internet and
            # fails with a network error, which reads like a piper bug.
            raise TtsError(f"no such piper voice file: {path}")
        config_path = self.cfg.model_config or f"{path}.json"
        try:
            self._voice = PiperVoice.load(path, config_path=config_path)
        except Exception as exc:
            raise TtsError(f"could not load piper voice {path}: {exc}") from exc
        return self._voice

    def warmup(self) -> None:
        """Load the voice ONNX now; ~1 s, which must not land inside a turn."""
        self._load()

    def synthesize(self, text: str) -> AudioFrame:
        text = (text or "").strip()
        if not text:
            return AudioFrame(np.zeros(0, dtype=np.float32), self.cfg.sample_rate)
        voice = self._load()
        syn_config = None
        if self.cfg.length_scale is not None:
            try:
                from piper import SynthesisConfig  # type: ignore

                syn_config = SynthesisConfig(length_scale=float(self.cfg.length_scale))
            except Exception as exc:  # pragma: no cover
                raise TtsError(f"could not build piper synthesis config: {exc}") from exc

        pieces = []
        try:
            for chunk in voice.synthesize(text, syn_config):
                raw = getattr(chunk, "audio_int16_bytes", None)
                if raw:
                    pieces.append(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
        except Exception as exc:
            raise TtsError(f"piper synthesis failed: {exc}") from exc
        if not pieces:
            raise TtsError("piper returned no audio")

        data = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        sr = int(voice.config.sample_rate)
        if sr != self.cfg.sample_rate:
            data = resample(data, sr, self.cfg.sample_rate)
            sr = self.cfg.sample_rate
        return AudioFrame(data, sr)

    def close(self) -> None:
        self._voice = None


class HttpTts:
    """OpenAI-compatible ``POST {base_url}/audio/speech`` returning WAV bytes."""

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._client = None

    def _http(self):
        if self._client is None:
            try:
                import httpx  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise TtsError("HttpTts needs httpx installed") from exc
            self._client = httpx.Client(timeout=self.cfg.timeout_s)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def synthesize(self, text: str) -> AudioFrame:
        client = self._http()
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
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.content
        except Exception as exc:
            raise TtsError(f"TTS request to {url} failed: {exc}") from exc
        return _parse_wav_bytes(data, self.cfg.sample_rate)


class CommandTts:
    """Run an external CLI that writes a WAV file, then read it back.

    ``cfg.command`` is a template containing ``{out}`` only, e.g.
    ``piper --model x.onnx -f {out}``.  The text is passed on **stdin**, never
    interpolated into the command, so LLM/ASR text cannot inject shell syntax.
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg

    def synthesize(self, text: str) -> AudioFrame:
        if not self.cfg.command:
            raise TtsError("CommandTts requires a 'command' template")
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            try:
                cmd = self.cfg.command.format(out=out_path)
            except KeyError as exc:
                raise TtsError(
                    f"unsupported placeholder in TTS command template: {exc}; "
                    "use {out} only — the text is passed on stdin"
                )
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
    if cfg.backend == "edge":
        return EdgeTts(cfg)
    if cfg.backend == "piper":
        return PiperTts(cfg)
    if cfg.backend == "http":
        return HttpTts(cfg)
    if cfg.backend == "command":
        return CommandTts(cfg)
    if cfg.backend == "null":
        return NullTts(cfg)
    return MockTts(cfg)
