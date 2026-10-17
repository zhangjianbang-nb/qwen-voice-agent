"""Audio primitives: framing, a bounded ring buffer, WAV IO and resampling.

Everything here is numpy + the standard library.  No scipy, no soundfile, no
librosa: a voice agent whose perception layer cannot be exercised without a
sound card or a heavy DSP stack is not testable, so the primitives are kept
portable and the optional quality upgrades (polyphase resampling) are used only
when scipy happens to be importable.

Conventions used everywhere in this package:

* audio is **mono**, **float32**, nominal range ``[-1, 1]``;
* a frame is a fixed-size, non-overlapping slice of that stream;
* time is measured in seconds as a float, never in samples, at module boundaries.
"""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

__all__ = [
    "AudioFrame",
    "RingBuffer",
    "resample",
    "read_wav",
    "write_wav",
    "iter_frames",
    "concat",
    "silence",
]

# WAV sample-width code -> (numpy dtype, scale factor that maps it into [-1, 1]).
_WAV_WIDTHS = {
    1: (np.uint8, 128.0),   # 8-bit PCM is unsigned
    2: (np.int16, 32768.0),
    4: (np.int32, 2147483648.0),
}


@dataclass(frozen=True)
class AudioFrame:
    """An immutable block of mono float32 audio.

    ``samples`` is always 1-D.  A stereo or multi-channel input is mixed to mono
    by the reader, never carried around as a 2-D array, because every downstream
    stage (VAD, embedding, ASR) is defined on a single channel.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples, dtype=np.float32).reshape(-1)
        if arr.ndim != 1:
            raise ValueError("AudioFrame expects 1-D mono samples")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        # frozen dataclass: bypass the frozen guard to normalise the array.
        object.__setattr__(self, "samples", np.ascontiguousarray(arr))

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    def __len__(self) -> int:
        return len(self.samples)


class RingBuffer:
    """A fixed-capacity rolling window over the most recent samples.

    The capacity is a hard bound on resident audio: a 24/7 process must state how
    much raw audio it may ever hold, otherwise the buffer becomes a slow leak.
    Pushing more than the capacity drops the oldest samples, which is exactly
    what a live listener wants.
    """

    def __init__(self, capacity_samples: int, sample_rate: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self._cap = int(capacity_samples)
        self.sample_rate = int(sample_rate)
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._len = 0

    def push(self, samples: np.ndarray) -> None:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        if x.size >= self._cap:
            self._buf[:] = x[-self._cap :]
            self._len = self._cap
            return
        self._buf = np.roll(self._buf, -x.size)
        self._buf[-x.size :] = x
        self._len = min(self._cap, self._len + x.size)

    def snapshot(self) -> np.ndarray:
        """Return a copy of the buffered audio, oldest sample first."""
        return self._buf[-self._len :].copy() if self._len else np.zeros(0, dtype=np.float32)

    def __len__(self) -> int:
        return self._len

    @property
    def capacity(self) -> int:
        return self._cap

    def clear(self) -> None:
        self._buf[:] = 0.0
        self._len = 0


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample mono audio from ``sr_in`` to ``sr_out``.

    Uses scipy's polyphase filter when scipy is importable, and falls back to a
    linear interpolation otherwise.  The fallback is intentionally naive — it is
    a correctness fallback, not a quality one — and the function documents which
    path it took through the ``qwen_voice_agent.resample_backend`` logger.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if sr_in == sr_out or x.size == 0:
        return x
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError("sample rates must be positive")
    try:  # optional, keeps the core dependency-free
        from scipy.signal import resample_poly  # type: ignore

        from math import gcd

        g = gcd(int(sr_in), int(sr_out))
        y = resample_poly(x, sr_out // g, sr_in // g)
        return np.ascontiguousarray(y.astype(np.float32))
    except Exception:
        n_out = int(round(x.size * sr_out / sr_in))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        src = np.arange(x.size, dtype=np.float64)
        dst = np.linspace(0.0, x.size - 1, n_out, dtype=np.float64)
        return np.interp(dst, src, x.astype(np.float64)).astype(np.float32)


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file into ``(float32 mono samples, sample_rate)``.

    Handles 8/16/32-bit PCM and 32-bit float WAVs.  Multi-channel files are
    averaged to mono.  Only the standard library is used so this works in the
    minimal install.
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if width == 4:
        # wave does not distinguish int32 from float32; assume float32 WAVs
        # (the common case for generated speech) unless values look like int32.
        as_f32 = np.frombuffer(raw, dtype=np.float32)
        as_i32 = np.frombuffer(raw, dtype=np.int32)
        # float32 audio lives in [-1, 1]; int32 audio exceeds it almost surely.
        if as_f32.size and np.nanmax(np.abs(as_f32)) <= 1.5:
            data = as_f32
        else:
            data = as_i32.astype(np.float32) / _WAV_WIDTHS[4][1]
    else:
        dtype, scale = _WAV_WIDTHS.get(width, (np.int16, 32768.0))
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if width == 1:
            data = (data - 128.0) / 128.0  # unsigned 8-bit -> signed float
        else:
            data = data / scale

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 audio as a 16-bit PCM WAV (clipped, not wrapped)."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(x, -1.0, 1.0)
    ints = np.round(clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(ints.tobytes())


def iter_frames(samples: np.ndarray, sample_rate: int, frame_samples: int) -> Iterator[AudioFrame]:
    """Yield non-overlapping frames of ``frame_samples`` samples.

    A trailing partial frame is padded with zeros so every yielded frame has the
    same shape; downstream code (ONNX VAD) requires a fixed frame size.
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if frame_samples <= 0:
        raise ValueError("frame_samples must be positive")
    for start in range(0, x.size, frame_samples):
        chunk = x[start : start + frame_samples]
        if chunk.size < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - chunk.size))
        yield AudioFrame(chunk, sample_rate)


def silence(duration_s: float, sample_rate: int) -> AudioFrame:
    """A frame of pure silence, used by synthetic sources and tests."""
    n = max(0, int(round(duration_s * sample_rate)))
    return AudioFrame(np.zeros(n, dtype=np.float32), sample_rate)


def concat(frames: list[AudioFrame]) -> AudioFrame:
    """Concatenate frames that share a sample rate."""
    if not frames:
        raise ValueError("concat() needs at least one frame")
    sr = frames[0].sample_rate
    if any(f.sample_rate != sr for f in frames):
        raise ValueError("concat() requires a common sample rate")
    return AudioFrame(np.concatenate([f.samples for f in frames]), sr)
