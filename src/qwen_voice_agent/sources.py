"""Audio sources and sinks.

A source is anything that yields a blocking stream of :class:`AudioFrame`; a sink
is anything that can play one.  The four sources cover the ways this agent is
actually run:

* :class:`FileSource` — replay a WAV (offline tests, batch transcription);
* :class:`MicSource` — a live microphone via ``sounddevice`` (the production path,
  and the only one that needs an extra);
* :class:`SyntheticSource` — a scripted speech/silence timeline, so the full-duplex
  loop and barge-in are testable with no audio hardware at all;
* :class:`NullSource` — endless silence; the correct default when the machine has
  no capture device, so a 24/7 process still starts and stays healthy.

Because this box currently has no capture device, ``NullSource`` / ``FileSource``
are the paths that actually run here; ``MicSource`` is written but exercised only
by ``doctor``.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Protocol, Tuple

import numpy as np

from .audio import AudioFrame, iter_frames, read_wav, write_wav
from .config import AudioConfig

__all__ = [
    "AudioSource",
    "AudioSink",
    "FileSource",
    "MicSource",
    "SyntheticSource",
    "NullSource",
    "NullSink",
    "FileSink",
    "SpeakerSink",
    "build_source",
    "build_sink",
]


class AudioSource(Protocol):
    def frames(self) -> Iterator[AudioFrame]:  # pragma: no cover - protocol
        ...

    def close(self) -> None:  # pragma: no cover - protocol
        ...


class AudioSink(Protocol):
    def play(self, audio: AudioFrame) -> None:  # pragma: no cover - protocol
        ...

    def close(self) -> None:  # pragma: no cover - protocol
        ...


class FileSource:
    """Replay a WAV file as frames, optionally looping."""

    def __init__(self, path: str, frame_samples: int, loop: bool = False) -> None:
        self.samples, self.sample_rate = read_wav(path)
        self.frame_samples = frame_samples
        self.loop = loop

    def frames(self) -> Iterator[AudioFrame]:
        if self.samples.size == 0:
            raise ValueError("FileSource: the WAV contains no audio samples")
        while True:
            yield from iter_frames(self.samples, self.sample_rate, self.frame_samples)
            if not self.loop:
                return

    def close(self) -> None:
        pass


class MicSource:
    """Live capture through ``sounddevice`` (requires the 'mic' extra)."""

    def __init__(self, cfg: AudioConfig, block_frames: int = 1) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("MicSource needs sounddevice; install the 'mic' extra") from exc
        self._sd = sd
        self.cfg = cfg
        self.block_frames = max(1, block_frames)
        self._stream = None

    def frames(self) -> Iterator[AudioFrame]:
        block = self.cfg.frame_samples * self.block_frames
        self._stream = self._sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block,
        )
        self._stream.start()
        try:
            while True:
                data, overflowed = self._stream.read(block)
                if overflowed:
                    # Dropping samples is preferable to blocking; the VAD will see
                    # the gap as silence, which is a safe failure mode.
                    pass
                mono = np.asarray(data, dtype=np.float32).reshape(-1)
                yield AudioFrame(mono, self.cfg.sample_rate)
        finally:
            self.close()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class SyntheticSource:
    """A scripted speech/silence timeline, looped forever.

    Each segment is ``(is_speech, duration_s)``.  Speech segments are a harmonic
    tone loud enough for the energy VAD to fire; silence segments are exact
    silence.  This is what makes the full-duplex loop and barge-in testable on a
    machine with no microphone.
    """

    def __init__(
        self,
        segments: List[Tuple[bool, float]],
        sample_rate: int = 16000,
        frame_samples: int = 320,
        amplitude: float = 0.3,
        tone_hz: float = 220.0,
        loop: bool = True,
    ) -> None:
        self.segments = list(segments)
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.amplitude = amplitude
        self.tone_hz = tone_hz
        self.loop = loop
        self._samples = self._render()

    def _render(self) -> np.ndarray:
        parts: List[np.ndarray] = []
        for is_speech, dur in self.segments:
            n = int(round(dur * self.sample_rate))
            if is_speech:
                t = np.arange(n, dtype=np.float32) / self.sample_rate
                parts.append((self.amplitude * np.sin(2 * np.pi * self.tone_hz * t)).astype(np.float32))
            else:
                parts.append(np.zeros(n, dtype=np.float32))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def frames(self) -> Iterator[AudioFrame]:
        if self._samples.size == 0:
            raise ValueError("SyntheticSource: no segments were configured")
        while True:
            yield from iter_frames(self._samples, self.sample_rate, self.frame_samples)
            if not self.loop:
                return

    def close(self) -> None:
        pass


class NullSource:
    """Endless silence — the default when no capture device exists."""

    def __init__(self, sample_rate: int = 16000, frame_samples: int = 320) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples

    def frames(self) -> Iterator[AudioFrame]:
        while True:
            yield AudioFrame(np.zeros(self.frame_samples, dtype=np.float32), self.sample_rate)

    def close(self) -> None:
        pass


class NullSink:
    """Discards audio — the headless default."""

    def play(self, audio: AudioFrame) -> None:
        pass

    def close(self) -> None:
        pass


class FileSink:
    """Append every played frame to a growing WAV (offline verification)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._chunks: List[np.ndarray] = []
        self._sr: Optional[int] = None

    def play(self, audio: AudioFrame) -> None:
        if self._sr is None:
            self._sr = audio.sample_rate
        elif audio.sample_rate != self._sr:
            raise ValueError("FileSink requires a single sample rate")
        self._chunks.append(audio.samples)

    def close(self) -> None:
        if self._chunks and self._sr:
            write_wav(self.path, np.concatenate(self._chunks), self._sr)


class SpeakerSink:
    """Playback through ``sounddevice`` (requires the 'mic' extra)."""

    def __init__(self) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("SpeakerSink needs sounddevice; install the 'mic' extra") from exc
        self._sd = sd

    def play(self, audio: AudioFrame) -> None:
        self._sd.play(audio.samples, audio.sample_rate, blocking=True)

    def close(self) -> None:
        self._sd.stop()


def build_source(cfg: AudioConfig, kind: str = "null", path: Optional[str] = None) -> AudioSource:
    """Factory for a source by kind (``file`` / ``mic`` / ``null``)."""
    if kind == "file":
        if not path:
            raise ValueError("file source requires a path")
        return FileSource(path, cfg.frame_samples, loop=True)
    if kind == "mic":
        return MicSource(cfg)
    return NullSource(cfg.sample_rate, cfg.frame_samples)


def build_sink(kind: str = "null", path: Optional[str] = None) -> AudioSink:
    """Factory for a sink by kind (``file`` / ``speaker`` / ``null``)."""
    if kind == "file":
        if not path:
            raise ValueError("file sink requires a path")
        return FileSink(path)
    if kind == "speaker":
        return SpeakerSink()
    return NullSink()
