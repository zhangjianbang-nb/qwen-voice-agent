"""Audio sources and sinks.

A source is anything that yields a blocking stream of :class:`AudioFrame`; a sink
is anything that can play one.  The sources cover the ways this agent is actually
run:

* :class:`FileSource` — replay a WAV (offline tests, batch transcription);
* :class:`ArecordSource` — live capture through the ALSA ``arecord`` binary;
* :class:`MicSource` — live capture through ``sounddevice``, when that extra is
  installed;
* :class:`SyntheticSource` — a scripted speech/silence timeline, so the full-duplex
  loop and barge-in are testable with no audio hardware at all;
* :class:`NullSource` — endless silence; the correct default when the machine has
  no capture device, so a 24/7 process still starts and stays healthy.

Sinks: :class:`AplaySink` (``aplay``), :class:`SpeakerSink` (``sounddevice``),
:class:`FileSink`, :class:`NullSink`.

**Playback is cancellable.**  ``play(audio, cancel=...)`` is part of the sink
protocol because "cancel generation" is not the same thing as "stop talking": a
sink that dumps the whole utterance into the audio device and blocks cannot be
interrupted, so the agent would keep speaking over the user who just barged in.
:class:`AplaySink` therefore writes in short chunks, paces them to real time, and
kills the player the moment ``cancel`` is set — playback stops within one chunk
(100 ms) instead of at the end of the sentence.

The ``arecord`` / ``aplay`` paths exist because they need no Python package at
all: this box has the ALSA tools but no ``sounddevice``, and a deployment that
cannot capture audio is not a voice agent.  Both are tried first only when the
``sounddevice`` extra is genuinely importable.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Iterator, List, Optional, Protocol, Tuple

import numpy as np

from .audio import AudioFrame, iter_frames, read_wav, write_wav
from .config import AudioConfig

__all__ = [
    "AudioSource",
    "AudioSink",
    "FileSource",
    "MicSource",
    "ArecordSource",
    "SyntheticSource",
    "NullSource",
    "NullSink",
    "FileSink",
    "SpeakerSink",
    "AplaySink",
    "build_source",
    "build_sink",
]

_S16_SCALE = 32768.0


class AudioSource(Protocol):
    #: True when this source is a live microphone, i.e. when it can hear the
    #: agent's own playback and the echo guard is worth applying.
    live: bool

    def frames(self) -> Iterator[AudioFrame]:  # pragma: no cover - protocol
        ...

    def close(self) -> None:  # pragma: no cover - protocol
        ...


class AudioSink(Protocol):
    #: True when this sink actually makes sound in the room, i.e. when the
    #: microphone can hear the agent and the echo guard is worth arming.
    acoustic: bool

    def play(self, audio: AudioFrame, cancel=None) -> None:  # pragma: no cover - protocol
        ...

    def close(self) -> None:  # pragma: no cover - protocol
        ...


def _read_exact(stream, n: int) -> bytes:
    """Read exactly ``n`` bytes from a pipe, or fewer if the writer closes."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


class FileSource:
    """Replay a WAV file as frames, optionally looping.

    The session paces frame delivery to real time, so a replay behaves like a live
    stream rather than being consumed as fast as the CPU allows.
    """

    live = False

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


class ArecordSource:
    """Live capture through the ALSA ``arecord`` binary.

    Chosen over ``sounddevice`` here because it needs no Python package: the ALSA
    tools are part of the base system, and a capture path that fails to import is
    a capture path that does not run.  ``arecord`` writes raw little-endian 16-bit
    mono, which is converted to the float32 convention the rest of the package
    uses.
    """

    live = True

    def __init__(self, cfg: AudioConfig, device: Optional[str] = None, block_frames: int = 1) -> None:
        self.cfg = cfg
        self.device = device or cfg.input_device
        self.block_frames = max(1, block_frames)
        self._proc: Optional[subprocess.Popen] = None

    def _cmd(self) -> List[str]:
        cmd = [
            "arecord",
            "-q",
            "-t", "raw",
            "-f", "S16_LE",
            "-r", str(self.cfg.sample_rate),
            "-c", "1",
        ]
        if self.device:
            cmd += ["-D", self.device]
        return cmd

    def frames(self) -> Iterator[AudioFrame]:
        if not shutil.which("arecord"):
            raise RuntimeError("ArecordSource needs the 'arecord' binary (alsa-utils)")
        block = self.cfg.frame_samples * self.block_frames
        need = block * 2  # 16-bit samples
        self._proc = subprocess.Popen(
            self._cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        try:
            while True:
                raw = _read_exact(self._proc.stdout, need)
                if len(raw) < need:
                    # The recorder died (device unplugged, or taken by someone
                    # else).  Failing loudly beats pretending to listen.
                    raise RuntimeError(
                        f"arecord exited while capturing (device={self.device or 'default'})"
                    )
                arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / _S16_SCALE
                yield AudioFrame(arr * self.cfg.input_gain, self.cfg.sample_rate)
        finally:
            self.close()

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


class MicSource:
    """Live capture through ``sounddevice`` (requires the 'mic' extra)."""

    live = True

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
                yield AudioFrame(mono * self.cfg.input_gain, self.cfg.sample_rate)
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

    live = False

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

    live = False

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

    acoustic = False

    def play(self, audio: AudioFrame, cancel=None) -> None:
        pass

    def close(self) -> None:
        pass


class FileSink:
    """Append every played frame to a growing WAV (offline verification)."""

    acoustic = False

    def __init__(self, path: str) -> None:
        self.path = path
        self._chunks: List[np.ndarray] = []
        self._sr: Optional[int] = None

    def play(self, audio: AudioFrame, cancel=None) -> None:
        if audio.samples.size == 0:
            return
        if self._sr is None:
            self._sr = audio.sample_rate
        elif audio.sample_rate != self._sr:
            raise ValueError("FileSink requires a single sample rate")
        self._chunks.append(audio.samples)

    def close(self) -> None:
        if self._chunks and self._sr:
            write_wav(self.path, np.concatenate(self._chunks), self._sr)


class SpeakerSink:
    """Playback through ``sounddevice`` (requires the 'mic' extra).

    ``sounddevice.play(blocking=True)`` cannot be interrupted mid-buffer, so this
    sink is only suitable where playback need not be cancellable; prefer
    :class:`AplaySink` for a full-duplex agent.
    """

    acoustic = True

    def __init__(self) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("SpeakerSink needs sounddevice; install the 'mic' extra") from exc
        self._sd = sd

    def play(self, audio: AudioFrame, cancel=None) -> None:
        self._sd.play(audio.samples, audio.sample_rate, blocking=True)

    def close(self) -> None:
        self._sd.stop()


class AplaySink:
    """Playback through the ALSA ``aplay`` binary, in cancellable chunks.

    The utterance is written to the player in ``chunk_ms`` slices, paced to real
    time so the device buffer never holds more than a slice.  That is what makes
    barge-in work: when ``cancel`` is set, the player is **killed** (SIGTERM is
    ignored while ``aplay`` drains, which used to leave ~2 s of speech still
    playing) and audio stops within one chunk, rather than after the whole
    sentence has drained through the buffer.  The device buffer is also capped
    (``buffer_time_us``) so the tail after the last write is short.
    """

    acoustic = True

    def __init__(
        self,
        device: Optional[str] = None,
        chunk_ms: float = 100.0,
        buffer_time_us: int = 200_000,
    ) -> None:
        if not shutil.which("aplay"):
            raise RuntimeError("AplaySink needs the 'aplay' binary (alsa-utils)")
        self.device = device
        self.chunk_ms = max(10.0, float(chunk_ms))
        # A small device buffer keeps the post-utterance drain short (~200 ms
        # instead of the ~500 ms default) while still staying ahead of the paced
        # 100 ms writes, so it cannot underrun.
        self.buffer_time_us = int(buffer_time_us)
        self._proc: Optional[subprocess.Popen] = None

    def _cmd(self, sample_rate: int) -> List[str]:
        cmd = [
            "aplay",
            "-q",
            "-B", str(self.buffer_time_us),
            "-t", "raw",
            "-f", "S16_LE",
            "-r", str(int(sample_rate)),
            "-c", "1",
        ]
        if self.device:
            cmd += ["-D", self.device]
        return cmd

    def play(self, audio: AudioFrame, cancel=None) -> None:
        if audio.samples.size == 0:
            return
        sr = int(audio.sample_rate)
        ints = np.round(np.clip(audio.samples, -1.0, 1.0) * 32767.0).astype("<i2")
        chunk = max(1, int(sr * self.chunk_ms / 1000.0))
        proc = subprocess.Popen(
            self._cmd(sr),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._proc = proc
        started = time.monotonic()
        try:
            for i, start in enumerate(range(0, ints.size, chunk)):
                if cancel is not None and cancel.is_set():
                    break
                try:
                    proc.stdin.write(ints[start : start + chunk].tobytes())
                except (BrokenPipeError, OSError) as exc:
                    detail = b""
                    if proc.stderr is not None:
                        try:
                            detail = proc.stderr.read() or b""
                        except Exception:
                            detail = b""
                    raise RuntimeError(
                        f"aplay failed on device {self.device or 'default'}: "
                        f"{detail.decode('utf-8', 'ignore').strip() or exc}"
                    ) from exc
                # Pace the writes so the device buffer stays ~one chunk deep.
                target = started + (i + 1) * chunk / float(sr)
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        finally:
            self._proc = None
            interrupted = cancel is not None and cancel.is_set()
            try:
                if interrupted:
                    # SIGTERM is not enough: `aplay` does not exit while draining
                    # its buffer, so a barge-in would leave ~2 s of speech still
                    # playing.  SIGKILL stops it within one chunk.
                    proc.kill()
                else:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except Exception:
                    pass

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                pass


def build_source(
    cfg: AudioConfig,
    kind: str = "null",
    path: Optional[str] = None,
    loop: bool = True,
) -> AudioSource:
    """Factory for a source by kind (``file`` / ``mic`` / ``arecord`` / ``null``).

    ``mic`` prefers ``sounddevice`` when it is importable and falls back to
    ``arecord``, so the live path works on a box with no extra installed.

    ``loop`` applies to the ``file`` source: looping suits a continuous live-style
    run, while ``loop=False`` is what a batch pass over a recording wants — the
    stream ends and the last reply is allowed to finish.
    """
    if kind == "file":
        if not path:
            raise ValueError("file source requires a path")
        return FileSource(path, cfg.frame_samples, loop=loop)
    if kind == "arecord":
        return ArecordSource(cfg)
    if kind == "mic":
        try:
            return MicSource(cfg)
        except RuntimeError:
            return ArecordSource(cfg)
    return NullSource(cfg.sample_rate, cfg.frame_samples)


def build_sink(kind: str = "null", path: Optional[str] = None, device: Optional[str] = None) -> AudioSink:
    """Factory for a sink by kind (``file`` / ``speaker`` / ``aplay`` / ``null``).

    ``speaker`` prefers ``sounddevice`` when importable and otherwise uses
    ``aplay`` — which is also the cancellable one.
    """
    if kind == "file":
        if not path:
            raise ValueError("file sink requires a path")
        return FileSink(path)
    if kind == "aplay":
        return AplaySink(device=device)
    if kind == "speaker":
        try:
            return SpeakerSink()
        except RuntimeError:
            return AplaySink(device=device)
    return NullSink()
