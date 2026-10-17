"""Voice activity detection and utterance segmentation.

Two interchangeable backends sit behind one protocol:

* :class:`EnergyVad` — a dependency-free adaptive-noise-floor detector.  Its
  threshold is *relative* to the recent noise floor rather than an absolute dB
  level, so one config survives both a quiet room and a noisy one.
* :class:`SileroVad` — the Silero VAD v5 ONNX model through ``onnxruntime``.
  Note its contract: the LSTM state is the pair ``h``/``c`` of shape
  ``(2, 1, 64)``, not a single ``state`` tensor.  Feeding ``{"input", "state",
  "sr"}`` fails.

:class:`Segmenter` turns a frame stream into utterances using hysteresis: a turn
opens after ``start_frames`` consecutive speech frames and closes after
``end_frames`` consecutive silence frames.  ``end_frames`` is deliberately larger
than ``start_frames`` because a turn that closes early is far cheaper than one
that never closes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import numpy as np

from .audio import AudioFrame, RingBuffer, concat
from .config import VadConfig

__all__ = ["VadBackend", "EnergyVad", "SileroVad", "Utterance", "Segmenter", "build_vad"]

# Silero v5 consumes a fixed 512-sample window at 16 kHz (256 at 8 kHz).
_SILERO_WINDOW = {8000: 256, 16000: 512}


class VadBackend(Protocol):
    """Anything that can score a frame's probability of being speech."""

    def speech_prob(self, frame: AudioFrame) -> float:  # pragma: no cover - protocol
        ...

    def reset(self) -> None:  # pragma: no cover - protocol
        ...


def _rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


class EnergyVad:
    """Adaptive-noise-floor energy detector.

    The noise floor follows the quietest frames with an exponential moving
    average; speech is decided by whether the current frame rises
    ``energy_threshold_db`` above that floor.  ``energy_range_db`` bounds how far
    the floor may drift so a long silence does not make the detector deaf.
    """

    def __init__(self, cfg: VadConfig) -> None:
        self.cfg = cfg
        self._floor_db: Optional[float] = None

    def reset(self) -> None:
        self._floor_db = None

    def speech_prob(self, frame: AudioFrame) -> float:
        level = _rms_db(frame.samples)
        if self._floor_db is None:
            self._floor_db = level
            return 0.0
        # Adapt the floor toward the current level, but only when the frame is
        # quiet relative to the floor (so loud speech does not raise the floor).
        if level <= self._floor_db + self.cfg.energy_threshold_db * 0.5:
            self._floor_db = (
                self.cfg.noise_adapt * self._floor_db + (1.0 - self.cfg.noise_adapt) * level
            )
        # Clamp the floor's distance from the very first observation is handled
        # implicitly by the EMA; energy_range_db caps the decision margin.
        margin = min(self.cfg.energy_range_db, max(0.0, level - self._floor_db))
        return float(min(1.0, margin / max(1e-6, self.cfg.energy_threshold_db)))


class SileroVad:
    """Silero VAD v5 via onnxruntime.

    The model expects 512-sample windows at 16 kHz; shorter frames are buffered
    internally until a full window is available, at which point the recurrent
    state advances.  ``speech_prob`` returns the most recent window's probability.
    """

    def __init__(self, cfg: VadConfig, model_path: Optional[str] = None) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "SileroVad needs onnxruntime; install the 'onnx' extra"
            ) from exc
        path = model_path or cfg.model_path
        if not path:
            raise ValueError("SileroVad requires a model_path")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self._sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._in_names = [i.name for i in self._sess.get_inputs()]
        self._out_names = [o.name for o in self._sess.get_outputs()]
        self._buf = np.zeros(0, dtype=np.float32)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._last = 0.0

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._last = 0.0

    def _feed(self, window: np.ndarray, sample_rate: int) -> float:
        inp = window.reshape(1, -1).astype(np.float32)
        sr = np.array([sample_rate], dtype=np.int64)
        feed = {}
        for name in self._in_names:
            low = name.lower()
            if "sr" in low or "rate" in low:
                feed[name] = sr
            elif low == "h" or low.startswith("h_"):
                feed[name] = self._h
            elif low == "c" or low.startswith("c_"):
                feed[name] = self._c
            elif "input" in low or low in ("x", "waveform"):
                feed[name] = inp
            else:
                # Refuse rather than silently feed audio into an unknown slot
                # (e.g. a v4 combined `state` tensor) — a shape mismatch here
                # would surface far from the cause.
                raise RuntimeError(
                    f"unsupported Silero input {name!r}; this build expects the "
                    "h/c recurrent-state variant with inputs (input, sr, h, c)"
                )
        outs = self._sess.run(self._out_names, feed)
        by_name = dict(zip(self._out_names, outs))
        prob = 0.0
        for name, val in by_name.items():
            low = name.lower()
            if low.startswith("output") or low == "prob" or low == "out":
                prob = float(np.asarray(val).reshape(-1)[0])
        for name, val in by_name.items():
            low = name.lower()
            if low.startswith("hn"):
                self._h = np.asarray(val, dtype=np.float32)
            elif low.startswith("cn"):
                self._c = np.asarray(val, dtype=np.float32)
        return prob

    def speech_prob(self, frame: AudioFrame) -> float:
        sr = frame.sample_rate
        window = _SILERO_WINDOW.get(sr, 512)
        self._buf = np.concatenate([self._buf, frame.samples.astype(np.float32)])
        while self._buf.size >= window:
            chunk, self._buf = self._buf[:window], self._buf[window:]
            self._last = self._feed(chunk, sr)
        return self._last


def build_vad(cfg: VadConfig) -> VadBackend:
    """Instantiate the backend named by ``cfg.backend``."""
    if cfg.backend == "silero":
        return SileroVad(cfg)
    return EnergyVad(cfg)


@dataclass
class Utterance:
    """A closed speech segment with the pre-roll that preceded it."""

    samples: np.ndarray
    sample_rate: int
    start_s: float
    end_s: float
    peak_prob: float = 0.0
    mean_prob: float = 0.0
    frames: int = 0

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    def as_frame(self) -> AudioFrame:
        return AudioFrame(self.samples, self.sample_rate)


class Segmenter:
    """Frame stream -> utterances, with hysteresis and a bounded pre-roll.

    State machine per frame:

    * ``SILENCE``: count consecutive speech frames; open a turn at
      ``start_frames``.  The pre-roll ring is always being filled so the opening
      consonant is not clipped.
    * ``SPEECH``: count consecutive silence frames; close at ``end_frames``.  A
      segment is force-closed at ``max_speech_ms`` and dropped below
      ``min_speech_ms``.
    """

    def __init__(self, cfg: VadConfig, sample_rate: int, frame_samples: int) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        pre_roll_samples = int(round(sample_rate * cfg.pre_roll_ms / 1000.0))
        self._pre = RingBuffer(max(pre_roll_samples, frame_samples), sample_rate)
        self._speech_run = 0
        self._silence_run = 0
        self._active = False
        self._collected: List[np.ndarray] = []
        self._probs: List[float] = []
        self._start_s = 0.0
        self._t = 0.0  # seconds of audio consumed so far
        self._frame_dur = frame_samples / float(sample_rate)

    def reset(self) -> None:
        self._pre.clear()
        self._speech_run = self._silence_run = 0
        self._active = False
        self._collected = []
        self._probs = []
        self._t = 0.0

    def push(self, frame: AudioFrame, prob: float) -> Optional[Utterance]:
        """Feed one frame and its speech probability; return a closed utterance."""
        self._pre.push(frame.samples)
        self._t += frame.duration_s
        thr = self.cfg.speech_prob
        is_speech = prob >= thr

        if not self._active:
            if is_speech:
                self._speech_run += 1
                if self._speech_run >= self.cfg.start_frames:
                    self._active = True
                    self._collected = [self._pre.snapshot()]
                    self._probs = [prob]
                    self._start_s = max(0.0, self._t - self._frame_dur * self.cfg.start_frames)
                    self._silence_run = 0
            else:
                self._speech_run = 0
            return None

        # active
        self._collected.append(frame.samples)
        self._probs.append(prob)
        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1

        max_frames = max(1, int(round(self.cfg.max_speech_ms / 1000.0 / self._frame_dur)))
        if self._silence_run >= self.cfg.end_frames or len(self._collected) >= max_frames:
            return self._close()

        return None

    def flush(self) -> Optional[Utterance]:
        """Close any in-progress utterance (call at end of stream)."""
        if self._active and self._collected:
            return self._close()
        return None

    def _close(self) -> Optional[Utterance]:
        samples = np.concatenate(self._collected) if self._collected else np.zeros(0, np.float32)
        probs = self._probs or [0.0]
        end_s = self._t
        # Derive the start from the sample count so start_s/end_s and the audio
        # length agree (the segment includes the pre-roll).
        start_s = max(0.0, end_s - samples.size / float(self.sample_rate))
        utt = Utterance(
            samples=np.ascontiguousarray(samples, dtype=np.float32),
            sample_rate=self.sample_rate,
            start_s=start_s,
            end_s=end_s,
            peak_prob=float(max(probs)),
            mean_prob=float(np.mean(probs)),
            frames=len(self._collected),
        )
        self._active = False
        self._collected = []
        self._probs = []
        self._speech_run = self._silence_run = 0
        min_samples = int(round(self.cfg.min_speech_ms / 1000.0 * self.sample_rate))
        if utt.samples.size < min_samples:
            return None
        return utt
