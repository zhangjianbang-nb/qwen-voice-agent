"""The full-duplex turn pipeline.

This is the heart of the agent.  The design goal is *full duplex*: while the
agent is generating (ASR -> LLM -> TTS -> playback) it must keep listening, so
that the user can interrupt — "barge in" — and the agent stops talking at once.

The mechanism:

* :meth:`Pipeline.feed` is called once per audio frame by the driving loop.  It
  never blocks on generation.
* When a :class:`~qwen_voice_agent.vad.Segmenter` closes an utterance, a
  **generation worker thread** is started with its *own* cancel event; the
  pipeline enters ``GENERATING`` and the main loop keeps feeding frames.
* While ``GENERATING``, if the VAD reports ``start_frames`` consecutive speech
  frames, that worker's cancel event is set: the worker stops at its next
  checkpoint, the pipeline returns to ``LISTENING``, and the frame that triggered
  the barge-in starts a fresh utterance.
* A worker that outlives ``session.worker_timeout_s`` is cancelled too, so a
  stuck ASR/LLM call cannot freeze the loop forever.
* A finished worker appends its :class:`Turn` to a thread-safe queue; ``feed``
  drains it on the next call, so generation and listening are decoupled.

Each worker gets its own cancel event (not a shared one) precisely because a
second utterance can close before the first worker finishes: a shared event would
be replaced and the older worker would never observe the cancellation.

Every completed turn is handed to the :class:`~qwen_voice_agent.memory.MemoryManager`
exactly once, on the thread that drains it.

Note on interruption granularity: cancel is checked *between* stages and before
``sink.play``.  A backend whose ``play`` blocks (e.g. ``sounddevice``) cannot be
interrupted mid-sample by this layer; such a sink must stop itself when its own
cancellation is requested.  The ``null``/``file`` sinks used in tests return
immediately.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .audio import AudioFrame
from .config import VoiceAgentConfig
from .llm import LlmBackend, render_system_prompt
from .memory import MemoryManager
from .sources import AudioSink, AudioSource
from .vad import Segmenter, Utterance, VadBackend

__all__ = ["Turn", "Pipeline", "PipelineState"]


class PipelineState:
    LISTENING = "listening"
    GENERATING = "generating"


@dataclass
class Turn:
    """One completed exchange: a user utterance and the agent's reply."""

    index: int
    person_id: Optional[str]
    person_name: Optional[str]
    user_text: str
    reply: str
    embedding: Optional[np.ndarray] = None
    identification_score: Optional[float] = None
    started_s: float = 0.0
    ended_s: float = 0.0
    interrupted: bool = False
    audio: Optional[AudioFrame] = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.ended_s - self.started_s)


class Pipeline:
    """Full-duplex: listen while generating, stop on barge-in or worker timeout."""

    def __init__(
        self,
        cfg: VoiceAgentConfig,
        *,
        vad: VadBackend,
        segmenter: Segmenter,
        asr,
        tts,
        llm: LlmBackend,
        memory: MemoryManager,
        sink: AudioSink,
        voiceprints=None,
    ) -> None:
        self.cfg = cfg
        self.vad = vad
        self.segmenter = segmenter
        self.asr = asr
        self.tts = tts
        self.llm = llm
        self.memory = memory
        self.sink = sink
        self.voiceprints = voiceprints

        self._state = PipelineState.LISTENING
        self._barge_run = 0
        self._barge_frames: List[Tuple[AudioFrame, float]] = []
        self._workers: List[threading.Thread] = []
        self._cancel = threading.Event()      # the *current* worker's event
        self._worker_started = 0.0
        self._completed: List[Turn] = []
        self._errors: List[BaseException] = []
        self._lock = threading.Lock()
        self._turn_index = 0

    # ---- public --------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    def feed(self, frame: AudioFrame) -> List[Turn]:
        """Feed one frame; return any turns completed as a result (usually none)."""
        prob = self.vad.speech_prob(frame)

        if self._state == PipelineState.GENERATING:
            if prob >= self.cfg.vad.speech_prob:
                self._barge_run += 1
                self._barge_frames.append((frame, prob))
                if self._barge_run >= self.cfg.vad.start_frames:
                    # Barge-in: cancel the worker, then replay every frame that
                    # confirmed the interruption into the segmenter so the
                    # opening syllables of the user's new utterance are not lost.
                    self._cancel.set()
                    self._state = PipelineState.LISTENING
                    self._barge_run = 0
                    self.segmenter.reset()
                    pending, self._barge_frames = self._barge_frames, []
                    for f, p in pending:
                        self._maybe_start(f, p)
            else:
                self._barge_run = 0
                self._barge_frames = []
            self._check_worker_timeout()
            if self._has_completed():
                self._state = PipelineState.LISTENING
        else:
            self._maybe_start(frame, prob)

        return self._drain()

    def run(self, source: AudioSource) -> Iterator[Turn]:
        """Drive the pipeline from a source, yielding each completed turn."""
        for frame in source.frames():
            for turn in self.feed(frame):
                yield turn
        # Drain anything still in flight at end of stream.
        self._join_workers()
        for turn in self._drain():
            yield turn
        tail = self.segmenter.flush()
        if tail is not None:
            self._process_now(tail)
            for turn in self._drain():
                yield turn

    def close(self) -> None:
        """Cancel any in-flight workers, wait for them, flush turns, close the sink."""
        self._cancel.set()
        self._join_workers()
        try:
            self._drain()
        except Exception:
            pass
        try:
            self.sink.close()
        except Exception:
            pass

    # ---- state helpers -------------------------------------------------
    def _maybe_start(self, frame: AudioFrame, prob: float) -> None:
        utt = self.segmenter.push(frame, prob)
        if utt is not None:
            self._start_worker(utt)
            self._state = PipelineState.GENERATING
            self._barge_run = 0

    def _check_worker_timeout(self) -> None:
        alive = [t for t in self._workers if t.is_alive()]
        if alive and (time.monotonic() - self._worker_started) > self.cfg.session.worker_timeout_s:
            self._cancel.set()

    # ---- worker --------------------------------------------------------
    def _start_worker(self, utt: Utterance) -> None:
        # Reap finished threads so the list does not grow unboundedly.
        self._workers = [t for t in self._workers if t.is_alive()]
        cancel = threading.Event()
        self._cancel = cancel
        self._worker_started = time.monotonic()
        t = threading.Thread(
            target=self._worker_body, args=(utt, cancel), name="qva-generate", daemon=True
        )
        self._workers.append(t)
        t.start()

    def _worker_body(self, utt: Utterance, cancel: threading.Event) -> None:
        try:
            transcript = self.asr.transcribe(utt.as_frame())
            ident = None
            if self.voiceprints is not None and not cancel.is_set():
                ident = self.voiceprints.identify(utt.as_frame())
            person_id = ident.person_id if ident and ident.accepted else None
            person_name = ident.name if ident and ident.accepted else "there"

            reply = ""
            audio = None
            interrupted = cancel.is_set()
            if not interrupted:
                context = self.memory.context_block(person_name, transcript)
                system_prompt = render_system_prompt(self.cfg.llm, person_name, context)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript},
                ]
                reply = self.llm.chat(messages)
                interrupted = cancel.is_set()
            if not interrupted:
                audio = self.tts.synthesize(reply)
                interrupted = cancel.is_set()
            if not interrupted:
                self.sink.play(audio)

            turn = Turn(
                index=self._next_index(),
                person_id=person_id,
                person_name=person_name,
                user_text=transcript,
                reply=reply,
                embedding=(ident.embedding if ident else None),
                identification_score=(ident.score if ident else None),
                started_s=utt.start_s,
                ended_s=utt.end_s,
                interrupted=interrupted,
                audio=audio,
            )
            with self._lock:
                self._completed.append(turn)
        except BaseException as exc:  # surfaced to the main thread on drain
            with self._lock:
                self._errors.append(exc)

    def _next_index(self) -> int:
        with self._lock:
            idx = self._turn_index
            self._turn_index += 1
        return idx

    # ---- drain / sync --------------------------------------------------
    def _has_completed(self) -> bool:
        with self._lock:
            return bool(self._completed) or bool(self._errors)

    def _drain(self) -> List[Turn]:
        with self._lock:
            turns = self._completed
            self._completed = []
            errors = self._errors
            self._errors = []
        if errors:
            raise errors[0]
        for turn in turns:
            self.memory.observe(turn)
        return turns

    def _join_workers(self, timeout: Optional[float] = None) -> None:
        if timeout is None:
            timeout = max(5.0, self.cfg.session.worker_timeout_s)
        for t in list(self._workers):
            if t.is_alive():
                t.join(timeout=timeout)
        self._workers = [t for t in self._workers if t.is_alive()]

    def _process_now(self, utt: Utterance) -> None:
        """Process an utterance synchronously (used only for the tail flush)."""
        cancel = threading.Event()
        self._cancel = cancel
        self._worker_body(utt, cancel)
