"""The full-duplex turn pipeline.

This is the heart of the agent.  The design goal is *full duplex*: while the
agent is generating (ASR -> LLM -> TTS -> playback) it must keep listening, so
that the user can interrupt — "barge in" — and the agent stops talking at once.

The mechanism:

* :meth:`Pipeline.feed` is called once per audio frame by the driving loop.  It
  never blocks on generation.
* When a :class:`~qwen_voice_agent.vad.Segmenter` closes an utterance, a
  **generation worker thread** is started; the pipeline enters ``GENERATING`` and
  the main loop keeps feeding frames.
* While ``GENERATING``, if the VAD reports ``start_frames`` consecutive speech
  frames, the worker's cancel event is set: the worker stops at its next
  checkpoint, the pipeline returns to ``LISTENING``, and the frame that triggered
  the barge-in starts a fresh utterance.
* A finished worker appends its :class:`Turn` to a thread-safe queue; ``feed``
  drains it on the next call, so generation and listening are decoupled.

Every completed turn is handed to the :class:`~qwen_voice_agent.memory.MemoryManager`
exactly once, on the thread that drains it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

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
    """Full-duplex: listen while generating, stop instantly on barge-in."""

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
        self._worker: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._completed: List[Turn] = []
        self._lock = threading.Lock()
        self._turn_index = 0
        self._error: Optional[BaseException] = None

    # ---- public --------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    def feed(self, frame: AudioFrame) -> List[Turn]:
        """Feed one frame; return any turns completed as a result (usually none)."""
        prob = self.vad.speech_prob(frame)
        if self._state == PipelineState.GENERATING:
            return self._feed_generating(frame, prob)
        return self._feed_listening(frame, prob)

    def run(self, source: AudioSource) -> Iterator[Turn]:
        """Drive the pipeline from a source, yielding each completed turn."""
        for frame in source.frames():
            for turn in self.feed(frame):
                yield turn
        # Drain anything still in flight at end of stream.
        self._join_worker()
        for turn in self._drain():
            yield turn
        tail = self.segmenter.flush()
        if tail is not None:
            for turn in self._process_now(tail):
                yield turn

    def close(self) -> None:
        """Cancel any in-flight worker and wait for it, then close the sink."""
        self._cancel.set()
        self._join_worker()
        try:
            self.sink.close()
        except Exception:
            pass

    # ---- listening -----------------------------------------------------
    def _feed_listening(self, frame: AudioFrame, prob: float) -> List[Turn]:
        utt = self.segmenter.push(frame, prob)
        if utt is None:
            return []
        self._start_worker(utt)
        self._state = PipelineState.GENERATING
        self._barge_run = 0
        return []

    # ---- generating ----------------------------------------------------
    def _feed_generating(self, frame: AudioFrame, prob: float) -> List[Turn]:
        out: List[Turn] = []
        if prob >= self.cfg.vad.speech_prob:
            self._barge_run += 1
            if self._barge_run >= self.cfg.vad.start_frames:
                # Barge-in: stop the agent, then treat this frame as the start of
                # a fresh utterance.
                self._cancel.set()
                self._state = PipelineState.LISTENING
                self._barge_run = 0
                self.segmenter.reset()
                out.extend(self._drain())
                out.extend(self._feed_listening(frame, prob))
                return out
        else:
            self._barge_run = 0

        if self._has_completed():
            self._state = PipelineState.LISTENING
            out.extend(self._drain())
        return out

    # ---- worker --------------------------------------------------------
    def _start_worker(self, utt: Utterance) -> None:
        self._cancel = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_body, args=(utt,), name="qva-generate", daemon=True
        )
        self._worker.start()

    def _worker_body(self, utt: Utterance) -> None:
        try:
            transcript = self.asr.transcribe(utt.as_frame())
            ident = None
            if self.voiceprints is not None and not self._cancel.is_set():
                ident = self.voiceprints.identify(utt.as_frame())
            person_id = ident.person_id if ident and ident.accepted else None
            person_name = ident.name if ident and ident.accepted else "there"

            reply = ""
            audio = None
            interrupted = self._cancel.is_set()
            if not interrupted:
                context = self.memory.context_block(person_name, transcript)
                system_prompt = render_system_prompt(self.cfg.llm, person_name, context)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript},
                ]
                reply = self.llm.chat(messages)
                interrupted = self._cancel.is_set()
            if not interrupted:
                audio = self.tts.synthesize(reply)
                interrupted = self._cancel.is_set()
            if not interrupted:
                self.sink.play(audio)

            turn = self._build_turn(utt, transcript, reply, ident, audio, interrupted)
            with self._lock:
                self._completed.append(turn)
        except BaseException as exc:  # surfaced to the main thread on drain
            with self._lock:
                self._error = exc

    def _build_turn(self, utt, transcript, reply, ident, audio, interrupted) -> Turn:
        with self._lock:
            idx = self._turn_index
            self._turn_index += 1
        return Turn(
            index=idx,
            person_id=(ident.person_id if ident and ident.accepted else None),
            person_name=(ident.name if ident and ident.accepted else "there"),
            user_text=transcript,
            reply=reply,
            embedding=(ident.embedding if ident else None),
            identification_score=(ident.score if ident else None),
            started_s=utt.start_s,
            ended_s=utt.end_s,
            interrupted=interrupted,
            audio=audio,
        )

    # ---- drain / sync --------------------------------------------------
    def _has_completed(self) -> bool:
        with self._lock:
            return bool(self._completed) or self._error is not None

    def _drain(self) -> List[Turn]:
        with self._lock:
            turns = self._completed
            self._completed = []
            err = self._error
            self._error = None
        if err is not None:
            raise err
        for turn in turns:
            self.memory.observe(turn)
        return turns

    def _join_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=max(5.0, self.cfg.session.worker_timeout_s))
        self._worker = None

    def _process_now(self, utt: Utterance) -> List[Turn]:
        """Process an utterance synchronously (used only for the tail flush)."""
        self._cancel = threading.Event()
        self._worker_body(utt)
        return self._drain()
