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

Note on interruption granularity: cancel is checked *between* stages and passed to
``sink.play(audio, cancel=...)``.  A sink that honours the event stops mid-utterance
(:class:`~qwen_voice_agent.sources.AplaySink` terminates its player within one
100 ms chunk), which is what makes barge-in mean "the agent stopped talking"
rather than merely "the agent stopped generating".  A sink whose ``play`` blocks
and ignores the event (``sounddevice``) can only be interrupted between
utterances; the ``null``/``file`` sinks used in tests return immediately.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

from .audio import AudioFrame
from .config import VoiceAgentConfig, WakeConfig
from .llm import LlmBackend, chat_with_tools, render_system_prompt
from .memory import MemoryManager
from .scheduler import Schedule, Scheduler
from .sources import AudioSink, AudioSource
from .tools import ToolRegistry
from .vad import Segmenter, Utterance, VadBackend

__all__ = ["Turn", "Pipeline", "PipelineState", "strip_wake_word", "contains_wake_word"]

log = logging.getLogger("qwen_voice_agent.pipeline")

# Punctuation and whitespace that ASR sprinkles through a sentence; the wake word
# has to be found across it ("小丽同学，" vs "小丽同学").
_WAKE_NOISE = set("，。！？、；：,.!?;: 　\n\r\t“”\"'()（）[]【】-—…")


def _arguments_json(arguments) -> str:
    """Normalise a tool call's arguments to the JSON string the API expects.

    llama-server returns a string; a test double may hand back a dict.  Both
    must be replayed verbatim in the assistant message, or the follow-up request
    fails to match the tool result to its call.
    """
    if isinstance(arguments, str):
        return arguments or "{}"
    return json.dumps(arguments, ensure_ascii=False)


def _normalise(text: str) -> tuple:
    """Return ``(stripped_text, index_map)`` so a match can be mapped back.

    ASR returns one line with punctuation; comparing "小丽同学" against
    "小丽同学，" or "小 丽 同 学" needs the noise removed — but the remainder of
    the sentence must still come back with its original characters, so each
    stripped character remembers where it came from.
    """
    kept = []
    index_map = []
    for i, ch in enumerate(text):
        if ch in _WAKE_NOISE:
            continue
        kept.append(ch)
        index_map.append(i)
    return "".join(kept), index_map


def contains_wake_word(text: str, wake: WakeConfig) -> bool:
    """True if ``text`` contains the wake word (or an alias), fuzzily.

    ASR mangles a fixed phrase in predictable ways ("小丽同学" -> "小莉同学",
    "小丽同 学"), so an exact substring test misses real wakes.  A sliding window
    compared with :func:`difflib.SequenceMatcher` catches those without matching
    arbitrary speech: the threshold is deliberately high (``fuzzy_ratio``).
    """
    return _wake_span(text, wake) is not None


def _wake_span(text: str, wake: WakeConfig) -> Optional[tuple]:
    """Return ``(start, end)`` indices in ``text`` covering the wake word."""
    norm, index_map = _normalise(text)
    if not norm:
        return None
    candidates = [wake.wake_word] + list(wake.aliases or [])
    for cand in candidates:
        needle, _ = _normalise(cand or "")
        if not needle:
            continue
        pos = norm.find(needle)
        if pos >= 0:
            return (index_map[pos], index_map[pos + len(needle) - 1] + 1)

    # Fuzzy pass: slide a window the length of the wake word and take the best
    # match.  Only worth doing for a short prefix of the utterance — the wake
    # word is said first, and searching the whole sentence would match noise.
    import difflib

    best = None
    for cand in candidates:
        needle, _ = _normalise(cand or "")
        n = len(needle)
        if n < 2 or len(norm) < n:
            continue
        limit = min(len(norm) - n, 12)  # the wake word is at the start
        for start in range(0, limit + 1):
            window = norm[start : start + n]
            ratio = difflib.SequenceMatcher(None, window, needle).ratio()
            if ratio >= wake.fuzzy_ratio and (best is None or ratio > best[0]):
                best = (ratio, start, start + n)
    if best is None:
        return None
    _, start, end = best
    return (index_map[start], index_map[end - 1] + 1)


def strip_wake_word(text: str, wake: WakeConfig) -> Optional[str]:
    """Remove the wake word and return what is left ("" if it stood alone).

    Returns ``None`` when the wake word is absent, so the caller can tell
    "no wake word" from "wake word, nothing else".
    """
    span = _wake_span(text, wake)
    if span is None:
        return None
    start, end = span
    remainder = (text[:start] + text[end:]).strip()
    return remainder


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
    # What the model actually did before speaking: [{"name", "arguments",
    # "result", "ok"}].  Empty when the turn needed no tool.
    tool_calls: List[dict] = field(default_factory=list)
    # A proactive turn has no user utterance: the agent spoke a due reminder or
    # ran a scheduled task.  `proactive_kind` is "reminder" or "task".
    proactive: bool = False
    proactive_kind: str = "reminder"
    schedule_id: Optional[str] = None

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
        live_capture: bool = False,
        tools: Optional[ToolRegistry] = None,
        scheduler: Optional[Scheduler] = None,
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
        # None means "no tool loop": the worker calls the LLM exactly once, the
        # pre-tool behaviour.  A registry turns on the tool_calls -> execute ->
        # second-round path in `_chat_with_tools`.
        self.tools = tools
        # The same Scheduler the reminder tools write to.  `None` disables
        # proactive announcements; the pipeline then only answers utterances.
        self.scheduler = scheduler
        # Previous frame's speech probability: a due reminder waits while the
        # user is mid-sentence instead of talking over them.
        self._last_prob = 0.0
        self._next_proactive_poll = 0.0

        self._state = PipelineState.LISTENING
        self._barge_run = 0
        self._barge_frames: List[Tuple[AudioFrame, float]] = []
        self._barge_audio: List[AudioFrame] = []
        self._barge_cooldown_until = 0.0
        self._workers: List[threading.Thread] = []
        self._cancel = threading.Event()      # the *current* worker's event
        self._worker_started = 0.0
        self._completed: List[Turn] = []
        self._errors: List[BaseException] = []
        self._lock = threading.Lock()
        self._turn_index = 0
        # Wake-word bookkeeping: after a wake the agent stays addressable for
        # `wake.follow_up_s`, so a task can be given over several sentences.
        self._wake_until = 0.0
        self._ignored = 0
        self._last_ignore_reason: Optional[str] = None
        # Set while a task is running; the pipeline then ignores ambient speech
        # instead of starting new turns.  `task_gate` is the external policy (it
        # may let a correction through); `_task_busy` is set by the worker while
        # it is executing tool calls.
        self.task_gate: Optional[Callable[[str], bool]] = None
        self.on_progress: Optional[Callable[[str, str], None]] = None
        self._task_busy = False
        # While the agent is speaking through the same machine's speakers, the
        # microphone hears it.  The guard is only meaningful when *both* halves of
        # that acoustic loop exist: a live microphone that can hear the agent, and
        # a sink that actually makes sound.  A file source cannot hear anything,
        # so arming the guard there would silently drop real utterances.
        self._echo_guard = (
            bool(cfg.vad.echo_guard)
            and bool(live_capture)
            and bool(getattr(sink, "acoustic", False))
        )
        # Frames captured before this monotonic deadline are dropped.
        self._play_until = 0.0
        # Barge-in is VAD-only, so with a screen in the room it needs a higher
        # bar; without it every line of dialogue cuts the agent off mid-sentence.
        self._barge_in_frames = int(cfg.wake.barge_in_frames or cfg.vad.start_frames)
        # Barge-in is VAD-only by default, which is wrong in a room with a
        # television: the TV trips the VAD and cuts the agent off.  When the
        # owner is enrolled and required, verify the speaker before cancelling.
        self._barge_verify = (
            bool(cfg.vad.barge_in_verify)
            and voiceprints is not None
            and bool(cfg.wake.require_enrolled)
        )
        if cfg.wake.require_enrolled and voiceprints is None:
            log.warning(
                "wake.require_enrolled is set but no voiceprint backend is "
                "configured: the agent cannot tell the owner from a television"
            )

    # ---- public --------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    def feed(self, frame: AudioFrame) -> List[Turn]:
        """Feed one frame; return any turns completed as a result (usually none)."""
        # A reminder is an *input* as well as a turn: check it before processing
        # this frame so a due schedule starts speaking at the next idle moment.
        self.poll_proactive()
        if self._echo_guard and time.monotonic() < self._play_until:
            # The agent is talking and the microphone is hearing it.  Drop the
            # frame — but keep the worker-timeout check and the drain, so a stuck
            # worker is still cancelled and finished turns still reach memory.
            self._check_worker_timeout()
            if self._has_completed():
                self._state = PipelineState.LISTENING
            return self._drain()

        prob = self.vad.speech_prob(frame)
        self._last_prob = prob

        if self._state == PipelineState.GENERATING:
            if time.monotonic() < self._barge_cooldown_until:
                # Just rejected a non-owner barge-in; do not pay for another
                # embedding every 20 ms while the television keeps talking.
                self._reset_barge()
            elif prob >= self.cfg.vad.speech_prob:
                self._barge_run += 1
                self._barge_frames.append((frame, prob))
                if self._barge_verify:
                    # VAD cannot tell the owner from a television.  Collect
                    # enough audio for a speaker embedding, then decide.
                    self._barge_audio.append(frame)
                    if self._barge_audio_s() >= self.cfg.voiceprint.min_utterance_s:
                        if self._barge_is_owner():
                            self._trigger_barge_in()
                        else:
                            log.debug("ignored barge-in candidate: not the enrolled owner")
                            self._reset_barge()
                            self._barge_cooldown_until = time.monotonic() + max(
                                1.0, self.cfg.voiceprint.min_utterance_s
                            )
                elif self._barge_run >= self._barge_in_frames:
                    self._trigger_barge_in()
            else:
                self._reset_barge()
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

    def finish(self) -> Iterator[Turn]:
        """Let in-flight generation complete, then flush — for end-of-stream.

        :meth:`close` cancels, which is right when the operator asked to stop.  A
        source that simply *ends* (a batch WAV played once, not looped) is
        different: cancelling there would throw away the reply to the last
        utterance, so this waits for the workers bounded by ``worker_timeout_s``,
        drains their turns, and flushes a final open utterance.
        """
        self._join_workers()
        for turn in self._drain():
            yield turn
        tail = self.segmenter.flush()
        if tail is not None:
            self._process_now(tail)
            for turn in self._drain():
                yield turn

    def close(self) -> None:
        """Cancel any in-flight workers, wait for them, flush turns, close the sink.

        The inference backends are closed too, so a long-lived HTTP client (the
        per-process connection pool the http backends hold) is released rather
        than leaked at process exit.
        """
        self._cancel.set()
        self._join_workers()
        try:
            self._drain()
        except Exception:
            pass
        for obj in (self.sink, self.asr, self.tts, self.llm, self.scheduler):
            closer = getattr(obj, "close", None)
            if closer is None:
                continue
            try:
                closer()
            except Exception:
                pass

    # ---- state helpers -------------------------------------------------
    def _barge_audio_s(self) -> float:
        return sum(f.duration_s for f in self._barge_audio)

    def _barge_is_owner(self) -> bool:
        if self.voiceprints is None or not self._barge_audio:
            return False
        sr = self._barge_audio[0].sample_rate
        samples = np.concatenate([f.samples for f in self._barge_audio])
        try:
            return bool(self.voiceprints.identify(AudioFrame(samples, sr)).accepted)
        except Exception as exc:
            # Fail closed: an unreadable voiceprint must not hand the agent to
            # whoever is on the television.
            log.warning("barge-in speaker check failed: %s", exc)
            return False

    def _reset_barge(self) -> None:
        self._barge_run = 0
        self._barge_frames = []
        self._barge_audio = []

    def _trigger_barge_in(self) -> None:
        # Cancel the worker, then replay every frame that confirmed the
        # interruption into the segmenter so the opening syllables of the
        # owner's new utterance are not lost.
        log.info("barge-in: cancelling generation (turn in flight)")
        self._cancel.set()
        self._state = PipelineState.LISTENING
        pending, self._barge_frames = self._barge_frames, []
        self._barge_run = 0
        self._barge_audio = []
        self.segmenter.reset()
        for f, p in pending:
            self._maybe_start(f, p)

    def _maybe_start(self, frame: AudioFrame, prob: float) -> None:
        utt = self.segmenter.push(frame, prob)
        if utt is not None:
            self._start_worker(utt)
            self._state = PipelineState.GENERATING
            self._barge_run = 0

    def _check_worker_timeout(self) -> None:
        alive = [t for t in self._workers if t.is_alive()]
        if alive and (time.monotonic() - self._worker_started) > self.cfg.session.worker_timeout_s:
            log.warning(
                "worker exceeded worker_timeout_s=%.1fs; cancelling",
                self.cfg.session.worker_timeout_s,
            )
            self._cancel.set()

    # ---- proactive reminders -------------------------------------------
    def poll_proactive(self, now: Optional[float] = None) -> None:
        """Start speaking a due reminder if the agent is otherwise idle.

        Conditions are deliberately conservative: no generation in flight, no
        open utterance (the user is mid-sentence), no task busy, and the previous
        VAD frame was silence.  A reminder is an utterance the user did not ask
        for, so it must never cut across one they did.
        """
        if self.scheduler is None or self._state != PipelineState.LISTENING:
            return
        wall = time.time() if now is None else now
        if wall < self._next_proactive_poll:
            return
        self._next_proactive_poll = wall + max(0.05, float(self.cfg.scheduler.poll_interval_s))
        if self._task_busy or getattr(self.segmenter, "active", False):
            return
        if self._last_prob >= self.cfg.vad.speech_prob:
            return
        if any(t.is_alive() for t in self._workers):
            return
        if self._echo_guard and time.monotonic() < self._play_until:
            return
        item = self.scheduler.claim_due(wall)
        if item is not None:
            self._start_proactive(item)

    def _start_proactive(self, item: Schedule) -> None:
        self._workers = [t for t in self._workers if t.is_alive()]
        cancel = threading.Event()
        self._cancel = cancel
        self._worker_started = time.monotonic()
        log.info("schedule due: %s (%s) %r", item.id, item.kind, item.text)
        t = threading.Thread(
            target=self._proactive_body,
            args=(item, cancel),
            name="qva-reminder",
            daemon=True,
        )
        self._workers.append(t)
        self._state = PipelineState.GENERATING
        t.start()

    def _proactive_body(self, item: Schedule, cancel: threading.Event) -> None:
        tool_calls_made: List[dict] = []
        try:
            if item.kind == "task":
                # A scheduled task runs the prompt through the same LLM/tool loop
                # as a user turn, then speaks the answer.  It is not prefixed with
                # "提醒你：" because the result is the message.
                context = self.memory.context_block("there", item.text)
                system_prompt = render_system_prompt(self.cfg.llm, "there", context)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item.text},
                ]
                if self.tools is not None and self.tools.enabled:
                    text, tool_calls_made = self._chat_with_tools(
                        messages,
                        cancel,
                        context={"person_id": None, "person_name": "there"},
                    )
                else:
                    text = str(self.llm.chat(messages)).strip()
                if cancel.is_set():
                    self.scheduler.release(item.id)
                    return
                text = text.strip() or "定时任务已完成。"
            else:
                text = self.cfg.scheduler.speak_prefix + item.text
            audio = self.tts.synthesize(text)
            if cancel.is_set():
                self.scheduler.release(item.id)
                return
            if self._echo_guard:
                self._play_until = (
                    time.monotonic()
                    + audio.duration_s
                    + self.cfg.vad.echo_tail_ms / 1000.0
                )
            self.sink.play(audio, cancel=cancel)
            interrupted = cancel.is_set()
            if interrupted:
                # The owner talked over the reminder; retry after `retry_s`
                # rather than restarting it the instant they stop.
                self.scheduler.release(item.id)
            else:
                self.scheduler.complete(item.id)
            turn = Turn(
                index=self._next_index(),
                person_id=None,
                person_name="there",
                user_text="",
                reply=text,
                started_s=time.monotonic(),
                ended_s=time.monotonic(),
                interrupted=interrupted,
                audio=audio,
                tool_calls=tool_calls_made,
                proactive=True,
                proactive_kind=item.kind,
                schedule_id=item.id,
            )
            with self._lock:
                self._completed.append(turn)
        except Exception as exc:
            # A failed reminder must not take the 24/7 loop down; it is retried.
            log.warning("reminder %s failed: %s", item.id, exc)
            self.scheduler.release(item.id)

    # ---- worker --------------------------------------------------------
    def _start_worker(self, utt: Utterance) -> None:
        # Reap finished threads so the list does not grow unboundedly.
        self._workers = [t for t in self._workers if t.is_alive()]
        cancel = threading.Event()
        self._cancel = cancel
        self._worker_started = time.monotonic()
        log.debug(
            "worker start: %.2fs utterance, %d worker(s) alive",
            utt.duration_s,
            len(self._workers),
        )
        t = threading.Thread(
            target=self._worker_body, args=(utt, cancel), name="qva-generate", daemon=True
        )
        self._workers.append(t)
        t.start()

    # ---- gates ---------------------------------------------------------
    def _ignore(self, reason: str, text: str) -> None:
        self._ignored += 1
        self._last_ignore_reason = reason
        # DEBUG, not INFO: in a room with a screen this fires many times a minute,
        # and a log that is unreadable is a log nobody checks.
        log.debug("ignored (%s): %r", reason, text[:60])

    def _gate(self, transcript: str, ident) -> Optional[Tuple[str, str]]:
        """Decide whether this utterance is addressed to the agent.

        Returns ``(action, text)`` with ``action`` in ``{"respond", "ack"}``, or
        ``None`` to ignore the utterance entirely — no reply, no audio, no turn.

        Speaker comes first because a television can say the wake word; the wake
        word comes second because the owner says plenty that is not addressed to
        the agent.  Either gate alone leaves the other failure open, which is why
        both exist.
        """
        text = (transcript or "").strip()
        if not text:
            return None

        wake = self.cfg.wake

        # A task in flight owns the conversation; ambient speech is not a new
        # command.  `task_gate` is the external policy and may let a correction
        # through; when no policy is installed, a running tool loop still wins.
        if self._task_busy:
            if self.task_gate is None or not self.task_gate(text):
                self._ignore("busy with a task", text)
                return None
        elif self.task_gate is not None and not self.task_gate(text):
            self._ignore("busy with a task", text)
            return None

        if wake.require_enrolled and self.voiceprints is not None:
            if not (ident is not None and ident.accepted):
                self._ignore("speaker is not the enrolled owner", text)
                return None

        if not wake.enabled or not wake.wake_word:
            return ("respond", text)

        remainder = strip_wake_word(text, wake)
        if remainder is not None:
            self._wake_until = time.monotonic() + wake.follow_up_s
            return ("respond", remainder) if remainder else ("ack", text)

        if time.monotonic() <= self._wake_until:
            return ("respond", text)

        self._ignore("no wake word", text)
        return None

    def _worker_body(self, utt: Utterance, cancel: threading.Event) -> None:
        try:
            # Per-stage timing at DEBUG: "the agent is slow" is only actionable
            # once it is known which stage is slow, and the stages are a serial
            # chain (ASR -> voiceprint -> LLM -> TTS -> playback).
            t_asr = t_vp = t_llm = t_tts = 0.0
            _t = time.monotonic()
            transcript = self.asr.transcribe(utt.as_frame())
            t_asr = time.monotonic() - _t
            ident = None
            if self.voiceprints is not None and not cancel.is_set():
                _t = time.monotonic()
                ident = self.voiceprints.identify(utt.as_frame())
                t_vp = time.monotonic() - _t
            person_id = ident.person_id if ident and ident.accepted else None
            person_name = ident.name if ident and ident.accepted else "there"

            verdict = self._gate(transcript, ident)
            if verdict is None:
                return  # not addressed to the agent; nothing to say
            action, transcript = verdict

            reply = ""
            audio = None
            tool_calls_made: List[dict] = []
            interrupted = cancel.is_set()
            if not interrupted and action == "ack":
                # The wake word stood alone: answer it without troubling the LLM.
                reply = self.cfg.wake.ack
            elif not interrupted:
                context = self.memory.context_block(
                    person_name, transcript, person_id=person_id
                )
                system_prompt = render_system_prompt(self.cfg.llm, person_name, context)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript},
                ]
                _t = time.monotonic()
                if self.tools is not None and self.tools.enabled:
                    reply, tool_calls_made = self._chat_with_tools(
                        messages,
                        cancel,
                        context={"person_id": person_id, "person_name": person_name},
                    )
                else:
                    reply = self.llm.chat(messages)
                t_llm = time.monotonic() - _t
                interrupted = cancel.is_set()
            if not interrupted:
                _t = time.monotonic()
                audio = self.tts.synthesize(reply)
                t_tts = time.monotonic() - _t
                interrupted = cancel.is_set()
            if not interrupted:
                # Playback is handed the same cancel event: a sink that honours it
                # (AplaySink) stops mid-utterance, which is the difference between
                # "the agent stopped generating" and "the agent stopped talking".
                # The echo guard is armed for the utterance's own duration plus a
                # reverb tail, so the microphone's pickup of this speech cannot be
                # read as a new turn.
                if self._echo_guard:
                    self._play_until = (
                        time.monotonic() + audio.duration_s + self.cfg.vad.echo_tail_ms / 1000.0
                    )
                self.sink.play(audio, cancel=cancel)
                interrupted = cancel.is_set()

            log.debug(
                "worker done: interrupted=%s asr=%r reply_chars=%d tools=%d "
                "stage_ms(asr=%.0f vp=%.0f llm=%.0f tts=%.0f)",
                interrupted,
                transcript[:40],
                len(reply),
                len(tool_calls_made),
                t_asr * 1000,
                t_vp * 1000,
                t_llm * 1000,
                t_tts * 1000,
            )
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
                tool_calls=tool_calls_made,
            )
            with self._lock:
                self._completed.append(turn)
        except BaseException as exc:  # surfaced to the main thread on drain
            with self._lock:
                self._errors.append(exc)

    def _chat_with_tools(
        self,
        messages: List[dict],
        cancel: threading.Event,
        context: Optional[dict] = None,
    ) -> Tuple[str, List[dict]]:
        """Ask → execute → feed results back → ask again, bounded by ``max_rounds``.

        The second round is the whole point: a tool result is not spoken to the
        user, it is *context* the model turns into a sentence.  Every request and
        result is appended in the OpenAI message shape, so the server can match
        ``tool_call_id`` to its result; a malformed history would make the model
        repeat the call or answer from thin air.

        If the model keeps calling tools, the final request is sent without the
        ``tools`` array, which forces prose instead of another call.  Silence is
        not an acceptable outcome for a voice agent, so an empty answer falls
        back to a short acknowledgement.
        """
        assert self.tools is not None
        schemas = self.tools.schemas()
        msgs = list(messages)
        executed: List[dict] = []
        content = ""
        for _round in range(self.tools.max_rounds):
            if cancel.is_set():
                return content, executed
            reply = chat_with_tools(self.llm, msgs, schemas)
            content = reply.content
            if not reply.has_tool_calls:
                return content, executed

            msgs.append(
                {
                    "role": "assistant",
                    "content": reply.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": _arguments_json(call.arguments),
                            },
                        }
                        for call in reply.tool_calls
                    ],
                }
            )
            # While tools run, ambient speech must not start a competing turn;
            # `_gate` checks this flag and ignores new utterances.
            self._task_busy = True
            try:
                for call in reply.tool_calls:
                    if cancel.is_set():
                        return content, executed
                    result = self.tools.execute(call, context=context)
                    executed.append(
                        {
                            "name": result.name,
                            "arguments": call.arguments,
                            "result": result.content,
                            "ok": result.ok,
                        }
                    )
                    log.debug("tool %s(%s) -> %r", result.name, call.arguments, result.content[:80])
                    if self.on_progress is not None:
                        try:
                            self.on_progress(result.name, result.content)
                        except Exception:
                            log.warning("on_progress callback failed", exc_info=True)
                    msgs.append(
                        {"role": "tool", "tool_call_id": call.id, "content": result.content}
                    )
            finally:
                self._task_busy = False

        if cancel.is_set():
            return content, executed
        # Max rounds reached: no tools this time, so the model has to answer.
        final = str(self.llm.chat(msgs)).strip()
        return (final or content or "好的。"), executed

    def _next_index(self) -> int:
        with self._lock:
            idx = self._turn_index
            self._turn_index += 1
        return idx

    # ---- drain / sync --------------------------------------------------
    def _has_completed(self) -> bool:
        with self._lock:
            if self._completed or self._errors:
                return True
            # A worker that stays silent (ambient speech, a stranger, no wake
            # word) returns without producing a Turn.  Without this check the
            # pipeline would stay in GENERATING after that worker finished, so
            # the next real utterance would be logged as a barge-in against a
            # turn that no longer exists.
            return bool(self._workers) and not any(t.is_alive() for t in self._workers)

    def _drain(self) -> List[Turn]:
        with self._lock:
            turns = self._completed
            self._completed = []
            errors = self._errors
            self._errors = []
        if errors:
            raise errors[0]
        for turn in turns:
            if getattr(turn, "proactive", False):
                self.memory.observe_proactive(turn)
            else:
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
