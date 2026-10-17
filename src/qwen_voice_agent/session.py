"""The 24/7 supervision loop, and the factory that assembles a full agent.

:class:`VoiceSession` is the outer loop: it pulls frames from a source, drives the
:class:`~qwen_voice_agent.pipeline.Pipeline`, and owns the things that keep a
long-running process healthy and *stoppable*:

* ``max_seconds`` / ``max_turns`` — a hard exit so the agent is testable and
  bounded when asked to be;
* ``idle_timeout_s`` — exit after a long silence (optional; ``None`` means run
  forever, which is the real 24/7 mode);
* ``maintenance_interval_s`` — periodic memory pruning and persistence;
* ``worker_timeout_s`` — a stuck generation worker is cancelled, the single most
  common way a long-running voice agent dies quietly;
* graceful shutdown on SIGTERM/SIGINT — stop the sink, flush memory, close the
  source.

:func:`assemble` builds every component from a :class:`VoiceAgentConfig`, so the
CLI and the tests share one wiring path.
"""

from __future__ import annotations

import signal
import threading
import time
from typing import Callable, Iterator, Optional, Tuple

from .config import VoiceAgentConfig
from .asr import build_asr
from .llm import build_llm
from .memory import MemoryManager
from .pipeline import Pipeline, Turn
from .sources import AudioSink, AudioSource, build_sink, build_source
from .tts import build_tts
from .vad import Segmenter, build_vad
from .voiceprint import VoiceprintStore

__all__ = ["VoiceSession", "assemble"]


class VoiceSession:
    """The 24/7 supervision loop."""

    def __init__(
        self,
        cfg: VoiceAgentConfig,
        pipeline: Pipeline,
        source: AudioSource,
        *,
        memory: Optional[MemoryManager] = None,
        on_turn: Optional[Callable[[Turn], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self.source = source
        self.memory = memory or pipeline.memory
        self.on_turn = on_turn
        self._clock = clock
        self._stop = threading.Event()
        self._turns = 0
        self._start = 0.0
        self._last_activity = 0.0
        self._next_maintenance = 0.0

    # ---- control -------------------------------------------------------
    def request_stop(self) -> None:
        """Ask the loop to exit at the next frame boundary (signal-safe)."""
        self._stop.set()

    @property
    def turns_handled(self) -> int:
        return self._turns

    # ---- loop ----------------------------------------------------------
    def run(self) -> int:
        """Run until a limit is hit or ``request_stop`` is called; return turns handled."""
        self._install_signal_handlers()
        sess = self.cfg.session
        self._start = self._clock()
        self._last_activity = self._start
        self._next_maintenance = self._start + sess.maintenance_interval_s

        try:
            for frame in self.source.frames():
                if self._stop.is_set():
                    break
                for turn in self.pipeline.feed(frame):
                    self._turns += 1
                    self._last_activity = self._clock()
                    if self.on_turn is not None:
                        self.on_turn(turn)
                    if sess.max_turns is not None and self._turns >= sess.max_turns:
                        self._stop.set()
                        break
                if self._stop.is_set():
                    break

                now = self._clock()
                if sess.max_seconds is not None and now - self._start >= sess.max_seconds:
                    self._stop.set()
                if sess.idle_timeout_s is not None and now - self._last_activity >= sess.idle_timeout_s:
                    self._stop.set()
                if now >= self._next_maintenance:
                    self._maintain()
                    self._next_maintenance = now + sess.maintenance_interval_s
        finally:
            self._shutdown()
        return self._turns

    def _maintain(self) -> None:
        """Periodic upkeep: prune expired working memory, persist procedural skills."""
        try:
            self.memory.prune()
        except Exception:
            # Maintenance must never take the agent down.
            pass

    def _shutdown(self) -> None:
        """Graceful stop: cancel workers, close the sink/source, flush memory."""
        try:
            self.pipeline.close()
        except Exception:
            pass
        try:
            self.source.close()
        except Exception:
            pass
        if self.memory is not None:
            try:
                self.memory.close()
            except Exception:
                pass
        self._restore_signal_handlers()

    # ---- signals -------------------------------------------------------
    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return  # signals can only be handled on the main thread
        self._prev = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._prev[sig] = signal.getsignal(sig)
                signal.signal(sig, lambda *_: self.request_stop())
            except (ValueError, OSError):
                pass

    def _restore_signal_handlers(self) -> None:
        prev = getattr(self, "_prev", {})
        for sig, handler in prev.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


def assemble(
    cfg: VoiceAgentConfig,
    *,
    source_kind: str = "null",
    source_path: Optional[str] = None,
    sink_kind: str = "null",
    sink_path: Optional[str] = None,
    memory_root: Optional[str] = None,
) -> Tuple[Pipeline, VoiceSession]:
    """Wire every component from ``cfg`` and return ``(pipeline, session)``."""
    vad = build_vad(cfg.vad)
    segmenter = Segmenter(cfg.vad, cfg.audio.sample_rate, cfg.audio.frame_samples)

    voiceprints = None
    if cfg.voiceprint.enabled:
        voiceprints = VoiceprintStore(cfg.voiceprint)

    memory = MemoryManager(cfg.memory, root=memory_root)
    asr = build_asr(cfg.asr)
    tts = build_tts(cfg.tts)
    llm = build_llm(cfg.llm)
    sink = build_sink(sink_kind, sink_path)
    source = build_source(cfg.audio, source_kind, source_path)

    pipeline = Pipeline(
        cfg,
        vad=vad,
        segmenter=segmenter,
        asr=asr,
        tts=tts,
        llm=llm,
        memory=memory,
        sink=sink,
        voiceprints=voiceprints,
    )
    session = VoiceSession(cfg, pipeline, source, memory=memory)
    return pipeline, session
