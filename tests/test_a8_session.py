"""A8 — the 24/7 supervision loop: limits, maintenance, graceful shutdown."""

import threading
import time

import pytest

from qwen_voice_agent.config import VoiceAgentConfig
from qwen_voice_agent.session import VoiceSession, assemble
from qwen_voice_agent.sources import NullSource, SyntheticSource

SR = 16000
FRAME = 320


def test_max_turns_exit(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    cfg.session.max_turns = 1
    cfg.session.max_seconds = 10.0
    cfg.session.idle_timeout_s = None
    pipeline, _ = assemble(cfg, source_kind="null", memory_root=str(tmp_path))
    src = SyntheticSource([(False, 0.2), (True, 0.6), (False, 0.4)], sample_rate=SR, frame_samples=FRAME, loop=True)
    session = VoiceSession(cfg, pipeline, src, memory=pipeline.memory)
    turns = session.run()
    assert turns >= 1
    assert session.turns_handled >= 1


def test_max_seconds_exit(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.session.max_seconds = 0.2
    cfg.session.idle_timeout_s = None
    cfg.session.maintenance_interval_s = 10.0
    pipeline, _ = assemble(cfg, source_kind="null", memory_root=str(tmp_path))
    src = NullSource(SR, FRAME)
    session = VoiceSession(cfg, pipeline, src, memory=pipeline.memory)
    t0 = time.monotonic()
    session.run()
    assert time.monotonic() - t0 < 5.0


def test_idle_timeout_exit(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.session.max_seconds = None
    cfg.session.idle_timeout_s = 0.2
    pipeline, _ = assemble(cfg, source_kind="null", memory_root=str(tmp_path))
    src = NullSource(SR, FRAME)
    session = VoiceSession(cfg, pipeline, src, memory=pipeline.memory)
    t0 = time.monotonic()
    session.run()
    assert time.monotonic() - t0 < 5.0


def test_request_stop_is_graceful(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.session.max_seconds = None
    cfg.session.idle_timeout_s = None
    pipeline, _ = assemble(cfg, source_kind="null", memory_root=str(tmp_path))
    src = NullSource(SR, FRAME)
    session = VoiceSession(cfg, pipeline, src, memory=pipeline.memory)
    threading.Timer(0.2, session.request_stop).start()
    turns = session.run()
    assert turns == 0


def test_maintenance_runs_periodically(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.session.max_seconds = 0.4
    cfg.session.idle_timeout_s = None
    cfg.session.maintenance_interval_s = 0.05
    pipeline, _ = assemble(cfg, source_kind="null", memory_root=str(tmp_path))
    src = NullSource(SR, FRAME)
    session = VoiceSession(cfg, pipeline, src, memory=pipeline.memory)

    calls = {"n": 0}
    original = pipeline.memory.prune

    def counted():
        calls["n"] += 1
        return original()

    pipeline.memory.prune = counted
    session.run()
    assert calls["n"] >= 1


def test_shutdown_closes_memory(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.session.max_seconds = 0.1
    cfg.session.idle_timeout_s = None
    pipeline, _ = assemble(cfg, source_kind="null", memory_root=str(tmp_path))
    src = NullSource(SR, FRAME)
    session = VoiceSession(cfg, pipeline, src, memory=pipeline.memory)
    session.run()
    # closing the semantic store twice would raise if it were not closed; a
    # closed sqlite connection rejects new queries.
    with pytest.raises(Exception):
        pipeline.memory.semantic.recall("anything")


def test_assemble_builds_working_agent(tmp_path):
    cfg = VoiceAgentConfig()
    pipeline, session = assemble(cfg, source_kind="null", memory_root=str(tmp_path))
    assert pipeline is not None
    assert session is not None
    assert session.memory is pipeline.memory
