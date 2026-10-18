"""A7 — the full-duplex pipeline: turn production and barge-in."""

import time

import numpy as np
import pytest

from qwen_voice_agent.asr import MockAsr
from qwen_voice_agent.audio import AudioFrame
from qwen_voice_agent.config import VoiceAgentConfig
from qwen_voice_agent.llm import MockLlm
from qwen_voice_agent.memory import MemoryManager
from qwen_voice_agent.pipeline import Pipeline, PipelineState
from qwen_voice_agent.sources import NullSink
from qwen_voice_agent.vad import EnergyVad, Segmenter

SR = 16000
FRAME = 320


class FastTts:
    def synthesize(self, text: str) -> AudioFrame:
        return AudioFrame(np.zeros(1600, dtype=np.float32), SR)


class SlowTts:
    """Sleeps so the worker is still generating when the user barges in."""

    def __init__(self, delay: float = 0.5) -> None:
        self.delay = delay

    def synthesize(self, text: str) -> AudioFrame:
        time.sleep(self.delay)
        return AudioFrame(np.zeros(1600, dtype=np.float32), SR)


def _tone():
    t = np.arange(FRAME, dtype=np.float32) / SR
    return AudioFrame((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR)


def _silence():
    return AudioFrame(np.zeros(FRAME, dtype=np.float32), SR)


def _make(cfg, tts, tmp_path, voiceprints=None):
    return Pipeline(
        cfg,
        vad=EnergyVad(cfg.vad),
        segmenter=Segmenter(cfg.vad, cfg.audio.sample_rate, cfg.audio.frame_samples),
        asr=MockAsr(),
        tts=tts,
        llm=MockLlm(),
        memory=MemoryManager(cfg.memory, root=str(tmp_path)),
        sink=NullSink(),
        voiceprints=voiceprints,
    )


def _open_and_close(p, speech_frames=20, silence_frames=10):
    turns = []
    for _ in range(5):
        turns += p.feed(_silence())  # establish the noise floor
    for _ in range(speech_frames):
        turns += p.feed(_tone())     # opens and sustains an utterance
    for _ in range(silence_frames):
        # closes it -> worker starts; a fast worker may also finish here, so
        # the turns it produced must not be dropped on the floor.
        turns += p.feed(_silence())
    return turns


def test_pipeline_produces_a_turn(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    p = _make(cfg, FastTts(), tmp_path)
    turns = _open_and_close(p)
    for _ in range(200):
        turns += p.feed(_silence())
        if turns:
            break
        time.sleep(0.005)
    assert turns, "pipeline produced no turn"
    t = turns[0]
    assert t.user_text.startswith("<mock:")
    assert t.reply
    assert t.interrupted is False
    assert t.person_name == "there"  # no voiceprints configured
    p.close()


def test_pipeline_records_turn_in_memory(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    p = _make(cfg, FastTts(), tmp_path)
    turns = _open_and_close(p)
    for _ in range(200):
        turns += p.feed(_silence())
        if turns:
            break
        time.sleep(0.005)
    assert turns
    # observe() was called by _drain, so working memory holds both sides
    assert len(p.memory.working.items()) >= 2
    p.close()


def test_barge_in_stops_generation(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    p = _make(cfg, SlowTts(delay=0.6), tmp_path)
    _open_and_close(p)
    assert p.state == PipelineState.GENERATING, f"expected generating, got {p.state}"

    # The user speaks again: barge-in must fire within start_frames frames.
    for _ in range(cfg.vad.start_frames):
        p.feed(_tone())
    assert p.state == PipelineState.LISTENING
    assert p._cancel.is_set()

    # The in-flight worker must actually stop, not merely be flagged.
    deadline = time.time() + 3.0
    while time.time() < deadline and any(t.is_alive() for t in p._workers):
        time.sleep(0.02)
    assert not any(t.is_alive() for t in p._workers), "worker thread did not stop after barge-in"
    p.close()


def test_no_barge_in_when_agent_finishes(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    p = _make(cfg, FastTts(), tmp_path)
    _open_and_close(p)
    # let the fast worker finish, then feed silence: state returns to listening
    for _ in range(200):
        p.feed(_silence())
        if p.state == PipelineState.LISTENING:
            break
        time.sleep(0.005)
    assert p.state == PipelineState.LISTENING
    p.close()


def test_pipeline_run_over_synthetic_source(tmp_path):
    from qwen_voice_agent.sources import SyntheticSource

    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    p = _make(cfg, FastTts(), tmp_path)
    src = SyntheticSource(
        [(False, 0.2), (True, 0.6), (False, 0.4)],
        sample_rate=SR,
        frame_samples=FRAME,
        loop=True,
    )
    turns = []
    for turn in p.run(src):
        turns.append(turn)
        if len(turns) >= 2:
            break
    assert len(turns) >= 2
    assert [t.index for t in turns] == list(range(len(turns)))
    p.close()


class FakeVoiceprints:
    """Stands in for VoiceprintStore: every utterance is (or is not) the owner."""

    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted
        self.calls = 0

    def identify(self, frame):
        from qwen_voice_agent.voiceprint import IdentificationResult

        self.calls += 1
        return IdentificationResult(
            person_id="p1" if self.accepted else None,
            name="Ada" if self.accepted else None,
            score=0.9 if self.accepted else 0.05,
            accepted=self.accepted,
            ranking=[("p1", 0.9 if self.accepted else 0.05)],
        )


def test_a_television_does_not_barge_in(tmp_path):
    """VAD alone would let any voice in the room cut the agent off."""
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    cfg.wake.require_enrolled = True
    vp = FakeVoiceprints(accepted=False)
    p = _make(cfg, HangingTts(), tmp_path, voiceprints=vp)
    _open_and_close(p)
    assert p.state == PipelineState.GENERATING

    # 40 frames = 0.8 s, past voiceprint.min_utterance_s (0.6 s).
    for _ in range(40):
        p.feed(_tone())
    assert vp.calls >= 1, "the barge-in candidate was never checked against the voiceprint"
    assert p.state == PipelineState.GENERATING, "a non-owner voice cut the agent off"
    assert not p._cancel.is_set()
    p.close()


def test_the_owner_can_still_barge_in(tmp_path):
    """Verification must not make the agent deaf to the person it serves."""
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    cfg.wake.require_enrolled = True
    vp = FakeVoiceprints(accepted=True)
    p = _make(cfg, HangingTts(), tmp_path, voiceprints=vp)
    _open_and_close(p)
    assert p.state == PipelineState.GENERATING

    for _ in range(40):
        p.feed(_tone())
    assert p.state == PipelineState.LISTENING
    assert p._cancel.is_set()
    p.close()


class HangingTts:
    """Never returns within the test window, simulating a stuck worker."""

    def synthesize(self, text: str) -> AudioFrame:
        time.sleep(3.0)
        return AudioFrame(np.zeros(1600, dtype=np.float32), SR)


def test_ignored_worker_returns_to_listening(tmp_path):
    """Ambient speech must not leave the pipeline stuck in GENERATING.

    A worker that decides an utterance is not addressed to the agent returns
    without producing a Turn.  If the state machine only notices *turns*, that
    finished worker is invisible: the pipeline stays GENERATING, and the next
    real utterance is logged as a barge-in against a turn that no longer exists.
    """
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    cfg.wake.enabled = True
    cfg.wake.wake_word = "小丽同学"
    p = _make(cfg, FastTts(), tmp_path)
    _open_and_close(p)
    assert p.state == PipelineState.GENERATING

    deadline = time.time() + 3.0
    while time.time() < deadline and any(t.is_alive() for t in p._workers):
        time.sleep(0.02)
    assert not any(t.is_alive() for t in p._workers)

    for _ in range(10):
        p.feed(_silence())
    assert p.state == PipelineState.LISTENING
    assert p._ignored == 1
    p.close()


def test_worker_timeout_cancels_stuck_worker(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    cfg.session.worker_timeout_s = 0.15
    p = _make(cfg, HangingTts(), tmp_path)
    _open_and_close(p)
    assert p.state == PipelineState.GENERATING

    t0 = time.time()
    while time.time() - t0 < 2.0:
        p.feed(_silence())
        if p._cancel.is_set():
            break
        time.sleep(0.01)
    assert p._cancel.is_set(), "worker_timeout_s did not cancel a stuck worker"
    p.close()
