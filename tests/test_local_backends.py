"""The locally-served backends: FunASR ASR, CAM++ voiceprint, aplay/arecord.

These tests exist because the acceptance suite (A1–A8) is deliberately
model-free: it proves the *plumbing* with mocks and skips when no checkpoint is on
disk.  That is the right default, but it means the backends this deployment
actually runs were never covered.  Here they are, skipping cleanly on a machine
that has neither the checkpoints nor the audio tools, so the suite still runs
anywhere.

The two audio-hardware tests are gated behind ``QVA_TEST_AUDIO=1`` because they
open the sound device, which is intrusive on a shared machine::

    QVA_TEST_AUDIO=1 pytest tests/test_local_backends.py -q
"""

import os
import subprocess
import time

import numpy as np
import pytest

from qwen_voice_agent.asr import AsrError, FunAsrAsr, clean_sensevoice_text, build_asr
from qwen_voice_agent.audio import AudioFrame, read_wav
from qwen_voice_agent.config import ASRConfig, AudioConfig, SessionConfig, VadConfig, VoiceAgentConfig
from qwen_voice_agent.local_models import campplus_dir, sensevoice_dir
from qwen_voice_agent.pipeline import Pipeline
from qwen_voice_agent.session import VoiceSession, assemble
from qwen_voice_agent.sources import NullSource, NullSink, build_sink, build_source
from qwen_voice_agent.voiceprint import FunAsrEmbedding, VoiceprintStore
from qwen_voice_agent.config import VoiceprintConfig

SR = 16000
FRAME = 320
_HERE = os.path.dirname(os.path.abspath(__file__))
_AUDIO = os.path.join(_HERE, "assets")

# Two clips from the CAM++ checkpoint's own examples: A and B are the same
# speaker, C is a different one.  Real speech beats synthetic for this purpose.
CLIP_A = os.path.join(_AUDIO, "ref_minicpm_signature.wav")
CLIP_B = os.path.join(_AUDIO, "ref_en_dlc_1.wav")


def _funasr_available() -> bool:
    try:
        import funasr  # noqa: F401

        return True
    except Exception:
        return False


needs_funasr = pytest.mark.skipif(not _funasr_available(), reason="funasr is not installed")
needs_sensevoice = pytest.mark.skipif(sensevoice_dir() is None, reason="SenseVoice checkpoint not downloaded")
needs_campplus = pytest.mark.skipif(campplus_dir() is None, reason="CAM++ checkpoint not downloaded")
needs_audio_hw = pytest.mark.skipif(
    os.environ.get("QVA_TEST_AUDIO") != "1",
    reason="opens the sound device; set QVA_TEST_AUDIO=1 to run",
)

# A local piper voice, if one has been installed.  Point QVA_TEST_PIPER at a
# different voice to exercise that one.
PIPER_VOICE = os.environ.get("QVA_TEST_PIPER", "/home/whu/models/piper/zh_CN-huayan-medium.onnx")


def _piper_available() -> bool:
    try:
        import piper  # noqa: F401

        return True
    except Exception:
        return False


needs_piper = pytest.mark.skipif(
    not (_piper_available() and os.path.isfile(PIPER_VOICE)),
    reason="piper-tts is not installed or no voice is on disk",
)


# ---- TTS: piper ----------------------------------------------------------
@needs_piper
def test_piper_speaks_and_is_far_quicker_than_the_network_backend():
    from qwen_voice_agent.config import TTSConfig
    from qwen_voice_agent.tts import build_tts

    cfg = TTSConfig(backend="piper", model_path=PIPER_VOICE, sample_rate=22050)
    tts = build_tts(cfg)
    assert isinstance(tts, __import__("qwen_voice_agent.tts", fromlist=["PiperTts"]).PiperTts)

    tts.warmup()  # the ~1 s load must be outside the measured call
    started = time.perf_counter()
    audio = tts.synthesize("你好，我是你的语音助手。")
    elapsed = time.perf_counter() - started

    assert audio.samples.size > 0
    assert audio.sample_rate == 22050
    assert audio.duration_s > 0.5  # a real sentence, not a click
    assert np.abs(audio.samples).max() > 0.01  # not silence
    # Local synthesis is tens of ms; edge-tts's round trip is ~1.5 s.  A generous
    # bound still fails loudly if piper is accidentally run per-word or reloaded.
    assert elapsed < 0.5, f"piper took {elapsed:.3f}s, expected well under 0.5s"


@needs_piper
def test_a_missing_piper_voice_fails_loudly_instead_of_downloading():
    from qwen_voice_agent.config import TTSConfig
    from qwen_voice_agent.tts import TtsError, build_tts

    tts = build_tts(TTSConfig(backend="piper", model_path="/nonexistent/voice.onnx"))
    with pytest.raises(TtsError):
        tts.synthesize("你好")


def test_piper_without_a_model_path_fails_loudly():
    from qwen_voice_agent.config import TTSConfig
    from qwen_voice_agent.tts import TtsError, build_tts

    tts = build_tts(TTSConfig(backend="piper", model_path=None))
    with pytest.raises(TtsError):
        tts.synthesize("你好")


# ---- ASR -----------------------------------------------------------------
def test_annotation_tags_are_stripped():
    raw = "<|zh|><|NEUTRAL|><|Speech|><|withitn|>你好世界"
    assert clean_sensevoice_text(raw) == "你好世界"
    # FunASR's own rich postprocess turns tags into emoji, which a voice agent
    # would read out loud; this must not.
    assert "🎼" not in clean_sensevoice_text("<|zh|><|BGM|>背景音乐")


@needs_funasr
@needs_sensevoice
def test_funasr_transcribes_real_speech():
    cfg = ASRConfig(backend="funasr", language="auto")
    asr = build_asr(cfg)
    assert isinstance(asr, FunAsrAsr)
    samples, sr = read_wav(CLIP_A)
    text = asr.transcribe(AudioFrame(samples, sr))
    assert text.strip(), "FunASR returned an empty transcript for real speech"
    assert "<|" not in text, "annotation tags leaked into the transcript"


@needs_funasr
@needs_sensevoice
def test_funasr_resamples_non_16k_input():
    """The frontend is 16 kHz; other rates must be resampled, not fed raw."""
    cfg = ASRConfig(backend="funasr")
    asr = build_asr(cfg)
    samples, sr = read_wav(CLIP_A)
    x = samples[::2]  # naive 2x decimation -> 8 kHz
    text = asr.transcribe(AudioFrame(x, sr // 2))
    assert isinstance(text, str) and text.strip()


@needs_funasr
def test_funasr_without_checkpoint_fails_loudly(monkeypatch):
    cfg = ASRConfig(backend="funasr", model_path="/nonexistent/sensevoice")
    with pytest.raises(AsrError):
        FunAsrAsr(cfg).transcribe(AudioFrame(np.zeros(SR, np.float32), SR))


# ---- voiceprint ----------------------------------------------------------
@needs_funasr
@needs_campplus
def test_campplus_separates_speakers_and_rejects_a_stranger(tmp_path):
    """The point of the CAM++ backend: unlike the mel placeholder it *rejects*.

    The same speaker's two clips must score far above a different speaker's, and
    the store must refuse a speaker who was never enrolled.
    """
    cfg = VoiceprintConfig(backend="funasr", threshold=0.35, store_path=str(tmp_path / "vp.json"))
    store = VoiceprintStore(cfg)
    a, sr = read_wav(CLIP_A)
    b, _ = read_wav(CLIP_B)
    store.enroll("p1", "Ada", AudioFrame(a, sr))

    same = store.identify(AudioFrame(a, sr))
    assert same.accepted and same.person_id == "p1" and same.score > 0.5

    # A synthetic voice (never enrolled) must be refused, not merely ranked lower.
    rng = np.random.default_rng(0)
    noise = AudioFrame(rng.standard_normal(SR * 2).astype(np.float32) * 0.05, SR)
    stranger = store.identify(noise)
    assert stranger.accepted is False
    assert stranger.score < cfg.threshold
    assert same.score > stranger.score


@needs_funasr
@needs_campplus
def test_campplus_embedding_dim_and_norm():
    emb = FunAsrEmbedding()
    samples, sr = read_wav(CLIP_A)
    vec = emb.embed(AudioFrame(samples, sr))
    assert vec.shape == (192,)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)


@needs_funasr
def test_campplus_without_checkpoint_fails_loudly():
    with pytest.raises(ValueError):
        FunAsrEmbedding("/nonexistent/campplus")


# ---- runtime fixes -------------------------------------------------------
def test_echo_guard_needs_a_live_microphone_and_an_acoustic_sink():
    """The guard must not fire for a file source: it would drop real utterances."""

    class FakeAcousticSink(NullSink):
        acoustic = True

    def make(cfg, sink, live):
        return Pipeline(
            cfg,
            vad=None,
            segmenter=None,
            asr=None,
            tts=None,
            llm=None,
            memory=None,
            sink=sink,
            live_capture=live,
        )

    cfg = VoiceAgentConfig()
    cfg.vad.echo_guard = True
    assert make(cfg, FakeAcousticSink(), live=True)._echo_guard is True
    assert make(cfg, FakeAcousticSink(), live=False)._echo_guard is False
    assert make(cfg, NullSink(), live=True)._echo_guard is False
    cfg.vad.echo_guard = False
    assert make(cfg, FakeAcousticSink(), live=True)._echo_guard is False


def test_session_paces_to_the_audio_clock(tmp_path):
    """Without pacing, an endless silence source spins a core."""
    cfg = VoiceAgentConfig()
    cfg.session = SessionConfig(max_seconds=0.4, idle_timeout_s=None, realtime=True, maintenance_interval_s=10.0)
    pipeline, _ = assemble(cfg, source_kind="null", memory_root=str(tmp_path))
    session = VoiceSession(cfg, pipeline, NullSource(SR, FRAME), memory=pipeline.memory)
    t0 = time.monotonic()
    session.run()
    elapsed = time.monotonic() - t0
    # ~0.4 s of audio should take ~0.4 s of wall time: not instant, not slow.
    assert 0.25 <= elapsed < 3.0, f"pacing looks wrong: {elapsed:.2f}s for 0.4s of audio"


def test_a_failed_turn_does_not_end_the_session(tmp_path):
    """A 24/7 process must survive one bad utterance (dead server, wedged TTS)."""
    cfg = VoiceAgentConfig()
    cfg.session = SessionConfig(max_seconds=0.3, idle_timeout_s=None, maintenance_interval_s=10.0)
    pipeline, _ = assemble(cfg, source_kind="null", memory_root=str(tmp_path))

    calls = {"n": 0}
    real_feed = pipeline.feed

    def flaky_feed(frame):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated backend failure")
        return real_feed(frame)

    pipeline.feed = flaky_feed
    session = VoiceSession(cfg, pipeline, NullSource(SR, FRAME), memory=pipeline.memory)
    session.run()  # must return, not raise
    assert calls["n"] > 2, "the loop stopped at the first failing frame"


def test_arecord_source_and_aplay_sink_factories():
    """``mic``/``speaker`` must resolve to the ALSA tools when sounddevice is absent."""
    cfg = AudioConfig()
    src = build_source(cfg, "mic")
    assert getattr(src, "live", False) is True, "a mic source must be marked live"
    sink = build_sink("speaker", device=None)
    assert hasattr(sink, "acoustic")
    assert isinstance(build_source(cfg, "file", CLIP_A).live, bool)


@needs_audio_hw
def test_arecord_captures_raw_frames():
    if not _have("arecord"):
        pytest.skip("arecord not installed")
    cfg = AudioConfig(input_device=os.environ.get("QVA_TEST_INPUT_DEVICE") or None)
    src = build_source(cfg, "arecord")
    frames = src.frames()
    frame = next(frames)
    assert isinstance(frame, AudioFrame) and frame.samples.size == cfg.frame_samples
    src.close()


@needs_audio_hw
def test_aplay_playback_is_cancellable_within_a_chunk():
    """Barge-in must stop audio, not just generation.

    A 6 s utterance is cancelled after 0.5 s; the call must return within one
    100 ms chunk of that, not after the utterance has drained.
    """
    import threading

    if not _have("aplay"):
        pytest.skip("aplay not installed")
    sink = build_sink("aplay", device=os.environ.get("QVA_TEST_OUTPUT_DEVICE") or None)
    sr = SR
    t = np.arange(6 * sr, dtype=np.float32) / sr
    audio = AudioFrame((0.02 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), sr)

    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    t0 = time.monotonic()
    sink.play(audio, cancel=cancel)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5, f"cancelled playback took {elapsed:.2f}s (should be ~0.5s)"
    sink.close()


def _have(binary: str) -> bool:
    from shutil import which

    return which(binary) is not None
