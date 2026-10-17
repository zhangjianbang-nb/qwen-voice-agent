"""A3 — voice activity detection and utterance segmentation on real speech."""

import numpy as np
import pytest

from qwen_voice_agent.audio import AudioFrame, iter_frames, read_wav
from qwen_voice_agent.config import VadConfig
from qwen_voice_agent.vad import EnergyVad, Segmenter, SileroVad, build_vad

SR = 16000
FRAME = 320


def _tone(n=FRAME, amp=0.3, hz=220.0):
    t = np.arange(n, dtype=np.float32) / SR
    return AudioFrame((amp * np.sin(2 * np.pi * hz * t)).astype(np.float32), SR)


def _silence(n=FRAME):
    return AudioFrame(np.zeros(n, dtype=np.float32), SR)


def test_energy_vad_separates_speech_from_silence():
    cfg = VadConfig()
    vad = EnergyVad(cfg)
    for _ in range(3):
        vad.speech_prob(_silence())  # establish the noise floor
    assert vad.speech_prob(_tone()) > 0.5
    assert vad.speech_prob(_silence()) < 0.5


def test_build_vad_returns_energy_by_default():
    assert isinstance(build_vad(VadConfig(backend="energy")), EnergyVad)


def test_segmenter_segments_real_speech(speech_wav):
    cfg = VadConfig(min_speech_ms=200.0, start_frames=3, end_frames=8)
    samples, sr = read_wav(speech_wav)
    vad = EnergyVad(cfg)
    seg = Segmenter(cfg, sr, FRAME)
    utts = []
    for frame in iter_frames(samples, sr, FRAME):
        u = seg.push(frame, vad.speech_prob(frame))
        if u is not None:
            utts.append(u)
    tail = seg.flush()
    if tail is not None:
        utts.append(tail)

    assert len(utts) >= 1, "expected at least one utterance in a speech WAV"
    for u in utts:
        ms = u.duration_s * 1000.0
        assert cfg.min_speech_ms - 1 <= ms <= cfg.max_speech_ms + 1
        assert u.peak_prob >= cfg.speech_prob


def test_segmenter_drops_too_short_segments():
    cfg = VadConfig(min_speech_ms=500.0, start_frames=2, end_frames=2)
    vad = EnergyVad(cfg)
    seg = Segmenter(cfg, SR, FRAME)
    for _ in range(3):
        seg.push(_silence(), vad.speech_prob(_silence()))
    # three speech frames ~60 ms, below the 500 ms floor
    for _ in range(3):
        seg.push(_tone(), vad.speech_prob(_tone()))
    for _ in range(3):
        seg.push(_silence(), vad.speech_prob(_silence()))
    assert seg.flush() is None


def test_silero_vad_responds_to_real_speech(speech_wav, silero_model):
    pytest.importorskip("onnxruntime")
    cfg = VadConfig(backend="silero", model_path=silero_model)
    vad = SileroVad(cfg)
    samples, sr = read_wav(speech_wav)
    probs = [vad.speech_prob(f) for f in iter_frames(samples, sr, 512)]
    assert max(probs) > 0.5, f"silero never fired on real speech (max={max(probs):.3f})"


def test_silero_stays_quiet_on_silence(silero_model):
    pytest.importorskip("onnxruntime")
    cfg = VadConfig(backend="silero", model_path=silero_model)
    vad = SileroVad(cfg)
    for _ in range(8):
        p = vad.speech_prob(_silence(512))
    assert p < 0.3
