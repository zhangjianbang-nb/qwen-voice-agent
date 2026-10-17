"""A4 — voiceprint enrolment, identification, ranking and persistence.

The default ``mel`` backend is a *deterministic placeholder*, not a trained
speaker model: it discriminates audio by spectral envelope and is **not** robust
to a speaker saying different words.  These tests therefore assert what the
placeholder genuinely guarantees — identical audio scores ~1.0, a different clip
scores strictly lower, and the full ranking is returned — and do NOT claim it
rejects a stranger (that needs the ``onnx`` backend).
"""

import numpy as np
import pytest

from qwen_voice_agent.audio import AudioFrame, read_wav
from qwen_voice_agent.config import VoiceprintConfig
from qwen_voice_agent.voiceprint import MelEmbedding, VoiceprintStore


def test_mel_embedding_shape_and_norm():
    emb = MelEmbedding(n_mels=40)
    vec = emb.embed(AudioFrame(np.random.default_rng(0).standard_normal(16000).astype(np.float32), 16000))
    assert vec.shape == (80,)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)


def test_enrol_and_identify_self(speech_wav, tmp_path):
    cfg = VoiceprintConfig(store_path=str(tmp_path / "vp.json"))
    store = VoiceprintStore(cfg)
    samples, sr = read_wav(speech_wav)
    frame = AudioFrame(samples, sr)
    store.enroll("p1", "Ada", frame)
    res = store.identify(frame)
    assert res.accepted
    assert res.person_id == "p1"
    assert res.name == "Ada"
    assert res.score > 0.99


def test_different_clip_scores_strictly_lower(speech_wav, speech_wav_2, tmp_path):
    cfg = VoiceprintConfig(store_path=str(tmp_path / "vp.json"))
    store = VoiceprintStore(cfg)
    s1, sr1 = read_wav(speech_wav)
    store.enroll("p1", "Ada", AudioFrame(s1, sr1))
    self_score = store.identify(AudioFrame(s1, sr1)).score
    s2, sr2 = read_wav(speech_wav_2)
    other = store.identify(AudioFrame(s2, sr2))
    # A different clip must score strictly below an identical clip, and the
    # ranking must still list the enrolled speaker (the placeholder does not
    # guarantee rejection — see the module docstring).
    assert other.score < self_score
    assert other.ranking and other.ranking[0][0] == "p1"


def test_enrol_averages_multiple_clips(speech_wav, tmp_path):
    cfg = VoiceprintConfig(store_path=str(tmp_path / "vp.json"))
    store = VoiceprintStore(cfg)
    samples, sr = read_wav(speech_wav)
    half = len(samples) // 2
    a = AudioFrame(samples[:half], sr)
    b = AudioFrame(samples[half:], sr)
    emb_a = store.embedder.embed(a)
    emb_b = store.embedder.embed(b)

    store.enroll("p1", "Ada", a)
    merged = store.enroll("p1", "Ada", b)  # returns the stored (running-mean) embedding

    expected = emb_a + emb_b
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(merged, expected, atol=1e-5), "enrolment did not average the two clips"
    assert len(store.speakers()) == 1


def test_persistence_roundtrip(speech_wav, tmp_path):
    path = str(tmp_path / "vp.json")
    cfg = VoiceprintConfig(store_path=path)
    store = VoiceprintStore(cfg)
    samples, sr = read_wav(speech_wav)
    store.enroll("p1", "Ada", AudioFrame(samples, sr))
    store.save()

    store2 = VoiceprintStore(VoiceprintConfig(store_path=path))
    assert "p1" in store2.speakers()
    res = store2.identify(AudioFrame(samples, sr))
    assert res.person_id == "p1"
    assert res.score > 0.99


def test_identify_with_empty_store(tmp_path):
    store = VoiceprintStore(VoiceprintConfig(store_path=str(tmp_path / "vp.json")))
    res = store.identify(AudioFrame(np.zeros(16000, np.float32), 16000))
    assert res.accepted is False
    assert res.person_id is None
    assert res.ranking == []
