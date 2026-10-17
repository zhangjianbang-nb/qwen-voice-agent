"""A2 — configuration round-trips, env overrides, and loud failure on typos."""

import pytest

from qwen_voice_agent.config import AudioConfig, VoiceAgentConfig


def test_json_roundtrip(tmp_path):
    cfg = VoiceAgentConfig()
    cfg.persons = [{"person_id": "p1", "name": "Ada"}]
    path = tmp_path / "cfg.json"
    cfg.save_json(str(path))
    again = VoiceAgentConfig.from_json_file(str(path))
    assert again.to_dict() == cfg.to_dict()


def test_env_overrides_with_type_coercion(monkeypatch):
    monkeypatch.setenv("QVA_VAD_START_FRAMES", "5")
    monkeypatch.setenv("QVA_VOICEPRINT_ENABLED", "0")
    monkeypatch.setenv("QVA_LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("QVA_ASR_SCRIPT", "a,b,c")
    monkeypatch.setenv("QVA_PERSONS", '[{"person_id": "p9", "name": "Zoe"}]')
    cfg = VoiceAgentConfig.from_env()
    assert cfg.vad.start_frames == 5
    assert cfg.voiceprint.enabled is False
    assert cfg.llm.temperature == pytest.approx(0.2)
    assert cfg.asr.script == ["a", "b", "c"]
    assert cfg.persons == [{"person_id": "p9", "name": "Zoe"}]


def test_unknown_keys_raise():
    with pytest.raises(ValueError):
        VoiceAgentConfig.from_dict({"vad": {"bogus_key": 1}})
    with pytest.raises(ValueError):
        VoiceAgentConfig.from_dict({"not_a_section": {}})


def test_frame_samples_derivation():
    a = AudioConfig(sample_rate=16000, frame_ms=20.0)
    assert a.frame_samples == 320
    assert a.ring_samples == 16000 * 30


def test_env_bool_variants(monkeypatch):
    for raw, expected in [("1", True), ("true", True), ("yes", True), ("on", True), ("0", False), ("false", False)]:
        monkeypatch.setenv("QVA_VOICEPRINT_ENABLED", raw)
        assert VoiceAgentConfig.from_env().voiceprint.enabled is expected


def test_env_override_for_none_defaulted_numeric(monkeypatch):
    """A field whose default is None must still coerce numbers, not keep a string."""
    monkeypatch.setenv("QVA_SESSION_MAX_SECONDS", "10")
    monkeypatch.setenv("QVA_SESSION_MAX_TURNS", "3")
    monkeypatch.setenv("QVA_SESSION_IDLE_TIMEOUT_S", "none")
    cfg = VoiceAgentConfig.from_env()
    assert cfg.session.max_seconds == 10
    assert isinstance(cfg.session.max_seconds, (int, float))
    assert cfg.session.max_turns == 3
    assert cfg.session.idle_timeout_s is None


def test_idle_timeout_defaults_to_none():
    assert VoiceAgentConfig().session.idle_timeout_s is None

