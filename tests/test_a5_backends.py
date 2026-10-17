"""A5 — the inference backends behave behind their protocols."""

import numpy as np
import pytest

from qwen_voice_agent.asr import AsrError, HttpAsr, MockAsr, ScriptedAsr, build_asr
from qwen_voice_agent.audio import AudioFrame
from qwen_voice_agent.config import ASRConfig, LLMConfig, TTSConfig
from qwen_voice_agent.llm import LlmError, HttpLlm, MockLlm, build_llm, render_system_prompt
from qwen_voice_agent.tts import HttpTts, MockTts, NullTts, TtsError, build_tts


def _frame(n=1600):
    return AudioFrame(np.zeros(n, dtype=np.float32), 16000)


def test_mock_asr_is_deterministic_and_varies_with_audio():
    a = AudioFrame(np.sin(np.arange(1600, dtype=np.float32) / 50.0), 16000)
    asr = MockAsr()
    assert asr.transcribe(a) == asr.transcribe(a)
    assert asr.transcribe(a) != asr.transcribe(_frame())


def test_scripted_asr_cycles():
    asr = ScriptedAsr(ASRConfig(script=["one", "two"]))
    assert asr.transcribe(_frame()) == "one"
    assert asr.transcribe(_frame()) == "two"
    assert asr.transcribe(_frame()) == "one"


def test_build_asr_defaults_to_mock():
    assert isinstance(build_asr(ASRConfig()), MockAsr)


def test_mock_and_null_tts():
    cfg = TTSConfig()
    mock = MockTts(cfg)
    out = mock.synthesize("hello world")
    assert isinstance(out, AudioFrame) and len(out) > 0
    assert mock.synthesize("hello world").samples.shape == out.samples.shape  # deterministic
    null = NullTts(cfg)
    assert np.allclose(null.synthesize("hi").samples, 0.0)


def test_build_tts_variants():
    assert isinstance(build_tts(TTSConfig(backend="mock")), MockTts)
    assert isinstance(build_tts(TTSConfig(backend="null")), NullTts)


def test_mock_llm_returns_text():
    reply = MockLlm().chat([{"role": "user", "content": "hello there"}])
    assert isinstance(reply, str) and reply
    assert "hello there" in reply


def test_render_system_prompt():
    cfg = LLMConfig()
    text = render_system_prompt(cfg, "Ada", "Known facts:\n- location: Beijing")
    assert "Ada" in text
    assert "Beijing" in text


def test_http_asr_fails_loudly_not_hangs():
    cfg = ASRConfig(backend="http", base_url="http://127.0.0.1:1/v1", timeout_s=1.0)
    with pytest.raises(AsrError):
        HttpAsr(cfg).transcribe(_frame())


def test_http_tts_fails_loudly():
    cfg = TTSConfig(backend="http", base_url="http://127.0.0.1:1/v1", timeout_s=1.0)
    with pytest.raises(TtsError):
        HttpTts(cfg).synthesize("hello")


def test_http_llm_fails_loudly():
    cfg = LLMConfig(backend="http", base_url="http://127.0.0.1:1/v1", timeout_s=1.0)
    with pytest.raises(LlmError):
        HttpLlm(cfg).chat([{"role": "user", "content": "hi"}])
