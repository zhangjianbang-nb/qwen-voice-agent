"""Audio primitives: framing, ring buffer, resampling, WAV round-trip."""

import numpy as np
import pytest

from qwen_voice_agent.audio import (
    AudioFrame,
    RingBuffer,
    concat,
    iter_frames,
    read_wav,
    resample,
    silence,
    write_wav,
)


def test_frame_normalises_to_float32_mono():
    f = AudioFrame([1, 2, 3], 16000)
    assert f.samples.dtype == np.float32
    assert f.samples.shape == (3,)
    assert f.duration_s == pytest.approx(3 / 16000)


def test_frame_rejects_bad_rate():
    with pytest.raises(ValueError):
        AudioFrame(np.zeros(4, np.float32), 0)


def test_ringbuffer_is_bounded_and_keeps_newest():
    rb = RingBuffer(5, 16000)
    rb.push(np.arange(3, dtype=np.float32))
    assert len(rb) == 3
    rb.push(np.arange(10, dtype=np.float32))
    assert len(rb) == 5
    assert np.allclose(rb.snapshot(), np.arange(10)[-5:])


def test_ringbuffer_oversized_push():
    rb = RingBuffer(4, 16000)
    rb.push(np.arange(10, dtype=np.float32))
    assert np.allclose(rb.snapshot(), [6, 7, 8, 9])


def test_resample_lengths():
    x = np.sin(np.linspace(0, 10, 16000)).astype(np.float32)
    down = resample(x, 16000, 8000)
    assert abs(len(down) - 8000) <= 2
    up = resample(down, 8000, 16000)
    assert abs(len(up) - 16000) <= 4


def test_resample_identity():
    x = np.arange(100, dtype=np.float32)
    assert np.array_equal(resample(x, 16000, 16000), x)


def test_wav_roundtrip(tmp_path):
    x = np.sin(np.linspace(0, 20, 8000)).astype(np.float32)
    path = tmp_path / "a.wav"
    write_wav(str(path), x, 16000)
    y, sr = read_wav(str(path))
    assert sr == 16000
    assert len(y) == 8000
    assert np.max(np.abs(y - x)) < 0.01  # 16-bit quantisation tolerance


def test_wav_clips_instead_of_wrapping(tmp_path):
    x = np.array([2.0, -2.0, 0.5], dtype=np.float32)
    path = tmp_path / "clip.wav"
    write_wav(str(path), x, 16000)
    y, _ = read_wav(str(path))
    assert np.max(np.abs(y)) <= 1.0


def test_iter_frames_pads_tail():
    frames = list(iter_frames(np.arange(1000, dtype=np.float32), 16000, 320))
    assert all(len(f) == 320 for f in frames)
    assert len(frames) == 4  # ceil(1000 / 320)


def test_silence_and_concat():
    s = silence(0.1, 16000)
    assert len(s) == 1600
    assert np.allclose(s.samples, 0.0)
    c = concat([s, s])
    assert len(c) == 3200
    with pytest.raises(ValueError):
        concat([])


def test_concat_rejects_mixed_rates():
    with pytest.raises(ValueError):
        concat([AudioFrame(np.zeros(10, np.float32), 16000), AudioFrame(np.zeros(10, np.float32), 8000)])
