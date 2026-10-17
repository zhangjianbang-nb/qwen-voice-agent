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


def _write_float32_wav(path, samples, sr):
    import struct

    data = np.asarray(samples, dtype=np.float32).tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, sr, sr * 4, 4, 32)  # audioFormat=3 (IEEE float)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    riff = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body
    with open(path, "wb") as fh:
        fh.write(riff)


def test_read_wav_24bit(tmp_path):
    import struct
    import wave

    n = 100
    vals = [int(0.5 * 8388607) * (1 if i % 2 else -1) for i in range(n)]
    raw = b"".join(struct.pack("<i", v)[:3] for v in vals)
    path = tmp_path / "24.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)
        wf.setframerate(16000)
        wf.writeframes(raw)
    y, sr = read_wav(str(path))
    assert sr == 16000
    assert len(y) == n, "24-bit samples misread (an int16 fallback would give 1.5x length)"
    assert abs(abs(y[0]) - 0.5) < 0.01


def test_read_wav_float32_uses_format_code(tmp_path):
    path = tmp_path / "f.wav"
    _write_float32_wav(str(path), [0.25, -0.5, 0.75], 16000)
    y, sr = read_wav(str(path))
    assert sr == 16000
    assert np.allclose(y, [0.25, -0.5, 0.75])


def test_read_wav_quiet_int32_not_misread(tmp_path):
    import wave

    path = tmp_path / "i32.wav"
    x = np.array([100, -100, 200], dtype=np.int32)  # quiet int32 PCM
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(16000)
        wf.writeframes(x.tobytes())
    y, sr = read_wav(str(path))
    assert len(y) == 3
    assert abs(y[0] - 100 / 2147483648.0) < 1e-9
