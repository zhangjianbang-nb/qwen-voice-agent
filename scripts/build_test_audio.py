#!/usr/bin/env python
"""Build the WAV files the smoke test drives the agent with.

Speech comes from ``edge-tts`` — a *material generator* only; the agent itself
speaks through local ``piper``.  Edge is used here because the stranger-rejection
test needs two distinct voices and only one piper voice is installed.  The clips
are laid out with explicit silences so the VAD/segmenter sees clean utterances:
the gap is wide enough for the CPU LLM (~1-2 s) plus synthesis and playback to
finish, so a turn completes instead of being interrupted by the next question.

    python scripts/build_test_audio.py [outdir]
"""

from __future__ import annotations

import asyncio
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qwen_voice_agent.audio import resample, write_wav  # noqa: E402

SR = 16000
VOICE = os.environ.get("QVA_TEST_VOICE", "zh-CN-XiaoxiaoNeural")
# The gap must outlast generation *plus* playback, or the next question barges in
# before the agent has finished speaking.  Measured on this CPU: ASR 83 ms,
# voiceprint 19 ms, LLM 0.8-2.3 s, synthesis 1.4-2.8 s, playback at real time
# (2.9-5.2 s for these replies) -> a turn is not finished before ~9.7 s.  A 7 s
# gap sits exactly on the barge-in boundary and the run becomes a coin flip; 12 s
# clears it with margin.  Raise QVA_TEST_GAP on a slower machine.
GAP_S = float(os.environ.get("QVA_TEST_GAP", "12.0"))


async def synth(text: str, voice: str = VOICE) -> np.ndarray:
    import edge_tts
    import soundfile as sf

    buf = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk.get("type") == "audio":
            buf += chunk["data"]
    if not buf:
        raise RuntimeError(f"edge-tts returned no audio for {text!r}")
    data, sr = sf.read(io.BytesIO(bytes(buf)), dtype="float32")
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return resample(data, sr, SR) if sr != SR else data


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(round(seconds * SR)), dtype=np.float32)


async def main(outdir: str) -> int:
    os.makedirs(outdir, exist_ok=True)
    u1 = await synth("你好，请用一句话介绍一下你自己。")
    u2 = await synth("我刚才说了什么？")
    stream = np.concatenate([silence(0.6), u1, silence(GAP_S), u2, silence(GAP_S)])
    dialog = os.path.join(outdir, "dialog.wav")
    write_wav(dialog, stream, SR)
    print(f"{dialog}: {stream.size / SR:.2f}s")

    stranger = await synth("这是一段用来验证陌生人会被拒识的录音。", "zh-CN-YunxiNeural")
    stranger_path = os.path.join(outdir, "stranger.wav")
    write_wav(stranger_path, stranger, SR)
    print(f"{stranger_path}: {stranger.size / SR:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "state/smoke")))
