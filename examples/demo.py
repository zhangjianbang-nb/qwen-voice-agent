"""End-to-end demo: a scripted conversation through the full-duplex pipeline.

Runs with **no model, no microphone, no GPU**: synthetic audio drives the VAD,
the mock ASR/LLM/TTS backends produce turns, and the script prints each exchange.
It also demonstrates barge-in — once the first reply is being generated, the
source emits speech again and the agent stops.

    python examples/demo.py

Requires only numpy (the core dependency).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qwen_voice_agent.asr import MockAsr
from qwen_voice_agent.audio import AudioFrame
from qwen_voice_agent.config import VoiceAgentConfig
from qwen_voice_agent.llm import MockLlm
from qwen_voice_agent.memory import MemoryManager
from qwen_voice_agent.pipeline import Pipeline, PipelineState
from qwen_voice_agent.sources import NullSink, SyntheticSource
from qwen_voice_agent.vad import EnergyVad, Segmenter

SR = 16000
FRAME = 320


class SlowTts:
    """A deliberately slow TTS so barge-in has time to happen."""

    def __init__(self, delay: float = 0.4) -> None:
        self.delay = delay

    def synthesize(self, text: str) -> AudioFrame:
        time.sleep(self.delay)
        return AudioFrame(np.zeros(3200, dtype=np.float32), SR)


def main() -> int:
    cfg = VoiceAgentConfig()
    cfg.vad.min_speech_ms = 200.0
    cfg.memory.context_budget_chars = 800

    with tempfile.TemporaryDirectory() as tmp:
        pipeline = Pipeline(
            cfg,
            vad=EnergyVad(cfg.vad),
            segmenter=Segmenter(cfg.vad, SR, FRAME),
            asr=MockAsr(),
            tts=SlowTts(),
            llm=MockLlm(),
            memory=MemoryManager(cfg.memory, root=tmp),
            sink=NullSink(),
        )

        # 0.3 s silence, 0.8 s speech, 0.3 s silence — repeated.
        source = SyntheticSource([(False, 0.3), (True, 0.8), (False, 0.3)], sample_rate=SR, frame_samples=FRAME)

        print("driving the pipeline (barge-in expected on the first turn)...")
        seen = 0
        for turn in pipeline.run(source):
            tag = " [interrupted]" if turn.interrupted else ""
            print(f"  turn {turn.index}: user={turn.user_text!r}")
            print(f"           agent={turn.reply!r}{tag}")
            seen += 1
            if seen >= 3:
                break

        print(f"\ncompleted {seen} turn(s); final state = {pipeline.state}")
        print(f"working-memory items = {len(pipeline.memory.working.items())}")
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
