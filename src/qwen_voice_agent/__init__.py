"""qwen-voice-agent -- a 24/7 full-duplex voice interaction agent.

The package is split along the four axes the agent is specified in:

* **perception** -- `audio`, `vad`, `voiceprint`: microphone framing, voice
  activity detection, and speaker (voiceprint) embedding + identification.
* **inference** -- `asr`, `tts`, `llm`: Qwen ASR, Qwen TTS and a Qwen small LLM,
  each behind a backend protocol so the real model can be an HTTP server, a local
  `transformers` checkpoint, or a deterministic stub for tests.
* **memory** -- `memory`: the five stores (voice / working / semantic / episodic
  / procedural) plus the manager that decides what reaches the prompt.
* **runtime** -- `sources`, `pipeline`, `session`: audio sources and sinks, the
  full-duplex turn state machine with barge-in, and the 24/7 supervision loop.

Nothing in the core imports torch, scipy, onnxruntime or sounddevice; those are
optional extras and are imported lazily by the backend that needs them.
"""

from .config import VoiceAgentConfig, load_config

__version__ = "0.1.0"

__all__ = ["VoiceAgentConfig", "load_config", "__version__"]
