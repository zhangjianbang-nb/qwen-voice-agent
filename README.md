# qwen-voice-agent

A **24/7 full-duplex voice interaction agent**: it listens continuously, recognises
*who* is speaking, transcribes with Qwen ASR, replies with a Qwen small LLM through
Qwen TTS, and remembers the conversation across five kinds of memory — all while
staying interruptible, so the user can talk over it at any moment.

Built around four axes:

| Axis | Modules | What it does |
|------|---------|--------------|
| **Perception** | `audio`, `vad`, `voiceprint` | Microphone framing, voice-activity detection, speaker (voiceprint) embedding & identification |
| **Inference** | `asr`, `tts`, `llm` | Qwen ASR, Qwen TTS and a Qwen small LLM, each behind a backend protocol |
| **Memory** | `memory/` | Five stores — voice, working, semantic, episodic, procedural — plus a manager that decides what reaches the prompt |
| **Runtime** | `sources`, `pipeline`, `session` | Audio sources/sinks, the full-duplex turn state machine with barge-in, and the 24/7 supervision loop |

## Design principles

1. **The runtime core depends only on numpy.** No torch, no scipy, no
   onnxruntime, no sounddevice at import time. A voice agent that cannot run on a
   machine without a microphone or a GPU is not *testable*, so every heavy
   dependency is an optional extra, imported lazily by the backend that needs it.
2. **Every model sits behind a protocol.** ASR, TTS, LLM and the speaker embedder
   each have four-or-so interchangeable backends (`http` for a real server,
   `transformers` for a local checkpoint, `scripted`/`mock` for deterministic
   tests). The pipeline never knows which one it is talking to.
3. **Failures are loud, never silent.** A dead ASR server raises with the URL and
   the transport error rather than hanging the loop.
4. **A 24/7 process states its bounds.** The ring buffer bounds resident audio;
   working memory has a count cap *and* a TTL; the generation worker has a
   timeout; the session has idle and wall-clock exits.

## Install

```bash
# core only — enough to run the agent against HTTP model servers
pip install -e .

# with extras
pip install -e ".[onnx,mic,dsp,dev]"   # silero VAD + onnx speaker + mic + scipy + pytest
```

## Quickstart

```bash
# What can this machine do?
qwen-voice-agent doctor

# Print (or write) the default configuration
qwen-voice-agent config
qwen-voice-agent config --write config.json

# Enrol a speaker from a WAV file, then identify one
qwen-voice-agent enroll voice.wav --person-id p1 --name Ada
qwen-voice-agent identify voice.wav

# Transcribe a WAV with the configured ASR backend
qwen-voice-agent transcribe voice.wav

# Run the agent (bounded here so it exits; drop --seconds for 24/7)
qwen-voice-agent run --source file --source-path voice.wav --seconds 20
```

Configuration is per-stage dataclasses, overridable from the environment for a
systemd-launched process:

```bash
QVA_VAD_BACKEND=silero \
QVA_VAD_MODEL_PATH=/path/silero_vad.onnx \
QVA_ASR_BACKEND=http QVA_ASR_BASE_URL=http://127.0.0.1:8000/v1 \
QVA_TTS_BACKEND=http QVA_TTS_BASE_URL=http://127.0.0.1:8000/v1 \
QVA_LLM_BACKEND=http QVA_LLM_BASE_URL=http://127.0.0.1:8000/v1 QVA_LLM_MODEL=qwen3.8-8b \
qwen-voice-agent run --source mic --sink speaker
```

## Full duplex and barge-in

The `Pipeline` starts a **generation worker thread** when the VAD closes an
utterance, and keeps listening while that worker runs ASR → LLM → TTS →
playback. If the VAD then reports `start_frames` consecutive speech frames, the
worker's cancel event is set: the agent stops at its next checkpoint, returns to
`LISTENING`, and the interrupting frame opens a fresh utterance. Generation and
listening are decoupled through a thread-safe queue, so neither blocks the other.

## The five memories

| Store | Backing | Question it answers |
|-------|---------|---------------------|
| **voice** | JSONL | Who said what, when — speaker identity across sessions |
| **working** | in-process ring + TTL | The last few turns, so "that" and "it" still resolve |
| **semantic** | SQLite | Durable facts ("the user lives in Beijing") |
| **episodic** | JSONL | What happened ("at 14:03 Ada asked about the weather") |
| **procedural** | JSON | How to behave ("lead with the answer") |

> **Voiceprint caveat.** The default `voiceprint.backend="mel"` is a deterministic
> *placeholder*, not a trained speaker model — it ranks clips by spectral envelope
> and does **not** reliably reject a stranger. Set `voiceprint.backend="onnx"`
> (ECAPA / WeSpeaker / 3D-Speaker) for real identification, and calibrate
> `threshold` on enrolment audio.

`MemoryManager.context_block(person, query)` assembles what fits in a **character
budget** (`context_budget_chars`) — highest-priority blocks first — so latency does
not grow with conversation length. The memory layer never depends on a tokenizer.

## Acceptance criteria

The suite encodes eight acceptance criteria (see `docs/DESIGN.md`):

| # | Criterion |
|---|-----------|
| A1 | Core imports pull in no heavy dependency |
| A2 | Config round-trips; env overrides work; unknown keys raise |
| A3 | VAD segments real speech; both energy and Silero backends fire |
| A4 | Voiceprint enrol/identify: self > 0.99, other clips score strictly lower, full ranking returned |
| A5 | ASR/TTS/LLM backends behave; HTTP backends fail loudly, never hang |
| A6 | Five stores round-trip; `context_block` respects its budget |
| A7 | Full-duplex: turns are produced; barge-in cancels generation |
| A8 | 24/7 loop: max-turns/seconds/idle exits, maintenance, graceful shutdown |

```bash
python -m pytest tests/ -q
```

## Hardware notes

This project was developed on an **NVIDIA GB10** box with **no capture device**
(`arecord -l` is empty) and a **CPU-only torch**. That is exactly why the core is
thin and the models are behind protocols: the agent runs end-to-end here against
`mock`/`scripted` backends and real WAV files, and switches to live microphone +
HTTP-served Qwen models by changing configuration, not code. `doctor` reports
which of those paths are available.

## License

Apache-2.0.
