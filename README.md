# qwen-voice-agent

A **24/7 full-duplex voice interaction agent**: it listens continuously, recognises
*who* is speaking, transcribes with Qwen ASR, replies with a Qwen small LLM through
Qwen TTS, and remembers the conversation across five kinds of memory — all while
staying interruptible, so the user can talk over it at any moment.

Built around five axes:

| Axis | Modules | What it does |
|------|---------|--------------|
| **Perception** | `audio`, `vad`, `voiceprint` | Microphone framing, voice-activity detection, speaker (voiceprint) embedding & identification |
| **Inference** | `asr`, `tts`, `llm` | Qwen ASR, Qwen TTS and a Qwen small LLM, each behind a backend protocol |
| **Memory** | `memory/` | Five stores — voice, working, semantic, episodic, procedural — plus a manager that decides what reaches the prompt |
| **Action** | `tools` | OpenAI-style tool calls: the model asks, the runtime executes, the result goes back to the model for a spoken answer |
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

# List the tools the model may call (name + JSON schema)
qwen-voice-agent tools

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

## Tool calling

A voice chatbot can only *say* things; a voice agent can *do* them. With
`tools.enabled=true`, each turn sends the tool schemas to the LLM. If the model
answers with `tool_calls` instead of prose, the pipeline executes them, appends
the results as `role: "tool"` messages, and asks the model for a final spoken
sentence. The loop is bounded by `tools.max_rounds`; after that the final request
is sent without tools, so the agent always says something.

The built-ins cover five groups:

| Group | Tools | What they do |
|-------|-------|--------------|
| Time & memory | `get_current_time`, `remember_fact`, `recall_fact` | Stop the model from guessing the clock; store/read durable facts |
| Files | `write_file`, `read_file`, `list_files` | Read and write UTF-8 text inside `tools.workspace_root`; `..` and symlink escapes are rejected, reads/writes are bounded by `max_file_bytes` |
| Browser & apps | `open_browser`, `open_app` | Open an `http`/`https` URL or search Baidu (`query`); launch an app from `allowed_apps` as argv, never through a shell |
| Desktop (opt-in) | `type_text`, `press_key`, `screenshot` | Synthesise input into a window (or the focused one) and capture the screen; requires `tools.control_enabled=true` |
| Reminders | `set_reminder`, `schedule_task`, `list_reminders`, `cancel_reminder` | One-shot or repeating (`daily`/`weekdays`/`weekly`) reminders. `set_reminder` speaks fixed text; `schedule_task` runs a prompt through the LLM at due time and speaks the answer |

There is deliberately **no shell/exec tool**: a voice agent that can run arbitrary
commands is a remote-code-execution endpoint with a microphone. The action tools
are constrained instead: files stay in a workspace, URLs must be http/https,
apps come from a configured allow-list, and desktop input is off by default
because it types into whatever window has focus.

```json
"tools": {
  "enabled": true, "max_rounds": 3,
  "builtins": ["time", "remember_fact", "recall_fact",
               "write_file", "read_file", "list_files",
               "open_browser", "open_app",
               "set_reminder", "schedule_task", "list_reminders", "cancel_reminder",
               "type_text", "press_key", "screenshot"],
  "control_enabled": true,
  "workspace_root": "/home/whu/workspace/qwen-voice-agent/agent_files",
  "max_file_bytes": 65536,
  "browser_command": "xdg-open",
  "allowed_apps": {"chrome": "google-chrome", "files": "nautilus", "terminal": "gnome-terminal"}
}
```

## Scheduled reminders and tasks

`set_reminder` accepts human time expressions — `10分钟后`, `明天早上8点`,
`每天8点`, `周一9点`, `2026-10-20 09:00` — and writes them to
`scheduler.store_path` (a JSON file written atomically). The pipeline polls the
scheduler on every audio frame and, when the agent is idle (no generation, no
open utterance, silence on the VAD), synthesises `scheduler.speak_prefix + text`
through the normal TTS/sink path. A reminder is therefore barge-in
interruptible: talking over it cancels the speech and retries after
`scheduler.retry_s`.

`schedule_task` uses the same time expressions but stores a *prompt*: at due time
the pipeline runs it through the LLM (with the tool loop) and speaks the answer.
That is what makes "每天8点播报天气" a task rather than a fixed announcement; the
result is not prefixed with "提醒你：".

Delivery uses a claim: a due schedule is marked in flight before it is spoken, so
the next audio frame cannot start it twice. A crash mid-speech leaves the claim
to expire; on restart claims are reset, so the reminder is retried rather than
lost. One-shot reminders are removed after a successful delivery; repeating ones
advance to their next occurrence. A scheduled task whose LLM call fails is also
retried after `scheduler.retry_s`.

```json
"scheduler": {"enabled": true,
              "store_path": "/home/whu/workspace/qwen-voice-agent/state/schedules.json",
              "poll_interval_s": 0.5, "max_schedules": 100,
              "speak_prefix": "提醒你：", "retry_s": 60.0}
```

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

The suite encodes ten acceptance criteria (see `docs/DESIGN.md`):

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
| A9 | Tool loop: a `tool_call` is executed and fed back; rounds are bounded; failures are data, not crashes |
| A10 | Action tools: file access stays in the workspace, browser rejects non-http(s), apps come from the allow-list, desktop tools require opt-in; a due reminder is spoken proactively, a scheduled task runs through the LLM, and repeating schedules advance |

```bash
python -m pytest tests/ -q
```

## Hardware notes

This deployment runs on a 24-core box with an **RTX 5060 Laptop (8 GB)**, a
**CPU-only torch** build, and an internal DMIC (`plughw:1,6`). The LLM is served
by llama.cpp's CUDA backend with all layers offloaded (`NGLAYERS=99`), so torch is
not on the inference path. That is exactly why the core is thin and the models are
behind protocols: the agent runs end-to-end here against `mock`/`scripted`
backends and real WAV files, and switches to live microphone + HTTP-served Qwen
models by changing configuration, not code. `doctor` reports which of those paths
are available.

Desktop-control caveat: this is a **GNOME/Wayland** session. `xdotool` input
works for X11/XWayland windows, but native Wayland clients (Chrome, GNOME apps)
ignore it; `screenshot` needs a Wayland-capable backend (`gnome-screenshot` or
`grim`) that is not installed here. `type_text`/`press_key`/`screenshot` fail
loudly with a readable error rather than pretending to work.

## License

Apache-2.0.
