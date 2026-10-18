"""Configuration objects for the voice agent.

Every component takes its own dataclass, and `VoiceAgentConfig` composes them.
The reason for one dataclass per stage rather than a flat dict is that the
stages fail differently: a wrong `vad` threshold costs you false turns, a wrong
`llm` timeout costs you a hung worker, and keeping them in separate objects
means a typo in one cannot silently reconfigure another.

`from_env` exists because a 24/7 process is usually started by systemd, where
editing a JSON file per deployment is worse than exporting `QVA_*` variables.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

ENV_PREFIX = "QVA_"


@dataclass
class AudioConfig:
    """Microphone framing and buffering.

    `ring_seconds` is the hard bound on how much raw audio is ever held in RAM.
    A 24/7 process must bound this explicitly or it grows without limit; 30 s at
    16 kHz mono float32 is 1.9 MB, which is the whole point of choosing a bound
    rather than trusting a garbage collector.
    """

    sample_rate: int = 16000
    frame_ms: float = 20.0
    ring_seconds: float = 30.0
    input_gain: float = 1.0
    # ALSA device names for the ``arecord`` / ``aplay`` capture and playback
    # paths (e.g. "plughw:1,0").  ``None`` lets each tool pick its own default,
    # which is right when the box has exactly one card; naming a device is how
    # you keep the agent off the HDMI output when a headset is also attached.
    input_device: Optional[str] = None
    output_device: Optional[str] = None

    @property
    def frame_samples(self) -> int:
        return max(1, int(round(self.sample_rate * self.frame_ms / 1000.0)))

    @property
    def ring_samples(self) -> int:
        return max(self.frame_samples, int(round(self.sample_rate * self.ring_seconds)))


@dataclass
class VadConfig:
    """Voice activity detection and utterance segmentation.

    `energy_threshold_db` is relative to an adaptive noise floor, not an absolute
    level, so the same config survives a quiet room and a noisy one.  The
    hysteresis pair (`start_frames` / `end_frames`) is what stops a single
    door-slam from opening a turn; `end_frames` is deliberately the larger of the
    two because a turn that closes early is far cheaper than one that never
    closes.

    `echo_guard` is the honest answer to a single-box deployment: when the agent
    plays through the same machine's speakers, its own voice is picked up by the
    microphone, the VAD fires, and the agent interrupts itself.  With the guard on,
    captured frames are dropped while the agent is speaking (plus `echo_tail_ms`
    of reverb tail), which makes the loop usable — at the price of true barge-in
    during playback.  Wear headphones, or set this to ``False``, to get real
    full-duplex interruption back.  Proper echo cancellation (AEC with a playback
    reference signal) is the fix that keeps both.
    """

    backend: str = "energy"  # "energy" | "silero"
    model_path: Optional[str] = None  # silero_vad.onnx when backend == "silero"
    speech_prob: float = 0.5
    energy_threshold_db: float = 6.0
    energy_range_db: float = 18.0
    noise_adapt: float = 0.95
    start_frames: int = 3
    end_frames: int = 8
    min_speech_ms: float = 250.0
    max_speech_ms: float = 15000.0
    pre_roll_ms: float = 300.0
    echo_guard: bool = False  # drop capture while the agent speaks (see above)
    echo_tail_ms: float = 400.0  # keep dropping this long after playback ends


@dataclass
class VoiceprintConfig:
    """Speaker embedding and identification.

    `threshold` is the cosine similarity above which an embedding is accepted as
    a known speaker.  The default is a starting point, not a calibrated value:
    it must be tuned per embedding backend on real enrolment audio, and the
    `identify` path returns the full ranking so a caller can see how close the
    decision was instead of only the verdict.
    """

    enabled: bool = True
    # "mel" (placeholder) | "funasr" (local CAM++) | "onnx" | "resemblyzer"
    backend: str = "mel"
    model_path: Optional[str] = None
    threshold: float = 0.75
    min_utterance_s: float = 0.6
    store_path: Optional[str] = None  # defaults to <memory.root>/voiceprints.json


@dataclass
class ASRConfig:
    """Qwen ASR backend selection.

    `backend`:
      * ``funasr`` -- a **local** FunASR checkpoint (SenseVoice).  This is the
        path this deployment runs: no server, no network, ~0.1 s per utterance
        on CPU.  ``model_path`` names the checkpoint directory; ``None`` means
        "use the cached ModelScope download";
      * ``http`` -- an OpenAI-compatible ``/v1/audio/transcriptions`` endpoint
        (this is how Qwen ASR is normally reached in production);
      * ``transformers`` -- a local checkpoint loaded through `transformers`;
      * ``scripted`` -- pops the next line from a fixed script (demos/tests);
      * ``mock`` -- deterministic, derived from the audio itself.
    """

    backend: str = "mock"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen-asr"
    api_key: Optional[str] = None
    language: Optional[str] = None
    timeout_s: float = 30.0
    script: List[str] = field(default_factory=list)
    # ``funasr`` backend only: checkpoint directory, compute device, and whether
    # to inverse-text-normalise the output (digits vs spelled-out numbers).
    model_path: Optional[str] = None
    device: str = "cpu"
    use_itn: bool = True


@dataclass
class TTSConfig:
    """Qwen TTS backend selection.

    ``http`` speaks the OpenAI ``/v1/audio/speech`` shape; ``edge`` synthesises
    locally through the ``edge-tts`` package (no subprocess, no temp file — the
    online voices are the best Chinese quality available here); ``command`` runs
    an external CLI that writes a WAV file, for anything else.

    ``voice`` is the backend's own voice name: an OpenAI voice for ``http``, an
    Edge voice (``zh-CN-XiaoxiaoNeural``) for ``edge``.

    ``piper`` runs the voice ONNX locally.  Synthesis is ~50 ms per sentence
    against edge's ~1.5 s network round trip, so it is the low-latency choice, at
    some cost in naturalness.  ``model_path`` points at the voice ``.onnx``;
    ``model_config`` defaults to ``<model_path>.json``.  ``length_scale`` slows
    (``>1``) or speeds (``<1``) the delivery without changing the voice.
    """

    backend: str = "mock"  # "http" | "edge" | "piper" | "command" | "mock" | "null"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen-tts"
    voice: str = "alloy"
    api_key: Optional[str] = None
    timeout_s: float = 30.0
    sample_rate: int = 16000
    model_path: Optional[str] = None  # piper voice .onnx
    model_config: Optional[str] = None  # defaults to <model_path>.json
    length_scale: Optional[float] = None  # piper rate; 1.0 is the voice's native
    command: Optional[str] = None  # e.g. "piper --model x.onnx -f {out}"


@dataclass
class LLMConfig:
    """Qwen small-LLM backend selection.

    `system_prompt` is a template with `{context}` and `{person}` placeholders;
    it is rendered per turn from the memory manager's context block.  Keeping
    the template here rather than in code means the agent's persona is
    configuration, not a source edit.
    """

    backend: str = "mock"  # "http" | "local" | "mock"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen3.8-8b"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 256
    timeout_s: float = 60.0
    system_prompt: str = (
        "You are a warm, concise voice assistant talking out loud to {person}. "
        "Reply in one or two short sentences; never use markdown or lists. "
        "Use the remembered context when it is relevant.\n\n{context}"
    )


@dataclass
class MemoryConfig:
    """The five memory stores.

    `context_budget_chars` is a character budget, not a token budget, because the
    memory layer must not depend on a tokenizer.  It is deliberately conservative:
    the context block is prepended to every turn, so an unbounded one would make
    latency grow with conversation length.
    """

    root: str = "memory"
    working_max_items: int = 64
    working_ttl_s: float = 900.0
    semantic_db: Optional[str] = None  # defaults to <root>/semantic.sqlite3
    episodic_path: Optional[str] = None  # defaults to <root>/episodic.jsonl
    procedural_path: Optional[str] = None  # defaults to <root>/procedural.json
    voice_utterance_log: Optional[str] = None  # defaults to <root>/voice_utterances.jsonl
    context_budget_chars: int = 1200
    extract_facts: bool = True
    seed_skills: bool = True


@dataclass
class SessionConfig:
    """The 24/7 supervision loop.

    ``max_seconds`` / ``max_turns`` are optional bounds (``None`` == run forever,
    the real 24/7 mode).  ``idle_timeout_s`` is also optional and defaults to
    ``None`` so a silent-but-alive agent is not killed; set it if you want an idle
    exit.  ``worker_timeout_s`` is the point at which a generation worker is
    considered stuck and its cancellation event is set.  Without it a hung LLM
    request would freeze the turn loop forever, which is the single most common
    way a long-running voice agent dies quietly.

    ``realtime`` paces the loop to the audio's own clock: one frame of audio per
    frame-duration of wall time.  It is on by default because without it a
    non-blocking source (a WAV replay, or the endless silence of ``NullSource``)
    is consumed as fast as the CPU allows — which both spins a core and makes the
    agent deaf to its own timing.  A blocking source (microphone capture) already
    paces itself, and the pacer then simply never sleeps.
    """

    max_seconds: Optional[float] = None  # None == run forever
    idle_timeout_s: Optional[float] = None  # None == never exit on idleness
    worker_timeout_s: float = 90.0
    maintenance_interval_s: float = 30.0
    max_turns: Optional[int] = None
    log_level: str = "INFO"
    realtime: bool = True


@dataclass
class WakeConfig:
    """Wake word and speaker gating.

    Two gates, because a voice agent sharing a room with a screen needs both.
    The **voiceprint** answers "is this my owner?" — without it every line of
    dialogue from a nearby video becomes a turn, which is exactly what happened
    on this machine.  The **wake word** answers "is this utterance addressed to
    me?" — without it every overheard sentence from the owner does.

    ``follow_up_s`` keeps the agent listening for a while after a wake, so a task
    can be given across several sentences ("小丽同学…" then "把报告发我") without
    repeating the wake word before every clause.  ``aliases`` holds the ASR
    variants a wake word tends to come back as.

    Both gates are **off by default**: the acceptance suite (A1-A8) and a plain
    conversation assume every utterance is for the agent.  A room with a screen
    turns them on, which is what this deployment does.
    """

    enabled: bool = False
    wake_word: str = "小丽同学"
    aliases: List[str] = field(default_factory=list)
    follow_up_s: float = 8.0
    ack: str = "在呢，你说。"
    require_enrolled: bool = False
    fuzzy_ratio: float = 0.75  # tolerance for the ASR mangling the wake word
    # Barge-in is decided from VAD alone (no speaker check), so in a room with a
    # screen every line of dialogue would cut the agent off mid-sentence.  When
    # set, interrupting requires this many consecutive speech frames instead of
    # `vad.start_frames`; speech from a screen rarely holds that long.
    barge_in_frames: Optional[int] = None


@dataclass
class VoiceAgentConfig:
    """The whole agent's configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    voiceprint: VoiceprintConfig = field(default_factory=VoiceprintConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    # Persons enrolled ahead of time: [{"person_id": "p1", "name": "Ada"}].
    persons: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def resolve_paths(self) -> "VoiceAgentConfig":
        """Fill in path defaults that are derived from ``memory.root``.

        The voiceprint store is the one path that *must* agree between the agent
        and the ``enroll``/``identify`` commands: if it defaults to ``None`` the
        two silently use different files, and enrolment appears to have no effect.
        Deriving it from the memory root makes every entry point share one store.
        """
        base = os.path.abspath(self.memory.root)
        if self.voiceprint.store_path is None:
            self.voiceprint.store_path = os.path.join(base, "voiceprints.json")
        return self

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceAgentConfig":
        """Build a config from a nested dict, ignoring unknown keys loudly.

        Unknown keys raise rather than being dropped: a silently ignored
        `memory.rot` typo is a config change that appears to work and does not.
        """
        cfg = cls()
        for section in fields(cls):
            value = data.get(section.name)
            if value is None:
                continue
            if section.name == "persons":
                cfg.persons = list(value)
                continue
            sub_cls = section.type if isinstance(section.type, type) else None
            if sub_cls is None:
                # `from __future__ import annotations` makes types strings; resolve
                # against this module's globals.
                sub_cls = globals().get(section.type)  # type: ignore[arg-type]
            if not isinstance(value, dict) or sub_cls is None:
                raise ValueError(f"config section {section.name!r} must be an object")
            known = {f.name for f in fields(sub_cls)}
            unknown = set(value) - known
            if unknown:
                raise ValueError(
                    f"unknown key(s) in config section {section.name!r}: {sorted(unknown)}"
                )
            setattr(cfg, section.name, sub_cls(**value))
        extra = set(data) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError(f"unknown top-level config key(s): {sorted(extra)}")
        return cfg

    @classmethod
    def from_json_file(cls, path: str) -> "VoiceAgentConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def from_env(cls, base: Optional["VoiceAgentConfig"] = None) -> "VoiceAgentConfig":
        """Apply `QVA_<SECTION>_<FIELD>` environment overrides.

        Values are parsed against the dataclass field's current type, so
        `QVA_VAD_START_FRAMES=5` becomes an int and `QVA_VOICEPRINT_ENABLED=0`
        becomes a bool without a hand-written coercion table.
        """
        cfg = base or cls()
        for section in fields(cls):
            if section.name == "persons":
                raw = os.environ.get(ENV_PREFIX + "PERSONS")
                if raw:
                    cfg.persons = json.loads(raw)
                continue
            sub_cls = globals().get(section.type)  # type: ignore[arg-type]
            if sub_cls is None:
                continue
            current = getattr(cfg, section.name)
            for f in fields(sub_cls):
                key = f"{ENV_PREFIX}{section.name.upper()}_{f.name.upper()}"
                if key not in os.environ:
                    continue
                setattr(current, f.name, _coerce(os.environ[key], getattr(current, f.name)))
        return cfg


def _coerce(raw: str, reference: Any) -> Any:
    """Coerce an environment string to the type of the field it overrides.

    For a field whose default is ``None`` (``max_seconds``, ``max_turns``,
    ``idle_timeout_s``, ``model_path`` ...) the target type is unknown, so we try
    int, then float, then leave the string; ``none``/``null``/empty mean ``None``.
    Without this, ``QVA_SESSION_MAX_SECONDS=10`` would become the *string* "10"
    and crash the loop at the first comparison.
    """
    if isinstance(reference, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(reference, int):
        return int(raw)
    if isinstance(reference, float):
        return float(raw)
    if isinstance(reference, list):
        return [item for item in raw.split(",") if item != ""]
    if reference is None:
        s = raw.strip()
        if s.lower() in {"", "none", "null"}:
            return None
        for cast in (int, float):
            try:
                return cast(s)
            except ValueError:
                continue
        return raw
    return raw


def load_config(path: Optional[str] = None) -> VoiceAgentConfig:
    """Load config from a JSON file if given, apply `QVA_*` overrides, resolve paths."""
    base = VoiceAgentConfig.from_json_file(path) if path else VoiceAgentConfig()
    return VoiceAgentConfig.from_env(base).resolve_paths()
