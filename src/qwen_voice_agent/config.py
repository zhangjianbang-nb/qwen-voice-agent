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
    backend: str = "mel"  # "mel" | "onnx" | "resemblyzer"
    model_path: Optional[str] = None
    threshold: float = 0.75
    min_utterance_s: float = 0.6
    store_path: Optional[str] = None  # defaults to <memory.root>/voiceprints.json


@dataclass
class ASRConfig:
    """Qwen ASR backend selection.

    `backend`:
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


@dataclass
class TTSConfig:
    """Qwen TTS backend selection.

    ``http`` speaks the OpenAI ``/v1/audio/speech`` shape; ``command`` runs an
    external CLI that writes a WAV file (this is how a local Qwen TTS or
    `edge-tts` + ffmpeg is wired without adding a Python dependency).
    """

    backend: str = "mock"  # "http" | "command" | "mock" | "null"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen-tts"
    voice: str = "alloy"
    api_key: Optional[str] = None
    timeout_s: float = 30.0
    sample_rate: int = 16000
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
    """

    max_seconds: Optional[float] = None  # None == run forever
    idle_timeout_s: Optional[float] = None  # None == never exit on idleness
    worker_timeout_s: float = 90.0
    maintenance_interval_s: float = 30.0
    max_turns: Optional[int] = None
    log_level: str = "INFO"


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
    # Persons enrolled ahead of time: [{"person_id": "p1", "name": "Ada"}].
    persons: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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
    """Load config from a JSON file if given, then apply `QVA_*` overrides."""
    base = VoiceAgentConfig.from_json_file(path) if path else VoiceAgentConfig()
    return VoiceAgentConfig.from_env(base)
