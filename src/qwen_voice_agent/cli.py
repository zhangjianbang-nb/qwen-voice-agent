"""Command-line entry point.

Subcommands:

* ``run``        — start the agent (the 24/7 loop; bound it with ``--seconds`` /
                   ``--turns`` for a smoke run);
* ``doctor``     — report what this machine can and cannot do (deps, audio
                   devices, model files, and whether the *configured* servers
                   actually answer) so a failing deployment is diagnosable;
* ``enroll``     — add a speaker to the voiceprint store from a WAV file;
* ``identify``   — identify the speaker in a WAV file against the store;
* ``transcribe`` — run the configured ASR backend on a WAV file;
* ``voices``     — list the TTS voices installed on this machine;
* ``speak``      — synthesise one line and play it, so a voice can be auditioned
                   and chosen before it is wired into ``run``;
* ``config``     — print the default configuration, or write a template JSON.

The CLI is intentionally thin: all logic lives in the package so it is testable
without a terminal.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import List, Optional

from .audio import read_wav
from .config import VoiceAgentConfig, load_config


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qwen-voice-agent", description="24/7 full-duplex voice agent")
    p.add_argument("--config", "-c", help="path to a JSON config file")
    p.add_argument("--log-level", default=None, help="override session.log_level")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start the agent loop")
    run.add_argument("--source", choices=["null", "file", "mic", "arecord"], default="null")
    run.add_argument("--source-path", help="WAV path when --source=file")
    run.add_argument("--sink", choices=["null", "file", "speaker", "aplay"], default="null")
    run.add_argument("--sink-path", help="WAV path when --sink=file")
    run.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    run.add_argument("--turns", type=int, default=None, help="stop after N turns")
    run.add_argument("--memory-root", default=None, help="where memory files live")
    run.add_argument(
        "--no-loop",
        action="store_true",
        help="play a file source once and end, letting the last reply finish",
    )
    run.add_argument(
        "--no-realtime",
        action="store_true",
        help="do not pace frame consumption to the audio clock (batch mode)",
    )
    run.add_argument(
        "--tts-backend",
        choices=["piper", "edge", "http", "command"],
        default=None,
        help="override tts.backend (piper is local and fast, edge is natural but online)",
    )
    run.add_argument(
        "--tts-voice",
        default=None,
        help="override tts.voice (edge voice name, or a piper voice file name)",
    )
    run.add_argument(
        "--tts-model",
        default=None,
        help="override tts.model_path (piper voice .onnx)",
    )
    run.add_argument(
        "--tts-rate",
        type=float,
        default=None,
        help="override tts.length_scale (piper speaking rate, 1.0 = native)",
    )
    run.add_argument(
        "--no-wake",
        action="store_true",
        help="ignore the wake word for this run (every utterance is for the agent)",
    )
    run.add_argument(
        "--anyone",
        action="store_true",
        help="do not require an enrolled speaker (accept any voice)",
    )

    sub.add_parser("doctor", help="report environment readiness")

    voices = sub.add_parser("voices", help="list the TTS voices available on this machine")
    voices.add_argument("--tts-backend", default=None)
    voices.add_argument("--dir", default=None, help="directory to scan for piper .onnx voices")

    speak = sub.add_parser("speak", help="synthesise one line and play it (audition a voice)")
    speak.add_argument("text")
    speak.add_argument("--tts-backend", default=None)
    speak.add_argument("--tts-voice", default=None)
    speak.add_argument("--tts-model", default=None)
    speak.add_argument("--tts-rate", type=float, default=None)
    speak.add_argument("--out", default=None, help="write a WAV here instead of playing")
    speak.add_argument(
        "--sink",
        choices=["aplay", "file", "null"],
        default="aplay",
        help="aplay (default) plays it; file writes --out; null synthesises silently",
    )

    enroll = sub.add_parser("enroll", help="enrol a speaker from a WAV file")
    enroll.add_argument("wav")
    enroll.add_argument("--person-id", required=True)
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--store", default=None, help="voiceprint store path")

    ident = sub.add_parser("identify", help="identify the speaker in a WAV file")
    ident.add_argument("wav")
    ident.add_argument("--store", default=None)

    tr = sub.add_parser("transcribe", help="transcribe a WAV file")
    tr.add_argument("wav")

    cfg = sub.add_parser("config", help="print or write the default config")
    cfg.add_argument("--write", help="write the default config to this path")
    return p


def _configure_logging(cfg: VoiceAgentConfig, override: Optional[str]) -> None:
    level = (override or cfg.session.log_level or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _apply_tts_overrides(cfg: VoiceAgentConfig, args) -> None:
    """Let the voice be changed per run, without editing config.json.

    Switching voices is the thing users actually do, and an edge voice name
    (``zh-CN-YunxiNeural``) and a piper voice file (``zh_CN-huayan-medium``) are
    different kinds of name, so ``--tts-voice`` picks the backend too when the
    name looks like a piper file.
    """
    backend = getattr(args, "tts_backend", None)
    voice = getattr(args, "tts_voice", None)
    model = getattr(args, "tts_model", None)
    rate = getattr(args, "tts_rate", None)

    if backend:
        cfg.tts.backend = backend
    if voice:
        cfg.tts.voice = voice
        # A piper voice is identified by its file name, not by a voice id.
        if voice.endswith(".onnx") or os.path.sep in voice:
            cfg.tts.model_path = voice
            cfg.tts.backend = "piper"
    if model:
        cfg.tts.model_path = model
        cfg.tts.backend = "piper"
    if rate is not None:
        cfg.tts.length_scale = rate


def cmd_run(args) -> int:
    from .session import assemble

    cfg = load_config(args.config)
    _configure_logging(cfg, args.log_level)
    _apply_tts_overrides(cfg, args)
    if getattr(args, "no_wake", False):
        cfg.wake.enabled = False
    if getattr(args, "anyone", False):
        cfg.wake.require_enrolled = False
    if args.seconds is not None:
        cfg.session.max_seconds = args.seconds
    if args.turns is not None:
        cfg.session.max_turns = args.turns
    if args.no_realtime:
        cfg.session.realtime = False
    pipeline, session = assemble(
        cfg,
        source_kind=args.source,
        source_path=args.source_path,
        sink_kind=args.sink,
        sink_path=args.sink_path,
        memory_root=args.memory_root,
        source_loop=not args.no_loop,
        warmup=True,
    )

    # Print the wiring once at startup: a voice agent that runs but talks to the
    # wrong backend is the hardest kind of deployment bug to see in a log.
    print(
        "qwen-voice-agent: "
        f"source={args.source} sink={args.sink} "
        f"asr={cfg.asr.backend} llm={cfg.llm.backend}({cfg.llm.base_url}) "
        f"tts={cfg.tts.backend}({cfg.tts.voice}) "
        f"voiceprint={cfg.voiceprint.backend} vad={cfg.vad.backend} "
        f"realtime={cfg.session.realtime}"
    )

    def report(turn):
        tag = " [interrupted]" if turn.interrupted else ""
        who = turn.person_name or "unknown"
        print(f"[{turn.index}] {who}: {turn.user_text!r} -> {turn.reply!r}{tag}")

    session.on_turn = report
    turns = session.run()
    print(f"session ended after {turns} turn(s)")
    return 0


def _probe_url(url: str, timeout: float = 3.0) -> str:
    """Return 'OK' if an HTTP GET on ``url`` answers, else the failure reason."""
    try:
        import httpx  # type: ignore
    except Exception:
        return "httpx missing"
    try:
        with httpx.Client(timeout=timeout) as client:
            code = client.get(url).status_code
        return f"OK ({code})" if code < 500 else f"HTTP {code}"
    except Exception as exc:
        return f"FAIL ({type(exc).__name__})"


def cmd_doctor(args) -> int:
    cfg = load_config(args.config)
    print("qwen-voice-agent doctor")
    print(f"  python      : {sys.version.split()[0]}")

    for mod, why in [
        ("numpy", "core requirement"),
        ("httpx", "http ASR/TTS/LLM backends"),
        ("funasr", "local ASR (SenseVoice) / local voiceprint (CAM++)"),
        ("piper", "local TTS backend"),
        ("edge_tts", "edge TTS backend (online fallback)"),
        ("soundfile", "decoding TTS audio"),
        ("onnxruntime", "silero VAD / onnx speaker embedding"),
        ("sounddevice", "microphone capture / playback (optional)"),
        ("scipy", "polyphase resampling"),
    ]:
        try:
            __import__(mod)
            print(f"  {mod:12s}: OK   ({why})")
        except Exception:
            print(f"  {mod:12s}: MISSING   ({why})")

    # ---- audio devices -------------------------------------------------
    for tool in ("arecord", "aplay"):
        if not shutil.which(tool):
            print(f"  {tool:12s}: not installed")
            continue
        out = subprocess.run([tool, "-l"], capture_output=True, text=True).stdout
        cards = [ln.strip() for ln in out.splitlines() if "card" in ln]
        print(f"  {tool:12s}: {len(cards)} device(s)")
        for ln in cards[:6]:
            print(f"                {ln}")

    # ---- local model files ---------------------------------------------
    try:
        from .local_models import campplus_dir, sensevoice_dir

        sv = cfg.asr.model_path or sensevoice_dir()
        cp = cfg.voiceprint.model_path or campplus_dir()
        print(f"  sensevoice  : {'found ' + sv if sv else 'MISSING'}")
        print(f"  cam++       : {'found ' + cp if cp else 'MISSING'}")
    except Exception as exc:
        print(f"  local models: could not resolve ({exc})")

    # ---- configured wiring ---------------------------------------------
    mp = cfg.vad.model_path
    print(f"  vad backend : {cfg.vad.backend}" + (f" (model={mp})" if mp else ""))
    if cfg.vad.backend == "silero":
        print(f"  vad model   : {'found' if mp and os.path.exists(mp) else 'MISSING'}")

    print(f"  asr backend : {cfg.asr.backend}  model={cfg.asr.model}")
    print(f"  tts backend : {cfg.tts.backend}  voice={cfg.tts.voice}")
    if cfg.tts.backend == "piper":
        voice_path = cfg.tts.model_path or ""
        print(f"  piper voice : {'found ' + voice_path if voice_path and os.path.isfile(voice_path) else 'MISSING'}")
    print(f"  llm backend : {cfg.llm.backend}  model={cfg.llm.model}")
    print(f"  voiceprint  : {cfg.voiceprint.backend} (enabled={cfg.voiceprint.enabled})")
    print(f"  vp store    : {cfg.voiceprint.store_path}")
    print(f"  memory root : {os.path.abspath(cfg.memory.root)}")

    # ---- do the configured servers actually answer? --------------------
    if cfg.llm.backend == "http":
        print(f"  llm server  : {_probe_url(cfg.llm.base_url.rstrip('/') + '/models')}")
    if cfg.asr.backend == "http":
        print(f"  asr server  : {_probe_url(cfg.asr.base_url.rstrip('/') + '/models')}")
    if cfg.tts.backend == "http":
        print(f"  tts server  : {_probe_url(cfg.tts.base_url.rstrip('/') + '/models')}")
    return 0


def cmd_enroll(args) -> int:
    from .voiceprint import VoiceprintStore
    from .audio import AudioFrame

    cfg = load_config(args.config)
    if args.store:
        cfg.voiceprint.store_path = args.store
    store = VoiceprintStore(cfg.voiceprint)
    samples, sr = read_wav(args.wav)
    emb = store.enroll(args.person_id, args.name, AudioFrame(samples, sr))
    store.save()
    print(f"enrolled {args.name!r} ({args.person_id}) dim={emb.size} -> {store.path}")
    print(f"  threshold={cfg.voiceprint.threshold} backend={cfg.voiceprint.backend}")
    return 0


def cmd_identify(args) -> int:
    from .voiceprint import VoiceprintStore
    from .audio import AudioFrame

    cfg = load_config(args.config)
    if args.store:
        cfg.voiceprint.store_path = args.store
    store = VoiceprintStore(cfg.voiceprint)
    samples, sr = read_wav(args.wav)
    res = store.identify(AudioFrame(samples, sr))
    print(f"accepted={res.accepted} person={res.name} score={res.score:.4f}")
    for pid, score in res.ranking:
        print(f"  {pid}: {score:.4f}")
    return 0


def cmd_transcribe(args) -> int:
    from .asr import build_asr
    from .audio import AudioFrame

    cfg = load_config(args.config)
    samples, sr = read_wav(args.wav)
    asr = build_asr(cfg.asr)
    print(asr.transcribe(AudioFrame(samples, sr)))
    return 0


def cmd_config(args) -> int:
    cfg = load_config(args.config)
    text = cfg.to_json()
    if args.write:
        cfg.save_json(args.write)
        print(f"wrote config to {args.write}")
    else:
        print(text)
    return 0


_EDGE_VOICES = [
    "zh-CN-XiaoxiaoNeural",   # female, warm — the current default
    "zh-CN-XiaoyiNeural",     # female, brighter
    "zh-CN-YunxiNeural",      # male, young
    "zh-CN-YunjianNeural",    # male, newsreader
    "zh-CN-YunyangNeural",    # male, calm
    "zh-CN-liaoning-XiaobeiNeural",  # female, north-eastern accent
    "zh-CN-shaanxi-XiaoniNeural",    # female, shaanxi accent
]


def cmd_voices(args) -> int:
    """List the voices this machine can speak with.

    Two kinds, deliberately: local piper voices (fast, offline) and Edge voices
    (natural, online).  Audition one with ``speak`` before wiring it in.
    """
    cfg = load_config(args.config)
    if getattr(args, "tts_backend", None):
        cfg.tts.backend = args.tts_backend

    print(f"current: tts.backend={cfg.tts.backend} voice={cfg.tts.voice}")
    if cfg.tts.model_path:
        exists = "found" if os.path.isfile(cfg.tts.model_path) else "MISSING"
        print(f"  piper voice {cfg.tts.model_path} [{exists}]")

    print("\nlocal piper voices:")
    scan_dirs = [args.dir] if getattr(args, "dir", None) else []
    scan_dirs += ["/home/whu/models/piper", os.path.expanduser("~/.local/share/piper")]
    seen = set()
    found = False
    for d in scan_dirs:
        if not d or not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".onnx"):
                continue
            path = os.path.join(d, name)
            if path in seen:
                continue
            seen.add(path)
            found = True
            print(f"  {name[:-5]:<24} {path}")
    if not found:
        print("  (none — download one, e.g. from hf-mirror.com/rhasspy/piper-voices)")

    print("\nEdge voices (online, need the network; use --tts-backend edge):")
    for v in _EDGE_VOICES:
        print(f"  {v}")

    print("\nexamples:")
    print('  qwen-voice-agent -c config.json speak "你好，我是你的语音助手。"')
    print('  qwen-voice-agent -c config.json speak "你好" --tts-voice zh-CN-YunxiNeural')
    print('  qwen-voice-agent -c config.json speak "你好" --tts-voice /path/to/voice.onnx')
    return 0


def cmd_speak(args) -> int:
    """Synthesise ``args.text`` and play (or write) it — the voice audition path."""
    from .sources import build_sink
    from .tts import build_tts

    cfg = load_config(args.config)
    _configure_logging(cfg, args.log_level)
    _apply_tts_overrides(cfg, args)

    tts = build_tts(cfg.tts)
    started = time.perf_counter()
    audio = tts.synthesize(args.text)
    elapsed = time.perf_counter() - started
    print(
        f"tts={cfg.tts.backend}({cfg.tts.voice}) "
        f"synth={elapsed * 1000:.0f}ms audio={audio.duration_s:.2f}s sr={audio.sample_rate}"
    )

    if args.sink == "null":
        return 0
    if args.sink == "file" or args.out:
        out = args.out or "speech.wav"
        from .audio import write_wav

        write_wav(out, audio.samples, audio.sample_rate)
        print(f"wrote {out}")
        return 0

    sink = build_sink("aplay", None, cfg.audio.output_device)
    sink.play(audio)
    close = getattr(tts, "close", None)
    if callable(close):
        close()
    return 0


_DISPATCH = {
    "run": cmd_run,
    "doctor": cmd_doctor,
    "enroll": cmd_enroll,
    "identify": cmd_identify,
    "transcribe": cmd_transcribe,
    "voices": cmd_voices,
    "speak": cmd_speak,
    "config": cmd_config,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _DISPATCH[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
