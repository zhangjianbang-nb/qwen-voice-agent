"""Command-line entry point.

Subcommands:

* ``run``        — start the agent (the 24/7 loop; bound it with ``--seconds`` /
                   ``--turns`` for a smoke run);
* ``doctor``     — report what this machine can and cannot do (deps, audio
                   devices, model files) so a failing deployment is diagnosable;
* ``enroll``     — add a speaker to the voiceprint store from a WAV file;
* ``identify``   — identify the speaker in a WAV file against the store;
* ``transcribe`` — run the configured ASR backend on a WAV file;
* ``config``     — print the default configuration, or write a template JSON.

The CLI is intentionally thin: all logic lives in the package so it is testable
without a terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import List, Optional

from .audio import read_wav
from .config import VoiceAgentConfig, load_config


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qwen-voice-agent", description="24/7 full-duplex voice agent")
    p.add_argument("--config", "-c", help="path to a JSON config file")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start the agent loop")
    run.add_argument("--source", choices=["null", "file", "mic"], default="null")
    run.add_argument("--source-path", help="WAV path when --source=file")
    run.add_argument("--sink", choices=["null", "file", "speaker"], default="null")
    run.add_argument("--sink-path", help="WAV path when --sink=file")
    run.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    run.add_argument("--turns", type=int, default=None, help="stop after N turns")
    run.add_argument("--memory-root", default=None, help="where memory files live")

    sub.add_parser("doctor", help="report environment readiness")

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


def cmd_run(args) -> int:
    from .session import assemble

    cfg = load_config(args.config)
    if args.seconds is not None:
        cfg.session.max_seconds = args.seconds
    if args.turns is not None:
        cfg.session.max_turns = args.turns
    pipeline, session = assemble(
        cfg,
        source_kind=args.source,
        source_path=args.source_path,
        sink_kind=args.sink,
        sink_path=args.sink_path,
        memory_root=args.memory_root,
    )

    def report(turn):
        tag = " [interrupted]" if turn.interrupted else ""
        who = turn.person_name or "unknown"
        print(f"[{turn.index}] {who}: {turn.user_text!r} -> {turn.reply!r}{tag}")

    session.on_turn = report
    turns = session.run()
    print(f"session ended after {turns} turn(s)")
    return 0


def cmd_doctor(args) -> int:
    cfg = load_config(args.config)
    print("qwen-voice-agent doctor")
    print(f"  python      : {sys.version.split()[0]}")
    for mod, why in [
        ("numpy", "core requirement"),
        ("httpx", "http ASR/TTS/LLM backends"),
        ("onnxruntime", "silero VAD / onnx speaker embedding"),
        ("sounddevice", "microphone capture / playback"),
        ("scipy", "polyphase resampling"),
    ]:
        try:
            __import__(mod)
            print(f"  {mod:12s}: OK   ({why})")
        except Exception:
            print(f"  {mod:12s}: MISSING   ({why})")

    # Audio devices
    if shutil.which("arecord"):
        import subprocess

        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True).stdout
        have = any("card" in line for line in out.splitlines())
        print(f"  capture dev : {'present' if have else 'none'}")
    else:
        print("  capture dev : arecord not installed")

    # VAD model
    mp = cfg.vad.model_path
    print(f"  vad backend : {cfg.vad.backend}" + (f" (model={mp})" if mp else ""))
    if cfg.vad.backend == "silero":
        print(f"  vad model   : {'found' if mp and os.path.exists(mp) else 'MISSING'}")

    print(f"  asr backend : {cfg.asr.backend}  model={cfg.asr.model}")
    print(f"  tts backend : {cfg.tts.backend}  model={cfg.tts.model}")
    print(f"  llm backend : {cfg.llm.backend}  model={cfg.llm.model}")
    print(f"  voiceprint  : {cfg.voiceprint.backend} (enabled={cfg.voiceprint.enabled})")
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


_DISPATCH = {
    "run": cmd_run,
    "doctor": cmd_doctor,
    "enroll": cmd_enroll,
    "identify": cmd_identify,
    "transcribe": cmd_transcribe,
    "config": cmd_config,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _DISPATCH[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
