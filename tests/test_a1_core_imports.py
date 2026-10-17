"""A1 — the runtime core must not pull in heavy dependencies.

The whole point of keeping the core on numpy is that a voice agent is testable
without a GPU, a sound card, or a model.  These checks run in a *fresh*
interpreter (a subprocess), because once another test in the same session has
imported scipy, ``sys.modules`` would be polluted and the assertion meaningless.
"""

import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")

CORE_MODULES = [
    "qwen_voice_agent",
    "qwen_voice_agent.audio",
    "qwen_voice_agent.vad",
    "qwen_voice_agent.voiceprint",
    "qwen_voice_agent.asr",
    "qwen_voice_agent.tts",
    "qwen_voice_agent.llm",
    "qwen_voice_agent.sources",
    "qwen_voice_agent.pipeline",
    "qwen_voice_agent.session",
    "qwen_voice_agent.cli",
    "qwen_voice_agent.memory",
]

HEAVY = ["torch", "scipy", "onnxruntime", "sounddevice", "librosa", "transformers"]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_core_imports_do_not_pull_heavy_deps():
    imports = "".join(f"import {m}\n" for m in CORE_MODULES)
    code = (
        f"import sys; sys.path.insert(0, {SRC!r})\n"
        f"{imports}"
        f"print([m for m in {HEAVY!r} if m in sys.modules])\n"
    )
    out = _run(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"core import pulled heavy deps: {out.stdout.strip()}"


def test_isolated_interpreter_imports_package():
    code = f"import sys; sys.path.insert(0, {SRC!r}); import qwen_voice_agent; print('ok')"
    out = _run(code)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
