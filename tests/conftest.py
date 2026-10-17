"""Shared test fixtures.

Puts ``src/`` on the path (src-layout, not installed) and locates the real speech
WAVs used by the perception acceptance tests.  The WAVs live in the sibling
``omni-bot`` project; if they are absent the dependent tests skip rather than
fail, so the suite still runs on a bare checkout.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Candidate locations for real speech audio and the Silero VAD model.
_ASSET_DIRS = [
    "/home/whu/workspace/servers/omni-bot/assets/ref_audio",
    os.path.join(ROOT, "tests", "assets"),
]
_VAD_DIRS = [
    "/home/whu/workspace/servers/omni-bot/assets/vad_model",
    os.path.join(ROOT, "tests", "assets"),
]


def _find(dirs, name):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


@pytest.fixture(scope="session")
def speech_wav():
    p = _find(_ASSET_DIRS, "ref_minicpm_signature.wav")
    if not p:
        pytest.skip("no real speech WAV available (omni-bot assets missing)")
    return p


@pytest.fixture(scope="session")
def speech_wav_2():
    p = _find(_ASSET_DIRS, "ref_en_dlc_1.wav")
    if not p:
        pytest.skip("no second speech WAV available")
    return p


@pytest.fixture(scope="session")
def silero_model():
    p = _find(_VAD_DIRS, "silero_vad.onnx")
    if not p:
        pytest.skip("silero_vad.onnx not available")
    return p
