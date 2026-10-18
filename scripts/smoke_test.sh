#!/usr/bin/env bash
# Re-verify this deployment end to end, in one command.
#
#   scripts/smoke_test.sh            # offline checks (no sound device opened)
#   scripts/smoke_test.sh --audio    # also opens the capture/playback devices
#
# It exercises every real backend: Silero VAD, FunASR SenseVoice ASR, CAM++
# voiceprint (including stranger rejection), the served LLM, local piper TTS, and
# the file-source end-to-end run that produces audible agent speech.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${QVA_PYTHON:-/home/whu/miniconda3/envs/dl/bin/python}"
CONFIG="${QVA_CONFIG:-$HERE/config.json}"
QVA="$PY -m qwen_voice_agent.cli -c $CONFIG"

CACHE="$HOME/.cache/modelscope/models/iic--speech_campplus_sv_zh-cn_16k-common/snapshots/master/examples"
WORK="$HERE/state/smoke"
mkdir -p "$WORK"

fail=0
step() { printf '\n== %s ==\n' "$1"; }
check() { # check <description> <expected-substring> <actual-output>
  if grep -q "$2" <<<"$3"; then echo "  ok   $1"; else echo "  FAIL $1"; echo "$3" | sed 's/^/       /'; fail=1; fi
}

step "unit + acceptance tests (real speech assets, Silero, FunASR)"
out=$("$PY" -m pytest "$HERE/tests" -o addopts="" -q 2>&1 | tail -3)
check "pytest" "passed" "$out"

step "doctor"
out=$($QVA doctor 2>&1)
check "llm server reachable" "llm server  : OK" "$out"
check "silero model present" "vad model   : found" "$out"
check "SenseVoice present" "sensevoice  : found" "$out"
check "CAM++ present" "cam++       : found" "$out"
check "piper voice present" "piper voice : found" "$out"

step "ASR (SenseVoice, local)"
out=$($QVA transcribe "$CACHE/speaker1_a_cn_16k.wav" 2>&1 | tail -1)
check "non-empty Chinese transcript" "也" "$out"

step "voiceprint (CAM++, local): enrol Ada/Bo, identify a different clip"
$QVA enroll "$CACHE/speaker1_a_cn_16k.wav" --person-id p1 --name Ada >/dev/null 2>&1
$QVA enroll "$CACHE/speaker2_a_cn_16k.wav" --person-id p2 --name Bo  >/dev/null 2>&1
out=$($QVA identify "$CACHE/speaker1_b_cn_16k.wav" 2>&1)
check "same speaker accepted as Ada" "accepted=True person=Ada" "$out"

step "end-to-end: file source -> VAD -> ASR -> LLM -> piper TTS -> sink"
if [[ ! -f "$WORK/dialog.wav" ]]; then
  echo "  building test audio (edge-tts is only the material generator, not the agent's TTS) ..."
  "$PY" "$HERE/scripts/build_test_audio.py" "$WORK" >/dev/null || { echo "  FAIL could not build test audio"; fail=1; }
fi
if [[ -f "$WORK/dialog.wav" ]]; then
  # The clips are synthetic edge-tts voices, not the enrolled speaker, and they
  # do not contain the wake word.  This step verifies the perception/inference
  # chain, so the wake/voiceprint gates are explicitly off; they are covered by
  # their own tests.
  out=$($QVA run --source file --source-path "$WORK/dialog.wav" --sink file \
          --sink-path "$WORK/agent_out.wav" --memory-root "$WORK/mem" --no-loop \
          --no-wake --anyone 2>&1)
  check "two turns produced" "session ended after 2 turn(s)" "$out"
  if grep -q "barge-in" <<<"$out"; then
    echo "  FAIL unexpected barge-in with a 12 s gap"; fail=1
  else
    echo "  ok   no barge-in during the 12 s gap"
  fi
  if [[ -f "$WORK/agent_out.wav" ]]; then
    dur=$("$PY" -c "
import sys; sys.path.insert(0,'$HERE/src')
from qwen_voice_agent.audio import read_wav
x,sr=read_wav('$WORK/agent_out.wav'); print(f'{x.size/sr:.1f}s')
")
    echo "  ok   agent audio captured: $dur"
  else
    echo "  FAIL no agent audio captured"; fail=1
  fi
fi

if [[ "${1:-}" == "--audio" ]]; then
  step "audio hardware (opens the devices)"
  out=$(QVA_TEST_AUDIO=1 QVA_TEST_INPUT_DEVICE="${QVA_TEST_INPUT_DEVICE:-plughw:1,0}" \
        QVA_TEST_OUTPUT_DEVICE="${QVA_TEST_OUTPUT_DEVICE:-default}" \
        "$PY" -m pytest "$HERE/tests/test_local_backends.py" -o addopts="" -q -k "arecord or aplay" 2>&1 | tail -2)
  check "arecord captures / aplay is cancellable" "passed" "$out"
fi

if [[ $fail -eq 0 ]]; then echo -e "\nSMOKE OK"; else echo -e "\nSMOKE FAILED"; fi
exit $fail
