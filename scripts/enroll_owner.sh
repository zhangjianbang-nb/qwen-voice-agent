#!/usr/bin/env bash
# Enrol the owner's voice from the microphone, then (re)start the agent.
#
# The agent refuses every voice until the owner is enrolled
# (`wake.require_enrolled=true`), so this is the one-time setup step.  It stops a
# running agent first because the built-in DMIC is exclusive: only one process
# can open `plughw:1,6` at a time.
#
#   bash scripts/enroll_owner.sh          # 15 s of speech, then start the agent
#   QVA_ENROLL_SECONDS=20 bash scripts/enroll_owner.sh
#
# Speak naturally for the whole recording; several sentences give CAM++ a much
# sharper profile than one short phrase.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${QVA_PYTHON:-/home/whu/miniconda3/envs/dl/bin/python}"
CONFIG="${QVA_CONFIG:-$HERE/config.json}"
DEVICE="${QVA_INPUT_DEVICE:-plughw:1,6}"
SECONDS_TO_RECORD="${QVA_ENROLL_SECONDS:-15}"
PIDFILE="$HERE/state/agent.pid"

# ---- stop a running agent so the microphone is free -----------------------
if [[ -f "$PIDFILE" ]]; then
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "stopping agent (pid $pid) to free $DEVICE ..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
  fi
fi

mkdir -p "$HERE/state/enroll"
WAV="$HERE/state/enroll/owner.wav"

echo
echo "请对着麦克风连续说 ${SECONDS_TO_RECORD} 秒（3-5 句话，内容随意）："
echo "  例如：小丽同学，你好，我是这个电脑的主人。今天天气不错。"
echo
arecord -D "$DEVICE" -f S16_LE -r 16000 -c 1 -d "$SECONDS_TO_RECORD" "$WAV"

echo
echo "转写结果（确认这段确实是你，而不是电视/视频的声音）："
"$PY" -m qwen_voice_agent.cli -c "$CONFIG" transcribe "$WAV"
echo
read -r -p "确认是你的声音就按回车注册；否则 Ctrl-C 后重跑本脚本: " _

"$PY" -m qwen_voice_agent.cli -c "$CONFIG" enroll "$WAV" \
  --person-id owner --name "主人"

echo
echo "声纹已注册。启动 agent ..."
exec bash "$HERE/scripts/run_agent.sh" "$@"
