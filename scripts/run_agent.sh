#!/usr/bin/env bash
# Start the 24/7 voice agent with this deployment's configuration.
#
#   scripts/run_agent.sh                 # live microphone + speakers (foreground)
#   scripts/run_agent.sh --seconds 30    # a bounded smoke run
#   QVA_SINK=file scripts/run_agent.sh --source file --source-path x.wav \
#       --sink file --sink-path out.wav
#
# Everything is overridable from the environment so a systemd unit can set it
# without editing this file: QVA_PYTHON, QVA_CONFIG, QVA_SOURCE, QVA_SINK.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${QVA_PYTHON:-/home/whu/miniconda3/envs/dl/bin/python}"
CONFIG="${QVA_CONFIG:-$HERE/config.json}"
SOURCE="${QVA_SOURCE:-mic}"
SINK="${QVA_SINK:-aplay}"

if [[ ! -x "$PY" ]]; then
  echo "python interpreter not found: $PY (set QVA_PYTHON)" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG (set QVA_CONFIG)" >&2
  exit 1
fi

# `-u` matters when this is redirected to a log file: Python buffers stdout when
# it is not a TTY, so the startup banner and every turn would sit in a buffer
# instead of appearing in the log while the agent runs.
exec "$PY" -u -m qwen_voice_agent.cli -c "$CONFIG" run --source "$SOURCE" --sink "$SINK" "$@"
