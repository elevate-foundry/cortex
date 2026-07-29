#!/bin/bash
# cortex-mount-watcher.sh — Linux variant
# Triggered by udev rule when the CORTEX drive mounts.
#
# TODO: Install with:
#   sudo cp 99-cortex.rules /etc/udev/rules.d/
#   sudo udevadm control --reload-rules
#
# Example udev rule (99-cortex.rules):
#   ACTION=="add", SUBSYSTEM=="block", ENV{ID_FS_LABEL}=="CORTEX", \
#     RUN+="/media/$USER/CORTEX/cortex/bin/cortex-mount-watcher.sh"

set -euo pipefail

# Auto-detect mount point (common locations)
for MOUNT in "/media/$USER/CORTEX/cortex" "/mnt/CORTEX/cortex" "/run/media/$USER/CORTEX/cortex"; do
    if [ -d "$MOUNT/src" ]; then
        CORTEX_HOME="$MOUNT"
        break
    fi
done

if [ -z "${CORTEX_HOME:-}" ]; then
    echo "CORTEX drive not found at expected mount points"
    exit 0
fi

VENV="$CORTEX_HOME/.venv/bin/python3"
LOG="$CORTEX_HOME/logs/daemon.log"
PID_FILE="$CORTEX_HOME/logs/daemon.pid"

# Bail if already running
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    exit 0
fi

# Create logs dir if needed
mkdir -p "$CORTEX_HOME/logs"

# Run wake lifecycle
"$VENV" -c "
import sys
sys.path.insert(0, '$CORTEX_HOME')
from src.lifecycle import wake
checkpoint = wake()
if checkpoint:
    print(f'  Resumed: boot #{checkpoint.boot_count}')
" >> "$LOG" 2>&1 || true

# TTS announcement (espeak or piper, whichever is available)
if command -v espeak-ng &>/dev/null; then
    espeak-ng "Cortex waking up" &
elif command -v espeak &>/dev/null; then
    espeak "Cortex waking up" &
elif command -v piper &>/dev/null; then
    echo "Cortex waking up" | piper --output-raw | aplay -r 22050 -f S16_LE &
fi

# Desktop notification (freedesktop)
if command -v notify-send &>/dev/null; then
    notify-send "Cortex" "Daemon starting..." --icon=computer &
fi

# Start daemon
export CORTEX_HOME
cd "$CORTEX_HOME"
nohup "$VENV" -m src daemon --port 11411 >> "$LOG" 2>&1 &
DAEMON_PID=$!
echo "$DAEMON_PID" > "$PID_FILE"

# Wait for healthy (up to 15s)
for i in $(seq 1 30); do
    if curl -sf http://localhost:11411/health > /dev/null 2>&1; then
        MODELS=$(curl -s http://localhost:11411/health | python3 -c "import sys,json; print(json.load(sys.stdin).get('models_ready',0))" 2>/dev/null || echo "?")
        if command -v notify-send &>/dev/null; then
            notify-send "Cortex" "$MODELS models loaded. Ready." --icon=computer
        fi
        if command -v espeak-ng &>/dev/null; then
            espeak-ng "Cortex ready. $MODELS models loaded." &
        fi
        exit 0
    fi
    sleep 0.5
done

if command -v notify-send &>/dev/null; then
    notify-send "Cortex" "Failed to start. Check logs." --icon=dialog-error
fi
exit 1
