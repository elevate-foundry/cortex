#!/bin/sh
# Cortex initramfs init — minimal shell bootstrap.
# This runs inside the initramfs before switching to real root.
# Its only job: mount essentials, then exec into cortex-init.py

set -e

# Mount kernel virtual filesystems
mount -t proc  proc  /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev
mount -t tmpfs  tmpfs  /tmp

# Create basic device nodes if devtmpfs missing
[ -c /dev/null ] || mknod /dev/null c 1 3
[ -c /dev/tty ]  || mknod /dev/tty  c 5 0
[ -c /dev/tty1 ] || mknod /dev/tty1 c 4 1
[ -c /dev/random ] || mknod /dev/random c 1 8
[ -c /dev/urandom ] || mknod /dev/urandom c 1 9

# Set hostname
echo "cortex-ai" > /proc/sys/kernel/hostname

# Bring up loopback
ip link set lo up 2>/dev/null || true

# Export minimal env
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CORTEX_BOOT=1
export CORTEX_INITRAMFS=1

# If we have a persistent overlay (e.g. thumbdrive partition), mount it
PERSIST_DEV=$(blkid -L CORTEX 2>/dev/null || true)
if [ -n "$PERSIST_DEV" ] && [ -b "$PERSIST_DEV" ]; then
    mkdir -p /mnt/cortex
    mount -t ext4 "$PERSIST_DEV" /mnt/cortex 2>/dev/null || true
fi

# Exec into the Cortex init script (Python PID 1)
if [ -f /app/cortex-init.py ]; then
    cd /app
    exec python3 /app/cortex-init.py
elif [ -f /mnt/cortex/app/cortex-init.py ]; then
    cd /mnt/cortex/app
    exec python3 /mnt/cortex/app/cortex-init.py
else
    echo "Cortex not found. Dropping to rescue shell."
    exec /bin/sh
fi
