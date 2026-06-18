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
if [ -b /dev/sda2 ]; then
    mkdir -p /mnt/persist
    mount /dev/sda2 /mnt/persist 2>/dev/null || true
fi

# If cortex source is present (embedded in initramfs or on persist partition),
# exec into it. Otherwise fall back to a minimal shell for rescue.
if [ -f /app/src/__main__.py ]; then
    cd /app
    exec python3 -m src init
elif [ -f /mnt/persist/cortex/src/__main__.py ]; then
    cd /mnt/persist/cortex
    exec python3 -m src init
else
    echo "Cortex not found. Dropping to rescue shell."
    exec /bin/sh
fi
