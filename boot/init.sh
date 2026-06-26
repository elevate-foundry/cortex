#!/bin/sh
# Cortex initramfs init — minimal shell bootstrap.
# This runs inside the initramfs before switching to real root.
# Its only job: mount essentials, detect hardware, then exec into cortex-init.py
#
# Supports: x86_64, aarch64, any Linux-capable device
# Boot time target: <2 seconds to Python handoff

# Don't exit on error — we need to stay alive as PID 1
set +e

echo "[cortex] init.sh starting (PID $$)"

# ---------------------------------------------------------------------------
# 1. Mount kernel virtual filesystems
# ---------------------------------------------------------------------------
mount -t proc  proc  /proc 2>/dev/null
mount -t sysfs sysfs /sys  2>/dev/null
mount -t devtmpfs devtmpfs /dev 2>/dev/null
mkdir -p /dev/pts /dev/shm /tmp /run /mnt/cortex
mount -t devpts devpts /dev/pts 2>/dev/null
mount -t tmpfs  tmpfs  /tmp -o size=512M 2>/dev/null
mount -t tmpfs  tmpfs  /run -o size=64M 2>/dev/null

# Create basic device nodes if devtmpfs missing
[ -c /dev/null ]    || mknod -m 666 /dev/null c 1 3
[ -c /dev/zero ]    || mknod -m 666 /dev/zero c 1 5
[ -c /dev/tty ]     || mknod -m 666 /dev/tty  c 5 0
[ -c /dev/tty1 ]    || mknod -m 620 /dev/tty1 c 4 1
[ -c /dev/ttyS0 ]   || mknod -m 660 /dev/ttyS0 c 4 64
[ -c /dev/random ]  || mknod -m 444 /dev/random c 1 8
[ -c /dev/urandom ] || mknod -m 444 /dev/urandom c 1 9
[ -c /dev/console ] || mknod -m 600 /dev/console c 5 1

# ---------------------------------------------------------------------------
# 2. Parse kernel command line
# ---------------------------------------------------------------------------
CMDLINE=$(cat /proc/cmdline 2>/dev/null || echo "")
RESCUE=0
if echo "$CMDLINE" | grep -q "cortex.rescue=1"; then
    RESCUE=1
fi

# ---------------------------------------------------------------------------
# 3. Set hostname and networking basics
# ---------------------------------------------------------------------------
echo "cortex-ai" > /proc/sys/kernel/hostname 2>/dev/null
ip link set lo up 2>/dev/null || ifconfig lo up 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. GPU kernel module loading (early, before Python)
# ---------------------------------------------------------------------------
# Load NVIDIA if hardware present
if [ -d /sys/bus/pci ] && grep -qi "10de" /sys/bus/pci/devices/*/vendor 2>/dev/null; then
    echo "[cortex] NVIDIA GPU detected, loading driver..."
    modprobe nvidia 2>/dev/null || modprobe nouveau 2>/dev/null || true
    modprobe nvidia_uvm 2>/dev/null || true
fi

# Load AMD GPU
if [ -d /sys/bus/pci ] && grep -qi "1002" /sys/bus/pci/devices/*/vendor 2>/dev/null; then
    echo "[cortex] AMD GPU detected, loading amdgpu..."
    modprobe amdgpu 2>/dev/null || true
fi

# Intel GPU (usually auto-loaded but ensure it)
if [ -d /sys/bus/pci ] && grep -qi "8086" /sys/bus/pci/devices/*/vendor 2>/dev/null; then
    modprobe i915 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 5. Mount CORTEX persistence partition (USB stick data partition)
# ---------------------------------------------------------------------------
# Wait briefly for USB storage to enumerate
sleep 0.5

PERSIST_DEV=""
if command -v blkid >/dev/null 2>&1; then
    PERSIST_DEV=$(blkid -L CORTEX 2>/dev/null || true)
fi

# Fallback: scan partitions if blkid didn't find it
if [ -z "$PERSIST_DEV" ]; then
    for dev in /dev/sd[a-z][1-9]* /dev/nvme*p[1-9]* /dev/mmcblk*p[1-9]*; do
        [ -b "$dev" ] || continue
        if e2label "$dev" 2>/dev/null | grep -q "CORTEX"; then
            PERSIST_DEV="$dev"
            break
        fi
    done
fi

if [ -n "$PERSIST_DEV" ] && [ -b "$PERSIST_DEV" ]; then
    echo "[cortex] Mounting CORTEX partition: $PERSIST_DEV"
    mount -t ext4 -o rw,noatime "$PERSIST_DEV" /mnt/cortex 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 6. Export environment
# ---------------------------------------------------------------------------
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/app
export CORTEX_BOOT=1
export CORTEX_INITRAMFS=1
export HOME=/root
export TERM=linux

# If CORTEX partition mounted, set persistence env
if mountpoint -q /mnt/cortex 2>/dev/null; then
    export CORTEX_DATA_DIR=/mnt/cortex
    export CORTEX_DB=/mnt/cortex/var/lib/cortex.db
    echo "[cortex] Persistence: /mnt/cortex"
else
    export CORTEX_DB=/tmp/cortex.db
    echo "[cortex] No persistence (tmpfs mode)"
fi

# ---------------------------------------------------------------------------
# 7. Rescue mode check
# ---------------------------------------------------------------------------
if [ "$RESCUE" = "1" ]; then
    echo "[cortex] RESCUE MODE — dropping to shell"
    echo "  Type 'exit' to continue boot, or debug from here."
    exec /bin/sh
fi

# ---------------------------------------------------------------------------
# 8. Exec into Cortex Python init (becomes PID 1)
# ---------------------------------------------------------------------------
PYTHON=$(command -v python3 2>/dev/null || echo "/usr/bin/python3")

# Search for cortex-init.py in order of preference
for init_path in \
    /app/cortex-init.py \
    /mnt/cortex/app/cortex-init.py \
    /app/boot/cortex-init.py \
    /opt/cortex/boot/cortex-init.py; do
    if [ -f "$init_path" ]; then
        echo "[cortex] Handing off to $init_path ($(date +%H:%M:%S))"
        cd "$(dirname "$init_path")"
        exec "$PYTHON" "$init_path"
    fi
done

# Absolute fallback
echo "[cortex] ERROR: cortex-init.py not found!"
echo "  Searched: /app/ /mnt/cortex/app/ /opt/cortex/"
echo "  Dropping to rescue shell."
exec /bin/sh
