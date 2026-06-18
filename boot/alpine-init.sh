#!/usr/bin/env bash
# Build an Alpine Linux initramfs with Cortex embedded.
# Produces: boot/initramfs.cpio.gz + boot/vmlinuz-lts
#
# Requirements: Alpine container or host with apk, abuild, mkinitfs
# Tested on: Alpine Linux 3.19 (x86_64)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
INITRAMFS_DIR="$BUILD_DIR/initramfs"

echo "=========================================="
echo "  Cortex Alpine Initramfs Builder"
echo "=========================================="

# Clean and prep
rm -rf "$BUILD_DIR"
mkdir -p "$INITRAMFS_DIR"/{bin,sbin,usr/bin,usr/sbin,lib,lib64,usr/lib,usr/lib64,proc,sys,dev,tmp,mnt,etc,app}

# ---------------------------------------------------------------------------
# 1. Install Alpine base packages (mini rootfs)
# ---------------------------------------------------------------------------
echo "[*] Installing Alpine mini rootfs..."
ALPINE_VERSION="v3.19"
ALPINE_ARCH="x86_64"
MINIROOTFS_URL="https://dl-cdn.alpinelinux.org/alpine/${ALPINE_VERSION}/releases/${ALPINE_ARCH}/alpine-minirootfs-latest-${ALPINE_ARCH}.tar.gz"

curl -sL "$MINIROOTFS_URL" | tar xz -C "$INITRAMFS_DIR" \
    bin/busybox bin/sh sbin/apk etc/apk lib/libc.musl* lib/ld-musl* 2>/dev/null || {
    echo "    (Using local busybox if available)"
    cp /bin/busybox "$INITRAMFS_DIR/bin/" 2>/dev/null || true
    cp /bin/sh "$INITRAMFS_DIR/bin/" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# 2. Install Python (static or minimal)
# ---------------------------------------------------------------------------
echo "[*] Installing Python..."
if command -v apk &>/dev/null; then
    apk add --root "$INITRAMFS_DIR" --initdb --no-cache \
        python3 py3-pip sqlite-libs 2>/dev/null || true
fi

# Fallback: copy host Python if Alpine package unavailable
if [ ! -f "$INITRAMFS_DIR/usr/bin/python3" ]; then
    PYTHON_BIN="$(command -v python3)"
    if [ -n "$PYTHON_BIN" ]; then
        echo "[*] Copying host Python from $PYTHON_BIN..."
        mkdir -p "$INITRAMFS_DIR/usr/bin"
        cp -L "$PYTHON_BIN" "$INITRAMFS_DIR/usr/bin/python3"
        # Copy shared libs (best-effort)
        ldd "$PYTHON_BIN" 2>/dev/null | grep '=> /' | awk '{print $3}' | while read lib; do
            cp -L "$lib" "$INITRAMFS_DIR/lib/" 2>/dev/null || true
        done || true
    fi
fi

# ---------------------------------------------------------------------------
# 3. Embed Cortex source
# ---------------------------------------------------------------------------
echo "[*] Embedding Cortex source..."
rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    "$REPO_ROOT/src/" "$INITRAMFS_DIR/app/src/" 2>/dev/null || \
    cp -r "$REPO_ROOT/src" "$INITRAMFS_DIR/app/"

# Copy boot assets
cp "$SCRIPT_DIR/init.sh" "$INITRAMFS_DIR/init"
chmod +x "$INITRAMFS_DIR/init"
cp "$SCRIPT_DIR/cortex-init.py" "$INITRAMFS_DIR/app/cortex-init.py"
chmod +x "$INITRAMFS_DIR/app/cortex-init.py"

# ---------------------------------------------------------------------------
# 4. Create essential device nodes
# ---------------------------------------------------------------------------
echo "[*] Creating device nodes..."
mkdir -p "$INITRAMFS_DIR/dev"
mknod "$INITRAMFS_DIR/dev/null"    c 1 3  2>/dev/null || true
mknod "$INITRAMFS_DIR/dev/zero"     c 1 5  2>/dev/null || true
mknod "$INITRAMFS_DIR/dev/random"   c 1 8  2>/dev/null || true
mknod "$INITRAMFS_DIR/dev/urandom"  c 1 9  2>/dev/null || true
mknod "$INITRAMFS_DIR/dev/tty"      c 5 0  2>/dev/null || true
mknod "$INITRAMFS_DIR/dev/tty1"     c 4 1  2>/dev/null || true
mknod "$INITRAMFS_DIR/dev/console"  c 5 1  2>/dev/null || true

# ---------------------------------------------------------------------------
# 5. Build CPIO archive
# ---------------------------------------------------------------------------
echo "[*] Building initramfs.cpio.gz..."
cd "$INITRAMFS_DIR"
find . | cpio -o -H newc 2>/dev/null | gzip -9 > "$BUILD_DIR/initramfs.cpio.gz"
echo "    Size: $(du -h "$BUILD_DIR/initramfs.cpio.gz" | cut -f1)"

# ---------------------------------------------------------------------------
# 6. Try to get a kernel
# ---------------------------------------------------------------------------
echo "[*] Looking for kernel..."
KERNEL=""
for path in /boot/vmlinuz-* /boot/vmlinuz /usr/lib/modules/*/vmlinuz; do
    if [ -f "$path" ]; then
        KERNEL="$path"
        break
    fi
done

if [ -n "$KERNEL" ]; then
    cp "$KERNEL" "$BUILD_DIR/vmlinuz"
    echo "    Kernel: $KERNEL → build/vmlinuz"
else
    echo "    (No kernel found — you'll need to supply your own vmlinuz)"
fi

echo ""
echo "=========================================="
echo "  Build complete: $BUILD_DIR/"
echo "=========================================="
echo "  initramfs.cpio.gz   → boot with this"
echo "  vmlinuz             → (if found above)"
echo ""
echo "  To test in QEMU:"
echo "    qemu-system-x86_64 -kernel $BUILD_DIR/vmlinuz \\"
echo "      -initrd $BUILD_DIR/initramfs.cpio.gz -append 'console=ttyS0' \\"
echo "      -nographic -m 2G"
echo ""
