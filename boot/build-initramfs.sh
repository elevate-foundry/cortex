#!/usr/bin/env bash
# Build a functional initramfs with Cortex as PID 1.
# Targeted copy: only Python binary + stdlib + aiohttp deps + specific .so files.
#
# Usage: bash boot/build-initramfs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
INITRAMFS_DIR="$BUILD_DIR/initramfs"
VENV_DIR="$BUILD_DIR/venv"

echo "=========================================="
echo "  Cortex Initramfs Builder (targeted)"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. Clean and prep
# ---------------------------------------------------------------------------
rm -rf "$INITRAMFS_DIR"
mkdir -p "$INITRAMFS_DIR"/{bin,sbin,usr/bin,usr/sbin,lib,lib64,usr/lib,usr/lib64,proc,sys,dev,tmp,mnt,etc,app,root}

# ---------------------------------------------------------------------------
# 2. Reuse existing venv or create one
# ---------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating Python virtualenv..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet aiohttp
fi

VENV_PYTHON="$VENV_DIR/bin/python3"
PYTHON_VER="$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
VENV_SITE="$VENV_DIR/lib/python${PYTHON_VER}/site-packages"

# ---------------------------------------------------------------------------
# 3. Collect ALL shared libraries needed (Python + extensions)
# ---------------------------------------------------------------------------
echo "[*] Collecting required shared libraries..."
NEEDED_LIBS="$BUILD_DIR/needed_libs.txt"
> "$NEEDED_LIBS"

# Python binary itself
ldd "$VENV_DIR/bin/python3" 2>/dev/null | grep '=> /' | awk '{print $3}' >> "$NEEDED_LIBS"

# All .so in venv site-packages
find "$VENV_SITE" -name "*.so" 2>/dev/null | while read so; do
    ldd "$so" 2>/dev/null | grep '=> /' | awk '{print $3}' >> "$NEEDED_LIBS"
done

# Python stdlib .so
find "/usr/lib/python${PYTHON_VER}" -name "*.so" 2>/dev/null | while read so; do
    ldd "$so" 2>/dev/null | grep '=> /' | awk '{print $3}' >> "$NEEDED_LIBS"
done

# Deduplicate and copy
sort -u "$NEEDED_LIBS" | while read lib; do
    if [ -f "$lib" ] && [ ! -f "$INITRAMFS_DIR$lib" ]; then
        mkdir -p "$INITRAMFS_DIR$(dirname "$lib")"
        cp -L "$lib" "$INITRAMFS_DIR$lib" 2>/dev/null || true
    fi
done

# Also copy ld-linux
cp -L /lib64/ld-linux-x86-64.so.2 "$INITRAMFS_DIR/lib64/" 2>/dev/null || \
    cp -L /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 "$INITRAMFS_DIR/lib/x86_64-linux-gnu/" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Copy Python binary + stdlib + site-packages
# ---------------------------------------------------------------------------
echo "[*] Copying Python..."
mkdir -p "$INITRAMFS_DIR/usr/bin"
cp -L "$VENV_PYTHON" "$INITRAMFS_DIR/usr/bin/python3"
ln -sf python3 "$INITRAMFS_DIR/usr/bin/python"

# Stdlib
mkdir -p "$INITRAMFS_DIR/usr/lib"
cp -r "/usr/lib/python${PYTHON_VER}" "$INITRAMFS_DIR/usr/lib/" 2>/dev/null || true

# Site-packages
mkdir -p "$INITRAMFS_DIR/usr/local/lib/python${PYTHON_VER}/site-packages"
cp -r "$VENV_SITE/"* "$INITRAMFS_DIR/usr/local/lib/python${PYTHON_VER}/site-packages/" 2>/dev/null || true

# System dist-packages
SYS_SITE="/usr/lib/python${PYTHON_VER}/dist-packages"
if [ -d "$SYS_SITE" ]; then
    mkdir -p "$INITRAMFS_DIR$SYS_SITE"
    cp -r "$SYS_SITE/"* "$INITRAMFS_DIR$SYS_SITE/" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 5. Copy essential binaries (busybox + minimal set)
# ---------------------------------------------------------------------------
echo "[*] Copying essential binaries..."
for bin_src in /bin/busybox /bin/sh /bin/bash /bin/mount /bin/umount \
               /bin/mkdir /bin/rm /bin/cp /bin/mv /bin/cat /bin/ls \
               /bin/echo /bin/grep /bin/sed /bin/awk /bin/chmod; do
    if [ -f "$bin_src" ]; then
        bin_dst="$INITRAMFS_DIR$bin_src"
        mkdir -p "$(dirname "$bin_dst")"
        cp -L "$bin_src" "$bin_dst" 2>/dev/null || true
    fi
done

# busybox symlinks
if [ -f "$INITRAMFS_DIR/bin/busybox" ]; then
    for util in sh mount umount mkdir rm cp mv cat ls echo grep sed awk chmod; do
        ln -sf /bin/busybox "$INITRAMFS_DIR/bin/$util" 2>/dev/null || true
    done
fi

# ---------------------------------------------------------------------------
# 6. Embed Cortex source
# ---------------------------------------------------------------------------
echo "[*] Embedding Cortex source..."
mkdir -p "$INITRAMFS_DIR/app"
cp -r "$REPO_ROOT/src" "$INITRAMFS_DIR/app/"

cp "$SCRIPT_DIR/init.sh" "$INITRAMFS_DIR/init"
chmod +x "$INITRAMFS_DIR/init"
cp "$SCRIPT_DIR/cortex-init.py" "$INITRAMFS_DIR/app/cortex-init.py"
chmod +x "$INITRAMFS_DIR/app/cortex-init.py"

# ---------------------------------------------------------------------------
# 7. Device nodes
# ---------------------------------------------------------------------------
echo "[*] Creating device nodes..."
mkdir -p "$INITRAMFS_DIR/dev"
mknod "$INITRAMFS_DIR/dev/null"    c 1 3  2>/dev/null || true
mknod "$INITRAMFS_DIR/dev/zero"     c 1 5  2>/dev/null || true
mknod "$INITRAMFS_DIR/dev/random"   c 1 8  2>/dev/null || true
mknod "$INITRAMFS_DIR/dev/urandom"  c 1 9  2>/dev/null || true
mknod "$INITRAMFS_DIR/dev/tty"      c 5 0  2>/dev/null || true
mknod "$INITRAMFS_DIR/dev/tty1"     c 4 1  2>/dev/null || true
mknod "$INITRAMFS_DIR/dev/ttyS0"    c 4 64 2>/dev/null || true
mknod "$INITRAMFS_DIR/dev/console"  c 5 1  2>/dev/null || true

# ---------------------------------------------------------------------------
# 8. Build CPIO
# ---------------------------------------------------------------------------
echo "[*] Building initramfs.cpio.gz..."
cd "$INITRAMFS_DIR"
UNSIZE="$(du -sh . | cut -f1)"
echo "    Uncompressed: $UNSIZE"
find . -print0 | cpio --null -o -H newc 2>/dev/null | gzip -9 > "$BUILD_DIR/initramfs.cpio.gz"
CPIO_SIZE="$(du -h "$BUILD_DIR/initramfs.cpio.gz" | cut -f1)"
echo "    Compressed: $CPIO_SIZE"

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Build complete"
echo "=========================================="
ls -lh "$BUILD_DIR/" | grep -E "initramfs|vmlinuz"
echo ""
