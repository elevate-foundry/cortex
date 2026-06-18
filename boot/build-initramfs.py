#!/usr/bin/env python3
"""
Build a functional initramfs with Cortex as PID 1.

This script creates a minimal Linux initramfs containing:
  - Python 3 with stdlib + aiohttp
  - Cortex source code
  - PID-1 init script (init.sh)
  - Essential binaries (busybox)

Produces: build/initramfs.cpio.gz
"""

import gzip
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.resolve()
BUILD_DIR = REPO_ROOT / "build"
INITRAMFS_DIR = BUILD_DIR / "initramfs"
VENV_DIR = BUILD_DIR / "venv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    return result


def ldd_libs(binary: Path) -> list[Path]:
    """Return list of shared libraries needed by a binary."""
    result = run(["ldd", str(binary)])
    libs = []
    for line in result.stdout.splitlines():
        if "=>" in line and "/" in line:
            parts = line.split("=>")
            if len(parts) == 2:
                lib_path = parts[1].split()[0].strip()
                if lib_path.startswith("/"):
                    libs.append(Path(lib_path))
    return libs


def copy_file(src: Path, dst: Path):
    """Copy a file, creating parent directories."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_symlink():
        # Resolve symlink and copy the target
        real = src.resolve()
        shutil.copy2(real, dst)
    else:
        shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, exclude=None):
    """Copy a directory tree."""
    if exclude is None:
        exclude = {".git", "__pycache__", "*.pyc"}
    if not src.exists():
        return
    for item in src.rglob("*"):
        if any(part.startswith(".") or part in exclude for part in item.parts):
            continue
        rel = item.relative_to(src)
        dst_item = dst / rel
        if item.is_dir():
            dst_item.mkdir(parents=True, exist_ok=True)
        else:
            copy_file(item, dst_item)


def collect_needed_libs(initramfs_dir: Path, binaries: list[Path]) -> set[Path]:
    """Recursively collect all shared libraries needed by binaries."""
    needed = set()
    to_process = list(binaries)
    seen = set()

    while to_process:
        binary = to_process.pop()
        if binary in seen:
            continue
        seen.add(binary)
        for lib in ldd_libs(binary):
            if lib not in needed and lib.exists():
                needed.add(lib)
                to_process.append(lib)

    return needed


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------

def step_create_venv():
    print("[*] Creating Python virtualenv...")
    if not VENV_DIR.exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        pip = VENV_DIR / "bin" / "pip"
        subprocess.run([str(pip), "install", "--quiet", "aiohttp"], check=True)


def step_copy_libs():
    print("[*] Collecting shared libraries...")
    venv_python = VENV_DIR / "bin" / "python3"

    # Collect all .so files
    so_files = []
    site_packages = VENV_DIR / "lib"
    for sp in site_packages.rglob("*.so"):
        so_files.append(sp)

    stdlib_dir = Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}")
    if stdlib_dir.exists():
        for sp in stdlib_dir.rglob("*.so"):
            so_files.append(sp)

    # Collect all needed libs
    libs = collect_needed_libs(INITRAMFS_DIR, so_files + [venv_python])

    for lib in libs:
        dst = INITRAMFS_DIR / lib.relative_to("/")
        copy_file(lib, dst)

    # ld-linux
    for ld_path in [
        Path("/lib64/ld-linux-x86-64.so.2"),
        Path("/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"),
    ]:
        if ld_path.exists():
            dst = INITRAMFS_DIR / ld_path.relative_to("/")
            copy_file(ld_path, dst)
            break


def step_copy_python():
    print("[*] Copying Python...")
    venv_python = VENV_DIR / "bin" / "python3"

    # Python binary
    dst_bin = INITRAMFS_DIR / "usr" / "bin" / "python3"
    copy_file(venv_python, dst_bin)
    os.chmod(dst_bin, dst_bin.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Python stdlib
    stdlib = Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}")
    if stdlib.exists():
        dst_stdlib = INITRAMFS_DIR / "usr" / "lib" / stdlib.name
        copy_tree(stdlib, dst_stdlib)

    # Venv site-packages
    site_pkg = VENV_DIR / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if site_pkg.exists():
        dst_site = INITRAMFS_DIR / "usr" / "local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        copy_tree(site_pkg, dst_site)

    # System dist-packages
    sys_site = Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages")
    if sys_site.exists():
        dst_sys = INITRAMFS_DIR / sys_site.relative_to("/")
        copy_tree(sys_site, dst_sys)


def step_copy_binaries():
    print("[*] Copying essential binaries...")
    binaries = [
        "/bin/busybox", "/bin/sh", "/bin/bash",
        "/bin/mkdir", "/bin/rm", "/bin/cp", "/bin/mv", "/bin/cat", "/bin/ls",
        "/bin/echo", "/bin/grep", "/bin/sed", "/bin/awk", "/bin/chmod",
    ]
    for src_path in binaries:
        src = Path(src_path)
        if src.exists():
            dst = INITRAMFS_DIR / src.relative_to("/")
            copy_file(src, dst)

    # busybox symlinks
    busybox = INITRAMFS_DIR / "bin" / "busybox"
    if busybox.exists():
        for util in ["sh", "mount", "umount", "mkdir", "rm", "cp", "mv",
                     "cat", "ls", "echo", "grep", "sed", "awk", "chmod"]:
            link = INITRAMFS_DIR / "bin" / util
            if not link.exists():
                link.symlink_to("/bin/busybox")


def step_embed_cortex():
    print("[*] Embedding Cortex source...")
    dst_app = INITRAMFS_DIR / "app"
    copy_tree(REPO_ROOT / "src", dst_app / "src")

    # Boot assets
    init_sh = SCRIPT_DIR / "init.sh"
    if init_sh.exists():
        shutil.copy2(init_sh, INITRAMFS_DIR / "init")
        os.chmod(INITRAMFS_DIR / "init", 0o755)

    cortex_init = SCRIPT_DIR / "cortex-init.py"
    if cortex_init.exists():
        shutil.copy2(cortex_init, dst_app / "cortex-init.py")
        os.chmod(dst_app / "cortex-init.py", 0o755)


def step_devices():
    print("[*] Creating device nodes...")
    dev = INITRAMFS_DIR / "dev"
    dev.mkdir(exist_ok=True)

    devices = [
        ("null", 1, 3), ("zero", 1, 5), ("random", 1, 8),
        ("urandom", 1, 9), ("tty", 5, 0), ("tty1", 4, 1),
        ("ttyS0", 4, 64), ("console", 5, 1),
    ]
    for name, major, minor in devices:
        node = dev / name
        if not node.exists():
            try:
                os.mknod(str(node), 0o666 | stat.S_IFCHR, os.makedev(major, minor))
            except PermissionError:
                pass


def step_build_cpio():
    print("[*] Building initramfs.cpio.gz...")
    cpio_path = BUILD_DIR / "initramfs.cpio.gz"

    # Ensure empty directories exist in the archive by touching placeholder files
    for dirname in ["proc", "sys", "tmp", "mnt", "etc", "root"]:
        placeholder = INITRAMFS_DIR / dirname / ".keep"
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        placeholder.write_text("")

    # Use find + cpio via subprocess for proper CPIO format
    cmd = [
        "sh", "-c",
        f"cd '{INITRAMFS_DIR}' && find . | cpio -o -H newc | gzip -9 > '{cpio_path}'"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] cpio failed: {result.stderr}")
        sys.exit(1)

    size = cpio_path.stat().st_size
    print(f"    Size: {size / (1024*1024):.1f}MB")


def step_kernel():
    print("[*] Looking for kernel...")
    kernel_dst = BUILD_DIR / "vmlinuz"
    if kernel_dst.exists():
        print(f"    Kernel already exists: {kernel_dst}")
        return

    kernel_src = Path(f"/boot/vmlinuz-{os.uname().release}")
    if kernel_src.exists():
        try:
            shutil.copy2(kernel_src, kernel_dst)
            print(f"    Kernel: {kernel_src} → {kernel_dst}")
            return
        except PermissionError:
            print("    (Permission denied — will try downloading)")

    # Download Alpine netboot kernel
    alpine_url = "https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/x86_64/netboot/vmlinuz-lts"
    print(f"    Downloading Alpine kernel...")
    result = run(["curl", "-sL", "-o", str(kernel_dst), alpine_url])
    if result.returncode == 0 and kernel_dst.exists() and kernel_dst.stat().st_size > 1000000:
        print(f"    Kernel: {alpine_url} → {kernel_dst}")
    else:
        print("    [!] Could not download kernel")


def main():
    print("==========================================")
    print("  Cortex Initramfs Builder")
    print("==========================================")

    # Clean
    if INITRAMFS_DIR.exists():
        shutil.rmtree(INITRAMFS_DIR)
    INITRAMFS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure standard directories exist
    for dirname in ["proc", "sys", "dev", "tmp", "mnt", "etc", "root"]:
        (INITRAMFS_DIR / dirname).mkdir(parents=True, exist_ok=True)

    step_create_venv()
    step_copy_libs()
    step_copy_python()
    step_copy_binaries()
    step_embed_cortex()
    step_devices()
    step_build_cpio()
    step_kernel()

    print("")
    print("==========================================")
    print("  Build complete")
    print("==========================================")
    for f in BUILD_DIR.iterdir():
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.name:<30s} {size / (1024*1024):>8.1f}MB")
    print("")
    print("  To test in QEMU:")
    print("    qemu-system-x86_64 -kernel build/vmlinuz \\")
    print("      -initrd build/initramfs.cpio.gz -append 'console=ttyS0' \\")
    print("      -nographic -m 8G")
    print("")


if __name__ == "__main__":
    main()
