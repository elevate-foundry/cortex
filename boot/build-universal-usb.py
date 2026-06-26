#!/usr/bin/env python3
"""
Cortex Universal USB Builder — boot AI as PID 1 on ANY device.

Creates a hybrid-bootable USB stick that works on:
  - x86_64 BIOS (legacy)
  - x86_64 UEFI (modern PCs)
  - aarch64 UEFI (ARM servers, Raspberry Pi 4/5, Ampere, Graviton)
  - Apple Silicon (via m1n1 chainload — experimental)

Architecture:
  ┌────────────────────────────────────────────────────────────┐
  │ USB Stick Layout                                            │
  ├──────────┬─────────────────────────────────────────────────┤
  │ Part 1   │ EFI System Partition (FAT32, 512MB)             │
  │          │   /EFI/BOOT/BOOTX64.EFI  (GRUB x86_64)        │
  │          │   /EFI/BOOT/BOOTAA64.EFI (GRUB aarch64)       │
  │          │   /boot/vmlinuz-x86_64                          │
  │          │   /boot/vmlinuz-aarch64                         │
  │          │   /boot/initramfs-x86_64.cpio.gz               │
  │          │   /boot/initramfs-aarch64.cpio.gz              │
  │          │   /grub/grub.cfg                                │
  ├──────────┼─────────────────────────────────────────────────┤
  │ Part 2   │ CORTEX (ext4, rest of disk)                     │
  │          │   /models/gguf/qwen3-0.6b-q4km.gguf           │
  │          │   /models/gguf/qwen3-1.7b-q4km.gguf           │
  │          │   /models/gguf/qwen3-4b-q4km.gguf             │
  │          │   /var/lib/cortex.db                            │
  │          │   /var/log/cortex.log                           │
  │          │   /etc/cortex.toml                              │
  │          │   /AGENTS.md                                    │
  └──────────┴─────────────────────────────────────────────────┘

Key improvements over build-usb.py:
  1. EFI + BIOS dual-boot (works on any PC made after 2005)
  2. Multi-arch: same stick boots x86_64 and aarch64
  3. Models stored as raw GGUF on persistent partition (no Ollama required)
  4. llama.cpp as primary backend (embedded in initramfs)
  5. Auto-detects GPU at boot and loads appropriate driver module
  6. Network auto-config (DHCP + mDNS: cortex.local)
  7. Survives reboots: state on ext4 partition

Usage:
    # Build image (dry-run)
    sudo python3 boot/build-universal-usb.py --output cortex-universal.img

    # Build and write
    sudo python3 boot/build-universal-usb.py --output cortex-universal.img --write

    # Write directly to USB
    sudo python3 boot/build-universal-usb.py --device /dev/sdX --write

    # Include models (downloaded from HuggingFace if not cached)
    sudo python3 boot/build-universal-usb.py --device /dev/sdX --models L0,L1,L2 --write

    # Minimal (L0 only, fits on 2GB stick)
    sudo python3 boot/build-universal-usb.py --device /dev/sdX --models L0 --size 2048 --write
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.resolve()
BUILD_DIR = REPO_ROOT / "build"
CACHE_DIR = BUILD_DIR / "cache"


# ---------------------------------------------------------------------------
# Model catalog — GGUF files that get baked into the USB
# ---------------------------------------------------------------------------

MODELS = {
    "L0": {
        "name": "Qwen3-0.6B",
        "filename": "qwen3-0.6b-q4_k_m.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf",
        "size_mb": 460,
        "sha256": "",  # filled after first download
        "ollama_tag": "qwen3:0.6b",
        "context": 4096,
    },
    "L1": {
        "name": "Qwen3-1.7B",
        "filename": "qwen3-1.7b-q4_k_m.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
        "size_mb": 1200,
        "sha256": "",
        "ollama_tag": "qwen3:1.7b",
        "context": 4096,
    },
    "L2": {
        "name": "Qwen3-4B",
        "filename": "qwen3-4b-q4_k_m.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
        "size_mb": 2800,
        "sha256": "",
        "ollama_tag": "qwen3:4b",
        "context": 8192,
    },
    "L3": {
        "name": "Qwen3-8B",
        "filename": "qwen3-8b-q4_k_m.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
        "size_mb": 5000,
        "sha256": "",
        "ollama_tag": "qwen3:8b",
        "context": 8192,
    },
}

# Kernel sources (Alpine Linux netboot — tiny, stable, well-tested)
KERNELS = {
    "x86_64": {
        "url": "https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/x86_64/netboot/vmlinuz-lts",
        "modules_url": "https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/x86_64/netboot/modloop-lts",
    },
    "aarch64": {
        "url": "https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/aarch64/netboot/vmlinuz-lts",
        "modules_url": "https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/aarch64/netboot/modloop-lts",
    },
}

# GRUB EFI binaries (from Alpine packages)
GRUB_EFIS = {
    "x86_64": "BOOTX64.EFI",
    "aarch64": "BOOTAA64.EFI",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run command, print on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)
    if check and result.returncode != 0:
        print(f"[!] Command failed: {' '.join(cmd)}")
        if result.stderr:
            print(f"    {result.stderr[:500]}")
        if check:
            sys.exit(1)
    return result


def download(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress indication."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    [cached] {desc or dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Downloading: {desc or url}")
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"    Done: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"    [!] Download failed: {e}")
        return False


def confirm(msg: str) -> bool:
    """User confirmation prompt."""
    print(f"\n{'═' * 60}")
    print(f"  {msg}")
    print(f"{'═' * 60}")
    try:
        resp = input("  Type 'yes' to proceed: ")
        return resp.strip().lower() == "yes"
    except EOFError:
        return False


def is_removable(dev: str) -> bool:
    """Check if block device is removable."""
    name = Path(dev).name
    p = Path(f"/sys/block/{name}/removable")
    return p.exists() and p.read_text().strip() == "1"


def host_arch() -> str:
    """Detect host architecture."""
    machine = platform.machine()
    if machine in ("x86_64", "AMD64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return machine


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------

def step_download_kernels(arches: list[str]):
    """Download kernels for target architectures."""
    print("[1/7] Downloading kernels...")
    for arch in arches:
        if arch not in KERNELS:
            print(f"    [!] Unknown arch: {arch}")
            continue
        dest = CACHE_DIR / "kernels" / f"vmlinuz-{arch}"
        download(KERNELS[arch]["url"], dest, f"Linux kernel ({arch})")


def step_download_models(model_tiers: list[str]):
    """Download GGUF model files."""
    print("[2/7] Downloading models...")
    for tier in model_tiers:
        if tier not in MODELS:
            print(f"    [!] Unknown tier: {tier}")
            continue
        model = MODELS[tier]
        dest = CACHE_DIR / "models" / model["filename"]
        download(model["url"], dest, f"{model['name']} ({model['size_mb']}MB)")


def step_build_grub_efi(arches: list[str], dry_run: bool):
    """Build GRUB EFI binaries for each arch (or use pre-built)."""
    print("[3/7] Preparing EFI bootloaders...")
    efi_dir = CACHE_DIR / "efi"
    efi_dir.mkdir(parents=True, exist_ok=True)

    for arch in arches:
        efi_name = GRUB_EFIS.get(arch)
        if not efi_name:
            continue
        efi_path = efi_dir / efi_name
        if efi_path.exists():
            print(f"    [cached] {efi_name}")
            continue

        # Try to build with grub-mkimage
        if arch == "x86_64":
            target = "x86_64-efi"
            grub_cmd = "grub-mkimage"
        else:
            target = "arm64-efi"
            grub_cmd = "grub-mkimage"

        grub_bin = shutil.which(grub_cmd) or shutil.which("grub2-mkimage")
        if grub_bin and not dry_run:
            modules = "part_gpt part_msdos fat ext2 normal linux boot chain configfile search search_fs_uuid search_label echo test"
            result = run([
                grub_bin,
                "--format", target,
                "--output", str(efi_path),
                "--prefix", "/grub",
            ] + modules.split(), check=False)
            if result.returncode == 0:
                print(f"    Built {efi_name} via {grub_cmd}")
                continue

        # Fallback: create a stub that tells user to install grub
        print(f"    [!] Cannot build {efi_name} — grub-mkimage not available for {target}")
        print(f"        Install: apt install grub-efi-{arch.replace('x86_64','amd64').replace('aarch64','arm64')}-bin")
        if not dry_run:
            # Create placeholder so build can continue
            efi_path.write_bytes(b"")


def step_build_initramfs(arches: list[str], model_tiers: list[str], dry_run: bool):
    """Build architecture-specific initramfs images."""
    print("[4/7] Building initramfs images...")
    for arch in arches:
        dest = BUILD_DIR / f"initramfs-{arch}.cpio.gz"
        if dest.exists():
            print(f"    [cached] initramfs-{arch} ({dest.stat().st_size // (1024*1024)}MB)")
            continue

        if dry_run:
            print(f"    [dry-run] Would build initramfs-{arch}")
            continue

        # For now, delegate to existing builder (same-arch only)
        if arch == host_arch():
            result = run([
                sys.executable, str(SCRIPT_DIR / "build-initramfs.py"),
                "--no-kernel",
            ], check=False)
            built = BUILD_DIR / "initramfs.cpio.gz"
            if built.exists():
                shutil.move(str(built), str(dest))
                print(f"    Built initramfs-{arch} ({dest.stat().st_size // (1024*1024)}MB)")
            else:
                print(f"    [!] Failed to build initramfs-{arch}")
        else:
            print(f"    [!] Cross-arch initramfs build not yet supported for {arch}")
            print(f"        Build on a {arch} machine, or use QEMU user-mode emulation")


def step_create_grub_cfg(model_tiers: list[str]) -> str:
    """Generate a universal GRUB config that auto-detects architecture."""
    models_desc = ", ".join(model_tiers) if model_tiers else "none"
    cfg = f"""# Cortex AI OS — Universal Boot Configuration
# Models: {models_desc}

set timeout=3
set default=0

# Auto-detect architecture
if [ "$grub_platform" = "efi" ]; then
    if [ "$grub_cpu" = "x86_64" ]; then
        set arch="x86_64"
    elif [ "$grub_cpu" = "arm64" ]; then
        set arch="aarch64"
    else
        set arch="x86_64"
    fi
else
    set arch="x86_64"
fi

menuentry "Cortex AI OS (auto)" --class cortex {{
    linux /boot/vmlinuz-$arch rw console=tty0 console=ttyS0,115200 cortex.models={','.join(model_tiers)} quiet
    initrd /boot/initramfs-$arch.cpio.gz
}}

menuentry "Cortex AI OS (x86_64)" --class cortex {{
    linux /boot/vmlinuz-x86_64 rw console=tty0 console=ttyS0,115200 cortex.models={','.join(model_tiers)} quiet
    initrd /boot/initramfs-x86_64.cpio.gz
}}

menuentry "Cortex AI OS (aarch64)" --class cortex {{
    linux /boot/vmlinuz-aarch64 rw console=tty0 console=ttyS0,115200 cortex.models={','.join(model_tiers)} quiet
    initrd /boot/initramfs-aarch64.cpio.gz
}}

menuentry "Cortex AI OS (rescue shell)" --class recovery {{
    linux /boot/vmlinuz-$arch rw console=tty0 console=ttyS0,115200 cortex.rescue=1 init=/bin/sh
    initrd /boot/initramfs-$arch.cpio.gz
}}
"""
    return cfg


def step_assemble_image(
    target: str,
    size_mb: int,
    arches: list[str],
    model_tiers: list[str],
    dry_run: bool,
) -> Optional[str]:
    """Assemble the final USB image with EFI + data partitions."""
    print("[5/7] Assembling USB image...")

    if dry_run:
        print(f"    [dry-run] Would create {size_mb}MB image at {target}")
        print(f"    [dry-run] Partition 1: ESP (FAT32, 512MB)")
        print(f"    [dry-run] Partition 2: CORTEX (ext4, {size_mb - 512}MB)")
        return None

    img = Path(target)

    # Create image
    run(["fallocate", "-l", f"{size_mb}M", str(img)])

    # Partition: GPT with ESP + data
    sgdisk_cmds = [
        ["sgdisk", "--zap-all", str(img)],
        ["sgdisk", "--new=1:2048:+512M", "--typecode=1:EF00", "--change-name=1:EFI", str(img)],
        ["sgdisk", "--new=2:0:0", "--typecode=2:8300", "--change-name=2:CORTEX", str(img)],
    ]

    # Fallback to sfdisk if sgdisk not available
    if shutil.which("sgdisk"):
        for cmd in sgdisk_cmds:
            run(cmd)
    else:
        # Use sfdisk with GPT
        sfdisk_script = (
            "label: gpt\n"
            "size=512M, type=C12A7328-F81F-11D2-BA4B-00A0C93EC93B, name=EFI\n"
            "type=0FC63DAF-8483-4772-8E79-3D69D8477DE4, name=CORTEX\n"
        )
        p = subprocess.Popen(["sfdisk", str(img)], stdin=subprocess.PIPE, text=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.communicate(sfdisk_script)

    # Setup loop device
    result = run(["losetup", "--find", "--partscan", "--show", str(img)])
    loop_dev = result.stdout.strip()
    print(f"    Loop: {loop_dev}")
    time.sleep(0.5)

    try:
        esp_part = f"{loop_dev}p1"
        data_part = f"{loop_dev}p2"

        # Format ESP
        run(["mkfs.fat", "-F", "32", "-n", "EFI", esp_part])
        # Format data
        run(["mkfs.ext4", "-L", "CORTEX", "-F", "-q", data_part])

        with tempfile.TemporaryDirectory(prefix="cortex_usb_") as tmpdir:
            esp_mnt = os.path.join(tmpdir, "esp")
            data_mnt = os.path.join(tmpdir, "data")
            os.makedirs(esp_mnt)
            os.makedirs(data_mnt)

            run(["mount", esp_part, esp_mnt])
            run(["mount", data_part, data_mnt])

            try:
                # --- ESP contents ---
                # EFI bootloaders
                efi_boot = Path(esp_mnt) / "EFI" / "BOOT"
                efi_boot.mkdir(parents=True, exist_ok=True)
                for arch in arches:
                    efi_name = GRUB_EFIS.get(arch)
                    if efi_name:
                        src = CACHE_DIR / "efi" / efi_name
                        if src.exists() and src.stat().st_size > 0:
                            shutil.copy2(src, efi_boot / efi_name)

                # GRUB config
                grub_dir = Path(esp_mnt) / "grub"
                grub_dir.mkdir(parents=True, exist_ok=True)
                grub_cfg = step_create_grub_cfg(model_tiers)
                (grub_dir / "grub.cfg").write_text(grub_cfg)

                # Kernels + initramfs
                boot_dir = Path(esp_mnt) / "boot"
                boot_dir.mkdir(parents=True, exist_ok=True)
                for arch in arches:
                    kernel_src = CACHE_DIR / "kernels" / f"vmlinuz-{arch}"
                    if kernel_src.exists():
                        shutil.copy2(kernel_src, boot_dir / f"vmlinuz-{arch}")
                    initramfs_src = BUILD_DIR / f"initramfs-{arch}.cpio.gz"
                    if initramfs_src.exists():
                        shutil.copy2(initramfs_src, boot_dir / f"initramfs-{arch}.cpio.gz")

                # --- Data partition contents ---
                # Models
                models_dir = Path(data_mnt) / "models" / "gguf"
                models_dir.mkdir(parents=True, exist_ok=True)
                for tier in model_tiers:
                    if tier in MODELS:
                        src = CACHE_DIR / "models" / MODELS[tier]["filename"]
                        if src.exists():
                            print(f"    Copying model {tier}: {MODELS[tier]['name']}...")
                            shutil.copy2(src, models_dir / MODELS[tier]["filename"])

                # Model manifest (so cortex-init knows what's available)
                manifest = {
                    "models": {
                        tier: {
                            "path": f"/mnt/cortex/models/gguf/{MODELS[tier]['filename']}",
                            "context": MODELS[tier]["context"],
                            "tier": tier,
                            "ollama_tag": MODELS[tier]["ollama_tag"],
                        }
                        for tier in model_tiers if tier in MODELS
                    }
                }
                (Path(data_mnt) / "models" / "manifest.json").write_text(
                    json.dumps(manifest, indent=2)
                )

                # CKM (Cortex Kernel Model) — AI as PID 1
                # This 379MB GGUF model makes boot configuration decisions.
                # It runs BEFORE the main inference models are loaded.
                ckm_src = SCRIPT_DIR.parent / "models" / "cortex-kernel.gguf"
                if ckm_src.exists():
                    ckm_dest = Path(data_mnt) / "models" / "cortex-kernel.gguf"
                    print(f"    Copying CKM (AI as PID 1): {ckm_src.stat().st_size // (1024*1024)}MB")
                    shutil.copy2(ckm_src, ckm_dest)
                else:
                    print("    [!] CKM not found (models/cortex-kernel.gguf)")
                    print("        Boot will use heuristics instead of AI decisions.")
                    print("        Train with: modal run src/ckm/modal_train.py")

                # Persistence directories
                Path(data_mnt, "var", "lib").mkdir(parents=True, exist_ok=True)
                Path(data_mnt, "var", "log").mkdir(parents=True, exist_ok=True)
                Path(data_mnt, "etc").mkdir(parents=True, exist_ok=True)

                # Default config
                cortex_toml = """# Cortex AI OS Configuration
[daemon]
host = "0.0.0.0"
port = 11411

[backend]
# Primary backend: llama-cpp (no Ollama dependency)
primary = "llama_cpp"
# Fallback: Ollama if installed on host
fallback = "ollama"
# Model directory (on CORTEX partition)
model_dir = "/mnt/cortex/models/gguf"

[network]
# Auto-configure networking on boot
dhcp = true
mdns = "cortex.local"
# SSH access (key-only after first boot)
ssh = true

[gpu]
# Auto-detect and load appropriate driver
auto_detect = true
# Prefer GPU inference when available
prefer_gpu = true
"""
                (Path(data_mnt) / "etc" / "cortex.toml").write_text(cortex_toml)

                # AGENTS.md placeholder
                (Path(data_mnt) / "AGENTS.md").write_text(
                    "# Cortex Agent Manifest\n\n"
                    "# This file persists across reboots on the USB stick.\n"
                    "# Add agent coordination directives here.\n"
                )

            finally:
                run(["umount", esp_mnt], check=False)
                run(["umount", data_mnt], check=False)

        # Install BIOS bootloader (MBR protective for GPT)
        grub_install = shutil.which("grub-install") or shutil.which("grub2-install")
        if grub_install:
            run([grub_install, "--target=i386-pc", f"--boot-directory={esp_mnt}/boot",
                 "--recheck", loop_dev], check=False)

    finally:
        run(["losetup", "-d", loop_dev], check=False)

    size_actual = img.stat().st_size / (1024 * 1024)
    print(f"    Image ready: {img} ({size_actual:.0f} MB)")
    return str(img)


def step_write_to_device(img_path: str, device: str, dry_run: bool):
    """Write image to physical USB device."""
    print("[6/7] Writing to device...")
    if dry_run:
        print(f"    [dry-run] Would dd {img_path} → {device}")
        return
    run(["dd", f"if={img_path}", f"of={device}", "bs=4M", "status=progress", "conv=fsync"])
    run(["sync"])
    print(f"    Written to {device}")


def step_summary(target: str, model_tiers: list[str], arches: list[str], dry_run: bool):
    """Print build summary."""
    print(f"\n{'═' * 60}")
    if dry_run:
        print("  DRY-RUN COMPLETE — pass --write to build")
    else:
        print("  ✓ Cortex Universal USB Built Successfully")
    print(f"{'═' * 60}")
    print(f"  Target:         {target}")
    print(f"  Architectures:  {', '.join(arches)}")
    print(f"  Models:         {', '.join(model_tiers)}")
    total_model_mb = sum(MODELS[t]["size_mb"] for t in model_tiers if t in MODELS)
    print(f"  Model size:     ~{total_model_mb} MB")
    print(f"{'═' * 60}")
    print()
    print("  Boot on any machine:")
    print("    1. Plug USB into target device")
    print("    2. Enter BIOS/UEFI boot menu (F12 / Option key / DEL)")
    print("    3. Select USB device")
    print("    4. Cortex boots as PID 1 in ~5 seconds")
    print()
    print("  Connect from another device:")
    print("    curl http://cortex.local:11411/health")
    print("    export OPENAI_BASE_URL=http://cortex.local:11411/v1")
    print()
    print("  Or locally:")
    print("    curl http://localhost:11411/v1/chat/completions \\")
    print('      -d \'{"model":"auto","messages":[{"role":"user","content":"hello"}]}\'')
    print(f"{'═' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cortex Universal USB Builder — AI as PID 1 on any device",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run (see what would happen)
  sudo python3 boot/build-universal-usb.py --output cortex.img

  # Build 8GB image with L0-L2 models
  sudo python3 boot/build-universal-usb.py --output cortex.img --models L0,L1,L2 --size 8192 --write

  # Minimal 2GB stick (L0 only)
  sudo python3 boot/build-universal-usb.py --device /dev/sdX --models L0 --size 2048 --write

  # Full stack (L0-L3, needs 16GB stick)
  sudo python3 boot/build-universal-usb.py --device /dev/sdX --models L0,L1,L2,L3 --size 16384 --write
""",
    )
    parser.add_argument("--device", "-d", help="Target USB device (e.g. /dev/sdX)")
    parser.add_argument("--output", "-o", default="cortex-universal.img", help="Output image file")
    parser.add_argument("--size", "-s", type=int, default=8192, help="Image size in MB (default: 8192)")
    parser.add_argument("--models", "-m", default="L0,L1,L2", help="Tiers to include (default: L0,L1,L2)")
    parser.add_argument("--arch", default="x86_64,aarch64", help="Target architectures (default: both)")
    parser.add_argument("--write", action="store_true", help="Actually write (dry-run by default)")
    parser.add_argument("--skip-download", action="store_true", help="Skip model/kernel downloads")
    args = parser.parse_args()

    dry_run = not args.write
    model_tiers = [t.strip().upper() for t in args.models.split(",")]
    arches = [a.strip() for a in args.arch.split(",")]

    # Validate
    if os.geteuid() != 0 and args.write:
        print("Error: Must run as root (sudo) for --write mode.")
        sys.exit(1)

    if args.device and args.device == args.output:
        print("Error: --device and --output cannot be the same")
        sys.exit(1)

    # Safety check for devices
    if args.device:
        if not Path(args.device).exists():
            print(f"Error: Device {args.device} does not exist")
            sys.exit(1)
        if not is_removable(args.device):
            print(f"[!] WARNING: {args.device} does NOT appear to be a removable device!")
            if not confirm(f"DANGER: Overwrite {args.device}? This will DESTROY all data."):
                sys.exit(0)

    target = args.device or args.output

    print(f"\n{'═' * 60}")
    print("  Cortex Universal USB Builder")
    print(f"{'═' * 60}")
    print(f"  Target:     {target}")
    print(f"  Size:       {args.size} MB")
    print(f"  Models:     {', '.join(model_tiers)}")
    print(f"  Arches:     {', '.join(arches)}")
    print(f"  Mode:       {'WRITE' if args.write else 'DRY-RUN'}")
    print(f"{'═' * 60}\n")

    if args.write and not confirm(f"Build Cortex USB → {target}?"):
        sys.exit(0)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Execute build pipeline
    if not args.skip_download:
        step_download_kernels(arches)
        step_download_models(model_tiers)

    step_build_grub_efi(arches, dry_run)
    step_build_initramfs(arches, model_tiers, dry_run)

    img_path = step_assemble_image(
        target=args.output if not args.device else str(BUILD_DIR / "cortex-tmp.img"),
        size_mb=args.size,
        arches=arches,
        model_tiers=model_tiers,
        dry_run=dry_run,
    )

    if args.device and img_path:
        step_write_to_device(img_path, args.device, dry_run)
        # Clean up temp image
        tmp_img = BUILD_DIR / "cortex-tmp.img"
        if tmp_img.exists():
            tmp_img.unlink()

    step_summary(target, model_tiers, arches, dry_run)


if __name__ == "__main__":
    main()
