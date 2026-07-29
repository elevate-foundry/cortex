#!/usr/bin/env python3
"""
Cortex Live USB Builder — plug in any computer and take over.

Builds a bootable USB image (or writes directly to a thumbdrive) containing:
  - Linux kernel + initramfs with Cortex + Ollama + models
  - GRUB bootloader (MBR + BIOS/UEFI compatible)
  - Persistent ext4 partition labeled "CORTEX" for DB + state

Usage:
    # Dry-run (shows what would happen)
    sudo python3 boot/build-usb.py --output cortex.img --size 2048

    # Build image
    sudo python3 boot/build-usb.py --output cortex.img --size 2048 --write

    # Write directly to USB device
    sudo python3 boot/build-usb.py --device /dev/sdX --write

    # Include specific models (default: qwen3:0.6b)
    sudo python3 boot/build-usb.py --device /dev/sdX --models qwen3:0.6b,qwen3:1.7b --write
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.resolve()
BUILD_DIR = REPO_ROOT / "build"


def run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a shell command, return result."""
    return subprocess.run(cmd, capture_output=True, text=True, check=check, **kwargs)


def confirm(msg: str) -> bool:
    """Ask user for confirmation."""
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")
    try:
        resp = input("  Type 'yes' to proceed: ")
        return resp.strip().lower() == "yes"
    except EOFError:
        return False


def is_removable_device(dev: str) -> bool:
    """Check if a block device is removable (USB stick, SD card)."""
    name = Path(dev).name
    removable_path = Path(f"/sys/block/{name}/removable")
    if removable_path.exists():
        return removable_path.read_text().strip() == "1"
    return False


def build_initramfs(models: str) -> Path:
    """Call build-initramfs.py with models included."""
    print("[*] Building initramfs with models...")
    cmd = [sys.executable, str(SCRIPT_DIR / "build-initramfs.py"), "--models", models, "--no-kernel"]
    result = run(cmd, check=False)
    if result.returncode != 0:
        print(f"[!] Initramfs build failed:\n{result.stderr}")
        sys.exit(1)
    initramfs = BUILD_DIR / "initramfs.cpio.gz"
    if not initramfs.exists():
        print("[!] Initramfs not found after build")
        sys.exit(1)
    size_mb = initramfs.stat().st_size / (1024 * 1024)
    print(f"    Initramfs: {initramfs} ({size_mb:.1f} MB)")
    return initramfs


def get_kernel() -> Path:
    """Find the host kernel."""
    kernel = Path(f"/boot/vmlinuz-{os.uname().release}")
    if kernel.exists():
        return kernel
    # Fallback to Alpine kernel
    alpine = BUILD_DIR / "vmlinuz"
    if alpine.exists():
        return alpine
    print("[!] No kernel found. Run build-initramfs.py first or provide --kernel")
    sys.exit(1)


def create_disk_image(path: Path, size_mb: int, dry_run: bool) -> str:
    """Create a raw disk image file of given size in MB."""
    print(f"[*] Creating disk image: {path} ({size_mb} MB)")
    if not dry_run:
        run(["fallocate", "-l", f"{size_mb}M", str(path)])
    return str(path)


def partition_device(dev: str, dry_run: bool):
    """Create MBR partition table with one primary ext4 partition."""
    print(f"[*] Partitioning {dev}...")
    # MBR label, one partition from 1MiB to end
    sfdisk_script = "label: dos\nstart=2048, type=83\n"
    if not dry_run:
        p = subprocess.Popen(
            ["sfdisk", dev],
            stdin=subprocess.PIPE,
            text=True,
        )
        p.communicate(sfdisk_script)
        if p.returncode != 0:
            print(f"[!] sfdisk failed")
            sys.exit(1)
        # Re-read partition table
        run(["partprobe", dev], check=False)
        time.sleep(1)


def setup_loop_device(img: str, dry_run: bool) -> str:
    """Attach image as loopback device, return device path."""
    if dry_run:
        return "/dev/loop0"
    result = run(["losetup", "--find", "--partscan", "--show", img])
    loop_dev = result.stdout.strip()
    print(f"    Loop device: {loop_dev}")
    time.sleep(0.5)
    return loop_dev


def teardown_loop_device(loop_dev: str, dry_run: bool):
    """Detach loopback device."""
    if dry_run:
        return
    run(["losetup", "-d", loop_dev], check=False)
    print(f"    Detached {loop_dev}")


def format_partition(part: str, dry_run: bool):
    """Format partition as ext4 labeled CORTEX."""
    print(f"[*] Formatting {part} as ext4...")
    if not dry_run:
        run(["mkfs.ext4", "-L", "CORTEX", "-F", part])


def mount_partition(part: str, mountpoint: str, dry_run: bool):
    """Mount partition."""
    print(f"[*] Mounting {part} -> {mountpoint}")
    if not dry_run:
        os.makedirs(mountpoint, exist_ok=True)
        run(["mount", part, mountpoint])


def umount_partition(mountpoint: str, dry_run: bool):
    """Unmount partition."""
    print(f"[*] Unmounting {mountpoint}")
    if not dry_run:
        run(["umount", mountpoint], check=False)


def install_grub(dev: str, boot_dir: str, dry_run: bool):
    """Install GRUB for BIOS to MBR."""
    print(f"[*] Installing GRUB to {dev}...")
    if not dry_run:
        run([
            "grub-install",
            "--target=i386-pc",
            f"--boot-directory={boot_dir}",
            "--recheck",
            dev,
        ])


def create_grub_cfg(boot_dir: str, dry_run: bool):
    """Create grub.cfg with Cortex boot entry."""
    print("[*] Creating GRUB config...")
    cfg_dir = Path(boot_dir) / "grub"
    cfg_path = cfg_dir / "grub.cfg"
    grub_cfg = """set timeout=5
set default=0

menuentry "Cortex AI OS" {
    linux /boot/vmlinuz rw console=tty0 console=ttyS0,115200
    initrd /boot/initramfs.cpio.gz
}
"""
    if not dry_run:
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(grub_cfg)


def copy_boot_files(mountpoint: str, kernel: Path, initramfs: Path, dry_run: bool):
    """Copy kernel and initramfs to /boot on the partition."""
    print("[*] Copying boot files...")
    boot_dir = Path(mountpoint) / "boot"
    if not dry_run:
        boot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kernel, boot_dir / "vmlinuz")
        shutil.copy2(initramfs, boot_dir / "initramfs.cpio.gz")
        print(f"    Kernel:     {kernel.name}")
        print(f"    Initramfs:  {initramfs.name}")


def create_persistence_dirs(mountpoint: str, dry_run: bool):
    """Create directories for persistent state."""
    print("[*] Creating persistence directories...")
    dirs = [
        Path(mountpoint) / "var" / "lib",
        Path(mountpoint) / "var" / "log",
    ]
    for d in dirs:
        if not dry_run:
            d.mkdir(parents=True, exist_ok=True)
    # Touch AGENTS.md placeholder
    agents = Path(mountpoint) / "AGENTS.md"
    if not dry_run:
        agents.write_text("# Cortex Agent Manifest\n\n# Add agent directives here.\n")


def build_usb(args: argparse.Namespace):
    """Main build routine."""
    dry_run = not args.write
    if dry_run:
        print("=" * 60)
        print("  DRY-RUN MODE — no writes will be performed")
        print("  Pass --write to actually build")
        print("=" * 60)

    # Validate root
    if os.geteuid() != 0:
        print("Error: Must run as root (sudo).")
        sys.exit(1)

    # Determine target
    if args.device and args.output:
        print("Error: Cannot specify both --device and --output")
        sys.exit(1)

    if args.device:
        if not Path(args.device).exists():
            print(f"Error: Device {args.device} does not exist")
            sys.exit(1)
        if not is_removable_device(args.device):
            print(f"[!] WARNING: {args.device} does not appear to be a removable device!")
            if not confirm("Are you sure you want to overwrite this device?"):
                print("Aborted.")
                sys.exit(1)
        target_dev = args.device
        target_desc = f"device {args.device}"
        is_image = False
    else:
        target_dev = str(Path(args.output).resolve())
        target_desc = f"image {target_dev}"
        is_image = True

    # Pre-flight: check tools
    required_tools = ["grub-install", "sfdisk", "mkfs.ext4", "partprobe"]
    if is_image:
        required_tools += ["fallocate", "losetup"]
    missing = [t for t in required_tools if shutil.which(t) is None]
    if missing:
        print(f"Error: Missing tools: {', '.join(missing)}")
        print("Install: sudo apt install grub-common grub-pc-bin util-linux parted")
        sys.exit(1)

    # Summary
    models = args.models
    size_mb = args.size
    print(f"\n{'=' * 60}")
    print(f"  Cortex USB Builder")
    print(f"{'=' * 60}")
    print(f"  Target:     {target_desc}")
    print(f"  Size:       {size_mb} MB")
    print(f"  Models:     {models}")
    print(f"  Kernel:     {args.kernel}")
    print(f"  Write:      {'YES' if args.write else 'NO (dry-run)'}")
    print(f"{'=' * 60}\n")

    if not dry_run and not confirm(f"Proceed with {target_desc}?"):
        print("Aborted.")
        sys.exit(0)

    # Build initramfs
    initramfs = build_initramfs(models)
    kernel = Path(args.kernel)
    if not kernel.exists():
        kernel = get_kernel()
    print(f"    Kernel: {kernel}")

    # Build image or prepare device
    with tempfile.TemporaryDirectory(prefix="cortex_usb_") as tmpdir:
        mountpoint = os.path.join(tmpdir, "mnt")

        if is_image:
            img_path = Path(target_dev)
            create_disk_image(img_path, size_mb, dry_run)
            partition_device(str(img_path), dry_run)
            loop_dev = setup_loop_device(str(img_path), dry_run)
            part = f"{loop_dev}p1"
        else:
            # Real device
            partition_device(target_dev, dry_run)
            part = f"{target_dev}1"

        format_partition(part, dry_run)
        mount_partition(part, mountpoint, dry_run)

        # Install bootloader
        if is_image:
            install_grub(loop_dev, os.path.join(mountpoint, "boot"), dry_run)
        else:
            install_grub(target_dev, os.path.join(mountpoint, "boot"), dry_run)

        create_grub_cfg(os.path.join(mountpoint, "boot"), dry_run)
        copy_boot_files(mountpoint, kernel, initramfs, dry_run)
        create_persistence_dirs(mountpoint, dry_run)

        umount_partition(mountpoint, dry_run)

        if is_image:
            teardown_loop_device(loop_dev, dry_run)

    if dry_run:
        print(f"\n{'=' * 60}")
        print("  DRY-RUN complete — pass --write to actually build")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print("  Build complete!")
        print(f"{'=' * 60}")
        if is_image:
            print(f"  Image: {target_dev}")
            print(f"  Flash to USB:")
            print(f"    sudo dd if={target_dev} of=/dev/sdX bs=4M status=progress")
        else:
            print(f"  Device: {target_dev}")
            print(f"  Eject and plug into target machine.")
        print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Build bootable Cortex USB")
    parser.add_argument("--device", "-d", help="Target USB device (e.g. /dev/sdX)")
    parser.add_argument("--output", "-o", help="Output image file (e.g. cortex.img)")
    parser.add_argument("--size", "-s", type=int, default=2048, help="Image size in MB (default: 2048)")
    parser.add_argument("--kernel", default=f"/boot/vmlinuz-{os.uname().release}", help="Kernel path")
    parser.add_argument("--models", default="qwen3:0.6b", help="Comma-separated Ollama models")
    parser.add_argument("--write", action="store_true", help="Actually write (dry-run by default)")
    args = parser.parse_args()
    build_usb(args)


if __name__ == "__main__":
    main()
