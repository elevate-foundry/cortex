#!/usr/bin/env python3
"""
Cortex Init — PID 1 bootstrap for AI-native operating system.

When the Linux kernel boots, it looks for /init.  This file IS /init.
It reaps zombies, mounts filesystems, starts essential services,
detects hardware, and then becomes the Cortex inference daemon.

Usage (as PID 1):
    exec python3 /app/src/cortex-init.py

Usage (dry-run for testing):
    sudo python3 /app/src/cortex-init.py --dry-run
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("cortex.init")

# ---------------------------------------------------------------------------
# Service registry
# ---------------------------------------------------------------------------

SERVICES: dict[str, dict] = {}


def register_service(name: str, cmd: list[str], env: dict | None = None) -> None:
    """Register a service to be managed by the init process."""
    SERVICES[name] = {
        "cmd": cmd,
        "env": env or {},
        "pid": None,
        "status": "stopped",
        "started_at": 0,
    }


def start_service(name: str) -> int | None:
    """Start a registered service, return its PID or None."""
    svc = SERVICES.get(name)
    if not svc:
        return None
    if svc["pid"] is not None:
        logger.info("Service %s already running (pid=%s)", name, svc["pid"])
        return svc["pid"]

    env = os.environ.copy()
    env.update(svc["env"])
    try:
        proc = subprocess.Popen(
            svc["cmd"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        svc["pid"] = proc.pid
        svc["status"] = "running"
        svc["started_at"] = time.monotonic()
        logger.info("Started service %s (pid=%d)", name, proc.pid)
        return proc.pid
    except FileNotFoundError as e:
        logger.warning("Cannot start %s: %s not found", name, svc["cmd"][0])
        svc["status"] = "failed"
        return None


def stop_service(name: str) -> bool:
    """Stop a running service."""
    svc = SERVICES.get(name)
    if not svc or svc["pid"] is None:
        return False
    try:
        os.kill(svc["pid"], signal.SIGTERM)
        svc["pid"] = None
        svc["status"] = "stopped"
        logger.info("Stopped service %s", name)
        return True
    except ProcessLookupError:
        svc["pid"] = None
        svc["status"] = "stopped"
        return True


def service_status(name: str) -> dict:
    """Return current status of a service."""
    svc = SERVICES.get(name, {})
    pid = svc.get("pid")
    if pid is not None:
        # Check if still alive
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            svc["pid"] = None
            svc["status"] = "exited"
    return {
        "name": name,
        "status": svc.get("status", "unknown"),
        "pid": svc.get("pid"),
        "reason": svc.get("reason", ""),
    }


def all_services() -> list[dict]:
    """Return status of all registered services."""
    return [service_status(n) for n in SERVICES]


# ---------------------------------------------------------------------------
# PID-1 duties
# ---------------------------------------------------------------------------

ZOMBIES_REAPED = 0


def _reap_zombies(signum, frame):
    """SIGCHLD handler — reap terminated child processes."""
    global ZOMBIES_REAPED
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
            ZOMBIES_REAPED += 1
            # Update service status if this was a known service
            for svc in SERVICES.values():
                if svc.get("pid") == pid:
                    svc["pid"] = None
                    svc["status"] = "exited"
                    logger.info("Service exited (pid=%d)", pid)
                    break
        except ChildProcessError:
            break


def _setup_pid1():
    """Install signal handlers required for PID-1 operation."""
    signal.signal(signal.SIGCHLD, _reap_zombies)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    # Ignore SIGPIPE (common when writing to closed sockets)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)


def _shutdown(signum=None, frame=None):
    """Graceful shutdown."""
    sig_name = signal.Signals(signum).name if signum else "UNKNOWN"
    logger.info("Init received %s, shutting down services...", sig_name)
    for name in list(SERVICES):
        stop_service(name)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Boot sequence
# ---------------------------------------------------------------------------

def mount_virtualfs():
    """Mount proc, sys, dev, tmp — the bare minimum for a Linux system."""
    mounts = [
        ("proc", "/proc", "proc"),
        ("sysfs", "/sys", "sysfs"),
        ("devtmpfs", "/dev", "devtmpfs"),
        ("tmpfs", "/tmp", "tmpfs"),
    ]
    for src, dst, fstype in mounts:
        Path(dst).mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(["mount", "-t", fstype, src, dst], check=False, capture_output=True)
            logger.info("Mounted %s on %s", fstype, dst)
        except FileNotFoundError:
            logger.warning("mount command not found; skipping %s", dst)


def detect_hardware() -> dict:
    """Quick hardware probe for boot-time decisions."""
    profile = {}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    profile["cpu"] = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    profile["ram_kb"] = int(line.split()[1])
                    break
    except Exception:
        pass

    # Check for GPU
    profile["has_gpu"] = Path("/dev/nvidia0").exists() or Path("/dev/dri").exists()
    profile["has_network"] = Path("/sys/class/net/eth0").exists() or Path("/sys/class/net/wlan0").exists()

    return profile


def mount_cortex_partition() -> str | None:
    """Find and mount the CORTEX USB partition for persistence. Returns mount point or None."""
    try:
        result = subprocess.run(
            ["blkid", "-L", "CORTEX"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            dev = result.stdout.strip()
            os.makedirs("/mnt/cortex", exist_ok=True)
            subprocess.run(
                ["mount", "-t", "ext4", dev, "/mnt/cortex"],
                capture_output=True, timeout=10, check=False
            )
            if os.path.ismount("/mnt/cortex"):
                logger.info("Mounted CORTEX partition at /mnt/cortex (%s)", dev)
                # Ensure state directories exist
                Path("/mnt/cortex/var/lib").mkdir(parents=True, exist_ok=True)
                Path("/mnt/cortex/var/log").mkdir(parents=True, exist_ok=True)
                return "/mnt/cortex"
    except FileNotFoundError:
        logger.warning("blkid not found; cannot search for CORTEX partition")
    except Exception as e:
        logger.warning("Failed to mount CORTEX partition: %s", e)
    return None


def wait_for_port(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for a TCP port to become reachable."""
    import socket
    for _ in range(timeout):
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(1)
    return False


def boot_sequence(dry_run: bool = False):
    """Full boot sequence for Cortex as PID 1."""
    logger.info("=== Cortex Init Boot Sequence ===")
    logger.info("PID: %d | PPID: %d | dry_run=%s", os.getpid(), os.getppid(), dry_run)

    # 1. PID-1 signal plumbing
    if os.getpid() == 1:
        _setup_pid1()
        logger.info("Installed PID-1 signal handlers (SIGCHLD, SIGTERM)")

    # 2. Mount virtual filesystems
    if not dry_run:
        mount_virtualfs()
    else:
        logger.info("[DRY-RUN] Skipping mount_virtualfs()")

    # 3. Hardware detection
    hw = detect_hardware()
    logger.info("Hardware: %s", hw)

    # 3.5 Mount CORTEX partition for persistence (optional)
    cortex_mount = mount_cortex_partition()
    if cortex_mount:
        os.environ["CORTEX_DB"] = f"{cortex_mount}/var/lib/cortex.db"
        os.environ["CORTEX_AGENTS"] = f"{cortex_mount}/AGENTS.md"
        logger.info("Persistence enabled: DB at %s/var/lib/cortex.db", cortex_mount)
    else:
        logger.info("No CORTEX partition found; running without persistence")

    # 4. Register services based on hardware
    # getty on tty1 if we have a tty
    if Path("/dev/tty1").exists():
        register_service(
            "getty",
            cmd=["/sbin/getty", "38400", "tty1"],
            reason="tty1 detected",
        )
        if not dry_run:
            start_service("getty")
    else:
        logger.info("No tty1; skipping getty")

    # sshd if network detected
    if hw.get("has_network"):
        register_service(
            "sshd",
            cmd=["/usr/sbin/sshd", "-D"],
            reason="network interface detected",
        )
        if not dry_run:
            start_service("sshd")
    else:
        logger.info("No network; skipping sshd")

    # 4.5 Start Ollama server (required for Cortex inference)
    ollama_path = "/usr/local/bin/ollama"
    if Path(ollama_path).exists():
        register_service(
            "ollama",
            cmd=[ollama_path, "serve"],
            env={"OLLAMA_HOST": "127.0.0.1:11434", "HOME": "/root"},
        )
        if not dry_run:
            pid = start_service("ollama")
            if pid:
                logger.info("Waiting for Ollama to be ready...")
                if wait_for_port("127.0.0.1", 11434, timeout=60):
                    logger.info("Ollama is ready")
                else:
                    logger.warning("Ollama did not become ready within 60s")
    else:
        logger.warning("Ollama not found at %s; inference will not work", ollama_path)

    # 5. Start Cortex daemon (this process becomes the daemon)
    logger.info("Handing over to Cortex daemon...")
    if dry_run:
        logger.info("[DRY-RUN] Would exec into Cortex daemon now")
        return

    # Import and start daemon inline (same process, becomes asyncio)
    sys.path.insert(0, "/app")
    try:
        from src.daemon import DaemonServer, run_daemon
        from src.hardware_detect import detect_system

        profile = detect_system()
        db_path = os.environ.get("CORTEX_DB", "/tmp/cortex.db")
        logger.info("Cortex DB path: %s", db_path)
        asyncio.run(run_daemon(
            host="0.0.0.0",
            port=11411,
            profile=profile,
            db_path=db_path,
        ))
    except Exception as e:
        logger.exception("Cortex daemon failed: %s", e)
        # Fallback: keep init alive so we can debug
        while True:
            time.sleep(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cortex Init — AI as PID 1")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without doing it")
    parser.add_argument("--services", action="store_true",
                        help="Print registered services and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.services:
        for svc in all_services():
            print(f"{svc['name']:15s} {svc['status']:10s} pid={svc.get('pid')}")
        return

    boot_sequence(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
