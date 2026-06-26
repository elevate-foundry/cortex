#!/usr/bin/env python3
"""
Cortex Init — PID 1 bootstrap for AI-native operating system.

When the Linux kernel boots, it looks for /init.  This file IS /init.
It reaps zombies, mounts filesystems, detects hardware, loads GPU drivers,
configures networking, starts llama.cpp (or Ollama), and becomes the
Cortex inference daemon.

Design principles:
  - Zero external dependencies at boot (no Ollama required)
  - GPU auto-detection: NVIDIA, AMD, Intel, Apple Metal
  - Network auto-config: DHCP + mDNS (cortex.local)
  - Model manifest from CORTEX partition (pre-baked GGUF files)
  - Fallback chain: llama.cpp → Ollama → CPU-only inference
  - Survives any failure: always drops to a shell or keeps running

Usage (as PID 1):
    exec python3 /app/cortex-init.py

Usage (dry-run for testing):
    python3 boot/cortex-init.py --dry-run

Kernel command line options:
    cortex.models=L0,L1,L2    Which tiers to load
    cortex.rescue=1           Drop to shell instead of daemon
    cortex.port=11411         Override daemon port
    cortex.gpu=nvidia         Force GPU type
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.init")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = "2.0.0"
CORTEX_MOUNT = "/mnt/cortex"
MODEL_DIR = f"{CORTEX_MOUNT}/models/gguf"
MANIFEST_PATH = f"{CORTEX_MOUNT}/models/manifest.json"
CONFIG_PATH = f"{CORTEX_MOUNT}/etc/cortex.toml"
DB_PATH_DEFAULT = f"{CORTEX_MOUNT}/var/lib/cortex.db"
LLAMA_SERVER_PORT = 8080
DAEMON_PORT = 11411


# ---------------------------------------------------------------------------
# Service registry
# ---------------------------------------------------------------------------

SERVICES: dict[str, dict] = {}


def register_service(
    name: str,
    cmd: list[str],
    env: Optional[dict] = None,
    reason: str = "",
    restart: bool = False,
) -> None:
    """Register a service to be managed by the init process."""
    SERVICES[name] = {
        "cmd": cmd,
        "env": env or {},
        "pid": None,
        "status": "stopped",
        "started_at": 0,
        "reason": reason,
        "restart": restart,
        "restart_count": 0,
    }


def start_service(name: str) -> Optional[int]:
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
    except FileNotFoundError:
        logger.warning("Cannot start %s: %s not found", name, svc["cmd"][0])
        svc["status"] = "failed"
        return None
    except Exception as e:
        logger.warning("Cannot start %s: %s", name, e)
        svc["status"] = "failed"
        return None


def stop_service(name: str) -> bool:
    """Stop a running service."""
    svc = SERVICES.get(name)
    if not svc or svc["pid"] is None:
        return False
    try:
        os.kill(svc["pid"], signal.SIGTERM)
        # Give it 2 seconds then SIGKILL
        time.sleep(0.5)
        try:
            os.kill(svc["pid"], 0)  # check alive
            time.sleep(1.5)
            os.kill(svc["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
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
        "restart_count": svc.get("restart_count", 0),
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
            pid, status = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
            ZOMBIES_REAPED += 1
            for svc_name, svc in SERVICES.items():
                if svc.get("pid") == pid:
                    svc["pid"] = None
                    svc["status"] = "exited"
                    logger.info("Service %s exited (pid=%d, status=%d)", svc_name, pid, status)
                    # Emit SCL lifecycle record
                    try:
                        from src.lifecycle_scl import service_failed, service_restart
                        service_failed(svc_name, reason="process_exit", exit_code=status)
                    except Exception:
                        pass
                    # Auto-restart if configured
                    if svc.get("restart") and svc["restart_count"] < 5:
                        svc["restart_count"] += 1
                        logger.info("Restarting %s (attempt %d)", svc_name, svc["restart_count"])
                        start_service(svc_name)
                        # Emit SCL restart record
                        try:
                            from src.lifecycle_scl import service_restart as _sr
                            _sr(svc_name, attempt=svc["restart_count"], status="success")
                        except Exception:
                            pass
                    break
        except ChildProcessError:
            break


def _setup_pid1():
    """Install signal handlers required for PID-1 operation."""
    signal.signal(signal.SIGCHLD, _reap_zombies)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)


def _shutdown(signum=None, frame=None):
    """Graceful shutdown — stop all services in reverse order."""
    sig_name = signal.Signals(signum).name if signum else "UNKNOWN"
    logger.info("Init received %s, shutting down services...", sig_name)
    for name in reversed(list(SERVICES)):
        stop_service(name)
    logger.info("All services stopped. Goodbye.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Kernel command line parsing
# ---------------------------------------------------------------------------

def parse_cmdline() -> dict:
    """Parse kernel command line for cortex.* parameters."""
    params = {}
    try:
        cmdline = Path("/proc/cmdline").read_text().strip()
    except Exception:
        return params

    for token in cmdline.split():
        if token.startswith("cortex."):
            key_val = token[7:]  # strip "cortex."
            if "=" in key_val:
                key, val = key_val.split("=", 1)
                params[key] = val
            else:
                params[key_val] = "1"
    return params


# ---------------------------------------------------------------------------
# Hardware detection & GPU driver loading
# ---------------------------------------------------------------------------

def detect_gpu() -> dict:
    """Detect GPU type and load appropriate kernel module."""
    gpu = {"type": "none", "name": "", "vram_mb": 0, "driver_loaded": False}

    # NVIDIA
    if Path("/dev/nvidia0").exists() or _pci_has_vendor("10de"):
        gpu["type"] = "nvidia"
        gpu["name"] = _get_nvidia_name()
        gpu["vram_mb"] = _get_nvidia_vram()
        gpu["driver_loaded"] = _load_module("nvidia") or Path("/dev/nvidia0").exists()
        return gpu

    # AMD
    if _pci_has_vendor("1002"):
        gpu["type"] = "amd"
        gpu["name"] = "AMD GPU"
        gpu["driver_loaded"] = _load_module("amdgpu") or Path("/dev/dri/renderD128").exists()
        return gpu

    # Intel Arc
    if _pci_has_vendor("8086") and Path("/dev/dri").exists():
        gpu["type"] = "intel"
        gpu["name"] = "Intel GPU"
        gpu["driver_loaded"] = True  # i915 usually auto-loaded
        return gpu

    # Check for Apple Silicon (won't have /proc/bus/pci)
    try:
        uname = os.uname()
        if "apple" in uname.machine.lower() or "arm64" == uname.machine:
            # Could be running on Apple Silicon via Asahi Linux
            if Path("/dev/dri").exists():
                gpu["type"] = "apple"
                gpu["name"] = "Apple Silicon GPU"
                gpu["driver_loaded"] = True
                return gpu
    except Exception:
        pass

    return gpu


def _pci_has_vendor(vendor_id: str) -> bool:
    """Check if any PCI device matches a vendor ID."""
    try:
        result = subprocess.run(
            ["lspci", "-n"], capture_output=True, text=True, timeout=5
        )
        return vendor_id in result.stdout.lower()
    except Exception:
        # Fallback: check /sys/bus/pci
        pci_path = Path("/sys/bus/pci/devices")
        if pci_path.exists():
            for dev_dir in pci_path.iterdir():
                try:
                    vendor = (dev_dir / "vendor").read_text().strip()
                    if vendor_id in vendor.lower():
                        return True
                except Exception:
                    continue
        return False


def _get_nvidia_name() -> str:
    """Get NVIDIA GPU name."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "NVIDIA GPU"


def _get_nvidia_vram() -> int:
    """Get NVIDIA VRAM in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return 0


def _load_module(module: str) -> bool:
    """Try to load a kernel module."""
    try:
        result = subprocess.run(
            ["modprobe", module], capture_output=True, timeout=10
        )
        if result.returncode == 0:
            logger.info("Loaded kernel module: %s", module)
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Network configuration
# ---------------------------------------------------------------------------

def configure_network(dry_run: bool = False) -> dict:
    """Auto-configure networking: DHCP + mDNS."""
    net = {"interfaces": [], "ip": None, "mdns": False}

    # Find network interfaces
    net_path = Path("/sys/class/net")
    if not net_path.exists():
        return net

    for iface in net_path.iterdir():
        name = iface.name
        if name == "lo":
            continue
        # Check if carrier is present (cable plugged in or wifi connected)
        try:
            carrier = (iface / "carrier").read_text().strip()
            if carrier == "1":
                net["interfaces"].append(name)
        except Exception:
            # carrier file may not exist or may error — check operstate
            try:
                state = (iface / "operstate").read_text().strip()
                if state in ("up", "unknown"):
                    net["interfaces"].append(name)
            except Exception:
                continue

    if not net["interfaces"]:
        logger.info("No active network interfaces found")
        return net

    if dry_run:
        logger.info("[DRY-RUN] Would configure DHCP on %s", net["interfaces"])
        return net

    # Bring up loopback
    subprocess.run(["ip", "link", "set", "lo", "up"], capture_output=True)

    # DHCP on first active interface
    primary = net["interfaces"][0]
    logger.info("Configuring DHCP on %s...", primary)
    subprocess.run(["ip", "link", "set", primary, "up"], capture_output=True)

    # Try dhclient, udhcpc (busybox), or dhcpcd
    for dhcp_cmd in [
        ["dhclient", "-1", primary],
        ["udhcpc", "-i", primary, "-n", "-q"],
        ["dhcpcd", "-1", primary],
    ]:
        if shutil.which(dhcp_cmd[0]):
            result = subprocess.run(dhcp_cmd, capture_output=True, timeout=15)
            if result.returncode == 0:
                break

    # Get assigned IP
    try:
        result = subprocess.run(
            ["ip", "-4", "addr", "show", primary],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "inet " in line:
                net["ip"] = line.strip().split()[1].split("/")[0]
                break
    except Exception:
        pass

    if net["ip"]:
        logger.info("Network configured: %s = %s", primary, net["ip"])
    else:
        logger.warning("DHCP failed on %s", primary)

    # Set hostname
    subprocess.run(["hostname", "cortex-ai"], capture_output=True)

    # Start mDNS (avahi) if available
    avahi = shutil.which("avahi-daemon")
    if avahi:
        register_service(
            "avahi",
            cmd=[avahi, "--no-drop-root", "--no-chroot"],
            reason="mDNS: cortex.local",
            restart=True,
        )
        start_service("avahi")
        net["mdns"] = True

    return net


# ---------------------------------------------------------------------------
# Model & Backend discovery
# ---------------------------------------------------------------------------

def load_model_manifest() -> dict:
    """Load model manifest from CORTEX partition."""
    if Path(MANIFEST_PATH).exists():
        try:
            return json.loads(Path(MANIFEST_PATH).read_text())
        except Exception as e:
            logger.warning("Failed to load model manifest: %s", e)
    return {"models": {}}


def find_llama_server() -> Optional[str]:
    """Find llama-server (llama.cpp) binary."""
    for path in [
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
        "/app/bin/llama-server",
        "/opt/llama.cpp/build/bin/llama-server",
    ]:
        if Path(path).exists():
            return path
    # Check PATH
    import shutil as _shutil
    return _shutil.which("llama-server")


def find_ollama() -> Optional[str]:
    """Find Ollama binary."""
    for path in ["/usr/local/bin/ollama", "/usr/bin/ollama"]:
        if Path(path).exists():
            return path
    import shutil as _shutil
    return _shutil.which("ollama")


def start_inference_backend(
    manifest: dict,
    gpu: dict,
    requested_models: list[str],
    dry_run: bool = False,
) -> str:
    """
    Start the best available inference backend.

    Priority:
      1. llama-server with GGUF models from CORTEX partition
      2. Ollama (if installed)
      3. Fail gracefully (daemon starts but no models)

    Returns: backend type ("llama_cpp", "ollama", "none")
    """
    models = manifest.get("models", {})

    # Determine which model to load first (smallest available)
    model_path = None
    model_context = 4096
    for tier in ["L0", "L1", "L2", "L3"]:
        if tier in models:
            candidate = Path(models[tier]["path"])
            if candidate.exists():
                model_path = str(candidate)
                model_context = models[tier].get("context", 4096)
                logger.info("Selected model for boot: %s (%s)", tier, candidate.name)
                break

    # Strategy 1: llama-server (preferred — no Ollama dependency)
    llama = find_llama_server()
    if llama and model_path:
        logger.info("Starting llama-server with %s", model_path)

        llama_cmd = [
            llama,
            "--model", model_path,
            "--host", "127.0.0.1",
            "--port", str(LLAMA_SERVER_PORT),
            "--ctx-size", str(model_context),
            "--threads", str(max(1, os.cpu_count() - 1) if os.cpu_count() else 4),
            "--parallel", "2",
        ]

        # GPU-specific flags
        if gpu["type"] == "nvidia" and gpu["driver_loaded"]:
            llama_cmd += ["--n-gpu-layers", "999"]  # offload all layers
        elif gpu["type"] == "amd" and gpu["driver_loaded"]:
            llama_cmd += ["--n-gpu-layers", "999"]
        elif gpu["type"] == "apple":
            llama_cmd += ["--n-gpu-layers", "999"]  # Metal
        # else: CPU-only (no flag needed)

        # Flash attention if supported
        llama_cmd += ["--flash-attn"]

        register_service(
            "llama-server",
            cmd=llama_cmd,
            reason=f"llama.cpp inference ({Path(model_path).name})",
            restart=True,
        )
        if not dry_run:
            start_service("llama-server")
            # Wait for llama-server to be ready
            if wait_for_port("127.0.0.1", LLAMA_SERVER_PORT, timeout=30):
                logger.info("llama-server ready on :%d", LLAMA_SERVER_PORT)
                return "llama_cpp"
            else:
                logger.warning("llama-server failed to start")
        else:
            return "llama_cpp"

    # Strategy 2: Ollama
    ollama = find_ollama()
    if ollama:
        logger.info("Falling back to Ollama")
        register_service(
            "ollama",
            cmd=[ollama, "serve"],
            env={"OLLAMA_HOST": "127.0.0.1:11434", "HOME": "/root"},
            reason="Ollama inference backend",
            restart=True,
        )
        if not dry_run:
            start_service("ollama")
            if wait_for_port("127.0.0.1", 11434, timeout=60):
                logger.info("Ollama ready on :11434")
                return "ollama"
            else:
                logger.warning("Ollama failed to start")
        else:
            return "ollama"

    # Strategy 3: No backend available
    logger.warning("No inference backend available. Daemon will start but cannot serve models.")
    return "none"


# ---------------------------------------------------------------------------
# Mount & filesystem
# ---------------------------------------------------------------------------

def mount_virtualfs():
    """Mount proc, sys, dev, tmp — the bare minimum for a Linux system."""
    mounts = [
        ("proc", "/proc", "proc", ""),
        ("sysfs", "/sys", "sysfs", ""),
        ("devtmpfs", "/dev", "devtmpfs", ""),
        ("tmpfs", "/tmp", "tmpfs", "size=512M"),
        ("devpts", "/dev/pts", "devpts", ""),
        ("tmpfs", "/run", "tmpfs", "size=64M"),
    ]
    for src, dst, fstype, opts in mounts:
        Path(dst).mkdir(parents=True, exist_ok=True)
        try:
            cmd = ["mount", "-t", fstype]
            if opts:
                cmd += ["-o", opts]
            cmd += [src, dst]
            subprocess.run(cmd, check=False, capture_output=True)
        except FileNotFoundError:
            pass


def mount_cortex_partition() -> Optional[str]:
    """Find and mount the CORTEX USB partition for persistence."""
    # Try blkid first
    try:
        result = subprocess.run(
            ["blkid", "-L", "CORTEX"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            dev = result.stdout.strip()
            os.makedirs(CORTEX_MOUNT, exist_ok=True)
            subprocess.run(
                ["mount", "-t", "ext4", "-o", "rw,noatime", dev, CORTEX_MOUNT],
                capture_output=True, timeout=10, check=False
            )
            if os.path.ismount(CORTEX_MOUNT):
                logger.info("Mounted CORTEX partition at %s (%s)", CORTEX_MOUNT, dev)
                return CORTEX_MOUNT
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("blkid failed: %s", e)

    # Fallback: scan /dev/sd* and /dev/nvme* for CORTEX label
    for pattern in ["/dev/sd[a-z][1-9]*", "/dev/nvme*p[1-9]*", "/dev/mmcblk*p[1-9]*"]:
        import glob
        for dev in sorted(glob.glob(pattern)):
            try:
                result = subprocess.run(
                    ["e2label", dev], capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0 and "CORTEX" in result.stdout:
                    os.makedirs(CORTEX_MOUNT, exist_ok=True)
                    subprocess.run(
                        ["mount", "-t", "ext4", "-o", "rw,noatime", dev, CORTEX_MOUNT],
                        capture_output=True, timeout=10, check=False
                    )
                    if os.path.ismount(CORTEX_MOUNT):
                        logger.info("Mounted CORTEX partition at %s (%s)", CORTEX_MOUNT, dev)
                        return CORTEX_MOUNT
            except Exception:
                continue

    logger.info("No CORTEX partition found")
    return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def wait_for_port(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for a TCP port to become reachable."""
    import socket
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


def detect_ram_mb() -> int:
    """Get total RAM in MB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def detect_arch() -> str:
    """Detect CPU architecture."""
    machine = os.uname().machine
    if machine in ("x86_64", "AMD64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return machine


# ---------------------------------------------------------------------------
# Boot sequence
# ---------------------------------------------------------------------------

def boot_sequence(dry_run: bool = False):
    """Full boot sequence for Cortex as PID 1."""
    t0 = time.monotonic()

    # Banner (print + flush before any log output)
    sys.stdout.write(f"""
┌─────────────────────────────────────────────────────────────┐
│  Cortex AI OS v{VERSION} — PID 1 Boot                          │
│  The machine thinks before it boots.                        │
└─────────────────────────────────────────────────────────────┘

""")
    sys.stdout.flush()

    logger.info("=== Cortex Init v%s ===", VERSION)
    logger.info("PID: %d | arch: %s | dry_run: %s", os.getpid(), detect_arch(), dry_run)

    # 1. PID-1 signal plumbing
    if os.getpid() == 1:
        _setup_pid1()
        logger.info("[1/8] PID-1 signal handlers installed")
    else:
        logger.info("[1/8] Running as PID %d (not init)", os.getpid())

    # 2. Mount virtual filesystems
    logger.info("[2/8] Mounting virtual filesystems...")
    if not dry_run:
        mount_virtualfs()
    else:
        logger.info("  [DRY-RUN] skipped")

    # Parse kernel cmdline
    cmdline = parse_cmdline()
    logger.info("  Kernel params: %s", cmdline)

    # Rescue mode?
    if cmdline.get("rescue") == "1":
        logger.info("RESCUE MODE — dropping to shell")
        os.execv("/bin/sh", ["/bin/sh"])
        return

    # 3. Mount CORTEX partition
    logger.info("[3/9] Searching for CORTEX partition...")
    cortex_mount = None if dry_run else mount_cortex_partition()
    if cortex_mount:
        os.environ["CORTEX_DATA_DIR"] = cortex_mount
        os.environ["CORTEX_DB"] = f"{cortex_mount}/var/lib/cortex.db"
    else:
        os.environ.setdefault("CORTEX_DB", "/tmp/cortex.db")
        logger.info("  Running without persistence (tmpfs)")

    # 3.5. Check boot cache (self-modifying OS)
    # If we've seen this hardware before, skip expensive detection
    telemetry = None
    cached_config = None
    try:
        # Add search paths for src/ imports (initramfs: /app, dev: repo root)
        for p in ["/app", str(Path(__file__).resolve().parent.parent), "."]:
            if p not in sys.path:
                sys.path.insert(0, p)
        from src.boot_telemetry import BootTelemetry, BootRecord, detect_fingerprint, HardwareFingerprint
        from src.micro_engine import probe_and_configure

        data_dir = f"{cortex_mount}/var/lib" if cortex_mount else "/tmp"
        telemetry = BootTelemetry(data_dir)

        # Check if we already know this hardware
        if not dry_run:
            hw_fp = detect_fingerprint()
            cached_config = telemetry.get_cached_config(hw_fp.fingerprint)
            if cached_config:
                logger.info("  ⚡ CACHED CONFIG found (boot #%d on this hardware)", cached_config.boot_count)
                logger.info("    Best boot: %.0fms | Avg TTFT: %.0fms",
                            cached_config.best_boot_ms, cached_config.avg_ttft_ms)
    except ImportError:
        logger.info("  Boot telemetry not available (first boot?)")
    except Exception as e:
        logger.warning("  Boot cache check failed: %s", e)

    # 4. Hardware detection + GPU (skip if cached)
    logger.info("[4/9] Detecting hardware...")
    t_hw = time.monotonic()
    if cached_config and cached_config.skip_gpu_detect and not dry_run:
        logger.info("  Using cached GPU config (skipping detection)")
        gpu = {"type": "nvidia" if cached_config.gpu_layers > 0 else "none",
               "name": "cached", "vram_mb": 0, "driver_loaded": cached_config.gpu_layers > 0}
        ram_mb = detect_ram_mb()
    else:
        ram_mb = detect_ram_mb()
        gpu = detect_gpu() if not dry_run else {"type": "none", "name": "", "vram_mb": 0, "driver_loaded": False}
    arch = detect_arch()
    gpu_detect_ms = (time.monotonic() - t_hw) * 1000
    logger.info("  RAM: %d MB | Arch: %s | GPU: %s (%s, %d MB VRAM) [%.0fms]",
                ram_mb, arch, gpu["type"], gpu["name"], gpu["vram_mb"], gpu_detect_ms)

    # 4.5. CKM probe — AI as PID 1 makes the boot decision
    #   The Cortex Kernel Model (0.5B GGUF) infers optimal config from hardware state.
    #   Falls back to heuristics if CKM not available.
    probe_config = None
    t_probe = time.monotonic()
    try:
        if not dry_run:
            probe_config = probe_and_configure()
            probe_ms = (time.monotonic() - t_probe) * 1000
            source = probe_config.get("_source", "unknown")
            if source == "ckm":
                logger.info("  🧠 CKM DECISION (AI as PID 1): threads=%d, gpu_layers=%d, "
                            "ctx=%d, models=%s [%.0fms]",
                            probe_config.get("thread_count", 0),
                            probe_config.get("gpu_layers", 0),
                            probe_config.get("context_size", 4096),
                            probe_config.get("hot_models", []),
                            probe_ms)
            else:
                logger.info("  Heuristic probe (%s): threads=%d, gpu_layers=%d, models=%s [%.0fms]",
                            source,
                            probe_config.get("thread_count", 0),
                            probe_config.get("gpu_layers", 0),
                            probe_config.get("hot_models", []),
                            probe_ms)
    except Exception as e:
        logger.warning("  Boot probe failed (using defaults): %s", e)

    # 5. Network configuration
    logger.info("[5/9] Configuring network...")
    t_net = time.monotonic()
    if cached_config and cached_config.skip_network and not dry_run:
        logger.info("  Skipping network (cached: not needed)")
        net = {"interfaces": [], "ip": None, "mdns": False}
    else:
        net = configure_network(dry_run=dry_run)
    network_ms = (time.monotonic() - t_net) * 1000
    if net.get("ip"):
        logger.info("  IP: %s | mDNS: %s [%.0fms]", net["ip"],
                    "cortex.local" if net["mdns"] else "disabled", network_ms)

    # 6. Start inference backend
    logger.info("[6/9] Starting inference backend...")
    t_backend = time.monotonic()
    manifest = load_model_manifest()
    requested_models = cmdline.get("models", "L0,L1,L2").split(",")

    # Apply cached/probed config to backend startup
    if cached_config:
        # Use learned optimal settings
        os.environ.setdefault("CORTEX_THREADS", str(cached_config.thread_count))
        os.environ.setdefault("CORTEX_GPU_LAYERS", str(cached_config.gpu_layers))
        os.environ.setdefault("CORTEX_CTX_SIZE", str(cached_config.context_size))
    elif probe_config:
        # Use micro-engine probe results
        os.environ.setdefault("CORTEX_THREADS", str(probe_config.get("thread_count", 4)))
        os.environ.setdefault("CORTEX_GPU_LAYERS", str(probe_config.get("gpu_layers", 0)))
        os.environ.setdefault("CORTEX_CTX_SIZE", str(probe_config.get("context_size", 4096)))

    backend_type = start_inference_backend(manifest, gpu, requested_models, dry_run=dry_run)
    backend_ms = (time.monotonic() - t_backend) * 1000
    logger.info("  Backend: %s [%.0fms]", backend_type, backend_ms)

    # Set env vars for Cortex daemon
    if backend_type == "llama_cpp":
        os.environ["OLLAMA_URL"] = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"
    elif backend_type == "ollama":
        os.environ["OLLAMA_URL"] = "http://127.0.0.1:11434"

    # 7. Optional services
    logger.info("[7/9] Starting optional services...")
    # getty
    if Path("/dev/tty1").exists() and not dry_run:
        register_service("getty", cmd=["/sbin/getty", "38400", "tty1"], reason="console")
        start_service("getty")
    # sshd
    if net.get("interfaces") and not dry_run:
        sshd = shutil.which("sshd") if hasattr(shutil, "which") else "/usr/sbin/sshd"
        import shutil as _shutil
        sshd_path = _shutil.which("sshd")
        if sshd_path:
            register_service("sshd", cmd=[sshd_path, "-D"], reason="remote access", restart=True)
            start_service("sshd")

    # 7b. Optional: background CKM self-training (if idle and configured)
    ckm_auto_train = cmdline.get("ckm_train", "false").lower() in ("1", "true", "yes")
    if ckm_auto_train and not dry_run:
        logger.info("  Background CKM training scheduled (post-boot)")
        try:
            from src.lifecycle_scl import boot_phase
            boot_phase("ckm_train_scheduled", status="queued")
        except Exception:
            pass
        # Fork background trainer (non-blocking)
        try:
            trainer_pid = os.fork()
            if trainer_pid == 0:
                # Child process — run training
                try:
                    sys.path.insert(0, "/app")
                    from src.ckm.cli import main as ckm_main
                    ckm_main(["run", "--target", "ckm", "--time-budget", "5m",
                              "--data-dir", "/mnt/cortex/var/lib/ckm/data",
                              "--output-dir", "/mnt/cortex/var/lib/ckm/output"])
                except Exception as e:
                    logger.warning("CKM background training failed: %s", e)
                os._exit(0)
            else:
                register_service("ckm_trainer", cmd=[], reason="self-training",
                                 restart=False)
                SERVICES["ckm_trainer"]["pid"] = trainer_pid
                SERVICES["ckm_trainer"]["status"] = "running"
        except Exception as e:
            logger.warning("  CKM trainer fork failed: %s", e)

    # 8. Log boot telemetry (self-modifying OS)
    logger.info("[8/9] Recording boot telemetry...")
    elapsed = time.monotonic() - t0
    if telemetry and not dry_run:
        try:
            import uuid
            boot_record = BootRecord(
                boot_id=str(uuid.uuid4())[:8],
                timestamp_ms=int(time.time() * 1000),
                hardware_fp=hw_fp.fingerprint if 'hw_fp' in dir() else "",
                hardware={"gpu_type": gpu["type"], "gpu_name": gpu["name"],
                          "ram_mb": ram_mb, "arch": arch},
                boot_total_ms=elapsed * 1000,
                gpu_detect_ms=gpu_detect_ms,
                network_ms=network_ms,
                backend_start_ms=backend_ms,
                backend_type=backend_type,
                models_loaded=requested_models,
                gpu_layers=int(os.environ.get("CORTEX_GPU_LAYERS", "0")),
                thread_count=int(os.environ.get("CORTEX_THREADS", "0")),
                context_size=int(os.environ.get("CORTEX_CTX_SIZE", "4096")),
                used_cache=cached_config is not None,
            )
            telemetry.log_boot(boot_record)
            logger.info("  Boot logged (id=%s, %.0fms, cache=%s)",
                        boot_record.boot_id, elapsed * 1000, boot_record.used_cache)

            # Background: optimize config for next boot
            if 'hw_fp' in dir():
                optimized = telemetry.optimize(hw_fp.fingerprint)
                if optimized:
                    logger.info("  Config optimized for next boot (boot #%d)", optimized.boot_count)
        except Exception as e:
            logger.warning("  Telemetry logging failed: %s", e)

    # 9. Start Cortex daemon
    logger.info("[9/9] Starting Cortex daemon...")
    daemon_port = int(cmdline.get("port", str(DAEMON_PORT)))

    logger.info("Boot sequence complete in %.1fs. Handing over to Cortex.", elapsed)

    if dry_run:
        print(f"\n  [DRY-RUN] Boot sequence would complete in {elapsed:.1f}s")
        print(f"  [DRY-RUN] Daemon would listen on 0.0.0.0:{daemon_port}")
        print(f"  [DRY-RUN] Backend: {backend_type}")
        print(f"  [DRY-RUN] Models: {list(manifest.get('models', {}).keys())}")
        return

    # Import and start daemon inline (same process, becomes asyncio)
    sys.path.insert(0, "/app")
    try:
        from src.daemon import run_daemon
        from src.hardware_detect import detect_system

        profile = detect_system()
        db_path = os.environ.get("CORTEX_DB", "/tmp/cortex.db")

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Cortex AI OS — PID 1 Active                                ║
╠══════════════════════════════════════════════════════════════╣
║  Listen:   http://0.0.0.0:{daemon_port:<5d}                            ║
║  Backend:  {backend_type:<20s}                        ║
║  GPU:      {gpu['name'] or 'CPU-only':<20s}                        ║
║  RAM:      {ram_mb} MB                                         ║
║  Network:  {net.get('ip') or 'none':<15s}                             ║
║  mDNS:     {'cortex.local' if net.get('mdns') else 'disabled':<15s}                             ║
╠══════════════════════════════════════════════════════════════╣
║  export OPENAI_BASE_URL=http://{'cortex.local' if net.get('mdns') else net.get('ip', 'localhost')}:{daemon_port}/v1  ║
╚══════════════════════════════════════════════════════════════╝
""")

        asyncio.run(run_daemon(
            host="0.0.0.0",
            port=daemon_port,
            profile=profile,
            db_path=db_path,
        ))
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down")
        _shutdown(signal.SIGINT)
    except Exception as e:
        logger.exception("Cortex daemon failed: %s", e)
        print(f"\n[!] Daemon crashed: {e}")
        print("    Dropping to rescue shell. Type 'reboot' to restart.")
        # Fallback: exec into shell so we can debug
        if Path("/bin/sh").exists():
            os.execv("/bin/sh", ["/bin/sh"])
        else:
            # Absolute fallback: keep PID 1 alive
            while True:
                time.sleep(60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Cortex Init — AI as PID 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run (test boot sequence without mounting/starting)
  python3 boot/cortex-init.py --dry-run

  # Show what services would be registered
  python3 boot/cortex-init.py --dry-run --services

  # Real boot (as PID 1 inside initramfs)
  exec python3 /app/cortex-init.py
""",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate boot without side effects")
    parser.add_argument("--services", action="store_true",
                        help="Print registered services and exit")
    parser.add_argument("--version", action="version", version=f"cortex-init {VERSION}")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.services:
        # Quick boot to register services, then show
        boot_sequence(dry_run=True)
        print("\nRegistered services:")
        for svc in all_services():
            print(f"  {svc['name']:15s} {svc['status']:10s} {svc.get('reason', '')}")
        return

    boot_sequence(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
