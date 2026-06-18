# Cortex Boot — AI as PID 1

This directory contains everything needed to build a **bootable Cortex initramfs**:
a Linux system where the first userspace process (`/init`) is Cortex itself.

## Philosophy

Traditional Linux boot: `kernel → initramfs → /sbin/init (systemd) → services → apps`

Cortex boot: `kernel → initramfs → /init (cortex-init.py) → Cortex daemon → AI-native OS`

Cortex is PID 1.  It mounts the filesystems, detects hardware, loads the
L0–L2 inference stack, starts essential services (getty, sshd, networking),
and then listens on :11411 for inference requests.

The system **thinks before it boots**.

## Files

| File | Purpose |
|------|---------|
| `init.sh` | Minimal shell init for initramfs |
| `cortex-init.py` | The real init — PID 1 Python process |
| `build-initramfs.py` | Python builder for the initramfs (tested & working) |
| `Makefile` | `make iso`, `make dry-run`, `make test` (QEMU) |

## Build & Test (Verified Working)

```bash
# 1. Build initramfs (~29MB compressed)
python3 boot/build-initramfs.py

# 2. Test in QEMU (no thumbdrive needed)
make test

# 3. Burn to thumbdrive (DANGER — wipes target device)
make thumb THUMB=/dev/sdX
```

### Verified boot flow

```
[    3.015] Run /init as init process
[    3.990] random: crng init done
Failed to pull qwen3:0.6b, may already be available
Health check failed for unsloth/Qwen3-0.6B-GGUF:Q4_K_M — backend not reachable
Failed to pull qwen3:1.7b, may already be available
Health check failed for unsloth/Qwen3-1.7B-GGUF:Q4_K_M — backend not reachable
Failed to pull qwen3:4b, may already be available
Health check failed for unsloth/Qwen3-4B-GGUF:Q4_K_M — backend not reachable

╔══════════════════════════════════════════════════════════════╗
║  Cortex daemon — AI-native OS inference proxy               ║
╠══════════════════════════════════════════════════════════════╣
║  Listening:  http://0.0.0.0:11411                          ║
║  Compatible with: Cursor, VS Code, Cline, aider, ...       ║
╚══════════════════════════════════════════════════════════════╝

  ○ L0: unsloth/Qwen3-0.6B-GGUF:Q4_K_M [HOT]
  ○ L1: unsloth/Qwen3-1.7B-GGUF:Q4_K_M [HOT]
  ○ L2: unsloth/Qwen3-4B-GGUF:Q4_K_M [HOT]
```

**Notes:**
- Models show ○ (not ●) because Ollama is not available inside QEMU — expected
- On real hardware with Ollama installed, models load as ● ready
- The daemon starts and listens on :11411 regardless of model availability

## Boot Flow

```
1. BIOS/UEFI loads kernel + initramfs
2. Kernel mounts rootfs (or tmpfs overlay)
3. Kernel execs /init → cortex-init.py
4. cortex-init.py:
     a. Reaps zombie processes (SIGCHLD)
     b. Mounts /proc, /sys, /dev, /tmp
     c. Detects hardware profile
     d. Starts getty on tty1 (optional)
     e. Starts sshd if network detected (optional)
     f. Execs into Cortex daemon (asyncio loop)
5. Cortex daemon:
     a. Boots L0–L2 models (if Ollama available)
     b. Exposes /v1/services endpoint
     c. Accepts inference requests on :11411
```

## Services Endpoint

Once booted, query what Cortex thinks should be running:

```bash
curl http://localhost:11411/v1/services
```

Response:
```json
{
  "services": [
    {"name": "cortex-daemon", "status": "running", "pid": 1, "reason": "AI-native OS PID 1"},
    {"name": "cortex-l0", "status": "ready", "model": "qwen3:0.6b", "reason": "Router model loaded"},
    {"name": "cortex-memory", "status": "ready", "db_size_mb": 0.2, "reason": "SQLite WAL active"},
    {"name": "cortex-policy", "status": "active", "mutations": 0, "reason": "Self-modifying policy engine"}
  ]
}
```

## Requirements

- Python 3.10+
- 1GB RAM minimum (for L0–L2)
- 4GB RAM recommended (for L3)
- USB 2.0+ thumbdrive (for ISO)
- GPU optional (CPU inference works, slower)

## License

Same as Cortex — MIT
