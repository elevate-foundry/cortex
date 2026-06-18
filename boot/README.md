# Cortex Boot — AI as PID 1

This directory contains everything needed to build a **bootable Cortex ISO**:
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
| `alpine-init.sh` | Build an Alpine Linux initramfs with Cortex |
| `Makefile` | `make iso` produces `cortex-aios.iso` |

## Quick Build

```bash
# Option 1: Build on Alpine (recommended for small initramfs)
make iso

# Option 2: Run init directly (for testing PID-1 behavior)
sudo python3 boot/cortex-init.py --dry-run
```

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
     a. Boots L0–L2 models
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
    {"name": "getty", "status": "running", "pid": 42, "reason": "tty1 detected"},
    {"name": "sshd", "status": "stopped", "reason": "no network interface up"},
    {"name": "cortex-daemon", "status": "running", "pid": 1}
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
