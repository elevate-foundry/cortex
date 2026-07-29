# Cortex Boot — AI as PID 1 on Any Device

Plug a thumbdrive into **any machine** — laptop, server, Raspberry Pi, workstation —
and it boots directly into an AI inference system. No OS install required.

```
Traditional: BIOS → kernel → systemd → 200 services → maybe AI
Cortex:      BIOS → kernel → cortex-init.py → AI is PID 1 (5 seconds)
```

## What's New (v2.0)

| Feature | Before | Now |
|---------|--------|-----|
| Boot mode | BIOS only (GRUB MBR) | **EFI + BIOS hybrid** (boots anywhere) |
| Architecture | x86_64 only | **x86_64 + aarch64** (same stick) |
| Inference | Ollama required | **llama.cpp primary** (zero dependencies) |
| Models | Downloaded at boot | **Pre-baked GGUF on USB** (instant inference) |
| GPU | Manual setup | **Auto-detect + driver loading** (NVIDIA/AMD/Intel/Apple) |
| Network | None | **DHCP + mDNS** (cortex.local) |
| Recovery | Stuck if crash | **Rescue shell** (cortex.rescue=1) |
| Persistence | tmpfs (lost on reboot) | **ext4 partition** (state survives) |

## Quick Start

```bash
# 1. Build universal USB image (8GB, includes L0+L1+L2 models)
cd boot && make usb-image

# 2. Flash to USB stick
make flash THUMB=/dev/sdX

# 3. Boot any machine from USB, then:
export OPENAI_BASE_URL=http://cortex.local:11411/v1
curl $OPENAI_BASE_URL/chat/completions \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}]}'
```

## USB Stick Layout

```
┌────────────────────────────────────────────────────────────┐
│ Partition 1: EFI System Partition (FAT32, 512MB)           │
│   /EFI/BOOT/BOOTX64.EFI   ← GRUB for x86_64 EFI         │
│   /EFI/BOOT/BOOTAA64.EFI  ← GRUB for aarch64 EFI        │
│   /boot/vmlinuz-x86_64    ← Linux kernel (x86)           │
│   /boot/vmlinuz-aarch64   ← Linux kernel (ARM)           │
│   /boot/initramfs-*.gz    ← Cortex + Python + llama.cpp  │
│   /grub/grub.cfg          ← Auto-detects architecture    │
├────────────────────────────────────────────────────────────┤
│ Partition 2: CORTEX (ext4, rest of disk)                   │
│   /models/gguf/*.gguf     ← Pre-baked model weights      │
│   /models/manifest.json   ← Model registry               │
│   /var/lib/cortex.db      ← Persistent memory (SQLite)   │
│   /var/log/cortex.log     ← Boot + request logs          │
│   /etc/cortex.toml        ← Configuration                │
│   /AGENTS.md              ← Agent coordination manifest  │
└────────────────────────────────────────────────────────────┘
```

## Boot Flow (v2.0)

```
1. BIOS/UEFI loads GRUB from EFI partition
2. GRUB auto-detects CPU arch, loads correct kernel + initramfs
3. Kernel execs /init (init.sh):
     a. Mount /proc, /sys, /dev, /tmp
     b. Load GPU kernel modules (nvidia, amdgpu, i915)
     c. Mount CORTEX persistence partition
     d. Set environment, check for rescue mode
     e. Exec into cortex-init.py
4. cortex-init.py (PID 1):
     a. Install SIGCHLD handler (zombie reaping)
     b. Detect hardware: RAM, GPU type, VRAM
     c. Configure network (DHCP + mDNS: cortex.local)
     d. Load model manifest from CORTEX partition
     e. Start llama-server with GPU offloading
     f. Start optional services (getty, sshd, avahi)
     g. Become Cortex daemon (asyncio, port 11411)
5. Ready for inference in ~5 seconds
```

## Files

| File | Purpose |
|------|---------|
| `init.sh` | Shell bootstrap (mounts, GPU drivers, env) |
| `cortex-init.py` | Python PID 1 (hardware, network, backend, daemon) |
| `build-universal-usb.py` | **New:** Multi-arch USB builder |
| `build-initramfs.py` | Initramfs builder (Python + libs + Cortex) |
| `build-usb.py` | Legacy USB builder (BIOS-only, deprecated) |
| `Makefile` | All build/test targets |

## Build Variants

```bash
# Minimal (2GB USB, L0 only, x86_64, CPU inference)
make usb-minimal

# Standard (8GB USB, L0+L1+L2, dual-arch)
make usb-image

# Full (16GB USB, L0-L3, dual-arch, heavy inference)
make usb-full

# Custom
make usb-image MODELS=L0,L1 SIZE=4096 ARCH=x86_64
```

## Testing

```bash
# Dry-run (no root, no hardware, shows boot sequence)
make dry-run

# QEMU x86_64 (with port forward)
make test
# → curl http://localhost:11411/health from host

# QEMU aarch64
make test-arm

# QEMU EFI boot (full USB image test)
make test-usb
```

## Kernel Command Line Options

Pass these via GRUB to control boot behavior:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `cortex.models=L0,L1,L2` | `L0,L1,L2` | Which tiers to load |
| `cortex.port=11411` | `11411` | Daemon listen port |
| `cortex.rescue=1` | off | Drop to shell for debugging |
| `cortex.gpu=nvidia` | auto | Force GPU type |

## Hardware Support

| Device Type | Status | Notes |
|-------------|--------|-------|
| x86_64 PC (UEFI) | ✅ Full | Any modern PC/laptop |
| x86_64 PC (BIOS) | ✅ Full | Legacy machines (2005+) |
| NVIDIA GPU | ✅ Auto | Driver loaded at boot |
| AMD GPU (ROCm) | ✅ Auto | amdgpu module |
| Intel Arc | ✅ Auto | i915 module |
| Raspberry Pi 4/5 | ✅ Full | aarch64 kernel |
| AWS Graviton | ✅ Full | aarch64 EFI |
| Apple Silicon | ⚠️ Experimental | Via Asahi Linux kernel |
| CPU-only | ✅ Full | Slower but works everywhere |

## Inference Backend Priority

```
1. llama-server + GGUF from USB  → Zero dependencies, fastest boot
2. Ollama (if installed on host) → Broader model support
3. None (daemon starts, no model) → API responds with 503
```

## Network Access

Once booted:
```bash
# From another machine on the same network
curl http://cortex.local:11411/health         # mDNS
curl http://192.168.1.x:11411/health          # Direct IP

# SSH into the Cortex OS
ssh root@cortex.local

# Use with any OpenAI-compatible client
export OPENAI_BASE_URL=http://cortex.local:11411/v1
export OPENAI_API_KEY=local
```

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| USB stick | 2 GB | 8-16 GB |
| RAM | 1 GB | 4+ GB |
| CPU | Any x86_64 or aarch64 | 4+ cores |
| GPU | None (CPU works) | NVIDIA 8GB+ |
| Boot mode | BIOS or UEFI | UEFI preferred |

## License

Same as Cortex — MIT
