# Cortex Boot Makefile
# Build a bootable AI system on a thumbdrive for ANY device.
#
# Quick start:
#   make dry-run          # Test boot sequence locally
#   make usb-image        # Build universal USB image (8GB, L0+L1+L2)
#   make usb-minimal      # Minimal 2GB image (L0 only)
#   make flash THUMB=/dev/sdX  # Write image to physical USB
#   make test             # Boot in QEMU (x86_64)
#   make test-arm         # Boot in QEMU (aarch64)

.PHONY: all usb-image usb-minimal initramfs iso clean test test-arm dry-run flash

REPO_ROOT := $(shell cd .. && pwd)
BUILD_DIR := $(REPO_ROOT)/build
BOOT_DIR  := $(REPO_ROOT)/boot
CACHE_DIR := $(BUILD_DIR)/cache

# Configurable
MODELS ?= L0,L1,L2
SIZE ?= 8192
ARCH ?= x86_64,aarch64
THUMB ?= /dev/sdX

all: usb-image

# ---------------------------------------------------------------------------
# Universal USB Image (EFI + BIOS, multi-arch, pre-baked models)
# ---------------------------------------------------------------------------
usb-image:
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║  Building Cortex Universal USB Image                     ║"
	@echo "║  Models: $(MODELS) | Size: $(SIZE)MB                    ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@sudo python3 $(BOOT_DIR)/build-universal-usb.py \
		--output $(BUILD_DIR)/cortex-universal.img \
		--models $(MODELS) \
		--size $(SIZE) \
		--arch $(ARCH) \
		--write

usb-minimal:
	@echo "[*] Building minimal USB (L0 only, 2GB)..."
	@sudo python3 $(BOOT_DIR)/build-universal-usb.py \
		--output $(BUILD_DIR)/cortex-minimal.img \
		--models L0 \
		--size 2048 \
		--arch x86_64 \
		--write

usb-full:
	@echo "[*] Building full USB (L0-L3, 16GB)..."
	@sudo python3 $(BOOT_DIR)/build-universal-usb.py \
		--output $(BUILD_DIR)/cortex-full.img \
		--models L0,L1,L2,L3 \
		--size 16384 \
		--arch $(ARCH) \
		--write

# ---------------------------------------------------------------------------
# Flash to physical USB device
# ---------------------------------------------------------------------------
flash: $(BUILD_DIR)/cortex-universal.img
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║  WARNING: This will DESTROY all data on $(THUMB)         ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@read -p "  Type YES to continue: " confirm && [ "$$confirm" = "YES" ] || exit 1
	@sudo dd if=$(BUILD_DIR)/cortex-universal.img of=$(THUMB) bs=4M status=progress conv=fsync
	@sync
	@echo "[✓] Cortex USB ready: $(THUMB)"
	@echo "    Plug into any device and boot from USB."

# ---------------------------------------------------------------------------
# Build initramfs only (for development)
# ---------------------------------------------------------------------------
initramfs: $(BUILD_DIR)/initramfs.cpio.gz

$(BUILD_DIR)/initramfs.cpio.gz:
	@echo "[*] Building initramfs..."
	@mkdir -p $(BUILD_DIR)
	@python3 $(BOOT_DIR)/build-initramfs.py

# ---------------------------------------------------------------------------
# Legacy ISO (BIOS-only, for older hardware)
# ---------------------------------------------------------------------------
iso: $(BUILD_DIR)/cortex-aios.iso

$(BUILD_DIR)/cortex-aios.iso: $(BUILD_DIR)/initramfs.cpio.gz
	@echo "[*] Building bootable ISO..."
	@mkdir -p $(BUILD_DIR)/iso/boot/grub
	@echo 'set timeout=5\nset default=0\nmenuentry "Cortex AI OS" {\n  linux /boot/vmlinuz console=ttyS0\n  initrd /boot/initramfs.cpio.gz\n}' > $(BUILD_DIR)/iso/boot/grub/grub.cfg
	@cp $(BUILD_DIR)/initramfs.cpio.gz $(BUILD_DIR)/iso/boot/
	@[ -f $(BUILD_DIR)/vmlinuz ] && cp $(BUILD_DIR)/vmlinuz $(BUILD_DIR)/iso/boot/ || true
	@grub-mkrescue -o $@ $(BUILD_DIR)/iso 2>/dev/null || \
		grub2-mkrescue -o $@ $(BUILD_DIR)/iso 2>/dev/null || \
		echo "[!] grub-mkrescue not found. Install grub2-tools."
	@echo "[✓] ISO ready: $@"

# ---------------------------------------------------------------------------
# Dry-run (test boot logic without side effects)
# ---------------------------------------------------------------------------
dry-run:
	@echo "[*] Dry-run: Cortex init (PID 1 simulation)..."
	@python3 $(BOOT_DIR)/cortex-init.py --dry-run

dry-run-services:
	@python3 $(BOOT_DIR)/cortex-init.py --dry-run --services

# ---------------------------------------------------------------------------
# Test in QEMU (x86_64)
# ---------------------------------------------------------------------------
test: $(BUILD_DIR)/initramfs.cpio.gz
	@echo "[*] Booting Cortex in QEMU (x86_64)..."
	@qemu-system-x86_64 \
		-kernel $(BUILD_DIR)/vmlinuz \
		-initrd $(BUILD_DIR)/initramfs.cpio.gz \
		-append "console=ttyS0 cortex.models=L0 quiet" \
		-nographic -m 4G \
		-smp 4 \
		-netdev user,id=net0,hostfwd=tcp::11411-:11411 \
		-device virtio-net-pci,netdev=net0 \
		$(QEMU_OPTS) || \
		echo "[!] QEMU failed. Install: apt install qemu-system-x86_64"

# Test with USB image (tests full EFI boot path)
test-usb: $(BUILD_DIR)/cortex-universal.img
	@echo "[*] Booting USB image in QEMU (EFI)..."
	@qemu-system-x86_64 \
		-drive file=$(BUILD_DIR)/cortex-universal.img,format=raw \
		-bios /usr/share/ovmf/OVMF.fd \
		-nographic -m 4G \
		-smp 4 \
		-netdev user,id=net0,hostfwd=tcp::11411-:11411 \
		-device virtio-net-pci,netdev=net0 \
		$(QEMU_OPTS) || \
		echo "[!] QEMU+OVMF failed. Install: apt install qemu-system-x86_64 ovmf"

# Test aarch64 (requires qemu-system-aarch64)
test-arm: $(BUILD_DIR)/initramfs-aarch64.cpio.gz
	@echo "[*] Booting Cortex in QEMU (aarch64)..."
	@qemu-system-aarch64 \
		-machine virt -cpu cortex-a72 \
		-kernel $(BUILD_DIR)/cache/kernels/vmlinuz-aarch64 \
		-initrd $(BUILD_DIR)/initramfs-aarch64.cpio.gz \
		-append "console=ttyAMA0 cortex.models=L0 quiet" \
		-nographic -m 4G \
		-smp 4 \
		-netdev user,id=net0,hostfwd=tcp::11411-:11411 \
		-device virtio-net-pci,netdev=net0 \
		$(QEMU_OPTS) || \
		echo "[!] QEMU aarch64 failed. Install: apt install qemu-system-aarch64"

# ---------------------------------------------------------------------------
# Download models only (useful for pre-caching)
# ---------------------------------------------------------------------------
download-models:
	@python3 $(BOOT_DIR)/build-universal-usb.py \
		--models $(MODELS) \
		--skip-download=false \
		2>/dev/null || \
		echo "Use: python3 boot/build-universal-usb.py --models L0,L1,L2 (dry-run downloads)"

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	@echo "[*] Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)

clean-cache:
	@echo "[*] Cleaning download cache..."
	@rm -rf $(CACHE_DIR)

clean-all: clean clean-cache

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "  Cortex Boot — AI as PID 1 on any device"
	@echo ""
	@echo "  Build targets:"
	@echo "    make usb-image     Build 8GB universal USB (EFI+BIOS, x86+ARM)"
	@echo "    make usb-minimal   Build 2GB minimal USB (x86 only, L0)"
	@echo "    make usb-full      Build 16GB full USB (all models)"
	@echo "    make flash THUMB=/dev/sdX   Write to physical USB"
	@echo ""
	@echo "  Test targets:"
	@echo "    make dry-run       Test boot logic (no side effects)"
	@echo "    make test          Boot in QEMU (x86_64)"
	@echo "    make test-arm      Boot in QEMU (aarch64)"
	@echo "    make test-usb      Boot USB image in QEMU (EFI)"
	@echo ""
	@echo "  Variables:"
	@echo "    MODELS=L0,L1,L2    Which model tiers to include"
	@echo "    SIZE=8192          USB image size in MB"
	@echo "    ARCH=x86_64,aarch64  Target architectures"
	@echo "    THUMB=/dev/sdX     USB device for 'make flash'"
	@echo ""
