# Cortex Boot Makefile
# Build a bootable ISO or initramfs with Cortex as PID 1.

.PHONY: all iso initramfs clean test dry-run

REPO_ROOT := $(shell cd .. && pwd)
BUILD_DIR := $(REPO_ROOT)/build
BOOT_DIR  := $(REPO_ROOT)/boot

all: initramfs

# ---------------------------------------------------------------------------
# Build initramfs only (fast, no ISO)
# ---------------------------------------------------------------------------
initramfs: $(BUILD_DIR)/initramfs.cpio.gz

$(BUILD_DIR)/initramfs.cpio.gz:
	@echo "[*] Building initramfs..."
	@mkdir -p $(BUILD_DIR)
	@python3 $(BOOT_DIR)/build-initramfs.py

# ---------------------------------------------------------------------------
# Build bootable ISO (requires grub2-mkrescue or xorriso)
# ---------------------------------------------------------------------------
iso: $(BUILD_DIR)/cortex-aios.iso

$(BUILD_DIR)/cortex-aios.iso: $(BUILD_DIR)/initramfs.cpio.gz
	@echo "[*] Building bootable ISO..."
	@mkdir -p $(BUILD_DIR)/iso/boot/grub
	@cp $(BOOT_DIR)/grub.cfg $(BUILD_DIR)/iso/boot/grub/grub.cfg 2>/dev/null || \
		echo 'set timeout=5\nset default=0\nmenuentry "Cortex AI OS" {\n  linux /boot/vmlinuz console=ttyS0\n  initrd /boot/initramfs.cpio.gz\n}' > $(BUILD_DIR)/iso/boot/grub/grub.cfg
	@cp $(BUILD_DIR)/initramfs.cpio.gz $(BUILD_DIR)/iso/boot/
	@if [ -f $(BUILD_DIR)/vmlinuz ]; then \
		cp $(BUILD_DIR)/vmlinuz $(BUILD_DIR)/iso/boot/; \
	fi
	@if command -v grub2-mkrescue >/dev/null 2>&1; then \
		grub2-mkrescue -o $(BUILD_DIR)/cortex-aios.iso $(BUILD_DIR)/iso 2>/dev/null; \
	elif command -v grub-mkrescue >/dev/null 2>&1; then \
		grub-mkrescue -o $(BUILD_DIR)/cortex-aios.iso $(BUILD_DIR)/iso 2>/dev/null; \
	else \
		echo "[!] grub-mkrescue not found. Install grub2-tools or build manually."; \
		exit 1; \
	fi
	@echo "[*] ISO ready: $(BUILD_DIR)/cortex-aios.iso"

# ---------------------------------------------------------------------------
# Dry-run test (run cortex-init.py without actually mounting/pivoting)
# ---------------------------------------------------------------------------
dry-run:
	@echo "[*] Dry-run: Cortex init (PID 1 simulation)..."
	@python3 $(BOOT_DIR)/cortex-init.py --dry-run

# ---------------------------------------------------------------------------
# Test in QEMU (requires qemu-system-x86_64)
# ---------------------------------------------------------------------------
test: $(BUILD_DIR)/initramfs.cpio.gz
	@echo "[*] Booting in QEMU..."
	@qemu-system-x86_64 \
		-kernel $(BUILD_DIR)/vmlinuz \
		-initrd $(BUILD_DIR)/initramfs.cpio.gz \
		-append "console=ttyS0 quiet" \
		-nographic -m 2G \
		$(QEMU_OPTS) || \
		echo "[!] QEMU failed. Install qemu-system-x86_64 or run dry-run instead."

# ---------------------------------------------------------------------------
# Clean build artifacts
# ---------------------------------------------------------------------------
clean:
	@echo "[*] Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)

# ---------------------------------------------------------------------------
# Install to thumbdrive (DANGER — wipes target device)
# ---------------------------------------------------------------------------
THUMB ?= /dev/sdX
thumb: $(BUILD_DIR)/cortex-aios.iso
	@echo "[!] WARNING: This will wipe $(THUMB)"
	@read -p "Type YES to continue: " confirm && [ "$$confirm" = "YES" ] || exit 1
	@dd if=$(BUILD_DIR)/cortex-aios.iso of=$(THUMB) bs=4M status=progress
	@echo "[*] Thumbdrive ready: $(THUMB)"
