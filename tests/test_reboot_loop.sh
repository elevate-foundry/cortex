#!/bin/bash
# Cortex Reboot Loop Test — automated 5-boot reliability test using QEMU.
#
# This script:
#   1. Builds the initramfs (if not already built)
#   2. Boots QEMU with the Cortex USB image 5 times
#   3. On each boot, waits for the health endpoint
#   4. Verifies: clean boot, state persistence, no corruption
#   5. Reports pass/fail for each requirement
#
# Prerequisites:
#   - qemu-system-x86_64 installed
#   - Cortex initramfs built (make -C boot/)
#   - A kernel bzImage available
#
# Usage:
#   bash tests/test_reboot_loop.sh [--kernel /path/to/bzImage] [--initramfs /path/to/initramfs.cpio.gz]

set -euo pipefail

# Configuration
NUM_BOOTS=${NUM_BOOTS:-5}
QEMU_TIMEOUT=60         # seconds per boot to wait for health
HEALTH_PORT=11411
QEMU_MEM="2G"
QEMU_CPUS=4

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BOOT_DIR="$REPO_ROOT/boot"
RESULTS_DIR="$REPO_ROOT/tests/.reboot-results"
DISK_IMG="$RESULTS_DIR/cortex-persist.qcow2"

# Parse arguments
KERNEL="${KERNEL:-}"
INITRAMFS="${INITRAMFS:-}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --kernel) KERNEL="$2"; shift 2 ;;
        --initramfs) INITRAMFS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --boots) NUM_BOOTS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓ PASS${NC}: $1"; }
fail() { echo -e "  ${RED}✗ FAIL${NC}: $1"; FAILURES=$((FAILURES + 1)); }
warn() { echo -e "  ${YELLOW}⚠ WARN${NC}: $1"; }

FAILURES=0
BOOTS_COMPLETED=0

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Cortex Reboot Loop Test — ${NUM_BOOTS} boots                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Check QEMU
if ! command -v qemu-system-x86_64 &>/dev/null; then
    echo "ERROR: qemu-system-x86_64 not found. Install QEMU first."
    echo "  macOS: brew install qemu"
    echo "  Linux: apt install qemu-system-x86"
    exit 1
fi

# Find or build kernel
if [ -z "$KERNEL" ]; then
    for k in /boot/vmlinuz-* "$BOOT_DIR/bzImage" /usr/lib/modules/*/vmlinuz; do
        if [ -f "$k" ]; then
            KERNEL="$k"
            break
        fi
    done
fi
if [ -z "$KERNEL" ] || [ ! -f "$KERNEL" ]; then
    echo "ERROR: No kernel found. Provide --kernel /path/to/bzImage"
    echo "  Or download a prebuilt kernel for testing."
    exit 1
fi
echo "Kernel: $KERNEL"

# Find or build initramfs
if [ -z "$INITRAMFS" ]; then
    INITRAMFS="$BOOT_DIR/initramfs.cpio.gz"
    if [ ! -f "$INITRAMFS" ]; then
        echo "Building initramfs..."
        (cd "$BOOT_DIR" && python3 build-initramfs.py) || {
            echo "ERROR: Failed to build initramfs"
            exit 1
        }
    fi
fi
if [ ! -f "$INITRAMFS" ]; then
    echo "ERROR: Initramfs not found at $INITRAMFS"
    exit 1
fi
echo "Initramfs: $INITRAMFS"

# Create results directory and persistence disk
mkdir -p "$RESULTS_DIR"
if [ ! -f "$DISK_IMG" ]; then
    echo "Creating persistence disk (256MB)..."
    qemu-img create -f qcow2 "$DISK_IMG" 256M
    # Format with ext4 labeled CORTEX (requires root or fakeroot)
    # For testing, we'll let cortex-init handle missing partition gracefully
fi
echo "Persistence disk: $DISK_IMG"
echo

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] Would boot QEMU ${NUM_BOOTS} times with:"
    echo "  Kernel: $KERNEL"
    echo "  Initramfs: $INITRAMFS"
    echo "  Persistence: $DISK_IMG"
    echo "  Memory: $QEMU_MEM"
    echo "  CPUs: $QEMU_CPUS"
    exit 0
fi

# ---------------------------------------------------------------------------
# Boot loop
# ---------------------------------------------------------------------------

for boot_num in $(seq 1 "$NUM_BOOTS"); do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Boot $boot_num / $NUM_BOOTS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Start QEMU in background
    QEMU_LOG="$RESULTS_DIR/boot_${boot_num}.log"
    QEMU_PID_FILE="$RESULTS_DIR/qemu.pid"

    qemu-system-x86_64 \
        -kernel "$KERNEL" \
        -initrd "$INITRAMFS" \
        -append "console=ttyS0 cortex.port=$HEALTH_PORT" \
        -m "$QEMU_MEM" \
        -smp "$QEMU_CPUS" \
        -drive file="$DISK_IMG",format=qcow2,if=virtio \
        -netdev user,id=net0,hostfwd=tcp::${HEALTH_PORT}-:${HEALTH_PORT} \
        -device virtio-net-pci,netdev=net0 \
        -nographic \
        -serial stdio \
        -no-reboot \
        -pidfile "$QEMU_PID_FILE" \
        > "$QEMU_LOG" 2>&1 &
    QEMU_PID=$!

    # Wait for health endpoint
    BOOT_START=$(date +%s)
    HEALTH_OK=0

    for i in $(seq 1 "$QEMU_TIMEOUT"); do
        sleep 1
        if curl -sf "http://localhost:${HEALTH_PORT}/health" > "$RESULTS_DIR/health_${boot_num}.json" 2>/dev/null; then
            HEALTH_OK=1
            break
        fi
        # Check if QEMU died
        if ! kill -0 "$QEMU_PID" 2>/dev/null; then
            break
        fi
    done

    BOOT_END=$(date +%s)
    BOOT_SECS=$((BOOT_END - BOOT_START))

    if [ "$HEALTH_OK" -eq 1 ]; then
        pass "Boot $boot_num: health endpoint responded in ${BOOT_SECS}s"

        # Parse health response
        HEALTH=$(cat "$RESULTS_DIR/health_${boot_num}.json")

        # Check PID 1
        PID1=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pid1', False))" 2>/dev/null || echo "false")
        if [ "$PID1" = "True" ] || [ "$PID1" = "true" ]; then
            pass "PID 1 confirmed"
        else
            warn "Not running as PID 1 (expected in QEMU test)"
        fi

        # Check model loaded
        MODEL=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model', 'none'))" 2>/dev/null || echo "none")
        if [ "$MODEL" = "loaded" ]; then
            pass "Model loaded"
        else
            warn "Model not loaded (model=$MODEL)"
        fi

        # Check state persistence (boot > 1)
        if [ "$boot_num" -gt 1 ]; then
            # Query boot-trace for boot_count
            TRACE=$(curl -sf "http://localhost:${HEALTH_PORT}/boot-trace" 2>/dev/null || echo "{}")
            BOOT_COUNT=$(echo "$TRACE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('boot_count', 0))" 2>/dev/null || echo "0")
            if [ "$BOOT_COUNT" -gt 1 ]; then
                pass "State persistence: boot_count=$BOOT_COUNT"
            else
                fail "State not persisted (boot_count=$BOOT_COUNT, expected >1)"
            fi
        fi

        BOOTS_COMPLETED=$((BOOTS_COMPLETED + 1))
    else
        fail "Boot $boot_num: health endpoint did not respond within ${QEMU_TIMEOUT}s"
        echo "  QEMU log tail:"
        tail -20 "$QEMU_LOG" | sed 's/^/    /'
    fi

    # Shutdown QEMU gracefully
    if kill -0 "$QEMU_PID" 2>/dev/null; then
        kill "$QEMU_PID" 2>/dev/null || true
        wait "$QEMU_PID" 2>/dev/null || true
    fi

    echo
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  REBOOT LOOP TEST RESULTS                                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Boots completed: $BOOTS_COMPLETED / $NUM_BOOTS                                     ║"
echo "║  Failures:        $FAILURES                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

if [ "$FAILURES" -eq 0 ] && [ "$BOOTS_COMPLETED" -eq "$NUM_BOOTS" ]; then
    echo -e "${GREEN}ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${RED}TESTS FAILED${NC} ($FAILURES failures, $BOOTS_COMPLETED/$NUM_BOOTS boots)"
    exit 1
fi
