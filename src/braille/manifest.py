"""
Braille Manifests — encode tier/system/routing info as compact Braille strings.

Provides fixed-width Braille encodings for:
  - Routing signatures (4 chars: tier|category|confidence|flags)
  - Tier manifests (capability summary per tier)
  - System manifests (full hardware profile summary)

These are designed for:
  - Compact audit log entries
  - Agent heartbeat payloads
  - Quick visual diffing of system state
"""

from ..tiers import Tier, TIER_SPECS
from ..router import RouteDecision, TaskCategory
from ..hardware_detect import SystemProfile
from .codec import encode_int, decode_int, encode, decode


# Category → single byte mapping
_CATEGORY_TO_BYTE: dict[str, int] = {
    "classify": 0x01,
    "tool_call": 0x02,
    "multi_tool": 0x03,
    "code": 0x04,
    "debug": 0x05,
    "plan": 0x06,
    "analyze": 0x07,
    "generate": 0x08,
    "safety": 0x09,
    "unknown": 0x00,
}

_BYTE_TO_CATEGORY: dict[int, str] = {v: k for k, v in _CATEGORY_TO_BYTE.items()}

# OS → byte mapping
_OS_TO_BYTE: dict[str, int] = {
    "Darwin": 0x01,
    "Linux": 0x02,
    "Windows": 0x03,
}

# Arch → byte mapping
_ARCH_TO_BYTE: dict[str, int] = {
    "arm64": 0x01,
    "x86_64": 0x02,
    "aarch64": 0x01,
}

# Accelerator → byte mapping
_ACCEL_TO_BYTE: dict[str, int] = {
    "apple_metal": 0x01,
    "nvidia_cuda": 0x02,
    "amd_rocm": 0x03,
    "cpu_only": 0x00,
}


def routing_signature(decision: RouteDecision) -> str:
    """Encode a routing decision as a 4-character Braille signature.

    Format: [tier][category][confidence][flags]
      - tier:       1 byte (0–7 for L0–L7)
      - category:   1 byte (see _CATEGORY_TO_BYTE)
      - confidence:  1 byte (0–255, maps to 0.0–1.0)
      - flags:      1 byte (bit 0: has_escalation_hint)

    Total: 4 Braille characters = 32 bits.

    Args:
        decision: The RouteDecision to encode.

    Returns:
        4-character Braille string.
    """
    tier_byte = decision.tier.value
    cat_byte = _CATEGORY_TO_BYTE.get(decision.category.value, 0x00)
    conf_byte = min(255, max(0, int(decision.confidence * 255)))

    flags = 0
    if decision.escalation_hint is not None:
        flags |= 0x01

    data = bytes([tier_byte, cat_byte, conf_byte, flags])
    return encode(data)


def decode_routing_signature(sig: str) -> dict:
    """Decode a 4-character routing signature back to a dict.

    Args:
        sig: 4-character Braille routing signature.

    Returns:
        Dict with keys: tier, category, confidence, has_escalation_hint.
    """
    data = decode(sig)
    if len(data) != 4:
        raise ValueError(f"Expected 4-byte signature, got {len(data)}")

    tier_val = data[0]
    cat_val = data[1]
    conf_val = data[2]
    flags = data[3]

    return {
        "tier": f"L{tier_val}",
        "category": _BYTE_TO_CATEGORY.get(cat_val, "unknown"),
        "confidence": round(conf_val / 255, 3),
        "has_escalation_hint": bool(flags & 0x01),
    }


def tier_manifest(tier: Tier, profile: SystemProfile) -> str:
    """Encode a tier's capability manifest as a Braille string.

    Format (6 bytes = 6 Braille chars):
      [tier][param_min_gb][param_max_gb][vram_req_gb][always_hot][feasible]

    Args:
        tier: The tier to encode.
        profile: System profile for feasibility check.

    Returns:
        6-character Braille string.
    """
    spec = TIER_SPECS.get(tier)
    if spec is None:
        return encode(bytes(6))

    # Approximate VRAM requirements in GB
    vram_budget = getattr(profile, "total_vram_mb", 0)
    if hasattr(profile, "gpus") and profile.gpus:
        vram_budget = profile.gpus[0].vram_mb
    elif hasattr(profile, "memory"):
        vram_budget = int(profile.memory.total_mb * 0.75)

    vram_req_gb = int(spec.min_params_b * 0.6)  # rough Q4 estimate
    feasible = 1 if (vram_budget / 1024) >= vram_req_gb else 0

    data = bytes([
        tier.value,                              # tier
        int(spec.min_params_b),                  # min params (GB)
        int(spec.max_params_b),                  # max params (GB)
        min(255, vram_req_gb),                   # VRAM req (GB, capped)
        1 if spec.always_hot else 0,             # always hot
        feasible,                                # feasible on this hardware
    ])
    return encode(data)


def system_manifest(profile: SystemProfile) -> str:
    """Encode a full system profile as a Braille manifest.

    Format (8 bytes = 8 Braille chars):
      [os][arch][accel][vram_gb][ram_gb][cores][max_tier][num_backends]

    Args:
        profile: The system profile to encode.

    Returns:
        8-character Braille string.
    """
    os_byte = _OS_TO_BYTE.get(profile.os_name, 0x00)
    arch_byte = _ARCH_TO_BYTE.get(profile.arch, 0x00)
    accel_byte = _ACCEL_TO_BYTE.get(profile.primary_accelerator, 0x00)

    vram_gb = min(255, profile.total_vram_mb // 1024)

    ram_gb = 0
    if hasattr(profile, "memory"):
        ram_gb = min(255, profile.memory.total_mb // 1024)

    cores = 0
    if hasattr(profile, "cpu"):
        cores = min(255, profile.cpu.physical_cores)

    # Count available backends
    num_backends = sum(1 for b in profile.backends if b.available)

    # Estimate max tier (simplified)
    max_tier = min(7, vram_gb // 2)  # rough heuristic

    data = bytes([
        os_byte, arch_byte, accel_byte, vram_gb,
        ram_gb, cores, max_tier, num_backends,
    ])
    return encode(data)


def decode_system_manifest(manifest: str) -> dict:
    """Decode a system manifest back to a human-readable dict."""
    data = decode(manifest)
    if len(data) != 8:
        raise ValueError(f"Expected 8-byte manifest, got {len(data)}")

    os_map = {v: k for k, v in _OS_TO_BYTE.items()}
    arch_map = {v: k for k, v in _ARCH_TO_BYTE.items()}
    accel_map = {v: k for k, v in _ACCEL_TO_BYTE.items()}

    return {
        "os": os_map.get(data[0], "unknown"),
        "arch": arch_map.get(data[1], "unknown"),
        "accelerator": accel_map.get(data[2], "unknown"),
        "vram_gb": data[3],
        "ram_gb": data[4],
        "cores": data[5],
        "max_tier": f"L{data[6]}",
        "num_backends": data[7],
    }
