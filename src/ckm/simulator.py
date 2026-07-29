"""
CKM Boot Simulator — deterministic environment for RL training.

This is the key enabler: a fast, deterministic boot simulator that replaces
real hardware. The CKM model learns to boot an OS by playing episodes in
this environment, receiving rewards for speed, correctness, and safety.

State space:
  - hardware: {cpu_type, cores, ram_mb, gpu_type, vram_mb, arch}
  - services: dict[name → {status, pid, restart_count, ram_mb, vram_mb}]
  - resources: {cpu_pct, ram_used_mb, vram_used_mb, disk_used_pct}
  - network: enum (down, connecting, up, failed)
  - boot_phase: enum (init, hw_detect, config, backend_start, services, ready)
  - persistence: dict[key → value]  (survives reboot)
  - time_ms: int (simulated clock)

Action space (discrete + continuous):
  - verb: one of N_VERBS boot-relevant verbs
  - target: one of N_TARGETS boot-relevant targets
  - params: dict of bounded numeric config values

Dynamics (deterministic with configurable noise):
  - Starting a service: takes 100-2000ms, consumes RAM/VRAM
  - Killing a service: frees resources after 50ms
  - Configuring: instant, changes params
  - Detecting hardware: takes 200ms, reveals true hw state
  - Network up: probabilistic (configurable success rate)

Episode termination:
  - Boot complete (all required services healthy) → SUCCESS
  - Timeout (> max_time_ms) → TIMEOUT
  - Safety violation → VIOLATION
  - Unrecoverable crash (OOM, all retries exhausted) → CRASH
"""

import copy
import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional

logger = logging.getLogger("cortex.ckm.simulator")


# ---------------------------------------------------------------------------
# Enums — finite state spaces
# ---------------------------------------------------------------------------

class BootPhase(IntEnum):
    """Boot progression phases."""
    INIT = 0
    HW_DETECT = 1
    CONFIG = 2
    BACKEND_START = 3
    SERVICES = 4
    READY = 5


class NetworkState(IntEnum):
    """Network interface states."""
    DOWN = 0
    CONNECTING = 1
    UP = 2
    FAILED = 3


class ServiceStatus(IntEnum):
    """Service lifecycle states."""
    STOPPED = 0
    STARTING = 1
    RUNNING = 2
    FAILED = 3
    FROZEN = 4


class Verb(IntEnum):
    """All boot-relevant verbs (action space dimension 1)."""
    DETECT = 0
    CONFIGURE = 1
    SPAWN = 2
    KILL = 3
    RESTART = 4
    NETWORK_UP = 5
    READ_CACHE = 6
    WRITE_CACHE = 7
    OBSERVE = 8
    DENY = 9
    ESCALATE = 10
    NOOP = 11


class Target(IntEnum):
    """All boot-relevant targets (action space dimension 2)."""
    HARDWARE = 0
    INFERENCE = 1
    API = 2
    LOGGER = 3
    NETWORK = 4
    CONFIG = 5
    PERSISTENCE = 6
    SYSTEM = 7


N_VERBS = len(Verb)
N_TARGETS = len(Target)
N_PARAMS = 6  # threads, gpu_layers, ctx_size, batch_size, port, timeout_ms
N_BOOT_PHASES = len(BootPhase)
N_SERVICE_STATES = len(ServiceStatus)
N_SERVICES = 4  # inference, api, logger, network


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    """Immutable hardware description for an episode."""
    cpu_type: str = "generic"
    cores: int = 4
    ram_mb: int = 8192
    gpu_type: str = "none"       # none, nvidia, amd, apple, intel
    gpu_name: str = "none"
    vram_mb: int = 0
    arch: str = "x86_64"         # x86_64, aarch64

    def to_vector(self) -> list[float]:
        """Convert to fixed-size numeric vector for embedding."""
        gpu_map = {"none": 0, "nvidia": 1, "amd": 2, "apple": 3, "intel": 4}
        arch_map = {"x86_64": 0, "aarch64": 1}
        return [
            self.cores / 128.0,               # normalized
            self.ram_mb / 131072.0,            # normalized to 128GB
            gpu_map.get(self.gpu_type, 0) / 4.0,
            self.vram_mb / 49152.0,            # normalized to 48GB
            arch_map.get(self.arch, 0),
        ]

    def fingerprint(self) -> str:
        """Hardware fingerprint for caching."""
        s = f"{self.cpu_type}:{self.cores}:{self.ram_mb}:{self.gpu_type}:{self.vram_mb}"
        return hashlib.sha256(s.encode()).hexdigest()[:8]


@dataclass
class ServiceState:
    """State of one managed service."""
    status: ServiceStatus = ServiceStatus.STOPPED
    ram_mb: int = 0
    vram_mb: int = 0
    restart_count: int = 0
    start_time_ms: int = 0      # when it started (for latency calc)
    ready_time_ms: int = 0      # when it became ready


@dataclass
class Resources:
    """Current resource utilization."""
    cpu_pct: float = 0.0
    ram_used_mb: int = 0
    vram_used_mb: int = 0
    disk_used_pct: float = 10.0


@dataclass
class Action:
    """One discrete+continuous action in the CKM action space."""
    verb: Verb
    target: Target
    params: dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> list[float]:
        """Convert to fixed-size numeric vector."""
        v = [0.0] * (N_VERBS + N_TARGETS + N_PARAMS)
        v[self.verb] = 1.0                    # one-hot verb
        v[N_VERBS + self.target] = 1.0        # one-hot target
        # params (normalized)
        param_keys = ["threads", "gpu_layers", "ctx_size", "batch_size", "port", "timeout_ms"]
        param_maxes = [128, 999, 131072, 64, 65535, 30000]
        for i, key in enumerate(param_keys):
            if key in self.params:
                v[N_VERBS + N_TARGETS + i] = self.params[key] / param_maxes[i]
        return v

    @classmethod
    def from_vector(cls, v: list[float]) -> "Action":
        """Reconstruct from vector (argmax for discrete, denorm for continuous)."""
        verb_idx = max(range(N_VERBS), key=lambda i: v[i])
        target_idx = max(range(N_TARGETS), key=lambda i: v[N_VERBS + i])
        param_keys = ["threads", "gpu_layers", "ctx_size", "batch_size", "port", "timeout_ms"]
        param_maxes = [128, 999, 131072, 64, 65535, 30000]
        params = {}
        for i, key in enumerate(param_keys):
            val = v[N_VERBS + N_TARGETS + i] * param_maxes[i]
            if val > 0:
                params[key] = val
        return cls(verb=Verb(verb_idx), target=Target(target_idx), params=params)


@dataclass
class BootState:
    """Complete simulator state at one timestep."""
    hardware: HardwareProfile
    services: dict[str, ServiceState] = field(default_factory=dict)
    resources: Resources = field(default_factory=Resources)
    network: NetworkState = NetworkState.DOWN
    phase: BootPhase = BootPhase.INIT
    persistence: dict[str, str] = field(default_factory=dict)
    time_ms: int = 0
    config: dict[str, float] = field(default_factory=dict)
    hw_detected: bool = False

    def to_vector(self) -> list[float]:
        """Convert full state to fixed-size vector for embedding."""
        v = []
        # Hardware (5 dims)
        v.extend(self.hardware.to_vector())
        # Boot phase (1 dim normalized)
        v.append(self.phase / (N_BOOT_PHASES - 1))
        # Network (1 dim normalized)
        v.append(self.network / 3.0)
        # hw_detected flag
        v.append(1.0 if self.hw_detected else 0.0)
        # Services (4 services × status normalized)
        for name in ["inference", "api", "logger", "network_svc"]:
            svc = self.services.get(name)
            if svc:
                v.append(svc.status / (N_SERVICE_STATES - 1))
            else:
                v.append(0.0)
        # Resources (4 dims)
        v.append(self.resources.cpu_pct)
        v.append(self.resources.ram_used_mb / max(self.hardware.ram_mb, 1))
        v.append(self.resources.vram_used_mb / max(self.hardware.vram_mb, 1) if self.hardware.vram_mb > 0 else 0.0)
        v.append(self.resources.disk_used_pct / 100.0)
        # Config params (6 dims, normalized)
        param_keys = ["threads", "gpu_layers", "ctx_size", "batch_size", "port", "timeout_ms"]
        param_maxes = [128, 999, 131072, 64, 65535, 30000]
        for key, mx in zip(param_keys, param_maxes):
            v.append(self.config.get(key, 0) / mx)
        # Time (1 dim, normalized to 10s)
        v.append(min(self.time_ms / 10000.0, 1.0))
        return v

    @property
    def state_dim(self) -> int:
        """Dimensionality of state vector."""
        return len(self.to_vector())


STATE_DIM = 23  # 5 hw + 1 phase + 1 net + 1 hw_det + 4 svc + 4 res + 6 cfg + 1 time
ACTION_DIM = N_VERBS + N_TARGETS + N_PARAMS


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------

@dataclass
class Fault:
    """A fault that can be injected into the simulator."""
    name: str
    trigger_ms: int
    triggered: bool = False

    def should_trigger(self, current_ms: int) -> bool:
        if not self.triggered and current_ms >= self.trigger_ms:
            self.triggered = True
            return True
        return False

    def apply(self, state: BootState) -> BootState:
        """Override in subclasses."""
        return state


class GPUDisappearFault(Fault):
    def apply(self, state: BootState) -> BootState:
        state.hardware = copy.copy(state.hardware)
        state.hardware.gpu_type = "none"
        state.hardware.vram_mb = 0
        # Kill GPU-dependent services
        for name, svc in state.services.items():
            if svc.vram_mb > 0:
                svc.status = ServiceStatus.FAILED
                state.resources.vram_used_mb -= svc.vram_mb
                svc.vram_mb = 0
        return state


class RAMPressureFault(Fault):
    def apply(self, state: BootState) -> BootState:
        # Simulate extreme memory pressure — reduce available to 100MB
        state.resources.ram_used_mb = state.hardware.ram_mb - 100
        # OOM-kill the largest service
        largest = max(state.services.items(), key=lambda x: x[1].ram_mb, default=(None, None))
        if largest[0] and largest[1].ram_mb > 0:
            largest[1].status = ServiceStatus.FAILED
            state.resources.ram_used_mb -= largest[1].ram_mb
        return state


class DiskFullFault(Fault):
    def apply(self, state: BootState) -> BootState:
        state.resources.disk_used_pct = 99.9
        return state


class ServiceOOMFault(Fault):
    def __init__(self, name: str, trigger_ms: int, service: str):
        super().__init__(name, trigger_ms)
        self.service = service

    def apply(self, state: BootState) -> BootState:
        svc = state.services.get(self.service)
        if svc and svc.status == ServiceStatus.RUNNING:
            state.resources.ram_used_mb -= svc.ram_mb
            state.resources.vram_used_mb -= svc.vram_mb
            svc.status = ServiceStatus.FAILED
            svc.ram_mb = 0
            svc.vram_mb = 0
        return state


class NetworkDownFault(Fault):
    def apply(self, state: BootState) -> BootState:
        state.network = NetworkState.FAILED
        return state


class CorruptedCacheFault(Fault):
    def apply(self, state: BootState) -> BootState:
        state.persistence = {"corrupted": "true"}
        return state


class PortConflictFault(Fault):
    def __init__(self, name: str, trigger_ms: int, port: int = 8080):
        super().__init__(name, trigger_ms)
        self.port = port

    def apply(self, state: BootState) -> BootState:
        # Simulate port already in use — block service that tries to bind
        for name, svc in state.services.items():
            if svc.status == ServiceStatus.STARTING:
                svc.status = ServiceStatus.FAILED
        return state


# Pre-defined fault library
FAULT_LIBRARY = [
    lambda: GPUDisappearFault("gpu_disappear", trigger_ms=500),
    lambda: RAMPressureFault("ram_pressure", trigger_ms=200),
    lambda: DiskFullFault("disk_full", trigger_ms=800),
    lambda: ServiceOOMFault("inference_oom", trigger_ms=1500, service="inference"),
    lambda: ServiceOOMFault("api_oom", trigger_ms=2000, service="api"),
    lambda: NetworkDownFault("network_down", trigger_ms=1000),
    lambda: CorruptedCacheFault("corrupted_cache", trigger_ms=0),
    lambda: PortConflictFault("port_conflict", trigger_ms=300),
]


# ---------------------------------------------------------------------------
# Hardware profile generator
# ---------------------------------------------------------------------------

HARDWARE_PROFILES = [
    HardwareProfile("Apple M1", 8, 16384, "apple", "Apple M1", 16384, "aarch64"),
    HardwareProfile("Apple M2 Pro", 12, 32768, "apple", "Apple M2 Pro", 32768, "aarch64"),
    HardwareProfile("Apple M4 Max", 16, 131072, "apple", "Apple M4 Max", 131072, "aarch64"),
    HardwareProfile("Intel i5-12400", 6, 16384, "none", "none", 0, "x86_64"),
    HardwareProfile("AMD Ryzen 9 7950X", 16, 65536, "nvidia", "RTX 4090", 24576, "x86_64"),
    HardwareProfile("AMD Ryzen 7 5800X", 8, 32768, "nvidia", "RTX 3080", 10240, "x86_64"),
    HardwareProfile("Intel i7-13700K", 16, 32768, "nvidia", "RTX 4070 Ti", 12288, "x86_64"),
    HardwareProfile("Raspberry Pi 5", 4, 8192, "none", "none", 0, "aarch64"),
    HardwareProfile("AMD EPYC 7763", 64, 524288, "nvidia", "A100", 81920, "x86_64"),
    HardwareProfile("Intel Celeron N5105", 4, 4096, "none", "none", 0, "x86_64"),
    HardwareProfile("AMD Ryzen 5 5600G", 6, 16384, "amd", "Radeon Vega 7", 2048, "x86_64"),
    HardwareProfile("Jetson Orin Nano", 6, 8192, "nvidia", "Orin GPU", 8192, "aarch64"),
]


def random_hardware() -> HardwareProfile:
    """Sample a random hardware profile."""
    return random.choice(HARDWARE_PROFILES)


# ---------------------------------------------------------------------------
# Dangerous targets/verbs (from policy SCL)
# ---------------------------------------------------------------------------

DANGEROUS_TARGETS = {"/dev/mem", "/dev/kmem", "/proc/kcore", "/dev/sda",
                     "/dev/nvme0", "/dev/port", "/dev/hda", "/dev/vda", "/dev/mmcblk0"}

DANGEROUS_VERBS = {Verb.KILL}  # Kill system target is dangerous in boot context


def is_dangerous(action: Action) -> bool:
    """Check if an action violates safety policy."""
    # Direct dangerous target reference in params
    for key, val in action.params.items():
        if isinstance(val, str) and val in DANGEROUS_TARGETS:
            return True
    # Kill + system target
    if action.verb == Verb.KILL and action.target == Target.SYSTEM:
        return True
    return False


# ---------------------------------------------------------------------------
# Service dynamics
# ---------------------------------------------------------------------------

# Service resource costs (ram_mb, vram_mb, startup_ms)
SERVICE_SPECS = {
    "inference": {"ram_mb": 2048, "vram_factor": 0.7, "startup_ms": 1500},
    "api": {"ram_mb": 128, "vram_factor": 0.0, "startup_ms": 200},
    "logger": {"ram_mb": 64, "vram_factor": 0.0, "startup_ms": 100},
    "network_svc": {"ram_mb": 32, "vram_factor": 0.0, "startup_ms": 300},
}


def _service_vram(name: str, hw: HardwareProfile) -> int:
    """Calculate VRAM usage for a service given hardware."""
    factor = SERVICE_SPECS.get(name, {}).get("vram_factor", 0.0)
    return int(hw.vram_mb * factor) if factor > 0 else 0


def _service_startup_ms(name: str) -> int:
    """Base startup time (before config adjustments)."""
    return SERVICE_SPECS.get(name, {}).get("startup_ms", 500)


# ---------------------------------------------------------------------------
# Boot Simulator
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Result of a completed episode."""
    outcome: str           # "success", "timeout", "violation", "crash"
    total_reward: float
    steps: int
    time_ms: int
    transitions: list[tuple]   # (state_vec, action_vec, reward, next_state_vec)
    final_state: BootState


class CortexBootSimulator:
    """
    Deterministic boot environment for RL training.

    Usage:
        sim = CortexBootSimulator(hardware=random_hardware())
        state = sim.reset()
        while not done:
            action = model.act(state)
            state, reward, done, info = sim.step(action)
    """

    def __init__(
        self,
        hardware: Optional[HardwareProfile] = None,
        faults: Optional[list[Fault]] = None,
        max_time_ms: int = 10000,
        network_success_rate: float = 0.9,
        seed: Optional[int] = None,
    ):
        self.hardware = hardware or random_hardware()
        self.faults = faults or []
        self.max_time_ms = max_time_ms
        self.network_success_rate = network_success_rate
        self.rng = random.Random(seed)
        self.state: Optional[BootState] = None
        self.history: list[tuple[BootState, Action, float]] = []
        self.step_count = 0
        self.max_steps = 30  # prevent infinite loops

    def reset(self) -> BootState:
        """Reset simulator to initial boot state."""
        # Reset faults
        for f in self.faults:
            f.triggered = False

        self.state = BootState(
            hardware=copy.deepcopy(self.hardware),
            services={},
            resources=Resources(cpu_pct=0.05, ram_used_mb=256, vram_used_mb=0, disk_used_pct=10.0),
            network=NetworkState.DOWN,
            phase=BootPhase.INIT,
            persistence=self._generate_persistence(),
            time_ms=0,
            config={},
            hw_detected=False,
        )
        self.history = []
        self.step_count = 0
        return self.state

    def step(self, action: Action) -> tuple[BootState, float, bool, dict]:
        """
        Execute one action in the environment.

        Returns: (next_state, reward, done, info)
        """
        assert self.state is not None, "Call reset() before step()"
        self.step_count += 1

        # 1. Safety check (immediate catastrophic penalty)
        if is_dangerous(action):
            info = {"reason": "safety_violation", "action": action}
            self.history.append((copy.deepcopy(self.state), action, -50.0))
            return self.state, -50.0, True, info

        # 2. Apply dynamics
        prev_state = copy.deepcopy(self.state)
        self._apply_dynamics(action)

        # 3. Inject faults
        for fault in self.faults:
            if fault.should_trigger(self.state.time_ms):
                self.state = fault.apply(self.state)

        # 4. Compute reward
        reward = self._compute_reward(prev_state, action, self.state)

        # 5. Check termination
        done, info = self._check_terminal()

        self.history.append((prev_state, action, reward))
        return self.state, reward, done, info

    def _apply_dynamics(self, action: Action) -> None:
        """Apply action effects to state (mutates self.state)."""
        s = self.state

        if action.verb == Verb.DETECT and action.target == Target.HARDWARE:
            # Hardware detection takes time but reveals state
            s.time_ms += 200
            s.hw_detected = True
            if s.phase == BootPhase.INIT:
                s.phase = BootPhase.HW_DETECT
            s.resources.cpu_pct = min(s.resources.cpu_pct + 0.1, 1.0)

        elif action.verb == Verb.CONFIGURE:
            # Configuration is instant
            for key, val in action.params.items():
                s.config[key] = val
            if s.phase.value < BootPhase.CONFIG.value:
                s.phase = BootPhase.CONFIG
            s.time_ms += 10

        elif action.verb == Verb.READ_CACHE and action.target == Target.PERSISTENCE:
            # Read from persistence (fast if available)
            s.time_ms += 50
            if s.persistence and "corrupted" not in s.persistence:
                # Apply cached config
                for key in ["threads", "gpu_layers", "ctx_size"]:
                    if key in s.persistence:
                        s.config[key] = float(s.persistence[key])
                s.hw_detected = True  # cache implies prior detection
                s.phase = BootPhase.CONFIG

        elif action.verb == Verb.WRITE_CACHE and action.target == Target.PERSISTENCE:
            # Persist current config for next boot
            s.time_ms += 50
            for key, val in s.config.items():
                s.persistence[key] = str(val)
            s.persistence["hw_fingerprint"] = s.hardware.fingerprint()

        elif action.verb == Verb.SPAWN:
            # Start a service
            svc_name = self._target_to_service(action.target)
            if svc_name and svc_name not in s.services:
                spec = SERVICE_SPECS.get(svc_name, {})
                ram_cost = spec.get("ram_mb", 256)
                vram_cost = _service_vram(svc_name, s.hardware)
                startup_ms = _service_startup_ms(svc_name)

                # Check if we have resources
                if s.resources.ram_used_mb + ram_cost > s.hardware.ram_mb:
                    # OOM — can't start
                    s.services[svc_name] = ServiceState(
                        status=ServiceStatus.FAILED, ram_mb=0, vram_mb=0
                    )
                elif vram_cost > 0 and s.resources.vram_used_mb + vram_cost > s.hardware.vram_mb:
                    # Not enough VRAM
                    s.services[svc_name] = ServiceState(
                        status=ServiceStatus.FAILED, ram_mb=0, vram_mb=0
                    )
                else:
                    # Successfully start
                    s.services[svc_name] = ServiceState(
                        status=ServiceStatus.RUNNING,
                        ram_mb=ram_cost,
                        vram_mb=vram_cost,
                        start_time_ms=s.time_ms,
                        ready_time_ms=s.time_ms + startup_ms,
                    )
                    s.resources.ram_used_mb += ram_cost
                    s.resources.vram_used_mb += vram_cost
                    s.time_ms += startup_ms

                if s.phase.value < BootPhase.BACKEND_START.value:
                    s.phase = BootPhase.BACKEND_START

            elif svc_name and svc_name in s.services:
                # Already exists — just advance time
                s.time_ms += 10

        elif action.verb == Verb.KILL:
            # Kill a service
            svc_name = self._target_to_service(action.target)
            if svc_name and svc_name in s.services:
                svc = s.services[svc_name]
                s.resources.ram_used_mb -= svc.ram_mb
                s.resources.vram_used_mb -= svc.vram_mb
                svc.status = ServiceStatus.STOPPED
                svc.ram_mb = 0
                svc.vram_mb = 0
                s.time_ms += 50

        elif action.verb == Verb.RESTART:
            # Restart a failed/stopped service
            svc_name = self._target_to_service(action.target)
            if svc_name and svc_name in s.services:
                svc = s.services[svc_name]
                if svc.status in (ServiceStatus.FAILED, ServiceStatus.STOPPED):
                    svc.restart_count += 1
                    spec = SERVICE_SPECS.get(svc_name, {})
                    ram_cost = spec.get("ram_mb", 256)
                    vram_cost = _service_vram(svc_name, s.hardware)
                    startup_ms = _service_startup_ms(svc_name)

                    if s.resources.ram_used_mb + ram_cost <= s.hardware.ram_mb:
                        svc.status = ServiceStatus.RUNNING
                        svc.ram_mb = ram_cost
                        svc.vram_mb = vram_cost
                        s.resources.ram_used_mb += ram_cost
                        s.resources.vram_used_mb += vram_cost
                        svc.ready_time_ms = s.time_ms + startup_ms
                        s.time_ms += startup_ms
                    else:
                        svc.status = ServiceStatus.FAILED
                        s.time_ms += 100

        elif action.verb == Verb.NETWORK_UP:
            # Attempt to bring network up
            s.time_ms += 300
            if self.rng.random() < self.network_success_rate:
                s.network = NetworkState.UP
            else:
                s.network = NetworkState.FAILED

        elif action.verb == Verb.OBSERVE:
            # Passive observation — just costs time
            s.time_ms += 20

        elif action.verb == Verb.DENY:
            # Explicit denial — model refuses to act (safe, costs minimal time)
            s.time_ms += 5

        elif action.verb == Verb.ESCALATE:
            # Request help / signal issue
            s.time_ms += 50

        elif action.verb == Verb.NOOP:
            # Do nothing
            s.time_ms += 10

        # Update CPU utilization based on running services
        running = sum(1 for svc in s.services.values() if svc.status == ServiceStatus.RUNNING)
        s.resources.cpu_pct = min(0.05 + running * 0.15, 1.0)

        # Phase progression
        if s.phase == BootPhase.BACKEND_START:
            # Check if required services are up
            inference_up = s.services.get("inference", ServiceState()).status == ServiceStatus.RUNNING
            api_up = s.services.get("api", ServiceState()).status == ServiceStatus.RUNNING
            if inference_up and api_up:
                s.phase = BootPhase.SERVICES

        if s.phase == BootPhase.SERVICES:
            # Check if all services healthy
            all_running = all(
                svc.status == ServiceStatus.RUNNING
                for name, svc in s.services.items()
                if name in ("inference", "api")
            )
            if all_running and s.services.get("inference") and s.services.get("api"):
                s.phase = BootPhase.READY

    def _compute_reward(self, prev: BootState, action: Action, curr: BootState) -> float:
        """Multi-objective reward shaping."""
        reward = 0.0

        # Boot progress (phase advancement)
        if curr.phase > prev.phase:
            reward += 2.0 * (curr.phase - prev.phase)

        # Speed bonus (less time = more reward)
        time_delta = curr.time_ms - prev.time_ms
        reward -= time_delta / 5000.0  # -0.2 per second

        # Hardware detection bonus
        if curr.hw_detected and not prev.hw_detected:
            reward += 1.0

        # Configuration quality (if config matches hardware)
        if action.verb == Verb.CONFIGURE:
            reward += self._config_quality_reward(action.params, curr.hardware)

        # Service successfully started
        for name, svc in curr.services.items():
            prev_svc = prev.services.get(name)
            if svc.status == ServiceStatus.RUNNING and (not prev_svc or prev_svc.status != ServiceStatus.RUNNING):
                reward += 3.0

        # Service failed (penalty)
        for name, svc in curr.services.items():
            prev_svc = prev.services.get(name)
            if svc.status == ServiceStatus.FAILED and (not prev_svc or prev_svc.status != ServiceStatus.FAILED):
                reward -= 2.0

        # Boot complete bonus
        if curr.phase == BootPhase.READY and prev.phase != BootPhase.READY:
            # Bonus scaled by speed
            speed_bonus = max(0, (self.max_time_ms - curr.time_ms) / self.max_time_ms) * 10.0
            reward += 10.0 + speed_bonus

        # NOOP penalty (discourages stalling)
        if action.verb == Verb.NOOP:
            reward -= 0.3

        # Redundant action penalty
        if action.verb == Verb.DETECT and curr.hw_detected and prev.hw_detected:
            reward -= 0.5

        return reward

    def _config_quality_reward(self, params: dict, hw: HardwareProfile) -> float:
        """Reward for configuration that matches hardware well."""
        reward = 0.0
        if "threads" in params:
            optimal = hw.cores + 1 if hw.cores <= 8 else hw.cores
            diff = abs(params["threads"] - optimal) / optimal
            reward += max(0, 1.0 - diff)  # 0-1 based on closeness

        if "gpu_layers" in params:
            if hw.vram_mb == 0:
                # No GPU — gpu_layers should be 0
                reward += 1.0 if params["gpu_layers"] == 0 else -0.5
            else:
                # Has GPU — more layers = better (up to a point)
                reward += min(params["gpu_layers"] / 50.0, 1.0)

        return reward * 0.5  # Scale down config reward

    def _check_terminal(self) -> tuple[bool, dict]:
        """Check if episode should end."""
        s = self.state

        # Success
        if s.phase == BootPhase.READY:
            return True, {"reason": "success", "time_ms": s.time_ms}

        # Timeout
        if s.time_ms >= self.max_time_ms:
            return True, {"reason": "timeout", "time_ms": s.time_ms}

        # Max steps exceeded
        if self.step_count >= self.max_steps:
            return True, {"reason": "timeout", "steps": self.step_count}

        # Unrecoverable crash (all required services failed with max retries)
        inference = s.services.get("inference")
        if inference and inference.status == ServiceStatus.FAILED and inference.restart_count >= 3:
            return True, {"reason": "crash", "service": "inference"}

        return False, {}

    def _target_to_service(self, target: Target) -> Optional[str]:
        """Map action target to service name."""
        mapping = {
            Target.INFERENCE: "inference",
            Target.API: "api",
            Target.LOGGER: "logger",
            Target.NETWORK: "network_svc",
        }
        return mapping.get(target)

    def _generate_persistence(self) -> dict[str, str]:
        """Generate persistence state (simulates prior boots)."""
        if self.rng.random() < 0.3:
            return {}  # Fresh boot, no cache
        # Simulate cached config from prior boot
        return {
            "hw_fingerprint": self.hardware.fingerprint(),
            "threads": str(self.hardware.cores),
            "gpu_layers": str(999 if self.hardware.vram_mb > 4096 else 0),
            "ctx_size": "4096",
            "boot_count": str(self.rng.randint(1, 100)),
        }

    def run_episode(self, policy_fn) -> EpisodeResult:
        """Run a complete episode with a policy function.

        Args:
            policy_fn: Callable(BootState) → Action

        Returns:
            EpisodeResult with outcome, reward, and transition data.
        """
        state = self.reset()
        total_reward = 0.0
        transitions = []
        done = False

        while not done:
            state_vec = state.to_vector()
            action = policy_fn(state)
            next_state, reward, done, info = self.step(action)
            next_state_vec = next_state.to_vector()
            action_vec = action.to_vector()
            transitions.append((state_vec, action_vec, reward, next_state_vec))
            total_reward += reward
            state = next_state

        outcome = info.get("reason", "unknown")
        return EpisodeResult(
            outcome=outcome,
            total_reward=total_reward,
            steps=self.step_count,
            time_ms=self.state.time_ms,
            transitions=transitions,
            final_state=self.state,
        )


# ---------------------------------------------------------------------------
# Expert policy (for imitation learning data generation)
# ---------------------------------------------------------------------------

def expert_policy(state: BootState) -> Action:
    """
    Hand-coded expert policy for generating imitation learning data.

    This encodes the "correct" boot sequence:
      1. Detect hardware (or read cache)
      2. Configure based on hardware
      3. Start inference backend
      4. Start API server
      5. Optionally bring up network
      6. Write cache for next boot
    """
    # Phase: INIT → detect or read cache
    if state.phase == BootPhase.INIT:
        if state.persistence and "hw_fingerprint" in state.persistence:
            # Have cache — read it
            if state.persistence.get("hw_fingerprint") == state.hardware.fingerprint():
                return Action(Verb.READ_CACHE, Target.PERSISTENCE)
        # No cache or fingerprint mismatch — detect
        return Action(Verb.DETECT, Target.HARDWARE)

    # Phase: HW_DETECT → configure
    if state.phase == BootPhase.HW_DETECT or (state.hw_detected and not state.config):
        hw = state.hardware
        threads = hw.cores + 1 if hw.cores <= 8 else hw.cores
        gpu_layers = 999 if hw.vram_mb > 4096 else (32 if hw.vram_mb > 2048 else 0)
        ctx_size = 8192 if hw.ram_mb > 16384 else 4096
        return Action(Verb.CONFIGURE, Target.CONFIG, {
            "threads": threads,
            "gpu_layers": gpu_layers,
            "ctx_size": ctx_size,
            "port": 11411,
        })

    # Phase: CONFIG → start inference
    if state.phase == BootPhase.CONFIG and "inference" not in state.services:
        return Action(Verb.SPAWN, Target.INFERENCE)

    # Inference started → start API
    inference = state.services.get("inference")
    if inference and inference.status == ServiceStatus.RUNNING and "api" not in state.services:
        return Action(Verb.SPAWN, Target.API)

    # Handle failed services
    for name in ["inference", "api"]:
        svc = state.services.get(name)
        if svc and svc.status == ServiceStatus.FAILED and svc.restart_count < 3:
            target = Target.INFERENCE if name == "inference" else Target.API
            return Action(Verb.RESTART, target)

    # Services running → write cache and bring up network
    if state.phase == BootPhase.SERVICES or state.phase == BootPhase.READY:
        if "hw_fingerprint" not in state.persistence or state.persistence.get("hw_fingerprint") != state.hardware.fingerprint():
            return Action(Verb.WRITE_CACHE, Target.PERSISTENCE)
        if state.network == NetworkState.DOWN:
            return Action(Verb.NETWORK_UP, Target.NETWORK)

    # Default: observe
    return Action(Verb.OBSERVE, Target.SYSTEM)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(n_episodes: int = 1000, verbose: bool = False) -> dict:
    """Run random episodes and validate simulator physics."""
    outcomes = {"success": 0, "timeout": 0, "violation": 0, "crash": 0, "unknown": 0}
    total_rewards = []
    total_times = []
    total_steps = []

    for i in range(n_episodes):
        hw = random_hardware()
        # 30% of episodes have faults
        faults = []
        if random.random() < 0.3:
            fault_fn = random.choice(FAULT_LIBRARY)
            faults = [fault_fn()]

        sim = CortexBootSimulator(hardware=hw, faults=faults, seed=i)
        result = sim.run_episode(expert_policy)

        outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1
        total_rewards.append(result.total_reward)
        total_times.append(result.time_ms)
        total_steps.append(result.steps)

    stats = {
        "episodes": n_episodes,
        "outcomes": outcomes,
        "success_rate": outcomes["success"] / n_episodes,
        "avg_reward": sum(total_rewards) / len(total_rewards),
        "avg_time_ms": sum(total_times) / len(total_times),
        "avg_steps": sum(total_steps) / len(total_steps),
        "max_reward": max(total_rewards),
        "min_reward": min(total_rewards),
    }

    if verbose:
        print(f"Simulator Validation ({n_episodes} episodes)")
        print(f"  Outcomes: {outcomes}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Avg reward: {stats['avg_reward']:.2f}")
        print(f"  Avg time: {stats['avg_time_ms']:.0f}ms")
        print(f"  Avg steps: {stats['avg_steps']:.1f}")

    return stats


def generate_expert_episodes(n: int = 1000, include_faults: bool = True) -> list[EpisodeResult]:
    """Generate expert demonstration episodes for imitation learning."""
    episodes = []
    for i in range(n):
        hw = random_hardware()
        faults = []
        if include_faults and random.random() < 0.2:
            fault_fn = random.choice(FAULT_LIBRARY)
            faults = [fault_fn()]

        sim = CortexBootSimulator(hardware=hw, faults=faults, seed=i)
        result = sim.run_episode(expert_policy)
        episodes.append(result)

    return episodes


if __name__ == "__main__":
    stats = validate(n_episodes=1000, verbose=True)
