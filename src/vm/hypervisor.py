"""
Cortex Hypervisor Abstraction — unified VM lifecycle across platforms.

The AI doesn't care whether it's using Apple's Virtualization.framework,
libvirt, Hyper-V, or containers. It just says "give me a machine" and
this layer figures out the how.

Architecture:
  Cortex → VMManager → HypervisorBackend (platform-specific)
                     → VMInstance (lifecycle: create → boot → run → snapshot → destroy)
                     → VMPool (reusable warm instances)
"""

import logging
import os
import platform
import subprocess
import time
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger("cortex.vm")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class VMState(Enum):
    """VM lifecycle states."""
    CREATING = "creating"
    BOOTING = "booting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DESTROYED = "destroyed"
    ERROR = "error"


class VMPurpose(Enum):
    """Why this VM exists."""
    SANDBOX = "sandbox"         # Disposable code execution
    CLONE = "clone"             # Self-replication for testing
    TENANT = "tenant"           # Isolated Cortex instance
    PROVISION = "provision"     # Full OS being managed


class HypervisorType(Enum):
    """Available hypervisor backends."""
    APPLE_VZ = "apple_vz"       # macOS Virtualization.framework
    LIBVIRT = "libvirt"         # Linux KVM/QEMU
    HYPERV = "hyperv"           # Windows Hyper-V
    DOCKER = "docker"           # Container fallback (not true VM)
    PODMAN = "podman"           # Rootless container fallback


@dataclass
class VMSpec:
    """Specification for a VM to create."""
    name: str = ""
    purpose: VMPurpose = VMPurpose.SANDBOX
    cpus: int = 2
    memory_mb: int = 2048
    disk_gb: int = 10
    image: str = ""             # OS image path or container image
    network: bool = True
    gpu_passthrough: bool = False
    mount_cortex: bool = False  # Mount the CORTEX drive inside the VM
    startup_script: str = ""    # Script to run after boot
    timeout_s: int = 300        # Max lifetime for sandboxes
    snapshot_on_create: bool = True  # Snapshot base state for fast reset


@dataclass
class VMInstance:
    """A running or stopped VM."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    spec: VMSpec = field(default_factory=VMSpec)
    state: VMState = VMState.CREATING
    hypervisor: HypervisorType = HypervisorType.DOCKER
    ip_address: str = ""
    ssh_port: int = 0
    cortex_port: int = 0       # If running a Cortex instance inside
    pid: int = 0
    created_at: float = field(default_factory=time.time)
    boot_time_s: float = 0.0
    error: str = ""

    @property
    def is_alive(self) -> bool:
        return self.state in (VMState.RUNNING, VMState.BOOTING, VMState.PAUSED)

    @property
    def age_s(self) -> float:
        return time.time() - self.created_at


# ---------------------------------------------------------------------------
# Hypervisor backend interface
# ---------------------------------------------------------------------------

class HypervisorBackend(ABC):
    """Abstract backend — each platform implements this."""

    @abstractmethod
    def available(self) -> bool:
        """Check if this hypervisor is usable on the current system."""
        ...

    @abstractmethod
    def create(self, spec: VMSpec) -> VMInstance:
        """Create a new VM from spec."""
        ...

    @abstractmethod
    def start(self, vm: VMInstance) -> VMInstance:
        """Boot a stopped VM."""
        ...

    @abstractmethod
    def stop(self, vm: VMInstance, force: bool = False) -> VMInstance:
        """Stop a running VM."""
        ...

    @abstractmethod
    def destroy(self, vm: VMInstance) -> None:
        """Permanently delete a VM and its disk."""
        ...

    @abstractmethod
    def exec(self, vm: VMInstance, command: str, timeout: int = 30) -> tuple[int, str]:
        """Execute a command inside the VM. Returns (exit_code, output)."""
        ...

    @abstractmethod
    def snapshot(self, vm: VMInstance, name: str) -> str:
        """Create a snapshot. Returns snapshot ID."""
        ...

    @abstractmethod
    def restore(self, vm: VMInstance, snapshot_id: str) -> VMInstance:
        """Restore VM to a snapshot."""
        ...

    @abstractmethod
    def list_vms(self) -> list[VMInstance]:
        """List all VMs managed by this backend."""
        ...


# ---------------------------------------------------------------------------
# Docker/Podman backend (works everywhere, immediate MVP)
# ---------------------------------------------------------------------------

class ContainerBackend(HypervisorBackend):
    """
    Container-based backend using Docker or Podman.

    Not a true hypervisor but provides:
    - Sandboxed execution (isolated filesystem, network, PIDs)
    - Fast create/destroy (~1s)
    - Works on macOS, Linux, Windows
    - Rootless with Podman
    """

    def __init__(self, runtime: str = "auto"):
        if runtime == "auto":
            self.runtime = self._detect_runtime()
        else:
            self.runtime = runtime
        self._instances: dict[str, VMInstance] = {}

    def _detect_runtime(self) -> str:
        """Find docker or podman."""
        for rt in ["podman", "docker"]:
            try:
                subprocess.run([rt, "version"], capture_output=True, timeout=5)
                return rt
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return ""

    def available(self) -> bool:
        return bool(self.runtime)

    def create(self, spec: VMSpec) -> VMInstance:
        image = spec.image or "ubuntu:22.04"
        name = spec.name or f"cortex-{spec.purpose.value}-{uuid.uuid4().hex[:6]}"

        cmd = [
            self.runtime, "run", "-d",
            "--name", name,
            "--memory", f"{spec.memory_mb}m",
            "--cpus", str(spec.cpus),
        ]

        if spec.network:
            # Expose SSH and optionally Cortex port
            ssh_port = self._find_free_port(2200, 2300)
            cmd += ["-p", f"{ssh_port}:22"]
            if spec.mount_cortex:
                cortex_port = self._find_free_port(11500, 11600)
                cmd += ["-p", f"{cortex_port}:11411"]
        else:
            cmd += ["--network", "none"]
            ssh_port = 0
            cortex_port = 0

        if spec.mount_cortex:
            # Mount the CORTEX drive read-only inside the container
            cortex_path = os.environ.get("CORTEX_HOME", "/Volumes/CORTEX/cortex")
            cmd += ["-v", f"{cortex_path}:/cortex:ro"]

        if spec.timeout_s > 0 and spec.purpose == VMPurpose.SANDBOX:
            # Auto-kill sandbox after timeout
            cmd += ["--stop-timeout", str(spec.timeout_s)]

        # Keep container alive
        cmd += [image, "sleep", "infinity"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                vm = VMInstance(name=name, spec=spec, state=VMState.ERROR,
                               error=result.stderr.strip())
                return vm

            container_id = result.stdout.strip()[:12]
            vm = VMInstance(
                id=container_id,
                name=name,
                spec=spec,
                state=VMState.RUNNING,
                hypervisor=HypervisorType.DOCKER if "docker" in self.runtime else HypervisorType.PODMAN,
                ssh_port=ssh_port,
                cortex_port=cortex_port,
            )

            # Run startup script if provided
            if spec.startup_script:
                self.exec(vm, spec.startup_script, timeout=60)

            self._instances[vm.id] = vm
            vm.boot_time_s = time.time() - vm.created_at
            logger.info("Created %s container: %s (boot=%.1fs)", spec.purpose.value, name, vm.boot_time_s)
            return vm

        except subprocess.TimeoutExpired:
            return VMInstance(name=name, spec=spec, state=VMState.ERROR,
                             error="Container creation timed out")

    def start(self, vm: VMInstance) -> VMInstance:
        subprocess.run([self.runtime, "start", vm.name], capture_output=True, timeout=10)
        vm.state = VMState.RUNNING
        return vm

    def stop(self, vm: VMInstance, force: bool = False) -> VMInstance:
        cmd = [self.runtime, "kill" if force else "stop", vm.name]
        subprocess.run(cmd, capture_output=True, timeout=15)
        vm.state = VMState.STOPPED
        return vm

    def destroy(self, vm: VMInstance) -> None:
        subprocess.run([self.runtime, "rm", "-f", vm.name], capture_output=True, timeout=10)
        vm.state = VMState.DESTROYED
        self._instances.pop(vm.id, None)
        logger.info("Destroyed container: %s", vm.name)

    def exec(self, vm: VMInstance, command: str, timeout: int = 30) -> tuple[int, str]:
        result = subprocess.run(
            [self.runtime, "exec", vm.name, "bash", "-c", command],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr

    def snapshot(self, vm: VMInstance, name: str) -> str:
        """Commit container state as a new image."""
        snap_tag = f"cortex-snap:{vm.name}-{name}"
        subprocess.run(
            [self.runtime, "commit", vm.name, snap_tag],
            capture_output=True, timeout=60,
        )
        return snap_tag

    def restore(self, vm: VMInstance, snapshot_id: str) -> VMInstance:
        """Destroy current, recreate from snapshot image."""
        self.destroy(vm)
        new_spec = VMSpec(**{**vm.spec.__dict__, "image": snapshot_id})
        return self.create(new_spec)

    def list_vms(self) -> list[VMInstance]:
        return list(self._instances.values())

    def _find_free_port(self, start: int, end: int) -> int:
        """Find a free port in range."""
        import socket
        for port in range(start, end):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                    return port
            except OSError:
                continue
        return start


# ---------------------------------------------------------------------------
# Apple Virtualization.framework backend (macOS native)
# ---------------------------------------------------------------------------

class AppleVZBackend(HypervisorBackend):
    """
    Native macOS hypervisor using Virtualization.framework.

    Requires macOS 12+ and Apple Silicon or Intel with VT-x.
    Uses `tart` CLI (open source, Virtualization.framework wrapper) or
    direct Swift bridge (future).

    TODO: Implement when testing on macOS
    - tart: https://github.com/cirruslabs/tart (brew install tart)
    - Supports Linux VMs on Apple Silicon natively
    - Fast boot (~3s to login prompt)
    """

    def available(self) -> bool:
        if platform.system() != "Darwin":
            return False
        # Check for tart CLI
        try:
            subprocess.run(["tart", "--version"], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def create(self, spec: VMSpec) -> VMInstance:
        raise NotImplementedError("Apple VZ backend TODO — install tart: brew install tart")

    def start(self, vm: VMInstance) -> VMInstance:
        raise NotImplementedError

    def stop(self, vm: VMInstance, force: bool = False) -> VMInstance:
        raise NotImplementedError

    def destroy(self, vm: VMInstance) -> None:
        raise NotImplementedError

    def exec(self, vm: VMInstance, command: str, timeout: int = 30) -> tuple[int, str]:
        raise NotImplementedError

    def snapshot(self, vm: VMInstance, name: str) -> str:
        raise NotImplementedError

    def restore(self, vm: VMInstance, snapshot_id: str) -> VMInstance:
        raise NotImplementedError

    def list_vms(self) -> list[VMInstance]:
        return []


# ---------------------------------------------------------------------------
# Libvirt/QEMU backend (Linux)
# ---------------------------------------------------------------------------

class LibvirtBackend(HypervisorBackend):
    """
    Linux KVM/QEMU via libvirt.

    TODO: Implement when testing on Linux workstation
    - virsh/virt-install for lifecycle
    - GPU passthrough for CUDA models inside VMs
    - Cloud-init for automated provisioning
    """

    def available(self) -> bool:
        if platform.system() != "Linux":
            return False
        try:
            subprocess.run(["virsh", "version"], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def create(self, spec: VMSpec) -> VMInstance:
        raise NotImplementedError("Libvirt backend TODO — test on Linux workstation")

    def start(self, vm: VMInstance) -> VMInstance:
        raise NotImplementedError

    def stop(self, vm: VMInstance, force: bool = False) -> VMInstance:
        raise NotImplementedError

    def destroy(self, vm: VMInstance) -> None:
        raise NotImplementedError

    def exec(self, vm: VMInstance, command: str, timeout: int = 30) -> tuple[int, str]:
        raise NotImplementedError

    def snapshot(self, vm: VMInstance, name: str) -> str:
        raise NotImplementedError

    def restore(self, vm: VMInstance, snapshot_id: str) -> VMInstance:
        raise NotImplementedError

    def list_vms(self) -> list[VMInstance]:
        return []


# ---------------------------------------------------------------------------
# VM Manager — the brain that decides what to spin up
# ---------------------------------------------------------------------------

class VMManager:
    """
    Cortex VM Manager — orchestrates VM lifecycle and purpose.

    Capabilities:
      - sandbox(command) → run code in disposable container, return output
      - clone() → boot a Cortex replica for testing changes
      - provision(os, purpose) → create a full VM with an OS
      - tenant(user_id) → isolated Cortex instance for a user
    """

    def __init__(self, cortex_home: Optional[str] = None):
        self.cortex_home = cortex_home or os.environ.get("CORTEX_HOME", "/Volumes/CORTEX/cortex")
        self.backends: list[HypervisorBackend] = []
        self._active_vms: dict[str, VMInstance] = {}
        self._discover_backends()

    def _discover_backends(self):
        """Find available hypervisor backends in priority order."""
        # Try native hypervisors first, fall back to containers
        candidates = [
            AppleVZBackend(),
            LibvirtBackend(),
            ContainerBackend(),
        ]
        for backend in candidates:
            if backend.available():
                self.backends.append(backend)
                logger.info("VM backend available: %s", type(backend).__name__)

    @property
    def primary(self) -> Optional[HypervisorBackend]:
        """Best available backend."""
        return self.backends[0] if self.backends else None

    @property
    def available(self) -> bool:
        return len(self.backends) > 0

    # ------------------------------------------------------------------
    # High-level operations
    # ------------------------------------------------------------------

    def sandbox(
        self,
        command: str,
        image: str = "python:3.12-slim",
        timeout: int = 60,
        memory_mb: int = 512,
    ) -> tuple[int, str]:
        """
        Execute a command in a disposable sandbox.

        Creates a container, runs the command, captures output, destroys it.
        Maximum isolation for untrusted code.
        """
        if not self.primary:
            return -1, "No VM backend available"

        spec = VMSpec(
            purpose=VMPurpose.SANDBOX,
            image=image,
            memory_mb=memory_mb,
            cpus=1,
            network=False,  # No network for sandboxes by default
            timeout_s=timeout,
        )

        vm = self.primary.create(spec)
        if vm.state == VMState.ERROR:
            return -1, f"Failed to create sandbox: {vm.error}"

        try:
            exit_code, output = self.primary.exec(vm, command, timeout=timeout)
            return exit_code, output
        finally:
            self.primary.destroy(vm)

    def clone(
        self,
        test_script: Optional[str] = None,
        branch: str = "main",
    ) -> VMInstance:
        """
        Boot a Cortex clone for self-testing.

        The clone gets a read-only mount of the current CORTEX drive.
        It can run tests, validate changes, and report back.
        """
        if not self.primary:
            raise RuntimeError("No VM backend available")

        startup = f"""
apt-get update -qq && apt-get install -y -qq python3 python3-pip curl > /dev/null 2>&1
cd /cortex && pip install -q -r requirements.txt 2>/dev/null || true
"""
        if test_script:
            startup += f"\n{test_script}"

        spec = VMSpec(
            name=f"cortex-clone-{uuid.uuid4().hex[:6]}",
            purpose=VMPurpose.CLONE,
            image="python:3.12-slim",
            memory_mb=2048,
            cpus=2,
            mount_cortex=True,
            startup_script=startup,
            timeout_s=600,
        )

        vm = self.primary.create(spec)
        self._active_vms[vm.id] = vm
        return vm

    def provision(
        self,
        image: str = "ubuntu:22.04",
        purpose: str = "general",
        cpus: int = 2,
        memory_mb: int = 4096,
        disk_gb: int = 20,
        startup_script: str = "",
    ) -> VMInstance:
        """
        Provision a full OS — Cortex as infrastructure manager.

        The VM gets network access, persistent disk, and optionally
        a Cortex daemon running inside it.
        """
        if not self.primary:
            raise RuntimeError("No VM backend available")

        spec = VMSpec(
            name=f"cortex-vm-{purpose}-{uuid.uuid4().hex[:4]}",
            purpose=VMPurpose.PROVISION,
            image=image,
            cpus=cpus,
            memory_mb=memory_mb,
            disk_gb=disk_gb,
            network=True,
            mount_cortex=True,
            startup_script=startup_script,
            timeout_s=0,  # No auto-kill for provisioned VMs
        )

        vm = self.primary.create(spec)
        self._active_vms[vm.id] = vm
        return vm

    def tenant(
        self,
        user_id: str,
        memory_mb: int = 2048,
    ) -> VMInstance:
        """
        Create an isolated Cortex instance for a user/context.

        Each tenant gets their own:
        - Cortex daemon on a unique port
        - Isolated training data
        - Own model routing state
        """
        if not self.primary:
            raise RuntimeError("No VM backend available")

        startup = """
cd /cortex
python3 -m src daemon --port 11411 &
"""

        spec = VMSpec(
            name=f"cortex-tenant-{user_id[:8]}",
            purpose=VMPurpose.TENANT,
            image="python:3.12-slim",
            memory_mb=memory_mb,
            cpus=2,
            mount_cortex=True,
            network=True,
            startup_script=startup,
            timeout_s=0,
        )

        vm = self.primary.create(spec)
        self._active_vms[vm.id] = vm
        return vm

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def list_active(self) -> list[VMInstance]:
        """List all active VMs."""
        return [vm for vm in self._active_vms.values() if vm.is_alive]

    def destroy_all(self, purpose: Optional[VMPurpose] = None) -> int:
        """Destroy all VMs, optionally filtered by purpose."""
        count = 0
        for vm_id, vm in list(self._active_vms.items()):
            if purpose and vm.spec.purpose != purpose:
                continue
            if self.primary:
                self.primary.destroy(vm)
            del self._active_vms[vm_id]
            count += 1
        return count

    def cleanup_expired(self) -> int:
        """Destroy sandboxes that have exceeded their timeout."""
        count = 0
        for vm_id, vm in list(self._active_vms.items()):
            if vm.spec.timeout_s > 0 and vm.age_s > vm.spec.timeout_s:
                if self.primary:
                    self.primary.destroy(vm)
                del self._active_vms[vm_id]
                count += 1
        return count

    def status(self) -> dict:
        """Current VM infrastructure status."""
        return {
            "backends": [type(b).__name__ for b in self.backends],
            "primary": type(self.primary).__name__ if self.primary else None,
            "active_vms": len(self._active_vms),
            "by_purpose": {
                p.value: len([v for v in self._active_vms.values() if v.spec.purpose == p])
                for p in VMPurpose
            },
        }
