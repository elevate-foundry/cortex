"""
Cortex hardware detection module.

Detects:
  - OS and architecture
  - CPU capabilities (cores, model, AVX/VNNI support)
  - RAM (total, available)
  - GPU: NVIDIA (CUDA), AMD (ROCm), Apple Silicon (Metal/ANE), Intel Arc (OneAPI)
  - Installed inference backends (vLLM, llama.cpp, MLX, TensorRT-LLM, etc.)
  - Python version and key package versions
"""

import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class AcceleratorType(str, Enum):
    NVIDIA_CUDA = "nvidia_cuda"
    AMD_ROCM = "amd_rocm"
    APPLE_METAL = "apple_metal"
    INTEL_ARC = "intel_arc"
    CPU_ONLY = "cpu_only"


@dataclass
class GPUInfo:
    name: str
    accelerator: AcceleratorType
    vram_mb: int
    driver_version: str = ""
    compute_capability: str = ""  # NVIDIA-specific (e.g. "8.9")
    count: int = 1


@dataclass
class CPUInfo:
    model: str
    arch: str  # x86_64, arm64, etc.
    physical_cores: int
    logical_cores: int
    features: list[str] = field(default_factory=list)  # AVX2, VNNI, NEON, etc.


@dataclass
class MemoryInfo:
    total_mb: int
    available_mb: int


@dataclass
class BackendAvailability:
    name: str
    available: bool
    version: str = ""
    path: str = ""


@dataclass
class SystemProfile:
    os_name: str  # Linux, Darwin, Windows
    os_version: str
    arch: str
    cpu: CPUInfo
    memory: MemoryInfo
    gpus: list[GPUInfo] = field(default_factory=list)
    backends: list[BackendAvailability] = field(default_factory=list)
    python_version: str = ""
    hostname: str = ""

    @property
    def primary_accelerator(self) -> AcceleratorType:
        if self.gpus:
            return self.gpus[0].accelerator
        return AcceleratorType.CPU_ONLY

    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_mb for g in self.gpus)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["primary_accelerator"] = self.primary_accelerator.value
        d["total_vram_mb"] = self.total_vram_mb
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        lines = [
            f"=== System Profile ===",
            f"OS:       {self.os_name} {self.os_version} ({self.arch})",
            f"CPU:      {self.cpu.model} ({self.cpu.physical_cores}P/{self.cpu.logical_cores}L cores)",
            f"RAM:      {self.memory.total_mb:,} MB total, {self.memory.available_mb:,} MB available",
        ]
        if self.cpu.features:
            lines.append(f"CPU Feat: {', '.join(self.cpu.features)}")
        if self.gpus:
            for i, gpu in enumerate(self.gpus):
                lines.append(
                    f"GPU[{i}]:   {gpu.name} ({gpu.vram_mb:,} MB VRAM) [{gpu.accelerator.value}]"
                )
        else:
            lines.append(f"GPU:      None detected (CPU-only)")
        if self.backends:
            avail = [b for b in self.backends if b.available]
            if avail:
                lines.append(
                    f"Backends: {', '.join(f'{b.name} {b.version}' for b in avail)}"
                )
            else:
                lines.append("Backends: None installed")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 10) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def detect_os() -> tuple[str, str, str]:
    """Returns (os_name, os_version, architecture)."""
    os_name = platform.system()  # Linux, Darwin, Windows
    os_version = platform.release()
    arch = platform.machine()  # x86_64, arm64, aarch64
    return os_name, os_version, arch


def detect_cpu() -> CPUInfo:
    """Detect CPU model, cores, and ISA features."""
    arch = platform.machine()
    physical = os.cpu_count() or 1
    logical = physical
    model = platform.processor() or "Unknown"
    features: list[str] = []

    system = platform.system()

    if system == "Linux":
        cpuinfo = _run(["cat", "/proc/cpuinfo"])
        if cpuinfo:
            for line in cpuinfo.splitlines():
                if line.startswith("model name") and model in ("Unknown", ""):
                    model = line.split(":", 1)[1].strip()
                if line.startswith("flags"):
                    flags = line.split(":", 1)[1].strip().split()
                    for feat in ("avx", "avx2", "avx512f", "vnni", "amx_tile", "f16c", "fma"):
                        if feat in flags:
                            features.append(feat.upper())
                    break
        # Physical cores via lscpu
        lscpu = _run(["lscpu"])
        if lscpu:
            for line in lscpu.splitlines():
                if "Core(s) per socket" in line:
                    try:
                        cores_per = int(line.split(":")[1].strip())
                    except ValueError:
                        cores_per = 0
                if "Socket(s)" in line:
                    try:
                        sockets = int(line.split(":")[1].strip())
                    except ValueError:
                        sockets = 1
            try:
                physical = cores_per * sockets
            except NameError:
                pass
            logical = os.cpu_count() or physical

    elif system == "Darwin":
        # macOS: use sysctl
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            model = brand
        phys = _run(["sysctl", "-n", "hw.physicalcpu"])
        log = _run(["sysctl", "-n", "hw.logicalcpu"])
        if phys:
            physical = int(phys)
        if log:
            logical = int(log)
        # Check for Apple Silicon features
        if "arm64" in arch or "Apple" in model:
            features.extend(["NEON", "FP16", "METAL"])
        else:
            feat_str = _run(["sysctl", "-n", "machdep.cpu.features"])
            if feat_str:
                cpu_feats = feat_str.split()
                for feat in ("AVX", "AVX2", "AVX512", "FMA", "F16C"):
                    if feat in cpu_feats:
                        features.append(feat)

    elif system == "Windows":
        # Use wmic or platform
        wmic_out = _run(["wmic", "cpu", "get", "name", "/value"])
        if wmic_out:
            for line in wmic_out.splitlines():
                if line.startswith("Name="):
                    model = line.split("=", 1)[1].strip()
        phys = _run(
            ["powershell", "-Command",
             "(Get-CimInstance Win32_Processor).NumberOfCores"]
        )
        if phys:
            try:
                physical = int(phys.strip())
            except ValueError:
                pass
        logical = os.cpu_count() or physical

    return CPUInfo(
        model=model,
        arch=arch,
        physical_cores=physical,
        logical_cores=logical,
        features=features,
    )


def detect_memory() -> MemoryInfo:
    """Detect total and available system RAM."""
    system = platform.system()

    if system == "Linux":
        meminfo = _run(["cat", "/proc/meminfo"])
        total = avail = 0
        if meminfo:
            for line in meminfo.splitlines():
                if line.startswith("MemTotal"):
                    total = int(line.split()[1]) // 1024  # kB -> MB
                elif line.startswith("MemAvailable"):
                    avail = int(line.split()[1]) // 1024
        return MemoryInfo(total_mb=total, available_mb=avail)

    elif system == "Darwin":
        total_str = _run(["sysctl", "-n", "hw.memsize"])
        total = int(total_str) // (1024 * 1024) if total_str else 0
        # vm_stat for available
        vm = _run(["vm_stat"])
        avail = 0
        if vm:
            page_size = 4096  # default macOS page size
            free_pages = 0
            for line in vm.splitlines():
                if "page size of" in line:
                    try:
                        page_size = int(line.split()[-2])
                    except (ValueError, IndexError):
                        pass
                if "Pages free" in line:
                    try:
                        free_pages += int(line.split()[-1].rstrip("."))
                    except ValueError:
                        pass
                if "Pages inactive" in line:
                    try:
                        free_pages += int(line.split()[-1].rstrip("."))
                    except ValueError:
                        pass
            avail = (free_pages * page_size) // (1024 * 1024)
        return MemoryInfo(total_mb=total, available_mb=avail)

    elif system == "Windows":
        total_str = _run(
            ["powershell", "-Command",
             "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"]
        )
        avail_str = _run(
            ["powershell", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"]
        )
        total = int(total_str) // (1024 * 1024) if total_str else 0
        avail = int(avail_str) // 1024 if avail_str else 0  # FreePhysicalMemory is in KB
        return MemoryInfo(total_mb=total, available_mb=avail)

    return MemoryInfo(total_mb=0, available_mb=0)


def detect_nvidia_gpus() -> list[GPUInfo]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    gpus: list[GPUInfo] = []
    smi = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if not smi:
        return gpus
    for line in smi.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append(GPUInfo(
                name=parts[0],
                accelerator=AcceleratorType.NVIDIA_CUDA,
                vram_mb=int(float(parts[1])),
                driver_version=parts[2],
                compute_capability=parts[3],
            ))
    # Set count
    if gpus:
        for g in gpus:
            g.count = 1
    return gpus


def detect_amd_gpus() -> list[GPUInfo]:
    """Detect AMD GPUs via rocm-smi."""
    gpus: list[GPUInfo] = []
    rocm = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not rocm:
        return gpus
    try:
        data = json.loads(rocm)
        for card_id, card_data in data.items():
            if not isinstance(card_data, dict):
                continue
            name = card_data.get("Card series", card_data.get("Card Model", "AMD GPU"))
            vram = 0
            for k, v in card_data.items():
                if "Total" in k and "vram" in k.lower():
                    try:
                        vram = int(v) // (1024 * 1024)  # bytes -> MB
                    except (ValueError, TypeError):
                        pass
            gpus.append(GPUInfo(
                name=name,
                accelerator=AcceleratorType.AMD_ROCM,
                vram_mb=vram,
            ))
    except (json.JSONDecodeError, AttributeError):
        pass
    return gpus


def detect_apple_silicon() -> list[GPUInfo]:
    """Detect Apple Silicon GPU (unified memory)."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []

    # Apple Silicon uses unified memory — GPU can access all RAM
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    total_mem_str = _run(["sysctl", "-n", "hw.memsize"])
    total_mem_mb = int(total_mem_str) // (1024 * 1024) if total_mem_str else 0

    chip_name = brand if brand else "Apple Silicon"
    # Estimate GPU-available memory (typically ~75% of unified memory)
    gpu_mem = int(total_mem_mb * 0.75)

    # Use the CPU brand string as GPU name (avoids slow system_profiler call)
    gpu_name = chip_name

    return [GPUInfo(
        name=gpu_name,
        accelerator=AcceleratorType.APPLE_METAL,
        vram_mb=gpu_mem,
    )]


def detect_gpus() -> list[GPUInfo]:
    """Detect all available GPUs across vendors."""
    gpus: list[GPUInfo] = []

    # Check NVIDIA first (most common for inference)
    nvidia = detect_nvidia_gpus()
    if nvidia:
        gpus.extend(nvidia)

    # Check AMD ROCm
    amd = detect_amd_gpus()
    if amd:
        gpus.extend(amd)

    # Check Apple Silicon
    apple = detect_apple_silicon()
    if apple:
        gpus.extend(apple)

    return gpus


def detect_backends() -> list[BackendAvailability]:
    """Check which inference backends are installed and available."""
    backends: list[BackendAvailability] = []

    # Python packages to check
    python_backends = [
        ("vllm", "vLLM"),
        ("tensorrt_llm", "TensorRT-LLM"),
        ("mlx_lm", "MLX-LM"),
        ("mlx", "MLX"),
        ("ctransformers", "CTransformers"),
        ("exllamav2", "ExLlamaV2"),
    ]

    for pkg, name in python_backends:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            backends.append(BackendAvailability(
                name=name,
                available=True,
                version=version,
            ))
        except ImportError:
            backends.append(BackendAvailability(name=name, available=False))

    # Check llama.cpp (binary)
    for binary_name in ("llama-server", "llama-cli", "llama.cpp"):
        path = shutil.which(binary_name)
        if path:
            version = _run([path, "--version"]) or "unknown"
            backends.append(BackendAvailability(
                name="llama.cpp",
                available=True,
                version=version.split("\n")[0][:50],
                path=path,
            ))
            break
    else:
        backends.append(BackendAvailability(name="llama.cpp", available=False))

    # Check ollama
    ollama_path = shutil.which("ollama")
    if ollama_path:
        version = _run(["ollama", "--version"]) or "unknown"
        backends.append(BackendAvailability(
            name="Ollama",
            available=True,
            version=version.split("\n")[0][:50],
            path=ollama_path,
        ))
    else:
        backends.append(BackendAvailability(name="Ollama", available=False))

    return backends


def detect_system() -> SystemProfile:
    """Run full system detection and return a SystemProfile."""
    os_name, os_version, arch = detect_os()
    cpu = detect_cpu()
    memory = detect_memory()
    gpus = detect_gpus()
    backends = detect_backends()

    return SystemProfile(
        os_name=os_name,
        os_version=os_version,
        arch=arch,
        cpu=cpu,
        memory=memory,
        gpus=gpus,
        backends=backends,
        python_version=platform.python_version(),
        hostname=platform.node(),
    )


# ---------------------------------------------------------------------------
# Simulated profiles for cross-platform testing
# ---------------------------------------------------------------------------

SIMULATED_PROFILES: dict[str, SystemProfile] = {
    "linux-4090": SystemProfile(
        os_name="Linux",
        os_version="6.8.0-45-generic",
        arch="x86_64",
        cpu=CPUInfo(
            model="AMD Ryzen 9 7950X",
            arch="x86_64",
            physical_cores=16,
            logical_cores=32,
            features=["AVX", "AVX2", "AVX512", "FMA", "F16C"],
        ),
        memory=MemoryInfo(total_mb=65536, available_mb=58000),
        gpus=[GPUInfo(
            name="NVIDIA GeForce RTX 4090",
            accelerator=AcceleratorType.NVIDIA_CUDA,
            vram_mb=24576,
            driver_version="550.90.07",
            compute_capability="8.9",
        )],
        backends=[
            BackendAvailability("vLLM", True, "0.6.4"),
            BackendAvailability("llama.cpp", True, "b4321"),
            BackendAvailability("Ollama", True, "0.5.1"),
            BackendAvailability("TensorRT-LLM", False),
            BackendAvailability("MLX-LM", False),
            BackendAvailability("MLX", False),
            BackendAvailability("CTransformers", False),
            BackendAvailability("ExLlamaV2", True, "0.2.1"),
        ],
        python_version="3.12.4",
        hostname="gpu-workstation",
    ),

    "linux-2x4090": SystemProfile(
        os_name="Linux",
        os_version="6.8.0-45-generic",
        arch="x86_64",
        cpu=CPUInfo(
            model="AMD EPYC 9474F",
            arch="x86_64",
            physical_cores=48,
            logical_cores=96,
            features=["AVX", "AVX2", "AVX512", "FMA", "F16C", "VNNI"],
        ),
        memory=MemoryInfo(total_mb=131072, available_mb=120000),
        gpus=[
            GPUInfo(
                name="NVIDIA GeForce RTX 4090",
                accelerator=AcceleratorType.NVIDIA_CUDA,
                vram_mb=24576,
                driver_version="550.90.07",
                compute_capability="8.9",
            ),
            GPUInfo(
                name="NVIDIA GeForce RTX 4090",
                accelerator=AcceleratorType.NVIDIA_CUDA,
                vram_mb=24576,
                driver_version="550.90.07",
                compute_capability="8.9",
            ),
        ],
        backends=[
            BackendAvailability("vLLM", True, "0.6.4"),
            BackendAvailability("llama.cpp", True, "b4321"),
            BackendAvailability("Ollama", True, "0.5.1"),
            BackendAvailability("TensorRT-LLM", False),
            BackendAvailability("MLX-LM", False),
            BackendAvailability("MLX", False),
            BackendAvailability("CTransformers", False),
            BackendAvailability("ExLlamaV2", True, "0.2.1"),
        ],
        python_version="3.12.4",
        hostname="dual-gpu-server",
    ),

    "linux-h100": SystemProfile(
        os_name="Linux",
        os_version="6.5.0-cloud",
        arch="x86_64",
        cpu=CPUInfo(
            model="Intel Xeon w9-3495X",
            arch="x86_64",
            physical_cores=56,
            logical_cores=112,
            features=["AVX", "AVX2", "AVX512", "VNNI", "AMX_TILE"],
        ),
        memory=MemoryInfo(total_mb=262144, available_mb=240000),
        gpus=[GPUInfo(
            name="NVIDIA H100 80GB HBM3",
            accelerator=AcceleratorType.NVIDIA_CUDA,
            vram_mb=81920,
            driver_version="550.90.07",
            compute_capability="9.0",
        )],
        backends=[
            BackendAvailability("vLLM", True, "0.6.4"),
            BackendAvailability("TensorRT-LLM", True, "0.15.0"),
            BackendAvailability("llama.cpp", True, "b4321"),
            BackendAvailability("Ollama", False),
            BackendAvailability("MLX-LM", False),
            BackendAvailability("MLX", False),
            BackendAvailability("CTransformers", False),
            BackendAvailability("ExLlamaV2", True, "0.2.1"),
        ],
        python_version="3.11.9",
        hostname="cloud-h100",
    ),

    "linux-cpu": SystemProfile(
        os_name="Linux",
        os_version="6.1.0-22-amd64",
        arch="x86_64",
        cpu=CPUInfo(
            model="Intel Core i7-12700",
            arch="x86_64",
            physical_cores=12,
            logical_cores=20,
            features=["AVX", "AVX2", "FMA", "F16C"],
        ),
        memory=MemoryInfo(total_mb=32768, available_mb=28000),
        gpus=[],
        backends=[
            BackendAvailability("vLLM", False),
            BackendAvailability("TensorRT-LLM", False),
            BackendAvailability("llama.cpp", True, "b4321"),
            BackendAvailability("Ollama", True, "0.5.1"),
            BackendAvailability("MLX-LM", False),
            BackendAvailability("MLX", False),
            BackendAvailability("CTransformers", False),
            BackendAvailability("ExLlamaV2", False),
        ],
        python_version="3.11.2",
        hostname="dev-server",
    ),

    "linux-amd-7900xtx": SystemProfile(
        os_name="Linux",
        os_version="6.8.0-45-generic",
        arch="x86_64",
        cpu=CPUInfo(
            model="AMD Ryzen 9 7900X",
            arch="x86_64",
            physical_cores=12,
            logical_cores=24,
            features=["AVX", "AVX2", "AVX512", "FMA"],
        ),
        memory=MemoryInfo(total_mb=65536, available_mb=58000),
        gpus=[GPUInfo(
            name="AMD Radeon RX 7900 XTX",
            accelerator=AcceleratorType.AMD_ROCM,
            vram_mb=24576,
        )],
        backends=[
            BackendAvailability("vLLM", True, "0.6.4"),
            BackendAvailability("TensorRT-LLM", False),
            BackendAvailability("llama.cpp", True, "b4321"),
            BackendAvailability("Ollama", True, "0.5.1"),
            BackendAvailability("MLX-LM", False),
            BackendAvailability("MLX", False),
            BackendAvailability("CTransformers", False),
            BackendAvailability("ExLlamaV2", False),
        ],
        python_version="3.12.4",
        hostname="amd-workstation",
    ),

    "windows-4090": SystemProfile(
        os_name="Windows",
        os_version="10.0.22631",
        arch="AMD64",
        cpu=CPUInfo(
            model="Intel Core i9-14900K",
            arch="AMD64",
            physical_cores=24,
            logical_cores=32,
            features=["AVX", "AVX2", "FMA"],
        ),
        memory=MemoryInfo(total_mb=65536, available_mb=55000),
        gpus=[GPUInfo(
            name="NVIDIA GeForce RTX 4090",
            accelerator=AcceleratorType.NVIDIA_CUDA,
            vram_mb=24576,
            driver_version="556.12",
            compute_capability="8.9",
        )],
        backends=[
            BackendAvailability("vLLM", False),  # vLLM doesn't support Windows
            BackendAvailability("TensorRT-LLM", False),
            BackendAvailability("llama.cpp", True, "b4321"),
            BackendAvailability("Ollama", True, "0.5.1"),
            BackendAvailability("MLX-LM", False),
            BackendAvailability("MLX", False),
            BackendAvailability("CTransformers", False),
            BackendAvailability("ExLlamaV2", True, "0.2.1"),
        ],
        python_version="3.12.4",
        hostname="GAMING-PC",
    ),

    "windows-cpu": SystemProfile(
        os_name="Windows",
        os_version="10.0.19045",
        arch="AMD64",
        cpu=CPUInfo(
            model="Intel Core i5-10400",
            arch="AMD64",
            physical_cores=6,
            logical_cores=12,
            features=["AVX", "AVX2"],
        ),
        memory=MemoryInfo(total_mb=16384, available_mb=12000),
        gpus=[],
        backends=[
            BackendAvailability("vLLM", False),
            BackendAvailability("TensorRT-LLM", False),
            BackendAvailability("llama.cpp", False),
            BackendAvailability("Ollama", True, "0.5.1"),
            BackendAvailability("MLX-LM", False),
            BackendAvailability("MLX", False),
            BackendAvailability("CTransformers", False),
            BackendAvailability("ExLlamaV2", False),
        ],
        python_version="3.11.9",
        hostname="OFFICE-PC",
    ),

    "mac-m4-ultra": SystemProfile(
        os_name="Darwin",
        os_version="25.5.0",
        arch="arm64",
        cpu=CPUInfo(
            model="Apple M4 Ultra",
            arch="arm64",
            physical_cores=32,
            logical_cores=32,
            features=["NEON", "FP16", "METAL"],
        ),
        memory=MemoryInfo(total_mb=196608, available_mb=180000),
        gpus=[GPUInfo(
            name="Apple M4 Ultra",
            accelerator=AcceleratorType.APPLE_METAL,
            vram_mb=147456,  # 75% of 192GB
        )],
        backends=[
            BackendAvailability("vLLM", False),
            BackendAvailability("TensorRT-LLM", False),
            BackendAvailability("llama.cpp", True, "b4321"),
            BackendAvailability("Ollama", True, "0.17.7"),
            BackendAvailability("MLX-LM", False),
            BackendAvailability("MLX", False),
            BackendAvailability("CTransformers", False),
            BackendAvailability("ExLlamaV2", False),
        ],
        python_version="3.12.4",
        hostname="mac-studio",
    ),

    "linux-bare": SystemProfile(
        os_name="Linux",
        os_version="6.1.0-minimal",
        arch="x86_64",
        cpu=CPUInfo(
            model="Intel Celeron N5105",
            arch="x86_64",
            physical_cores=4,
            logical_cores=4,
            features=["AVX", "AVX2"],
        ),
        memory=MemoryInfo(total_mb=8192, available_mb=6500),
        gpus=[],
        backends=[
            BackendAvailability("vLLM", False),
            BackendAvailability("TensorRT-LLM", False),
            BackendAvailability("llama.cpp", False),
            BackendAvailability("Ollama", False),
            BackendAvailability("MLX-LM", False),
            BackendAvailability("MLX", False),
            BackendAvailability("CTransformers", False),
            BackendAvailability("ExLlamaV2", False),
        ],
        python_version="3.11.2",
        hostname="mini-server",
    ),
}


def get_simulated_profile(name: str) -> Optional[SystemProfile]:
    """Get a simulated system profile by name. Returns None if not found."""
    return SIMULATED_PROFILES.get(name)


def list_simulated_profiles() -> list[str]:
    """List all available simulated profile names."""
    return list(SIMULATED_PROFILES.keys())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    profile = detect_system()
    print(profile.summary())
    print()

    if "--json" in sys.argv:
        print(profile.to_json())
