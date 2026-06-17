"""
Shared test fixtures for the Cortex test suite.

Provides:
  - Simulated SystemProfile objects for different hardware configs
  - Mock BackendAdapter that returns canned responses
  - In-memory Memory instance (SQLite :memory:)
  - Deterministic CortexConfig for reproducible tests
  - Sample SCL records and documents
"""

import pytest
from unittest.mock import MagicMock

from src.hardware_detect import (
    SystemProfile, CPUInfo, MemoryInfo, GPUInfo,
    BackendAvailability, AcceleratorType,
)
from src.backend_adapter import CompletionResponse, BackendType
from src.cortex import CortexConfig
from src.tiers import Tier
from src.scl.types import Anchor, Relation, Scope, SCLRecord, SCLDocument


# ---------------------------------------------------------------------------
# Hardware profiles
# ---------------------------------------------------------------------------

@pytest.fixture
def profile_m1_pro() -> SystemProfile:
    """Apple M1 Pro, 16GB unified memory."""
    return SystemProfile(
        os_name="Darwin",
        os_version="25.5.0",
        arch="arm64",
        cpu=CPUInfo(
            model="Apple M1 Pro",
            arch="arm64",
            physical_cores=10,
            logical_cores=10,
            features=["NEON", "FP16", "METAL"],
        ),
        memory=MemoryInfo(total_mb=16384, available_mb=8192),
        gpus=[GPUInfo(
            name="Apple M1 Pro",
            accelerator=AcceleratorType.APPLE_METAL,
            vram_mb=12288,
        )],
        backends=[
            BackendAvailability(name="Ollama", available=True, version="0.17.7"),
            BackendAvailability(name="llama.cpp", available=True),
            BackendAvailability(name="MLX", available=True),
        ],
    )


@pytest.fixture
def profile_rtx4090() -> SystemProfile:
    """Linux workstation with RTX 4090, 24GB VRAM."""
    return SystemProfile(
        os_name="Linux",
        os_version="6.5.0",
        arch="x86_64",
        cpu=CPUInfo(
            model="AMD Ryzen 9 7950X",
            arch="x86_64",
            physical_cores=16,
            logical_cores=32,
            features=["AVX2", "AVX-512"],
        ),
        memory=MemoryInfo(total_mb=65536, available_mb=50000),
        gpus=[GPUInfo(
            name="NVIDIA RTX 4090",
            accelerator=AcceleratorType.NVIDIA_CUDA,
            vram_mb=24576,
            compute_capability="8.9",
        )],
        backends=[
            BackendAvailability(name="vLLM", available=True, version="0.4.0"),
            BackendAvailability(name="Ollama", available=True),
            BackendAvailability(name="llama.cpp", available=True),
        ],
    )


@pytest.fixture
def profile_cpu_only() -> SystemProfile:
    """Minimal CPU-only system, 8GB RAM."""
    return SystemProfile(
        os_name="Linux",
        os_version="6.1.0",
        arch="x86_64",
        cpu=CPUInfo(
            model="Intel Core i5-1240P",
            arch="x86_64",
            physical_cores=4,
            logical_cores=8,
            features=["AVX2"],
        ),
        memory=MemoryInfo(total_mb=8192, available_mb=4096),
        gpus=[],
        backends=[
            BackendAvailability(name="llama.cpp", available=True),
        ],
    )


# ---------------------------------------------------------------------------
# Mock adapters
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_completion_response() -> CompletionResponse:
    """A canned CompletionResponse for testing."""
    return CompletionResponse(
        content="This is a test response from the mock adapter.",
        model="qwen3:4b",
        backend=BackendType.OLLAMA,
        total_ms=150.0,
        ttft_ms=25.0,
        tokens_generated=12,
    )


@pytest.fixture
def mock_adapter(mock_completion_response):
    """A mock BackendAdapter that returns canned responses."""
    adapter = MagicMock()
    adapter.complete_sync.return_value = mock_completion_response
    adapter.default_model = "qwen3:4b"
    return adapter


# ---------------------------------------------------------------------------
# Deterministic config
# ---------------------------------------------------------------------------

@pytest.fixture
def deterministic_config() -> CortexConfig:
    """CortexConfig with fixed thresholds for reproducible tests."""
    return CortexConfig(
        challenge_threshold=0.75,
        swarm_threshold=0.50,
        large_swarm_threshold=0.30,
        use_model_router=False,  # use heuristic for determinism
    )


# ---------------------------------------------------------------------------
# SCL fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_scl_record() -> SCLRecord:
    """A sample SCL record for testing."""
    return SCLRecord(
        anchor=Anchor("router"),
        relation=Relation("select"),
        scope=Scope({"model": "qwen3:4b", "confidence": "0.82"}),
    )


@pytest.fixture
def sample_scl_document() -> SCLDocument:
    """A sample SCL document with multiple records."""
    return SCLDocument(records=[
        SCLRecord(
            anchor=Anchor("task"),
            relation=Relation("classify"),
            scope=Scope({"category": "code", "complexity": "0.45"}),
        ),
        SCLRecord(
            anchor=Anchor("router"),
            relation=Relation("select"),
            scope=Scope({"tier": "L3", "confidence": "0.82"}),
        ),
        SCLRecord(
            anchor=Anchor("core"),
            relation=Relation("generate"),
            scope=Scope({"model": "qwen3:8b", "ms": "142"}),
        ),
    ])
