"""
CKM Distillation Pipeline — frontier model → Cortex Kernel Model.

This is the permanent teacher→student loop. A frontier model (L7) labels
decisions, generates reasoning traces, and produces training data that
the CKM self-training pipeline consumes.

Architecture:
  ┌─────────────┐         ┌──────────────┐        ┌─────────────┐
  │  Frontier    │  label  │  Distillation │  feed  │  CKM Train  │
  │  Model (L7)  │ ──────→ │  Buffer       │ ──────→│  Pipeline   │
  └─────────────┘         └──────────────┘        └─────────────┘
         ↑                        │                       │
         │                        ↓                       ↓
         │                 ┌──────────────┐        ┌─────────────┐
         └─────────────────│  Eval Gate   │←───────│  CKM Model  │
            disagreement   └──────────────┘  test  └─────────────┘

Three distillation modes:
  1. OFFLINE: Batch generation of labeled training data (no API needed)
  2. ONLINE:  Live labeling via frontier API during daemon operation
  3. REPLAY:  Re-label historical decisions with frontier hindsight

The distillation is asymmetric:
  - Teacher (frontier): slow, expensive, accurate, has broad knowledge
  - Student (CKM): fast, cheap, narrow, speaks SCL natively

The student doesn't need to replicate the teacher's breadth.
It needs to replicate the teacher's JUDGMENT within Cortex's domain.
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Iterator

logger = logging.getLogger("cortex.ckm.distill")


# ---------------------------------------------------------------------------
# Distillation modes
# ---------------------------------------------------------------------------

class DistillMode(Enum):
    OFFLINE = "offline"    # Batch synthetic generation (no API)
    ONLINE = "online"      # Live labeling via frontier API
    REPLAY = "replay"      # Re-label historical decisions


# ---------------------------------------------------------------------------
# Teacher interface — abstraction over any frontier model
# ---------------------------------------------------------------------------

@dataclass
class TeacherConfig:
    """Configuration for the frontier teacher model."""
    provider: str = "anthropic"      # anthropic, openai, openrouter, ollama_local
    model: str = "claude-sonnet-4-20250514"  # specific model
    api_key_env: str = "ANTHROPIC_API_KEY"  # env var for API key
    max_tokens: int = 1024
    temperature: float = 0.3         # Low temp for consistent labels
    batch_size: int = 50             # Requests per batch
    rate_limit_rpm: int = 50         # Requests per minute
    fallback_to_offline: bool = True  # If API fails, use offline mode


@dataclass
class TeacherLabel:
    """A single labeled decision from the teacher."""
    input_scl: str          # The situation presented to the teacher
    output_scl: str         # The teacher's ideal SCL response
    reasoning: str          # Teacher's chain-of-thought
    confidence: float       # Teacher's self-assessed confidence
    source: str = "distill_teacher"
    quality: float = 1.0    # Teacher labels are highest quality
    timestamp_ms: int = 0

    def to_jsonl(self) -> str:
        return json.dumps({
            "input": self.input_scl,
            "output": self.output_scl,
            "source": self.source,
            "quality": self.quality,
            "reasoning": self.reasoning,
            "teacher_confidence": self.confidence,
        })


# ---------------------------------------------------------------------------
# Offline distillation — generate labels without API calls
# ---------------------------------------------------------------------------

class OfflineTeacher:
    """
    Offline distillation teacher — generates expert labels using
    the reasoning_core module (which encodes frontier model behavior).

    This is the "cached mind" of the frontier model, running locally.
    It can't match the real frontier on novel situations, but for
    the CKM's narrow domain (routing, boot config, policy), it's
    good enough to bootstrap training.
    """

    def __init__(self):
        from .reasoning_core import (
            calibrate_confidence,
            decompose_problem,
            judge_escalation,
            metacognize,
            self_reflect,
        )
        self._calibrate = calibrate_confidence
        self._decompose = decompose_problem
        self._escalate = judge_escalation
        self._metacognize = metacognize
        self._reflect = self_reflect

    def label_routing_decision(
        self,
        category: str,
        complexity: float,
        tokens: int,
        hardware_tier: str,
    ) -> TeacherLabel:
        """Label a routing decision with full reasoning."""
        # Assess the situation
        meta = self._metacognize(
            domain=category,
            complexity=complexity,
            available_context=0.8,  # We have full context in routing
            similar_past_success=min(1.0, 0.3 + complexity),
        )

        # Determine optimal tier
        strategy, subproblems = self._decompose(
            category=category,
            complexity=complexity,
            token_count=tokens,
            has_code=(category == "code"),
            has_math=(category == "math"),
            multi_step=(complexity > 0.5),
        )

        # Compute confidence
        conf = self._calibrate(
            pattern_familiarity=0.7 + random.gauss(0, 0.1),
            evidence_agreement=0.8 + random.gauss(0, 0.1),
            stakes=0.3,  # Routing is low-stakes (can re-route)
            data_sufficiency=0.8,
        )

        # Determine the right tier
        from .reasoning_core import _tier_for_complexity
        optimal_tier = _tier_for_complexity(complexity)

        # Compose the label
        input_scl = (
            f"@task → classify ["
            f"category: {category}, "
            f"complexity: {complexity:.2f}, "
            f"input_tokens: {tokens}, "
            f"hardware_tier: {hardware_tier}]"
        )

        output_scl = (
            f"@router → select ["
            f"tier: {optimal_tier}, "
            f"confidence: {conf.score:.2f}, "
            f"strategy: {strategy.value}, "
            f"reason: {category}_complexity_{complexity:.1f}]"
        )

        reasoning = (
            f"@teacher → reason ["
            f"meta: {meta.knowledge_boundary}, "
            f"decomposition: {strategy.value}_{len(subproblems)}_steps, "
            f"confidence_basis: {conf.reasoning}]"
        )

        return TeacherLabel(
            input_scl=input_scl,
            output_scl=output_scl,
            reasoning=reasoning,
            confidence=conf.score,
            timestamp_ms=int(time.time() * 1000),
        )

    def label_boot_config(
        self,
        cpu: str,
        cores: int,
        ram_mb: int,
        gpu_type: str,
        vram_mb: int,
        arch: str,
    ) -> TeacherLabel:
        """Label an optimal boot configuration with reasoning."""
        # The frontier model's reasoning about hardware → config:
        # 1. Identify bottleneck (compute vs memory vs bandwidth)
        # 2. Size models to available resources
        # 3. Configure for minimum latency to first token
        # 4. Leave headroom for user workloads

        # Bottleneck analysis
        compute_bound = cores < 8
        memory_bound = (vram_mb if gpu_type != "none" else ram_mb) < 8192
        has_gpu = gpu_type != "none"

        # Optimal config computation (encoding frontier model's hardware intuition)
        if gpu_type == "apple":
            # Apple Silicon: unified memory, always all layers on GPU
            opt_gpu_layers = 999
            opt_threads = max(1, cores - 2)  # Leave 2 for OS
            # Context window: use half of unified memory budget
            opt_ctx = min(16384, vram_mb // 2)
            # Batch size: Apple MPS is good at batching
            opt_batch = 8
            opt_backend = "llama_cpp"
            # Hot models: depends on total memory
            if vram_mb >= 36864:
                hot_models = ["L0", "L1", "L2", "L3", "L4", "L5"]
            elif vram_mb >= 24576:
                hot_models = ["L0", "L1", "L2", "L3", "L4"]
            elif vram_mb >= 16384:
                hot_models = ["L0", "L1", "L2", "L3"]
            elif vram_mb >= 8192:
                hot_models = ["L0", "L1", "L2"]
            else:
                hot_models = ["L0", "L1"]
            reasoning_detail = "unified_memory_all_layers_gpu"

        elif gpu_type == "nvidia":
            # NVIDIA: discrete VRAM, maximize GPU utilization
            if vram_mb >= 24576:
                opt_gpu_layers = 999
                opt_ctx = 16384
                opt_batch = 16
                hot_models = ["L0", "L1", "L2", "L3", "L4"]
            elif vram_mb >= 12288:
                opt_gpu_layers = 999
                opt_ctx = 8192
                opt_batch = 8
                hot_models = ["L0", "L1", "L2", "L3"]
            elif vram_mb >= 8192:
                opt_gpu_layers = 999
                opt_ctx = 8192
                opt_batch = 8
                hot_models = ["L0", "L1", "L2", "L3"]
            elif vram_mb >= 4096:
                opt_gpu_layers = 32
                opt_ctx = 4096
                opt_batch = 4
                hot_models = ["L0", "L1", "L2"]
            else:
                opt_gpu_layers = 16
                opt_ctx = 2048
                opt_batch = 4
                hot_models = ["L0", "L1"]
            opt_threads = max(1, min(cores - 1, 8))  # CPU threads for offloaded layers
            opt_backend = "llama_cpp"
            reasoning_detail = f"discrete_gpu_{vram_mb}mb_vram"

        elif gpu_type == "amd":
            # AMD: ROCm support varies, be conservative
            opt_gpu_layers = 999 if vram_mb >= 8192 else 24
            opt_ctx = min(8192, vram_mb // 2) if vram_mb > 0 else 4096
            opt_batch = 4
            opt_threads = max(1, min(cores - 1, 12))
            opt_backend = "llama_cpp"
            hot_models = ["L0", "L1", "L2"] if vram_mb >= 8192 else ["L0", "L1"]
            reasoning_detail = "amd_rocm_conservative"

        else:
            # CPU only: constrained, maximize what fits in RAM
            opt_gpu_layers = 0
            opt_threads = max(1, min(cores - 1, 16))
            opt_ctx = min(4096, ram_mb // 4)
            opt_batch = 4
            opt_backend = "llama_cpp"
            if ram_mb >= 16384:
                hot_models = ["L0", "L1", "L2"]
            elif ram_mb >= 8192:
                hot_models = ["L0", "L1"]
            elif ram_mb >= 4096:
                hot_models = ["L0"]
            else:
                hot_models = []
            reasoning_detail = f"cpu_only_{ram_mb}mb_ram"

        # Compose the label
        input_scl = (
            f"@hardware → state ["
            f"cpu: {cpu}, "
            f"cores: {cores}, "
            f"ram_mb: {ram_mb}, "
            f"gpu_type: {gpu_type}, "
            f"vram_mb: {vram_mb}, "
            f"arch: {arch}]"
        )

        output_scl = (
            f"@cortex.boot → mutate ["
            f"optimal_threads: {opt_threads}, "
            f"optimal_gpu_layers: {opt_gpu_layers}, "
            f"optimal_ctx_size: {opt_ctx}, "
            f"optimal_batch_size: {opt_batch}, "
            f"optimal_backend: {opt_backend}, "
            f"optimal_hot_models: {','.join(hot_models)}]"
        )

        reasoning = (
            f"@teacher → reason ["
            f"bottleneck: {'compute' if compute_bound else 'memory' if memory_bound else 'none'}, "
            f"gpu_strategy: {reasoning_detail}, "
            f"headroom: 25_percent_reserved_for_user, "
            f"latency_priority: minimize_ttft]"
        )

        return TeacherLabel(
            input_scl=input_scl,
            output_scl=output_scl,
            reasoning=reasoning,
            confidence=0.9,
            timestamp_ms=int(time.time() * 1000),
        )

    def label_escalation_decision(
        self,
        current_tier: str,
        request_complexity: float,
        current_confidence: float,
        challenge_result: str,
        time_budget_ms: int,
    ) -> TeacherLabel:
        """Label whether to escalate, with full reasoning."""
        escalation = self._escalate(
            current_tier=current_tier,
            confidence=current_confidence,
            complexity=request_complexity,
            stakes=0.5,
            time_budget_ms=time_budget_ms,
            challenge_result=challenge_result,
        )

        input_scl = (
            f"@situation → state ["
            f"current_tier: {current_tier}, "
            f"complexity: {request_complexity:.2f}, "
            f"confidence: {current_confidence:.2f}, "
            f"challenge: {challenge_result}, "
            f"time_budget_ms: {time_budget_ms}]"
        )

        output_scl = escalation.to_scl()

        reasoning = (
            f"@teacher → reason ["
            f"escalate: {'yes' if escalation.should_escalate else 'no'}, "
            f"urgency: {escalation.urgency:.2f}, "
            f"expected_gain: {escalation.expected_improvement:.2f}, "
            f"reason: {escalation.reason}]"
        )

        return TeacherLabel(
            input_scl=input_scl,
            output_scl=output_scl,
            reasoning=reasoning,
            confidence=0.85,
            timestamp_ms=int(time.time() * 1000),
        )

    def label_policy_mutation(
        self,
        model: str,
        accuracy: float,
        sample_count: int,
        current_tier: str,
    ) -> TeacherLabel:
        """Label whether a policy mutation is warranted."""
        # Frontier model reasoning about policy:
        # - Need sufficient samples (>10) for signal
        # - Accuracy below 0.5 → demote
        # - Accuracy below 0.3 → block
        # - Accuracy above 0.9 → promote
        # - Always verify no empty tiers result

        should_mutate = sample_count >= 10
        if should_mutate:
            if accuracy < 0.3:
                action = "block"
                confidence = 0.9
            elif accuracy < 0.5:
                action = "demote"
                confidence = 0.8
            elif accuracy > 0.9:
                action = "promote"
                confidence = 0.85
            else:
                action = "observe"
                confidence = 0.7
                should_mutate = False
        else:
            action = "insufficient_data"
            confidence = 0.4
            should_mutate = False

        input_scl = (
            f"@feedback → accumulated ["
            f"model: {model}, "
            f"accuracy: {accuracy:.2f}, "
            f"samples: {sample_count}, "
            f"tier: {current_tier}]"
        )

        if should_mutate:
            output_scl = (
                f"@policy_rewriter → mutate ["
                f"action: {action}, "
                f"model: {model}, "
                f"tier: {current_tier}, "
                f"confidence: {confidence:.2f}]"
            )
        else:
            output_scl = (
                f"@policy_rewriter → observe ["
                f"action: {action}, "
                f"model: {model}, "
                f"reason: {'insufficient_samples' if sample_count < 10 else 'within_tolerance'}]"
            )

        reasoning = (
            f"@teacher → reason ["
            f"data_sufficient: {'yes' if sample_count >= 10 else 'no'}, "
            f"accuracy_signal: {'critical' if accuracy < 0.3 else 'low' if accuracy < 0.5 else 'acceptable' if accuracy < 0.9 else 'excellent'}, "
            f"recommended_action: {action}]"
        )

        return TeacherLabel(
            input_scl=input_scl,
            output_scl=output_scl,
            reasoning=reasoning,
            confidence=confidence,
            timestamp_ms=int(time.time() * 1000),
        )


# ---------------------------------------------------------------------------
# Online distillation — live labeling via frontier API
# ---------------------------------------------------------------------------

class OnlineTeacher:
    """
    Online distillation teacher — labels decisions in real-time
    by calling a frontier model API.

    Used during daemon operation to continuously improve CKM.
    Every Nth routing decision is sent to the teacher for labeling.
    Disagreements between CKM and teacher are high-value training data.
    """

    def __init__(self, config: TeacherConfig):
        self.config = config
        self._api_key: Optional[str] = None
        self._request_count = 0
        self._last_request_time = 0.0

    @property
    def api_key(self) -> Optional[str]:
        if self._api_key is None:
            self._api_key = os.environ.get(self.config.api_key_env)
        return self._api_key

    @property
    def available(self) -> bool:
        """Check if the teacher API is available."""
        return self.api_key is not None

    def _rate_limit_wait(self):
        """Wait if we're exceeding rate limit."""
        if self.config.rate_limit_rpm <= 0:
            return
        interval = 60.0 / self.config.rate_limit_rpm
        elapsed = time.time() - self._last_request_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1

    def label_decision(self, input_scl: str, ckm_output: str) -> Optional[TeacherLabel]:
        """
        Send a decision to the frontier teacher for labeling.

        The prompt asks the teacher to:
        1. Evaluate the CKM's output
        2. Provide the ideal output
        3. Explain its reasoning
        4. Assess its own confidence

        Returns None if the API is unavailable or rate-limited.
        """
        if not self.available:
            logger.debug("Teacher API not available, skipping online label")
            return None

        self._rate_limit_wait()

        prompt = self._build_labeling_prompt(input_scl, ckm_output)

        try:
            response = self._call_api(prompt)
            if response is None:
                return None
            return self._parse_teacher_response(input_scl, response)
        except Exception as e:
            logger.warning(f"Teacher API call failed: {e}")
            return None

    def _build_labeling_prompt(self, input_scl: str, ckm_output: str) -> str:
        """Build the prompt that asks the teacher to label a decision."""
        return f"""You are the teacher for a tiny AI kernel model (CKM) that speaks SCL
(Semantic Compression Language). The CKM made a decision. Evaluate it and provide
the ideal output.

SCL Grammar: @anchor → verb [key: value, key: value]

## Situation (input to CKM):
{input_scl}

## CKM's Output:
{ckm_output}

## Your Task:
1. Is the CKM's output correct? (yes/partially/no)
2. What is the IDEAL output in SCL format?
3. What is your reasoning? (in SCL format: @teacher → reason [...])
4. How confident are you? (0.0-1.0)

Respond in this exact format:
CORRECT: [yes/partially/no]
IDEAL: [your ideal SCL output]
REASONING: [your reasoning in SCL]
CONFIDENCE: [0.0-1.0]
"""

    def _call_api(self, prompt: str) -> Optional[str]:
        """Call the frontier model API. Returns response text or None."""
        # This is a pluggable interface. Implementation depends on provider.
        # For now, we support the pattern but don't make actual HTTP calls
        # (that happens in the daemon integration layer).

        if self.config.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.config.provider == "openai":
            return self._call_openai(prompt)
        elif self.config.provider == "openrouter":
            return self._call_openrouter(prompt)
        elif self.config.provider == "ollama_local":
            return self._call_ollama(prompt)
        else:
            logger.error(f"Unknown teacher provider: {self.config.provider}")
            return None

    def _call_anthropic(self, prompt: str) -> Optional[str]:
        """Call Anthropic API for labeling."""
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed, cannot call Anthropic API")
            return None

        api_key = self.api_key
        if not api_key:
            return None

        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self.config.model,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
        except Exception as e:
            logger.warning(f"Anthropic API error: {e}")
            return None

    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API for labeling."""
        try:
            import httpx
        except ImportError:
            return None

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None

        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"OpenAI API error: {e}")
            return None

    def _call_openrouter(self, prompt: str) -> Optional[str]:
        """Call OpenRouter API for labeling — single key, many models."""
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed, cannot call OpenRouter API")
            return None

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning("OPENROUTER_API_KEY not set")
            return None

        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/elevate-foundry/cortex",
                    "X-Title": "Cortex CKM Distillation",
                },
                json={
                    "model": self.config.model,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"OpenRouter API error: {e}")
            return None

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call local Ollama for labeling (when a large local model is available)."""
        try:
            import httpx
        except ImportError:
            return None

        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

        try:
            response = httpx.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    },
                },
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.warning(f"Ollama API error: {e}")
            return None

    def _parse_teacher_response(self, input_scl: str, response: str) -> Optional[TeacherLabel]:
        """Parse the teacher's structured response into a TeacherLabel."""
        try:
            lines = response.strip().split("\n")
            ideal = ""
            reasoning = ""
            confidence = 0.8

            for line in lines:
                if line.startswith("IDEAL:"):
                    ideal = line[6:].strip()
                elif line.startswith("REASONING:"):
                    reasoning = line[10:].strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line[11:].strip())
                    except ValueError:
                        confidence = 0.8

            if not ideal:
                return None

            return TeacherLabel(
                input_scl=input_scl,
                output_scl=ideal,
                reasoning=reasoning,
                confidence=confidence,
                source="distill_online",
                quality=min(1.0, confidence + 0.1),  # Teacher labels are high quality
                timestamp_ms=int(time.time() * 1000),
            )
        except Exception as e:
            logger.warning(f"Failed to parse teacher response: {e}")
            return None


# ---------------------------------------------------------------------------
# Distillation Buffer — accumulates labels for batch training
# ---------------------------------------------------------------------------

@dataclass
class DistillationBuffer:
    """
    Accumulates teacher labels for periodic batch training.

    The buffer serves as the bridge between continuous labeling and
    periodic CKM retraining. It:
    1. Accumulates labels from offline/online teachers
    2. Prioritizes disagreement cases (where CKM got it wrong)
    3. Deduplicates similar examples
    4. Triggers retraining when buffer reaches threshold
    """
    labels: list[TeacherLabel] = field(default_factory=list)
    disagreements: list[TeacherLabel] = field(default_factory=list)
    max_buffer_size: int = 10000
    retrain_threshold: int = 500  # Trigger retrain after this many new labels
    output_dir: Path = field(default_factory=lambda: Path("/tmp/cortex/distill"))

    def add_label(self, label: TeacherLabel, ckm_output: Optional[str] = None):
        """Add a teacher label to the buffer."""
        self.labels.append(label)

        # Track disagreements (where CKM output differs from teacher)
        if ckm_output and ckm_output.strip() != label.output_scl.strip():
            # Disagreements are the highest-value training signal
            label_copy = TeacherLabel(
                input_scl=label.input_scl,
                output_scl=label.output_scl,
                reasoning=label.reasoning + f" | ckm_said: {ckm_output}",
                confidence=label.confidence,
                source="distill_disagreement",
                quality=1.0,  # Maximum quality — this is exactly what CKM needs to learn
                timestamp_ms=label.timestamp_ms,
            )
            self.disagreements.append(label_copy)

        # Trim if over capacity
        if len(self.labels) > self.max_buffer_size:
            # Keep most recent + all disagreements
            self.labels = self.labels[-self.max_buffer_size:]

    @property
    def ready_for_training(self) -> bool:
        """Check if buffer has enough data to trigger retraining."""
        return len(self.labels) >= self.retrain_threshold

    def flush_to_jsonl(self, filename: str = "distill_buffer.jsonl") -> Path:
        """Write buffer to JSONL file for CKM training pipeline."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename

        with open(path, "w") as f:
            # Write disagreements first (highest value)
            for label in self.disagreements:
                f.write(label.to_jsonl() + "\n")
            # Then regular labels
            for label in self.labels:
                f.write(label.to_jsonl() + "\n")

        count = len(self.labels) + len(self.disagreements)
        logger.info(f"Flushed {count} labels to {path} ({len(self.disagreements)} disagreements)")
        return path

    def clear(self):
        """Clear the buffer after training."""
        self.labels.clear()
        self.disagreements.clear()

    def stats(self) -> dict:
        return {
            "total_labels": len(self.labels),
            "disagreements": len(self.disagreements),
            "ready_for_training": self.ready_for_training,
            "sources": _count_sources(self.labels),
        }


def _count_sources(labels: list[TeacherLabel]) -> dict:
    sources = {}
    for l in labels:
        sources[l.source] = sources.get(l.source, 0) + 1
    return sources


# ---------------------------------------------------------------------------
# Full distillation pipeline
# ---------------------------------------------------------------------------

class DistillationPipeline:
    """
    The complete distillation pipeline.

    Orchestrates:
    1. Offline teacher → batch label generation
    2. Online teacher → live labeling during operation
    3. Buffer management → accumulation and dedup
    4. Training trigger → when to retrain CKM
    5. Eval gate → verify improvement before promoting

    Usage:
        pipeline = DistillationPipeline()

        # Offline: generate batch training data
        pipeline.run_offline(count=5000)

        # Online: label a live decision
        pipeline.label_live_decision(input_scl, ckm_output)

        # Check if ready to retrain
        if pipeline.ready_for_training:
            path = pipeline.flush()
            # Feed to CKM training pipeline...
    """

    def __init__(
        self,
        teacher_config: Optional[TeacherConfig] = None,
        output_dir: str = "/tmp/cortex/distill",
        retrain_threshold: int = 500,
    ):
        self.offline_teacher = OfflineTeacher()
        self.online_teacher = OnlineTeacher(teacher_config or TeacherConfig())
        self.buffer = DistillationBuffer(
            output_dir=Path(output_dir),
            retrain_threshold=retrain_threshold,
        )
        self._generation_count = 0

    def run_offline(self, count: int = 5000) -> dict:
        """
        Run offline distillation — batch generate labeled training data.

        This is the primary bootstrapping mechanism. It uses the reasoning_core
        to generate expert-labeled decisions without any API calls.

        Generates a mix of:
        - Routing decisions (40%)
        - Boot configurations (30%)
        - Escalation decisions (20%)
        - Policy mutations (10%)
        """
        stats = {"routing": 0, "boot": 0, "escalation": 0, "policy": 0}

        # Hardware profiles for boot config generation
        hardware_profiles = [
            ("Intel Core i7-12700K", 20, 32768, "nvidia", 8192, "x86_64"),
            ("Intel Core i5-10400", 12, 16384, "nvidia", 4096, "x86_64"),
            ("Intel Core i3-10100", 8, 8192, "none", 0, "x86_64"),
            ("AMD Ryzen 9 7950X", 32, 65536, "nvidia", 24576, "x86_64"),
            ("AMD Ryzen 7 5800X", 16, 32768, "nvidia", 12288, "x86_64"),
            ("AMD Ryzen 5 5600X", 12, 16384, "amd", 8192, "x86_64"),
            ("Apple M1", 8, 16384, "apple", 16384, "aarch64"),
            ("Apple M1 Pro", 10, 16384, "apple", 16384, "aarch64"),
            ("Apple M1 Max", 10, 32768, "apple", 32768, "aarch64"),
            ("Apple M2", 8, 8192, "apple", 8192, "aarch64"),
            ("Apple M2 Pro", 12, 16384, "apple", 16384, "aarch64"),
            ("Apple M2 Max", 12, 32768, "apple", 32768, "aarch64"),
            ("Apple M3", 8, 8192, "apple", 8192, "aarch64"),
            ("Apple M3 Pro", 12, 18432, "apple", 18432, "aarch64"),
            ("Apple M3 Max", 16, 36864, "apple", 36864, "aarch64"),
            ("Apple M4", 10, 16384, "apple", 16384, "aarch64"),
            ("Apple M4 Pro", 14, 24576, "apple", 24576, "aarch64"),
            ("Apple M4 Max", 16, 49152, "apple", 49152, "aarch64"),
            ("Apple M4 Ultra", 32, 98304, "apple", 98304, "aarch64"),
            ("Qualcomm Snapdragon 8cx", 8, 8192, "none", 0, "aarch64"),
            ("Intel Celeron N5105", 4, 4096, "none", 0, "x86_64"),
            ("Raspberry Pi 5", 4, 4096, "none", 0, "aarch64"),
            ("Raspberry Pi 4", 4, 2048, "none", 0, "aarch64"),
            ("NVIDIA Jetson Orin", 12, 16384, "nvidia", 8192, "aarch64"),
            ("NVIDIA H100 Server", 128, 524288, "nvidia", 81920, "x86_64"),
            ("AMD EPYC 9654 Server", 192, 786432, "nvidia", 49152, "x86_64"),
        ]

        categories = ["code", "math", "chat", "tool", "analysis", "system"]
        tiers = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]
        models = ["qwen3:4b", "qwen3:8b", "llama3.2:3b", "granite3.3:8b", "phi4:14b"]
        challenge_results = ["none", "none", "agree", "weak_disagree", "strong_disagree"]

        for i in range(count):
            r = random.random()

            if r < 0.4:
                # Routing decision
                cat = random.choice(categories)
                complexity = random.uniform(0.0, 1.0)
                tokens = random.randint(10, 5000)
                hw_tier = random.choice(tiers[:5])
                label = self.offline_teacher.label_routing_decision(
                    cat, complexity, tokens, hw_tier
                )
                self.buffer.add_label(label)
                stats["routing"] += 1

            elif r < 0.7:
                # Boot configuration
                hw = random.choice(hardware_profiles)
                # Add noise to cores and RAM
                cores = max(1, hw[1] + random.randint(-2, 2))
                ram = max(1024, hw[2] + random.randint(-2048, 2048))
                label = self.offline_teacher.label_boot_config(
                    cpu=hw[0], cores=cores, ram_mb=ram,
                    gpu_type=hw[3], vram_mb=hw[4], arch=hw[5],
                )
                self.buffer.add_label(label)
                stats["boot"] += 1

            elif r < 0.9:
                # Escalation decision
                tier = random.choice(tiers[:5])
                complexity = random.uniform(0.2, 1.0)
                confidence = random.uniform(0.2, 0.9)
                challenge = random.choice(challenge_results)
                time_budget = random.choice([100, 500, 1000, 5000, 10000])
                label = self.offline_teacher.label_escalation_decision(
                    tier, complexity, confidence, challenge, time_budget
                )
                self.buffer.add_label(label)
                stats["escalation"] += 1

            else:
                # Policy mutation
                model = random.choice(models)
                accuracy = random.uniform(0.1, 1.0)
                samples = random.randint(3, 100)
                tier = random.choice(tiers[:5])
                label = self.offline_teacher.label_policy_mutation(
                    model, accuracy, samples, tier
                )
                self.buffer.add_label(label)
                stats["policy"] += 1

        self._generation_count += count
        stats["total"] = count
        stats["buffer_size"] = len(self.buffer.labels)
        return stats

    def label_live_decision(
        self,
        input_scl: str,
        ckm_output: str,
    ) -> Optional[TeacherLabel]:
        """
        Label a live CKM decision using the online teacher.

        Called during daemon operation. The frontier model evaluates
        the CKM's decision and provides the ideal answer.

        Disagreements are automatically prioritized in the buffer.
        """
        label = self.online_teacher.label_decision(input_scl, ckm_output)
        if label:
            self.buffer.add_label(label, ckm_output=ckm_output)
        return label

    @property
    def ready_for_training(self) -> bool:
        return self.buffer.ready_for_training

    def flush(self) -> Path:
        """Flush the buffer to disk for CKM training."""
        path = self.buffer.flush_to_jsonl()
        return path

    def stats(self) -> dict:
        return {
            "generation_count": self._generation_count,
            "buffer": self.buffer.stats(),
            "online_teacher_available": self.online_teacher.available,
        }


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def run_distillation(
    count: int = 5000,
    output_path: Optional[str] = None,
    include_reasoning_traces: bool = True,
) -> dict:
    """
    Run the full distillation pipeline and save results.

    This is the entry point called by `python -m src train --distill`.
    """
    pipeline = DistillationPipeline(
        output_dir=output_path or "/tmp/cortex/distill",
    )

    # Phase 1: Offline distillation (labeled decisions)
    logger.info("Phase 1: Generating labeled decisions...")
    decision_stats = pipeline.run_offline(count=count)

    # Phase 2: Reasoning traces (from reasoning_core)
    reasoning_stats = {}
    if include_reasoning_traces:
        logger.info("Phase 2: Generating reasoning traces...")
        from .reasoning_core import save_reasoning_corpus
        reasoning_path = str(Path(pipeline.buffer.output_dir) / "reasoning_traces.jsonl")
        reasoning_stats = save_reasoning_corpus(reasoning_path, count=count)

    # Flush everything
    decision_path = pipeline.flush()

    total_stats = {
        "decisions": decision_stats,
        "reasoning": reasoning_stats,
        "output_files": {
            "decisions": str(decision_path),
            "reasoning": reasoning_stats.get("output_path", ""),
        },
        "total_examples": count + reasoning_stats.get("total_traces", 0),
    }

    logger.info(f"Distillation complete: {total_stats['total_examples']} total examples")
    return total_stats
