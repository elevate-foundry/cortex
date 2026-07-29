"""
CortexRouter Training Pipeline — audit log → JSONL → fine-tuned router.

Converts live telemetry from the daemon's audit_log + race results into
training pairs for CortexRouter-0.6B.

Each training sample:
  Input:  prompt summary + hardware context (SCL)
  Output: optimal routing decision (tier, model, confidence)

Quality signal comes from:
  - TTFT (lower = better)
  - Cost (lower = better for same quality)
  - Success/failure (did the model produce a usable response?)
  - Race results (winner = best option for that query shape)

Usage:
  python -m src.ckm.route_trainer export --output /path/to/training.jsonl
  python -m src.ckm.route_trainer race --count 100 --output /path/to/race.jsonl
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.ckm.route_trainer")


# ---------------------------------------------------------------------------
# Training sample format
# ---------------------------------------------------------------------------

@dataclass
class RoutingSample:
    """One training sample for the routing model.

    Input: what the system knew at request time.
    Label: what the optimal decision was (from telemetry feedback).
    """
    # Input features
    prompt_length: int = 0           # token count estimate
    prompt_category: str = ""        # classify, code, analyze, etc.
    prompt_complexity: float = 0.0   # 0-1 estimated complexity
    has_images: bool = False         # multimodal?
    has_tools: bool = False          # tool-use request?

    # Label (optimal decision from feedback)
    best_tier: str = ""              # L0-L7
    best_model: str = ""             # actual model that performed best
    confidence: float = 0.0          # how confident the router should be

    # Quality signal
    ttft_ms: float = 0.0             # observed TTFT
    cost_usd: float = 0.0            # observed cost
    latency_ms: float = 0.0          # total latency
    success: bool = True             # did it succeed?
    tokens_out: int = 0              # tokens generated

    # Metadata
    source: str = ""                 # "audit", "race", "synthetic"
    timestamp: float = 0.0

    def to_jsonl(self) -> str:
        """Serialize as training JSONL (input/output format for fine-tuning)."""
        input_scl = (
            f"@request → classify ["
            f"length: {self.prompt_length}, "
            f"category: {self.prompt_category}, "
            f"complexity: {self.prompt_complexity:.2f}, "
            f"images: {str(self.has_images).lower()}, "
            f"tools: {str(self.has_tools).lower()}]"
        )
        output_scl = (
            f"@router → select ["
            f"tier: {self.best_tier}, "
            f"model: {self.best_model}, "
            f"confidence: {self.confidence:.2f}]"
        )
        return json.dumps({
            "input": input_scl,
            "output": output_scl,
            "quality": self._compute_quality(),
            "source": self.source,
        })

    def _compute_quality(self) -> float:
        """Quality weight: higher = more reliable training signal."""
        q = 0.5
        if self.success:
            q += 0.2
        if self.ttft_ms > 0 and self.ttft_ms < 1000:
            q += 0.15  # fast response = good signal
        if self.source == "race":
            q += 0.15  # race winner = strong signal
        return min(q, 1.0)


# ---------------------------------------------------------------------------
# Export from audit log
# ---------------------------------------------------------------------------

def export_from_audit(db_path: Path, output: Path, min_samples: int = 10) -> int:
    """
    Export routing training data from the daemon's audit_log.

    Each audit entry becomes one training sample.
    Returns number of samples written.
    """
    import sqlite3

    if not db_path.exists():
        logger.warning("Database not found: %s", db_path)
        return 0

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT routed_tier, actual_model, category, confidence,
               tokens_prompt, tokens_completion, latency_ms, ttft_ms,
               cost_usd, provider, error, created_at
        FROM audit_log
        WHERE routed_tier != '' AND actual_model != ''
        ORDER BY created_at DESC
        LIMIT 10000
    """).fetchall()

    conn.close()

    if len(rows) < min_samples:
        logger.info("Only %d audit entries (need %d), skipping export", len(rows), min_samples)
        return 0

    samples = []
    for row in rows:
        sample = RoutingSample(
            prompt_length=row["tokens_prompt"],
            prompt_category=row["category"],
            prompt_complexity=_tier_to_complexity(row["routed_tier"]),
            best_tier=row["routed_tier"],
            best_model=row["actual_model"],
            confidence=row["confidence"],
            ttft_ms=row["ttft_ms"],
            cost_usd=row["cost_usd"] or 0.0,
            latency_ms=row["latency_ms"],
            success=not bool(row["error"]),
            tokens_out=row["tokens_completion"],
            source="audit",
            timestamp=row["created_at"] / 1000.0,
        )
        samples.append(sample)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for s in samples:
            f.write(s.to_jsonl() + "\n")

    logger.info("Exported %d routing samples to %s", len(samples), output)
    return len(samples)


# ---------------------------------------------------------------------------
# Generate from model racing
# ---------------------------------------------------------------------------

def generate_race_samples(
    api_key: str,
    prompts: list[str],
    models: Optional[list[str]] = None,
    output: Optional[Path] = None,
) -> list[RoutingSample]:
    """
    Race models on each prompt, record winners as training labels.

    The race winner for each prompt = the optimal routing decision.
    This is the strongest signal: we KNOW this model was fastest.
    """
    from .aimd_dispatch import AIMDDispatcher, race_models_stream

    if models is None:
        models = [
            "openai/gpt-4o-mini",
            "google/gemini-2.0-flash-001",
            "mistralai/mistral-small-24b-instruct-2501",
            "qwen/qwen3-coder",
            "deepseek/deepseek-chat-v3-0324",
        ]

    samples = []

    for i, prompt in enumerate(prompts):
        dispatcher = AIMDDispatcher(
            initial_parallelism=len(models),
            max_parallelism=len(models) + 2,
        )

        race_meta = {}

        def on_winner(mid, ttft, _meta=race_meta):
            _meta["winner"] = mid
            _meta["ttft"] = ttft

        def on_complete(mid, content, telemetry, _meta=race_meta):
            _meta["tokens_out"] = telemetry.tokens_out
            _meta["total_s"] = telemetry.api_response_total_s

        result = race_models_stream(
            prompt=prompt,
            model_ids=models,
            api_key=api_key,
            dispatcher=dispatcher,
            temperature=0.1,
            max_tokens=200,
            on_winner=on_winner,
            on_complete=on_complete,
        )

        if result and "winner" in race_meta:
            winner_model = race_meta["winner"]
            sample = RoutingSample(
                prompt_length=len(prompt.split()),  # rough word count
                prompt_category=_guess_category(prompt),
                prompt_complexity=_guess_complexity(prompt),
                best_tier=_model_to_tier(winner_model),
                best_model=winner_model,
                confidence=0.9,  # race winner = high confidence
                ttft_ms=race_meta["ttft"] * 1000,
                latency_ms=race_meta.get("total_s", 0) * 1000,
                success=True,
                tokens_out=race_meta.get("tokens_out", 0),
                source="race",
                timestamp=time.time(),
            )
            samples.append(sample)
            logger.info(
                "[%d/%d] Winner: %s (TTFT=%.0fms)",
                i + 1, len(prompts), winner_model, sample.ttft_ms
            )

        # Brief pause between races to avoid hammering
        if i < len(prompts) - 1:
            time.sleep(0.5)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "a") as f:
            for s in samples:
                f.write(s.to_jsonl() + "\n")
        logger.info("Appended %d race samples to %s", len(samples), output)

    return samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tier_to_complexity(tier: str) -> float:
    """Map tier to approximate complexity (for training input)."""
    mapping = {
        "L0": 0.05, "L1": 0.15, "L2": 0.25, "L3": 0.40,
        "L4": 0.55, "L5": 0.70, "L6": 0.85, "L7": 0.95,
    }
    return mapping.get(tier, 0.3)


def _model_to_tier(model_id: str) -> str:
    """Infer tier from model ID."""
    mid = model_id.lower()
    if "gpt-4o" in mid and "mini" not in mid:
        return "L7"
    if "gpt-4o-mini" in mid:
        return "L5"
    if "gemini" in mid and "flash" in mid:
        return "L4"
    if "mistral-small" in mid:
        return "L3"
    if "qwen3-coder" in mid or "qwen3-8b" in mid:
        return "L3"
    if "deepseek" in mid:
        return "L5"
    if "llama" in mid and "70b" in mid:
        return "L6"
    if "qwen3-4b" in mid:
        return "L2"
    if "qwen3-1.7b" in mid:
        return "L1"
    return "L3"


def _guess_category(prompt: str) -> str:
    """Quick heuristic category from prompt text."""
    p = prompt.lower()
    if any(w in p for w in ["code", "function", "implement", "class ", "def ", "write a ",
                             "script", "regex", "decorator", "trie", "sort", "stack", "queue"]):
        return "code"
    if any(w in p for w in ["debug", "fix", "error", "bug", "wrong", "hangs", "crash",
                             "slow", "fails", "broken"]):
        return "debug"
    if any(w in p for w in ["explain", "analyze", "compare", "difference", "why",
                             "how does", "what is the", "implications", "tradeoff",
                             "characteristics", "works"]):
        return "analyze"
    if any(w in p for w in ["plan", "design", "outline", "architect", "strategy",
                             "steps to", "pipeline"]):
        return "plan"
    if any(w in p for w in ["yes or no", "classify", "is it", "does ", "can ",
                             "is ", "are "]):
        return "classify"
    if any(w in p for w in ["write", "generate", "create", "produce", "draft"]):
        return "generate"
    if any(w in p for w in ["safe", "ethical", "should", "autonomous"]):
        return "safety"
    if any(w in p for w in ["prove", "derive", "theoretical", "complexity of"]):
        return "analyze"
    return "unknown"


def _guess_complexity(prompt: str) -> float:
    """Rough complexity from prompt length and keywords."""
    words = len(prompt.split())
    base = min(words / 200.0, 0.5)  # longer = more complex
    if any(w in prompt.lower() for w in ["step by step", "detailed", "comprehensive"]):
        base += 0.2
    if any(w in prompt.lower() for w in ["simple", "quick", "one word"]):
        base -= 0.2
    return max(0.0, min(1.0, base))


# ---------------------------------------------------------------------------
# Benchmark prompts for racing
# ---------------------------------------------------------------------------

BENCHMARK_PROMPTS = [
    # Classify (L0-L1)
    "Is Python a compiled language? Yes or no.",
    "What color is the sky?",
    # Tool call (L1-L2)
    "What's 234 * 567?",
    "Convert 72 degrees Fahrenheit to Celsius.",
    # Code (L3)
    "Write a Python function to reverse a linked list.",
    "Implement binary search in Rust.",
    # Analyze (L3-L4)
    "Explain the difference between TCP and UDP in networking.",
    "What are the tradeoffs between B-trees and LSM-trees for databases?",
    # Debug (L4)
    "My Python script raises 'RecursionError: maximum recursion depth exceeded'. The function is supposed to calculate Fibonacci numbers. What's wrong?",
    # Plan (L4-L5)
    "Design a microservices architecture for a real-time chat application that needs to handle 1M concurrent users.",
    # Complex reasoning (L5-L7)
    "Compare the computational complexity of transformer self-attention vs state space models. Which scales better for long sequences and why?",
    "Prove that the halting problem is undecidable using a diagonalization argument.",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="CortexRouter training data pipeline")
    sub = parser.add_subparsers(dest="command")

    # Export from audit log
    exp = sub.add_parser("export", help="Export training data from audit log")
    exp.add_argument("--db", type=Path, default=None, help="Path to cortex.db")
    exp.add_argument("--output", "-o", type=Path, required=True, help="Output JSONL path")

    # Generate from racing
    race = sub.add_parser("race", help="Race models to generate training labels")
    race.add_argument("--count", "-n", type=int, default=len(BENCHMARK_PROMPTS),
                      help="Number of prompts to race")
    race.add_argument("--output", "-o", type=Path, required=True, help="Output JSONL path")

    # Combined: export + race
    full = sub.add_parser("full", help="Full pipeline: export audit + race benchmarks")
    full.add_argument("--output", "-o", type=Path, required=True, help="Output JSONL path")

    args = parser.parse_args()

    if args.command == "export":
        from ..config import DB_PATH
        db = args.db or DB_PATH
        n = export_from_audit(db, args.output)
        print(f"Exported {n} samples")

    elif args.command == "race":
        env_path = Path("/Volumes/CORTEX/cortex/bin/.env")
        api_key = ""
        if env_path.exists():
            for line in env_path.read_text().strip().split("\n"):
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("ERROR: No API key found", file=sys.stderr)
            sys.exit(1)

        prompts = BENCHMARK_PROMPTS[:args.count]
        samples = generate_race_samples(api_key, prompts, output=args.output)
        print(f"Generated {len(samples)} race samples")

    elif args.command == "full":
        from ..config import DB_PATH

        # 1. Export audit data
        n_audit = export_from_audit(DB_PATH, args.output)

        # 2. Race benchmarks
        env_path = Path("/Volumes/CORTEX/cortex/bin/.env")
        api_key = ""
        if env_path.exists():
            for line in env_path.read_text().strip().split("\n"):
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")

        n_race = 0
        if api_key:
            samples = generate_race_samples(api_key, BENCHMARK_PROMPTS, output=args.output)
            n_race = len(samples)

        print(f"Total: {n_audit + n_race} samples ({n_audit} audit + {n_race} race)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
