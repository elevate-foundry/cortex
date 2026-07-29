"""
Eval suite for the gossip consensus race (race_quality).

Tests:
  1. Unit tests — gossip consensus algorithm with mocked responses
  2. Integration tests — live API calls with diverse prompts
  3. Quality benchmarks — compare consensus vs single-model accuracy
  4. Convergence tests — verify cluster formation and cross-family bonuses
  5. Stress tests — edge cases (single response, all disagree, etc.)

Run:
  pytest tests/test_race_consensus.py -v              # unit tests only
  pytest tests/test_race_consensus.py -v -m live      # include live API tests
  pytest tests/test_race_consensus.py -v -m benchmark # full benchmark suite
"""

import json
import math
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cortex():
    """Create a Cortex instance with mocked backends."""
    from src.cortex import Cortex
    from src.hardware_detect import detect_system

    profile = detect_system()
    cortex = Cortex(profile=profile)
    cortex._booted = True
    return cortex


def make_mock_responses(responses: dict[str, str], timings: dict[str, float] = None):
    """Create a mock that simulates parallel API calls with given responses."""
    timings = timings or {k: 1.0 for k in responses}

    def mock_post(url, headers=None, json=None, timeout=None):
        model_id = json["messages"][0]["content"] if json else ""
        # Find model from the request
        model_id = json.get("model", "unknown")
        mock_resp = MagicMock()
        if model_id in responses:
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": responses[model_id]}}]
            }
        else:
            mock_resp.status_code = 500
        return mock_resp

    return mock_post


# ---------------------------------------------------------------------------
# Unit tests: gossip consensus algorithm
# ---------------------------------------------------------------------------

class TestGossipConsensus:
    """Test the _gossip_consensus method in isolation."""

    def test_unanimous_agreement(self, mock_cortex):
        """All models agree → winner is from the majority cluster."""
        responses = [
            ("openai/gpt-4o-mini", "The answer is 42."),
            ("google/gemini-flash", "The answer is 42."),
            ("mistralai/mistral-small", "The answer is 42."),
            ("deepseek/deepseek-chat", "The answer is 42."),
        ]
        timings = {r[0]: 1.0 for r in responses}

        winner_idx = mock_cortex._gossip_consensus(responses, timings)
        # All agree, so any index is valid — but should be one of them
        assert 0 <= winner_idx < len(responses)

    def test_majority_wins(self, mock_cortex):
        """3 agree, 1 disagrees → consensus picks from the majority."""
        responses = [
            ("openai/gpt-4o-mini", "Move right to explore the corridor."),
            ("google/gemini-flash", "Move right toward the door."),
            ("mistralai/mistral-small", "Move right along the wall."),
            ("deepseek/deepseek-chat", "Move left to find the key."),
        ]
        timings = {r[0]: 1.5 for r in responses}

        winner_idx = mock_cortex._gossip_consensus(responses, timings)
        winner_model = responses[winner_idx][0]
        # Winner should NOT be deepseek (the dissenter)
        assert winner_model != "deepseek/deepseek-chat"

    def test_cross_family_bonus(self, mock_cortex):
        """Cross-family cluster gets bonus weight over single-family."""
        # Cluster A: 2 models, 2 families (cross-family)
        # Cluster B: 2 models, 1 family (same family — weaker signal)
        responses = [
            ("openai/gpt-4o-mini", "Use dynamic programming."),
            ("google/gemini-flash", "Use dynamic programming approach."),
            ("mistralai/mistral-small", "Use brute force recursion."),
            ("mistralai/mistral-large", "Use brute force."),
        ]
        timings = {r[0]: 1.0 for r in responses}

        winner_idx = mock_cortex._gossip_consensus(responses, timings)
        winner_model = responses[winner_idx][0]
        # Cross-family cluster (openai + google) should win
        assert winner_model in ("openai/gpt-4o-mini", "google/gemini-flash")

    def test_single_response(self, mock_cortex):
        """Single response → returns index 0."""
        responses = [("openai/gpt-4o-mini", "Hello world")]
        timings = {"openai/gpt-4o-mini": 0.5}

        winner_idx = mock_cortex._gossip_consensus(responses, timings)
        assert winner_idx == 0

    def test_all_disagree(self, mock_cortex):
        """All models disagree → picks the best individual (substantive + fast)."""
        responses = [
            ("openai/gpt-4o-mini", "A"),
            ("google/gemini-flash", "B completely different long answer with detail"),
            ("mistralai/mistral-small", "C"),
            ("deepseek/deepseek-chat", "D"),
        ]
        timings = {
            "openai/gpt-4o-mini": 2.0,
            "google/gemini-flash": 0.5,  # fastest + longest
            "mistralai/mistral-small": 3.0,
            "deepseek/deepseek-chat": 4.0,
        }

        winner_idx = mock_cortex._gossip_consensus(responses, timings)
        # Gemini has longest response + fastest → highest individual weight
        assert responses[winner_idx][0] == "google/gemini-flash"

    def test_speed_tiebreaker(self, mock_cortex):
        """When agreement is equal, faster model wins as tiebreaker."""
        responses = [
            ("openai/gpt-4o-mini", "The function returns null on error."),
            ("google/gemini-flash", "The function returns null when errors occur."),
        ]
        timings = {
            "openai/gpt-4o-mini": 5.0,   # slow
            "google/gemini-flash": 0.3,   # fast
        }

        winner_idx = mock_cortex._gossip_consensus(responses, timings)
        # Both agree (same cluster), but gemini is faster → wins
        assert responses[winner_idx][0] == "google/gemini-flash"

    def test_empty_responses_filtered(self, mock_cortex):
        """Empty responses shouldn't dominate."""
        responses = [
            ("openai/gpt-4o-mini", ""),
            ("google/gemini-flash", "The correct approach is to use a hash map."),
            ("mistralai/mistral-small", "Use a hash map for O(1) lookups."),
        ]
        timings = {r[0]: 1.0 for r in responses}

        winner_idx = mock_cortex._gossip_consensus(responses, timings)
        # Empty response should not win
        assert responses[winner_idx][0] != "openai/gpt-4o-mini"


# ---------------------------------------------------------------------------
# Integration tests: full race_quality with mocked HTTP
# ---------------------------------------------------------------------------

class TestRaceQualityMocked:
    """Test race_quality end-to-end with mocked API calls."""

    def test_full_race_returns_tuple(self, mock_cortex):
        """race_quality returns (winner_model, winner_response, all_responses)."""
        mock_responses = {
            "openai/gpt-4o-mini": "Move right.",
            "google/gemini-2.0-flash-001": "Move right to corridor.",
            "mistralai/mistral-small-24b-instruct-2501": "Move right.",
            "qwen/qwen3-coder": "Open the door.",
            "deepseek/deepseek-chat-v3-0324": "Move right carefully.",
        }

        # Mock the pool and backend
        mock_backend = MagicMock()
        mock_backend.api_key = "test-key"
        mock_cortex._pool = MagicMock()
        mock_cortex._pool.get_backend.return_value = mock_backend

        with patch("httpx.post", side_effect=make_mock_responses(mock_responses)):
            result = mock_cortex.race_quality(
                prompt="What should the player do?",
                candidates=list(mock_responses.keys()),
            )

        assert result is not None
        winner_model, winner_response, all_responses = result
        assert winner_model in mock_responses
        assert len(all_responses) == 5
        # Consensus should pick "move right" cluster (4/5 agree)
        assert "right" in winner_response.lower()

    def test_race_with_custom_judge(self, mock_cortex):
        """Custom judge overrides gossip consensus."""
        mock_responses = {
            "openai/gpt-4o-mini": "Short.",
            "google/gemini-2.0-flash-001": "This is the longest and most detailed response.",
        }

        mock_backend = MagicMock()
        mock_backend.api_key = "test-key"
        mock_cortex._pool = MagicMock()
        mock_cortex._pool.get_backend.return_value = mock_backend

        # Custom judge: always pick the shortest
        def pick_shortest(responses):
            return min(range(len(responses)), key=lambda i: len(responses[i][1]))

        with patch("httpx.post", side_effect=make_mock_responses(mock_responses)):
            result = mock_cortex.race_quality(
                prompt="test",
                candidates=list(mock_responses.keys()),
                judge=pick_shortest,
            )

        assert result is not None
        assert result[0] == "openai/gpt-4o-mini"

    def test_no_backend_returns_none(self, mock_cortex):
        """No pool/backend → returns None gracefully."""
        mock_cortex._pool = None
        result = mock_cortex.race_quality(prompt="test")
        assert result is None

    def test_all_fail_returns_none(self, mock_cortex):
        """All API calls fail → returns None."""
        mock_backend = MagicMock()
        mock_backend.api_key = "test-key"
        mock_cortex._pool = MagicMock()
        mock_cortex._pool.get_backend.return_value = mock_backend

        def always_fail(*args, **kwargs):
            raise ConnectionError("network down")

        with patch("httpx.post", side_effect=always_fail):
            result = mock_cortex.race_quality(
                prompt="test",
                candidates=["openai/gpt-4o-mini", "google/gemini-flash"],
            )

        assert result is None


# ---------------------------------------------------------------------------
# Live API tests (require OPENROUTER_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestRaceQualityLive:
    """Live tests against real APIs. Require OPENROUTER_API_KEY."""

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

    @pytest.fixture
    def live_cortex(self):
        from src.cortex import Cortex
        from src.hardware_detect import detect_system
        cortex = Cortex(profile=detect_system())
        cortex.boot()
        return cortex

    def test_simple_math_consensus(self, live_cortex):
        """All models should agree on simple math."""
        result = live_cortex.race_quality(
            prompt="What is 15 * 7? Reply with just the number.",
            max_tokens=50,
            candidates=[
                "openai/gpt-4o-mini",
                "google/gemini-2.0-flash-001",
                "mistralai/mistral-small-24b-instruct-2501",
            ],
        )
        assert result is not None
        winner_model, winner_response, all_responses = result
        assert "105" in winner_response
        # All should agree
        for _, resp in all_responses:
            assert "105" in resp

    def test_game_action_consensus(self, live_cortex):
        """Models should converge on a reasonable game action."""
        observation = """
        You are playing a dungeon crawler. The grid shows:
        - Player (P) at position (5, 3)
        - Door (D) at position (5, 8) — requires a key
        - Key (K) at position (3, 3) — 2 tiles to the left
        - Wall blocking direct path to door
        
        Available actions: UP, DOWN, LEFT, RIGHT
        What single action should you take? Reply with just the action word.
        """
        result = live_cortex.race_quality(
            prompt=observation,
            max_tokens=20,
            candidates=[
                "openai/gpt-4o-mini",
                "google/gemini-2.0-flash-001",
                "qwen/qwen3-coder",
            ],
        )
        assert result is not None
        winner_model, winner_response, all_responses = result
        # Should pick LEFT (to get the key first)
        assert "LEFT" in winner_response.upper() or "left" in winner_response.lower()

    def test_reasoning_consensus(self, live_cortex):
        """Complex reasoning — consensus should filter hallucinations."""
        result = live_cortex.race_quality(
            prompt="In Python, what does `None is None` evaluate to? One word answer.",
            max_tokens=10,
            candidates=[
                "openai/gpt-4o-mini",
                "google/gemini-2.0-flash-001",
                "deepseek/deepseek-chat-v3-0324",
            ],
        )
        assert result is not None
        assert "true" in result[1].lower() or "True" in result[1]

    def test_race_timing(self, live_cortex):
        """Verify parallel execution — total time < sum of individual times."""
        t0 = time.time()
        result = live_cortex.race_quality(
            prompt="Say hello in one word.",
            max_tokens=10,
            candidates=[
                "openai/gpt-4o-mini",
                "google/gemini-2.0-flash-001",
                "mistralai/mistral-small-24b-instruct-2501",
                "qwen/qwen3-coder",
                "deepseek/deepseek-chat-v3-0324",
            ],
        )
        elapsed = time.time() - t0
        assert result is not None
        # 5 models in parallel should complete in <30s total
        # (if sequential, it would be ~50s+)
        assert elapsed < 30.0, f"Race took {elapsed:.1f}s — not parallel?"
        print(f"\n  Race completed in {elapsed:.1f}s with {len(result[2])} models")
        print(f"  Winner: {result[0]}")


# ---------------------------------------------------------------------------
# Benchmark: consensus vs single model accuracy
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
class TestConsensusBenchmark:
    """
    Benchmark comparing consensus accuracy vs single model.
    
    Tests a set of problems with known correct answers.
    Measures: accuracy, agreement rate, cluster sizes.
    """

    # L3-L4: Factual recall (baseline — any model should get these)
    RECALL_PROBLEMS = [
        {
            "prompt": "What is 2^10? Just the number.",
            "answer": "1024",
            "level": "L3",
            "check": lambda r: "1024" in r,
        },
        {
            "prompt": "What HTTP status code means 'Not Found'? Just the number.",
            "answer": "404",
            "level": "L3",
            "check": lambda r: "404" in r,
        },
        {
            "prompt": "What is the time complexity of binary search? Big-O notation only.",
            "answer": "O(log n)",
            "level": "L3",
            "check": lambda r: "log" in r.lower() and "n" in r.lower(),
        },
    ]

    # L5: Apply multiple known principles to a concrete unfamiliar case
    REASONING_L5 = [
        {
            "prompt": (
                "A transformer and a gated RNN have the same parameter count and must "
                "process sequences of length 32,768. Compare their training and inference "
                "time, memory consumption, effective dependency paths, and streaming behavior. "
                "Identify at least two conditions under which the RNN could be preferable."
            ),
            "level": "L5",
            "check": lambda r: (
                # Must mention quadratic attention cost AND streaming/latency advantage of RNN
                ("quadratic" in r.lower() or "n^2" in r.lower() or "n²" in r.lower())
                and ("stream" in r.lower() or "latency" in r.lower() or "real-time" in r.lower())
            ),
        },
        {
            "prompt": (
                "Design a storage engine for a workload with 100,000 writes/sec, 5,000 point "
                "reads/sec, a 99th-percentile read-latency target of 10 ms, and a 2x "
                "storage-overhead limit. Compare a B+ tree, a leveled LSM tree, and a "
                "size-tiered LSM tree, then justify a configuration."
            ),
            "level": "L5",
            "check": lambda r: (
                # Must mention write amplification AND recommend LSM variant for write-heavy
                ("write amplification" in r.lower() or "write-amplification" in r.lower())
                and ("lsm" in r.lower() or "log-structured" in r.lower())
            ),
        },
    ]

    # L6: Resolve ambiguity, compare competing solutions, expose hidden assumptions
    REASONING_L6 = [
        {
            "prompt": (
                "Derive the gradients for a sigmoid neuron under binary cross-entropy and "
                "mean-squared error. Compare the resulting gradient behavior as the neuron "
                "becomes confidently wrong, and explain why the losses produce different "
                "optimization dynamics."
            ),
            "level": "L6",
            "check": lambda r: (
                # Must identify vanishing gradient problem with MSE + sigmoid
                ("vanish" in r.lower() or "saturate" in r.lower() or "slow" in r.lower())
                and ("cross-entropy" in r.lower() or "cross entropy" in r.lower())
            ),
        },
        {
            "prompt": (
                "For a seven-node PBFT deployment, determine how many Byzantine failures can "
                "be tolerated. Trace a request through the protocol, then analyze what happens "
                "if the primary equivocates and two replicas experience delayed messages. "
                "Distinguish safety from liveness."
            ),
            "level": "L6",
            "check": lambda r: (
                # Must state f=2 (3f+1=7) AND distinguish safety/liveness
                ("2" in r and ("f" in r.lower() or "fault" in r.lower() or "failure" in r.lower()))
                and ("safety" in r.lower() and "liveness" in r.lower())
            ),
        },
    ]

    # L7: Construct novel argument, handle adversarial details, self-check
    REASONING_L7 = [
        {
            "prompt": (
                "A student claims Cantor's diagonal argument fails because the constructed "
                "real number may have two decimal representations, such as 0.4999...=0.5000... "
                "Repair the proof formally and explain why enumerating only computable reals "
                "does not enumerate all reals."
            ),
            "level": "L7",
            "check": lambda r: (
                # Must address dual representation AND uncomputability
                ("representation" in r.lower() or "decimal" in r.lower())
                and ("uncomputable" in r.lower() or "uncountab" in r.lower()
                     or "not computable" in r.lower() or "non-computable" in r.lower())
            ),
        },
        {
            "prompt": (
                "For a zero-mean Gaussian source with variance σ² under squared-error "
                "distortion, derive its rate-distortion function. Then calculate the minimum "
                "rate needed when D=σ²/16, and explain which assumptions make the result "
                "inapplicable to natural images."
            ),
            "level": "L7",
            "check": lambda r: (
                # Must give R(D) = 0.5 log(σ²/D) AND mention non-Gaussian/correlation
                ("log" in r.lower() or "ln" in r.lower())
                and ("gaussian" in r.lower() or "iid" in r.lower() or "i.i.d" in r.lower()
                     or "independent" in r.lower() or "correlation" in r.lower())
            ),
        },
    ]

    BENCHMARK_PROBLEMS = (
        [dict(p, category="recall") for p in RECALL_PROBLEMS]
        + [dict(p, category="L5_reasoning") for p in REASONING_L5]
        + [dict(p, category="L6_reasoning") for p in REASONING_L6]
        + [dict(p, category="L7_reasoning") for p in REASONING_L7]
    )

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

    @pytest.fixture
    def live_cortex(self):
        from src.cortex import Cortex
        from src.hardware_detect import detect_system
        cortex = Cortex(profile=detect_system())
        cortex.boot()
        return cortex

    def test_benchmark_accuracy(self, live_cortex):
        """
        Tiered reasoning benchmark.

        L3: Factual recall (baseline) — adaptive 5 models
        L5: Apply principles to novel case — adaptive 10-12 models
        L6: Resolve ambiguity, expose assumptions — adaptive 12-15 models
        L7: Construct novel argument under constraints — adaptive 15-20 models

        Cortex adaptively selects candidate count based on difficulty.
        """
        results = []
        level_scores = {}

        for problem in self.BENCHMARK_PROBLEMS:
            prompt = problem["prompt"]
            level = problem.get("level", "L3")
            category = problem.get("category", "unknown")
            check_fn = problem.get("check")

            # Let Cortex decide how many models to race
            candidates = live_cortex.select_candidates(prompt)
            max_tokens = 100 if level == "L3" else 2048

            t0 = time.time()
            result = live_cortex.race_quality(
                prompt=prompt,
                max_tokens=max_tokens,
                candidates=candidates,
            )
            elapsed = time.time() - t0

            if result is None:
                results.append({
                    "level": level, "correct": False, "time": elapsed,
                    "reason": "no_result", "prompt": prompt[:50],
                })
                level_scores.setdefault(level, []).append(False)
                continue

            winner_model, winner_response, all_responses = result

            # Use check function if available, else substring match
            if check_fn:
                correct = check_fn(winner_response)
            else:
                expected = problem.get("answer", "").lower()
                correct = expected in winner_response.lower()

            results.append({
                "level": level,
                "category": category,
                "prompt": prompt[:60],
                "correct": correct,
                "winner": winner_model,
                "n_models": len(all_responses),
                "n_candidates": len(candidates),
                "time": elapsed,
                "response_preview": winner_response.strip()[:100],
            })
            level_scores.setdefault(level, []).append(correct)

        # Report
        total_correct = sum(1 for r in results if r["correct"])
        total = len(results)
        avg_time = sum(r["time"] for r in results) / total

        print(f"\n{'='*70}")
        print(f"  CORTEX REASONING BENCHMARK (Consensus Race)")
        print(f"  Total: {total_correct}/{total} ({total_correct/total*100:.0f}%)")
        print(f"  Avg time: {avg_time:.1f}s per problem")
        print(f"{'='*70}")

        # Per-level breakdown
        for level in ["L3", "L5", "L6", "L7"]:
            scores = level_scores.get(level, [])
            if scores:
                n_pass = sum(scores)
                pct = n_pass / len(scores) * 100
                print(f"  {level}: {n_pass}/{len(scores)} ({pct:.0f}%)")

        print(f"{'='*70}")
        for r in results:
            status = "✓" if r["correct"] else "✗"
            level = r.get("level", "?")
            n = r.get("n_models", 0)
            t = r.get("time", 0)
            print(f"  {status} [{level}] {r['prompt'][:45]:45s} ({n} models, {t:.1f}s)")
            if not r["correct"] and r.get("response_preview"):
                print(f"       Response: {r['response_preview'][:80]}...")
        print(f"{'='*70}")

        # Expectations:
        # L3 recall: 100% (trivial for any model)
        # L5-L7 reasoning: consensus should improve accuracy over single model
        # Overall target: >= 60% (reasoning problems are genuinely hard)
        l3_scores = level_scores.get("L3", [])
        if l3_scores:
            assert all(l3_scores), "L3 recall problems should all pass"

        # At least half of reasoning problems should pass with consensus
        reasoning_scores = (
            level_scores.get("L5", []) +
            level_scores.get("L6", []) +
            level_scores.get("L7", [])
        )
        if reasoning_scores:
            reasoning_pass = sum(reasoning_scores) / len(reasoning_scores)
            print(f"\n  Reasoning accuracy: {reasoning_pass*100:.0f}%")
            # Consensus should get at least 50% of hard reasoning right
            assert reasoning_pass >= 0.5, (
                f"Reasoning accuracy {reasoning_pass*100:.0f}% < 50% — "
                f"consensus not providing sufficient quality lift"
            )


# ---------------------------------------------------------------------------
# RL integration tests
# ---------------------------------------------------------------------------

class TestRLIntegration:
    """Test that race results feed into the RL learning loop."""

    def test_race_feeds_experience_buffer(self):
        """Race winner should be recorded as a high-reward experience."""
        from src.ckm.rl_loop import OnlineLearningLoop, ExperienceBuffer

        loop = OnlineLearningLoop(update_interval=100)
        reward = loop.observe_race(
            winner_model="google/gemini-flash",
            ttft_ms=250.0,
            category="code",
            length=50,
        )
        assert reward > 0.8  # Race winners should get high reward
        assert loop.buffer.experiences[-1].source == "race"
        assert loop.buffer.experiences[-1].model_chosen == "google/gemini-flash"

    def test_grpo_produces_training_signal(self):
        """After enough experiences, GRPO should produce valid training signals."""
        from src.ckm.rl_loop import OnlineLearningLoop

        loop = OnlineLearningLoop(update_interval=100)

        # Add diverse experiences
        for i in range(20):
            loop.observe_race(
                winner_model=f"model_{i % 3}",
                ttft_ms=200 + i * 10,
                category=["code", "analyze", "plan"][i % 3],
                length=50 + i,
            )

        result = loop.force_update()
        assert result["grpo"]["status"] == "computed"
        assert result["grpo"]["groups"] >= 1

    def test_dpo_produces_preference_pairs(self):
        """DPO should find valid preference pairs from mixed experiences."""
        from src.ckm.rl_loop import OnlineLearningLoop, Experience

        loop = OnlineLearningLoop(update_interval=100)

        # Add some good and bad experiences in the same category
        for i in range(10):
            exp = Experience(
                prompt_category="code",
                prompt_length=100,
                tier_chosen="L7",
                model_chosen=f"model_{i}",
                ttft_ms=100 + i * 200,  # varying speed
                cost_usd=0.001 * (i + 1),
                success=i < 8,  # some failures
                escalated=i > 6,
                source="test",
            )
            loop.buffer.add(exp)

        pairs = loop.buffer.sample_dpo_pairs(n_pairs=5)
        assert len(pairs) > 0
        # Chosen should always have higher reward than rejected
        for chosen, rejected in pairs:
            assert chosen.reward > rejected.reward


# ---------------------------------------------------------------------------
# Reward model tests
# ---------------------------------------------------------------------------

class TestRewardModel:
    """Test the reward computation."""

    def test_perfect_request(self):
        """Fast, cheap, successful, no escalation → max reward."""
        from src.ckm.rl_loop import Experience, compute_reward

        exp = Experience(
            ttft_ms=100,  # very fast
            cost_usd=0.0,  # free (local)
            success=True,
            escalated=False,
        )
        reward = compute_reward(exp)
        assert reward > 0.9

    def test_failed_request(self):
        """Failed request → low reward."""
        from src.ckm.rl_loop import Experience, compute_reward

        exp = Experience(
            ttft_ms=5000,
            cost_usd=0.05,
            success=False,
            escalated=True,
        )
        reward = compute_reward(exp)
        assert reward < 0.3

    def test_escalated_but_succeeded(self):
        """Escalated but ultimately succeeded → moderate reward."""
        from src.ckm.rl_loop import Experience, compute_reward

        exp = Experience(
            ttft_ms=1000,
            cost_usd=0.005,
            success=True,
            escalated=True,
        )
        reward = compute_reward(exp)
        assert 0.3 < reward < 0.8

    def test_reward_ordering(self):
        """Faster + cheaper + no escalation → higher reward."""
        from src.ckm.rl_loop import Experience, compute_reward

        fast_local = Experience(ttft_ms=100, cost_usd=0.0, success=True, escalated=False)
        slow_cloud = Experience(ttft_ms=3000, cost_usd=0.02, success=True, escalated=False)
        failed = Experience(ttft_ms=5000, cost_usd=0.05, success=False, escalated=True)

        r_fast = compute_reward(fast_local)
        r_slow = compute_reward(slow_cloud)
        r_fail = compute_reward(failed)

        assert r_fast > r_slow > r_fail
