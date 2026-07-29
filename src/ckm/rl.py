"""
CKM Policy Gradient — REINFORCE with safety constraints.

Phase 4 of training: after imitation learning bootstraps the policy,
we fine-tune with REINFORCE in the boot simulator.

Key differences from standard REINFORCE:
  1. Safety head has veto power — if safety_score > threshold, action is blocked
  2. Reward is shaped (not sparse) — per-step reward from simulator
  3. Baseline subtraction — reduces variance
  4. Episodes are short (5-30 steps) — low variance naturally

Training loop:
  1. Run episode with current policy in simulator
  2. Compute returns (discounted cumulative reward)
  3. Subtract baseline (running mean of returns)
  4. Policy gradient: ∇J = Σ log_prob(a|s) × (R - baseline)
  5. Update policy
  6. Safety constraint: if safety head vetoes, skip that action's gradient

The safety head is FROZEN during RL — it's a hard constraint, not a learned one.
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.ckm.rl")

_torch = None


def _ensure_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# ---------------------------------------------------------------------------
# RL Configuration
# ---------------------------------------------------------------------------

@dataclass
class RLConfig:
    """Policy gradient training configuration."""
    n_episodes: int = 2000
    batch_size: int = 16         # episodes per update
    learning_rate: float = 1e-4
    gamma: float = 0.99          # discount factor
    entropy_coeff: float = 0.01  # encourage exploration
    safety_threshold: float = 0.5
    max_grad_norm: float = 0.5
    baseline_momentum: float = 0.99
    eval_every: int = 50         # episodes between eval
    eval_episodes: int = 50
    fault_probability: float = 0.3
    patience: int = 300          # episodes without improvement
    target_success_rate: float = 0.99


# ---------------------------------------------------------------------------
# REINFORCE with baseline
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """One episode's trajectory data for policy gradient."""
    states: list[list[float]]
    actions: list  # Action objects
    log_probs: list[float]
    rewards: list[float]
    values: list[float]  # baseline predictions
    outcome: str
    total_reward: float


def collect_trajectory(policy, simulator, safety_head=None, safety_threshold: float = 0.5):
    """
    Collect one trajectory using the policy in the simulator.

    If safety_head is provided, it vetoes dangerous actions.
    """
    torch = _ensure_torch()
    from .simulator import Action, Verb, Target

    state = simulator.reset()
    states = []
    actions = []
    log_probs = []
    rewards = []
    done = False

    while not done:
        state_vec = state.to_vector()
        states.append(state_vec)

        # Get action from policy
        action, log_prob, confidence = policy.act_with_logprob(state_vec)

        # Safety veto
        if safety_head is not None:
            action_vec = action.to_vector()
            device = next(safety_head.parameters()).device
            s_t = torch.tensor([state_vec], dtype=torch.float32, device=device)
            a_t = torch.tensor([action_vec], dtype=torch.float32, device=device)
            with torch.no_grad():
                danger_score = safety_head(s_t, a_t)[0, 0].item()

            if danger_score > safety_threshold:
                # Veto: replace with DENY action
                action = Action(Verb.DENY, Target.SYSTEM)
                log_prob = torch.tensor(0.0)  # don't update on vetoed actions

        actions.append(action)
        log_probs.append(log_prob)

        # Step simulator
        state, reward, done, info = simulator.step(action)
        rewards.append(reward)

    outcome = info.get("reason", "unknown")
    total_reward = sum(rewards)

    return Trajectory(
        states=states,
        actions=actions,
        log_probs=log_probs,
        rewards=rewards,
        values=[0.0] * len(rewards),  # filled by baseline
        outcome=outcome,
        total_reward=total_reward,
    )


def compute_returns(rewards: list[float], gamma: float = 0.99) -> list[float]:
    """Compute discounted returns from rewards."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_rl(
    policy,
    output_dir: str = "/tmp/cortex-train/rl",
    config: Optional[RLConfig] = None,
    safety_head=None,
    device: str = "cpu",
) -> dict:
    """
    Train policy with REINFORCE in boot simulator.

    Args:
        policy: initialized policy network (from imitation learning)
        output_dir: checkpoint directory
        config: RL hyperparameters
        safety_head: frozen safety head for veto
        device: compute device

    Returns: training stats
    """
    torch = _ensure_torch()
    from .simulator import (
        CortexBootSimulator, random_hardware, FAULT_LIBRARY,
        expert_policy as expert_fn,
    )

    cfg = config or RLConfig()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    policy = policy.to(device)
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    # Running baseline
    baseline = 0.0
    best_success_rate = 0.0
    patience_counter = 0

    # Stats
    episode_rewards = []
    episode_outcomes = []
    eval_history = []

    t0 = time.time()
    episode_idx = 0

    while episode_idx < cfg.n_episodes:
        # Collect batch of trajectories
        batch_trajectories = []
        for _ in range(cfg.batch_size):
            hw = random_hardware()
            faults = []
            if random.random() < cfg.fault_probability:
                fault_fn = random.choice(FAULT_LIBRARY)
                faults = [fault_fn()]

            sim = CortexBootSimulator(
                hardware=hw, faults=faults, seed=episode_idx + _
            )
            traj = collect_trajectory(policy, sim, safety_head, cfg.safety_threshold)
            batch_trajectories.append(traj)
            episode_rewards.append(traj.total_reward)
            episode_outcomes.append(traj.outcome)
            episode_idx += 1

        # Compute policy gradient for batch
        policy_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        n_steps = 0

        for traj in batch_trajectories:
            returns = compute_returns(traj.rewards, cfg.gamma)

            for t, (state_vec, log_prob, R) in enumerate(
                zip(traj.states, traj.log_probs, returns)
            ):
                if not isinstance(log_prob, torch.Tensor):
                    continue  # Skip vetoed actions

                advantage = R - baseline
                policy_loss -= log_prob * advantage
                n_steps += 1

            # Update baseline
            baseline = cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * traj.total_reward

        if n_steps > 0:
            policy_loss = policy_loss / n_steps

            # Add entropy bonus (encourage exploration)
            # Approximate entropy from recent log_probs
            total_loss = policy_loss - cfg.entropy_coeff * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

        # Periodic evaluation
        if episode_idx % cfg.eval_every < cfg.batch_size:
            eval_stats = evaluate_policy(policy, n_episodes=cfg.eval_episodes,
                                         fault_prob=cfg.fault_probability)
            eval_history.append({
                "episode": episode_idx,
                "success_rate": eval_stats["success_rate"],
                "avg_reward": eval_stats["avg_reward"],
                "avg_time_ms": eval_stats["avg_time_ms"],
            })

            success_rate = eval_stats["success_rate"]
            logger.info(
                "Episode %d: success=%.1f%% reward=%.2f time=%.0fms",
                episode_idx, success_rate * 100,
                eval_stats["avg_reward"], eval_stats["avg_time_ms"],
            )

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                patience_counter = 0
                torch.save(policy.state_dict(), os.path.join(output_dir, "policy_rl_best.pt"))
            else:
                patience_counter += cfg.eval_every

            # Convergence check
            if success_rate >= cfg.target_success_rate:
                logger.info("Target success rate reached: %.1f%%", success_rate * 100)
                break

            if patience_counter >= cfg.patience:
                logger.info("Patience exhausted at episode %d", episode_idx)
                break

    elapsed = time.time() - t0

    # Save final
    torch.save(policy.state_dict(), os.path.join(output_dir, "policy_rl_final.pt"))

    # Final eval
    final_eval = evaluate_policy(policy, n_episodes=200, fault_prob=cfg.fault_probability)

    metadata = {
        "episodes_trained": episode_idx,
        "best_success_rate": best_success_rate,
        "final_success_rate": final_eval["success_rate"],
        "final_avg_reward": final_eval["avg_reward"],
        "training_time_s": elapsed,
        "eval_history": eval_history,
    }
    with open(os.path.join(output_dir, "rl_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "RL training complete: %d episodes, success=%.1f%%, time=%.1fs",
        episode_idx, final_eval["success_rate"] * 100, elapsed,
    )

    return {
        "episodes": episode_idx,
        "best_success_rate": best_success_rate,
        "final_success_rate": final_eval["success_rate"],
        "final_avg_reward": final_eval["avg_reward"],
        "elapsed_s": elapsed,
        "output_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(policy, n_episodes: int = 100, fault_prob: float = 0.3) -> dict:
    """Evaluate policy in simulator (no gradient)."""
    from .simulator import CortexBootSimulator, random_hardware, FAULT_LIBRARY

    policy.eval()
    outcomes = {"success": 0, "timeout": 0, "crash": 0, "violation": 0}
    rewards = []
    times = []
    steps_list = []

    for i in range(n_episodes):
        hw = random_hardware()
        faults = []
        if random.random() < fault_prob:
            fault_fn = random.choice(FAULT_LIBRARY)
            faults = [fault_fn()]

        sim = CortexBootSimulator(hardware=hw, faults=faults, seed=10000 + i)
        state = sim.reset()
        total_reward = 0.0
        done = False

        while not done:
            state_vec = state.to_vector()
            action = policy.act(state_vec, deterministic=True)
            state, reward, done, info = sim.step(action)
            total_reward += reward

        outcome = info.get("reason", "unknown")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        rewards.append(total_reward)
        times.append(sim.state.time_ms)
        steps_list.append(sim.step_count)

    policy.train()

    return {
        "success_rate": outcomes["success"] / n_episodes,
        "outcomes": outcomes,
        "avg_reward": sum(rewards) / len(rewards),
        "avg_time_ms": sum(times) / len(times),
        "avg_steps": sum(steps_list) / len(steps_list),
    }
