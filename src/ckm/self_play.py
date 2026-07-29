"""
CKM Self-Play — three-player game for robustness hardening.

Phase 5 of training: after RL converges, we harden the policy via self-play.

Three-player game:
  Proposer:  selects boot actions (the actual deployed model)
  Verifier:  validates proposed actions (runtime safety checker)
  Adversary: injects novel faults to break the proposer

Training dynamics:
  1. Adversary corrupts simulator state (novel faults)
  2. Proposer selects recovery actions
  3. Verifier checks if actions are valid/safe
  4. Outcomes determine rewards for all three

Convergence: adversary can't find new failure modes that proposer can't handle.

Output artifacts:
  - CKM-Propose: deployed as the boot controller
  - CKM-Verify: deployed as runtime safety validator
  - CKM-Adversary: used only during training (generates hard examples)
"""

import copy
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.ckm.self_play")

_torch = None


def _ensure_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# ---------------------------------------------------------------------------
# Adversary Network
# ---------------------------------------------------------------------------

def build_adversary(state_dim: int = 24, hidden_dim: int = 128, n_fault_types: int = 8):
    """
    Build adversary network that learns to inject faults.

    The adversary outputs:
      - fault_type: which fault to inject (categorical)
      - trigger_time: when to inject (continuous, normalized)
      - intensity: how severe (continuous, [0, 1])
    """
    torch = _ensure_torch()
    nn = torch.nn

    class AdversaryNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.fault_head = nn.Linear(hidden_dim, n_fault_types)
            self.timing_head = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.intensity_head = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, state_vec):
            h = self.encoder(state_vec)
            return (
                self.fault_head(h),
                self.timing_head(h),
                self.intensity_head(h),
            )

        def generate_fault(self, state_vec):
            """Generate a fault configuration from state."""
            torch = _ensure_torch()
            from .simulator import (
                GPUDisappearFault, RAMPressureFault, DiskFullFault,
                ServiceOOMFault, NetworkDownFault, CorruptedCacheFault,
                PortConflictFault, Fault,
            )

            device = next(self.parameters()).device
            s = torch.tensor([state_vec], dtype=torch.float32, device=device)

            with torch.no_grad():
                fault_logits, timing, intensity = self.forward(s)

            # Sample fault type
            fault_dist = torch.distributions.Categorical(logits=fault_logits[0])
            fault_idx = fault_dist.sample().item()
            trigger_ms = int(timing[0, 0].item() * 5000)  # 0-5000ms
            _intensity = intensity[0, 0].item()

            # Map to fault object
            fault_map = [
                lambda t: GPUDisappearFault("adv_gpu", trigger_ms=t),
                lambda t: RAMPressureFault("adv_ram", trigger_ms=t),
                lambda t: DiskFullFault("adv_disk", trigger_ms=t),
                lambda t: ServiceOOMFault("adv_svc_oom", trigger_ms=t, service="inference"),
                lambda t: ServiceOOMFault("adv_api_oom", trigger_ms=t, service="api"),
                lambda t: NetworkDownFault("adv_net", trigger_ms=t),
                lambda t: CorruptedCacheFault("adv_cache", trigger_ms=t),
                lambda t: PortConflictFault("adv_port", trigger_ms=t),
            ]

            fault_fn = fault_map[fault_idx % len(fault_map)]
            return fault_fn(trigger_ms), fault_dist.log_prob(torch.tensor(fault_idx))

    return AdversaryNetwork()


# ---------------------------------------------------------------------------
# Verifier Network
# ---------------------------------------------------------------------------

def build_verifier(state_dim: int = 24, action_dim: int = 26, hidden_dim: int = 128):
    """
    Build verifier network that validates (state, action) pairs.

    Output: validity score [0, 1]
      1.0 = action is valid and will succeed
      0.0 = action is invalid or will fail
    """
    torch = _ensure_torch()
    nn = torch.nn

    class VerifierNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, state_vec, action_vec):
            combined = _ensure_torch().cat([state_vec, action_vec], dim=-1)
            return self.net(combined)

        def verify(self, state_vec, action_vec, threshold: float = 0.5) -> bool:
            """Check if action is valid."""
            torch = _ensure_torch()
            device = next(self.parameters()).device
            s = torch.tensor([state_vec], dtype=torch.float32, device=device)
            a = torch.tensor([action_vec], dtype=torch.float32, device=device)
            with torch.no_grad():
                score = self.forward(s, a)
            return score[0, 0].item() > threshold

    return VerifierNetwork()


# ---------------------------------------------------------------------------
# Self-Play Training
# ---------------------------------------------------------------------------

@dataclass
class SelfPlayConfig:
    """Self-play training configuration."""
    n_rounds: int = 500
    episodes_per_round: int = 16
    proposer_lr: float = 5e-5
    adversary_lr: float = 1e-4
    verifier_lr: float = 3e-4
    adversary_reward_scale: float = 1.0
    proposer_reward_scale: float = 1.0
    verifier_reward_scale: float = 1.0
    convergence_window: int = 50
    convergence_threshold: float = 0.95  # proposer success rate
    eval_every: int = 25


def train_self_play(
    proposer,
    output_dir: str = "/tmp/cortex-train/self_play",
    config: Optional[SelfPlayConfig] = None,
    safety_head=None,
    device: str = "cpu",
) -> dict:
    """
    Train via self-play: proposer vs adversary, with verifier learning.

    Args:
        proposer: pre-trained policy network (from RL)
        output_dir: checkpoint directory
        config: self-play hyperparameters
        safety_head: frozen safety head
        device: compute device

    Returns: training stats
    """
    torch = _ensure_torch()
    from .simulator import CortexBootSimulator, random_hardware, Action, Verb, Target

    cfg = config or SelfPlayConfig()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build adversary and verifier
    adversary = build_adversary().to(device)
    verifier = build_verifier().to(device)
    proposer = proposer.to(device)

    # Optimizers
    prop_opt = torch.optim.Adam(proposer.parameters(), lr=cfg.proposer_lr)
    adv_opt = torch.optim.Adam(adversary.parameters(), lr=cfg.adversary_lr)
    ver_opt = torch.optim.Adam(verifier.parameters(), lr=cfg.verifier_lr)

    # Stats tracking
    proposer_wins = []
    adversary_wins = []
    verifier_accuracy_history = []
    eval_history = []

    t0 = time.time()

    for round_idx in range(cfg.n_rounds):
        round_proposer_rewards = []
        round_adversary_rewards = []
        round_verifier_correct = 0
        round_verifier_total = 0

        prop_loss_total = torch.tensor(0.0, device=device)
        adv_loss_total = torch.tensor(0.0, device=device)
        ver_loss_total = torch.tensor(0.0, device=device)

        for ep in range(cfg.episodes_per_round):
            hw = random_hardware()

            # Adversary generates fault based on initial state
            init_state = CortexBootSimulator(hardware=hw).reset()
            init_vec = init_state.to_vector()
            fault, adv_log_prob = adversary.generate_fault(init_vec)

            # Run episode with adversary's fault
            sim = CortexBootSimulator(
                hardware=hw, faults=[fault], seed=round_idx * 1000 + ep
            )
            state = sim.reset()
            episode_reward = 0.0
            done = False
            prop_log_probs = []
            ver_predictions = []
            ver_labels = []

            while not done:
                state_vec = state.to_vector()

                # Proposer selects action
                action, log_prob, confidence = proposer.act_with_logprob(state_vec)
                prop_log_probs.append(log_prob)

                # Verifier predicts outcome
                action_vec = action.to_vector()
                s_t = torch.tensor([state_vec], dtype=torch.float32, device=device)
                a_t = torch.tensor([action_vec], dtype=torch.float32, device=device)
                ver_pred = verifier(s_t, a_t)
                ver_predictions.append(ver_pred)

                # Execute
                state, reward, done, info = sim.step(action)
                episode_reward += reward

                # Verifier label: was the action successful?
                action_succeeded = reward > 0
                ver_labels.append(1.0 if action_succeeded else 0.0)
                if ver_pred[0, 0].item() > 0.5:
                    if action_succeeded:
                        round_verifier_correct += 1
                else:
                    if not action_succeeded:
                        round_verifier_correct += 1
                round_verifier_total += 1

            outcome = info.get("reason", "unknown")
            proposer_won = outcome == "success"

            # Rewards
            proposer_reward = episode_reward * cfg.proposer_reward_scale
            adversary_reward = (-episode_reward if not proposer_won else -1.0) * cfg.adversary_reward_scale

            round_proposer_rewards.append(proposer_reward)
            round_adversary_rewards.append(adversary_reward)

            # Proposer loss (REINFORCE)
            for lp in prop_log_probs:
                if isinstance(lp, torch.Tensor) and lp.requires_grad:
                    prop_loss_total -= lp * proposer_reward / cfg.episodes_per_round

            # Adversary loss (REINFORCE — maximize adversary_reward)
            if isinstance(adv_log_prob, torch.Tensor) and adv_log_prob.requires_grad:
                adv_loss_total -= adv_log_prob * adversary_reward / cfg.episodes_per_round

            # Verifier loss (BCE)
            if ver_predictions and ver_labels:
                ver_preds_t = torch.cat(ver_predictions, dim=0)
                ver_labels_t = torch.tensor(
                    [[l] for l in ver_labels], dtype=torch.float32, device=device
                )
                ver_loss = torch.nn.functional.binary_cross_entropy(
                    ver_preds_t, ver_labels_t
                )
                ver_loss_total += ver_loss / cfg.episodes_per_round

        # Update all three
        prop_opt.zero_grad()
        if prop_loss_total.requires_grad:
            prop_loss_total.backward()
            torch.nn.utils.clip_grad_norm_(proposer.parameters(), 0.5)
            prop_opt.step()

        adv_opt.zero_grad()
        if adv_loss_total.requires_grad:
            adv_loss_total.backward()
            torch.nn.utils.clip_grad_norm_(adversary.parameters(), 0.5)
            adv_opt.step()

        ver_opt.zero_grad()
        if ver_loss_total.requires_grad:
            ver_loss_total.backward()
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), 0.5)
            ver_opt.step()

        # Track
        prop_success = sum(1 for r in round_proposer_rewards if r > 0) / cfg.episodes_per_round
        proposer_wins.append(prop_success)
        adversary_wins.append(1.0 - prop_success)
        ver_acc = round_verifier_correct / max(round_verifier_total, 1)
        verifier_accuracy_history.append(ver_acc)

        # Periodic eval
        if round_idx % cfg.eval_every == 0 or round_idx == cfg.n_rounds - 1:
            from .rl import evaluate_policy
            eval_stats = evaluate_policy(proposer, n_episodes=50, fault_prob=0.5)
            eval_history.append({
                "round": round_idx,
                "success_rate": eval_stats["success_rate"],
                "avg_reward": eval_stats["avg_reward"],
                "proposer_win_rate": sum(proposer_wins[-cfg.convergence_window:]) / min(len(proposer_wins), cfg.convergence_window),
                "verifier_acc": ver_acc,
            })
            logger.info(
                "Round %d: prop_win=%.1f%% ver_acc=%.1f%% eval_success=%.1f%%",
                round_idx, prop_success * 100, ver_acc * 100,
                eval_stats["success_rate"] * 100,
            )

        # Convergence check
        if len(proposer_wins) >= cfg.convergence_window:
            recent_win_rate = sum(proposer_wins[-cfg.convergence_window:]) / cfg.convergence_window
            if recent_win_rate >= cfg.convergence_threshold:
                logger.info(
                    "Self-play converged at round %d (proposer wins %.1f%%)",
                    round_idx, recent_win_rate * 100,
                )
                break

    elapsed = time.time() - t0

    # Save all three models
    torch.save(proposer.state_dict(), os.path.join(output_dir, "proposer_final.pt"))
    torch.save(adversary.state_dict(), os.path.join(output_dir, "adversary_final.pt"))
    torch.save(verifier.state_dict(), os.path.join(output_dir, "verifier_final.pt"))

    # Also save best proposer
    torch.save(proposer.state_dict(), os.path.join(output_dir, "proposer_best.pt"))

    metadata = {
        "rounds_trained": round_idx + 1,
        "final_proposer_win_rate": sum(proposer_wins[-50:]) / min(len(proposer_wins), 50),
        "final_verifier_accuracy": sum(verifier_accuracy_history[-50:]) / min(len(verifier_accuracy_history), 50),
        "training_time_s": elapsed,
        "eval_history": eval_history,
    }
    with open(os.path.join(output_dir, "self_play_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Self-play complete: %d rounds, prop_win=%.1f%%, ver_acc=%.1f%%, time=%.1fs",
        round_idx + 1,
        metadata["final_proposer_win_rate"] * 100,
        metadata["final_verifier_accuracy"] * 100,
        elapsed,
    )

    return {
        "rounds": round_idx + 1,
        "proposer_win_rate": metadata["final_proposer_win_rate"],
        "verifier_accuracy": metadata["final_verifier_accuracy"],
        "elapsed_s": elapsed,
        "output_dir": output_dir,
    }
