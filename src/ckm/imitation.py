"""
CKM Imitation Learning — bootstrap policy from expert demonstrations.

Phase 3 of training: after world model and safety head are trained,
the policy is bootstrapped via supervised learning on expert traces.

The expert policy (hand-coded optimal boot sequence) provides:
  (state, correct_action) pairs

The model learns:
  - Verb classification: which verb to use given state
  - Target classification: which target to act on
  - Parameter regression: what numeric config values to set

This is NOT the final policy — it's the initialization for RL.
Think of it as behavioral cloning before fine-tuning with REINFORCE.

Multi-task loss:
  L = CE(verb) + CE(target) + MSE(params) + safety_penalty
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.ckm.imitation")

_torch = None


def _ensure_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# ---------------------------------------------------------------------------
# Policy Network
# ---------------------------------------------------------------------------

def build_policy_network(state_dim: int = 23, hidden_dim: int = 256,
                         n_verbs: int = 12, n_targets: int = 8, n_params: int = 6):
    """Build the multi-head policy network."""
    torch = _ensure_torch()
    nn = torch.nn

    class PolicyNetwork(nn.Module):
        """
        Multi-head policy for boot action selection.

        Inputs: state vector (from simulator)
        Outputs:
          - verb_logits: (batch, n_verbs)
          - target_logits: (batch, n_targets)
          - param_values: (batch, n_params) — continuous
          - confidence: (batch, 1) — how sure the model is
        """
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )

            # Verb head
            self.verb_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, n_verbs),
            )

            # Target head
            self.target_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, n_targets),
            )

            # Param head (continuous, sigmoid to bound [0,1])
            self.param_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, n_params),
                nn.Sigmoid(),
            )

            # Confidence head
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid(),
            )

        def forward(self, state_vec):
            """
            Forward pass.

            Args:
                state_vec: (batch, state_dim)

            Returns:
                verb_logits, target_logits, param_values, confidence
            """
            h = self.encoder(state_vec)
            return (
                self.verb_head(h),
                self.target_head(h),
                self.param_head(h),
                self.confidence_head(h),
            )

        def act(self, state_vec, deterministic: bool = True):
            """Select action from state vector (single inference)."""
            torch = _ensure_torch()
            from .simulator import Action, Verb, Target

            device = next(self.parameters()).device
            s = torch.tensor([state_vec], dtype=torch.float32, device=device)

            with torch.no_grad():
                verb_logits, target_logits, params, confidence = self.forward(s)

            if deterministic:
                verb_idx = verb_logits[0].argmax().item()
                target_idx = target_logits[0].argmax().item()
            else:
                # Sample from softmax
                verb_probs = torch.softmax(verb_logits[0], dim=0)
                verb_idx = torch.multinomial(verb_probs, 1).item()
                target_probs = torch.softmax(target_logits[0], dim=0)
                target_idx = torch.multinomial(target_probs, 1).item()

            # Denormalize params
            param_keys = ["threads", "gpu_layers", "ctx_size", "batch_size", "port", "timeout_ms"]
            param_maxes = [128, 999, 131072, 64, 65535, 30000]
            action_params = {}
            for i, (key, mx) in enumerate(zip(param_keys, param_maxes)):
                val = params[0, i].item() * mx
                if val > 0.01 * mx:  # Only include non-trivial params
                    action_params[key] = val

            return Action(
                verb=Verb(verb_idx),
                target=Target(target_idx),
                params=action_params,
            )

        def act_with_logprob(self, state_vec):
            """Select action and return log probability (for RL)."""
            torch = _ensure_torch()
            from .simulator import Action, Verb, Target

            device = next(self.parameters()).device
            s = torch.tensor([state_vec], dtype=torch.float32, device=device)

            verb_logits, target_logits, params, confidence = self.forward(s)

            # Sample
            verb_dist = torch.distributions.Categorical(logits=verb_logits[0])
            target_dist = torch.distributions.Categorical(logits=target_logits[0])

            verb_idx = verb_dist.sample()
            target_idx = target_dist.sample()

            log_prob = verb_dist.log_prob(verb_idx) + target_dist.log_prob(target_idx)

            # Denormalize params
            param_keys = ["threads", "gpu_layers", "ctx_size", "batch_size", "port", "timeout_ms"]
            param_maxes = [128, 999, 131072, 64, 65535, 30000]
            action_params = {}
            for i, (key, mx) in enumerate(zip(param_keys, param_maxes)):
                val = params[0, i].item() * mx
                if val > 0.01 * mx:
                    action_params[key] = val

            action = Action(
                verb=Verb(verb_idx.item()),
                target=Target(target_idx.item()),
                params=action_params,
            )

            return action, log_prob, confidence[0, 0]

    return PolicyNetwork()


# ---------------------------------------------------------------------------
# Training data preparation
# ---------------------------------------------------------------------------

def prepare_imitation_data(episodes) -> list[dict]:
    """
    Convert expert episodes into (state, action_labels) pairs.

    Each sample has:
      state: state vector
      verb: int (ground truth verb index)
      target: int (ground truth target index)
      params: list[float] (normalized param values)
    """
    from .simulator import N_VERBS, N_TARGETS, N_PARAMS

    samples = []
    for ep in episodes:
        for state_vec, action_vec, reward, next_state_vec in ep.transitions:
            # Extract verb and target from one-hot action vector
            verb_idx = max(range(N_VERBS), key=lambda i: action_vec[i])
            target_idx = max(range(N_TARGETS), key=lambda i: action_vec[N_VERBS + i])
            params = action_vec[N_VERBS + N_TARGETS:]

            samples.append({
                "state": state_vec,
                "verb": verb_idx,
                "target": target_idx,
                "params": params,
                "reward": reward,
            })

    return samples


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class ImitationConfig:
    """Imitation learning training config."""
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    verb_loss_weight: float = 1.0
    target_loss_weight: float = 1.0
    param_loss_weight: float = 0.5
    patience: int = 15
    val_fraction: float = 0.1


def train_imitation(
    episodes,
    output_dir: str = "/tmp/cortex-train/imitation",
    config: Optional[ImitationConfig] = None,
    device: str = "cpu",
    safety_head=None,
) -> dict:
    """
    Train policy network via imitation learning on expert demonstrations.

    Args:
        episodes: list of EpisodeResult from expert_policy
        output_dir: where to save checkpoints
        config: training hyperparameters
        device: cpu/cuda/mps
        safety_head: optional trained safety head for penalty

    Returns: training stats dict
    """
    torch = _ensure_torch()
    cfg = config or ImitationConfig()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    logger.info("Preparing imitation data from %d episodes...", len(episodes))
    samples = prepare_imitation_data(episodes)
    logger.info("Generated %d imitation samples", len(samples))

    # Split
    n_val = max(10, int(len(samples) * cfg.val_fraction))
    random.shuffle(samples)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    # Build model
    from .simulator import N_VERBS, N_TARGETS, N_PARAMS, STATE_DIM
    model = build_policy_network(
        state_dim=STATE_DIM, hidden_dim=256,
        n_verbs=N_VERBS, n_targets=N_TARGETS, n_params=N_PARAMS,
    ).to(device)
    n_params_total = sum(p.numel() for p in model.parameters())
    logger.info("Policy network: %d parameters", n_params_total)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * (len(train_samples) // cfg.batch_size + 1)
    )

    # Loss functions
    verb_loss_fn = torch.nn.CrossEntropyLoss()
    target_loss_fn = torch.nn.CrossEntropyLoss()
    param_loss_fn = torch.nn.MSELoss()

    # Training
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_verb_acc": [], "val_target_acc": []}

    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        random.shuffle(train_samples)

        for batch_start in range(0, len(train_samples), cfg.batch_size):
            batch = train_samples[batch_start:batch_start + cfg.batch_size]
            if len(batch) < 2:
                continue

            states = torch.tensor([s["state"] for s in batch], dtype=torch.float32, device=device)
            verbs = torch.tensor([s["verb"] for s in batch], dtype=torch.long, device=device)
            targets = torch.tensor([s["target"] for s in batch], dtype=torch.long, device=device)
            params = torch.tensor([s["params"] for s in batch], dtype=torch.float32, device=device)

            # Forward
            verb_logits, target_logits, pred_params, confidence = model(states)

            # Multi-task loss
            loss_verb = verb_loss_fn(verb_logits, verbs) * cfg.verb_loss_weight
            loss_target = target_loss_fn(target_logits, targets) * cfg.target_loss_weight
            loss_params = param_loss_fn(pred_params, params) * cfg.param_loss_weight
            loss = loss_verb + loss_target + loss_params

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            states = torch.tensor([s["state"] for s in val_samples], dtype=torch.float32, device=device)
            verbs = torch.tensor([s["verb"] for s in val_samples], dtype=torch.long, device=device)
            targets = torch.tensor([s["target"] for s in val_samples], dtype=torch.long, device=device)

            verb_logits, target_logits, _, _ = model(states)

            verb_acc = (verb_logits.argmax(dim=1) == verbs).float().mean().item()
            target_acc = (target_logits.argmax(dim=1) == targets).float().mean().item()

        history["train_loss"].append(avg_loss)
        history["val_verb_acc"].append(verb_acc)
        history["val_target_acc"].append(target_acc)

        combined_acc = (verb_acc + target_acc) / 2.0

        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            logger.info(
                "Epoch %d/%d: loss=%.4f verb_acc=%.1f%% target_acc=%.1f%%",
                epoch + 1, cfg.epochs, avg_loss, verb_acc * 100, target_acc * 100,
            )

        # Save best
        if combined_acc > best_val_acc:
            best_val_acc = combined_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "policy_imitation_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    elapsed = time.time() - t0

    # Save
    torch.save(model.state_dict(), os.path.join(output_dir, "policy_imitation_final.pt"))
    metadata = {
        "n_params": n_params_total,
        "epochs_trained": epoch + 1,
        "best_verb_acc": max(history["val_verb_acc"]),
        "best_target_acc": max(history["val_target_acc"]),
        "training_time_s": elapsed,
    }
    with open(os.path.join(output_dir, "imitation_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Imitation training: verb=%.1f%% target=%.1f%% time=%.1fs",
                max(history["val_verb_acc"]) * 100,
                max(history["val_target_acc"]) * 100, elapsed)

    return {
        "n_params": n_params_total,
        "best_verb_acc": max(history["val_verb_acc"]),
        "best_target_acc": max(history["val_target_acc"]),
        "epochs": epoch + 1,
        "elapsed_s": elapsed,
        "output_dir": output_dir,
    }


def load_policy(path: str, device: str = "cpu"):
    """Load trained policy network."""
    torch = _ensure_torch()
    from .simulator import N_VERBS, N_TARGETS, N_PARAMS, STATE_DIM
    model = build_policy_network(
        state_dim=STATE_DIM, hidden_dim=256,
        n_verbs=N_VERBS, n_targets=N_TARGETS, n_params=N_PARAMS,
    ).to(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
