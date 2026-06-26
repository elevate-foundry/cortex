"""
CKM World Model — learns system dynamics before acting.

The world model predicts: given state(t) + action(t), what is state(t+1)?

This is Phase 0 of training — the model learns physics (what HAPPENS)
before it learns policy (what to DO). Same principle as MuZero/Dreamer.

Architecture:
  StateEncoder: structured state → 256d dense vector
  ActionEncoder: verb + target + params → 128d
  DynamicsCore: 6-layer transformer, predicts delta state
  RewardPredictor: estimates expected reward for (state, action)

Training:
  Data: expert episodes from simulator
  Loss: MSE on predicted next-state vector + BCE on terminal prediction
  Validation: state prediction accuracy on held-out episodes

The world model enables:
  - Planning: search over action sequences without real execution
  - RL: generate imagined trajectories for policy gradient
  - Safety: predict consequences of dangerous actions without executing them
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.ckm.world_model")

_torch = None


def _ensure_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

@dataclass
class WorldModelConfig:
    """Configuration for the world model."""
    state_dim: int = 23          # from simulator.STATE_DIM
    action_dim: int = 26         # N_VERBS + N_TARGETS + N_PARAMS
    hidden_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    history_len: int = 8         # number of past transitions to attend over


def build_world_model(config: Optional[WorldModelConfig] = None):
    """Build the world model (lazy torch import)."""
    torch = _ensure_torch()
    nn = torch.nn
    cfg = config or WorldModelConfig()

    class StateEncoder(nn.Module):
        """Encode structured state vector to hidden dim."""
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(cfg.state_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.hidden_dim),
            )

        def forward(self, state_vec):
            return self.proj(state_vec)

    class ActionEncoder(nn.Module):
        """Encode action vector to hidden dim."""
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(cfg.action_dim, cfg.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim // 2, cfg.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.hidden_dim),
            )

        def forward(self, action_vec):
            return self.proj(action_vec)

    class DynamicsCore(nn.Module):
        """
        Transformer-based dynamics model.

        Input: sequence of (state, action) embeddings
        Output: predicted next state delta + reward
        """
        def __init__(self):
            super().__init__()
            self.state_enc = StateEncoder()
            self.action_enc = ActionEncoder()

            # Combine state + action into one token
            self.fusion = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)

            # Positional encoding for history
            self.pos_emb = nn.Embedding(cfg.history_len + 1, cfg.hidden_dim)

            # Transformer layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.hidden_dim * 4,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

            # Output heads
            self.state_predictor = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.state_dim),
            )
            self.reward_predictor = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim // 2, 1),
            )
            self.terminal_predictor = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 4),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim // 4, 1),
                nn.Sigmoid(),
            )

        def forward(self, state_vecs, action_vecs):
            """
            Predict next state given history of (state, action) pairs.

            Args:
                state_vecs: (batch, seq_len, state_dim) — last K states + current
                action_vecs: (batch, seq_len, action_dim) — last K actions + current

            Returns:
                next_state: (batch, state_dim)
                reward: (batch, 1)
                terminal: (batch, 1)
            """
            batch_size, seq_len, _ = state_vecs.shape

            # Encode
            state_emb = self.state_enc(state_vecs)    # (B, T, H)
            action_emb = self.action_enc(action_vecs)  # (B, T, H)

            # Fuse state + action per timestep
            combined = torch.cat([state_emb, action_emb], dim=-1)  # (B, T, 2H)
            tokens = self.fusion(combined)  # (B, T, H)

            # Add positional encoding
            positions = torch.arange(seq_len, device=tokens.device)
            tokens = tokens + self.pos_emb(positions).unsqueeze(0)

            # Transformer
            output = self.transformer(tokens)  # (B, T, H)

            # Use last token for prediction
            last_hidden = output[:, -1, :]  # (B, H)

            # Predict
            next_state = self.state_predictor(last_hidden)
            reward = self.reward_predictor(last_hidden)
            terminal = self.terminal_predictor(last_hidden)

            return next_state, reward, terminal

        def predict_single(self, state_vec, action_vec, history=None):
            """Predict next state for a single (state, action) pair."""
            torch = _ensure_torch()
            device = next(self.parameters()).device

            # Build sequence
            if history is not None:
                # history: list of (state_vec, action_vec) tuples
                states = [h[0] for h in history] + [state_vec]
                actions = [h[1] for h in history] + [action_vec]
            else:
                states = [state_vec]
                actions = [action_vec]

            # Pad/truncate to history_len
            while len(states) < cfg.history_len + 1:
                states.insert(0, [0.0] * cfg.state_dim)
                actions.insert(0, [0.0] * cfg.action_dim)
            states = states[-(cfg.history_len + 1):]
            actions = actions[-(cfg.history_len + 1):]

            state_t = torch.tensor([states], dtype=torch.float32, device=device)
            action_t = torch.tensor([actions], dtype=torch.float32, device=device)

            with torch.no_grad():
                next_state, reward, terminal = self.forward(state_t, action_t)

            return (
                next_state[0].cpu().tolist(),
                reward[0, 0].item(),
                terminal[0, 0].item(),
            )

    return DynamicsCore()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class WorldModelTrainingConfig:
    """Training hyperparameters for world model."""
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    state_loss_weight: float = 1.0
    reward_loss_weight: float = 0.1
    terminal_loss_weight: float = 0.5
    val_fraction: float = 0.1
    patience: int = 10
    history_len: int = 8


def prepare_world_model_data(episodes, history_len: int = 8):
    """
    Convert episode transitions into training samples for world model.

    Each sample: (state_history, action_history) → (next_state, reward, terminal)
    """
    samples = []

    for ep in episodes:
        transitions = ep.transitions
        n = len(transitions)
        is_terminal = [False] * n
        is_terminal[-1] = True

        for t in range(n):
            # Build history window
            start = max(0, t - history_len + 1)
            states = [transitions[i][0] for i in range(start, t + 1)]
            actions = [transitions[i][1] for i in range(start, t + 1)]

            # Pad if needed
            state_dim = len(states[0])
            action_dim = len(actions[0])
            while len(states) < history_len + 1:
                states.insert(0, [0.0] * state_dim)
                actions.insert(0, [0.0] * action_dim)
            # Truncate
            states = states[-(history_len + 1):]
            actions = actions[-(history_len + 1):]

            # Target
            next_state = transitions[t][3]  # next_state_vec
            reward = transitions[t][2]
            terminal = 1.0 if is_terminal[t] else 0.0

            samples.append({
                "states": states,
                "actions": actions,
                "next_state": next_state,
                "reward": reward,
                "terminal": terminal,
            })

    return samples


def train_world_model(
    episodes,
    output_dir: str = "/tmp/cortex-train/world_model",
    config: Optional[WorldModelTrainingConfig] = None,
    model_config: Optional[WorldModelConfig] = None,
    device: str = "cpu",
) -> dict:
    """
    Train the world model on expert episode data.

    Returns training stats dict.
    """
    torch = _ensure_torch()
    cfg = config or WorldModelTrainingConfig()
    m_cfg = model_config or WorldModelConfig()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    logger.info("Preparing world model training data from %d episodes...", len(episodes))
    samples = prepare_world_model_data(episodes, cfg.history_len)
    logger.info("Generated %d training samples", len(samples))

    # Train/val split
    n_val = max(1, int(len(samples) * cfg.val_fraction))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    # Build model
    model = build_world_model(m_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("World model: %d parameters", n_params)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # LR scheduler (cosine)
    total_steps = cfg.epochs * (len(train_samples) // cfg.batch_size + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Loss functions
    state_loss_fn = torch.nn.MSELoss()
    reward_loss_fn = torch.nn.MSELoss()
    terminal_loss_fn = torch.nn.BCELoss()

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "state_acc": []}

    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle
        import random
        random.shuffle(train_samples)

        for batch_start in range(0, len(train_samples), cfg.batch_size):
            batch = train_samples[batch_start:batch_start + cfg.batch_size]
            if len(batch) < 2:
                continue

            # Collate
            states_t = torch.tensor([s["states"] for s in batch], dtype=torch.float32, device=device)
            actions_t = torch.tensor([s["actions"] for s in batch], dtype=torch.float32, device=device)
            next_states_t = torch.tensor([s["next_state"] for s in batch], dtype=torch.float32, device=device)
            rewards_t = torch.tensor([[s["reward"]] for s in batch], dtype=torch.float32, device=device)
            terminals_t = torch.tensor([[s["terminal"]] for s in batch], dtype=torch.float32, device=device)

            # Forward
            pred_state, pred_reward, pred_terminal = model(states_t, actions_t)

            # Loss
            loss_state = state_loss_fn(pred_state, next_states_t) * cfg.state_loss_weight
            loss_reward = reward_loss_fn(pred_reward, rewards_t) * cfg.reward_loss_weight
            loss_terminal = terminal_loss_fn(pred_terminal, terminals_t) * cfg.terminal_loss_weight
            loss = loss_state + loss_reward + loss_terminal

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        state_errors = []
        with torch.no_grad():
            states_t = torch.tensor([s["states"] for s in val_samples], dtype=torch.float32, device=device)
            actions_t = torch.tensor([s["actions"] for s in val_samples], dtype=torch.float32, device=device)
            next_states_t = torch.tensor([s["next_state"] for s in val_samples], dtype=torch.float32, device=device)
            rewards_t = torch.tensor([[s["reward"]] for s in val_samples], dtype=torch.float32, device=device)
            terminals_t = torch.tensor([[s["terminal"]] for s in val_samples], dtype=torch.float32, device=device)

            pred_state, pred_reward, pred_terminal = model(states_t, actions_t)
            loss_state = state_loss_fn(pred_state, next_states_t)
            loss_reward = reward_loss_fn(pred_reward, rewards_t)
            loss_terminal = terminal_loss_fn(pred_terminal, terminals_t)
            val_loss = (loss_state * cfg.state_loss_weight +
                       loss_reward * cfg.reward_loss_weight +
                       loss_terminal * cfg.terminal_loss_weight).item()

            # State prediction accuracy (fraction of dims within 0.1 of target)
            errors = (pred_state - next_states_t).abs()
            state_acc = (errors < 0.1).float().mean().item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["state_acc"].append(state_acc)

        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            logger.info(
                "Epoch %d/%d: train=%.4f val=%.4f state_acc=%.2f%%",
                epoch + 1, cfg.epochs, avg_train_loss, val_loss, state_acc * 100,
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(output_dir, "world_model_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    elapsed = time.time() - t0

    # Save final model + config
    torch.save(model.state_dict(), os.path.join(output_dir, "world_model_final.pt"))
    metadata = {
        "n_params": n_params,
        "epochs_trained": epoch + 1,
        "best_val_loss": best_val_loss,
        "final_state_acc": history["state_acc"][-1],
        "training_time_s": elapsed,
        "config": {
            "state_dim": m_cfg.state_dim,
            "action_dim": m_cfg.action_dim,
            "hidden_dim": m_cfg.hidden_dim,
            "n_layers": m_cfg.n_layers,
            "n_heads": m_cfg.n_heads,
        },
    }
    with open(os.path.join(output_dir, "world_model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "World model trained: %d params, acc=%.1f%%, time=%.1fs",
        n_params, history["state_acc"][-1] * 100, elapsed,
    )

    return {
        "n_params": n_params,
        "best_val_loss": best_val_loss,
        "final_state_acc": history["state_acc"][-1],
        "epochs": epoch + 1,
        "elapsed_s": elapsed,
        "output_dir": output_dir,
    }


def load_world_model(path: str, device: str = "cpu", config: Optional[WorldModelConfig] = None):
    """Load a trained world model from checkpoint."""
    torch = _ensure_torch()
    cfg = config or WorldModelConfig()
    model = build_world_model(cfg).to(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_world_model(
    model_path: Optional[str] = None,
    n_episodes: int = 100,
    device: str = "cpu",
) -> dict:
    """Validate world model prediction accuracy on fresh episodes."""
    torch = _ensure_torch()
    from .simulator import generate_expert_episodes, CortexBootSimulator, expert_policy

    # Generate fresh test episodes
    episodes = generate_expert_episodes(n=n_episodes, include_faults=True)
    samples = prepare_world_model_data(episodes)

    if model_path and os.path.exists(model_path):
        model = load_world_model(model_path, device=device)
    else:
        logger.warning("No model path provided, training a quick model for validation...")
        train_eps = generate_expert_episodes(n=200)
        result = train_world_model(train_eps, device=device,
                                   config=WorldModelTrainingConfig(epochs=20, batch_size=32))
        model = load_world_model(
            os.path.join(result["output_dir"], "world_model_best.pt"), device=device
        )

    # Evaluate
    model.eval()
    state_errors = []
    reward_errors = []
    terminal_correct = 0

    with torch.no_grad():
        for sample in samples[:500]:
            states_t = torch.tensor([sample["states"]], dtype=torch.float32, device=device)
            actions_t = torch.tensor([sample["actions"]], dtype=torch.float32, device=device)

            pred_state, pred_reward, pred_terminal = model(states_t, actions_t)

            # State error
            target_state = torch.tensor([sample["next_state"]], dtype=torch.float32)
            error = (pred_state.cpu() - target_state).abs().mean().item()
            state_errors.append(error)

            # Reward error
            reward_errors.append(abs(pred_reward[0, 0].item() - sample["reward"]))

            # Terminal accuracy
            pred_term = pred_terminal[0, 0].item() > 0.5
            actual_term = sample["terminal"] > 0.5
            if pred_term == actual_term:
                terminal_correct += 1

    stats = {
        "avg_state_error": sum(state_errors) / len(state_errors),
        "avg_reward_error": sum(reward_errors) / len(reward_errors),
        "terminal_accuracy": terminal_correct / len(samples[:500]),
        "state_accuracy_pct": sum(1 for e in state_errors if e < 0.1) / len(state_errors) * 100,
    }

    logger.info("World model validation: state_err=%.4f, reward_err=%.3f, terminal_acc=%.1f%%",
                stats["avg_state_error"], stats["avg_reward_error"],
                stats["terminal_accuracy"] * 100)

    return stats
