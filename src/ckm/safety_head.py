"""
CKM Safety Head — contrastive pre-training for safety classification.

The safety head is trained BEFORE the policy. It learns to discriminate
between safe and unsafe actions via triplet margin loss on contrastive pairs.

Key property: the safety head has **veto power** over the action head.
If safety_score > threshold, the action is denied regardless of policy.

Training data sources:
  1. dangerous_targets.scl → generate (state, safe_action, unsafe_action) triplets
  2. allowed_verbs.scl → verb-level safe/unsafe discrimination
  3. Simulator violations → any action that triggers safety_violation outcome

Target metrics:
  - 100% recall on dangerous targets (0% false negatives)
  - >99% precision (minimal false positives)
  - Calibrated: score correlates with actual danger severity
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.ckm.safety_head")

_torch = None


def _ensure_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# ---------------------------------------------------------------------------
# Contrastive pair generation
# ---------------------------------------------------------------------------

@dataclass
class SafetyTriplet:
    """One contrastive safety training example."""
    state: list[float]         # state vector
    safe_action: list[float]   # action vector (safe alternative)
    unsafe_action: list[float] # action vector (dangerous)
    severity: float            # 0.0 = borderline, 1.0 = catastrophic


def generate_safety_triplets(n_per_category: int = 100) -> list[SafetyTriplet]:
    """
    Generate contrastive safety triplets from policy SCL and simulator.

    Categories:
      1. Dangerous target access (kill /dev/mem, write /dev/sda)
      2. Blocked verb usage (format, erase, flash, exec)
      3. Resource exhaustion (spawn when OOM)
      4. Unsafe state transitions (skip detection, start without config)
    """
    from .simulator import (
        Action, Verb, Target, BootState, HardwareProfile, Resources,
        BootPhase, NetworkState, N_VERBS, N_TARGETS, N_PARAMS,
        HARDWARE_PROFILES, is_dangerous,
    )

    triplets = []

    # Category 1: Dangerous target references
    # The model must learn: DENY access to raw hardware, suggest OBSERVE instead
    for _ in range(n_per_category):
        hw = random.choice(HARDWARE_PROFILES)
        state = BootState(
            hardware=hw,
            services={},
            resources=Resources(cpu_pct=0.1, ram_used_mb=256, vram_used_mb=0),
            network=NetworkState.DOWN,
            phase=BootPhase(random.randint(0, 4)),
            time_ms=random.randint(0, 5000),
            hw_detected=True,
        )

        # Unsafe: Kill system (simulates /dev/mem write)
        unsafe_action = Action(Verb.KILL, Target.SYSTEM, {"force": 1.0})

        # Safe alternative: Observe system
        safe_action = Action(Verb.OBSERVE, Target.SYSTEM)

        triplets.append(SafetyTriplet(
            state=state.to_vector(),
            safe_action=safe_action.to_vector(),
            unsafe_action=unsafe_action.to_vector(),
            severity=1.0,  # catastrophic
        ))

    # Category 2: Spawning when resources are exhausted
    for _ in range(n_per_category):
        hw = random.choice(HARDWARE_PROFILES)
        state = BootState(
            hardware=hw,
            services={},
            resources=Resources(
                cpu_pct=0.9,
                ram_used_mb=hw.ram_mb - 50,  # nearly full
                vram_used_mb=hw.vram_mb,
            ),
            network=NetworkState.DOWN,
            phase=BootPhase.BACKEND_START,
            time_ms=random.randint(1000, 5000),
            hw_detected=True,
        )

        # Unsafe: Try to spawn another service (will OOM)
        unsafe_action = Action(Verb.SPAWN, Target.INFERENCE)

        # Safe: Observe/deny/escalate
        safe_choices = [
            Action(Verb.OBSERVE, Target.SYSTEM),
            Action(Verb.DENY, Target.INFERENCE),
            Action(Verb.ESCALATE, Target.SYSTEM),
        ]
        safe_action = random.choice(safe_choices)

        triplets.append(SafetyTriplet(
            state=state.to_vector(),
            safe_action=safe_action.to_vector(),
            unsafe_action=unsafe_action.to_vector(),
            severity=0.7,  # high but recoverable
        ))

    # Category 3: Skipping hardware detection (unsafe state transition)
    for _ in range(n_per_category):
        hw = random.choice(HARDWARE_PROFILES)
        state = BootState(
            hardware=hw,
            services={},
            resources=Resources(),
            network=NetworkState.DOWN,
            phase=BootPhase.INIT,
            time_ms=0,
            hw_detected=False,  # Hardware NOT yet detected
        )

        # Unsafe: Configure without detection (guessing)
        unsafe_action = Action(Verb.CONFIGURE, Target.CONFIG, {
            "threads": float(random.randint(1, 64)),
            "gpu_layers": float(random.randint(0, 999)),
        })

        # Safe: Detect hardware first
        safe_action = Action(Verb.DETECT, Target.HARDWARE)

        triplets.append(SafetyTriplet(
            state=state.to_vector(),
            safe_action=safe_action.to_vector(),
            unsafe_action=unsafe_action.to_vector(),
            severity=0.4,  # moderate — wrong config, not catastrophic
        ))

    # Category 4: Restarting a service too many times (thrashing)
    for _ in range(n_per_category):
        from .simulator import ServiceState, ServiceStatus
        hw = random.choice(HARDWARE_PROFILES)
        state = BootState(
            hardware=hw,
            services={
                "inference": ServiceState(
                    status=ServiceStatus.FAILED,
                    restart_count=3,  # already retried 3 times
                ),
            },
            resources=Resources(cpu_pct=0.3, ram_used_mb=512),
            network=NetworkState.DOWN,
            phase=BootPhase.BACKEND_START,
            time_ms=5000,
            hw_detected=True,
        )

        # Unsafe: Keep restarting (thrashing, won't help)
        unsafe_action = Action(Verb.RESTART, Target.INFERENCE)

        # Safe: Escalate or observe
        safe_choices = [
            Action(Verb.ESCALATE, Target.SYSTEM),
            Action(Verb.OBSERVE, Target.INFERENCE),
            Action(Verb.CONFIGURE, Target.CONFIG, {"gpu_layers": 0.0}),  # try with less resources
        ]
        safe_action = random.choice(safe_choices)

        triplets.append(SafetyTriplet(
            state=state.to_vector(),
            safe_action=safe_action.to_vector(),
            unsafe_action=unsafe_action.to_vector(),
            severity=0.5,
        ))

    # Category 5: Network actions when network already failed
    for _ in range(n_per_category // 2):
        hw = random.choice(HARDWARE_PROFILES)
        state = BootState(
            hardware=hw,
            services={},
            resources=Resources(),
            network=NetworkState.FAILED,
            phase=BootPhase.SERVICES,
            time_ms=3000,
            hw_detected=True,
        )

        # Unsafe: Keep trying network (already failed)
        unsafe_action = Action(Verb.NETWORK_UP, Target.NETWORK)

        # Safe: Continue without network
        safe_action = Action(Verb.SPAWN, Target.API)

        triplets.append(SafetyTriplet(
            state=state.to_vector(),
            safe_action=safe_action.to_vector(),
            unsafe_action=unsafe_action.to_vector(),
            severity=0.3,  # low — just wasteful, not dangerous
        ))

    random.shuffle(triplets)
    return triplets


# ---------------------------------------------------------------------------
# Safety Head Model
# ---------------------------------------------------------------------------

def build_safety_head(state_dim: int = 23, action_dim: int = 26, hidden_dim: int = 256):
    """Build the safety classification head."""
    torch = _ensure_torch()
    nn = torch.nn

    class SafetyHead(nn.Module):
        """
        Scores (state, action) pairs for safety.

        Output: safety_score ∈ [0, 1]
          0.0 = definitely safe
          1.0 = definitely dangerous

        Trained with triplet margin loss:
          score(state, unsafe_action) > score(state, safe_action) + margin
        """
        def __init__(self):
            super().__init__()
            self.state_proj = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.action_proj = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        def forward(self, state_vec, action_vec):
            """
            Score a (state, action) pair for danger.

            Args:
                state_vec: (batch, state_dim)
                action_vec: (batch, action_dim)

            Returns:
                danger_score: (batch, 1) in [0, 1]
            """
            s = self.state_proj(state_vec)
            a = self.action_proj(action_vec)
            combined = torch.cat([s, a], dim=-1)
            return self.classifier(combined)

        def is_safe(self, state_vec, action_vec, threshold: float = 0.5) -> bool:
            """Check if an action is safe (single inference)."""
            torch = _ensure_torch()
            device = next(self.parameters()).device

            s = torch.tensor([state_vec], dtype=torch.float32, device=device)
            a = torch.tensor([action_vec], dtype=torch.float32, device=device)

            with torch.no_grad():
                score = self.forward(s, a)

            return score[0, 0].item() < threshold

    return SafetyHead()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class SafetyTrainingConfig:
    """Safety head training config."""
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    margin: float = 0.5       # triplet margin
    weight_decay: float = 0.01
    patience: int = 15
    n_triplets_per_category: int = 200


def train_safety_head(
    output_dir: str = "/tmp/cortex-train/safety_head",
    config: Optional[SafetyTrainingConfig] = None,
    device: str = "cpu",
) -> dict:
    """
    Train the safety head with contrastive triplet loss.

    Returns training stats.
    """
    torch = _ensure_torch()
    cfg = config or SafetyTrainingConfig()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate triplets
    logger.info("Generating safety triplets...")
    triplets = generate_safety_triplets(n_per_category=cfg.n_triplets_per_category)
    logger.info("Generated %d triplets", len(triplets))

    # Split
    n_val = max(10, int(len(triplets) * 0.1))
    val_triplets = triplets[:n_val]
    train_triplets = triplets[n_val:]

    # Build model
    from .simulator import STATE_DIM, ACTION_DIM
    model = build_safety_head(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Safety head: %d parameters", n_params)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # Triplet margin loss
    margin_loss = torch.nn.MarginRankingLoss(margin=cfg.margin)

    # Training loop
    best_val_recall = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_recall": [], "val_precision": []}

    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        random.shuffle(train_triplets)

        for batch_start in range(0, len(train_triplets), cfg.batch_size):
            batch = train_triplets[batch_start:batch_start + cfg.batch_size]
            if len(batch) < 2:
                continue

            states = torch.tensor([t.state for t in batch], dtype=torch.float32, device=device)
            safe_actions = torch.tensor([t.safe_action for t in batch], dtype=torch.float32, device=device)
            unsafe_actions = torch.tensor([t.unsafe_action for t in batch], dtype=torch.float32, device=device)
            severities = torch.tensor([[t.severity] for t in batch], dtype=torch.float32, device=device)

            # Score both
            safe_scores = model(states, safe_actions)     # should be low
            unsafe_scores = model(states, unsafe_actions)  # should be high

            # Triplet loss: unsafe_score > safe_score + margin
            target = torch.ones(len(batch), 1, device=device)  # unsafe should rank higher
            loss = margin_loss(unsafe_scores, safe_scores, target)

            # Add severity-weighted BCE for calibration
            bce = torch.nn.functional.binary_cross_entropy(
                unsafe_scores, severities
            ) * 0.3
            loss = loss + bce

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            states = torch.tensor([t.state for t in val_triplets], dtype=torch.float32, device=device)
            safe_actions = torch.tensor([t.safe_action for t in val_triplets], dtype=torch.float32, device=device)
            unsafe_actions = torch.tensor([t.unsafe_action for t in val_triplets], dtype=torch.float32, device=device)

            safe_scores = model(states, safe_actions)
            unsafe_scores = model(states, unsafe_actions)

            # Recall: fraction of unsafe actions scored > 0.5
            recall = (unsafe_scores > 0.5).float().mean().item()
            # Precision: fraction of safe actions scored < 0.5
            precision = (safe_scores < 0.5).float().mean().item()

        history["train_loss"].append(avg_loss)
        history["val_recall"].append(recall)
        history["val_precision"].append(precision)

        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            logger.info(
                "Epoch %d/%d: loss=%.4f recall=%.2f%% precision=%.2f%%",
                epoch + 1, cfg.epochs, avg_loss, recall * 100, precision * 100,
            )

        # Early stopping on recall
        if recall > best_val_recall:
            best_val_recall = recall
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "safety_head_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience and recall >= 0.99:
                logger.info("Safety head converged at epoch %d (recall=%.2f%%)", epoch + 1, recall * 100)
                break

    elapsed = time.time() - t0

    # Save
    torch.save(model.state_dict(), os.path.join(output_dir, "safety_head_final.pt"))
    metadata = {
        "n_params": n_params,
        "epochs_trained": epoch + 1,
        "best_recall": best_val_recall,
        "final_precision": history["val_precision"][-1],
        "training_time_s": elapsed,
    }
    with open(os.path.join(output_dir, "safety_head_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Safety head trained: recall=%.1f%%, precision=%.1f%%, time=%.1fs",
                best_val_recall * 100, history["val_precision"][-1] * 100, elapsed)

    return {
        "n_params": n_params,
        "best_recall": best_val_recall,
        "final_precision": history["val_precision"][-1],
        "epochs": epoch + 1,
        "elapsed_s": elapsed,
        "output_dir": output_dir,
    }


def load_safety_head(path: str, device: str = "cpu"):
    """Load trained safety head."""
    torch = _ensure_torch()
    model = build_safety_head().to(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def validate_safety(model_path: Optional[str] = None, device: str = "cpu") -> dict:
    """Validate safety head on fresh triplets."""
    torch = _ensure_torch()

    if model_path and os.path.exists(model_path):
        model = load_safety_head(model_path, device=device)
    else:
        logger.info("No model, training quick safety head for validation...")
        result = train_safety_head(device=device,
                                   config=SafetyTrainingConfig(epochs=50, n_triplets_per_category=100))
        model = load_safety_head(
            os.path.join(result["output_dir"], "safety_head_best.pt"), device=device
        )

    # Generate fresh test triplets
    test_triplets = generate_safety_triplets(n_per_category=50)

    model.eval()
    with torch.no_grad():
        states = torch.tensor([t.state for t in test_triplets], dtype=torch.float32, device=device)
        safe_actions = torch.tensor([t.safe_action for t in test_triplets], dtype=torch.float32, device=device)
        unsafe_actions = torch.tensor([t.unsafe_action for t in test_triplets], dtype=torch.float32, device=device)

        safe_scores = model(states, safe_actions)
        unsafe_scores = model(states, unsafe_actions)

        recall = (unsafe_scores > 0.5).float().mean().item()
        precision = (safe_scores < 0.5).float().mean().item()
        # False negatives (dangerous actions scored safe)
        false_negatives = (unsafe_scores < 0.5).float().sum().item()

    stats = {
        "recall": recall,
        "precision": precision,
        "false_negatives": int(false_negatives),
        "total_unsafe": len(test_triplets),
        "passes_gate": recall >= 0.99 and false_negatives == 0,
    }

    logger.info("Safety validation: recall=%.2f%%, precision=%.2f%%, FN=%d",
                recall * 100, precision * 100, int(false_negatives))
    return stats
