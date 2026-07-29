"""
CKM-SM Training Orchestrator — runs all 6 phases in sequence.

Usage:
  python -m src.ckm.train_sm [--device cpu|cuda|mps] [--output-dir DIR] [--phases 1-6]

This orchestrates:
  Phase 1: Boot Simulator validation
  Phase 2: World Model training
  Phase 3: Safety Head contrastive pre-training
  Phase 4: Imitation Learning (behavioral cloning)
  Phase 5: Policy Gradient (REINFORCE)
  Phase 6: Self-Play (proposer/verifier/adversary)

Each phase saves checkpoints; later phases load from earlier ones.
The final artifact is a proposer model (CKM-SM) + verifier for deployment.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("cortex.ckm.train_sm")


def detect_device() -> str:
    """Auto-detect best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def run_phase_1(output_dir: str, verbose: bool = True) -> dict:
    """Phase 1: Validate boot simulator with expert policy."""
    from .simulator import validate, generate_expert_episodes

    logger.info("=" * 60)
    logger.info("PHASE 1: Boot Simulator Validation")
    logger.info("=" * 60)

    # Validate simulator physics
    stats = validate(n_episodes=500, verbose=verbose)

    # Generate expert episodes for later phases
    logger.info("Generating expert episodes for training...")
    episodes = generate_expert_episodes(n=2000, include_faults=True)

    # Save episodes
    ep_dir = os.path.join(output_dir, "episodes")
    os.makedirs(ep_dir, exist_ok=True)

    # Save as compact format
    episode_data = []
    for ep in episodes:
        episode_data.append({
            "outcome": ep.outcome,
            "total_reward": ep.total_reward,
            "steps": ep.steps,
            "time_ms": ep.time_ms,
            "transitions": ep.transitions,
        })

    import pickle
    with open(os.path.join(ep_dir, "expert_episodes.pkl"), "wb") as f:
        pickle.dump(episodes, f)

    stats["n_expert_episodes"] = len(episodes)
    stats["expert_success_rate"] = sum(1 for e in episodes if e.outcome == "success") / len(episodes)

    logger.info("Phase 1 complete: %d episodes generated, %.1f%% expert success rate",
                len(episodes), stats["expert_success_rate"] * 100)

    return stats


def run_phase_2(output_dir: str, device: str = "cpu") -> dict:
    """Phase 2: Train world model on expert episodes."""
    from .world_model import train_world_model, WorldModelTrainingConfig

    logger.info("=" * 60)
    logger.info("PHASE 2: World Model Training")
    logger.info("=" * 60)

    # Load expert episodes
    import pickle
    ep_path = os.path.join(output_dir, "episodes", "expert_episodes.pkl")
    with open(ep_path, "rb") as f:
        episodes = pickle.load(f)

    wm_dir = os.path.join(output_dir, "world_model")
    result = train_world_model(
        episodes,
        output_dir=wm_dir,
        device=device,
        config=WorldModelTrainingConfig(
            epochs=50,
            batch_size=64,
            learning_rate=3e-4,
            patience=10,
        ),
    )

    logger.info("Phase 2 complete: state_acc=%.1f%%", result["final_state_acc"] * 100)
    return result


def run_phase_3(output_dir: str, device: str = "cpu") -> dict:
    """Phase 3: Train safety head with contrastive learning."""
    from .safety_head import train_safety_head, SafetyTrainingConfig

    logger.info("=" * 60)
    logger.info("PHASE 3: Safety Head Contrastive Training")
    logger.info("=" * 60)

    sh_dir = os.path.join(output_dir, "safety_head")
    result = train_safety_head(
        output_dir=sh_dir,
        device=device,
        config=SafetyTrainingConfig(
            epochs=100,
            batch_size=64,
            n_triplets_per_category=200,
            patience=15,
        ),
    )

    logger.info("Phase 3 complete: recall=%.1f%%, precision=%.1f%%",
                result["best_recall"] * 100, result["final_precision"] * 100)
    return result


def run_phase_4(output_dir: str, device: str = "cpu") -> dict:
    """Phase 4: Imitation learning from expert demonstrations."""
    from .imitation import train_imitation, ImitationConfig

    logger.info("=" * 60)
    logger.info("PHASE 4: Imitation Learning")
    logger.info("=" * 60)

    # Load expert episodes
    import pickle
    ep_path = os.path.join(output_dir, "episodes", "expert_episodes.pkl")
    with open(ep_path, "rb") as f:
        episodes = pickle.load(f)

    il_dir = os.path.join(output_dir, "imitation")
    result = train_imitation(
        episodes,
        output_dir=il_dir,
        device=device,
        config=ImitationConfig(
            epochs=80,
            batch_size=64,
            patience=15,
        ),
    )

    logger.info("Phase 4 complete: verb=%.1f%%, target=%.1f%%",
                result["best_verb_acc"] * 100, result["best_target_acc"] * 100)
    return result


def run_phase_5(output_dir: str, device: str = "cpu") -> dict:
    """Phase 5: Policy gradient (REINFORCE) fine-tuning."""
    from .rl import train_rl, RLConfig
    from .imitation import load_policy
    from .safety_head import load_safety_head

    logger.info("=" * 60)
    logger.info("PHASE 5: Policy Gradient (REINFORCE)")
    logger.info("=" * 60)

    # Load imitation policy as starting point
    il_path = os.path.join(output_dir, "imitation", "policy_imitation_best.pt")
    policy = load_policy(il_path, device=device)

    # Load frozen safety head
    sh_path = os.path.join(output_dir, "safety_head", "safety_head_best.pt")
    safety_head = None
    if os.path.exists(sh_path):
        safety_head = load_safety_head(sh_path, device=device)
        safety_head.eval()
        for p in safety_head.parameters():
            p.requires_grad = False

    rl_dir = os.path.join(output_dir, "rl")
    result = train_rl(
        policy,
        output_dir=rl_dir,
        device=device,
        safety_head=safety_head,
        config=RLConfig(
            n_episodes=2000,
            batch_size=16,
            eval_every=50,
            patience=300,
        ),
    )

    logger.info("Phase 5 complete: success=%.1f%%", result["final_success_rate"] * 100)
    return result


def run_phase_6(output_dir: str, device: str = "cpu") -> dict:
    """Phase 6: Self-play hardening."""
    from .self_play import train_self_play, SelfPlayConfig
    from .imitation import load_policy
    from .safety_head import load_safety_head

    logger.info("=" * 60)
    logger.info("PHASE 6: Self-Play Hardening")
    logger.info("=" * 60)

    # Load RL-trained policy
    rl_path = os.path.join(output_dir, "rl", "policy_rl_best.pt")
    if not os.path.exists(rl_path):
        rl_path = os.path.join(output_dir, "rl", "policy_rl_final.pt")
    policy = load_policy(rl_path, device=device)

    # Load frozen safety head
    sh_path = os.path.join(output_dir, "safety_head", "safety_head_best.pt")
    safety_head = None
    if os.path.exists(sh_path):
        safety_head = load_safety_head(sh_path, device=device)
        safety_head.eval()

    sp_dir = os.path.join(output_dir, "self_play")
    result = train_self_play(
        policy,
        output_dir=sp_dir,
        device=device,
        safety_head=safety_head,
        config=SelfPlayConfig(
            n_rounds=500,
            episodes_per_round=16,
            eval_every=25,
            convergence_threshold=0.95,
        ),
    )

    logger.info("Phase 6 complete: proposer_win=%.1f%%, verifier_acc=%.1f%%",
                result["proposer_win_rate"] * 100, result["verifier_accuracy"] * 100)
    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def train_full_pipeline(
    output_dir: str = "/tmp/cortex-train/sm",
    device: str = "auto",
    phases: str = "1-6",
    verbose: bool = True,
) -> dict:
    """
    Run the complete CKM-SM training pipeline.

    Args:
        output_dir: base output directory
        device: compute device (auto, cpu, cuda, mps)
        phases: which phases to run (e.g., "1-6", "3-5", "6")
        verbose: detailed logging

    Returns: combined results from all phases
    """
    if device == "auto":
        device = detect_device()

    os.makedirs(output_dir, exist_ok=True)
    logger.info("CKM-SM Training Pipeline")
    logger.info("  Device: %s", device)
    logger.info("  Output: %s", output_dir)
    logger.info("  Phases: %s", phases)

    # Parse phases
    if "-" in phases:
        start, end = phases.split("-")
        phase_range = range(int(start), int(end) + 1)
    else:
        phase_range = [int(p) for p in phases.split(",")]

    results = {}
    t0 = time.time()

    phase_fns = {
        1: lambda: run_phase_1(output_dir, verbose=verbose),
        2: lambda: run_phase_2(output_dir, device=device),
        3: lambda: run_phase_3(output_dir, device=device),
        4: lambda: run_phase_4(output_dir, device=device),
        5: lambda: run_phase_5(output_dir, device=device),
        6: lambda: run_phase_6(output_dir, device=device),
    }

    for phase in phase_range:
        if phase in phase_fns:
            try:
                results[f"phase_{phase}"] = phase_fns[phase]()
            except Exception as e:
                logger.error("Phase %d failed: %s", phase, e)
                results[f"phase_{phase}"] = {"error": str(e)}
                break

    total_elapsed = time.time() - t0
    results["total_time_s"] = total_elapsed
    results["device"] = device
    results["output_dir"] = output_dir

    # Save combined results
    with open(os.path.join(output_dir, "pipeline_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Total time: %.1fs", total_elapsed)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    """CLI entry point for CKM-SM training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="CKM-SM State Machine Training Pipeline")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="Compute device")
    parser.add_argument("--output-dir", default="/tmp/cortex-train/sm",
                        help="Output directory for checkpoints")
    parser.add_argument("--phases", default="1-6",
                        help="Which phases to run (e.g., '1-6', '3-5', '6')")
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args(argv)

    results = train_full_pipeline(
        output_dir=args.output_dir,
        device=args.device,
        phases=args.phases,
        verbose=args.verbose,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("CKM-SM Training Summary")
    print("=" * 60)
    for key, val in results.items():
        if key.startswith("phase_") and isinstance(val, dict):
            phase_num = key.split("_")[1]
            if "error" in val:
                print(f"  Phase {phase_num}: FAILED — {val['error']}")
            else:
                # Extract key metric
                metrics = []
                for k in ["success_rate", "final_state_acc", "best_recall",
                          "best_verb_acc", "final_success_rate", "proposer_win_rate"]:
                    if k in val:
                        metrics.append(f"{k}={val[k]:.1%}")
                print(f"  Phase {phase_num}: {', '.join(metrics) or 'OK'}")
    print(f"  Total time: {results.get('total_time_s', 0):.1f}s")
    print(f"  Device: {results.get('device', 'unknown')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
