"""
CKM CLI — cortex train subcommand.

Usage:
  cortex train --target ckm --time-budget 10m --from-scratch --auto-profile
  cortex train --target ckm --model ckm-5m --epochs 3
  cortex train --eval /path/to/model
  cortex train --rollback
  cortex train --status

The self-training loop:
  1. Probe hardware
  2. Pick model size
  3. Pick batch size / accumulation
  4. Load cached dataset (or generate)
  5. Train candidate
  6. Run eval suite
  7. Stage if better
  8. Keep rollback
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("cortex.ckm.cli")


def parse_time_budget(value: str) -> float:
    """Parse time budget string like '10m', '1h', '30s' to minutes."""
    value = value.strip().lower()
    if value.endswith("m"):
        return float(value[:-1])
    elif value.endswith("h"):
        return float(value[:-1]) * 60
    elif value.endswith("s"):
        return float(value[:-1]) / 60
    else:
        return float(value)


def cmd_train(args) -> int:
    """Execute the full self-training loop."""
    from .profile import detect_hardware, select_training_profile
    from .dataset import TokenizedDataset
    from .data_generator import CKMDataset
    from .train_scratch import train_ckm
    from .promote import ModelRegistry

    time_budget = parse_time_budget(args.time_budget)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Cortex Self-Training Loop                                   ║")
    print(f"║  Target: CKM/SCL control model                               ║")
    print(f"║  Time budget: {time_budget:.1f} minutes                                    ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print()

    # Step 1: Probe hardware
    print("[1/8] Probing hardware...")
    hw = detect_hardware()
    print(f"  CPU: {hw.cpu_cores} cores, {hw.ram_mb}MB RAM")
    print(f"  GPU: {hw.gpu_name} ({hw.gpu_type}, {hw.vram_mb}MB VRAM)")
    print()

    # Step 2: Pick training profile
    print("[2/8] Selecting training profile...")
    profile = select_training_profile(
        hardware=hw,
        time_budget_minutes=time_budget,
        target_model=args.model if args.model != "auto" else None,
    )
    print(f"  Model: {profile.model_spec.name} ({profile.model_spec.params:,} params)")
    print(f"  Device: {profile.device}, Precision: {profile.precision}")
    print(f"  Batch: {profile.batch_size} × {profile.gradient_accumulation} accum")
    print(f"  Max epochs: {profile.max_epochs}")
    print()

    # Step 3: Prepare dataset
    print("[3/8] Preparing dataset...")
    jsonl_path = data_dir / "ckm_training.jsonl"
    tokenized_dir = data_dir / "tokenized"

    # Generate or load JSONL
    if not jsonl_path.exists() or args.regenerate_data:
        print("  Generating training data...")
        dataset = CKMDataset(output_dir=str(data_dir))
        n_added = dataset.add_synthetic(
            boot_count=args.boot_count,
            route_count=args.route_count,
        )
        # Add trace data if available
        try:
            from .trace_generator import save_trace_corpus
            trace_path = data_dir / "trace_corpus.jsonl"
            stats = save_trace_corpus(str(trace_path), instances_per_template=10)
            # Merge trace data into main dataset
            with open(trace_path) as f:
                for line in f:
                    if line.strip():
                        pair = json.loads(line)
                        from .data_generator import TrainingPair
                        tp = TrainingPair(
                            input_scl=pair["input"],
                            output_scl=pair["output"],
                            source=pair.get("source", "trace"),
                            quality=pair.get("quality", 0.9),
                        )
                        dataset.pairs.append(tp)
            print(f"  Added {stats['total_pairs']} trace pairs")
        except Exception as e:
            print(f"  Trace generation skipped: {e}")

        path = dataset.save(filename="ckm_training.jsonl")
        print(f"  Dataset: {len(dataset.pairs)} pairs → {path}")
    else:
        print(f"  Using cached dataset: {jsonl_path}")

    # Build tokenized mmap cache
    needs_retokenize = not (tokenized_dir / "header.json").exists()
    if not needs_retokenize:
        try:
            existing = TokenizedDataset.load(str(tokenized_dir))
            if existing.is_stale(str(jsonl_path)):
                needs_retokenize = True
            existing.close()
        except Exception:
            needs_retokenize = True

    if needs_retokenize:
        print("  Tokenizing dataset (cached for future runs)...")
        tok_dataset = TokenizedDataset.build(
            jsonl_path=str(jsonl_path),
            output_dir=str(tokenized_dir),
            seq_len=profile.model_spec.max_seq_len,
            vocab_size=profile.model_spec.vocab_size,
        )
        print(f"  Tokenized: {tok_dataset.stats()}")
    else:
        print("  Using cached tokenized dataset")
    print()

    # Step 4: Train
    print("[4/8] Training model...")
    print(f"  This may take up to {time_budget:.1f} minutes...")
    print()

    candidate_dir = output_dir / "candidate"
    results = train_ckm(
        dataset_dir=str(tokenized_dir),
        output_dir=str(candidate_dir),
        device=profile.device,
        precision=profile.precision,
        batch_size=profile.batch_size,
        gradient_accumulation=profile.gradient_accumulation,
        learning_rate=profile.learning_rate,
        max_epochs=profile.max_epochs,
        time_budget_minutes=time_budget * 0.7,  # Reserve 30% for eval
        early_stop_patience=profile.early_stop_patience,
        model_name=profile.model_spec.name,
    )

    print()
    print(f"  Training complete: {results['total_steps']} steps, "
          f"{results['elapsed_minutes']:.1f}min, loss={results['final_loss']:.4f}")
    if results.get("stopped_early"):
        print("  (stopped early — eval plateaued)")
    print()

    # Step 5: Eval gate
    print("[5/8] Running eval gate...")
    from .eval import evaluate_model
    eval_report = evaluate_model(
        model_path=str(candidate_dir / "best_model.pt"),
        dataset_dir=str(tokenized_dir),
        device=profile.device,
    )
    print(f"  Gate: {'PASS' if eval_report.passed_gate else 'FAIL'}")
    for cat, metrics in eval_report.metrics.items():
        if isinstance(metrics, dict) and "pass_rate" in metrics:
            print(f"    {cat}: {metrics['pass_rate']:.1%} ({metrics['passed']}/{metrics['total']})")
    print()

    # Step 6: Stage and compare
    print("[6/8] Staging candidate...")
    registry = ModelRegistry(base_dir=str(output_dir / "registry"))
    verdict = registry.stage_candidate(
        model_dir=str(candidate_dir),
        dataset_dir=str(tokenized_dir),
        device=profile.device,
    )
    print(f"  Verdict: {verdict['verdict'].upper()}")
    print(f"  Reason: {verdict.get('reason', 'n/a')}")
    if "regressions" in verdict:
        for reg in verdict["regressions"]:
            print(f"    Regression: {reg}")
    print()

    # Step 7-8: Report
    if verdict["verdict"] == "promote":
        print("[7/8] Model promoted!")
        print(f"  Version: v{verdict['version']:03d}")
        print(f"  Path: {verdict.get('model_path', 'n/a')}")
        print()
        print("[8/8] Rollback available")
        prev = registry.previous_version()
        if prev:
            print(f"  Previous: v{prev.version:03d} ({prev.model_name})")
        else:
            print("  (first deployment, no rollback)")
    else:
        print("[7/8] Model rejected — incumbent retained")
        print("[8/8] No deployment change")

    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Result: {verdict['verdict'].upper()}")
    print(f"  Total time: {results['elapsed_minutes']:.1f} min")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return 0 if verdict["verdict"] == "promote" else 1


def cmd_eval(args) -> int:
    """Evaluate an existing model against the gate."""
    from .eval import evaluate_model

    print(f"Evaluating: {args.model_path}")
    report = evaluate_model(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        device=args.device,
    )

    print(f"\nGate: {'PASS' if report.passed_gate else 'FAIL'}")
    print(f"\nMetrics:")
    print(json.dumps(report.metrics, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nReport saved: {args.output}")

    return 0 if report.passed_gate else 1


def cmd_rollback(args) -> int:
    """Rollback to previous model version."""
    from .promote import ModelRegistry

    registry = ModelRegistry(base_dir=args.registry_dir)
    result = registry.rollback(reason=args.reason)

    if result:
        print(f"Rolled back: v{result['from_version']:03d} → v{result['to_version']:03d}")
        print(f"Reason: {result['reason']}")
        return 0
    else:
        print("No previous version available for rollback")
        return 1


def cmd_status(args) -> int:
    """Show current model registry status."""
    from .promote import ModelRegistry

    registry = ModelRegistry(base_dir=args.registry_dir)
    status = registry.status()

    print("CKM Model Registry")
    print("━━━━━━━━━━━━━━━━━━")
    print(f"  Current:  v{status['current']['version']:03d} ({status['current']['model']})"
          if status['current']['version'] else "  Current:  none")
    print(f"  Previous: v{status['previous']['version']:03d} ({status['previous']['model']})"
          if status['previous']['version'] else "  Previous: none")
    print(f"  Total versions: {status['total_versions']}")
    print(f"  Registry: {status['base_dir']}")

    return 0


def cmd_sm(args) -> int:
    """Run the CKM-SM state machine training pipeline."""
    from .train_sm import train_full_pipeline
    results = train_full_pipeline(
        output_dir=args.output_dir,
        device=args.device,
        phases=args.phases,
    )
    return 0 if "error" not in str(results) else 1


def main(argv=None) -> int:
    """CKM CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="cortex train",
        description="Cortex self-training loop — train, eval, promote CKM models",
    )
    sub = parser.add_subparsers(dest="command")

    # Train subcommand
    train_parser = sub.add_parser("run", help="Run the full self-training loop")
    train_parser.add_argument("--target", default="ckm", choices=["ckm"],
                             help="Training target (default: ckm)")
    train_parser.add_argument("--time-budget", default="10m",
                             help="Time budget (e.g. '10m', '1h', '30s')")
    train_parser.add_argument("--model", default="auto",
                             help="Model size (auto, ckm-1m, ckm-5m, ckm-15m, ckm-30m, ckm-60m)")
    train_parser.add_argument("--data-dir", default="/tmp/cortex-train/data",
                             help="Directory for training data")
    train_parser.add_argument("--output-dir", default="/tmp/cortex-train/output",
                             help="Directory for model outputs")
    train_parser.add_argument("--boot-count", type=int, default=1000,
                             help="Synthetic boot pairs to generate")
    train_parser.add_argument("--route-count", type=int, default=2000,
                             help="Synthetic routing pairs to generate")
    train_parser.add_argument("--regenerate-data", action="store_true",
                             help="Force regenerate training data")
    train_parser.set_defaults(func=cmd_train)

    # Eval subcommand
    eval_parser = sub.add_parser("eval", help="Evaluate a model against the gate")
    eval_parser.add_argument("model_path", help="Path to model checkpoint")
    eval_parser.add_argument("--dataset-dir", required=True,
                            help="Path to tokenized dataset")
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--output", help="Save report to JSON file")
    eval_parser.set_defaults(func=cmd_eval)

    # Rollback subcommand
    rb_parser = sub.add_parser("rollback", help="Rollback to previous model")
    rb_parser.add_argument("--registry-dir", default="/tmp/cortex-train/output/registry")
    rb_parser.add_argument("--reason", default="operator_override")
    rb_parser.set_defaults(func=cmd_rollback)

    # Status subcommand
    status_parser = sub.add_parser("status", help="Show model registry status")
    status_parser.add_argument("--registry-dir", default="/tmp/cortex-train/output/registry")
    status_parser.set_defaults(func=cmd_status)

    # SM (state machine) subcommand — new optimized training regime
    sm_parser = sub.add_parser("sm", help="Run CKM-SM state machine training (RL + self-play)")
    sm_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    sm_parser.add_argument("--output-dir", default="/tmp/cortex-train/sm")
    sm_parser.add_argument("--phases", default="1-6",
                          help="Which phases to run (e.g., '1-6', '3-5', '6')")
    sm_parser.set_defaults(func=cmd_sm)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
