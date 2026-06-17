"""
Cortex — AI-native OS inference kernel.

Usage:
    python -m src detect              # Detect hardware and show optimal config
    python -m src detect --json        # Output as JSON
    python -m src tiers                # Show which model tiers fit on this system
    python -m src route "prompt"        # Route a prompt to the appropriate tier
    python -m src serve                # Launch the optimal inference server
    python -m src benchmark            # Benchmark TTFT against a running server
"""

import argparse
import json
import sys
import time

from .hardware_detect import (
    detect_system,
    get_simulated_profile,
    list_simulated_profiles,
)
from .backend_selector import select_backend
from .tiers import (
    Tier,
    assess_tiers,
    max_feasible_tier,
    concurrent_vram_budget,
    print_tier_report,
)
from .router import route_heuristic


def _get_profile(args):
    """Get system profile — real detection or simulated."""
    sim = getattr(args, "simulate", None)
    if sim:
        profile = get_simulated_profile(sim)
        if profile is None:
            available = ", ".join(list_simulated_profiles())
            print(f"Unknown profile: {sim!r}", file=sys.stderr)
            print(f"Available: {available}", file=sys.stderr)
            sys.exit(1)
        return profile
    return detect_system()


def cmd_detect(args):
    """Detect system hardware and recommend optimal config."""
    sim = getattr(args, "simulate", None)
    if sim:
        print(f"Simulating: {sim}\n", file=sys.stderr)
    else:
        print("Scanning system...\n", file=sys.stderr)
    t0 = time.monotonic()
    profile = _get_profile(args)
    scan_time = time.monotonic() - t0

    if args.json:
        output = {
            "system": profile.to_dict(),
            "scan_time_ms": round(scan_time * 1000, 1),
        }
        # Also compute the recommended config
        config = select_backend(profile)
        output["recommended"] = {
            "backend": config.backend.value,
            "model": config.model.model_id,
            "quant": config.model.quant.value,
            "max_context": config.model.max_context,
            "estimated_vram_mb": config.model.estimated_vram_mb,
            "prefix_caching": config.prefix_caching,
            "launch_command": config.to_launch_command(),
        }
        print(json.dumps(output, indent=2))
    else:
        print(profile.summary())
        print(f"\n(scanned in {scan_time*1000:.0f}ms)\n")
        print()
        config = select_backend(profile)
        print(config.summary())
        print()
        cmd = config.to_launch_command()
        print(f"Launch command:")
        print(f"  {' '.join(cmd)}")

        if config.extra_args.get("install_hint"):
            print(f"\n⚠️  No backend detected. Install one first:")
            print(f"  {config.extra_args['install_hint']}")


def cmd_serve(args):
    """Launch the inference server with optimal settings."""
    import subprocess

    print("Detecting system...")
    profile = detect_system()
    config = select_backend(profile)

    print(f"\n{config.summary()}\n")

    cmd = config.to_launch_command()
    print(f"Launching: {' '.join(cmd)}\n")

    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
    except FileNotFoundError:
        print(f"\nError: Could not find '{cmd[0]}'. Is the backend installed?")
        if config.extra_args.get("install_hint"):
            print(f"Install: {config.extra_args['install_hint']}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        proc.terminate()


def cmd_tiers(args):
    """Show which model tiers are feasible on this system."""
    sim = getattr(args, "simulate", None)
    if sim:
        print(f"Simulating: {sim}\n")
    else:
        print("Scanning system...\n")
    profile = _get_profile(args)
    print(profile.summary())
    print()
    report = print_tier_report(profile)
    print(report)

    if args.json:
        assessments = assess_tiers(profile)
        budget = concurrent_vram_budget(profile)
        output = {
            "tiers": [
                {
                    "tier": a.tier.name,
                    "label": a.spec.label,
                    "feasible": a.feasible,
                    "model": a.model.model_id if a.model else None,
                    "vram_mb": a.model.vram_mb if a.model else 0,
                    "ttft_target_ms": a.spec.ttft_target_ms,
                    "always_hot": a.spec.always_hot,
                    "reason": a.reason,
                }
                for a in assessments
            ],
            "max_local_tier": max_feasible_tier(profile).name,
            "concurrent_budget": budget,
        }
        print()
        print(json.dumps(output, indent=2))


def cmd_route(args):
    """Test routing a prompt to a tier."""
    prompt = args.prompt
    if not prompt:
        print("Error: provide a prompt to route")
        sys.exit(1)

    profile = _get_profile(args)
    max_tier = max_feasible_tier(profile)
    assessments = assess_tiers(profile)
    available = [a.tier for a in assessments if a.feasible]

    decision = route_heuristic(prompt, max_tier=max_tier, available_tiers=available)

    print(f"Prompt:     {prompt!r}")
    print(f"Route to:   {decision.tier.name}")
    print(f"Category:   {decision.category.value}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Reason:     {decision.reason}")
    if decision.escalation_hint:
        print(f"Escalation: Would benefit from {decision.escalation_hint.name}")

    if args.json:
        output = {
            "tier": decision.tier.name,
            "category": decision.category.value,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "escalation_hint": decision.escalation_hint.name if decision.escalation_hint else None,
        }
        print()
        print(json.dumps(output, indent=2))


def cmd_simulate_all(args):
    """Run detect + tiers across all simulated profiles."""
    profiles = list_simulated_profiles()
    divider = "=" * 64

    for name in profiles:
        profile = get_simulated_profile(name)
        print(f"{divider}")
        print(f"  PROFILE: {name}")
        print(f"{divider}")
        print(profile.summary())
        print()

        config = select_backend(profile)
        print(config.summary())
        print()

        report = print_tier_report(profile)
        print(report)
        print()
        print()


def cmd_benchmark(args):
    """Benchmark TTFT against a running server."""
    import urllib.request

    url = args.url or "http://localhost:8000"
    n = args.n or 5
    prompt = args.prompt or "Hello, how are you?"

    print(f"Benchmarking TTFT against {url}")
    print(f"Prompt: {prompt!r}")
    print(f"Iterations: {n}\n")

    endpoint = f"{url}/v1/chat/completions"
    payload = json.dumps({
        "model": args.model or "default",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 1,
    }).encode()

    ttfts = []
    for i in range(n):
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req) as resp:
                # Read until we get the first data chunk
                first_chunk = resp.read(1)
                ttft = (time.monotonic() - t0) * 1000  # ms
                ttfts.append(ttft)
                print(f"  Run {i+1}: {ttft:.1f}ms")
        except Exception as e:
            print(f"  Run {i+1}: ERROR - {e}")

    if ttfts:
        print(f"\nResults ({len(ttfts)} successful runs):")
        print(f"  Min TTFT:    {min(ttfts):.1f}ms")
        print(f"  Max TTFT:    {max(ttfts):.1f}ms")
        print(f"  Mean TTFT:   {sum(ttfts)/len(ttfts):.1f}ms")
        print(f"  Median TTFT: {sorted(ttfts)[len(ttfts)//2]:.1f}ms")
    else:
        print("\nNo successful runs. Is the server running?")


def main():
    parser = argparse.ArgumentParser(
        prog="cortex",
        description="Cortex — AI-native OS inference kernel",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Common --simulate argument
    sim_help = ("Simulate a system profile instead of detecting real hardware. "
                f"Options: {', '.join(list_simulated_profiles())}")

    # detect
    p_detect = sub.add_parser("detect", help="Detect hardware and recommend config")
    p_detect.add_argument("--json", action="store_true", help="Output as JSON")
    p_detect.add_argument("--simulate", "-s", metavar="PROFILE", help=sim_help)

    # tiers
    p_tiers = sub.add_parser("tiers", help="Show tier feasibility for this system")
    p_tiers.add_argument("--json", action="store_true", help="Output as JSON")
    p_tiers.add_argument("--simulate", "-s", metavar="PROFILE", help=sim_help)

    # route
    p_route = sub.add_parser("route", help="Test routing a prompt to a tier")
    p_route.add_argument("prompt", nargs="?", default=None, help="Prompt to route")
    p_route.add_argument("--json", action="store_true", help="Output as JSON")
    p_route.add_argument("--simulate", "-s", metavar="PROFILE", help=sim_help)

    # simulate-all
    sub.add_parser("simulate-all", help="Run all simulated profiles")

    # serve
    p_serve = sub.add_parser("serve", help="Launch optimal inference server")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Benchmark TTFT")
    p_bench.add_argument("--url", default="http://localhost:8000", help="Server URL")
    p_bench.add_argument("--n", type=int, default=5, help="Number of iterations")
    p_bench.add_argument("--prompt", default="Hello, how are you?", help="Test prompt")
    p_bench.add_argument("--model", default=None, help="Model name for API")

    args = parser.parse_args()

    if args.command == "detect":
        cmd_detect(args)
    elif args.command == "tiers":
        cmd_tiers(args)
    elif args.command == "route":
        cmd_route(args)
    elif args.command == "simulate-all":
        cmd_simulate_all(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
