"""
Cortex — AI-native OS inference kernel.

Usage:
    python -m src detect              # Detect hardware and show optimal config
    python -m src detect --json        # Output as JSON
    python -m src tiers                # Show which model tiers fit on this system
    python -m src models              # Show all discovered Ollama models mapped to tiers
    python -m src route "prompt"        # Route a prompt to the appropriate tier
    python -m src serve                # Launch the optimal inference server
    python -m src smoke               # Smoke test the daemon
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


def cmd_models(args):
    """Show all Ollama models mapped to tiers."""
    from .tiers import print_discovered_models
    print(print_discovered_models())


def cmd_smoke(args):
    """Smoke test the daemon — probes API compatibility, routing, and response shape."""
    import urllib.request

    base_url = args.url.rstrip("/")
    passed = 0
    failed = 0

    def probe(name, method, path, body=None, expect_keys=None, expect_status=200):
        nonlocal passed, failed
        url = f"{base_url}{path}"
        print(f"\n{'─' * 50}")
        print(f"TEST: {name}")
        print(f"  {method} {url}")

        try:
            data = json.dumps(body).encode() if body else None
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"} if body else {},
                method=method,
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = resp.status
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            status = e.code
            try:
                result = json.loads(e.read())
            except Exception:
                result = {}
        except urllib.error.URLError as e:
            print(f"  ✗ FAIL — cannot connect: {e.reason}")
            print(f"  Is the daemon running? Start with: python -m src daemon")
            failed += 1
            return None
        except Exception as e:
            print(f"  ✗ FAIL — {e}")
            failed += 1
            return None

        if status != expect_status:
            print(f"  ✗ FAIL — expected {expect_status}, got {status}")
            failed += 1
            return result

        if expect_keys:
            missing = [k for k in expect_keys if k not in result]
            if missing:
                print(f"  ✗ FAIL — missing keys: {missing}")
                print(f"  Got: {json.dumps(result, indent=2)[:300]}")
                failed += 1
                return result

        print(f"  ✓ PASS (status={status})")
        if body and "messages" in body:
            routing = result.get("_routing", {})
            if routing:
                print(f"  Routed to: {routing.get('tier', '?')} "
                      f"({routing.get('category', '?')}, "
                      f"conf={routing.get('confidence', '?')})")
                print(f"  Backend model: {routing.get('backend_model', '?')}")

        passed += 1
        return result

    print("=" * 50)
    print("  CORTEX SMOKE TEST")
    print(f"  Target: {base_url}")
    print("=" * 50)

    # 1. Health check
    probe("Health endpoint", "GET", "/health",
          expect_keys=["status", "uptime_seconds"])

    # 2. Models list
    probe("List models", "GET", "/v1/models",
          expect_keys=["object", "data"])

    # 3. Status endpoint
    probe("Status endpoint", "GET", "/status",
          expect_keys=["daemon", "system", "cortex", "semantic"])

    # 3b. SCL audit endpoint
    probe("SCL audit endpoint", "GET", "/v1/audit/scl",
          expect_keys=["object", "data"])

    # 4. API compatibility — minimal ping
    probe("Chat completion (ping)", "POST", "/v1/chat/completions",
          body={
              "model": "auto",
              "messages": [{"role": "user", "content": "Say pong."}],
              "max_tokens": 5,
          },
          expect_keys=["id", "object", "choices"])

    # 5. Routing test — code task
    probe("Routing: code task", "POST", "/v1/chat/completions",
          body={
              "model": "auto",
              "messages": [{"role": "user", "content":
                  "Write a Python function that implements a red-black tree "
                  "with insert, delete, and rebalance operations. Include type hints."}],
              "max_tokens": 50,
          },
          expect_keys=["id", "choices"])

    # 6. Routing test — simple question
    probe("Routing: simple question", "POST", "/v1/chat/completions",
          body={
              "model": "auto",
              "messages": [{"role": "user", "content": "What is 2+2?"}],
              "max_tokens": 10,
          },
          expect_keys=["id", "choices"])

    # 7. Responses API
    probe("Responses API translation", "POST", "/v1/responses",
          body={
              "model": "auto",
              "input": "Say hello.",
              "max_output_tokens": 10,
          },
          expect_keys=["id", "object", "output_text"])

    # 8. Anthropic Messages API
    probe("Anthropic Messages API translation", "POST", "/v1/messages",
          body={
              "model": "auto",
              "system": "You are helpful.",
              "messages": [{"role": "user", "content": "Say hi."}],
              "max_tokens": 10,
          },
          expect_keys=["id", "type", "content"])

    # 9. Custom prompt if provided
    if args.prompt:
        probe(f"Custom prompt", "POST", "/v1/chat/completions",
              body={
                  "model": "auto",
                  "messages": [{"role": "user", "content": args.prompt}],
                  "max_tokens": 100,
              },
              expect_keys=["id", "choices"])

    # Summary
    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  All smoke tests passed!")
    else:
        print(f"  {failed} test(s) failed.")
    print(f"{'=' * 50}")

    sys.exit(1 if failed else 0)


def cmd_scl(args):
    """SCL operations."""
    if args.scl_command == "audit":
        from .memory import Memory
        mem = Memory()
        entries = mem.get_scl_audit(limit=args.last)
        print(f"{'Last':<6} {'Fingerprint':<10} {'Created':<18} SCL")
        print("-" * 60)
        for e in entries:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.created_at / 1000))
            preview = e.scl_text.replace("\n", " | ")[:80]
            print(f"{e.fingerprint or '⣿⣿⣿⣿':<6} {ts:<18} {preview}")
    else:
        print("Usage: python -m src scl audit [--last N]")


def cmd_init(args):
    """Run Cortex as PID 1 (init replacement)."""
    import os
    from .daemon import DaemonServer
    from .hardware_detect import detect_system

    if os.getpid() != 1 and not args.force:
        print("Warning: Not running as PID 1. Use --force to override.")
        print("For dry-run: python -m src init --dry-run")
        sys.exit(1)

    profile = detect_system()
    daemon = DaemonServer(
        host="0.0.0.0",
        port=args.port,
        profile=profile,
    )
    import asyncio
    asyncio.run(daemon.start())


def cmd_feedback(args):
    """Send routing feedback to the Cortex daemon."""
    import urllib.request

    payload = json.dumps({
        "request_id": args.request_id or "",
        "thread_id": args.thread_id or "",
        "category": args.category or "",
        "routed_tier": args.tier or "",
        "actual_model": args.model or "",
        "predicted_correct": 1 if args.predicted_correct else 0,
        "user_correct": args.user_correct,
        "tool_success": 1 if args.tool_success else 0,
        "latency_ms": args.latency_ms or 0.0,
    }).encode()

    req = urllib.request.Request(
        f"{args.url}/v1/feedback",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            print(f"Feedback recorded: {result.get('feedback_id')}")
    except urllib.error.HTTPError as e:
        print(f"Error: HTTP {e.code}")
    except Exception as e:
        print(f"Error: {e}")


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

    # daemon
    p_daemon = sub.add_parser("daemon", help="Start the OS-level inference proxy daemon")
    p_daemon.add_argument("--host", default="127.0.0.1", help="Bind address")
    p_daemon.add_argument("--port", type=int, default=11411, help="Proxy port (default: 11411)")
    p_daemon.add_argument("--simulate", "-s", metavar="PROFILE", help=sim_help)

    # models
    sub.add_parser("models", help="Show all Ollama models mapped to tiers")

    # serve
    p_serve = sub.add_parser("serve", help="Launch optimal inference server (single model)")

    # smoke
    p_smoke = sub.add_parser("smoke", help="Smoke test the daemon (API compatibility probe)")
    p_smoke.add_argument("--url", default="http://localhost:11411", help="Daemon URL")
    p_smoke.add_argument("--prompt", default=None, help="Custom prompt")

    # init (PID-1 mode)
    p_init = sub.add_parser("init", help="Run Cortex as PID 1 (boot into AI-native OS)")
    p_init.add_argument("--port", type=int, default=11411, help="Daemon port")
    p_init.add_argument("--force", action="store_true", help="Run even if not PID 1")
    p_init.add_argument("--dry-run", action="store_true", help="Print what would happen")

    # feedback
    p_feedback = sub.add_parser("feedback", help="Send routing feedback to daemon")
    p_feedback.add_argument("--url", default="http://localhost:11411", help="Daemon URL")
    p_feedback.add_argument("--request-id", help="Request ID to feedback on")
    p_feedback.add_argument("--thread-id", help="Thread ID")
    p_feedback.add_argument("--category", help="Task category")
    p_feedback.add_argument("--tier", help="Routed tier")
    p_feedback.add_argument("--model", help="Actual model used")
    p_feedback.add_argument("--predicted-correct", action="store_true", help="We predicted success")
    p_feedback.add_argument("--user-correct", type=int, choices=[0, 1], help="User confirms correctness")
    p_feedback.add_argument("--tool-success", action="store_true", help="Tools executed successfully")
    p_feedback.add_argument("--latency-ms", type=float, default=0.0, help="Request latency")

    # gossip
    p_gossip = sub.add_parser("gossip", help="Multi-host gossip protocol")
    gossip_sub = p_gossip.add_subparsers(dest="gossip_command", help="Gossip subcommand")
    p_gossip_add = gossip_sub.add_parser("add", help="Add a gossip peer")
    p_gossip_add.add_argument("--id", required=True, help="Peer node ID")
    p_gossip_add.add_argument("--url", required=True, help="Peer URL (http://host:port)")
    p_gossip_add.add_argument("--daemon-url", default="http://localhost:11411", help="Cortex daemon URL")
    p_gossip_list = gossip_sub.add_parser("list", help="List gossip peers")
    p_gossip_list.add_argument("--daemon-url", default="http://localhost:11411", help="Cortex daemon URL")
    p_gossip_state = gossip_sub.add_parser("state", help="Show local gossip state (fingerprint)")
    p_gossip_state.add_argument("--daemon-url", default="http://localhost:11411", help="Cortex daemon URL")
    p_gossip_stats = gossip_sub.add_parser("stats", help="Show gossip statistics")
    p_gossip_stats.add_argument("--daemon-url", default="http://localhost:11411", help="Cortex daemon URL")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Benchmark TTFT")
    p_bench.add_argument("--url", default="http://localhost:8000", help="Server URL")
    p_bench.add_argument("--n", type=int, default=5, help="Number of iterations")
    p_bench.add_argument("--prompt", default="Hello, how are you?", help="Test prompt")
    p_bench.add_argument("--model", default=None, help="Model name for API")

    # scl
    p_scl = sub.add_parser("scl", help="SCL operations")
    scl_sub = p_scl.add_subparsers(dest="scl_command", help="SCL subcommand")
    p_scl_audit = scl_sub.add_parser("audit", help="Show SCL audit trail")
    p_scl_audit.add_argument("--last", type=int, default=20, help="Number of entries to show")

    args = parser.parse_args()

def cmd_gossip(args):
    """Gossip CLI commands."""
    import urllib.request

    if args.gossip_command == "add":
        payload = json.dumps({"id": args.id, "url": args.url}).encode()
        req = urllib.request.Request(
            f"{args.daemon_url}/v1/gossip/peers",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                print(f"Peer added: {result.get('peer_id')}")
        except Exception as e:
            print(f"Error: {e}")

    elif args.gossip_command == "list":
        req = urllib.request.Request(f"{args.daemon_url}/v1/gossip/peers")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                peers = result.get("data", [])
                print(f"Known peers: {len(peers)}")
                for p in peers:
                    print(f"  {p['id']:<20s} {p['url']:<30s} last_sync={p.get('last_sync_seconds_ago', '?')}s")
        except Exception as e:
            print(f"Error: {e}")

    elif args.gossip_command == "state":
        req = urllib.request.Request(f"{args.daemon_url}/v1/gossip/state")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                print(f"Node:      {result['node_id']}")
                print(f"Fingerprint: {result['fingerprint']}")
                print(f"State keys:  {result['state_keys']}")
                print(f"Stream:      {result['stream_length']} deltas")
        except Exception as e:
            print(f"Error: {e}")

    elif args.gossip_command == "stats":
        req = urllib.request.Request(f"{args.daemon_url}/v1/gossip/stats")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python -m src gossip {add|list|state|stats} ...")


    if args.command == "init":
        if args.dry_run:
            print("[DRY-RUN] Would boot Cortex as PID 1")
            print("  port:", args.port)
            print("  host: 0.0.0.0")
            sys.exit(0)
        cmd_init(args)
    elif args.command == "feedback":
        cmd_feedback(args)
    elif args.command == "gossip":
        cmd_gossip(args)
    elif args.command == "detect":
        cmd_detect(args)
    elif args.command == "tiers":
        cmd_tiers(args)
    elif args.command == "route":
        cmd_route(args)
    elif args.command == "simulate-all":
        cmd_simulate_all(args)
    elif args.command == "models":
        cmd_models(args)
    elif args.command == "daemon":
        from .daemon import run_daemon
        profile = _get_profile(args)
        run_daemon(host=args.host, port=args.port, profile=profile)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "smoke":
        cmd_smoke(args)
    elif args.command == "scl":
        cmd_scl(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
