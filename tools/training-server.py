#!/usr/bin/env python3
"""
CKM Training Dashboard Server

Lightweight HTTP server that:
  1. Monitors Modal training output (via subprocess)
  2. Parses loss/step/epoch from log lines
  3. Serves a real-time dashboard with SSE updates

Usage:
    python tools/training-server.py              # Just serve dashboard (poll metrics file)
    python tools/training-server.py --train      # Start training AND serve dashboard

Open http://localhost:8091 in your browser.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from queue import Queue, Empty

PORT = 8091
METRICS_FILE = Path(__file__).parent.parent / "metrics_live.json"
LOG_QUEUE = Queue(maxsize=1000)
METRICS = []
STATE = {
    "status": "idle",  # idle | running | complete | failed
    "step": 0,
    "loss": None,
    "epoch": 0,
    "total_steps": 0,
    "elapsed_s": 0,
    "samples_per_sec": 0,
    "start_time": None,
}


def parse_training_line(line: str):
    """Extract metrics from a training log line."""
    # Match: "  [step  123] loss=0.4012 | lr=2.00e-04 | epoch=1.23 | 45 samples/s | GPU=12.3GB"
    step_match = re.search(r'\[step\s+(\d+)\]', line)
    if not step_match:
        # Match: " 123] loss=0.4012 | ..."
        step_match = re.search(r'(\d+)\]\s*loss=', line)
    if not step_match:
        return None

    step = int(step_match.group(1))
    loss_match = re.search(r'loss=([\d.]+)', line)
    lr_match = re.search(r'lr=([\d.e+-]+)', line)
    epoch_match = re.search(r'epoch=([\d.]+)', line)
    speed_match = re.search(r'([\d.]+)\s*samples/s', line)

    return {
        "step": step,
        "loss": float(loss_match.group(1)) if loss_match else None,
        "lr": lr_match.group(1) if lr_match else None,
        "epoch": float(epoch_match.group(1)) if epoch_match else None,
        "samples_per_sec": float(speed_match.group(1)) if speed_match else None,
        "timestamp": time.time(),
    }


def run_training():
    """Run modal training and capture output."""
    STATE["status"] = "running"
    STATE["start_time"] = time.time()
    LOG_QUEUE.put({"type": "log", "msg": "Starting Modal training pipeline..."})

    proc = subprocess.Popen(
        [sys.executable, "-m", "modal", "run", "src/ckm/modal_train.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )

    for line in iter(proc.stdout.readline, ''):
        line = line.rstrip()
        if not line:
            continue

        # Parse metrics
        metric = parse_training_line(line)
        if metric and metric["loss"] is not None:
            METRICS.append(metric)
            STATE["step"] = metric["step"]
            STATE["loss"] = metric["loss"]
            STATE["epoch"] = metric["epoch"] or 0
            STATE["elapsed_s"] = time.time() - STATE["start_time"]
            STATE["samples_per_sec"] = metric["samples_per_sec"] or 0

            LOG_QUEUE.put({"type": "metric", **metric})

            # Write to file for persistence
            with open(METRICS_FILE, "w") as f:
                json.dump(METRICS, f)

        # Forward log line
        # Clean ANSI codes
        clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
        clean = re.sub(r'\r', '', clean)
        if clean.strip():
            LOG_QUEUE.put({"type": "log", "msg": clean[:200]})

        # Detect total steps
        total_match = re.search(r'Total training steps:\s*~?(\d+)', line)
        if total_match:
            STATE["total_steps"] = int(total_match.group(1))

    proc.wait()
    STATE["status"] = "complete" if proc.returncode == 0 else "failed"
    LOG_QUEUE.put({"type": "log", "msg": f"Training {'complete' if proc.returncode == 0 else 'FAILED'}!"})


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the training dashboard."""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.serve_dashboard()
        elif self.path == "/status":
            self.serve_json(STATE)
        elif self.path == "/metrics":
            self.serve_json(METRICS)
        elif self.path == "/logs":
            self.serve_sse()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/train":
            if STATE["status"] != "running":
                thread = threading.Thread(target=run_training, daemon=True)
                thread.start()
                self.serve_json({"started": True})
            else:
                self.serve_json({"started": False, "reason": "already running"})
        else:
            self.send_error(404)

    def serve_dashboard(self):
        dashboard_path = Path(__file__).parent / "training-dashboard.html"
        content = dashboard_path.read_bytes()
        # Inject the correct API base URL
        content = content.replace(
            b"const API_BASE = 'http://127.0.0.1:8091'",
            f"const API_BASE = 'http://127.0.0.1:{PORT}'".encode()
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(content))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def serve_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            while True:
                try:
                    msg = LOG_QUEUE.get(timeout=5)
                    data = json.dumps(msg)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                except Empty:
                    # Send keepalive
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass  # Suppress HTTP access logs


def load_existing_metrics():
    """Load metrics from a previous run if available."""
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE) as f:
                data = json.load(f)
            METRICS.extend(data)
            if data:
                last = data[-1]
                STATE["step"] = last.get("step", 0)
                STATE["loss"] = last.get("loss")
                STATE["epoch"] = last.get("epoch", 0)
                STATE["status"] = "complete"
                print(f"  Loaded {len(data)} metrics from previous run")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="CKM Training Dashboard Server")
    parser.add_argument("--train", action="store_true", help="Start training immediately")
    parser.add_argument("--port", type=int, default=PORT, help=f"Server port (default: {PORT})")
    args = parser.parse_args()

    port = args.port

    load_existing_metrics()

    if args.train:
        print("  Starting training in background...")
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()

    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"""
╔══════════════════════════════════════════════════╗
║  CKM Training Dashboard                         ║
║  http://localhost:{port}                          ║
╠══════════════════════════════════════════════════╣
║  Status: {'Training...' if args.train else 'Idle (click Run Training)'}{'  ' if args.train else ''}║
║  Press Ctrl+C to stop                            ║
╚══════════════════════════════════════════════════╝
""")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
