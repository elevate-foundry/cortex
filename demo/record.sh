#!/bin/bash
# Demo recording script for Cortex
# Usage: asciinema rec demo.cast -c "./demo/record.sh"
#
# Or for a scripted feel with typing simulation:
#   asciinema rec demo.cast -c "bash demo/record.sh"

set -e
cd "$(dirname "$0")/.."

# Simulated typing effect
type_cmd() {
    echo ""
    echo -n "$ "
    echo "$1" | while IFS= read -r -n1 char; do
        echo -n "$char"
        sleep 0.04
    done
    echo ""
    sleep 0.3
}

clear
echo "┌─────────────────────────────────────────────────┐"
echo "│  Cortex — AI as PID 1                           │"
echo "│  The inference engine isn't an app. It's init.  │"
echo "└─────────────────────────────────────────────────┘"
sleep 2

# 1. Hardware detection
type_cmd "python -m src detect"
python -m src detect
sleep 3

# 2. Tier feasibility
type_cmd "python -m src tiers"
python -m src tiers
sleep 3

# 3. Routing examples — show the tiering in action
type_cmd 'python -m src route "is it raining?"'
python -m src route "is it raining?"
sleep 1.5

type_cmd 'python -m src route "open the config file and read the database settings"'
python -m src route "open the config file and read the database settings"
sleep 1.5

type_cmd 'python -m src route "explain how TCP congestion control works and compare it to QUIC"'
python -m src route "explain how TCP congestion control works and compare it to QUIC"
sleep 1.5

type_cmd 'python -m src route "write a Python function that implements binary search"'
python -m src route "write a Python function that implements binary search"
sleep 1.5

type_cmd 'python -m src route "debug this segfault: the code crashes when refactoring the shared pointer design pattern"'
python -m src route "debug this segfault: the code crashes when refactoring the shared pointer design pattern"
sleep 1.5

type_cmd 'python -m src route "analyze the security vulnerability in this SQL injection exploit"'
python -m src route "analyze the security vulnerability in this SQL injection exploit"
sleep 2

# 4. Model discovery
type_cmd "python -m src models"
python -m src models 2>&1 | head -35
sleep 3

echo ""
echo "─────────────────────────────────────────────────────"
echo "  ✓ Hardware scanned → model tiers mapped to VRAM"
echo "  ✓ Simple questions → L0 (0.6B, ~10ms)"
echo "  ✓ Tool calls → L1 (1.7B, ~20ms)"
echo "  ✓ Analysis → L3 (8B, ~60ms)"
echo "  ✓ Code/debug/safety → L4 (14B, ~100ms)"
echo "  ✓ If they disagree → cross-family challenger swarm"
echo ""
echo "  github.com/elevate-foundry/cortex"
echo "─────────────────────────────────────────────────────"
sleep 4
