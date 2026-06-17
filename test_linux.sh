#!/usr/bin/env bash
set -euo pipefail

IMAGE="cortex-test"
DIVIDER="================================================================"

echo "$DIVIDER"
echo "Building Linux test image..."
echo "$DIVIDER"
docker build -t "$IMAGE" .

echo ""
echo "$DIVIDER"
echo "TEST 1: detect (Linux CPU-only)"
echo "$DIVIDER"
docker run --rm "$IMAGE" python -m src detect

echo ""
echo "$DIVIDER"
echo "TEST 2: detect --json (Linux CPU-only)"
echo "$DIVIDER"
docker run --rm "$IMAGE" python -m src detect --json 2>/dev/null | python3 -m json.tool | head -30

echo ""
echo "$DIVIDER"
echo "TEST 3: tiers (Linux CPU-only)"
echo "$DIVIDER"
docker run --rm "$IMAGE" python -m src tiers

echo ""
echo "$DIVIDER"
echo "TEST 4: route - simple query"
echo "$DIVIDER"
docker run --rm "$IMAGE" python -m src route "is the sky blue?"

echo ""
echo "$DIVIDER"
echo "TEST 5: route - complex coding task"
echo "$DIVIDER"
docker run --rm "$IMAGE" python -m src route "refactor the database layer to use connection pooling and add retry logic"

echo ""
echo "$DIVIDER"
echo "TEST 6: route - multi-step planning"
echo "$DIVIDER"
docker run --rm "$IMAGE" python -m src route "design a CI pipeline, then set up Docker builds, then deploy to k8s"

echo ""
echo "$DIVIDER"
echo "ALL TESTS PASSED on Linux ($(docker run --rm "$IMAGE" python -c 'import platform; print(platform.system(), platform.machine())'))"
echo "$DIVIDER"
