#!/bin/bash
# One stream of permittivity-channel shape solves.
# Usage: run_eps_stream.sh <stream-name> <shape> ...
cd "$(dirname "$0")" || exit 1
source "$(dirname "$0")/env1.sh"
PY=../.venv312/bin/python
STREAM="$1"; shift
mkdir -p out_eps logs_eps
for SHAPE in "$@"; do
  echo "=== $STREAM $SHAPE start $(date +%H:%M:%S) ==="
  $PY -m adjoint2d.eps_solve "$SHAPE" out_eps > "logs_eps/${SHAPE}.log" 2>&1 \
    || echo "FAILED $SHAPE"
  tail -3 "logs_eps/${SHAPE}.log"
  echo "=== $STREAM $SHAPE end $(date +%H:%M:%S) ==="
done
echo "=== $STREAM DONE $(date +%H:%M:%S) ==="
