#!/bin/bash
# One stream of multi-start shape solves. Usage: run_ms_stream.sh <stream-name> <shape> ...
cd "$(dirname "$0")" || exit 1
source "$(dirname "$0")/env1.sh"
PY=../.venv312/bin/python
STREAM="$1"; shift
mkdir -p out_ms logs_ms
for SHAPE in "$@"; do
  echo "=== $STREAM $SHAPE $(date +%H:%M:%S) ==="
  $PY -m adjoint2d.ms_solve "$SHAPE" out_ms > "logs_ms/${SHAPE}.log" 2>&1 \
    || echo "FAILED $SHAPE"
  tail -3 "logs_ms/${SHAPE}.log"
done
echo "=== $STREAM DONE $(date +%H:%M:%S) ==="
