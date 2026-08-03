#!/bin/bash
# Filtered SINGLE cold start at the full budget. Separates the design filter
# from the multi-start, since the two changed together in the primary arm.
cd "$(dirname "$0")" || exit 1
source "$(dirname "$0")/env1.sh"
PY=../.venv312/bin/python
STREAM="$1"; shift
for SHAPE in "$@"; do
  echo "=== $STREAM $SHAPE $(date +%H:%M:%S) ==="
  $PY -m adjoint2d.ms_solve "$SHAPE" out_ms 40 1.5 cold \
    > "logs_ms/${SHAPE}_control_cold.log" 2>&1 || echo "FAILED $SHAPE"
  tail -2 "logs_ms/${SHAPE}_control_cold.log"
done
echo "=== $STREAM DONE $(date +%H:%M:%S) ==="
