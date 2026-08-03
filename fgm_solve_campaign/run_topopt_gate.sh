#!/bin/bash
cd "$(dirname "$0")" || exit 1
source "$(dirname "$0")/env1.sh"
PY=../.venv312/bin/python
mkdir -p out_topopt logs_topopt
for SHAPE in "$@"; do
  ( $PY -m adjoint2d.gate_topopt "$SHAPE" "out_topopt/gate_topopt_${SHAPE}.json" \
      > "logs_topopt/gate_${SHAPE}.log" 2>&1 || echo "GATE FAILED $SHAPE" ) &
done
wait
echo "=== GATES DONE $(date +%H:%M:%S) ==="
