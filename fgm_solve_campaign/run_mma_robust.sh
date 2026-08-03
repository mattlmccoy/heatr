#!/bin/bash
# The two acceptance gates for one MMA-retest arm, six shapes in parallel.
# Usage: run_mma_robust.sh <stem-suffix, e.g. _mma or _mma_b80 or "">
cd "$(dirname "$0")" || exit 1
source "$(dirname "$0")/env1.sh"
PY=../.venv312/bin/python
SUF="$1"
mkdir -p out_mma logs_mma
for SHAPE in square circle trapezoid triangle diamond rectangle; do
  STEM="${SHAPE}${SUF}"
  $PY -m adjoint2d.topopt_robust "$SHAPE" out_mma "" out_mma "$STEM" \
    > "logs_mma/${STEM}_robust.log" 2>&1 || echo "FAILED robust $STEM" &
done
wait
echo "robust done for suffix '${SUF}'"
