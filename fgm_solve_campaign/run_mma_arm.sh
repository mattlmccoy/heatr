#!/bin/bash
# One arm of the MMA retest, six shapes in six parallel single-threaded streams.
# Usage: run_mma_arm.sh <budget> <control|-> <optimizer> [shapes...]
cd "$(dirname "$0")" || exit 1
source "$(dirname "$0")/env1.sh"
PY=../.venv312/bin/python
BUDGET="$1"; CONTROL="$2"; OPT="$3"; shift 3
[ "$CONTROL" = "-" ] && CONTROL=""
SHAPES=("$@")
if [ ${#SHAPES[@]} -eq 0 ]; then
  SHAPES=(square circle trapezoid triangle diamond rectangle)
fi
mkdir -p out_mma logs_mma
for SHAPE in "${SHAPES[@]}"; do
  TAG="${SHAPE}_${OPT}_b${BUDGET}${CONTROL:+_$CONTROL}"
  $PY -m adjoint2d.topopt_solve "$SHAPE" out_mma "$BUDGET" "$CONTROL" "$OPT" \
    > "logs_mma/${TAG}.log" 2>&1 || echo "FAILED $TAG" &
done
wait
echo "arm done: budget=$BUDGET control=${CONTROL:-none} optimizer=$OPT"
