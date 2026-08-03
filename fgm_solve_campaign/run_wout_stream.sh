#!/bin/bash
# Solve-at-price: one process per (shape, out-of-bounds price), single threaded,
# parallelism taken across jobs exactly as env1.sh explains.
# Usage: run_wout_stream.sh <budget> <w_out> [shapes...]
cd "$(dirname "$0")" || exit 1
source "$(dirname "$0")/env1.sh"
PY=../.venv312/bin/python
BUDGET="$1"; WOUT="$2"; shift 2
SHAPES=("$@")
if [ ${#SHAPES[@]} -eq 0 ]; then
  SHAPES=(square hexagon triangle)
fi
mkdir -p out_wout logs_wout
for SHAPE in "${SHAPES[@]}"; do
  TAG="${SHAPE}_w${WOUT//./p}"
  # RESUME GUARD: a completed solve writes its own result file, so a rerun after
  # an interrupted pass costs nothing for the shapes that already finished.
  if [ -f "out_wout/${SHAPE}_w${WOUT//./p}.json" ]; then
    echo "$TAG already complete, skipped"
    continue
  fi
  s=$(date +%s)
  ( $PY run_wout_solve.py "$SHAPE" "$WOUT" "$BUDGET" 0.85 \
      > "logs_wout/solve_${TAG}.log" 2>&1 \
      || echo "FAILED $TAG" >> logs_wout/stream_timing.log
    echo "$TAG wall=$(( $(date +%s) - s ))s" >> logs_wout/stream_timing.log ) &
done
wait
echo "price done: budget=$BUDGET w_out=$WOUT shapes=${SHAPES[*]}"
