#!/bin/zsh
# EQS-02 shape re-ranking campaign launcher.
# 3 workers at a time, single-threaded BLAS: another agent's adjoint2d gates are
# holding ~10/12 cores, so this deliberately does NOT oversubscribe.
set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"
LOG="$ROOT/heatr3d_eqs02_rerank/logs"
mkdir -p "$LOG"
N="${N:-64}"
SHAPES="${SHAPES:-cone dumbbell sphere cylinder square lshape cross diamond}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

i=0
for s in ${=SHAPES}; do
  "$ROOT/.venv312/bin/python" -u "$ROOT/heatr3d_eqs02_rerank/run_rerank.py" \
      --shape "$s" --n "$N" > "$LOG/${s}_n${N}.log" 2>&1 &
  i=$((i+1))
  if [ $((i % 3)) -eq 0 ]; then wait; fi
done
wait
echo "CAMPAIGN_DONE n=$N"
