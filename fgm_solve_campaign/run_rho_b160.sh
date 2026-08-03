#!/bin/zsh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")"
for sh in "$@"; do
  s=$(date +%s)
  ../.venv312/bin/python -m adjoint2d.rho_solve "$sh" out_rho_b160 160 1.5 >> "logs_rho/${sh}_b160.log" 2>&1
  echo "$sh b160 rc=$? wall=$(( $(date +%s) - s ))s" >> logs_rho/stream_timing.log
done
