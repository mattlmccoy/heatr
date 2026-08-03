#!/bin/zsh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$(dirname "$0")"
for sh in "$@"; do ../.venv312/bin/python -m adjoint2d.make_rho_figures shape "$sh" out_rho figs_rho >> logs_rho/figs.log 2>&1; done
