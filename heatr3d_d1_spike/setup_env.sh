#!/bin/bash
# D1 spike environment: micromamba + fenics-dolfinx (conda-forge), macOS arm64.
#
# Self-contained: micromamba binary lands in ./mm/bin, the environment in ./env.
# Nothing is installed globally and no shell rc file is touched.
#
# Deviation from the plan text (documented, not silent): the plan's
#   MM="${PWD}/mm/bin/micromamba"; command -v micromamba >/dev/null 2>&1 && MM=micromamba
# aborts under `set -e` when micromamba is absent from PATH (the `&&` list exits 1).
# It also would prefer a *global* micromamba over the local one; the task brief
# requires the spike to stay self-contained, so we always use the local binary.
set -euo pipefail
cd "$(dirname "$0")"

MM="${PWD}/mm/bin/micromamba"
if [ ! -x "$MM" ]; then
  mkdir -p mm && cd mm
  curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xj bin/micromamba
  cd ..
fi
export MAMBA_ROOT_PREFIX="${PWD}/mm/root"

# try the complex-PETSc variant first; fall back to default (real) build
"$MM" create -y -p ./env -c conda-forge python=3.11 fenics-dolfinx "petsc=*=*complex*" gmsh python-gmsh pyvista || \
"$MM" create -y -p ./env -c conda-forge python=3.11 fenics-dolfinx gmsh python-gmsh pyvista

# check_env.py imports jit_fix first: this repo's path contains SPACES, which
# breaks FFCx's C JIT (conda bakes the prefix into sysconfig's whitespace-split
# CFLAGS). See jit_fix.py -- that is a recorded D1 deployability finding.
./env/bin/python check_env.py
