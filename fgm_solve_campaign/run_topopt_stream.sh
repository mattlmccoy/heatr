#!/bin/bash
# One stream of topology-optimization solves.
# Usage: run_topopt_stream.sh <mode: solve|control> <shape> ...
cd "$(dirname "$0")" || exit 1
source "$(dirname "$0")/env1.sh"
PY=../.venv312/bin/python
MODE="$1"; shift
mkdir -p out_topopt logs_topopt
for SHAPE in "$@"; do
  if [ "$MODE" = "control" ]; then
    $PY -m adjoint2d.topopt_solve "$SHAPE" out_topopt 40 filteronly \
      > "logs_topopt/${SHAPE}_control.log" 2>&1 || echo "FAILED control $SHAPE"
  else
    $PY -m adjoint2d.topopt_solve "$SHAPE" out_topopt \
      > "logs_topopt/${SHAPE}.log" 2>&1 || echo "FAILED solve $SHAPE"
  fi
done
