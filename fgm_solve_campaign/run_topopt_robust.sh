#!/bin/bash
cd "$(dirname "$0")" || exit 1
source "$(dirname "$0")/env1.sh"
PY=../.venv312/bin/python
for SHAPE in "$@"; do
  $PY -m adjoint2d.topopt_robust "$SHAPE" out_topopt \
    > "logs_topopt/${SHAPE}_robust.log" 2>&1 || echo "FAILED robust $SHAPE"
done
