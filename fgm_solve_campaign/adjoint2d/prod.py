"""Import shim for the production 2-D engine.

The prototype lives in a git worktree; the production engine
`rfam_eqs_coupled.py` lives in the main tree. Importing it here (once) keeps
every other module free of sys.path surgery and guarantees the prototype
compares against the SAME source file the campaigns used.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ENV = os.environ.get("RFAM_MAIN_TREE", "").strip()
if _ENV:
    MAIN_TREE = Path(_ENV)
else:
    MAIN_TREE = Path(
        "/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/"
        "research/binderjet/code/geo-prewarp"
    )

if not (MAIN_TREE / "rfam_eqs_coupled.py").exists():  # pragma: no cover
    raise RuntimeError(f"production engine not found under {MAIN_TREE}")

if str(MAIN_TREE) not in sys.path:
    sys.path.insert(0, str(MAIN_TREE))

import rfam_eqs_coupled as rfam  # noqa: E402

__all__ = ["rfam", "MAIN_TREE"]
