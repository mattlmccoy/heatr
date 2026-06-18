#!/usr/bin/env python3
"""heatr3d.py -- REDIRECT SHIM to the single canonical source of truth.

CFG-01 / REPRO-04 (audit hygiene): there used to be TWO diverging copies of the
3-D heatr solver -- this one (geo-prewarp) and the one in analysis-3dfgm. They
drifted (the analysis-3dfgm copy is the SUPERSET: prewarp-study shapes, premix,
powder-loss BC, heat-sink Idea-4 terms, eps-perturb hook, edge regularization,
plus the audit clamp instrumentation). Maintaining two copies is a reproducibility
hazard. This file is now a THIN REDIRECT that re-exports the canonical module, so
`import heatr3d` keeps working unchanged for every existing caller (heatr3d_job.py),
while there is exactly ONE implementation.

Production-inertness: the canonical copy was verified to produce BIT-IDENTICAL
output to the former geo-prewarp copy across every job path (densify on/off x
fgm none/melt/density) for the shapes both supported (np.array_equal on sigma_T,
T_phi90, Qrf, rho_final, sinter_metrics, shrinkage_analysis). See
audit_fixes_out/hygiene/RESULT.md.

Drift detection: the canonical file's SHA-256 is recorded below. If it changes,
this shim emits a WARNING (so silent drift is visible) but still loads it, so a
legitimate canonical update does not break geo-prewarp. Update CANONICAL_SHA256
deliberately after reviewing a canonical change.

Override: set HEATR3D_CANONICAL_PATH to point at the canonical heatr3d.py if the
default relative location moves (e.g. a different Dropbox root).
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# SHA-256 of the canonical analysis-3dfgm/heatr3d.py this shim was validated
# against (audit-fixes branch, after the THM-01/02 clamp instrumentation).
CANONICAL_SHA256 = "4c7307f37abbd6903e4283430e489aebb1544bd638617cb113c978f7b4816078"

_DEFAULT_REL = "../../../dissertation_materials/analysis-3dfgm/heatr3d.py"


def _resolve_canonical_path() -> Path:
    env = os.environ.get("HEATR3D_CANONICAL_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"HEATR3D_CANONICAL_PATH={env!r} does not point at a file"
        )
    here = Path(__file__).resolve().parent
    p = (here / _DEFAULT_REL).resolve()
    if p.is_file():
        return p
    raise FileNotFoundError(
        "Canonical heatr3d.py not found at the expected relative location "
        f"({p}). Set HEATR3D_CANONICAL_PATH to the canonical "
        "dissertation_materials/analysis-3dfgm/heatr3d.py."
    )


def _check_drift(path: Path) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != CANONICAL_SHA256:
        logger.warning(
            "CFG-01 canonical heatr3d.py DRIFT: %s now hashes to %s, expected %s. "
            "geo-prewarp is loading a CHANGED canonical solver. Review the change "
            "and update CANONICAL_SHA256 in this shim if it is intentional.",
            path, actual, CANONICAL_SHA256,
        )


def _load_canonical():
    path = _resolve_canonical_path()
    _check_drift(path)
    spec = importlib.util.spec_from_file_location("_heatr3d_canonical", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec so the canonical module's frozen dataclasses can
    # introspect their own module via sys.modules (dataclasses._is_type).
    sys.modules["_heatr3d_canonical"] = mod
    spec.loader.exec_module(mod)
    return mod, path


_canonical, CANONICAL_PATH = _load_canonical()

# Re-export the canonical public API into this module's namespace so existing
# `import heatr3d as H; H.run(...)` callers are unaffected.
for _name in dir(_canonical):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_canonical, _name)

__canonical_path__ = str(CANONICAL_PATH)
__canonical_sha256__ = CANONICAL_SHA256
