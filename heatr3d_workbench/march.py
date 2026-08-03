"""Parse workbench_job job.log into live progress + scalar march series.

Runs in the SERVER interpreter: stdlib only. The solver's verbose line
(heatr3d.py:1237) is:
    `  t= 123.4s  Tmax= 190.1  phi=0.453  rho=0.550`
The wrapper adds `PROGRESS <pct>` and `PHASE <name>` lines. Garbage lines are
ignored (the log also carries warnings and prints from the solver).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

_MARCH_RE = re.compile(
    r"^\s*t=\s*([0-9.]+)s\s+Tmax=\s*([-0-9.na]+)\s+phi=([0-9.na]+)\s+rho=([0-9.na]+)")

# March-derived progress is scaled into the window between the wrapper's
# pre-march and post-march PROGRESS marks.
_MARCH_LO, _MARCH_HI = 10.0, 92.0


def _f(tok: str) -> Optional[float]:
    try:
        return float(tok)
    except ValueError:
        return None


def parse_log(text: str, max_time_s: Optional[float] = None) -> Dict[str, Any]:
    """Parse a job.log -> {progress, phase, series:{t_s, T_max_c, phi_bar, rho_bar}}."""
    t_s, tmax, phi, rho = [], [], [], []
    prog_mark = 0.0
    phase = "queued"
    for line in text.splitlines():
        if line.startswith("PROGRESS"):
            try:
                prog_mark = float(line.split()[1])
            except (IndexError, ValueError):
                pass
            continue
        if line.startswith("PHASE"):
            parts = line.split(None, 1)
            if len(parts) == 2:
                phase = parts[1].strip()
            continue
        m = _MARCH_RE.match(line)
        if m:
            vals = [_f(g) for g in m.groups()]
            if vals[0] is None:
                continue
            t_s.append(vals[0]); tmax.append(vals[1])
            phi.append(vals[2]); rho.append(vals[3])

    progress = prog_mark
    if t_s and max_time_s and max_time_s > 0:
        frac = min(1.0, t_s[-1] / float(max_time_s))
        march_prog = _MARCH_LO + frac * (_MARCH_HI - _MARCH_LO)
        progress = max(progress, march_prog)
    progress = min(progress, 99.0) if progress < 100.0 else 100.0

    return {
        "progress": round(progress, 1),
        "phase": phase,
        "series": {"t_s": t_s, "T_max_c": tmax, "phi_bar": phi, "rho_bar": rho},
    }
