"""Trust badges + per-run gate flags for the heatr3d workbench.

Server-side module: STDLIB ONLY (the GUI server interpreter has an unreliable
numpy; nothing here may import it). Registry facts are hand-maintained in
badge_registry.json and only updated when a gate report lands.

Rules (spec section 6): every displayed number class carries the highest gate
badge; per-run flags (energy gate, clamp, CFL, melt-onset fallback, 250 C
ceiling) are RUN-level state, distinct from tool-level badges; absence of a
gate value renders as not_recorded, never as healthy.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).resolve().parent / "badge_registry.json"

ENERGY_GATE_ABS = 1e-2      # |energy_residual_frac| standing gate
CEILING_C = 250.0           # thermal ceiling


@lru_cache(maxsize=1)
def load_registry() -> Dict[str, Any]:
    """Load the hand-maintained badge registry."""
    return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))


def badge_for(quantity_class: str) -> Dict[str, Any]:
    """Badge dict for a quantity class; unknown classes are loudly unknown."""
    entry = load_registry()["classes"].get(quantity_class)
    if entry is None:
        return {"level": "unknown",
                "label": f"unknown quantity class '{quantity_class}' (no badge on file)",
                "detail": "", "evidence": ""}
    return dict(entry)


def _flag(fid: str, state: str, text: str) -> Dict[str, str]:
    return {"id": fid, "state": state, "text": text}


def run_flags(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Derive per-run gate flags from a results.json dict.

    States: pass | warn | fail | not_recorded. Missing gate values are
    not_recorded (a legacy run), which is visually distinct from pass.
    """
    flags: List[Dict[str, str]] = []

    e = results.get("energy_residual_frac")
    if e is None:
        flags.append(_flag("energy_gate", "not_recorded",
                           "energy residual not recorded (legacy run; gate unavailable)"))
    elif abs(float(e)) <= ENERGY_GATE_ABS:
        flags.append(_flag("energy_gate", "pass",
                           f"energy residual {float(e):+.2e} within the 1e-2 gate"))
    else:
        flags.append(_flag("energy_gate", "fail",
                           f"energy residual {float(e):+.2e} EXCEEDS the 1e-2 gate"))

    reached = results.get("reached_phi90")
    fallback = bool(results.get("MELT_ONSET_FALLBACK", False)) or reached is False
    if fallback:
        flags.append(_flag("melt_onset_fallback", "warn",
                           "phi_bar never crossed 0.90: sigma_T / T_max / t90 are "
                           "final-step reads, not melt-onset reads"))
    else:
        flags.append(_flag("melt_onset_fallback", "pass", "melt-onset read state reached"))

    if bool(results.get("clamp_bound", False)):
        flags.append(_flag("clamp_bound", "fail",
                           "a numerical limiter bound during the march; numbers suspect"))
    else:
        flags.append(_flag("clamp_bound", "pass", "no limiter bound"))

    if bool(results.get("cfl_violated", False)):
        flags.append(_flag("cfl", "fail",
                           "conduction CFL violated (enforce_cfl off); march untrustworthy"))
    else:
        flags.append(_flag("cfl", "pass", "conduction stability enforced"))

    t = results.get("T_max_C")
    if t is None:
        flags.append(_flag("ceiling_250c", "not_recorded", "T_max not recorded"))
    elif float(t) > CEILING_C:
        flags.append(_flag("ceiling_250c", "warn",
                           f"T_max {float(t):.1f} C exceeds the 250 C ceiling"))
    else:
        flags.append(_flag("ceiling_250c", "pass",
                           f"T_max {float(t):.1f} C within the 250 C ceiling"))

    return flags


def banner_state(flags: List[Dict[str, str]]) -> str:
    """Aggregate flag list -> run banner state: fail > warn > ok."""
    states = {f["state"] for f in flags}
    if "fail" in states:
        return "fail"
    if "warn" in states or "not_recorded" in states:
        return "warn"
    return "ok"
