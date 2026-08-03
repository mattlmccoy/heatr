"""THE READ STATE of a solve-mode run: which time index the deliverable is read at.

WHAT CHANGED IN v2.1.0, stated plainly. Through v2.0.x the stop was the argmin
over the arm's own stored trajectory of the melt-region shape-fidelity objective

    J_phi(s, t) = sum over the domain of (phi(x, t) - chi(x))^2

From v2.1.0 the DEFAULT stop is the argmin of the asymmetric
dense-if-and-only-if-in-bounds objective

    J_asym(s, t) = [ w_out * sum over the BED of phi(x, t)^2
                   + w_in  * sum over the PART of h(rho_rel(x, t))^2 ] / n_part

at the production out-of-bounds price w_out = 2.0 and density floor 0.85
relative density. `stop_rule = "j_phi"` restores the v2.0.x read state exactly.

WHY, and what is NOT changing with it. DENSE_IFF_INBOUNDS_REPORT.md Section 5.1
measured that the asymmetric objective "changes the STOP far more than it
changes the MAP": reading the SAME melt-solved 4-bits-per-pixel map at its own
J_asym argmin instead of at its melt argmin improved J_asym by 4.3 to 36.1
percent, while re-solving the map on top of that changed J_asym by only +17.5,
+6.5, +5.3, +0.1 and -15.6 percent (it lost on one shape of five). So the melt
objective stays the MAP driver and the asymmetric objective owns the STOP only.
That split is the adopted verdict, not an efficiency shortcut.

WHERE THE TWO PRODUCTION NUMBERS COME FROM.
  * `W_OUT_PRODUCTION = 2.0`. Report Section 6 measured the trade curve on the
    delivered maps and found melt IoU (intersection over union) is NOT monotone
    in the price and peaks near w_out = 2 to 3: on the square w_out 1 -> 2 buys
    growth 6.50 -> 2.12 percent and IoU 0.9366 -> 0.9731 for mean in-bounds
    relative density 0.8405 -> 0.8135, which is inside the stated 0.80 to 0.90
    acceptance band. 2.0 is the lower, more conservative end of the recommended
    range, so it gives up the least density.
  * `FLOOR_RHO_REL_PRODUCTION = 0.85`. Report Section 7: the floor sweep on the
    hexagon at 0.80 / 0.85 / 0.90 moved the outcome by only 0.79 growth points
    and 0.009 IoU points, and 0.85 is the only one of the three whose achieved
    mean in-bounds density lands INSIDE the band rather than on its edge.

CAVEAT that travels with these defaults, from the report's own Section 11
ASSUMED list: the trade curve was measured by RE-READING maps that were SOLVED
at w_out = 1, so w_out = 2.0 is calibrated on the read-state trade, and a map
solved at w_out = 2 was not run. That is exactly why this module changes the
stop and not the map.

BOTH stops and BOTH objective values are computed and recorded on every run
whichever rule is selected, so any v2.0.x number stays directly comparable.
"""
from __future__ import annotations

import numpy as np

from . import asym_objective as ao
from . import topopt_objective as tobj

STOP_RULES = ("j_asym", "j_phi")
DEFAULT_STOP_RULE = "j_asym"

W_OUT_PRODUCTION = 2.0          # out-of-bounds price; report Section 6
W_IN_PRODUCTION = 1.0           # in-bounds price; unchanged from the report
FLOOR_RHO_REL_PRODUCTION = 0.85  # relative density floor; report Section 7


# ---------------------------------------------------------------------------
# pure logic
# ---------------------------------------------------------------------------

def validate_stop_rule(rule: str) -> str:
    """Return the rule name, or raise with the allowed set named."""
    r = str(rule)
    if r not in STOP_RULES:
        raise ValueError(
            f"unknown fgm_solve stop_rule {rule!r}; allowed: {list(STOP_RULES)} "
            f"('j_asym' is the v2.1.0 default, 'j_phi' restores the v2.0.x "
            f"melt-region read state)")
    return r


def select_stop(rule: str, phi_stop: dict, asym_stop: dict) -> dict:
    """Choose the read state, carrying BOTH stops and BOTH objective values.

    Args:
        rule: "j_asym" (default) or "j_phi" (the v2.0.x read state).
        phi_stop: record of the melt-region argmin. Keys: index, time_s,
            at_horizon, J_phi, J_asym (the asymmetric objective read AT the melt
            stop, the cross value).
        asym_stop: record of the asymmetric argmin, same keys plus the guard
            flags J_asym_out, J_asym_in, stop_is_first_step, in_term_dead,
            flat_onset_gap_steps.

    Returns:
        The selected stop plus both records and the signed gap between them.

    Raises:
        ValueError: on an unknown rule.
    """
    r = validate_stop_rule(rule)
    chosen = asym_stop if r == "j_asym" else phi_stop
    return {
        "stop_rule": r,
        "stop_rule_default": DEFAULT_STOP_RULE,
        "index": int(chosen["index"]),
        "time_s": float(chosen["time_s"]),
        "at_horizon": bool(chosen["at_horizon"]),
        "J_phi_at_stop": float(chosen["J_phi"]),
        "J_asym_at_stop": float(chosen["J_asym"]),
        "j_phi_stop": dict(phi_stop),
        "j_asym_stop": dict(asym_stop),
        "stop_gap_steps": int(asym_stop["index"]) - int(phi_stop["index"]),
        "note": ("v2.1.0 default read state: the argmin of the "
                 "dense-if-and-only-if-in-bounds objective. The MAP is still "
                 "driven by the melt-region objective; only the STOP changed. "
                 "stop_rule 'j_phi' restores the v2.0.x read state."),
    }


# ---------------------------------------------------------------------------
# against a real trajectory
# ---------------------------------------------------------------------------

def dual_stop(tr, case, chi: np.ndarray, rule: str = DEFAULT_STOP_RULE,
              w_out: float = W_OUT_PRODUCTION,
              w_in: float = W_IN_PRODUCTION,
              floor: float = FLOOR_RHO_REL_PRODUCTION) -> dict:
    """Both argmins over one stored trajectory, plus both cross-read values.

    One pass of each objective over the stored outer steps. No forward run is
    repeated: the caller passes the trajectory it already has.
    """
    r = validate_stop_rule(rule)
    sp = tobj.optimal_stop(tr, case, chi)
    st = ao.asym_stop(tr, case, floor=floor, w_out=w_out, w_in=w_in)

    phi_rec = {
        "index": int(sp.index),
        "time_s": float(sp.time_s),
        "at_horizon": bool(sp.at_horizon),
        "J_phi": float(sp.J),
        "J_first": float(sp.J_first),
        "J_asym": float(ao.J_and_seeds(tr.T_at_end(sp.index),
                                       tr.rho_at_end(sp.index), case,
                                       floor=floor, w_out=w_out, w_in=w_in)[0]),
    }
    asym_rec = {
        "index": int(st.index),
        "time_s": float(st.time_s),
        "at_horizon": bool(st.at_horizon),
        "J_phi": float(tobj.J_and_seed(tr.T_at_end(st.index), case, chi)[0]),
        "J_asym": float(st.J),
        "J_asym_out": float(st.J_out),
        "J_asym_in": float(st.J_in),
        "stop_is_first_step": bool(st.stop_is_first_step),
        "in_term_dead": bool(st.in_term_dead),
        "flat_onset_gap_steps": int(st.flat_onset_gap_steps),
    }
    out = select_stop(r, phi_rec, asym_rec)
    out.update({"w_out": float(w_out), "w_in": float(w_in),
                "floor_rho_rel": float(floor)})
    return out
