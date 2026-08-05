"""Predicted-benefit gate: a correction must BEAT uniform, or it does not ship.

WHY THIS EXISTS. On the Tamper job (feb850ec, TAMPER_DIAGNOSIS.md) the Studio
shipped a correction that zeroed the dopant in 79 % of the part and made every
outcome worse than uniform: peak temperature 288.0 -> 448.4 C, out-of-part melt
0.032 -> 0.175, heating time 789 -> 1348 s, sigma_T 28.5 -> 53.2. Nothing in
the chain compared the corrected arm against the before arm, so nothing stopped
it. Every one of the three complete before/after pairs in the job archive turned
out to be a harm case of the same kind (feb850ec, c474d787, d4d50045). This
module closes that class: after the corrected arm completes, it is measured
against the BEFORE arm, and a correction that loses reverts to uniform with a
banner instead of being printed.

GUARD CLASSES
-------------
HARD guards (any one failing -> REJECTED):
  * out_of_part_melt -- melt fraction OUTSIDE the part mask. This is the hard
    failure side of the dense-iff-in-bounds hierarchy: bed spill fuses loose
    powder to the part and is not recoverable in post. It must never regress
    silently. UNKNOWN (not measured) is treated as a FAILURE, never as "fine".
  * T_ceiling -- peak temperature. Compared by VALUE, not by the boolean
    T_ceiling_ok flag: on the Tamper the before arm was ALREADY over the
    ceiling (288.0 C, flag False), so a naive "was ok, now not ok" test would
    have passed the catastrophe through.
  * horizon_cap -- the after arm consuming the whole time budget when the
    before arm reached its stop condition inside it.
  * clamp_bound / energy_residual -- the standing solver gates. A corrected arm
    that trips a limiter the before arm did not is not a measurement.
  * convention_mismatch -- arms compared at different grid, horizon or stop
    target are not a benefit measurement at all; refuse rather than emit a
    meaningless verdict.

DIAGNOSTIC tripwires (recorded, and rejecting only past a gross factor):
  sigma_T, rho_final_std, warp_std_pct. These are quality proxies, not safety
  properties, so a few percent of drift is noise and must not revert a
  correction. Past DIAGNOSTIC_REJECT_FACTOR they stop being drift: the AIRCOIL
  job tripled sigma_T (36.9 -> 111.5).

READ CONVENTION (recorded in the verdict for auditability). Each arm is read at
ITS OWN stop state -- the state the process would actually end in -- and the
two runs must share grid_n, max_time_s and stop_mean_rho. That is the same
convention TAMPER_DIAGNOSIS.md section 2(3) verified for the shipped job, so
this gate's verdicts are comparable with that analysis.

PRE-REGISTERED THRESHOLDS, 2026-08-04, never retuned against outcomes. Values
and the measurements behind them are in THRESHOLDS below.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["evaluate_correction", "apply_verdict", "out_of_part_melt_frac",
           "GUARD_KEYS", "THRESHOLDS", "REJECTED", "ACCEPTED"]

REJECTED = "REJECTED"
ACCEPTED = "ACCEPTED"

# ---------------------------------------------------------------------------
# PRE-REGISTERED 2026-08-04. Never retuned against outcomes.
#
# Measured from the real job archive (studio3d/tests/fixtures/tamper):
#   out-of-part melt fraction, before -> after
#     feb850ec  0.03209 -> 0.17537   (5.5x worse)  REJECT
#     c474d787  0.15561 -> 0.63595   (4.1x worse)  REJECT
#     d4d50045  0.02236 -> 0.16649   (7.4x worse)  REJECT
#     2ffdfc3c  0.00374 -> 0.00023   (improved)    accept side
#   The smallest harm ratio is 4.1x, so a 1.25x tolerance sits far below every
#   observed harm case while absorbing discretisation noise on a near-zero
#   baseline. ABS_FLOOR keeps a tiny absolute baseline from making the ratio
#   meaningless (0.001 -> 0.002 is a doubling of nothing).
#   sigma_T, before -> after: 28.5->53.2, 36.9->111.5, 52.2->135.4
#     i.e. 1.86x, 3.02x, 2.59x. A 1.5x diagnostic reject factor sits below all
#     three and well above ordinary run-to-run drift.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "out_of_part_melt_ratio": 1.25,
    "out_of_part_melt_abs_floor": 0.005,
    "T_max_regress_c": 10.0,
    "diagnostic_reject_factor": 1.5,
    "diagnostic_note_factor": 1.0,
    "pre_registered": "2026-08-04, never retuned against outcomes",
}

GUARD_KEYS = ("out_of_part_melt", "T_ceiling", "horizon_cap", "clamp_bound",
              "energy_residual", "convention_mismatch")
DIAGNOSTIC_KEYS = ("sigma_T", "rho_final_std", "warp_std_pct")


def out_of_part_melt_frac(phi_final: np.ndarray,
                          part: np.ndarray) -> Optional[float]:
    """Mean melt fraction OUTSIDE the part mask, or None if unmeasurable.

    Returns None (never 0.0) when the shapes disagree or there is no
    out-of-part region: an unmeasured guard must be distinguishable from a
    measured-clean one.
    """
    phi = np.asarray(phi_final, dtype=float)
    part = np.asarray(part, dtype=bool)
    if phi.shape != part.shape:
        return None
    outside = ~part
    if not outside.any():
        return None
    return float(phi[outside].mean())


def _get(d: Dict[str, Any], *path, default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _num(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def evaluate_correction(before: Dict[str, Any],
                        after: Dict[str, Any]) -> Dict[str, Any]:
    """Compare a corrected arm against the uniform BEFORE arm.

    Both arguments are studio3d.runner results dicts. Returns a verdict record
    that is written verbatim into the correction provenance.
    """
    guards: Dict[str, Any] = {}
    failed: list[str] = []

    # ---- read convention, recorded explicitly for auditability ---------- #
    matched = {
        "grid_n": before.get("grid_n") == after.get("grid_n"),
        "max_time_s": before.get("max_time_s") == after.get("max_time_s"),
        "stop_mean_rho": before.get("stop_mean_rho") == after.get("stop_mean_rho"),
    }
    read_convention = {
        "description": ("each arm read at its own stop state (the state the "
                        "process would actually end in), under matched "
                        "conventions: same grid_n, same max_time_s, same "
                        "stop_mean_rho"),
        "before_stop": {k: before.get(k) for k in
                        ("sim_time_s", "t_phi90_s", "max_time_s",
                         "stop_mean_rho", "grid_n")},
        "after_stop": {k: after.get(k) for k in
                       ("sim_time_s", "t_phi90_s", "max_time_s",
                        "stop_mean_rho", "grid_n")},
        "matched": matched,
    }
    conv_ok = all(matched.values())
    guards["convention_mismatch"] = {
        "status": "PASS" if conv_ok else "FAIL",
        "before": {k: before.get(k) for k in matched},
        "after": {k: after.get(k) for k in matched},
        "note": ("arms are comparable" if conv_ok else
                 "arms were run under DIFFERENT conventions; a benefit "
                 "comparison is not meaningful and is refused"),
    }
    if not conv_ok:
        failed.append("convention_mismatch")

    # ---- HARD guard: out-of-part melt (bed spill) ----------------------- #
    ob = _num(before.get("out_of_part_melt_frac"))
    oa = _num(after.get("out_of_part_melt_frac"))
    if ob is None or oa is None:
        guards["out_of_part_melt"] = {
            "status": "UNKNOWN", "before": ob, "after": oa,
            "note": ("out-of-part melt fraction was not recorded for one or "
                     "both arms; an UNMEASURED bed-spill guard is treated as "
                     "a FAILURE, never as clean"),
        }
        failed.append("out_of_part_melt")
    else:
        budget = max(ob * THRESHOLDS["out_of_part_melt_ratio"],
                     THRESHOLDS["out_of_part_melt_abs_floor"])
        ok = oa <= budget
        guards["out_of_part_melt"] = {
            "status": "PASS" if ok else "FAIL", "before": ob, "after": oa,
            "budget": budget,
            "ratio": (oa / ob if ob > 0 else None),
            "note": ("bed spill did not regress" if ok else
                     "MELT OUTSIDE THE PART REGRESSED: the corrected map fuses "
                     "more loose powder to the part than uniform does"),
        }
        if not ok:
            failed.append("out_of_part_melt")

    # ---- HARD guard: peak temperature, compared by VALUE ---------------- #
    # Read preference (Tamper stale-snapshot incident, 2026-08-05): heatr3d's
    # T_max_C is a MELT-ONSET snapshot (t90), and for densify runs the march
    # keeps heating long past t90 - the shipped Tamper acceptance compared
    # 239 vs 240 C snapshots while the true end-state peaks were ~70 C
    # higher. When both arms carry the end-state peak (gates.T_end_max_C,
    # recorded by studio3d.runner), the guard compares THAT; otherwise it
    # falls back to the melt-onset read and says so out loud.
    teb = _num(_get(before, "gates", "T_end_max_C"))
    tea = _num(_get(after, "gates", "T_end_max_C"))
    if teb is not None and tea is not None:
        tb, ta, read = teb, tea, "end_state_peak"
    else:
        tb = _num(_get(before, "gates", "T_max_C"))
        ta = _num(_get(after, "gates", "T_max_C"))
        read = "melt_onset_snapshot"
    if tb is None or ta is None:
        guards["T_ceiling"] = {"status": "UNKNOWN", "before": tb, "after": ta,
                               "read": read,
                               "note": "T_max_C missing; treated as a failure"}
        failed.append("T_ceiling")
    else:
        ok = ta <= tb + THRESHOLDS["T_max_regress_c"]
        note_ok = ("peak temperature did not regress"
                   if read == "end_state_peak" else
                   "peak temperature did not regress (melt-onset snapshot "
                   "read: T_end_max_C missing from one or both arms, so the "
                   "true end-state peak was NOT compared)")
        note_fail = ("PEAK TEMPERATURE REGRESSED (compared by value, because "
                     "the before arm may already be over the ceiling)"
                     if read == "end_state_peak" else
                     "PEAK TEMPERATURE REGRESSED on the melt-onset snapshot "
                     "read (T_end_max_C missing; the true end-state peak may "
                     "be worse still)")
        guards["T_ceiling"] = {
            "status": "PASS" if ok else "FAIL", "before": tb, "after": ta,
            "read": read,
            "ceiling_C": _get(before, "gates", "T_ceiling_C"),
            "before_already_over_ceiling":
                (_get(before, "gates", "T_ceiling_ok") is False),
            "note": note_ok if ok else note_fail,
        }
        if not ok:
            failed.append("T_ceiling")

    # ---- HARD guard: horizon cap --------------------------------------- #
    sb, sa = _num(before.get("sim_time_s")), _num(after.get("sim_time_s"))
    mb, ma = _num(before.get("max_time_s")), _num(after.get("max_time_s"))
    if None in (sb, sa, ma):
        guards["horizon_cap"] = {"status": "UNKNOWN", "before": sb, "after": sa,
                                 "note": "sim/max time missing"}
        failed.append("horizon_cap")
    else:
        before_capped = (mb is not None) and (sb >= mb - 1e-9)
        after_capped = sa >= ma - 1e-9
        ok = not (after_capped and not before_capped)
        guards["horizon_cap"] = {
            "status": "PASS" if ok else "FAIL", "before": sb, "after": sa,
            "max_time_s": ma, "before_hit_cap": before_capped,
            "after_hit_cap": after_capped,
            "note": ("the corrected arm did not newly exhaust the horizon"
                     if ok else
                     "the corrected arm ran out the whole time budget while "
                     "the uniform arm reached its stop condition inside it"),
        }
        if not ok:
            failed.append("horizon_cap")

    # ---- HARD guards: standing solver gates ---------------------------- #
    for key, gname in (("clamp_bound", "clamp_bound"),
                       ("energy_residual_ok", "energy_residual")):
        vb = _get(before, "gates", key)
        va = _get(after, "gates", key)
        if key == "clamp_bound":
            ok = not (bool(va) and not bool(vb))
            note = ("the corrected arm newly bound a numerical limiter; its "
                    "numbers are not physical")
        else:
            ok = not ((vb is True) and (va is False))
            note = "the corrected arm broke the standing energy-conservation gate"
        guards[gname] = {"status": "PASS" if ok else "FAIL",
                         "before": vb, "after": va,
                         "note": ("no new limiter" if ok else note)}
        if not ok:
            failed.append(gname)

    # ---- DIAGNOSTIC tripwires ------------------------------------------ #
    tripwires_worse: list[str] = []
    diagnostics: Dict[str, Any] = {}
    for key in DIAGNOSTIC_KEYS:
        vb, va = _num(before.get(key)), _num(after.get(key))
        rec: Dict[str, Any] = {"before": vb, "after": va}
        if vb is not None and va is not None and vb > 0:
            ratio = va / vb
            rec["ratio"] = ratio
            if ratio > THRESHOLDS["diagnostic_note_factor"]:
                tripwires_worse.append(key)
            if ratio > THRESHOLDS["diagnostic_reject_factor"]:
                rec["status"] = "FAIL"
                failed.append(key)
            else:
                rec["status"] = "PASS"
        else:
            rec["status"] = "UNKNOWN"
        diagnostics[key] = rec

    verdict = REJECTED if failed else ACCEPTED
    out = {
        "verdict": verdict,
        "failed_guards": failed,
        "guards": guards,
        "diagnostics": diagnostics,
        "tripwires_worse": tripwires_worse,
        "read_convention": read_convention,
        "thresholds": dict(THRESHOLDS),
        "policy": ("a correction must beat the uniform BEFORE arm on the hard "
                   "guards; on REJECTED the ACTIVE correction for packaging "
                   "reverts to UNIFORM and no_correction_applied is set"),
    }
    if verdict == REJECTED:
        logger.warning("correction gate REJECTED the map: failed guards %s",
                       failed)
    return out


def apply_verdict(grade_dir, verdict: Dict[str, Any]) -> Dict[str, Any]:
    """Make the verdict change what actually SHIPS.

    On REJECTED the ACTIVE packaging artifacts -- correction_sat.npz and
    correction_stack.npz, the two files the emitter and the manifest read --
    are REPLACED with explicit uniform maps, and correction_provenance.json
    gains no_correction_applied plus a banner the UI and the manifest surface.
    The verdict itself is recorded either way, so an accepted correction is
    auditable too.

    The rejected map is not deleted: it is kept alongside as
    correction_sat.rejected.npz so the failure can be inspected. It is simply
    no longer the map anything downstream reads.
    """
    import json as _json
    from pathlib import Path as _Path

    import numpy as _np

    out = _Path(grade_dir) / "heatr3d"
    prov_path = out / "correction_provenance.json"
    prov: Dict[str, Any] = {}
    if prov_path.exists():
        try:
            prov = _json.loads(prov_path.read_text())
        except (OSError, ValueError):
            prov = {}
    prov["correction_gate"] = verdict

    if verdict.get("verdict") != REJECTED:
        prov.setdefault("no_correction_applied", False)
        prov_path.write_text(_json.dumps(prov, indent=2, default=float))
        return prov

    sat_path = out / "correction_sat.npz"
    stack_path = out / "correction_stack.npz"
    if sat_path.exists():
        sat_path.replace(out / "correction_sat.rejected.npz")
        with _np.load(out / "correction_sat.rejected.npz") as d:
            part = d["part"].astype(bool)
        _np.savez_compressed(sat_path,
                             sat=_np.where(part, 1.0, 0.0).astype(_np.float32),
                             part=part)
    if stack_path.exists():
        stack_path.replace(out / "correction_stack.rejected.npz")
        with _np.load(out / "correction_stack.rejected.npz") as d:
            keep = {k: d[k] for k in d.files}
        keep["sat"] = _np.ones_like(_np.asarray(keep["sat"], _np.float32))
        _np.savez_compressed(stack_path, **keep)

    reason = ("the corrected arm LOST to uniform on "
              + ", ".join(verdict.get("failed_guards", [])))
    prov["engine_rejected"] = prov.get("engine")
    prov["engine"] = "uniform_no_correction"
    prov["no_correction_applied"] = True
    prov["no_correction_reason"] = reason
    prov["banner"] = ("NO CORRECTION APPLIED - this part ships with uniform "
                      "dopant. " + reason)
    prov["trust_badge"] = ("uniform (no correction applied) | predicted-benefit "
                           "gate REJECTED the graded map | sim-only")
    prov["rejected_artifacts"] = {
        "sat": str(out / "correction_sat.rejected.npz"),
        "stack": str(out / "correction_stack.rejected.npz"),
        "note": ("kept for inspection; nothing downstream reads these"),
    }
    prov_path.write_text(_json.dumps(prov, indent=2, default=float))
    logger.warning("correction REVERTED TO UNIFORM: %s", reason)
    return prov
