"""Correction source selection and build (spec section 6).

Engine registry, never blended:
- "solve3d_solved": exact part-hash match in the solved registry; the DG0
  artifact transfers support-aware onto the voxel grid.
- "heatr_25d_perslice": the deployable fallback, the existing 2.5-D
  verification's dopant_volume.npz transferred support-aware.
- "heatr3d_native_inversion": the last-rung legacy heuristic. KEPT, but now
  refuses to run on a degenerate proxy (see below).
Every provenance record names its engine, trust badge, and transfer state.

===========================================================================
DEGENERATE-PROXY REFUSAL ON THE INVERSION RUNG (TAMPER_DIAGNOSIS.md fix 2)
===========================================================================

WHAT WENT WRONG. On the Tamper job the rung ran `make_fgm` on the BEFORE arm's
`rho_final`. That arm marches until `stop_mean_rho` = 0.98, so its final density
is flat BY CONSTRUCTION. Percentile-normalising a saturated field mapped the
dense bulk to norm ~ 1 and emitted sat ~ 0: 79 % of the part was zeroed, and
every outcome got worse than uniform.

(a) PROXY SWITCH. When the before arm reached its density stop, `rho_final` is
    never used; the proxy becomes `T_phi90`. This is a structural rule with no
    threshold in it, and it is the PRIMARY fix: on all four captured parts the
    T_phi90 proxy is non-degenerate (frac-at-max 0.0000-0.0040), while the
    rho_final proxy is saturated on all three harm jobs.

(b) DEGENERACY FLOORS. These floors are
    pre-registered 2026-08-04, never retuned against outcomes.
    Measured in-part, uncorrected arm, from the real job archive
    (studio3d/tests/fixtures/tamper/manifest.json):

      job              p98-p2 spread  frac@max  ATOM RATIO  emitted sat zeros
      feb850ec Tamper      0.2551       0.3432      42.2         79.1 %
      c474d787 AIRCOIL     0.2436       0.3289      48.0         77.0 %
      d4d50045             0.3511       0.6566     187.3         87.7 %
      2ffdfc3c tube       (0.3794)     (0.0040)     (1.00)        2.4 %  <- healthy

    TWO MEASURED NEGATIVE RESULTS, recorded because each one changed the design.

    (i) A p98-p2 SPREAD floor placed between the Tamper (0.2551) and the
        healthy tube (0.3794) -- the calibration originally proposed -- does
        NOT catch d4d50045, whose spread is 0.3511 but which emitted the WORST
        map in the archive (87.7 % of the part zeroed). Spread is the wrong
        statistic for saturation.

    (ii) A bare frac-at-max floor is GRID DEPENDENT and false-positives on
        coarse grids. A smooth linear ramp on a 20 mm cube gives frac_at_max
        0.1667 at n=16, 0.1000 at n=32 and 0.0455 at n=64 -- the same field,
        three different answers -- because a smooth field ties 1/(number of
        levels) of its voxels at the top. A 0.05 floor would have refused every
        healthy coarse-grid part.

    So the saturation statistic is the ATOM RATIO: the occupancy of the maximum
    value divided by the typical occupancy of the other values near the top. It
    is 1.00 for ANY smooth field at ANY grid (measured: 1.00 for the ramp at
    n=16/32/64, 1.00 for the healthy tube, 1.00 for T_phi90 on all four parts)
    and 42-187 on the three harm cases. Floors:

      SATURATION  atom_ratio >= 5.0 AND frac_at_max >= 0.05 -> refuse
        ("saturated"). 5.0 is 8.4x below the smallest harm case (42.2) and 5x
        above every healthy and synthetic field measured (1.00).
      SPREAD      relative span < 0.02 of the proxy's own scale, and for
        T_phi90 an absolute floor of one melt window (dt_pc_c = 10 C)
        -> refuse ("flat"). This is set from PHYSICS, not from a two-point fit,
        because the two fixtures do not separate on it at all: the Tamper's
        T_phi90 spread is 111.88 C and the healthy tube's is 113.02 C. Anyone
        re-deriving this floor from those two numbers will find no gap; that is
        expected, and is why the saturation floor carries the weight.

KNOWN CEILING OF THIS RUNG, even when the proxy is NOT degenerate. Whole-part
proportional inversion reads the part INTERIOR, so for shapes whose interior
carries no usable signal it produces a null or misleading map -- the
cylinder-null mechanism. It is therefore a QUICK-LOOK generator, never an
optimiser: it has no objective, no budget, no gradient and no hold-out. What
protects the user is the predicted-benefit gate in studio3d/correction_gate.py,
which measures the corrected arm against uniform after the fact and reverts to
uniform when it loses. Do not treat a passing degeneracy check as evidence that
the map is good.

When the rung refuses and no other source exists, the build emits an explicit
UNIFORM correction with `no_correction_applied` set, so the UI and the manifest
can say so instead of shipping a heuristic map nobody checked.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from studio3d import registry as reg
from studio3d.runner import voxelize_stl
from studio3d.transfer import dg0_to_voxel, stack_to_voxel

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

BADGE_25D = ("HEATR 2.5-D per-slice | deployable grading path | sim-only")
BADGE_INVERSION = ("heatr3d native inversion | legacy heuristic (the direct "
                   "solve is the primary generator) | sim-only")


# PRE-REGISTERED 2026-08-04, never retuned against outcomes. See the module
# docstring for the measurements each number came from.
PROXY_FLOORS = {
    "atom_ratio": 5.0,          # saturation: an ATOM piled at the proxy max
    "frac_at_max": 0.05,        # ...and it must be a non-trivial share
    "rel_spread": 0.02,         # flatness: (p98-p2) / |scale| below this
    "abs_spread_T_phi90_c": 10.0,   # one melt window (Params.dt_pc_c)
    "pre_registered": "2026-08-04, never retuned against outcomes",
}


def _atom_ratio(v: np.ndarray) -> float:
    """Occupancy of the MAX value / typical occupancy of the other top values.

    Grid-INDEPENDENT. A bare frac-at-max is not: a smoothly varying field ties
    1/(number of levels) of its voxels at the top, so the SAME linear ramp
    reads frac_at_max 0.167 at n=16 and 0.046 at n=64 and would false-positive
    on coarse grids. This ratio is 1.0 for any smooth field at any grid, and
    large only when a clip or a density stop has piled an ATOM at one value.
    """
    vals, counts = np.unique(np.round(v, 12), return_counts=True)
    if len(vals) < 2:
        return float("inf")
    f = counts / counts.sum()
    others = f[-6:-1] if len(f) >= 6 else f[:-1]
    typ = float(np.median(others)) if len(others) else 0.0
    return float(f[-1] / typ) if typ > 0 else float("inf")


class DegenerateProxyError(RuntimeError):
    """The inversion rung refused: its proxy carries no rankable structure."""


def choose_inversion_proxy(before_results: Dict[str, Any]) -> tuple[str, str]:
    """Pick the inversion proxy for a BEFORE arm. Returns (name, why).

    Rule 2(a): an arm that REACHED its density stop has a flat rho_final by
    construction, so inverting it deletes dopant almost everywhere. Such an arm
    is proxied on T_phi90 instead. No threshold is involved.
    """
    stop = before_results.get("stop_mean_rho")
    rho_mean = before_results.get("rho_final_mean")
    try:
        stop_f = float(stop) if stop is not None else None
        rho_f = float(rho_mean) if rho_mean is not None else None
    except (TypeError, ValueError):
        stop_f = rho_f = None
    if stop_f is not None and rho_f is not None and rho_f >= stop_f - 1e-9:
        return "T_phi90", (
            f"the before arm REACHED its density stop (rho_final_mean="
            f"{rho_f:g} >= stop_mean_rho={stop_f:g}), so rho_final is flat by "
            f"construction and must not be inverted")
    return "rho_final", (
        "the before arm did not reach its density stop, so rho_final still "
        "carries structure")


def proxy_degeneracy(stats_or_array, part: np.ndarray | None = None,
                     proxy_name: str = "rho_final") -> Dict[str, Any]:
    """Judge whether a proxy field carries rankable structure.

    Accepts either a stats dict (as captured in the fixture manifest) or a raw
    array plus a part mask. See the module docstring for the pre-registered
    floors and the measurements behind them.
    """
    if isinstance(stats_or_array, dict):
        s = stats_or_array
    else:
        a = np.asarray(stats_or_array, dtype=float)
        m = np.ones(a.shape, bool) if part is None else np.asarray(part, bool)
        v = a[m]
        lo, hi = np.percentile(v, [2.0, 98.0])
        s = {"p2": float(lo), "p98": float(hi), "spread": float(hi - lo),
             "min": float(v.min()), "max": float(v.max()),
             "mean": float(v.mean()), "std": float(v.std()),
             "frac_at_max": float(np.mean(np.isclose(v, v.max(), rtol=0,
                                                     atol=1e-9))),
             "atom_ratio": _atom_ratio(v)}
    spread = float(s.get("spread", 0.0))
    frac_at_max = float(s.get("frac_at_max", 0.0))
    atom_ratio = float(s.get("atom_ratio", 1.0))
    scale = max(abs(float(s.get("mean", 0.0))), abs(float(s.get("max", 0.0))),
                1e-30)
    rel_spread = spread / scale

    floor_atom = PROXY_FLOORS["atom_ratio"]
    floor_frac = PROXY_FLOORS["frac_at_max"]
    floor_rel = PROXY_FLOORS["rel_spread"]
    floor_abs = (PROXY_FLOORS["abs_spread_T_phi90_c"]
                 if proxy_name == "T_phi90" else None)

    reason = None
    if atom_ratio >= floor_atom and frac_at_max >= floor_frac:
        reason = "saturated"
    elif rel_spread < floor_rel or (floor_abs is not None and spread < floor_abs):
        reason = "flat"

    return {
        "proxy": proxy_name,
        "spread": spread, "rel_spread": rel_spread,
        "frac_at_max": frac_at_max, "atom_ratio": atom_ratio,
        "floor_spread": floor_rel, "floor_abs_spread": floor_abs,
        "floor_frac_at_max": floor_frac, "floor_atom_ratio": floor_atom,
        "degenerate": reason is not None,
        "reason": reason,
        "pre_registered": PROXY_FLOORS["pre_registered"],
    }


def _is_null_map(sat: np.ndarray, part: np.ndarray) -> bool:
    return bool(part.any()) and float(np.abs(sat[part] - 1.0).max()) < 1e-6


def _native_inversion(grade_dir: Path, part: np.ndarray):
    """Density-targeted make_fgm from the BEFORE arm's own fields.

    heatr3d's proportional-inverse rule on rho_final: under-densified
    regions get more dopant. Legacy heuristic, labeled as such; it exists
    so every part gets a real modulated correction even when the 2.5-D
    rulebook refuses the shape and no solved map matches."""
    from types import SimpleNamespace
    import heatr3d as H

    before = grade_dir / "heatr3d" / "uncorrected" / "fields.npz"
    if not before.exists():
        raise FileNotFoundError(
            "no correction source: no solved map matches, the 2.5-D map is "
            "absent or null, and the BEFORE arm has not run yet (its fields "
            "feed the native inversion). Run the BEFORE densification "
            "first.")
    with np.load(before) as d:
        part_b = d["part"].astype(bool)
        rho = np.asarray(d["rho_final"], float)
        T = np.asarray(d["T_phi90"], float)
    if part_b.shape != part.shape or not np.array_equal(part_b, part):
        raise ValueError("BEFORE arm part mask does not match this "
                         "voxelization; rerun the BEFORE arm at this grid")

    # ---- 2(a) proxy choice: never invert a density-stopped rho_final ---- #
    before_results: Dict[str, Any] = {}
    rp = before.parent / "results.json"
    if rp.exists():
        try:
            before_results = json.loads(rp.read_text())
        except (OSError, ValueError) as exc:      # unreadable -> be careful
            logger.warning("could not read the BEFORE arm results.json (%s); "
                           "defaulting the inversion proxy to T_phi90", exc)
            before_results = {"stop_mean_rho": 1.0, "rho_final_mean": 1.0}
    else:
        # No results.json means we cannot prove the arm did NOT stop on
        # density. Choose the safe proxy rather than assume.
        before_results = {"stop_mean_rho": 1.0, "rho_final_mean": 1.0}
    proxy_name, proxy_reason = choose_inversion_proxy(before_results)
    proxy = rho if proxy_name == "rho_final" else T

    # ---- 2(b) degeneracy refusal on the CHOSEN proxy -------------------- #
    deg = proxy_degeneracy(proxy, part=part_b, proxy_name=proxy_name)
    if deg["degenerate"]:
        raise DegenerateProxyError(
            f"native inversion REFUSED: the {proxy_name} proxy is "
            f"{deg['reason']} (frac_at_max={deg['frac_at_max']:.4f} vs floor "
            f"{deg['floor_frac_at_max']}, p98-p2 spread={deg['spread']:.4g}). "
            f"Inverting it would emit a near-binary dopant-removal map. "
            f"Proxy choice: {proxy_reason}.")

    res = SimpleNamespace(T_phi90=T, part=part_b)
    sat = H.make_fgm(res, magnitude=1.0, bpp=4, proxy=proxy)
    # in-part map; outside-part value never reaches the solver (gamma
    # blends part * sat), transfer not applicable: same grid, no resample
    rec = {"sat": np.where(part, sat, 0.0),
           "state": "transfer_not_applicable",
           "method": "native_inversion_same_grid",
           "dopant_mass_move_rel": 0.0, "gate": None}
    prov = {"engine": "heatr3d_native_inversion",
            "trust_badge": BADGE_INVERSION,
            "artifact": str(before),
            "proxy": proxy_name,
            "proxy_reason": proxy_reason,
            "proxy_degeneracy": deg,
            "rung_ceiling_note": (
                "whole-part proportional inversion reads the part INTERIOR "
                "(the cylinder-null mechanism), so this is a QUICK-LOOK "
                "generator with no objective, budget, gradient or hold-out. A "
                "passing degeneracy check is NOT evidence the map is good; the "
                "predicted-benefit gate is what protects the user.")}
    return rec, prov


def _fallback_chain(grade_dir: Path, part: np.ndarray):
    """2.5-D per-slice if it modulates AND transfers within the gate; else
    the native inversion, with any 2.5-D gate failure RECORDED
    (measured_and_failed, never a silent or fatal drop)."""
    from studio3d.transfer import TransferError

    failed_25d = None
    dop = grade_dir / "heatr" / "dopant_volume.npz"
    if dop.exists():
        try:
            with np.load(dop) as d:
                rec = stack_to_voxel(d["sat"], d["part_mask"].astype(bool),
                                     d["z_mm"], part,
                                     chamber_m=float(d["chamber_m"])
                                     if "chamber_m" in d.files else 0.060)
            if not _is_null_map(rec["sat"], part):
                return rec, {"engine": "heatr_25d_perslice",
                             "trust_badge": BADGE_25D, "artifact": str(dop)}
        except TransferError as e:
            # e.g. holed parts: the 2.5-D map is built on FILLED slices,
            # so its support is the wrong geometry for the true part
            failed_25d = {"state": "measured_and_failed", "error": str(e),
                          "artifact": str(dop)}
            logger.warning("2.5-D transfer failed the gate, falling "
                           "through to the native inversion: %s", e)
    rec, prov = _native_inversion(grade_dir, part)
    if failed_25d is not None:
        prov["fallback_from_25d"] = failed_25d
    return rec, prov


def build_correction(grade_dir: str | Path, mesh_path: str, n: int,
                     registry_path: Path = reg.REGISTRY_PATH
                     ) -> Dict[str, Any]:
    """Build the active correction volume for this mesh at grid n.

    Writes grade_dir/heatr3d/correction_sat.npz (sat + part) and
    correction_provenance.json; returns the provenance record.
    """
    grade_dir = Path(grade_dir)
    part = voxelize_stl(mesh_path, n)

    # spec 7e: a FRESH direct-solve artifact for this job ranks first.
    # CHANGED 2026-08-05 (solve3d tranche-1 notify): "do not present a
    # red-gate solve as a correction". solved_label is the artifact's own
    # acceptance verdict (hold-out + smoothing, Phase C protocol); an
    # artifact without it is recorded and SKIPPED, never shipped with a
    # weaker badge. The Tamper is the standing counterexample: it meshes
    # perfectly and its uniform forward is unusable (clamp_bound True,
    # energy residual 3.07e-03).
    solve_skip: Dict[str, Any] | None = None
    fresh = grade_dir / "heatr3d" / "solve" / "studio_solve_map.npz"
    if fresh.exists():
        sr = json.loads(
            (fresh.parent / "studio_solve_results.json").read_text())
        if bool(sr.get("solved_label")):
            with np.load(fresh) as d:
                rec = dg0_to_voxel(d["centroids"], d["s_map"], d["volumes"],
                                   part, chamber_m=0.060)
            prov: Dict[str, Any] = {
                "engine": "solve3d_solved",
                "trust_badge": ("solve3d direct solve | solved_label true "
                                "(hold-out + smoothing gates) | sim-only"),
                "artifact": str(fresh),
                "solve_results": {k: sr.get(k) for k in
                                  ("solved_label", "improvement_pct", "gates",
                                   "warm_start")},
            }
            return _finish(grade_dir, n, part, rec, prov)
        solve_skip = {
            "artifact": str(fresh),
            "reason": ("solved_label is not true: the solve's own acceptance "
                       "gates did not pass, and a red-gate solve is never "
                       "presented as a correction (solve3d lane, 2026-08-05)"),
            "solve_results": {k: sr.get(k) for k in
                              ("solved_label", "improvement_pct", "gates")},
        }

    entry = reg.find_solved_map(part, registry_path=registry_path)
    if entry is not None:
        art = Path(entry["artifact"])
        if not art.is_absolute():
            art = ROOT / art
        with np.load(art) as d:
            rec = dg0_to_voxel(d["centroids"], d["s_map"], d["volumes"],
                               part, chamber_m=0.060)
        prov: Dict[str, Any] = {
            "engine": entry["engine"],
            "trust_badge": entry["trust_badge"],
            "artifact": str(entry["artifact"]),
            "registry_entry": entry["name"],
            "source": entry.get("source"),
        }
    else:
        try:
            rec, prov = _fallback_chain(grade_dir, part)
        except DegenerateProxyError as exc:
            # Nothing deliverable: ship UNIFORM and say so loudly, rather than
            # substituting a heuristic map nobody checked (TAMPER_DIAGNOSIS
            # fix 3 -- make the refusal terminal and honest).
            logger.warning("no deliverable correction: %s", exc)
            rec, prov = _uniform_correction(part, str(exc))

    if solve_skip is not None:
        prov["solve_artifact_skipped"] = solve_skip
    return _finish(grade_dir, n, part, rec, prov)


def _uniform_correction(part: np.ndarray, why: str):
    """An EXPLICIT uniform (sat == 1.0) correction plus the banner flag.

    Uniform is what the printer does with no grading at all, so this ships the
    honest baseline instead of a map that was never checked."""
    rec = {"sat": np.where(part, 1.0, 0.0),
           "state": "transfer_not_applicable",
           "method": "uniform_no_correction",
           "dopant_mass_move_rel": 0.0, "gate": None}
    prov = {
        "engine": "uniform_no_correction",
        "trust_badge": ("uniform (no correction applied) | every grading "
                        "source refused or lost to uniform | sim-only"),
        "artifact": None,
        "no_correction_applied": True,
        "no_correction_reason": why,
        "banner": ("NO CORRECTION APPLIED - this part ships with uniform "
                   "dopant. " + why),
    }
    return rec, prov


def _finish(grade_dir: Path, n: int, part: np.ndarray, rec: Dict[str, Any],
            prov: Dict[str, Any]) -> Dict[str, Any]:
    prov["transfer"] = {k: v for k, v in rec.items() if k != "sat"}
    prov["grid_n"] = int(n)
    # A map that never deviates from 1.0 in-part modulates nothing; the
    # AFTER arm will equal the BEFORE arm by construction. Say so loudly
    # (no false impression of a correction having been applied).
    max_dev = float(np.abs(rec["sat"][part] - 1.0).max()) if part.any() else 0.0
    prov.setdefault("no_correction_applied", False)
    prov["null_correction"] = bool(max_dev < 1e-6)
    if prov["null_correction"]:
        prov["null_note"] = (
            "correction is NULL for this part: the source map is 1.0 "
            "(unmodulated) everywhere in the part, so the corrected arm "
            "equals the uncorrected arm by construction")

    out = grade_dir / "heatr3d"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "correction_sat.npz",
                        sat=rec["sat"].astype(np.float32), part=part)
    # meteor-convention stack for the TIFF grading path: sat[k, iy, ix],
    # z_mm relative to the part bottom, 1.0 (unmodulated) outside the mask
    h = 0.060 / n
    zs = np.where(part.any(axis=(0, 1)))[0]
    stack_sat = np.transpose(rec["sat"][:, :, zs], (2, 1, 0))
    stack_mask = np.transpose(part[:, :, zs], (2, 1, 0))
    stack_sat = np.where(stack_mask, stack_sat, 1.0)
    np.savez_compressed(
        out / "correction_stack.npz",
        sat=stack_sat.astype(np.float32), part_mask=stack_mask,
        z_mm=(np.arange(len(zs)) + 0.5) * h * 1e3,
        chamber_m=0.060)
    (out / "correction_provenance.json").write_text(
        json.dumps(prov, indent=2, default=float))
    logger.info("correction built: engine=%s move=%.3e", prov["engine"],
                prov["transfer"]["dopant_mass_move_rel"])
    return prov
