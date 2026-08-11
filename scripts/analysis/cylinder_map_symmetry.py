"""Symmetry and field-frame decomposition of the Phase C cylinder dopant map.

CAMPAIGN SCRIPT. No solve3d solver module is modified; this only reads the
stored map npz and re-uses solve3d.design_chain.DesignChain for the filter.

THE GROUP, DERIVED FROM THE CASE CODE (not assumed)
---------------------------------------------------
The admissible symmetry group is the INTERSECTION of the part symmetry, the
electrostatic field symmetry, the objective symmetry and the CONVECTION
boundary symmetry. Read directly from the source:

  * part           solve3d/forward.py in_part_predicate("circle") is
                   sqrt(x^2 + y^2) <= 10 mm with NO z condition, and the mesher
                   builds occ.addCylinder(0,0,-L/2, 0,0,L, half)
                   (heatr3d_d1_spike/mesh_gmsh.py). The cylinder axis is z and
                   it spans the full chamber height, so the part is invariant
                   under x -> -x, y -> -y and z -> -z.
  * chamber        box_mesh / addBox on [-L/2, L/2]^3: invariant under all three.
  * electrodes     _electrode_dofs puts Dirichlet planes at y = -L/2 and
                   y = +L/2 (forward.py l.274-278). The deposited power goes as
                   |E|^2, which is even under y -> -y even though V is odd, and
                   the electrode planes are unmoved by x -> -x and z -> -z. The
                   electrodes DO fix the field direction along y, so there is
                   NO rotational invariance about the cylinder axis.
  * convection     _top_facet_measure restricts ds to the OPEN TOP FACE
                   y = +L/2 ONLY (forward.py l.111, l.563-577). This is
                   invariant under x -> -x and z -> -z, and it BREAKS y -> -y.

Intersection: G = {identity, x-mirror, z-mirror, xz-mirror}.

The y-mirror is EXCLUDED. Any top/bottom (in y) asymmetry in a solved map is
physical, produced by the one-sided convection, and projecting it away would
delete real design content and corrupt J. This is the same correction the 3-D
lane recorded in solve3d/results/symmetry_retro_3d.json, where the stage_b4
square scores 0.860 under the corrected group and only 0.699 under the full
three-mirror group.

Usage (pure numpy / scipy, no dolfinx needed):

    ./.venv312/bin/python scripts/analysis/cylinder_map_symmetry.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "solve3d" / "results"
SRC_MAP = RESULTS / "phase_c_map_solve_filter_only_asymmetric_scaled.npz"
OUT_MAP = RESULTS / "phase_c_map_symmetrized_filter_only_asymmetric_scaled.npz"
FILTER_RADIUS_M = 1.0e-3          # phase_c_preregistration.design_chain
SLAB_HALF_M = 0.5e-3              # the figure's mid-height slab, |z| <= 0.5 mm

# (sx, sy, sz) per group element
GROUPS = {
    "xz_corrected": {"identity": (+1, +1, +1), "x_mirror": (-1, +1, +1),
                     "z_mirror": (+1, +1, -1), "xz_mirror": (-1, +1, -1)},
    "x_only": {"identity": (+1, +1, +1), "x_mirror": (-1, +1, +1)},
    "full_xyz": {f"{a}{b}{c}": (a, b, c)
                 for a in (+1, -1) for b in (+1, -1) for c in (+1, -1)},
}
GROUP = "xz_corrected"


# --------------------------------------------------------------------------- #
def group_permutations(centroids: np.ndarray, group: dict):
    """Nearest-centroid index map for each element of the group."""
    tree = cKDTree(centroids)
    perms, dists = {}, {}
    for name, sgn in group.items():
        q = centroids * np.asarray(sgn, float)[None, :]
        d, idx = tree.query(q)
        perms[name] = idx.astype(np.int64)
        dists[name] = d
    return perms, dists


def project_group(s: np.ndarray, perms: dict) -> np.ndarray:
    return np.mean([s[p] for p in perms.values()], axis=0)


def variance_fraction(s: np.ndarray, s_proj: np.ndarray, w: np.ndarray):
    """Fraction of the volume-weighted spatial variance of s retained by
    s_proj, both measured about s's own volume-weighted mean."""
    mu = np.average(s, weights=w)
    var = np.average((s - mu) ** 2, weights=w)
    var_p = np.average((s_proj - mu) ** 2, weights=w)
    return float(var_p / var), float(var), float(var_p)


# --------------------------------------------------------------------------- #
def field_frame_basis(c: np.ndarray, n_r: int = 4, z_powers=(0, 2, 4)):
    """Smooth field-frame modes that are INVARIANT UNDER THE CORRECTED GROUP.

    theta is measured from +x, so the electrode axis y is theta = 90 deg.
    x -> -x acts as theta -> pi - theta, under which cos(m theta) is invariant
    for m EVEN and sin(m theta) is invariant for m ODD. z -> -z admits only
    EVEN powers of z. The invariant smooth set is therefore

        {cos(0 t), cos(2 t), cos(4 t)}   the electrode-field (|E|^2) modes,
        {sin(1 t), sin(3 t)}             the one-sided-convection modes, which
                                         are odd in y = r sin(theta),

    each multiplied by a low-order polynomial in r/R and an even power of
    z/(L/2). The basis is deliberately low order: its whole purpose is to
    separate slow field-frame structure from mesh-frame speckle, so a basis
    rich enough to represent speckle would defeat the measurement.
    """
    r = np.hypot(c[:, 0], c[:, 1])
    th = np.arctan2(c[:, 1], c[:, 0])
    rn = r / max(r.max(), 1e-30)
    zn = c[:, 2] / max(np.abs(c[:, 2]).max(), 1e-30)
    ang = ([(f"cos{m}", np.cos(m * th)) for m in (0, 2, 4)]
           + [(f"sin{m}", np.sin(m * th)) for m in (1, 3)])
    cols, names = [], []
    for aname, a in ang:
        for i in range(n_r):
            for j in z_powers:
                cols.append(a * rn ** i * zn ** j)
                names.append(f"{aname}_r{i}_z{j}")
    return np.column_stack(cols), names


def fit_fraction(s: np.ndarray, B: np.ndarray, w: np.ndarray):
    """Volume-weighted least squares; returns (R2 about the mean, fit, coef)."""
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(B * sw[:, None], s * sw, rcond=None)
    fit = B @ coef
    mu = np.average(s, weights=w)
    ss_tot = np.average((s - mu) ** 2, weights=w)
    ss_res = np.average((s - fit) ** 2, weights=w)
    return float(1.0 - ss_res / ss_tot), fit, coef


# --------------------------------------------------------------------------- #
def report(s, c, v, label: str, z_powers=(0, 2, 4)) -> dict:
    out = {"label": label, "n_cells": int(s.size),
           "mean": float(np.average(s, weights=v)),
           "min": float(s.min()), "max": float(s.max()),
           "std_vol_weighted": float(np.sqrt(variance_fraction(s, s, v)[1])),
           "symmetric_variance_fraction_by_group": {},
           "group_match_max_dist_m": {}}
    for gname, g in GROUPS.items():
        perms, dists = group_permutations(c, g)
        frac, _, _ = variance_fraction(s, project_group(s, perms), v)
        out["symmetric_variance_fraction_by_group"][gname] = frac
        out["group_match_max_dist_m"][gname] = float(
            max(d.max() for d in dists.values()))

    B, _ = field_frame_basis(c, z_powers=z_powers)
    r2, fit, _ = fit_fraction(s, B, v)
    # split the smooth fit into its electrode-field (cos, even m) part, its
    # one-sided-convection (sin, odd m) part, and the pure radial/axial m = 0
    B_cos, names = field_frame_basis(c, z_powers=z_powers)
    keep_cos = np.array([n.startswith("cos") for n in names])
    keep_m0 = np.array([n.startswith("cos0") for n in names])
    r2_cos, fit_cos, _ = fit_fraction(s, B_cos[:, keep_cos], v)
    r2_m0, fit_m0, _ = fit_fraction(s, B_cos[:, keep_m0], v)
    mu = np.average(s, weights=v)
    out.update({
        "field_frame_R2_all_invariant_modes": r2,
        "field_frame_R2_electrode_cos_modes_only": r2_cos,
        "field_frame_R2_radial_axial_m0_only": r2_m0,
        "rms_radial_axial_m0_component":
            float(np.sqrt(np.average((fit_m0 - mu) ** 2, weights=v))),
        "rms_azimuthal_modulation_m2_m4":
            float(np.sqrt(np.average((fit_cos - fit_m0) ** 2, weights=v))),
        "rms_convection_odd_in_y_component":
            float(np.sqrt(np.average((fit - fit_cos) ** 2, weights=v))),
        "n_basis_modes": int(B.shape[1]),
    })
    return out


def main() -> int:
    z = np.load(SRC_MAP)
    s = np.asarray(z["s_map"], float)
    c = np.asarray(z["centroids"], float)
    v = np.asarray(z["volumes"], float)

    out = {"source_map": str(SRC_MAP.relative_to(ROOT)),
           "group_used": GROUP, "group_elements": list(GROUPS[GROUP]),
           "group_justification":
               "part INTERSECT chamber INTERSECT electrode-field INTERSECT "
               "objective INTERSECT convection-boundary symmetry; the y-mirror "
               "is excluded because convection acts on the top face y = +L/2 "
               "only (forward.py l.111, l.563-577), and no rotation about the "
               "cylinder axis survives because the electrodes at y = +/- L/2 "
               "fix the field direction",
           "filter_radius_m": FILTER_RADIUS_M, "slab_half_m": SLAB_HALF_M,
           "diagnostics": {}}

    out["diagnostics"]["raw_full_3d"] = report(s, c, v, "raw solved map, all part cells")
    slab = np.abs(c[:, 2]) <= SLAB_HALF_M
    out["diagnostics"]["raw_midheight_slab"] = report(
        s[slab], c[slab], v[slab], "raw solved map, |z| <= 0.5 mm slab",
        z_powers=(0,))

    # ---- delivered map: project onto the corrected group, then filter once --- #
    import sys
    sys.path.insert(0, str(ROOT))
    from solve3d.design_chain import DesignChain
    perms, _ = group_permutations(c, GROUPS[GROUP])
    s_proj = project_group(s, perms)
    chain = DesignChain(c, v, FILTER_RADIUS_M, [0.0])
    s_sym = np.clip(chain.filter_apply(s_proj), 0.0, 1.0)

    out["delivered"] = report(s_sym, c, v,
                              "group-projected then 1.0 mm filtered")
    frac_vs_raw, _, _ = variance_fraction(s, s_sym, v)
    out["delivered"]["variance_fraction_of_raw_map_retained"] = frac_vs_raw
    out["delivered"]["dopant_rel_move_vs_raw"] = float(
        abs(np.dot(s_sym, v) - np.dot(s, v)) / np.dot(s, v))
    out["delivered_midheight_slab"] = report(
        s_sym[slab], c[slab], v[slab], "delivered map, |z| <= 0.5 mm slab",
        z_powers=(0,))

    np.savez_compressed(OUT_MAP, v_raw=s_proj, s_map=s_sym,
                        centroids=c, volumes=v, s_map_source=s)
    (ROOT / "scripts" / "analysis" / "cylinder_map_symmetry.json").write_text(
        json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
