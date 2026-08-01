"""FGM benefit re-run under the corrected qrf_gradient="masked" default.

    ./.venv312/bin/python heatr3d_eqs02_rerank/run_fgm_rerun.py --shape cone

The exposure: the PUBLISHED heatr3d FGM maps were designed by inverting a
LEGACY-Q temperature field (make_fgm(res) uses res.T_phi90), and were also
EVALUATED under legacy Q. The EQS-02 correction therefore hits the map design
AND the evaluation. Arms per shape (all n=64, phase_update="enthalpy"):

  A  legacy baseline                          (reproduces rerank_results.json)
  B  masked baseline                          (reproduces rerank_results.json)
  C  map designed from A's T_phi90, run under MASKED physics
        -> "as-published map, corrected world"
  Cl map designed from A's T_phi90, run under LEGACY physics
        -> in-campaign reproduction of the OLD-regime published benefit
  D  map designed from B's T_phi90, run under MASKED physics
        -> the honest go-forward benefit

Decision-relevant deltas: C vs B and D vs B.

Cost control: only THREE EQS solves per shape. The baseline solve serves A and B
(same V, two Q post-processings); the sat_C solve serves C and Cl the same way;
sat_D needs its own solve. Every march is driven through run(qrf_override=...),
which skips the internal EQS solve -- the same trick run_rerank.py used, which is
why A and B here are bit-comparable to the stored rerank arms.

heatr3d.py is imported READ-ONLY. Nothing under dissertation_materials/ is
written; the published cone/dumbbell dopant volumes are only READ, for a map
similarity check.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # heatr3d.py at the repo root
sys.path.insert(0, str(HERE))                 # run_rerank.py / fgm_arms.py

import heatr3d  # noqa: E402
from run_rerank import GEOM, SURFACE_BAND_H, build_part  # noqa: E402
from fgm_arms import benefit_pct, gate_pass  # noqa: E402

SHAPES = ("sphere", "cone", "dumbbell", "cylinder")
PHI_TARGET = 0.90
MAX_TIME_S = 1500.0

# PUBLISHED make_fgm settings, identical across the three published claims:
#   analysis-3dfgm/run_cone_dumbbell_fgm_fixed.py  (cone -31.9 %, dumbbell -44.2 %)
#   analysis-3dfgm/run_3d_study.py                 (sphere -54.0 %, study_summary.csv)
FGM_MAGNITUDE = 1.0
FGM_BASELINE = 0.5
FGM_BPP = 2

# Published FGM benefit (% change in sigma_T vs its own baseline), for the
# sign-survival verdict. Cylinder has NO published FGM claim -> control arm.
PUBLISHED_PCT = {"cone": -31.9, "dumbbell": -44.2, "sphere": -54.0, "cylinder": None}
PUBLISHED_SRC = {
    "cone": "analysis-3dfgm/cone_dumbbell_fgm_results_fixed.json (32.648 -> 22.247 C)",
    "dumbbell": "analysis-3dfgm/cone_dumbbell_fgm_results_fixed.json (20.127 -> 11.234 C)",
    "sphere": "analysis-3dfgm/study_summary.csv row sphere,fgm3d (32.92 -> 15.15 C)",
    "cylinder": "no published FGM claim (control)",
}
# READ-ONLY: the actual published dopant maps, for a map-similarity check.
PUBLISHED_VOL_NPZ = Path(
    "/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/"
    "dissertation_materials/analysis-3dfgm/cone_dumbbell_fgm_volumes_fixed.npz")

RERANK_JSON = HERE / "rerank_results.json"


def _rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(ru) / 1024.0 ** 3 if sys.platform == "darwin" else float(ru) / 1024.0 ** 2


def _qrf_stats(q: np.ndarray, part, interior, band) -> dict:
    qin = q[part]
    mu = float(qin.mean())
    return {"mean_w_per_m3": mu,
            "max_over_mean": float(qin.max() / mu),
            "cv": float(qin.std() / mu),
            "power_fraction_in_surface_band": float(q[band].sum() / qin.sum()),
            "interior_mean_w_per_m3": float(q[interior].mean()),
            "surface_mean_w_per_m3": float(q[band].mean())}


def _thermal_stats(res, part, interior, band, wall_s: float) -> dict:
    T = res.T_phi90
    gates = {"reached_phi90": bool(res.reached),
             "energy_residual_frac": float(res.energy_residual_frac),
             "clamp_bound": bool(res.clamp_bound),
             "cfl_violated": bool(res.cfl_violated),
             "n_substeps_used": int(res.n_substeps_used)}
    gates["pass"] = gate_pass(gates)
    return {"sigma_T_c": float(res.sigma_T),
            "t_phi90_s": float(res.t_phi90_s),
            "T_max_c": float(res.T_max_c),
            "T_mean_c": float(T[part].mean()),
            "T_min_c": float(T[part].min()),
            "interior_mean_c": float(T[interior].mean()),
            "surface_mean_c": float(T[band].mean()),
            "surface_minus_interior_mean_c": float(T[band].mean() - T[interior].mean()),
            "gates": gates,
            "wall_thermal_s": wall_s}


def _sat_stats(sat: np.ndarray, part: np.ndarray) -> dict:
    s = sat[part]
    levels = (1 << FGM_BPP) - 1
    hist = {f"{int(round(v * levels))}/{levels}": int((np.round(s * levels) == v * levels).sum())
            for v in np.unique(np.round(s * levels) / levels)}
    return {"mean": float(s.mean()), "min": float(s.min()), "max": float(s.max()),
            "level_counts": hist}


def _march(grid, part, p_th, q, label: str, interior, band) -> dict:
    print(f"    march {label} ...", flush=True)
    t0 = time.perf_counter()
    res = heatr3d.run(grid, part, p_th, qrf_override=q,
                      max_time_s=MAX_TIME_S, phi_target=PHI_TARGET)
    st = _thermal_stats(res, part, interior, band, time.perf_counter() - t0)
    g = st["gates"]
    print(f"      sigma_T={st['sigma_T_c']:.3f} C  t90={st['t_phi90_s']:.1f} s  "
          f"Tmax={st['T_max_c']:.2f} C  gate_pass={g['pass']} "
          f"reached={g['reached_phi90']} resid={g['energy_residual_frac']:.2e} "
          f"clamp={g['clamp_bound']} cfl={g['cfl_violated']}  "
          f"({st['wall_thermal_s']:.0f} s)", flush=True)
    return {"stats": st, "result": res}


def _solve(part, p, grid, sat, tag: str):
    """One EQS solve -> (gamma, V, wall_s). Reproduces run()'s internal path with
    default edge_width_m / premix (so a qrf_override march is equivalent to the
    non-override march it stands in for)."""
    t0 = time.perf_counter()
    gamma = heatr3d.build_gamma(part, p, sat, edge_width_m=0.0, h=grid.h,
                                premix_frac=0.0)
    V = heatr3d.solve_eqs_3d(gamma, grid, p)
    wall = time.perf_counter() - t0
    print(f"    EQS solve [{tag}] {wall:.1f} s", flush=True)
    return gamma, V, wall


def _published_map_similarity(shape: str, sat: np.ndarray, part: np.ndarray) -> dict | None:
    """Compare the arm-C map (legacy-designed here, enthalpy march) against the
    ACTUAL published dopant map. READ-ONLY access to dissertation_materials."""
    if shape not in ("cone", "dumbbell") or not PUBLISHED_VOL_NPZ.exists():
        return None
    with np.load(PUBLISHED_VOL_NPZ) as z:
        pub = np.asarray(z[f"{shape}_sat"], dtype=float)
        pub_part = np.asarray(z[f"{shape}_part"], dtype=bool)
    if pub.shape != sat.shape or not np.array_equal(pub_part, part):
        return {"comparable": False,
                "why": "published part mask / shape differs from this campaign's"}
    a, b = sat[part], pub[part]
    # The published map is stored as float32, so EXACT equality against our
    # float64 make_fgm output fails on the 1/3 and 2/3 levels for numerical
    # reasons alone. Compare on the QUANTIZATION LEVEL index instead (both maps
    # are bpp=2, i.e. levels k/3), which is the physically meaningful identity.
    levels = (1 << FGM_BPP) - 1
    same_level = np.round(a * levels) == np.round(b * levels)
    return {"comparable": True,
            "source": str(PUBLISHED_VOL_NPZ.name),
            "identical_voxel_fraction": float(same_level.mean()),
            "exact_float_equality_fraction": float((a == b).mean()),
            "mean_abs_diff": float(np.abs(a - b).mean()),
            "published_mean_sat": float(b.mean()),
            "this_campaign_mean_sat": float(a.mean()),
            "note": "published map came from an apparent_cp + legacy-Q march; "
                    "arm C's map comes from this campaign's enthalpy + legacy-Q march"}


def run_shape(shape: str, n: int) -> dict:
    grid = heatr3d.Grid(n=n)
    p = heatr3d.Params()
    p_th = dataclasses.replace(p, phase_update="enthalpy")
    part = build_part(shape, grid)
    depth = (distance_transform_edt(part) - 0.5) * grid.h
    interior = part & (depth > SURFACE_BAND_H * grid.h)
    band = part & ~interior
    print(f"[fgm-rerun] {shape} n={n}: {int(part.sum())} part voxels", flush=True)

    rec: dict = {"shape": shape, "n": n, "h_m": grid.h, "L_m": grid.L,
                 "geometry": dict(GEOM[shape]),
                 "n_voxels_in_part": int(part.sum()),
                 "phase_update": p_th.phase_update,
                 "phi_target": PHI_TARGET, "max_time_s": MAX_TIME_S,
                 "fgm_settings": {"magnitude": FGM_MAGNITUDE, "baseline": FGM_BASELINE,
                                  "bpp": FGM_BPP, "proxy": "T_phi90 (make_fgm default)"},
                 "published_pct": PUBLISHED_PCT[shape],
                 "published_source": PUBLISHED_SRC[shape],
                 "arms": {}, "qrf": {}, "sat": {}, "eqs_solves": {}}

    # ---- solve 1: uniform-dopant baseline; serves arms A and B ---------------
    gamma0, V0, w0 = _solve(part, p, grid, None, "baseline uniform")
    rec["eqs_solves"]["baseline"] = w0
    q0 = {arm: heatr3d.compute_qrf_3d(V0, gamma0, grid, p, doped=part, premix=False,
                                      qrf_gradient=arm) for arm in ("legacy", "masked")}
    rec["qrf"]["A_legacy_baseline"] = _qrf_stats(q0["legacy"], part, interior, band)
    rec["qrf"]["B_masked_baseline"] = _qrf_stats(q0["masked"], part, interior, band)

    A = _march(grid, part, p_th, q0["legacy"], "A (legacy baseline)", interior, band)
    B = _march(grid, part, p_th, q0["masked"], "B (masked baseline)", interior, band)
    rec["arms"]["A_legacy_baseline"] = A["stats"]
    rec["arms"]["B_masked_baseline"] = B["stats"]

    # ---- map design ---------------------------------------------------------
    sat_C = heatr3d.make_fgm(A["result"], magnitude=FGM_MAGNITUDE,
                             baseline=FGM_BASELINE, bpp=FGM_BPP)
    sat_D = heatr3d.make_fgm(B["result"], magnitude=FGM_MAGNITUDE,
                             baseline=FGM_BASELINE, bpp=FGM_BPP)
    rec["sat"]["C_legacy_designed"] = _sat_stats(sat_C, part)
    rec["sat"]["D_masked_designed"] = _sat_stats(sat_D, part)
    rec["sat"]["map_agreement_C_vs_D"] = {
        "identical_voxel_fraction": float((sat_C[part] == sat_D[part]).mean()),
        "mean_abs_diff": float(np.abs(sat_C[part] - sat_D[part]).mean())}
    rec["sat"]["C_vs_published_map"] = _published_map_similarity(shape, sat_C, part)

    # ---- solve 2: legacy-designed map; serves arms C (masked) and Cl (legacy) -
    gammaC, VC, wC = _solve(part, p, grid, sat_C, "legacy-designed FGM")
    rec["eqs_solves"]["fgm_legacy_designed"] = wC
    qC = {arm: heatr3d.compute_qrf_3d(VC, gammaC, grid, p, doped=part, premix=False,
                                      qrf_gradient=arm) for arm in ("legacy", "masked")}
    rec["qrf"]["C_legacyMap_masked"] = _qrf_stats(qC["masked"], part, interior, band)
    rec["qrf"]["Cl_legacyMap_legacy"] = _qrf_stats(qC["legacy"], part, interior, band)
    Cl = _march(grid, part, p_th, qC["legacy"], "Cl (legacy map, LEGACY physics)",
                interior, band)
    C = _march(grid, part, p_th, qC["masked"], "C (legacy map, masked physics)",
               interior, band)
    rec["arms"]["Cl_legacyMap_legacyPhysics"] = Cl["stats"]
    rec["arms"]["C_legacyMap_maskedPhysics"] = C["stats"]

    # ---- solve 3: masked-designed map; arm D --------------------------------
    gammaD, VD, wD = _solve(part, p, grid, sat_D, "masked-designed FGM")
    rec["eqs_solves"]["fgm_masked_designed"] = wD
    qD = heatr3d.compute_qrf_3d(VD, gammaD, grid, p, doped=part, premix=False,
                                qrf_gradient="masked")
    rec["qrf"]["D_maskedMap_masked"] = _qrf_stats(qD, part, interior, band)
    D = _march(grid, part, p_th, qD, "D (masked map, masked physics)", interior, band)
    rec["arms"]["D_maskedMap_maskedPhysics"] = D["stats"]

    # ---- deltas -------------------------------------------------------------
    sA = rec["arms"]["A_legacy_baseline"]["sigma_T_c"]
    sB = rec["arms"]["B_masked_baseline"]["sigma_T_c"]
    sC = rec["arms"]["C_legacyMap_maskedPhysics"]["sigma_T_c"]
    sCl = rec["arms"]["Cl_legacyMap_legacyPhysics"]["sigma_T_c"]
    sD = rec["arms"]["D_maskedMap_maskedPhysics"]["sigma_T_c"]
    rec["deltas_pct"] = {
        "old_regime_benefit_Cl_vs_A": benefit_pct(sCl, sA),
        "old_map_corrected_world_C_vs_B": benefit_pct(sC, sB),
        "corrected_design_D_vs_B": benefit_pct(sD, sB),
        "baseline_shift_B_vs_A": benefit_pct(sB, sA),
        "D_vs_C": benefit_pct(sD, sC),
    }

    # ---- reproduction check against the stored rerank arms ------------------
    if RERANK_JSON.exists():
        stored = json.loads(RERANK_JSON.read_text())["shapes"].get(shape)
        if stored is not None:
            rec["rerank_reproduction"] = {
                arm: {"stored_sigma_T_c": stored["thermal"][a0]["sigma_T_c"],
                      "here_sigma_T_c": here,
                      "rel_diff": abs(here - stored["thermal"][a0]["sigma_T_c"])
                      / max(abs(stored["thermal"][a0]["sigma_T_c"]), 1e-30)}
                for arm, a0, here in (("A_legacy_baseline", "legacy", sA),
                                      ("B_masked_baseline", "masked", sB))}

    rec["all_gates_pass"] = all(a["gates"]["pass"] for a in rec["arms"].values())
    rec["peak_rss_gb"] = _rss_gb()
    rec["heatr3d_run_default_qrf_gradient"] = "masked"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=SHAPES)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out-dir", default=str(HERE / "fgm_shards"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rec = run_shape(args.shape, args.n)
    rec["wall_total_s"] = time.perf_counter() - t0
    rec["python"] = sys.version.split()[0]
    path = out_dir / f"{args.shape}_n{args.n}.json"
    path.write_text(json.dumps(rec, indent=1))
    print(f"[fgm-rerun] wrote {path} ({rec['wall_total_s']:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
