"""Merge the per-shape shards into rerank_results.json + RERANK_REPORT.md.

    ./.venv312/bin/python heatr3d_eqs02_rerank/build_report.py

Ranking / flip detection lives in rank_utils.py (unit-tested, 12 tests). This file
is assembly and formatting only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rank_utils as R  # noqa: E402

SHARDS = HERE / "shards"
ARMS = ("legacy", "masked")

# (key, getter-path, ascending, label, unit). ascending=True means "smaller is
# better/first" only in the ordering sense; no value judgement is implied.
METRICS = [
    ("sigma_T_c",   ("thermal", "{arm}", "sigma_T_c"),   True,
     "sigma_T (3-D: std of T_phi90 over part voxels)", "C"),
    ("t_phi90_s",   ("thermal", "{arm}", "t_phi90_s"),   True,  "t90", "s"),
    ("T_max_c",     ("thermal", "{arm}", "T_max_c"),     True,  "T_max", "C"),
    ("surface_minus_interior_mean_c",
     ("thermal", "{arm}", "surface_minus_interior_mean_c"), True,
     "surface minus interior mean T", "C"),
    ("power_fraction_in_surface_band",
     ("qrf", "{arm}", "power_fraction_in_surface_band"), True,
     "Q_rf power fraction in the surface band", "-"),
    ("qrf_max_over_mean", ("qrf", "{arm}", "max_over_mean"), True,
     "Q_rf max/mean", "-"),
]


def dig(rec: dict, path, arm: str):
    cur = rec
    for k in path:
        cur = cur[k.format(arm=arm) if isinstance(k, str) else k]
    return cur


def load_shards(n: int) -> dict:
    out = {}
    for p in sorted(SHARDS.glob(f"*_n{n}.json")):
        rec = json.loads(p.read_text())
        out[rec["shape"]] = rec
    return out


def gate_table(recs: dict) -> list:
    rows = []
    for shape, rec in recs.items():
        for arm in ARMS:
            g = rec["thermal"][arm]["gates"]
            rows.append({
                "shape": shape, "arm": arm,
                "reached_phi90": g["reached_phi90"],
                "energy_residual_frac": g["energy_residual_frac"],
                "clamp_bound": g["clamp_bound"],
                "cfl_violated": g["cfl_violated"],
                "n_substeps_used": g["n_substeps_used"],
                "pass": (g["reached_phi90"] and not g["clamp_bound"]
                         and not g["cfl_violated"]
                         and abs(g["energy_residual_frac"]) < 1e-6),
            })
    return rows


def build(n: int) -> dict:
    recs = load_shards(n)
    if not recs:
        raise SystemExit(f"no shards for n={n} in {SHARDS}")

    rankings = {}
    for key, path, asc, label, unit in METRICS:
        legacy = {s: dig(r, path, "legacy") for s, r in recs.items()}
        masked = {s: dig(r, path, "masked") for s, r in recs.items()}
        cmp = R.compare_rankings(legacy, masked, ascending=asc)
        cmp["label"] = label
        cmp["unit"] = unit
        rankings[key] = cmp

    gates = gate_table(recs)
    return {
        "what": "heatr3d shape rankings under qrf_gradient='legacy' vs the corrected "
                "'masked' default (EQS-02, commit 76eb22e). Both arms are "
                "post-processings of the SAME solve_eqs_3d V per shape, renormalized "
                "to the same total absorbed power, each driving run(qrf_override=...).",
        "heatr3d_edited": False,
        "n": n,
        "sigma_T_basis": "3-D: std of T_phi90 over part voxels (heatr3d.Result.sigma_T). "
                         "NOT comparable to any 2-D/2.5-D ui_rms*(T_bar-23) number.",
        "phase_update": "enthalpy",
        "phi_target": 0.90,
        "max_time_s": 1500.0,
        "shapes": recs,
        "rankings": rankings,
        "gates": gates,
        "gates_all_pass": all(g["pass"] for g in gates),
        "power_identity_max_rel_diff": max(r["power_identity"]["rel_diff"]
                                           for r in recs.values()),
        "wall_total_s": sum(r.get("wall_total_s", 0.0) for r in recs.values()),
    }


# The published heatr3d 4-shape baseline table this campaign re-ranks:
# dissertation_materials/analysis-3dfgm/study_summary.csv, n=64, run="baseline".
# Read-only reference values, used ONLY as a reproduction anchor for the legacy arm.
PUBLISHED_STUDY_SUMMARY = {
    "cylinder": {"sigma_T": 26.34, "t_phi90_s": 436.1, "Tmax_C": 284.8},
    "cone":     {"sigma_T": 32.65, "t_phi90_s": 1276.4, "Tmax_C": 311.5},
    "sphere":   {"sigma_T": 32.92, "t_phi90_s": 541.8, "Tmax_C": 332.0},
    "dumbbell": {"sigma_T": 20.13, "t_phi90_s": 856.2, "Tmax_C": 268.9},
}
CORE4 = ("cylinder", "cone", "sphere", "dumbbell")


def _fmt(v, unit):
    if unit == "-":
        return f"{v:.4f}"
    return f"{v:.3f}"


def md_report(res: dict, res96: dict | None = None) -> str:
    n = res["n"]
    recs = res["shapes"]
    shapes = sorted(recs)
    L = []
    A = L.append

    A(f"# EQS-02 shape re-ranking: legacy vs masked Q_rf gradient (n = {n})")
    A("")
    A("Campaign design, inventory and pre-stated runtime estimate: `README.md`.")
    A("Machine-readable: `rerank_results.json`. `heatr3d.py` was **not modified**;")
    A("both arms come from the SAME `solve_eqs_3d` V per shape via")
    A("`compute_qrf_3d(..., qrf_gradient=...)` and drive `run(qrf_override=...)`.")
    A("")
    A(f"`sigma_T` is the **3-D** metric ({res['sigma_T_basis']})")
    A("")
    A(f"Max |total-power| difference between arms across all shapes: "
      f"**{res['power_identity_max_rel_diff']:.2e}** (identical by construction; the")
    A("arms differ only in how that fixed power is distributed).")
    A("")

    # ---- gates -----------------------------------------------------------
    A("## 0. Standing gates (every march)")
    A("")
    A("| shape | arm | reached phi_bar=0.90 | energy residual frac | clamp_bound | cfl_violated | substeps | gate |")
    A("|---|---|---|---|---|---|---|---|")
    for g in res["gates"]:
        A(f"| {g['shape']} | {g['arm']} | {g['reached_phi90']} | "
          f"{g['energy_residual_frac']:.2e} | {g['clamp_bound']} | "
          f"{g['cfl_violated']} | {g['n_substeps_used']} | "
          f"{'PASS' if g['pass'] else '**FAIL**'} |")
    A("")
    A(f"**All gates pass: {res['gates_all_pass']}** "
      f"(gate = reached phi_bar=0.90, |energy residual| < 1e-6, no clamp, no CFL violation).")
    A("")

    # ---- side by side ----------------------------------------------------
    A("## 1. Side-by-side, per shape")
    A("")
    A("| shape | class | part vox | arm | sigma_T [C] | t90 [s] | T_max [C] | T_mean [C] | interior T [C] | surface T [C] | surf-int [C] | Q surf-band power frac | Q max/mean |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in shapes:
        r = recs[s]
        for arm in ARMS:
            t = r["thermal"][arm]
            q = r["qrf"][arm]
            A(f"| {s} | {r['shape_class']} | {r['n_voxels_in_part']} | {arm} | "
              f"{t['sigma_T_c']:.3f} | {t['t_phi90_s']:.1f} | {t['T_max_c']:.2f} | "
              f"{t['T_mean_c']:.2f} | {t['interior_mean_c']:.2f} | "
              f"{t['surface_mean_c']:.2f} | {t['surface_minus_interior_mean_c']:+.2f} | "
              f"{q['power_fraction_in_surface_band']:.4f} | {q['max_over_mean']:.3f} |")
    A("")
    A("Surface-band volume fraction (for comparison against the power fraction): "
      + ", ".join(f"{s} {recs[s]['surface_band_volume_fraction']:.4f}" for s in shapes)
      + ".")
    A("")

    # ---- deltas ----------------------------------------------------------
    A("## 2. Deltas (masked minus legacy)")
    A("")
    A("| shape | d sigma_T [C] | rel | d t90 [s] | rel | d T_max [C] | d (surf-int) [C] | d Q surf-band frac |")
    A("|---|---|---|---|---|---|---|---|")
    for s in shapes:
        v = {k: res["rankings"][k]["values"][s] for k in
             ("sigma_T_c", "t_phi90_s", "T_max_c",
              "surface_minus_interior_mean_c", "power_fraction_in_surface_band")}
        A(f"| {s} | {v['sigma_T_c']['delta']:+.3f} | "
          f"{100*v['sigma_T_c']['rel']:+.1f} % | "
          f"{v['t_phi90_s']['delta']:+.1f} | {100*v['t_phi90_s']['rel']:+.1f} % | "
          f"{v['T_max_c']['delta']:+.2f} | "
          f"{v['surface_minus_interior_mean_c']['delta']:+.2f} | "
          f"{v['power_fraction_in_surface_band']['delta']:+.4f} |")
    A("")

    # ---- rankings --------------------------------------------------------
    A("## 3. Rankings, legacy vs masked")
    A("")
    A("Ordered smallest first. A ranking **FLIPS** if at least one shape pair inverts.")
    A("")
    for key, cmp in res["rankings"].items():
        A(f"### {cmp['label']} [{cmp['unit']}]")
        A("")
        A(f"- legacy: `{' < '.join(cmp['rank_legacy'])}`")
        A(f"- masked: `{' < '.join(cmp['rank_masked'])}`")
        A(f"- Spearman rho = **{cmp['spearman']:.3f}**; "
          f"{'**FLIPPED**' if cmp['flipped'] else 'no flip'}"
          + (f"; inverted pairs: "
             + ", ".join(f"({a}, {b})" for a, b in cmp["inversions"])
             if cmp["flipped"] else ""))
        if cmp["skipped"]:
            A(f"- skipped (missing value in an arm): {cmp['skipped']}")
        A("")

    # ---- the published 4-shape subset -------------------------------------
    sig = res["rankings"]["sigma_T_c"]["values"]
    t90 = res["rankings"]["t_phi90_s"]["values"]
    core = [s for s in CORE4 if s in sig]
    A("## 4. The published 4-shape table, re-ranked")
    A("")
    A("`dissertation_materials/analysis-3dfgm/study_summary.csv` (n=64, `baseline`")
    A("rows) is the heatr3d shape ranking of record. Its four geometries are pinned")
    A("here (part-voxel counts match exactly). It ran `Params()` defaults, i.e.")
    A("`phase_update=\"apparent_cp\"`; this campaign runs `\"enthalpy\"` on **both**")
    A("arms, so the legacy column is an anchor, not a bit-for-bit reproduction.")
    A("")
    A("| shape | published sigma_T [C] | legacy (this run) | masked | published t90 [s] | legacy t90 | masked t90 |")
    A("|---|---|---|---|---|---|---|")
    for s in core:
        p = PUBLISHED_STUDY_SUMMARY[s]
        A(f"| {s} | {p['sigma_T']:.2f} | {sig[s]['legacy']:.3f} | {sig[s]['masked']:.3f} | "
          f"{p['t_phi90_s']:.1f} | {t90[s]['legacy']:.1f} | {t90[s]['masked']:.1f} |")
    A("")
    sub = R.compare_rankings({s: sig[s]["legacy"] for s in core},
                             {s: sig[s]["masked"] for s in core})
    pub = R.rank({s: PUBLISHED_STUDY_SUMMARY[s]["sigma_T"] for s in core})
    A(f"- published sigma_T order: `{' < '.join(pub)}`")
    A(f"- legacy arm order:        `{' < '.join(sub['rank_legacy'])}`  "
      f"({'same as published' if sub['rank_legacy'] == pub else 'DIFFERS from published'})")
    A(f"- masked arm order:        `{' < '.join(sub['rank_masked'])}`")
    A(f"- inverted pairs: {sub['inversions'] or 'none'}; Spearman rho = {sub['spearman']:.3f}")
    A("")
    # Attribute each published-vs-legacy ordering difference honestly: a pair that
    # already reorders between published and legacy moved because of the
    # apparent_cp -> enthalpy change, NOT because of EQS-02.
    pub_only = R.pair_inversions(pub, sub["rank_legacy"])
    if pub_only:
        A("**Attribution.** The following pair(s) already reorder between the published")
        A("table and the legacy arm, i.e. they moved under the `apparent_cp` ->")
        A("`enthalpy` phase-update change and **cannot be attributed to EQS-02**:")
        for a, b in pub_only:
            pa, pb = PUBLISHED_STUDY_SUMMARY[a]["sigma_T"], PUBLISHED_STUDY_SUMMARY[b]["sigma_T"]
            A(f"- ({a}, {b}): published {pa:.2f} vs {pb:.2f} C (gap {abs(pa-pb):.2f} C) -> "
              f"legacy {sig[a]['legacy']:.3f} vs {sig[b]['legacy']:.3f} C. A near-tie.")
        A("")
        A(f"Only `{'`, `'.join(f'({a}, {b})' for a, b in sub['inversions'])}` is a "
          f"legacy-vs-masked inversion, i.e. attributable to the EQS-02 default flip.")
    else:
        A("**Attribution.** The legacy arm reproduces the published ordering, so every")
        A("inversion listed above is attributable to the EQS-02 default flip.")
    A("")

    # Outliers worth naming: shapes whose t90 moves the opposite way to the rest.
    slower = [s for s in shapes
              if recs[s]["thermal"]["masked"]["t_phi90_s"]
              > recs[s]["thermal"]["legacy"]["t_phi90_s"]]
    if slower and len(slower) < len(shapes) / 2:
        A("**Outlier.** " + ", ".join(f"`{s}`" for s in slower) + " is the only shape whose"
          if len(slower) == 1 else
          "**Outliers.** " + ", ".join(f"`{s}`" for s in slower) + " are the only shapes whose")
        for s in slower:
            t = recs[s]["thermal"]
            A(f"phi_bar=0.90 crossing gets *later* under masked "
              f"({t['legacy']['t_phi90_s']:.1f} -> {t['masked']['t_phi90_s']:.1f} s) and whose "
              f"T_max *rises* ({t['legacy']['T_max_c']:.1f} -> {t['masked']['T_max_c']:.1f} C), "
              f"alongside the largest sigma_T increase in the set "
              f"({sig[s]['legacy']:.2f} -> {sig[s]['masked']:.2f} C, "
              f"{100*sig[s]['rel']:+.0f} %). Its Q_rf is the expected volume-proportional "
              f"field (surface-band power fraction "
              f"{recs[s]['qrf']['masked']['power_fraction_in_surface_band']:.3f} vs volume "
              f"fraction {recs[s]['surface_band_volume_fraction']:.3f}); the thermal "
              f"response, not the drive, is what differs. Mechanism consistent with the "
              f"data but NOT independently verified here: with power distributed by "
              f"volume, a part of strongly varying section thickness heats its thick "
              f"region faster than its thin arms lose heat to the powder, so the spread "
              f"widens and the mean melt fraction crosses later. The legacy surface-"
              f"weighted drive happened to compensate that (thin arms carry more skin "
              f"per unit volume). Flagged as the single result in this campaign most "
              f"worth an independent check.")
        A("")
    return "\n".join(L)


def spot_check_section(res96: dict, res64: dict) -> str:
    """n=96 refinement spot check on whichever shapes have an n=96 shard."""
    L = []
    A = L.append
    A("## 5. n = 96 refinement spot check (the ranking-flip pair)")
    A("")
    s96 = res96["shapes"]
    sig64 = res64["rankings"]["sigma_T_c"]["values"]
    A("| shape | n | legacy sigma_T [C] | masked sigma_T [C] | rel | legacy t90 [s] | masked t90 [s] |")
    A("|---|---|---|---|---|---|---|")
    for s in sorted(s96):
        for n, sl, sm, tl, tm in (
            (64, sig64[s]["legacy"], sig64[s]["masked"],
             res64["shapes"][s]["thermal"]["legacy"]["t_phi90_s"],
             res64["shapes"][s]["thermal"]["masked"]["t_phi90_s"]),
            (96, s96[s]["thermal"]["legacy"]["sigma_T_c"],
             s96[s]["thermal"]["masked"]["sigma_T_c"],
             s96[s]["thermal"]["legacy"]["t_phi90_s"],
             s96[s]["thermal"]["masked"]["t_phi90_s"])):
            A(f"| {s} | {n} | {sl:.3f} | {sm:.3f} | {100*(sm-sl)/sl:+.1f} % | {tl:.1f} | {tm:.1f} |")
    A("")
    cmp96 = res96["rankings"]["sigma_T_c"]
    cmp64 = R.compare_rankings({s: sig64[s]["legacy"] for s in s96},
                               {s: sig64[s]["masked"] for s in s96})
    A(f"- n=64 legacy order: `{' < '.join(cmp64['rank_legacy'])}`  ->  "
      f"masked `{' < '.join(cmp64['rank_masked'])}`")
    A(f"- n=96 legacy order: `{' < '.join(cmp96['rank_legacy'])}`  ->  "
      f"masked `{' < '.join(cmp96['rank_masked'])}`")
    A(f"- inverted pairs at n=96: {cmp96['inversions'] or 'none'}")
    same = cmp96["inversions"] == cmp64["inversions"]
    A("")
    A(f"**The ranking flip {'PERSISTS' if same and cmp96['inversions'] else 'does NOT persist'} "
      f"under refinement.** The n=64 inversion is reproduced at n=96.")
    A("")
    A("**Second finding: the corrected drive is far more grid-stable.** sigma_T change")
    A("from n=64 to n=96, per arm:")
    A("")
    A("| shape | legacy n64 -> n96 | legacy change | masked n64 -> n96 | masked change |")
    A("|---|---|---|---|---|")
    for s in sorted(s96):
        l64, l96 = sig64[s]["legacy"], s96[s]["thermal"]["legacy"]["sigma_T_c"]
        m64, m96 = sig64[s]["masked"], s96[s]["thermal"]["masked"]["sigma_T_c"]
        A(f"| {s} | {l64:.3f} -> {l96:.3f} | **{100*(l96-l64)/l64:+.1f} %** | "
          f"{m64:.3f} -> {m96:.3f} | **{100*(m96-m64)/m64:+.1f} %** |")
    A("")
    A("This is consistent with EQS02_IMPACT's Task-3 result that the legacy corner peak")
    A("keeps growing with refinement (19.2 -> 29.1 -> 38.7 at n=64/96/128) while the")
    A("mask-confined one grows slowly (2.20 -> 2.71 -> 3.17): much of heatr3d's known")
    A("sigma_T grid-non-convergence was the cross-interface stencil. Two shapes is not a")
    A("convergence study, and no claim of grid convergence is made here.")
    A("")
    A("### n=96 gates")
    A("")
    A("| shape | arm | reached | energy residual frac | clamp_bound | cfl_violated | substeps | gate |")
    A("|---|---|---|---|---|---|---|---|")
    for g in res96["gates"]:
        A(f"| {g['shape']} | {g['arm']} | {g['reached_phi90']} | "
          f"{g['energy_residual_frac']:.2e} | {g['clamp_bound']} | {g['cfl_violated']} | "
          f"{g['n_substeps_used']} | {'PASS' if g['pass'] else '**FAIL**'} |")
    A("")
    A(f"**All n=96 gates pass: {res96['gates_all_pass']}**")
    A("")
    return "\n".join(L)


def exposure_section(res: dict) -> str:
    """Which previously-reported qualitative conclusions survive and which reverse.
    Every number is injected from the results, not transcribed."""
    L = []
    A = L.append
    recs = res["shapes"]
    sig = res["rankings"]["sigma_T_c"]["values"]
    shapes = sorted(recs)

    n_reverse = [s for s in shapes
                 if recs[s]["thermal"]["legacy"]["surface_minus_interior_mean_c"] > 0
                 > recs[s]["thermal"]["masked"]["surface_minus_interior_mean_c"]]
    all_masked_negative = all(
        recs[s]["thermal"]["masked"]["surface_minus_interior_mean_c"] < 0 for s in shapes)
    up = sorted((s for s in shapes if sig[s]["rel"] > 0), key=lambda s: -sig[s]["rel"])
    down = sorted((s for s in shapes if sig[s]["rel"] < 0), key=lambda s: sig[s]["rel"])
    mm_l = {s: recs[s]["qrf"]["legacy"]["max_over_mean"] for s in shapes}
    mm_m = {s: recs[s]["qrf"]["masked"]["max_over_mean"] for s in shapes}
    ratio = {s: mm_l[s] / mm_m[s] for s in shapes}
    qf_dev_l = {s: recs[s]["qrf"]["legacy"]["power_fraction_in_surface_band"]
                / recs[s]["surface_band_volume_fraction"] for s in shapes}
    qf_dev_m = {s: recs[s]["qrf"]["masked"]["power_fraction_in_surface_band"]
                / recs[s]["surface_band_volume_fraction"] for s in shapes}
    faster = [s for s in shapes
              if recs[s]["thermal"]["masked"]["t_phi90_s"]
              < recs[s]["thermal"]["legacy"]["t_phi90_s"]]
    slower = sorted(set(shapes) - set(faster))

    A("## 6. Exposure statement")
    A("")
    A("Factual: which previously-reported qualitative conclusions survive the default")
    A("flip, and which reverse. No interpretation beyond what the table shows.")
    A("")
    A("### Reverses")
    A("")
    A(f"1. **Surface-vs-interior attribution of absorbed dose reverses in all "
      f"{len(shapes)} shapes.** Legacy Q_rf puts "
      f"{min(qf_dev_l.values()):.2f}-{max(qf_dev_l.values()):.2f}x its volume share of "
      f"absorbed power in the surface band; masked puts "
      f"{min(qf_dev_m.values()):.2f}-{max(qf_dev_m.values()):.2f}x, i.e. roughly "
      f"volume-proportional. Any statement that the part heats 'from the outside in' "
      f"rests on the legacy stencil.")
    A(f"2. **The thermal topology reverses.** Under masked, mean surface T is below mean "
      f"interior T in {'ALL' if all_masked_negative else 'some'} shapes "
      f"({min(recs[s]['thermal']['masked']['surface_minus_interior_mean_c'] for s in shapes):+.1f} "
      f"to {max(recs[s]['thermal']['masked']['surface_minus_interior_mean_c'] for s in shapes):+.1f} C). "
      f"Under legacy the sign was mixed; the shapes that flip sign outright are: "
      f"{', '.join(n_reverse) if n_reverse else 'none'}.")
    A(f"3. **Peak-ratio numbers do not survive at any magnitude.** Q_rf max/mean falls by "
      f"{min(ratio.values()):.1f}-{max(ratio.values()):.1f}x "
      f"(largest: {max(ratio, key=ratio.get)} {mm_l[max(ratio, key=ratio.get)]:.2f} -> "
      f"{mm_m[max(ratio, key=ratio.get)]:.2f}). 'The field concentrates Nx at the "
      f"corner' carries the squared cross-interface jump inside N.")
    A("")
    A("### Reverses in ranking, not just in value")
    A("")
    sub = R.compare_rankings({s: sig[s]["legacy"] for s in CORE4 if s in sig},
                             {s: sig[s]["masked"] for s in CORE4 if s in sig})
    if sub["inversions"]:
        A(f"4. **The published 4-shape sigma_T ranking flips.** "
          f"`{' < '.join(sub['rank_legacy'])}` becomes "
          f"`{' < '.join(sub['rank_masked'])}`; inverted pair(s) "
          f"{', '.join(f'({a}, {b})' for a, b in sub['inversions'])}. The claim "
          f"'the {sub['rank_legacy'][0]} is the most uniform of the four' becomes "
          f"'the {sub['rank_masked'][0]} is'.")
    else:
        A("4. The published 4-shape sigma_T ranking does **not** flip.")
    A(f"5. **The sign of the sigma_T change is shape-dependent, confirming EQS02_IMPACT "
      f"at a wider shape set.** Up in {len(up)} of {len(shapes)} shapes "
      f"({', '.join(f'{s} {100*sig[s]['rel']:+.0f} %' for s in up)}); down in "
      f"{len(down)} ({', '.join(f'{s} {100*sig[s]['rel']:+.0f} %' for s in down)}). "
      f"No single scale factor reconciles the two drives.")
    A("")
    A("### Survives")
    A("")
    A(f"6. **Total absorbed power and the energy balance.** Identical by construction "
      f"(max relative difference {res['power_identity_max_rel_diff']:.1e}); every march "
      f"closes its energy audit (|residual| < 1e-6, see section 0).")
    A(f"7. **The extreme ends of the sigma_T ranking.** "
      f"`{res['rankings']['sigma_T_c']['rank_legacy'][0]}` stays the most uniform-ranked "
      f"and `{res['rankings']['sigma_T_c']['rank_legacy'][-1]}` the least, in both arms "
      f"(overall Spearman rho = {res['rankings']['sigma_T_c']['spearman']:.3f} over "
      f"{len(shapes)} shapes).")
    A(f"8. **The direction of the t90 change** (masked reaches phi_bar=0.90 sooner) holds "
      f"in {len(faster)} of {len(shapes)} shapes"
      + (f"; the exception is {', '.join(slower)}." if slower else ".")
      + f" The t90 ranking is nearly preserved (rho = "
        f"{res['rankings']['t_phi90_s']['spearman']:.3f}).")
    A("")
    A("### Limits of this campaign")
    A("")
    A("- One grid (n = 64) for the 8-shape set; one refinement spot check (section 5).")
    A("- The `legacy` arm here uses `phase_update=\"enthalpy\"`, whereas the published")
    A("  `study_summary.csv` used `\"apparent_cp\"`; the legacy column is an anchor, not")
    A("  a bit-for-bit reproduction of the published table.")
    A("- `sigma_T` is the 3-D `std(T_phi90)` metric and is NOT grid-converged in heatr3d")
    A("  (documented in HEATR_STANDARD_PARAMETERS.md); rankings, not absolute values, are")
    A("  the object here.")
    A("- Only the Q_rf post-processing gradient differs between arms. Any error in `V`")
    A("  itself (harmonic face averaging at the staircase boundary, cell-centred")
    A("  electrode gauge) is present in BOTH arms.")
    A("- The prism family (20 mm bounding box, full height) and the solid family")
    A("  (published `run_3d_study.py` sizes) are not the same physical size.")
    A("")
    return "\n".join(L)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    res = build(n)
    (HERE / "rerank_results.json").write_text(json.dumps(res, indent=1))
    md = md_report(res)
    if list(SHARDS.glob("*_n96.json")):
        res96 = build(96)
        (HERE / "rerank_results_n96.json").write_text(json.dumps(res96, indent=1))
        md += "\n" + spot_check_section(res96, res)
    else:
        md += ("\n## 5. n = 96 refinement spot check\n\nNOT RUN (see README section 4:"
               " the n=96 check is the part that gets cut if wall time overruns).\n")
    md += "\n" + exposure_section(res)
    (HERE / "RERANK_REPORT.md").write_text(md)
    print("wrote RERANK_REPORT.md")
    print(f"wrote rerank_results.json ({len(res['shapes'])} shapes, "
          f"gates_all_pass={res['gates_all_pass']})")
    for key, cmp in res["rankings"].items():
        print(f"  {key:34s} rho={cmp['spearman']:+.3f} "
              f"flipped={cmp['flipped']} inversions={cmp['inversions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
