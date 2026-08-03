#!/usr/bin/env python3
"""The cross-scatter variants figure and its merged table.

Three curves of the SAME quantity, the cross's rotating uniform-dopant
intersection over union at its own optimal stop, across the seven ladder grids:

  ORIGINAL   the published ladder (`ROTATING_GRID_LADDER_REPORT.md` Section 3.1)
  VARIANT A  the same ladder with both cross boundaries snapped to whole cell
             multiples at every grid, so the rasterization rounding the
             candidate mechanism blames is identically zero
  VARIANT B  the original runs re-scored with the sub-cell melt area fill, so
             the non-smooth `phi >= 0.5` threshold is removed from the metric

plus the keyhole's original rotating uniform arm as the converged reference,
because the whole question is whether the cross can be made to look like it.

Run:
  ./.venv312/bin/python scripts/analysis/make_cross_variants_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                            # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from rot_ladder_variants import pearson_r, spread             # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_rot_ladder"
FIGS = REPO / "fgm_solve_campaign/figs_rot_ladder"
ARMS = ("ROT_uniform", "STATIC_uniform", "QS_uniform", "ROT_transfer")
SOLVED_CLASS_IOU = 0.95


def merged(names: list[str]) -> dict:
    grids: dict[int, dict] = {}
    src = []
    for nm in names:
        p = OUT / f"{nm}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        src.append(str(p))
        for g, rec in d["grids"].items():
            grids[int(g)] = rec
    if not grids:
        raise SystemExit(f"no ladder files among {names}")
    return {"sources": src, "grids": grids}


def seq(m: dict, arm: str, key: str = "IoU"):
    ns = sorted(n for n in m["grids"] if arm in m["grids"][n]["arms"])
    return (np.array(ns, dtype=float),
            np.array([m["grids"][n]["arms"][arm][key] for n in ns], dtype=float))


def seq_smooth(sm: dict, arm: str, key: str):
    ns = sorted(int(g) for g in sm["grids"] if arm in sm["grids"][g]["arms"])
    return (np.array(ns, dtype=float),
            np.array([sm["grids"][str(n)]["arms"][arm][key] for n in ns],
                     dtype=float))


def rankings(m: dict) -> dict:
    """Does the rotating arm still beat static, and the map beat uniform?"""
    rows, ok = [], True
    for n in sorted(m["grids"]):
        a = m["grids"][n]["arms"]
        for better, worse, why in (("ROT_uniform", "STATIC_uniform",
                                    "rotation beats static"),
                                   ("ROT_transfer", "ROT_uniform",
                                    "the transferred map beats uniform")):
            if better not in a or worse not in a:
                continue
            for metric, sign in (("J", -1.0), ("IoU", +1.0)):
                p = bool(sign * (a[better][metric] - a[worse][metric]) > 0.0)
                ok = ok and p
                rows.append({"n_grid": n, "test": f"{why} on {metric}",
                             "better": a[better][metric],
                             "worse": a[worse][metric], "PASS": p})
    return {"n_tests": len(rows), "ALL_PASS": bool(ok) if rows else None,
            "status": ("PASS" if (rows and ok) else "FAIL" if rows
                       else "NOT CHECKED, no comparable arm pair"),
            "rows": rows}


def gates(m: dict) -> dict:
    ev, hz, ce, res = [], [], [], []
    for n in sorted(m["grids"]):
        for arm, a in m["grids"][n]["arms"].items():
            res.append(100.0 * a["energy_gate"]["rel_residual_at_index"])
            if not a["energy_gate"]["PASS"]:
                ev.append(f"{n}:{arm}")
            if a["t_stop_at_horizon"]:
                hz.append(f"{n}:{arm}")
            if a.get("over_ceiling_250c"):
                ce.append(f"{n}:{arm} {a['max_T_at_stop_c']:.1f} C")
    return {"energy_gate_violations": ev,
            "max_energy_residual_pct": max(res) if res else None,
            "horizon_flags": hz, "over_ceiling_250c": ce,
            "status": "PASS" if not ev else "FAIL"}


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    orig = merged(["cross_ladder", "cross_align_probe_ladder"])
    snapA = merged(["cross_snapA_ladder"])
    keyh = merged(["keyhole_ladder", "keyhole_align_probe_ladder"])
    smoothB = json.loads((OUT / "cross_smoothB.json").read_text())
    smoothB_k = json.loads((OUT / "keyhole_smoothB.json").read_text())
    raster = json.loads((OUT / "cross_raster_geometry.json").read_text())
    raster_snap = json.loads((OUT / "cross_raster_geometry_snapA.json").read_text())

    n_o, iou_o = seq(orig, "ROT_uniform")
    n_a, iou_a = seq(snapA, "ROT_uniform")
    n_b, iou_b = seq_smooth(smoothB, "ROT_uniform", "IoU_subcell_melt_vs_chi")
    n_k, iou_k = seq(keyh, "ROT_uniform")

    # the rasterization predictor, aligned to the grids each series has
    rr_by_n = {int(r["n_grid"]): float(r["limb_rel_err_pct"])
               for r in raster["rows"]}
    lim_o = np.array([rr_by_n[int(v)] for v in n_o])

    series = [
        ("ORIGINAL, binary metric", n_o, iou_o, "#1f77b4", "o", "-"),
        ("VARIANT A, snapped geometry", n_a, iou_a, "#d62728", "s", "-"),
        ("VARIANT B, sub-cell melt metric", n_b, iou_b, "#9467bd", "^", "--"),
        ("keyhole, original (converged reference)", n_k, iou_k, "#7f7f7f",
         "d", ":"),
    ]

    fig = plt.figure(figsize=(14.0, 5.2), dpi=180)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.95, 0.95], wspace=0.26,
                          left=0.05, right=0.99, top=0.84, bottom=0.14)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    # -- panel A, the three ladders ----------------------------------------
    box = []
    for lab, n, v, c, mk, ls in series:
        axa.plot(n, v, marker=mk, ls=ls, color=c, lw=1.9, ms=6, label=lab)
        sel = v[n >= 160]
        d = np.diff(v)
        box.append((lab.split(",")[0], c, float(sel.max() - sel.min()),
                    int(np.sum(np.sign(d)[1:] != np.sign(d)[:-1]))))
    axa.text(0.985, 0.985, "spread over n >= 160   sign changes",
             transform=axa.transAxes, fontsize=7.8, ha="right", va="top",
             color="k")
    for i, (lab, c, sp, sc) in enumerate(box):
        axa.text(0.985, 0.945 - 0.045 * i,
                 f"{lab}: {sp:.4f}          {sc}",
                 transform=axa.transAxes, fontsize=7.8, ha="right", va="top",
                 color=c)
    axa.axhline(SOLVED_CLASS_IOU, color="#2ca02c", ls="--", lw=1.1)
    axa.text(150, SOLVED_CLASS_IOU - 0.006, "SOLVED class 0.95", fontsize=8,
             color="#2ca02c", va="top", ha="center")
    axa.axvline(120, color="k", ls=":", lw=1.0, alpha=0.5)
    axa.text(121, 0.535, "grid 120,\nthe solve grid", fontsize=7.5, alpha=0.8)
    axa.set_xscale("log")
    axa.set_xticks([96, 120, 160, 180, 200, 240, 360])
    axa.set_xticklabels(["96", "120", "160", "180", "200", "240", "360"],
                        fontsize=8)
    axa.minorticks_off()
    axa.set_xlim(90, 380)
    axa.set_ylim(0.52, 1.04)
    axa.grid(alpha=0.25)
    axa.set_xlabel("grid number n (cells across the 60 mm chamber)", fontsize=9)
    axa.set_ylabel("intersection over union, rotating uniform arm", fontsize=9)
    axa.set_title("A. snapping removes the ALTERNATION; smoothing the metric "
                  "changes nothing", fontsize=9.5)
    axa.legend(fontsize=7.6, loc="lower left", framealpha=0.92)

    # -- panel B, the ladder step ------------------------------------------
    for lab, n, v, c, mk, ls in series:
        d = np.abs(np.diff(v))
        axb.plot(n[1:], np.maximum(d, 3e-4), marker=mk, ls=ls, color=c, lw=1.9,
                 ms=6, label=lab.split(",")[0])
    axb.set_yscale("log")
    axb.set_xscale("log")
    axb.set_ylim(3e-4, 3e-1)
    axb.set_xticks([120, 160, 180, 200, 240, 360])
    axb.set_xticklabels(["120", "160", "180", "200", "240", "360"], fontsize=8)
    axb.minorticks_off()
    axb.grid(alpha=0.25, which="both")
    axb.set_xlabel("grid number n, step from the previous grid", fontsize=9)
    axb.set_ylabel("size of the ladder step", fontsize=9)
    axb.set_title("B. only the keyhole's step walks down", fontsize=10)
    axb.legend(fontsize=7.6, loc="lower left", framealpha=0.92)

    # -- panel C, the candidate mechanism, before and after -----------------
    r_o = pearson_r(lim_o, iou_o)
    r_b = pearson_r(lim_o, iou_b)
    axc.scatter(lim_o, iou_o, c="#1f77b4", marker="o", s=48, zorder=3,
                label=f"ORIGINAL, r = {r_o:+.3f}")
    axc.scatter(lim_o, iou_b, c="#9467bd", marker="^", s=48, zorder=3,
                label=f"VARIANT B, r = {r_b:+.3f}")
    for xv, yv, nv in zip(lim_o, iou_o, n_o):
        axc.annotate(f"{int(nv)}", (xv, yv), fontsize=7, xytext=(3, 4),
                     textcoords="offset points", color="#1f77b4")
    axc.scatter(np.zeros_like(iou_a), iou_a, c="#d62728", marker="s", s=48,
                zorder=4, label="VARIANT A, rounding removed by construction")
    axc.vlines(0.0, float(iou_a.min()), float(iou_a.max()), color="#d62728",
               lw=2.0, zorder=2)
    axc.annotate(f"VARIANT A, at ZERO rounding error:\n"
                 f"{float(iou_a.max() - iou_a.min()):.4f} of range over the "
                 f"whole ladder,\n"
                 f"{spread(iou_a, 160, n_a):.4f} over n >= 160,\n"
                 f"monotone in the grid instead of alternating",
                 xy=(0.02, 0.62), xycoords="axes fraction", fontsize=8,
                 color="#d62728", va="center")
    axc.axvline(0.0, color="k", lw=0.8, alpha=0.4)
    axc.grid(alpha=0.25)
    axc.set_xlabel("limb-boundary rasterization rounding error, percent",
                   fontsize=9)
    axc.set_ylabel("intersection over union, rotating uniform arm", fontsize=9)
    axc.set_title("C. the predictor removed: range shrinks, does not vanish",
                  fontsize=9.5)
    axc.legend(fontsize=7.4, loc="lower right", framealpha=0.92)

    fig.suptitle(
        "Does the cross's rotating grid scatter survive snapping the geometry "
        "to the grid (variant A) and smoothing the melt indicator (variant B)? "
        "Forward runs only; variant B re-runs no physics at all.",
        fontsize=10.5, y=0.955)
    p = FIGS / "fig_cross_scatter_variants.png"
    fig.savefig(p)
    print(f"wrote {p}", flush=True)

    # -- the merged table ---------------------------------------------------
    def block(name, n, v, extra=None):
        d = {"grids": [int(x) for x in n], "IoU": [float(x) for x in v],
             "spread_all": spread(v),
             "spread_n_ge_160": spread(v, 160, n),
             "spread_n_ge_200": spread(v, 200, n),
             "steps": [float(x) for x in np.diff(v)],
             "last_step_abs": float(abs(np.diff(v)[-1])),
             "n_sign_changes_in_steps": int(np.sum(
                 np.sign(np.diff(v))[1:] != np.sign(np.diff(v))[:-1])),
             "pearson_r_vs_limb_rounding": pearson_r(
                 [rr_by_n[int(x)] for x in n], v)}
        if extra:
            d.update(extra)
        return {name: d}

    tab: dict = {"shape": "cross", "arm": "ROT_uniform"}
    tab.update(block("original_binary_metric", n_o, iou_o))
    tab.update(block("variantA_snapped_geometry", n_a, iou_a, {
        "pearson_r_vs_limb_rounding": None,
        "note": "the rasterization rounding is identically zero at every grid "
                "here, so the correlation against it is undefined and is "
                "reported as null rather than as a number",
        "per_grid_geometry_delta": [
            {"n_grid": g["n_grid"], **{k: g["snap"][k] for k in (
                "limb_cells", "arm_cells", "d_limb_m", "d_arm_m",
                "d_limb_frac_of_cell", "d_arm_frac_of_cell",
                "d_limb_pct_of_dimension", "d_arm_pct_of_dimension")}}
            for g in (snapA["grids"][n] for n in sorted(snapA["grids"]))],
        "raster_check": [
            {"n_grid": r["n_grid"], "limb_rel_err_pct": r["limb_rel_err_pct"],
             "arm_rel_err_pct": r["arm_rel_err_pct"],
             "raster_minus_area_pct": r["raster_minus_area_pct"]}
            for r in raster_snap["rows"]],
    }))
    tab.update(block("variantB_subcell_melt_metric", n_b, iou_b))
    smoothAB = json.loads((OUT / "cross_snapA_smoothB.json").read_text())
    n_ab, iou_ab = seq_smooth(smoothAB, "ROT_uniform",
                              "IoU_subcell_melt_vs_chi")
    tab.update(block("variantA_plus_B_both_together", n_ab, iou_ab, {
        "pearson_r_vs_limb_rounding": None,
        "note": "the snapped ladder re-scored with the sub-cell melt metric: "
                "both candidate causes removed at once",
    }))
    tab["variantAB_all_metrics"] = smoothAB["ladder_summary"]
    tab["variantAB_rescoring_gate"] = smoothAB["rescoring_gate"]
    tab.update(block("keyhole_original_reference", n_k, iou_k, {
        "note": "the converged shape, carried as the reference the cross is "
                "being compared against"}))

    # variant A moves the part itself by up to half a cell, so its own part
    # area is the predictor that REPLACES the rasterization rounding error.
    area_a = np.array([float(snapA["grids"][n]["chi"]["area_m2"])
                       for n in sorted(snapA["grids"])])
    area_o = np.array([float(orig["grids"][n]["chi"]["area_m2"])
                       for n in sorted(orig["grids"])])
    tab["variantA_area_predictor"] = {
        "chi_area_m2": [float(v) for v in area_a],
        "chi_area_rel_delta_pct_vs_nominal":
            [100.0 * (float(v) - float(np.mean(area_o))) / float(np.mean(area_o))
             for v in area_a],
        "pearson_r_IoU_vs_own_area": pearson_r(area_a, iou_a),
        "note": "the snap holds the RASTER exact and lets the PART move by up "
                "to half a cell, so the part area is no longer grid "
                "independent; this is the price of the variant and the "
                "correlation says how much of what is left it explains",
    }
    tab["variantB_all_metrics"] = smoothB["ladder_summary"]
    tab["variantB_keyhole_all_metrics"] = smoothB_k["ladder_summary"]
    tab["variantB_rescoring_gate"] = smoothB["rescoring_gate"]
    tab["variantB_keyhole_rescoring_gate"] = smoothB_k["rescoring_gate"]
    tab["gates"] = {"original": gates(orig), "variantA": gates(snapA),
                    "keyhole_original": gates(keyh)}
    tab["rankings"] = {"original": rankings(orig), "variantA": rankings(snapA),
                       "keyhole_original": rankings(keyh)}
    tab["reproduction_gates"] = {
        "original": json.loads((OUT / "cross_ladder.json").read_text())
                        .get("reproduction_gate", {}),
        "variantA": json.loads((OUT / "cross_snapA_ladder.json").read_text())
                        .get("reproduction_gate", {}),
    }
    tab["wall_s"] = {
        "variantA_total": json.loads(
            (OUT / "cross_snapA_ladder.json").read_text()).get("wall_s"),
        "variantA_by_grid": json.loads(
            (OUT / "cross_snapA_ladder.json").read_text()).get("wall_by_grid_s"),
        "variantB_rescoring": smoothB.get("wall_s"),
    }
    tab["sources"] = {"original": orig["sources"], "variantA": snapA["sources"],
                      "variantB": str(OUT / "cross_smoothB.json"),
                      "keyhole": keyh["sources"]}
    (OUT / "cross_variants_table.json").write_text(
        json.dumps(tab, indent=2, default=float))
    print(f"wrote {OUT / 'cross_variants_table.json'}", flush=True)

    for k in ("original_binary_metric", "variantA_snapped_geometry",
              "variantB_subcell_melt_metric", "variantA_plus_B_both_together",
              "keyhole_original_reference"):
        d = tab[k]
        r = d["pearson_r_vs_limb_rounding"]
        print(f"{k:34s} IoU " + " ".join(f"{v:.4f}" for v in d["IoU"]) +
              f"  spread>=160 {d['spread_n_ge_160']:.4f}"
              f"  last {d['last_step_abs']:.4f}"
              f"  sign changes {d['n_sign_changes_in_steps']}"
              + ("  r_vs_limb NOT DEFINED (rounding is zero at every grid)"
                 if r is None else f"  r_vs_limb {r:+.3f}"), flush=True)
    for k, v in tab["gates"].items():
        print(f"gate {k:18s} {v['status']}, max energy residual "
              f"{v['max_energy_residual_pct']:.2f} %, horizon {v['horizon_flags']}, "
              f"ceiling {len(v['over_ceiling_250c'])} flags", flush=True)
    for k, v in tab["rankings"].items():
        print(f"rankings {k:18s} {v['status']} ({v['n_tests']} tests)",
              flush=True)


if __name__ == "__main__":
    main()
