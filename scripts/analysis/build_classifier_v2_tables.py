#!/usr/bin/env python3
"""Tables and confusion matrices for classifier version 1 against version 2.

Reads the version-2 measurement pass (`out_intake/*_v2.json`, written by
`run_intake_classifier_v2.py`) and emits, to `out_intake/`:

  _tables_v2.md              the full per-geometry tables
  classifier_v2_summary.json every number in machine-readable form

Both classifier versions are RE-RUN here from the same measured residual
dictionaries, so the version-1 column is not copied from the earlier pass; it is
recomputed by `geometry_actuator.recommend` on the uniform-map residuals, which
reproduces the stored version-1 numbers (gear8 static 0.4440 and continuous
0.2280, for instance) and keeps the comparison honest.

ONE VERSION-1 NUMBER LEGITIMATELY MOVES. The octagon's symmetry order was
corrected from 10 to 8 in this pass, so its mode set is now static, continuous,
index2, index4, index8 instead of static, continuous, index2, index5, index10.
That changes which modes exist for the octagon under BOTH versions.

Run:  ./.venv312/bin/python scripts/analysis/build_classifier_v2_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from adjoint2d import geometry_actuator as ga            # noqa: E402
from adjoint2d import library_solve as lib               # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_intake"
OUT_LIB = REPO / "fgm_solve_campaign/out_lib"
ORDER = list(lib.SHAPES) + ["gear8", "keyhole"]


def load_all() -> list[dict]:
    rows = []
    for name in ORDER:
        p = OUT / f"{name}_v2.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    return rows


def static_solve_iou(name: str) -> float | None:
    """The library campaign's own solved-static 4 bits per pixel IoU."""
    p = OUT_LIB / f"{name}.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    a = j.get("arms", {}).get("A1_4bpp")
    return float(a["IoU"]) if a else None


def decide(row: dict) -> dict:
    """Both classifier versions, on all three injected maps."""
    order = int(row["symmetry"]["rotational_order"])
    res = row["residual"]
    out = {"v1": ga.recommend(res["uniform"], order).as_json()}
    for basis in ("prop_inverse", "solved"):
        if basis in res:
            out[f"v2_{basis}"] = ga.recommend_v2(res[basis], order,
                                                 basis=basis).as_json()
    return out


def confusion(rows: list[dict], key: str) -> dict:
    """Predicted rotation against the MEASURED rotation-versus-solved outcome."""
    tab = {"true_positive": [], "false_positive": [],
           "true_negative": [], "false_negative": []}
    for r in rows:
        gt = r.get("ground_truth")
        d = r["_decisions"].get(key)
        if not gt or d is None:
            continue
        pred = bool(d["rotation_recommended"])
        truth = gt["outcome"] == "rotation_wins"
        cell = ("true_positive" if (pred and truth) else
                "false_positive" if (pred and not truth) else
                "true_negative" if (not pred and not truth) else
                "false_negative")
        tab[cell].append(r["shape"])
    n_ok = len(tab["true_positive"]) + len(tab["true_negative"])
    n = sum(len(v) for v in tab.values())
    tab["correct"] = n_ok
    tab["n"] = n
    return tab


def _fmt(v, w=6, p=4):
    return f"{v:{w}.{p}f}" if v is not None else " " * w


def main() -> None:
    rows = load_all()
    for r in rows:
        r["_decisions"] = decide(r)
        r["_static_solve_iou"] = static_solve_iou(r["shape"])

    conf = {k: confusion(rows, k) for k in
            ("v1", "v2_prop_inverse", "v2_solved")}

    # out-of-sample context, the version-1 pass's own headline statistic
    spear = {}
    for basis, key in (("uniform", "v1"), ("prop_inverse", "v2_prop_inverse"),
                       ("solved", "v2_solved")):
        xs, ys = [], []
        for r in rows:
            if r["_static_solve_iou"] is None or basis not in r["residual"]:
                continue
            d = r["_decisions"].get(key)
            if d is None:
                continue
            xs.append(d["residual_anisotropy"])
            ys.append(r["_static_solve_iou"])
        if len(xs) >= 4:
            rho, p = spearmanr(xs, ys)
            spear[basis] = {"rho": float(rho), "p": float(p), "n": len(xs)}

    # the reduction-factor separation, the quantity the bands rest on
    sep = {}
    for basis in ("uniform", "prop_inverse", "solved"):
        wins, fails = [], []
        for r in rows:
            gt = r.get("ground_truth")
            if not gt or basis not in r["residual"]:
                continue
            d = r["residual"][basis]
            others = {k: v for k, v in d.items() if k != "static"}
            best = min(others.values()) if others else d["static"]
            red = d["static"] / max(min(best, d["static"]), 1e-30)
            (wins if gt["outcome"] == "rotation_wins" else fails).append(
                (r["shape"], red))
        wins.sort(key=lambda t: t[1])
        fails.sort(key=lambda t: t[1])
        sep[basis] = {
            "rotation_wins": [[a, float(b)] for a, b in wins],
            "rotation_fails": [[a, float(b)] for a, b in fails],
            "min_win": float(wins[0][1]) if wins else None,
            "max_fail": float(fails[-1][1]) if fails else None,
            "separates": bool(wins and fails and wins[0][1] > fails[-1][1]),
        }
        if sep[basis]["separates"]:
            sep[basis]["margin_pct"] = 100.0 * (sep[basis]["min_win"]
                                                / sep[basis]["max_fail"] - 1.0)

    summary = {
        "n_geometries": len(rows),
        "bands_v1": {"A_MODE_SUFFICES": ga.A_MODE_SUFFICES,
                     "A_PHYSICAL_LIMIT": ga.A_PHYSICAL_LIMIT,
                     "MIN_REDUCTION": ga.MIN_REDUCTION},
        "bands_v2": {"A2_MODE_SUFFICES": ga.A2_MODE_SUFFICES,
                     "A2_PHYSICAL_LIMIT": ga.A2_PHYSICAL_LIMIT,
                     "MIN_REDUCTION_V2": ga.MIN_REDUCTION_V2,
                     "default_basis": ga.DEFAULT_V2_BASIS,
                     "prop_inverse_magnitude": ga.PROP_INVERSE_MAGNITUDE},
        "confusion": conf,
        "spearman_best_residual_vs_static_solve_iou": spear,
        "reduction_separation": sep,
        "per_geometry": {r["shape"]: {
            "order": r["symmetry"]["rotational_order"],
            "point_group": r["symmetry"]["point_group"],
            "residual": r["residual"],
            "decisions": r["_decisions"],
            "static_solve_4bpp_iou": r["_static_solve_iou"],
            "ground_truth": r.get("ground_truth"),
            "magnitude_sensitivity": r["prop_inverse_magnitude_sensitivity"],
        } for r in rows},
    }
    (OUT / "classifier_v2_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))

    # ---------------- markdown ----------------
    L: list[str] = []
    L.append("# Classifier version 2 tables\n")
    L.append("Grid 120, electrical state B, conductivity channel, uniform arm "
             "calibrated to 500 W/m. The anisotropy metric is invariant to the "
             "drive scale. Every number COMPUTED in this pass.\n")

    L.append("\n## Table V1. Residual anisotropy per mode, three injected maps\n")
    L.append("| shape | order | mode | uniform (v1) | prop_inverse (free) | "
             "solved (one forward) |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        for mode in r["modes"]:
            L.append(f"| {r['shape']} | {r['symmetry']['rotational_order']} | "
                     f"{mode} | {r['residual']['uniform'][mode]:.4f} | "
                     f"{r['residual']['prop_inverse'][mode]:.4f} | "
                     + (f"{r['residual']['solved'][mode]:.4f} |"
                        if "solved" in r["residual"] else " n/a |"))

    L.append("\n## Table V2. Recommendation per geometry, version 1 against "
             "version 2\n")
    L.append("| shape | measured outcome | v1 class / mode / reduction | "
             "v2 free class / mode / reduction | v2 one-forward class / mode / "
             "reduction |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        gt = r.get("ground_truth")
        g = gt["outcome"] if gt else "not measured"
        cells = []
        for k in ("v1", "v2_prop_inverse", "v2_solved"):
            d = r["_decisions"].get(k)
            cells.append("n/a" if d is None else
                         f"{d['actuator_class']} / {d['mode']} / "
                         f"{d['reduction_factor']:.3f}")
        L.append(f"| {r['shape']} | {g} | " + " | ".join(cells) + " |")

    L.append("\n## Table V3. Confusion against the seven MEASURED outcomes\n")
    L.append("Prediction = does the classifier recommend rotation. Truth = did "
             "the best rotating arm beat the best SOLVED STATIC arm.\n")
    L.append("| classifier | correct | true positive | false positive | "
             "true negative | false negative |")
    L.append("|---|---|---|---|---|---|")
    for k, c in conf.items():
        L.append(f"| {k} | **{c['correct']} of {c['n']}** | "
                 f"{', '.join(c['true_positive']) or '-'} | "
                 f"{', '.join(c['false_positive']) or '-'} | "
                 f"{', '.join(c['true_negative']) or '-'} | "
                 f"{', '.join(c['false_negative']) or '-'} |")

    L.append("\n## Table V4. The reduction factor, sorted, per injected map\n")
    for basis, s in sep.items():
        L.append(f"\n**{basis}**: separates = {s['separates']}"
                 + (f", margin {s['margin_pct']:.1f} percent"
                    if s.get("margin_pct") is not None else ""))
        L.append("")
        L.append("| outcome | " + " | ".join(
            f"{a} {b:.3f}" for a, b in s["rotation_fails"]) + " |")
        L.append("|---|" + "---|" * len(s["rotation_fails"]))
        L.append("| rotation FAILS | " + " | ".join(
            f"{b:.3f}" for _a, b in s["rotation_fails"]) + " |")
        L.append("")
        L.append("| outcome | " + " | ".join(
            f"{a} {b:.3f}" for a, b in s["rotation_wins"]) + " |")
        L.append("|---|" + "---|" * len(s["rotation_wins"]))
        L.append("| rotation WINS | " + " | ".join(
            f"{b:.3f}" for _a, b in s["rotation_wins"]) + " |")

    L.append("\n## Table V5. Magnitude sensitivity of the FREE stand-in\n")
    L.append("Reduction factor of the best mode at three stand-in magnitudes.\n")
    L.append("| shape | measured outcome | m = 0.50 | m = 1.00 (default) | m = 1.50 |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        gt = r.get("ground_truth")
        cells = []
        for m in ("0.50", "1.00", "1.50"):
            d = r["prop_inverse_magnitude_sensitivity"].get(m)
            if d is None:
                cells.append("n/a")
                continue
            o = {k: v for k, v in d.items() if k != "static"}
            best = min(o.values()) if o else d["static"]
            cells.append(f"{d['static'] / max(min(best, d['static']), 1e-30):.3f}")
        L.append(f"| {r['shape']} | {gt['outcome'] if gt else '-'} | "
                 + " | ".join(cells) + " |")

    L.append("\n## Table V6. Out-of-sample rank correlation, for context\n")
    L.append("Spearman rank correlation between the best mode's residual and "
             "the library campaign's own solved-static 4 bits per pixel IoU. "
             "This is context, not the version-2 criterion.\n")
    L.append("| injected map | rho | p | n |")
    L.append("|---|---|---|---|")
    for b, s in spear.items():
        L.append(f"| {b} | {s['rho']:.3f} | {s['p']:.2e} | {s['n']} |")

    (OUT / "_tables_v2.md").write_text("\n".join(L) + "\n")
    print(json.dumps({k: {"correct": v["correct"], "n": v["n"],
                          "false_positive": v["false_positive"],
                          "false_negative": v["false_negative"]}
                      for k, v in conf.items()}, indent=2))
    print("separation:", json.dumps({k: {"separates": v["separates"],
                                         "margin_pct": v.get("margin_pct")}
                                     for k, v in sep.items()}, indent=2))
    print("wrote", OUT / "_tables_v2.md")


if __name__ == "__main__":
    main()
