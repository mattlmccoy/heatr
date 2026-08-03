"""Score the solve-at-price question and draw its one figure.

THE QUESTION. `HEATR_V2_ROLLOUT_NOTES.md` v2.1.0 set the production
out-of-bounds price to w_out = 2.0 on a READ-STATE trade curve that re-read maps
solved at w_out = 1. This compares, at each production price, three arms, every
one read at ITS OWN J_asym argmin over its own stored full-horizon trajectory:

  SOLVE_AT_PRICE   the map solved with the objective carrying that price.
  REREAD_AT_PRICE  the stored melt-region-solved 4-bits-per-pixel map re-read at
                   that price. This is the shipped v2.1.0 recipe.
  UNIFORM          uniform saturation, the control.

The decision floor is 3 percent of J_asym: a gain smaller than that is inside
the run-to-run and quantization noise of this campaign and is reported as no
difference.

Run:
  ./.venv312/bin/python analyze_wout.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adjoint2d.map_structure import r2_against_proxy

HERE = Path(__file__).resolve().parent
OUT_W = HERE / "out_wout"
OUT_A = HERE / "out_asym"
FIGS = HERE / "figs_wout"
SHAPES = ("square", "hexagon", "triangle")
PRICES = (2.0, 3.0)
DECISION_FLOOR_PCT = 3.0
ARMS = {"SOLVE_AT_PRICE": "ASYM_mma_4bpp",
        "REREAD_AT_PRICE": "PHI4_prev",
        "UNIFORM": "U_uniform"}


def tag(w: float) -> str:
    return f"_w{str(w).replace('.', 'p')}"


def load(shape: str, w: float) -> dict:
    return json.loads((OUT_W / f"{shape}{tag(w)}.json").read_text())


def row(m: dict, w: float) -> dict:
    """The scored line of one arm, decomposed and with its stop stated."""
    return {
        "J_asym": m["J_asym"],
        "J_out_weighted": m["J_asym_out"],
        "J_out_unweighted": m["J_asym_out"] / w,
        "J_in": m["J_asym_in"],
        "stop_s": m["asym_stop_s"],
        "stop_index": m["asym_stop_index"],
        "at_horizon": m["asym_stop_at_horizon"],
        "J_phi_at_asym_stop": m["J_phi_at_asym_stop"],
        "J_phi_at_own_melt_stop": m["J_phi"],
        "IoU": m["IoU"],
        "growth_pct": m["growth_pct"],
        "under_pct": m["under_pct"],
        "mean_rho_rel_part": m["mean_rho_rel_part"],
        "frac_at_or_above_floor": m["frac_part_at_or_above_floor"],
        "P_abs_W_per_m": m["P_abs_W_per_m"],
        "energy_gate_PASS": m["energy_gate_at_asym_stop"]["PASS"],
        "energy_gate_rel_residual": m["energy_gate_at_asym_stop"]["rel_residual"],
        "energy_gate_threshold": m["energy_gate_at_asym_stop"]["threshold"],
        "spec_PASS": m["spec"]["PASS"],
    }


def curve_at(J_out_c, J_in_c, w: float) -> float:
    """J_asym of a stored trajectory at an arbitrary price, the free identity."""
    tot = float(w) * np.asarray(J_out_c, float) + np.asarray(J_in_c, float)
    return float(np.min(tot))


def cross_check(shape: str, w: float, res: dict) -> dict:
    """The re-read arm must reproduce the STORED w_out = 1 pass exactly.

    `out_asym/<shape>.json` already carries the melt-solved map's trade-curve
    row at this price, computed from the full-resolution stored curves of that
    pass. The new run marches the same map again, so agreement is a determinism
    check on the whole forward and a reuse of the stored numbers rather than a
    second measurement of them.
    """
    old = json.loads((OUT_A / f"{shape}.json").read_text())
    stored = {r["w_out"]: r for r in old["arms"]["PHI4_prev"]["trade_curve"]}[w]
    new = res["arms"]["PHI4_prev"]
    d = abs(new["J_asym"] - stored["J_asym"]) / max(abs(stored["J_asym"]), 1e-30)
    return {"stored_J_asym": stored["J_asym"], "stored_index": stored["index"],
            "new_J_asym": new["J_asym"], "new_index": new["asym_stop_index"],
            "rel_diff": d, "index_match": bool(stored["index"] == new["asym_stop_index"])}


def map_family(shape: str, w: float) -> dict:
    """Is the solved-at-price map the melt-solved family, or a new one?"""
    new = np.load(OUT_W / f"{shape}{tag(w)}_maps.npz")
    old = np.load(OUT_A / f"{shape}_maps.npz")
    pm = np.asarray(new["part_mask"], bool)
    s = np.asarray(new["ASYM_mma_4bpp"], float)
    refs = {"melt_solved": np.asarray(new["PHI4_prev"], float),
            "asym_solved_w1": np.asarray(old["ASYM_mma_4bpp"], float),
            "uniform": np.asarray(new["U_uniform"], float)}
    out = {"mean_in_part": float(np.mean(s[pm])),
           "std_in_part": float(np.std(s[pm]))}
    for k, r in refs.items():
        out[f"r2_vs_{k}"] = r2_against_proxy(s, r, pm)
        out[f"rms_vs_{k}"] = float(np.sqrt(np.mean((s[pm] - r[pm]) ** 2)))
    return out


def collect() -> dict:
    res: dict = {"decision_floor_pct": DECISION_FLOOR_PCT, "grid": [120, 120],
                 "shapes": {}}
    for sh in SHAPES:
        e: dict = {"prices": {}}
        for w in PRICES:
            r = load(sh, w)
            arms = {k: row(r["arms"][v], w) for k, v in ARMS.items()
                    if v in r["arms"]}
            wp = OUT_W / f"{sh}{tag(w)}_warm.json"
            if wp.exists():
                wm = json.loads(wp.read_text())
                arms["WARM_AT_PRICE"] = row(wm["arms"]["WARM_mma_4bpp"], w)
                arms["WARM_AT_PRICE"]["filtered_melt_start_J_asym"] = \
                    wm["start_J_asym"]
                arms["WARM_AT_PRICE"]["n_gradient_evals"] = wm["n_gradient_evals"]
            solved = arms["SOLVE_AT_PRICE"]["J_asym"]
            reread = arms["REREAD_AT_PRICE"]["J_asym"]
            gain = 100.0 * (reread - solved) / reread
            e["prices"][str(w)] = {
                "arms": arms,
                "gain_solve_over_reread_pct": gain,
                "beats_reread": bool(gain > DECISION_FLOOR_PCT),
                "gain_solve_over_filtered_melt_pct": (
                    100.0 * (arms["WARM_AT_PRICE"]["filtered_melt_start_J_asym"]
                             - solved)
                    / arms["WARM_AT_PRICE"]["filtered_melt_start_J_asym"]
                    if "WARM_AT_PRICE" in arms else None),
                "cold_warm_spread_pct": (
                    100.0 * abs(arms["WARM_AT_PRICE"]["J_asym"] - solved)
                    / min(arms["WARM_AT_PRICE"]["J_asym"], solved)
                    if "WARM_AT_PRICE" in arms else None),
                "cross_check": cross_check(sh, w, r),
                "map_family": map_family(sh, w),
                "n_gradient_evals": r["cost"]["n_gradient_evals_per_optimizer"],
                "adjoint_forward_ratio": r["cost"]["ratio"],
                "wall_s": r["wall_s"],
                "energy_gate_violations": r["energy_gate_violations"],
                "solve_first_J": r["solves"]["mma"]["rows"][0]["J_asym"],
                "solve_best_J": min(x["J_asym"] for x in r["solves"]["mma"]["rows"]),
                "frac_followups_improving":
                    r["solves"]["mma"]["info"].get("frac_followups_improving"),
                "hinge_active_frac_first_eval":
                    r["solves"]["mma"]["rows"][0]["hinge_active_frac"],
            }
            # the stored curves of every arm of THIS run, for the figure
            e["prices"][str(w)]["reread_curve"] = {
                k: {"w": list(np.round(np.geomspace(0.25, 20.0, 25), 4)),
                    "J": [curve_at(r["arms"][v]["J_out_curve_stride10"],
                                   r["arms"][v]["J_in_curve_stride10"], ww)
                          for ww in np.geomspace(0.25, 20.0, 25)]}
                for k, v in ARMS.items() if v in r["arms"]}
        res["shapes"][sh] = e
    return res


def figure(res: dict, path: Path) -> None:
    """One figure: the trade curve with BOTH arms on it.

    The lines are what a stored map costs when it is RE-READ at each price (the
    free identity of the objective). The filled markers are what the map SOLVED
    at that price costs. The vertical distance between a marker and the
    melt-solved line at the same price IS the answer to the question.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4),
                             gridspec_kw={"height_ratios": [1.15, 1.0]})
    for j, sh in enumerate(SHAPES):
        ax = axes[0, j]
        e = res["shapes"][sh]
        c = e["prices"]["2.0"]["reread_curve"]
        ax.plot(c["UNIFORM"]["w"], c["UNIFORM"]["J"], color="0.55", lw=1.6,
                ls=":", label="uniform, re-read")
        ax.plot(c["REREAD_AT_PRICE"]["w"], c["REREAD_AT_PRICE"]["J"],
                color="#1f77b4", lw=2.2,
                label="melt-solved map, re-read (shipped recipe)")
        for w, mk in zip(PRICES, ("o", "s")):
            p = e["prices"][str(w)]
            js = p["arms"]["SOLVE_AT_PRICE"]["J_asym"]
            jr = p["arms"]["REREAD_AT_PRICE"]["J_asym"]
            ax.plot([w], [jr], mk, color="#1f77b4", ms=8, mfc="white", mew=1.8)
            ax.plot([w], [js], mk, color="#d62728", ms=9,
                    label="solved AT the price, cold start" if w == PRICES[0]
                    else None)
            ax.annotate("", xy=(w, js), xytext=(w, jr),
                        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6))
            g = p["gain_solve_over_reread_pct"]
            ax.text(w * (0.42 if w == PRICES[0] else 1.12), 0.5 * (js + jr),
                    f"{g:+.1f}%", color="#d62728", fontsize=9, va="center",
                    ha="right" if w == PRICES[0] else "left",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
            if "WARM_AT_PRICE" in p["arms"]:
                jw = p["arms"]["WARM_AT_PRICE"]["J_asym"]
                jf = p["arms"]["WARM_AT_PRICE"]["filtered_melt_start_J_asym"]
                ax.plot([w], [jw], mk, color="#ff7f0e", ms=8, mfc="none", mew=2.0,
                        label="solved AT the price, warm start"
                        if w == PRICES[0] else None)
                ax.plot([w], [jf], "_", color="0.25", ms=13, mew=2.0,
                        label="melt-solved map after the 1.0 mm design filter"
                        if w == PRICES[0] else None)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 3, 5, 10, 20])
        ax.set_xticklabels(["0.25", "0.5", "1", "2", "3", "5", "10", "20"])
        ax.set_xlabel("out-of-bounds price $w_{out}$")
        if j == 0:
            ax.set_ylabel("$J_{asym}$ at that arm's own argmin stop")
        wins = all(e["prices"][str(w)]["beats_reread"] for w in PRICES)
        ax.set_title(f"{sh}: solving at the price "
                     f"{'WINS' if wins else 'LOSES'}",
                     fontsize=11,
                     color=("#2a7f2a" if wins else "#b02020"))
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7.6, loc="upper left")

        # row 2: the outcome the price was chosen for
        ax2 = axes[1, j]
        w_arr = np.array(PRICES)
        for key, col, lab in (("REREAD_AT_PRICE", "#1f77b4", "re-read"),
                              ("SOLVE_AT_PRICE", "#d62728", "solved, cold"),
                              ("WARM_AT_PRICE", "#ff7f0e", "solved, warm")):
            if key not in e["prices"][str(PRICES[0])]["arms"]:
                continue
            gr = [e["prices"][str(w)]["arms"][key]["growth_pct"] for w in PRICES]
            rh = [e["prices"][str(w)]["arms"][key]["mean_rho_rel_part"] for w in PRICES]
            ax2.plot(gr, rh, "-", color=col, lw=1.4, alpha=0.7)
            ax2.scatter(gr, rh, s=[55, 95], color=col, label=lab, zorder=3)
            for g_, r_, w_ in zip(gr, rh, w_arr):
                ax2.annotate(f"w={w_:g}", (g_, r_), textcoords="offset points",
                             xytext=(6, -9), fontsize=7.5, color=col)
        ax2.axhline(0.85, color="k", ls="--", lw=1.0)
        ax2.text(0.02, 0.855, "density floor 0.85", fontsize=7.5,
                 transform=ax2.get_yaxis_transform())
        ax2.set_xlabel("bed growth, percent of part cell count")
        if j == 0:
            ax2.set_ylabel("mean in-bounds relative density")
            ax2.legend(fontsize=8, loc="best")
        ax2.grid(alpha=0.3)
    fig.suptitle("Solving at the production out-of-bounds price against re-reading "
                 "at it. Grid 120 x 120, floor 0.85, filtered production recipe, "
                 "method of moving asymptotes, 40 forward-equivalents.\n"
                 "Every arm read at its own $J_{asym}$ argmin over its own "
                 "full-horizon trajectory. Melted region is melt fraction >= 0.5.",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> dict:
    res = collect()
    (OUT_W / "wout_summary.json").write_text(json.dumps(res, indent=2, default=float))
    figure(res, FIGS / "fig_wout_solve_at_price.png")
    for sh in SHAPES:
        for w in PRICES:
            p = res["shapes"][sh]["prices"][str(w)]
            a = p["arms"]
            print(f"{sh:9s} w={w:g}  solved {a['SOLVE_AT_PRICE']['J_asym']:.5f}  "
                  f"reread {a['REREAD_AT_PRICE']['J_asym']:.5f}  "
                  f"uniform {a['UNIFORM']['J_asym']:.5f}  "
                  f"gain {p['gain_solve_over_reread_pct']:+.2f}%  "
                  f"beats={p['beats_reread']}  "
                  f"R2 vs melt-solved {p['map_family']['r2_vs_melt_solved']:+.3f}  "
                  f"xcheck rel {p['cross_check']['rel_diff']:.2e} "
                  f"idx_match={p['cross_check']['index_match']}", flush=True)
    return res


if __name__ == "__main__":
    main()
