#!/usr/bin/env python3
"""Collect the circle_PROPORTIONAL_m0* sweep and render the dissertation figure.

Reads (per run, per iteration):
  outputs_eqs/runs/circle/fgm_iterate/circle_PROPORTIONAL_m0*_n30/
      <name>_iterK/time_series.json   -> melt-onset sigma_T (dual read-state)
      <name>_iterK/fgm_*.npz          -> saturation map GENERATED FROM iter K
      <name>_iterK/fields.npz         -> part_mask (iter0 only, shared grid)
      convergence.json                -> iteration order + metadata

Map indexing convention: the npz written in iter-K's directory is the map
generated from iter-K's field and APPLIED in iter-(K+1).  We analyze the
applied-map sequence s_1..s_29 where s_{k} = npz from iter-(k-1).

Writes:
  prop_sweep_data.json   (all trajectories + lagged stats + verdicts)
  fig_proportional_sweep.png

Run:
  ./.venv312/bin/python fgm_solve_campaign/figs_prop_sweep/make_fig_proportional_sweep.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from rfam_eqs_coupled import dual_read_state_from_hist  # noqa: E402
from prop_sweep_analysis import lagged_stats, classify_dynamics  # noqa: E402

logger = logging.getLogger(__name__)

RUNS_DIR = REPO / "outputs_eqs" / "runs" / "circle" / "fgm_iterate"
SWEEP = [
    ("circle_PROPORTIONAL_m03_n30", 0.3),
    ("circle_PROPORTIONAL_m05_n30", 0.5),
    ("circle_PROPORTIONAL_m007_n30", 0.7),
    ("circle_PROPORTIONAL_m008_n30", 0.8),
    ("circle_PROPORTIONAL_m009_n30", 0.9),
]
PLATEAU_START = 10          # iterates >= this index are "plateau"
BAND_LO, BAND_HI = 14.9, 16.5   # the dissertation-quoted plateau band (C)


def melt_onset_sigma_t(iter_dir: Path) -> float | None:
    """Melt-onset sigma_T from time_series.json (never end-of-horizon)."""
    ts = iter_dir / "time_series.json"
    if not ts.exists():
        return None
    hist = json.loads(ts.read_text())
    dual = dual_read_state_from_hist(hist)
    return dual.get("sigma_T_melt_onset_c")


def load_run(name: str) -> Dict[str, Any] | None:
    parent = RUNS_DIR / name
    conv = parent / "convergence.json"
    if not conv.exists():
        logger.warning("missing %s", conv)
        return None
    meta = json.loads(conv.read_text())
    n = len(meta.get("iterations", []))
    sigma: List[float | None] = []
    maps: List[np.ndarray | None] = []
    mask: np.ndarray | None = None
    for k in range(n):
        it_dir = parent / f"{name}_iter{k}"
        sigma.append(melt_onset_sigma_t(it_dir))
        if mask is None and (it_dir / "fields.npz").exists():
            mask = np.load(it_dir / "fields.npz")["part_mask"].astype(bool)
        npzs = sorted(it_dir.glob("fgm_*_mag*.npz"))
        if npzs:
            maps.append(np.load(npzs[0])["sat_map"].astype(np.float64))
        else:
            maps.append(None)
    return {"name": name, "meta": meta, "sigma": sigma, "maps": maps, "mask": mask}


def analyze_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Lagged stats + verdict on the plateau slice of the APPLIED-map sequence."""
    mask = run["mask"]
    # applied map s_k (k>=1) = npz generated in iter k-1
    applied = [m for m in run["maps"][:-1] if m is not None]
    applied_masked = [m[mask] if m.shape == mask.shape else m.ravel() for m in applied]
    plateau = applied_masked[PLATEAU_START:]
    stats = lagged_stats(plateau)
    verdict = classify_dynamics(stats)
    # Settled-tail sensitivity window: last 8 applied maps.  For a damped
    # oscillation (m=0.9) the plateau window mixes transient and settled
    # phases; the tail shows what the iteration ultimately does.
    tail_stats = lagged_stats(applied_masked[-8:])
    tail_verdict = classify_dynamics(tail_stats)
    sig = [s for s in run["sigma"] if s is not None]
    plateau_sig = [s for s in run["sigma"][PLATEAU_START:] if s is not None]
    tail_sig = [s for s in run["sigma"][-8:] if s is not None]
    return {
        "verdict": verdict,
        "tail_verdict": tail_verdict,
        "tail_corr_lag1_mean": tail_stats["corr_lag1_mean"],
        "tail_corr_lag2_mean": tail_stats["corr_lag2_mean"],
        "tail_step_norm_mean": tail_stats["step_norm_mean"],
        "tail_sigma_mean": float(np.mean(tail_sig)) if tail_sig else None,
        "tail_sigma_std": float(np.std(tail_sig)) if tail_sig else None,
        "corr_lag1_mean": stats["corr_lag1_mean"],
        "corr_lag2_mean": stats["corr_lag2_mean"],
        "step_norm_mean": stats["step_norm_mean"],
        "step_norm_trend": stats["step_norm_trend"],
        "corr_lag1": stats["corr_lag1"],
        "corr_lag2": stats["corr_lag2"],
        "step_norms_plateau": stats["step_norms"],
        "plateau_sigma_mean": float(np.mean(plateau_sig)) if plateau_sig else None,
        "plateau_sigma_std": float(np.std(plateau_sig)) if plateau_sig else None,
        "iter1_is_min": (len(sig) > 1 and sig[1] == min(sig[1:])),
        "min_sigma": float(min(sig[1:])) if len(sig) > 1 else None,
        "min_sigma_iter": int(np.argmin(sig[1:]) + 1) if len(sig) > 1 else None,
        "iter0_sigma": sig[0] if sig else None,
        "iter1_sigma": sig[1] if len(sig) > 1 else None,
        "in_band": (plateau_sig and BAND_LO <= float(np.mean(plateau_sig)) <= BAND_HI),
    }


def full_step_norms(run: Dict[str, Any]) -> List[float]:
    """RMS step ||s_{k+1}-s_k|| over the whole applied sequence (for the figure)."""
    mask = run["mask"]
    applied = [m[mask] for m in run["maps"][:-1] if m is not None]
    return [float(np.sqrt(np.mean((b - a) ** 2))) for a, b in zip(applied, applied[1:])]


def render(runs: List[Dict[str, Any]], analyses: List[Dict[str, Any]],
           out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
        "text.color": "#222222", "xtick.color": "#444444",
        "ytick.color": "#444444", "font.size": 10.5,
        "axes.titlesize": 11.5, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = ["#4477aa", "#66ccee", "#228833", "#ccbb44", "#ee6677"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    # (a) sigma_T trajectories
    ax = axes[0]
    ax.axhspan(BAND_LO, BAND_HI, color="#dddddd", alpha=0.6, zorder=0,
               label=f"quoted band {BAND_LO}-{BAND_HI} C")
    for run, c in zip(runs, colors):
        m = run["meta"]["magnitude_base"]
        sig = run["sigma"]
        ax.plot(range(len(sig)), sig, "-o", ms=3.5, lw=1.4, color=c,
                label=f"m = {m:g}")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\sigma_T$ at melt onset (C)")
    ax.set_title("(a) proportional trajectories, constant m\n"
                 "(circle 20 mm, 4 bpp, enforced 500 W)")
    ax.annotate("m = 0.9: period-2 damps, settles at ~15.9 C",
                xy=(26.5, 16.4), xytext=(9.5, 17.3), fontsize=8.5,
                color="#c94f5f",
                arrowprops=dict(arrowstyle="->", color="#c94f5f", lw=1))
    ax.set_ylim(5.5, 21.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2,
              columnspacing=0.9, handlelength=1.6)

    # (b) step norms
    ax = axes[1]
    for run, c in zip(runs, colors):
        m = run["meta"]["magnitude_base"]
        steps = full_step_norms(run)
        ax.plot(range(1, len(steps) + 1), steps, "-o", ms=3, lw=1.3,
                color=c, label=f"m = {m:g}")
    ax.axvline(PLATEAU_START, color="#999999", ls=":", lw=1)
    ax.text(PLATEAU_START + 0.3, ax.get_ylim()[1] * 0.02, "plateau window",
            fontsize=8, color="#777777")
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"RMS$(s_{k+1}-s_k)$ inside part")
    ax.set_title("(b) map step size per iteration")
    ax.legend(fontsize=8.5, frameon=False)

    # (c) lag-1 vs lag-2 correlation on the plateau
    ax = axes[2]
    ms = [run["meta"]["magnitude_base"] for run in runs]
    w = 0.35
    xs = np.arange(len(ms))
    c1 = [a["corr_lag1_mean"] for a in analyses]
    c2 = [a["corr_lag2_mean"] for a in analyses]
    ax.bar(xs - w / 2, c1, w, color="#4477aa", label=r"corr$(s_k, s_{k+1})$")
    ax.bar(xs + w / 2, c2, w, color="#ee6677", label=r"corr$(s_k, s_{k+2})$")
    for x, a in zip(xs, analyses):
        ax.text(x, max(a["corr_lag1_mean"], a["corr_lag2_mean"]) + 0.015,
                a["verdict"], ha="center", fontsize=8, rotation=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{m:g}" for m in ms])
    ax.set_xlabel("magnitude m")
    ax.set_ylabel("plateau map correlation")
    ax.set_ylim(0.5, 1.12)
    ax.set_title("(c) plateau dynamics: lag-1 vs lag-2")
    ax.legend(fontsize=8.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2)

    fig.suptitle("Constant-magnitude proportional FGM sweep (circle, melt-onset read, "
                 "enforced-power drive)", fontsize=12, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    logger.info("wrote %s", out_png)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs, analyses = [], []
    for name, m in SWEEP:
        run = load_run(name)
        if run is None:
            logger.warning("skipping %s (not found)", name)
            continue
        runs.append(run)
        analyses.append(analyze_run(run))
    payload = {
        "plateau_start_iter": PLATEAU_START,
        "band_c": [BAND_LO, BAND_HI],
        "read": "sigma_T melt-onset (ui_rms_part x (mean_T_part - 23 C) at first "
                "mean_phi >= 0.90 crossing, from time_series.json)",
        "drive": "enforced generator power 500 W @ 2% transfer, 27.12 MHz, "
                 "grounded 860 V nominal, 20 mm circle, grid 120, t*=6.308 min",
        "runs": [
            {"name": r["name"], "m": r["meta"]["magnitude_base"],
             "sigma_T_melt_onset": r["sigma"], **a}
            for r, a in zip(runs, analyses)
        ],
    }
    out_json = HERE / "prop_sweep_data.json"
    out_json.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", out_json)
    for r in payload["runs"]:
        print(f"m={r['m']:g}  verdict={r['verdict']}  "
              f"lag1={r['corr_lag1_mean']:.3f} lag2={r['corr_lag2_mean']:.3f}  "
              f"plateau sigma_T={r['plateau_sigma_mean']}  "
              f"min sigma_T={r['min_sigma']} @ iter{r['min_sigma_iter']}  "
              f"iter1_is_min={r['iter1_is_min']}")
    render(runs, analyses, HERE / "fig_proportional_sweep.png")


if __name__ == "__main__":
    main()
