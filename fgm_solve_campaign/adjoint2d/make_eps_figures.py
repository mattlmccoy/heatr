"""Figures for the permittivity-channel pass. Each one says a single thing.

  fig_eps_census    the de-confounded census: solved arms against the best
                    historical mask now that both share an actuator
  fig_eps_channel   what the permittivity channel bought, at matched budget,
                    and the finite-difference gate that licenses it
  fig_eps_maps      the 18 delivered 4-bits-per-pixel dopant maps
  fig_eps_robust    rim and grid probes on the new square and cross maps
  fig_eps_<shape>   per-shape mover panels: dopant map, melted region against
                    the nominal part, for the new arm and the historical mask

Run:  ./.venv312/bin/python -m adjoint2d.make_eps_figures [shape ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_EPS = ROOT / "out_eps"
OUT_LIB = ROOT / "out_lib"
OUT_MS = ROOT / "out_ms"
FIGS = ROOT / "figs_eps"
DPI = 180
SHAPES = ("square", "circle", "hexagon", "triangle", "equilateral_triangle",
          "L_shape", "H_shape", "T_shape", "cross", "diamond", "ellipse",
          "octagon", "pentagon", "rectangle", "rounded_rect", "star", "star6",
          "trapezoid")
C = {"eps_best": "#C44E52", "eps_cold": "#DD8452", "hist": "#4C72B0",
     "ms": "#55A868", "unif": "#8C8C8C"}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def collect() -> list[dict]:
    rows = []
    for sh in SHAPES:
        e = _load(OUT_EPS / f"{sh}.json")
        if e is None:
            continue
        lib = _load(OUT_LIB / f"{sh}.json") or {"arms": {}}
        ms = _load(OUT_MS / f"{sh}.json") or {"arms": {}}
        ctl = _load(OUT_MS / f"{sh}_control_cold.json") or {"arms": {}}
        rows.append({"shape": sh, "e": e,
                     "eps_best": e["arms"]["EPS_best_4bpp"],
                     "eps_cold": e["arms"].get("EPS_cold_4bpp"),
                     "unif": e["arms"]["U_uniform"],
                     "hist": lib["arms"].get("HIST_best"),
                     "ms": ms["arms"].get("MS_4bpp"),
                     "ctl": ctl["arms"].get("MS_4bpp")})
    return rows


# ---------------------------------------------------------------------------

def fig_census(rows) -> Path:
    labels = [r["shape"] for r in rows]
    x = np.arange(len(labels))
    w = 0.20
    fig, axes = plt.subplots(3, 1, figsize=(16.0, 13.5))

    ax = axes[0]
    series = [("unif", "uniform dopant"), ("hist", "best historical mask (oracle)"),
              ("ms", "solved, conductivity only (out_ms)"),
              ("eps_best", "solved, permittivity channel (this pass)")]
    for i, (k, lab) in enumerate(series):
        vals = [np.nan if r[k] is None else float(r[k]["J"]) for r in rows]
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab,
               color=C.get(k, "#333333"), edgecolor="none")
    ax.set_yscale("log")
    ax.set_ylabel("J_phi at each arm's own J-stop (log scale)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=9, ncol=2)
    ax.set_title("A. The de-confounded census. Every arm reads J_phi = sum over the whole "
                 "domain of (phi - chi_part)^2 at its own argmin stop. The historical mask "
                 "and the new solved arm now share an actuator.", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for i, (k, lab) in enumerate(series):
        vals = [np.nan if r[k] is None else float(r[k]["IoU"]) for r in rows]
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab,
               color=C.get(k, "#333333"), edgecolor="none")
    ax.axhline(0.95, color="k", ls="--", lw=1.0)
    ax.text(len(labels) - 0.4, 0.955, "IoU 0.95", fontsize=8, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("intersection over union at grid 120")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_title("B. Intersection over union of the melted region (phi >= 0.5) with the "
                 "nominal part. GRID 120 QUALIFIER: this does not transfer to grid 160.",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    d_ms, d_eps = [], []
    for r in rows:
        h = None if r["hist"] is None else float(r["hist"]["J"])
        d_ms.append(np.nan if (h is None or r["ms"] is None)
                    else 100.0 * (h - float(r["ms"]["J"])) / abs(h))
        d_eps.append(np.nan if h is None
                     else 100.0 * (h - float(r["eps_best"]["J"])) / abs(h))
    ax.bar(x - 0.20, d_ms, 0.38, label="conductivity only, against the historical mask",
           color=C["ms"])
    ax.bar(x + 0.20, d_eps, 0.38, label="permittivity channel, against the historical mask",
           color=C["eps_best"])
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_ylabel("percent of J_phi removed against the historical mask\n"
                  "(positive = the solve wins)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    lo = -260
    ax.set_ylim(lo, 118)
    # Bars that run off the bottom are labelled with their true value rather
    # than silently clipped.
    for xi, (a, b) in enumerate(zip(d_ms, d_eps)):
        for off, v in ((-0.20, a), (0.20, b)):
            if np.isfinite(v) and v < lo:
                ax.annotate(f"{v:.0f}", (xi + off, lo + 8), ha="center", va="bottom",
                            fontsize=7.5, rotation=90, color="#222222")
    ax.legend(fontsize=9, loc="lower center")
    ax.set_title("C. The confound removed. Bars below zero are shapes where the stored "
                 "historical mask still wins. NOT power matched.", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    p = FIGS / "fig_eps_census.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p


def fig_channel(rows) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    ax = axes[0]
    pairs = [(r["shape"],
              100.0 * (float(r["ctl"]["J"]) - float(r["eps_cold"]["J"]))
              / max(abs(float(r["ctl"]["J"])), 1e-30))
             for r in rows if r["ctl"] is not None and r["eps_cold"] is not None]
    pairs.sort(key=lambda t: t[1])
    ax.barh([p[0] for p in pairs], [p[1] for p in pairs],
            color=[C["eps_best"] if p[1] > 0 else C["ms"] for p in pairs])
    ax.axvline(0.0, color="k", lw=1.0)
    ax.set_xlabel("percent of J_phi removed by switching the actuator\n"
                  "(same 40 forward-equivalents, same filter, same cold start)")
    ax.set_title("A. Permittivity channel against conductivity only,\nbudget matched",
                 fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    for sh, col in (("square", C["eps_best"]), ("cross", C["hist"])):
        g = _load(OUT_EPS / f"gate_eps_{sh}.json")
        if g is not None:
            for lay in g["layers"]:
                if not lay["eps_covary"]:
                    continue
                pr = lay["probes"]["gradient_direction"]
                e = [row["eps"] for row in pr["sweep"]]
                r = [row["rel_err"] for row in pr["sweep"]]
                tag = lay["layer"].split("_")[0]
                ax.loglog(e, r, "--" if tag == "E1" else "-", marker="o", ms=3.5,
                          color=col, alpha=0.55,
                          label=f"{sh} {tag}, coarse sweep")
        b = _load(OUT_EPS / f"gate_eps_bisect_{sh}.json")
        if b is not None:
            pr = b["refined_sweep_E2"]["probes"]["gradient_direction"]
            e = [row["eps"] for row in pr["sweep"]]
            r = [row["rel_err"] for row in pr["sweep"]]
            ax.loglog(e, r, "-", marker="s", ms=5.0, lw=2.0, color=col,
                      label=f"{sh} E2, refined sweep in the usable window")
    ax.axhline(1e-5, color="k", ls=":", lw=1.0)
    ax.axhline(1e-6, color="k", ls="--", lw=1.0)
    ax.text(1.2e-8, 1.3e-5, "1e-5 subgradient standard", fontsize=7.5)
    ax.text(1.2e-8, 1.3e-6, "1e-6 clean-smooth standard", fontsize=7.5)
    ax.set_xlabel("central-difference epsilon")
    ax.set_ylabel("relative error against the analytic gradient")
    ax.set_title("B. The finite-difference gate, permittivity channel,\n"
                 "gradient-direction probe. E1 has no design filter, E2 has it.",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    names, fracs = [], []
    for sh in ("square", "cross"):
        g = _load(OUT_EPS / f"gate_eps_{sh}.json")
        if g is None:
            continue
        for lay in g["layers"]:
            if "channel_split" in lay:
                names.append(f"{sh}\n{lay['layer'].split('_')[0]}")
                fracs.append(lay["channel_split"]["eps_term_relative_norm"])
    ax.bar(names, fracs, color=C["eps_best"])
    for i, v in enumerate(fracs):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("||dJ/ds with permittivity minus without|| / ||dJ/ds without||")
    ax.set_title("C. The size of the new term in the gradient\n"
                 "(a channel worth adding is a channel that moves it)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    p = FIGS / "fig_eps_channel.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p


def fig_maps(rows) -> Path:
    fig, axes = plt.subplots(3, 6, figsize=(16.0, 8.6))
    im = None
    for ax, r in zip(axes.ravel(), rows):
        with np.load(OUT_EPS / f"{r['shape']}_maps.npz") as z:
            s = np.asarray(z["EPS_best_4bpp"], dtype=float)
            pm = np.asarray(z["part_mask"], dtype=bool)
        im = ax.imshow(np.where(pm, s, np.nan), origin="lower", cmap="viridis",
                       vmin=0.0, vmax=1.0, interpolation="bilinear")
        a = r["eps_best"]
        ax.set_title(f"{r['shape']}  ({r['e']['winner_start']})\n"
                     f"J {a['J']:.1f}  IoU {a['IoU']:.3f}  "
                     f"{int(a['census_n_levels_used'])} levels", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.015,
                 label="binder saturation, 4 bits per pixel")
    fig.suptitle("The delivered permittivity-channel dopant maps, quantized to the "
                 "printer's 16 levels inside the part and held at the nominal 1 outside. "
                 "Winning start in brackets. Grid 120 x 120.", fontsize=11)
    p = FIGS / "fig_eps_maps.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# per-shape mover panels; these RE-RUN the forward to get the melted region
# ---------------------------------------------------------------------------

def fig_shape(shape: str) -> Path | None:
    from . import eps_solve as es
    from . import library_solve as lib
    from . import shape_objective as so
    from .pins import build_case, load_cfg
    from .verify_hist import load_stored_map

    e = _load(OUT_EPS / f"{shape}.json")
    if e is None:
        return None
    libj = _load(OUT_LIB / f"{shape}.json") or {"arms": {}}
    cfg = load_cfg(lib.shape_config(shape))
    case = build_case(cfg)
    pm = case.part_mask

    with np.load(OUT_EPS / f"{shape}_maps.npz") as z:
        s_eps = np.asarray(z["EPS_best_4bpp"], dtype=float)
    panels = [("permittivity-channel solve, 4 bits per pixel", s_eps, True,
               e["arms"]["EPS_best_4bpp"])]

    h = libj["arms"].get("HIST_best")
    if h and h.get("map_npz"):
        s_h = load_stored_map(case, Path(h["map_npz"]), cfg)
        if h.get("convention") == "outside1":
            s_h = np.where(pm, s_h, 1.0)
        panels.append(("best stored historical mask", s_h, True, h))

    ms_npz = OUT_MS / f"{shape}_maps.npz"
    if ms_npz.exists():
        with np.load(ms_npz) as z:
            s_ms = np.asarray(z["MS_4bpp"], dtype=float)
        msj = _load(OUT_MS / f"{shape}.json")
        panels.append(("conductivity-only solve, 4 bits per pixel", s_ms, False,
                       msj["arms"]["MS_4bpp"]))

    fig, axes = plt.subplots(2, len(panels), figsize=(4.6 * len(panels), 8.6))
    if len(panels) == 1:
        axes = axes.reshape(2, 1)
    for j, (title, s, covary, met) in enumerate(panels):
        tr = es.run_forward(case, s, eps_covary=covary)
        st = so.optimal_stop(tr, case)
        T = tr.T_at_end(st.index)
        phi = so.phi_field(T, case)[0]
        del tr
        ax = axes[0, j]
        ax.imshow(np.where(pm, s, np.nan), origin="lower", cmap="viridis",
                  vmin=0.0, vmax=1.0, interpolation="bilinear")
        ax.contour(pm.astype(float), levels=[0.5], colors="w", linewidths=0.9)
        ax.set_title(f"{title}\n"
                     f"{'permittivity co-varying' if covary else 'conductivity only'}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1, j]
        melted = (phi >= 0.5)
        img = np.ones(pm.shape + (3,))                          # bed, white
        img[melted & ~pm] = (0.86, 0.20, 0.18)                  # growth, red
        img[pm & ~melted] = (0.20, 0.35, 0.80)                  # under-melt, blue
        img[melted & pm] = (0.62, 0.62, 0.62)                   # correct, grey
        ax.imshow(img, origin="lower", interpolation="nearest")
        ax.contour(pm.astype(float), levels=[0.5], colors="k", linewidths=0.9)
        ax.set_title(f"J_phi {float(met['J']):.1f}   IoU {float(met['IoU']):.4f}\n"
                     f"grey = correct, red = growth into the bed, blue = unmelted part\n"
                     f"stop {float(met['t_stop_s']):.1f} s"
                     f"{' AT HORIZON' if met.get('t_stop_at_horizon') else ''}",
                     fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{shape}: the melted region at each arm's own J-stop, grid 120 x 120. "
                 f"Melted region is phi >= 0.5. Arms are NOT power matched.", fontsize=11)
    fig.tight_layout()
    p = FIGS / f"fig_eps_{shape}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p


def fig_robust(shapes=("square", "cross")) -> Path | None:
    have = [s for s in shapes if (OUT_EPS / f"{s}_robust.json").exists()]
    if not have:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # A. rim, as PERCENT of J_phi added by the blur, so the two shapes and the
    # previously measured conductivity-only arm sit on one axis.
    ax = axes[0]
    xs = np.arange(len(have))
    prev = {}
    for s in have:
        p = OUT_MS / f"{s}_robust.json"
        if p.exists():
            prev[s] = json.loads(p.read_text()).get("rim_verdict", {})
    series = [("r1.0", "permittivity map, blur 1 cell", C["eps_best"]),
              ("r2.0", "permittivity map, blur 2 cells", "#8C3F41")]
    for i, (k, lab, col) in enumerate(series):
        vals = []
        for s in have:
            j = json.loads((OUT_EPS / f"{s}_robust.json").read_text())
            vals.append(100.0 * j["rim_verdict"][f"dJ_rel_{k}"])
        ax.bar(xs + (i - 1) * 0.22, vals, 0.20, label=lab, color=col)
        for xi, v in zip(xs, vals):
            ax.annotate(f"{v:+.1f}%", (xi + (i - 1) * 0.22, v + 1.0), ha="center",
                        fontsize=8)
    pv = [100.0 * prev.get(s, {}).get("dJ_rel_r1.0", np.nan) for s in have]
    ax.bar(xs + 0.22, pv, 0.20, label="conductivity-only map, blur 1 cell (out_ms)",
           color=C["ms"])
    for xi, v in zip(xs, pv):
        if np.isfinite(v):
            ax.annotate(f"{v:+.1f}%", (xi + 0.22, v + 1.0), ha="center", fontsize=8)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xticks(xs); ax.set_xticklabels(have)
    ax.set_ylabel("percent of J_phi ADDED by the blur (lower is more robust)")
    ax.legend(fontsize=8.5)
    ax.set_title("A. Rim robustness at grid 120: part-masked Gaussian blur of the\n"
                 "solved continuous map, re-quantized and re-run", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    keys = [("EPS_4bpp_160", "solved permittivity\nmap at 160"),
            ("U_uniform_160", "uniform at 160"),
            ("EPS_4bpp_160_recal", "solved map at 160,\ndose matched")]
    xs2 = np.arange(len(keys))
    for i, s in enumerate(have):
        j = json.loads((OUT_EPS / f"{s}_robust.json").read_text())
        vals = [j["grid"][k]["IoU"] if k in j["grid"] else np.nan for k, _ in keys]
        ax.bar(xs2 + (i - 0.5) * 0.36, vals, 0.33, label=s)
        for xi, v in zip(xs2, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.3f}", (xi + (i - 0.5) * 0.36, v + 0.01),
                            ha="center", fontsize=8)
    for xi, (k, _lab) in enumerate(keys):
        if all(k not in json.loads((OUT_EPS / f"{s}_robust.json").read_text())["grid"]
               for s in have[1:]):
            pass
    ax.set_xticks(xs2)
    ax.set_xticklabels([lab for _k, lab in keys], fontsize=8.5)
    ax.axhline(0.95, color="k", ls="--", lw=1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("intersection over union at GRID 160")
    ax.legend(fontsize=9)
    ax.set_title("B. Grid hold-out: solved at 120, scored at 160. Missing bar means the "
                 "arm was\nnot run: the cross has no stored 160 dose calibration.",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    p = FIGS / "fig_eps_robust.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p


def main(argv) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = collect()
    if not rows:
        print("no out_eps results yet")
        return
    for f in (fig_census(rows), fig_channel(rows), fig_maps(rows)):
        print("wrote", f)
    r = fig_robust()
    if r:
        print("wrote", r)
    for sh in argv:
        f = fig_shape(sh)
        print("wrote", f)


if __name__ == "__main__":
    main(sys.argv[1:])
