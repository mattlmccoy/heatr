#!/usr/bin/env python3
"""Gate S4 supporting runs: (a) grid sensitivity of the predicted top-face
pattern, (b) figures. Neither changes any scored number.

(a) is the honest guard S2 has not yet supplied: if the predicted pattern moved
under refinement, the S4 pattern score would be a grid artifact, not a claim.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from heatr3d import Grid, Params, build_gamma, compute_qrf_3d, run, solve_eqs_3d  # noqa: E402
import s4_flir_lib as L  # noqa: E402
import s4_run as R  # noqa: E402

logger = logging.getLogger("s4x")


def coarse_prediction(duration_s: float, amb: float, power_w: float) -> np.ndarray:
    """Case-A prediction on a 1.5x coarser grid (h = 0.975 mm, 64 x 21 x 64),
    same physical geometry, same absorbed power."""
    nx = nz = 64
    ny = 21
    g = Grid(n=nx, L=R.CHAMBER_L)
    w, hy = 41, 10
    i0 = (nx - w) // 2
    part = np.zeros((nx, ny, nz), bool)
    part[i0:i0 + w, 0:hy, i0:i0 + w] = True
    p0 = Params()
    gam = build_gamma(part, p0)
    V = solve_eqs_3d(gam, g, p0)
    Q = compute_qrf_3d(V, gam, g, p0, part)
    Q *= power_w / (Q.sum() * g.dV)
    p = dataclasses.replace(p0, phase_update="enthalpy", t_pc_c=185.0, dt_s=0.3,
                            ambient_c=amb, preheat_c=amb)
    pr = R.march(g, part, p, Q, duration_s, R.N_CHECKPOINTS)
    return pr


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    res = json.loads((HERE / "results.json").read_text())
    fields = np.load(HERE / "fields.npz")
    ca = res["cases"]["A"]
    amb = ca["measured"]["e095"]["ambient_c"]
    dur = ca["measured"]["e095"]["duration_s"]
    power = res["gates"]["A"]["power_w"]

    pr = coarse_prediction(dur, amb, power)
    theta = L.normalized_rise_curve(pr["t_s"], pr["face_mean"], amb)
    imgs = pr["planeP"]
    gmax = imgs.max(axis=(1, 2)) - amb
    ref = int(np.argmax(gmax >= 0.6 * gmax.max()))
    rect = L.min_area_rect(L.largest_component_roi(imgs[ref], amb, frac=0.5))
    out = {"coarse_grid": dict(h_mm=0.975, n_xz=64, n_y=21, dt_s=0.3,
                               power_w=power,
                               energy_residual_frac=pr["energy_residual_frac"],
                               clamp_bound=pr["clamp_bound"],
                               cfl_violated=pr["cfl_violated"],
                               n_substeps_used=pr["n_substeps_used"],
                               face_max_end_c=float(pr["face_max"][-1]),
                               part_max_end_c=float(pr["part_max"][-1]))}
    for fr in R.MATCH_FRACS:
        j = int(np.argmin(np.abs(pr["t_s"] - L.time_at_fraction(pr["t_s"], theta, fr))))
        coarse = L.resample_unit_square(imgs[j], rect, ng=R.NG)
        fine = fields[f"pred_A_planeP_{fr:.2f}"]
        meas = fields[f"meas_A_{fr:.2f}"]
        out["coarse_grid"][f"r_coarse_vs_fine_{fr:.2f}"] = L.pattern_correlation(
            L.normalize_rise(coarse, amb), L.normalize_rise(fine, amb))
        out["coarse_grid"][f"r_coarse_vs_measured_{fr:.2f}"] = L.pattern_correlation(
            L.normalize_rise(coarse, amb), L.normalize_rise(meas, amb))
        out["coarse_grid"][f"chamfer_coarse_vs_fine_{fr:.2f}"] = L.chamfer_mm(fine, coarse)
        out["coarse_grid"][f"cX_coarse_{fr:.2f}"] = L.corner_edge_contrast(coarse)
    (HERE / "grid_sensitivity.json").write_text(json.dumps(out, indent=1, default=float))
    logger.info("grid sensitivity: %s", json.dumps(out["coarse_grid"], indent=1, default=float))

    make_figures(res, fields)


def make_figures(res: dict, fields) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = HERE / "figs"
    figs.mkdir(exist_ok=True)
    for k in ("A", "B", "C"):
        fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.2))
        for row, fr in enumerate(R.MATCH_FRACS):
            m = fields[f"meas_{k}_{fr:.2f}"]
            p = fields[f"pred_{k}_planeP_{fr:.2f}"]
            s = res["cases"][k]["scores"][f"planeP_{fr:.2f}"]
            for ax, img, ttl in ((axes[row, 0], m, "measured (FLIR)"),
                                 (axes[row, 1], p, "heatr3d plane P")):
                im = ax.imshow(img, cmap="inferno", origin="lower",
                               extent=[-18, 18, -18, 18])
                ax.set_title(f"{ttl}\nstate {fr:.0%}  max {img.max():.0f} C", fontsize=8)
                ax.set_xlabel("mm"); ax.tick_params(labelsize=7)
                fig.colorbar(im, ax=ax, fraction=0.046)
            ax = axes[row, 2]
            a = L.normalize_rise(m, res["cases"][k]["measured"]["e095"]["ambient_c"])
            b = L.normalize_rise(p, res["cases"][k]["predicted"]["face_mean"][0] * 0
                                 + res["cases"][k]["measured"]["e095"]["ambient_c"])
            im = ax.imshow(a - b, cmap="coolwarm", origin="lower", extent=[-18, 18, -18, 18],
                           vmin=-0.6, vmax=0.6)
            ax.set_title(f"normalized difference\nr={s['r']:+.3f}  chamfer={s['chamfer_mm']:.1f} mm",
                         fontsize=8)
            ax.set_xlabel("mm"); ax.tick_params(labelsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046)
        c = res["cases"][k]["case"]
        fig.suptitle(f"S4 case {k}: {c['seq']} ({c['cls']}) — part-relative registered fields",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(figs / f"case_{k}_fields.png", dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, k in zip(axes, ("A", "B", "C")):
        m = res["cases"][k]["measured"]["e095"]
        p = res["cases"][k]["predicted"]
        ax.plot(np.array(m["t_s"]) / 60, m["roi_mean_c"], "k-", lw=1.2, label="FLIR ROI mean")
        ax.plot(np.array(m["t_s"]) / 60, m["roi_max_c"], "k:", lw=1.0, label="FLIR ROI max")
        m85 = res["cases"][k]["measured"]["e085"]
        ax.fill_between(np.array(m["t_s"]) / 60, m["roi_mean_c"], m85["roi_mean_c"],
                        color="0.7", alpha=0.5, label="eps 0.85-0.95 band")
        ax.plot(np.array(p["t_s"]) / 60, p["face_mean"], "r-", lw=1.2, label="heatr3d face mean")
        ax.plot(np.array(p["t_s"]) / 60, p["face_max"], "r:", lw=1.0, label="heatr3d face max")
        ax.plot(np.array(p["t_s"]) / 60, p["part_max"], "b--", lw=1.0, label="heatr3d part max")
        ax.axhline(185, color="g", ls=":", lw=0.8)
        ax.set_title(f"case {k}: {res['cases'][k]['case']['seq']}\n"
                     f"fitted {res['gates'][k]['power_w']:.1f} W absorbed", fontsize=9)
        ax.set_xlabel("min"); ax.set_ylabel("T [C]"); ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figs / "curves.png", dpi=140)
    plt.close(fig)
    logger.info("figures in %s", figs)


if __name__ == "__main__":
    main()
