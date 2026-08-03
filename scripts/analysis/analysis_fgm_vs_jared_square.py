"""Compare Jared Allison's converged tuned-sigma map (COMSOL, square, z=5 mm)
against HEATR's FGM saturation map for the square.

Both maps are resampled onto a common unit-square part coordinate system
(u, v in [-1, 1]) and min-max normalized to [0, 1] so that the 40 mm (Jared)
and 20 mm (HEATR) parts can be compared shape-wise.

Outputs:
  - printed statistics (Pearson r, edge/center/corner structure, radial profile)
  - outputs_eqs/jared_ir_exp1/fgm_map_comparison.png

Verification gate: run against real artifacts and inspect printed stats/figure
(analysis script; no unit-testable pure logic beyond numpy calls on real data).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import zoom

REPO = Path(__file__).resolve().parents[2]  # repo root (moved to scripts/analysis/)
JARED_TXT = Path(
    "/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/JaredFiles/COMSOL Files/"
    "Powder Dispensing Machine/Square/Results/Z(0.005)_sigma.txt"
)
# Best-scoring applied FGM map: generated at iter0, applied to iter1 (best_iter=1
# per convergence.json of square_INTEGRAL_m007_n12_v2).
HEATR_NPZ = (
    REPO
    / "outputs_eqs/runs/square/fgm_iterate/square_INTEGRAL_m007_n12_v2/"
    "square_INTEGRAL_m007_n12_v2_iter0/"
    "fgm_square_INTEGRAL_m007_n12_v2_iter0_T_phi90_4bpp_mag0p70.npz"
)
OUT_DIR = REPO / "outputs_eqs/jared_ir_exp1"
N = 81  # common grid points per axis over u,v in [-1, 1]

JARED_HALF_M = 0.020  # 40 mm part full width -> half 20 mm
HEATR_HALF_MM = 10.0  # 20 mm part -> half 10 mm


def load_jared_grid() -> np.ndarray:
    """Load full-plane point cloud and grid onto (N, N) over u,v in [-1,1]."""
    d = np.loadtxt(JARED_TXT)
    x, y, _z, s = d.T
    u, v = x / JARED_HALF_M, y / JARED_HALF_M
    uu, vv = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    g = griddata((u, v), s, (uu, vv), method="linear")
    # fill any NaN at the extreme rim with nearest
    nn = griddata((u, v), s, (uu, vv), method="nearest")
    g = np.where(np.isfinite(g), g, nn)
    return g


def load_heatr_grid() -> tuple[np.ndarray, dict]:
    d = np.load(HEATR_NPZ)
    sat = d["sat_map"].astype(float)  # (120,120) full chamber
    x_mm, y_mm = d["x_mm"].astype(float), d["y_mm"].astype(float)
    meta = {k: d[k].item() for k in ("bpp", "magnitude", "baseline_saturation", "dead_band")}
    ix = np.where(np.abs(x_mm) <= HEATR_HALF_MM + 1e-9)[0]
    iy = np.where(np.abs(y_mm) <= HEATR_HALF_MM + 1e-9)[0]
    sub = sat[np.ix_(iy, ix)]
    zf = (N / sub.shape[0], N / sub.shape[1])
    g = zoom(sub, zf, order=1)
    return g[:N, :N], meta


def norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def structure_stats(g: np.ndarray, name: str) -> dict:
    uu, vv = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    r_inf = np.maximum(np.abs(uu), np.abs(vv))  # Chebyshev radius (square shells)
    center = g[r_inf < 0.25].mean()
    mid = g[(r_inf >= 0.25) & (r_inf < 0.65)].mean()
    edge = g[r_inf >= 0.85].mean()
    corner = g[(np.abs(uu) > 0.85) & (np.abs(vv) > 0.85)].mean()
    st = {
        "name": name,
        "center_mean(|r|<0.25)": round(float(center), 4),
        "mid_mean(0.25-0.65)": round(float(mid), 4),
        "edge_mean(|r|>0.85)": round(float(edge), 4),
        "corner_mean(both>0.85)": round(float(corner), 4),
        "edge_minus_center": round(float(edge - center), 4),
    }
    # radial (Chebyshev) profile, 10 shells
    prof = []
    for k in range(10):
        m = (r_inf >= k / 10) & (r_inf < (k + 1) / 10)
        prof.append(round(float(g[m].mean()), 3))
    st["shell_profile_r0_to_1"] = prof
    return st


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jared_raw = load_jared_grid()
    heatr_raw, meta = load_heatr_grid()
    print(f"Jared sigma raw range: [{jared_raw.min():.4g}, {jared_raw.max():.4g}] S/m")
    print(f"HEATR sat raw range:   [{heatr_raw.min():.3f}, {heatr_raw.max():.3f}]  meta={meta}")

    jared_n = norm01(jared_raw)
    heatr_n = norm01(heatr_raw)

    r = float(np.corrcoef(jared_n.ravel(), heatr_n.ravel())[0, 1])
    print(f"\nPearson r (normalized Jared sigma vs HEATR saturation): {r:+.3f}")

    stats = [structure_stats(jared_n, "jared_sigma_norm"), structure_stats(heatr_n, "heatr_sat_norm")]
    for s in stats:
        print(json.dumps(s, indent=2))

    # absorption-peak arithmetic (grounded in config constants)
    f = 27.12e6
    eps_r = 20.0
    sigma_star = 2 * np.pi * f * 8.854e-12 * eps_r
    print(f"\nsigma* = omega*eps0*eps_r = {sigma_star:.4f} S/m "
          f"(Jared clamp 0.0425; HEATR sigma0=0.04)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    ext = [-1, 1, -1, 1]
    im0 = axes[0].imshow(jared_n, origin="lower", extent=ext, cmap="viridis",
                         vmin=0, vmax=1, interpolation="bilinear")
    axes[0].set_title("Jared converged $\\sigma$ (COMSOL, z=5 mm)\nnormalized [0,1], 40 mm part")
    im1 = axes[1].imshow(heatr_n, origin="lower", extent=ext, cmap="viridis",
                         vmin=0, vmax=1, interpolation="bilinear")
    axes[1].set_title("HEATR FGM saturation (iter-0 map,\napplied to best iter-1), 20 mm part")
    for ax in axes[:2]:
        ax.set_xlabel("u (part half-width)")
        ax.set_ylabel("v")
    fig.colorbar(im0, ax=axes[:2], shrink=0.85, label="normalized level")

    sh = np.linspace(0.05, 0.95, 10)
    axes[2].plot(sh, stats[0]["shell_profile_r0_to_1"], "o-", label="Jared $\\sigma$ (norm)")
    axes[2].plot(sh, stats[1]["shell_profile_r0_to_1"], "s-", label="HEATR saturation (norm)")
    axes[2].set_xlabel("Chebyshev radius $r_\\infty$ (0=center, 1=edge)")
    axes[2].set_ylabel("shell-mean normalized level")
    axes[2].set_title(f"Radial structure — Pearson r = {r:+.3f}")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    out = OUT_DIR / "fgm_map_comparison.png"
    fig.savefig(out, dpi=180)
    print(f"\nFigure: {out}")

    with open(OUT_DIR / "fgm_map_comparison_stats.json", "w") as fh:
        json.dump({"pearson_r": r, "sigma_star_S_per_m": sigma_star,
                   "jared_raw_range": [float(jared_raw.min()), float(jared_raw.max())],
                   "heatr_meta": meta, "structure": stats}, fh, indent=2)
    print(f"Stats:  {OUT_DIR / 'fgm_map_comparison_stats.json'}")


if __name__ == "__main__":
    main()
