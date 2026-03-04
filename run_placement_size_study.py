#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
CFG_BASE = ROOT / "configs" / "shape_circle_placement_optimizer.yaml"
OUT_ROOT = ROOT / "outputs_eqs" / "placement_size_study"
TMP_CFG_DIR = ROOT / "configs" / "_tmp_placement_size_study"


def _load_yaml(path: Path) -> dict:
    d = yaml.safe_load(path.read_text())
    if not isinstance(d, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return d


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _score(row: dict) -> float:
    # Same weighting as knee helper in solver.
    return 0.35 * float(row.get("mean_rho", 0.0)) + 0.35 * float(row.get("min_rho", 0.0)) - 0.20 * float(row.get("std_rho", 0.0)) - 0.10 * float(row.get("temp_violation", 0.0))


def _circle_poly(cx: float, cy: float, r: float, n: int = 80) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _plot_version(out_dir: Path, size_mm: float, chamber_x: float, chamber_y: float, top_rows: list[dict]) -> None:
    if not top_rows:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)

    # Panel A: top layouts in chamber.
    ax0 = ax[0]
    hx = chamber_x * 0.5 * 1e3
    hy = chamber_y * 0.5 * 1e3
    ax0.plot([-hx, hx, hx, -hx, -hx], [-hy, -hy, hy, hy, -hy], "k-", lw=1.2, label="Chamber")
    rad_mm = size_mm * 0.5

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(top_rows)))
    for i, row in enumerate(top_rows):
        layout = row.get("layout", [])
        if not isinstance(layout, list):
            continue
        for cx, cy in layout:
            poly = _circle_poly(float(cx) * 1e3, float(cy) * 1e3, rad_mm)
            ax0.plot(poly[:, 0], poly[:, 1], "-", color=colors[i], lw=1.2, alpha=0.85)
        ax0.text(-hx + 1.0, hy - 2.0 - 2.4 * i, f"#{i+1}: mean={float(row.get('mean_rho',0)):.4f}, min={float(row.get('min_rho',0)):.4f}, std={float(row.get('std_rho',0)):.4f}", fontsize=7, color=colors[i])

    ax0.set_title(f"Top layouts (size={size_mm:.1f} mm)")
    ax0.set_xlabel("x [mm]")
    ax0.set_ylabel("y [mm]")
    ax0.set_aspect("equal")
    ax0.grid(alpha=0.2)

    # Panel B: metric bars for top candidates.
    ax1 = ax[1]
    idx = np.arange(len(top_rows))
    mean_vals = [float(r.get("mean_rho", 0.0)) for r in top_rows]
    min_vals = [float(r.get("min_rho", 0.0)) for r in top_rows]
    std_vals = [float(r.get("std_rho", 0.0)) for r in top_rows]
    ax1.bar(idx - 0.2, mean_vals, width=0.2, label="mean rho")
    ax1.bar(idx, min_vals, width=0.2, label="min rho")
    ax1.bar(idx + 0.2, std_vals, width=0.2, label="std rho")
    ax1.set_xticks(idx)
    ax1.set_xticklabels([f"C{i+1}" for i in idx])
    ax1.set_title("Top-candidate metrics")
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=8)

    fig.suptitle(f"Placement Version Report - {size_mm:.1f} mm circles", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "placement_version_report.png", bbox_inches="tight")
    plt.close(fig)


def _plot_cross_size(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: float(r["size_mm"]))
    x = [float(r["size_mm"]) for r in rows]
    mean_rho = [float(r["best_mean_rho"]) for r in rows]
    min_rho = [float(r["best_min_rho"]) for r in rows]
    std_rho = [float(r["best_std_rho"]) for r in rows]
    score = [float(r["best_score"]) for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=180)
    ax[0].plot(x, mean_rho, "o-", lw=1.8, label="best mean rho")
    ax[0].plot(x, min_rho, "s-", lw=1.8, label="best min rho")
    ax[0].plot(x, std_rho, "^-", lw=1.4, label="best std rho")
    ax[0].set_xlabel("Part size [mm]")
    ax[0].set_title("Best layout metrics by size")
    ax[0].grid(alpha=0.25)
    ax[0].legend(fontsize=8)

    ax[1].plot(x, score, "o-", lw=2.0, color="tab:red")
    best_i = int(np.argmax(np.asarray(score)))
    ax[1].scatter([x[best_i]], [score[best_i]], c="gold", s=90, edgecolors="k", zorder=3)
    ax[1].set_xlabel("Part size [mm]")
    ax[1].set_title("Composite score (higher better)")
    ax[1].grid(alpha=0.25)

    fig.suptitle("Placement Size Study Summary", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Run placement optimizer on smaller part sizes and generate figures")
    p.add_argument("--sizes-mm", default="4,6,8", help="Comma-separated circle diameters in mm")
    p.add_argument("--n-candidates", type=int, default=20)
    p.add_argument("--proxy-top-k", type=int, default=6)
    p.add_argument("--n-steps", type=int, default=240, help="Thermal steps for faster study runs")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = p.parse_args()

    sizes = [float(s.strip()) for s in str(args.sizes_mm).split(",") if s.strip()]
    if not sizes:
        raise ValueError("No sizes provided")

    base = _load_yaml(CFG_BASE)
    chamber_x = float(base.get("geometry", {}).get("chamber_x", 0.060))
    chamber_y = float(base.get("geometry", {}).get("chamber_y", 0.060))

    summary_rows: list[dict] = []

    for s_mm in sizes:
        cfg = json.loads(json.dumps(base))
        part = cfg.setdefault("geometry", {}).setdefault("part", {})
        part["shape"] = "circle"
        part["width"] = float(s_mm) * 1e-3
        part["height"] = float(s_mm) * 1e-3
        cfg.setdefault("thermal", {})["n_steps"] = int(args.n_steps)

        po = cfg.setdefault("placement_optimizer", {})
        po["enabled"] = True
        po["n_parts"] = 4
        po["shape_template"] = "circle"
        po["n_candidates"] = int(args.n_candidates)
        po["proxy_top_k"] = int(args.proxy_top_k)

        tag = f"size_{str(s_mm).replace('.', 'p')}mm"
        cfg_path = TMP_CFG_DIR / f"{tag}.yaml"
        out_dir = args.out_root / tag
        _write_yaml(cfg_path, cfg)

        cmd = ["python3", str(ROOT / "rfam_eqs_coupled.py"), "--config", str(cfg_path), "--output-dir", str(out_dir)]
        print("$", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            raise SystemExit(rc)

        top_path = out_dir / "placement_topk_coupled.json"
        best_path = out_dir / "placement_best.json"
        top_rows = json.loads(top_path.read_text()) if top_path.exists() else []
        best = json.loads(best_path.read_text()) if best_path.exists() else {}

        ranked = sorted(top_rows, key=_score, reverse=True)
        top6 = ranked[: min(6, len(ranked))]
        _plot_version(out_dir, s_mm, chamber_x, chamber_y, top6)

        summary_rows.append(
            {
                "size_mm": s_mm,
                "output_dir": str(out_dir),
                "best_mean_rho": float(best.get("mean_rho", 0.0)),
                "best_min_rho": float(best.get("min_rho", 0.0)),
                "best_std_rho": float(best.get("std_rho", 0.0)),
                "best_temp_violation": float(best.get("temp_violation", 0.0)),
                "best_score": float(_score(best)),
                "best_layout": best.get("layout", []),
            }
        )

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_json = args.out_root / "placement_size_study_summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2))
    _plot_cross_size(summary_rows, args.out_root / "placement_size_study_summary.png")

    if summary_rows:
        best_row = max(summary_rows, key=lambda r: float(r["best_score"]))
        print("\\nBest overall size variant:")
        print(json.dumps(best_row, indent=2))
        print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
