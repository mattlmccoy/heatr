#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("./.mplconfig").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


def _out_path(out_dir: Path, base_name: str, suffix: str) -> Path:
    p = Path(base_name)
    if not suffix:
        return out_dir / base_name
    return out_dir / f"{p.stem}.{suffix}{p.suffix}"


def _load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


def _render_scatter(rows: list[dict], pareto: list[dict], best: dict, out_png: Path) -> None:
    if not rows:
        raise ValueError("no rows to render")
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=180)
    means = [float(r.get("mean_rho", 0.0)) for r in rows]
    mins = [float(r.get("min_rho", 0.0)) for r in rows]
    stds = [float(r.get("std_rho", 0.0)) for r in rows]
    tv = [float(r.get("temp_violation", 0.0)) for r in rows]
    ax[0].scatter(means, mins, c=stds, cmap="viridis", s=20, alpha=0.35, label="all candidates")
    ax[0].set_xlabel("mean rho")
    ax[0].set_ylabel("min rho")
    ax[0].set_title("Densification tradeoff")
    ax[0].grid(alpha=0.25)
    sc_tv = ax[1].scatter(stds, means, c=tv, cmap="magma", s=20, alpha=0.35, label="all candidates")
    ax[1].set_xlabel("std rho")
    ax[1].set_ylabel("mean rho")
    ax[1].set_title("Uniformity vs mean")
    ax[1].grid(alpha=0.25)
    if pareto:
        ax[0].scatter(
            [float(r.get("mean_rho", 0.0)) for r in pareto],
            [float(r.get("min_rho", 0.0)) for r in pareto],
            marker="D", s=40, facecolors="none", edgecolors="black", linewidths=1.1, label="pareto front",
        )
        ax[1].scatter(
            [float(r.get("std_rho", 0.0)) for r in pareto],
            [float(r.get("mean_rho", 0.0)) for r in pareto],
            marker="D", s=40, facecolors="none", edgecolors="black", linewidths=1.1, label="pareto front",
        )
    bx = float(best.get("mean_rho", 0.0))
    by = float(best.get("min_rho", 0.0))
    ax[0].plot([bx], [by], "r*", ms=14)
    ax[1].plot([float(best.get("std_rho", 0.0))], [bx], "r*", ms=14)
    ax[0].annotate(
        "Selected Pareto point",
        xy=(bx, by),
        xytext=(0.03, 0.06),
        textcoords="axes fraction",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#444444", alpha=0.9),
    )
    cbar = plt.colorbar(sc_tv, ax=ax[1], shrink=0.88)
    cbar.set_label("temp ceiling violation [C]")
    for axi in ax:
        axi.legend(loc="best", fontsize=8)
    fig.suptitle("Placement Optimizer (backfilled)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill placement reports from existing run outputs.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--output-suffix", type=str, default="")
    args = ap.parse_args()

    out_dir = args.output_dir.resolve()
    topk_path = out_dir / "placement_topk_coupled.json"
    pareto_path = out_dir / "placement_pareto.json"
    best_path = out_dir / "placement_best.json"
    if not topk_path.exists():
        raise SystemExit("Missing placement_topk_coupled.json; cannot backfill placement report.")
    rows = _load_rows(topk_path)
    if not rows:
        raise SystemExit("No placement coupled candidates found.")
    pareto = _load_rows(pareto_path) if pareto_path.exists() else []
    best = json.loads(best_path.read_text()) if best_path.exists() else max(rows, key=lambda r: float(r.get("mean_rho", -math.inf)))
    out_png = _out_path(out_dir, "placement_report.png", str(args.output_suffix or "").strip())
    _render_scatter(rows, pareto, best, out_png)
    print(f"Backfilled placement report: {out_png}")


if __name__ == "__main__":
    main()

