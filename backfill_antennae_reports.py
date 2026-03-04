#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _load_instances(out_dir: Path) -> list[dict]:
    ant_path = out_dir / "antennae_summary.json"
    if ant_path.exists():
        try:
            raw = json.loads(ant_path.read_text())
            rows = raw.get("instances", []) if isinstance(raw, dict) else []
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
        except Exception:
            pass
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            raw = json.loads(summary_path.read_text())
            rows = raw.get("antennae_instances", []) if isinstance(raw, dict) else []
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
        except Exception:
            pass
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill antennae report from saved run artifacts.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--output-suffix", type=str, default="")
    args = ap.parse_args()

    out_dir = args.output_dir.resolve()
    fields_path = out_dir / "fields.npz"
    if not fields_path.exists():
        raise SystemExit("Missing fields.npz; cannot backfill antennae report.")
    fields = np.load(fields_path)
    x = np.asarray(fields["x"], dtype=float)
    y = np.asarray(fields["y"], dtype=float)
    qrf = np.asarray(fields["Qrf"], dtype=float) / 1.0e6
    part_mask = np.asarray(fields["part_mask"], dtype=bool)
    instances = _load_instances(out_dir)
    if not instances:
        raise SystemExit("No antennae instances found in antennae_summary.json or summary.json.")

    out_png = _out_path(out_dir, "antennae_report.png", str(args.output_suffix or "").strip())
    out_json = _out_path(out_dir, "antennae_anchors.json", str(args.output_suffix or "").strip())

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.7), dpi=180)
    qmin = float(np.nanmin(qrf))
    qmax = float(np.nanmax(qrf))
    if qmax <= qmin:
        qmax = qmin + 1e-6
    im = ax.imshow(
        qrf,
        extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        origin="lower",
        cmap="jet",
        vmin=qmin,
        vmax=qmax,
        interpolation="bilinear",
    )
    try:
        ax.contour(x, y, part_mask.astype(float), levels=[0.5], colors=["w"], linewidths=1.0)
    except Exception:
        pass

    for row in instances:
        cx = float(row.get("x_mm", 0.0))
        cy = float(row.get("y_mm", 0.0))
        rr = float(row.get("size_mm", row.get("global_size_mm", 1.0)))
        circ = plt.Circle((cx, cy), rr, edgecolor="#ff2d55", facecolor="none", linewidth=1.4, alpha=0.95)
        ax.add_patch(circ)
        ax.plot([cx], [cy], marker="o", ms=3.5, color="#ff2d55")
    ax.set_title("Antennae Overlay (backfilled)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    cbar = plt.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("Qrf [MW/m^3]")
    ax.text(
        0.01,
        0.01,
        f"antennae count: {len(instances)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#777777", alpha=0.92),
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_json.write_text(json.dumps({"instances": instances}, indent=2))
    print(f"Backfilled antennae report: {out_png.name}")


if __name__ == "__main__":
    main()

