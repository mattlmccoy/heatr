#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("./.mplconfig").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


def _out_path(out_dir: Path, base_name: str, suffix: str) -> Path:
    p = Path(base_name)
    if not suffix:
        return out_dir / base_name
    return out_dir / f"{p.stem}.{suffix}{p.suffix}"


def _write_svg(rows: list[dict], out_path: Path) -> None:
    by_shape: dict[str, list[dict]] = {}
    for row in rows:
        by_shape.setdefault(str(row.get("shape", "shape")), []).append(row)
    for shape_rows in by_shape.values():
        shape_rows.sort(key=lambda r: float(r.get("wall_thickness_mm", 0.0)))

    width, height = 960, 560
    pad_l, pad_r, pad_t, pad_b = 70, 24, 30, 48
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    all_t = [float(r.get("wall_thickness_mm", 0.0)) for r in rows]
    all_rho = [float(r.get("mean_rho_rel_part_final", 0.0)) for r in rows]
    tmin, tmax = min(all_t), max(all_t)
    rmin, rmax = min(all_rho), max(all_rho)
    if abs(tmax - tmin) < 1e-9:
        tmax = tmin + 1.0
    if abs(rmax - rmin) < 1e-9:
        rmax = rmin + 1e-3
    colors = ["#0c7bdc", "#dd614a", "#3a9f56", "#8c6ad9", "#e08b00"]
    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    lines.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#d0d7de"/>')
    lines.append(f'<text x="{pad_l}" y="20" font-size="16" font-family="Helvetica,Arial,sans-serif" fill="#111827">Shell Sweep: mean density vs wall thickness</text>')
    for i in range(6):
        yv = rmin + (rmax - rmin) * i / 5.0
        py = pad_t + plot_h - (yv - rmin) / (rmax - rmin) * plot_h
        lines.append(f'<line x1="{pad_l}" y1="{py:.2f}" x2="{pad_l + plot_w}" y2="{py:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l - 8}" y="{py + 4:.2f}" font-size="11" text-anchor="end" font-family="Helvetica,Arial,sans-serif" fill="#4b5563">{yv:.3f}</text>')
    for i, shape in enumerate(sorted(by_shape.keys())):
        shape_rows = by_shape[shape]
        color = colors[i % len(colors)]
        pts: list[str] = []
        for r in shape_rows:
            t = float(r.get("wall_thickness_mm", 0.0))
            rho = float(r.get("mean_rho_rel_part_final", 0.0))
            px = pad_l + (t - tmin) / (tmax - tmin) * plot_w
            py = pad_t + plot_h - (rho - rmin) / (rmax - rmin) * plot_h
            pts.append(f"{px:.2f},{py:.2f}")
            lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.5" fill="{color}"/>')
        lines.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    lines.append(f'<text x="{pad_l + plot_w / 2:.2f}" y="{height - 10}" font-size="12" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" fill="#374151">Wall thickness (mm)</text>')
    lines.append(
        f'<text x="20" y="{pad_t + plot_h / 2:.2f}" transform="rotate(-90 20 {pad_t + plot_h / 2:.2f})" '
        'font-size="12" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" fill="#374151">'
        'Mean relative density, rho_rel (-)</text>'
    )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines))


def _write_png(rows: list[dict], out_path: Path) -> None:
    by_shape: dict[str, list[dict]] = {}
    for row in rows:
        by_shape.setdefault(str(row.get("shape", "shape")), []).append(row)
    for shape_rows in by_shape.values():
        shape_rows.sort(key=lambda r: float(r.get("wall_thickness_mm", 0.0)))
    fig, ax = plt.subplots(1, 1, figsize=(9.6, 5.6), dpi=140)
    for shape in sorted(by_shape.keys()):
        shape_rows = by_shape[shape]
        x = [float(r.get("wall_thickness_mm", 0.0)) for r in shape_rows]
        y = [float(r.get("mean_rho_rel_part_final", 0.0)) for r in shape_rows]
        ax.plot(x, y, marker="o", lw=1.8, ms=4.5, label=shape)
    ax.set_title("Shell Sweep: mean density vs wall thickness")
    ax.set_xlabel("Wall thickness (mm)")
    ax.set_ylabel("Mean relative density, rho_rel (-)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill shell-sweep report images from existing summary.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--output-suffix", type=str, default="")
    args = ap.parse_args()
    out_dir = args.output_dir.resolve()
    summary_path = out_dir / "shell_sweep_summary.json"
    if not summary_path.exists():
        raise SystemExit("Missing shell_sweep_summary.json; cannot backfill shell sweep report.")
    data = json.loads(summary_path.read_text())
    rows = data.get("rows", []) if isinstance(data, dict) else []
    if not isinstance(rows, list) or not rows:
        raise SystemExit("No shell sweep rows found in shell_sweep_summary.json.")
    suffix = str(args.output_suffix or "").strip()
    out_svg = _out_path(out_dir, "shell_sweep_report.svg", suffix)
    out_png = _out_path(out_dir, "shell_sweep_report.png", suffix)
    _write_svg(rows, out_svg)
    _write_png(rows, out_png)
    print(f"Backfilled shell sweep reports: {out_png.name}, {out_svg.name}")


if __name__ == "__main__":
    main()

