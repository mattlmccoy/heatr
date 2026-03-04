#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import math
from pathlib import Path

import yaml
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CONFIGS = ROOT / "configs"
OUT = ROOT / "outputs_eqs" / "shell_sweep"
MPLCONFIG = ROOT / ".mplconfig"
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return data


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _set_shape(cfg: dict, shape: str) -> None:
    g = cfg.setdefault("geometry", {})
    part = g.setdefault("part", {})
    part["shape"] = shape


def _write_svg_report(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
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
        ly = pad_t + 18 + i * 18
        lx = pad_l + plot_w - 180
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
        lines.append(f'<text x="{lx + 24}" y="{ly + 4}" font-size="12" font-family="Helvetica,Arial,sans-serif" fill="#111827">{shape}</text>')

    for i in range(int(tmin), int(math.ceil(tmax)) + 1):
        px = pad_l + (float(i) - tmin) / (tmax - tmin) * plot_w
        lines.append(f'<line x1="{px:.2f}" y1="{pad_t + plot_h}" x2="{px:.2f}" y2="{pad_t + plot_h + 5}" stroke="#6b7280"/>')
        lines.append(f'<text x="{px:.2f}" y="{pad_t + plot_h + 20}" font-size="11" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" fill="#4b5563">{i}</text>')

    lines.append(f'<text x="{pad_l + plot_w / 2:.2f}" y="{height - 10}" font-size="12" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" fill="#374151">Wall thickness (mm)</text>')
    lines.append(
        f'<text x="20" y="{pad_t + plot_h / 2:.2f}" transform="rotate(-90 20 {pad_t + plot_h / 2:.2f})" '
        'font-size="12" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" fill="#374151">'
        'Mean relative density, rho_rel (-)</text>'
    )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines))


def _write_png_report(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
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
    p = argparse.ArgumentParser(description="Run shelled geometry sweep for circle/square thickness ladder")
    p.add_argument("--thickness-mm", default="2,3,4,5,6,7")
    p.add_argument("--shapes", default="circle,square")
    p.add_argument("--output-root", type=Path, default=OUT)
    args = p.parse_args()

    thicknesses = [float(x.strip()) for x in str(args.thickness_mm).split(",") if x.strip()]
    shapes = [str(x).strip() for x in str(args.shapes).split(",") if x.strip()]

    base_cfg_map = {
        "circle": CONFIGS / "shape_circle_6min.yaml",
        "square": CONFIGS / "rfam_eqs_xz_uniform_500w.yaml",
    }

    tmp_cfg_dir = ROOT / "configs" / "_tmp_shell_sweep"
    rows: list[dict] = []

    for shape in shapes:
        if shape not in base_cfg_map:
            raise ValueError(f"Unsupported shape for v1 shell sweep: {shape}")
        base_cfg = _load_yaml(base_cfg_map[shape])
        _set_shape(base_cfg, shape)
        for t_mm in thicknesses:
            cfg = json.loads(json.dumps(base_cfg))
            geom = cfg.setdefault("geometry", {})
            geom["shell"] = {
                "enabled": True,
                "wall_thickness_mm": float(t_mm),
                "method": "offset_inward",
            }

            run_name = f"{shape}_shell_t{str(t_mm).replace('.', 'p')}mm"
            cfg_path = tmp_cfg_dir / f"{run_name}.yaml"
            out_dir = args.output_root / run_name
            _write_yaml(cfg_path, cfg)

            cmd = [
                "python3",
                str(ROOT / "rfam_eqs_coupled.py"),
                "--config",
                str(cfg_path),
                "--output-dir",
                str(out_dir),
            ]
            print("$", " ".join(cmd))
            rc = subprocess.call(cmd, cwd=ROOT)
            if rc != 0:
                raise SystemExit(rc)

            summary_path = out_dir / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            rows.append(
                {
                    "shape": shape,
                    "wall_thickness_mm": t_mm,
                    "output_dir": str(out_dir),
                    "mean_rho_rel_part_final": float(summary.get("mean_rho_rel_part_final", 0.0)),
                    "max_T_part_final_c": float(summary.get("max_T_part_final_c", 0.0)),
                    "inter_part_mean_rho_std": float(summary.get("inter_part_mean_rho_std", 0.0)),
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    by_shape: dict[str, list[dict]] = {}
    for row in rows:
        by_shape.setdefault(str(row.get("shape", "shape")), []).append(row)
    for shape_rows in by_shape.values():
        shape_rows.sort(key=lambda r: float(r.get("wall_thickness_mm", 0.0)))

    report_path = args.output_root / "shell_sweep_report.svg"
    _write_svg_report(rows, report_path)
    report_png_path = args.output_root / "shell_sweep_report.png"
    _write_png_report(rows, report_png_path)

    ranked = sorted(rows, key=lambda r: float(r.get("mean_rho_rel_part_final", 0.0)), reverse=True)
    summary = {
        "best_overall_by_mean_rho": ranked[0] if ranked else None,
        "rows": rows,
    }
    summary_path = args.output_root / "shell_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {report_png_path}")


if __name__ == "__main__":
    main()
