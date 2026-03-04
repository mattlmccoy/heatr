#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import math
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", str(Path("./.mplconfig").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None


def _load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _knee_scores(rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.array([], dtype=float)
    arr_mean = np.array([float(r.get("mean_rho", 0.0)) for r in rows], dtype=float)
    arr_min = np.array([float(r.get("min_rho", 0.0)) for r in rows], dtype=float)
    arr_std = np.array([float(r.get("std_rho", 0.0)) for r in rows], dtype=float)
    arr_tv = np.array([float(r.get("temp_violation", 0.0)) for r in rows], dtype=float)
    n_mean = (arr_mean - np.min(arr_mean)) / max(np.max(arr_mean) - np.min(arr_mean), 1e-9)
    n_min = (arr_min - np.min(arr_min)) / max(np.max(arr_min) - np.min(arr_min), 1e-9)
    n_std = 1.0 - (arr_std - np.min(arr_std)) / max(np.max(arr_std) - np.min(arr_std), 1e-9)
    n_tv = 1.0 - (arr_tv - np.min(arr_tv)) / max(np.max(arr_tv) - np.min(arr_tv), 1e-9)
    return 0.35 * n_mean + 0.35 * n_min + 0.20 * n_std + 0.10 * n_tv


def _render_orientation_angle_report(
    angle_rows: list[dict],
    best: dict,
    output_png: Path,
    color_metric: str = "mean_rho",
    temp_ceiling_c: float | None = None,
    exposure_text: str | None = None,
) -> None:
    if not angle_rows:
        return
    angles = [float(r["rotation_deg"]) for r in angle_rows]
    scores = [float(r["best_effectiveness_score"]) for r in angle_rows]
    temp_v = [float(r.get("best_temp_violation", 0.0)) for r in angle_rows]
    mean_r = [float(r.get("best_mean_rho", 0.0)) for r in angle_rows]
    min_r = [float(r.get("best_min_rho", 0.0)) for r in angle_rows]
    pareto_flag = [bool(r.get("is_pareto_angle", False)) for r in angle_rows]

    metric = str(color_metric or "mean_rho").strip().lower()
    if metric == "temp_violation":
        cvals = np.array(temp_v, dtype=float)
        cmap = "magma_r"
        if temp_ceiling_c is not None:
            cbar_label = f"temp ceiling violation [C], ceiling={float(temp_ceiling_c):.1f} C"
        else:
            cbar_label = "temp ceiling violation [C]"
    elif metric == "min_rho":
        cvals = np.array(min_r, dtype=float)
        cmap = "viridis"
        cbar_label = "best min rho"
    else:
        metric = "mean_rho"
        cvals = np.array(mean_r, dtype=float)
        cmap = "viridis"
        cbar_label = "best mean rho"

    fig, ax = plt.subplots(1, 1, figsize=(10.2, 4.4), dpi=180)
    sc = ax.scatter(angles, scores, c=cvals, cmap=cmap, s=70, alpha=0.85, edgecolor="k", linewidth=0.35, label="angle candidates")
    for a, s, tv in zip(angles, scores, temp_v):
        if float(tv) > 0.0:
            ax.plot([a], [s], marker="o", ms=10, markerfacecolor="none", markeredgecolor="#d62728", markeredgewidth=1.25, linestyle="None")
    for a, s, pr in zip(angles, scores, pareto_flag):
        if pr:
            ax.plot([a], [s], marker="D", ms=7, markerfacecolor="none", markeredgecolor="black", linestyle="None")
    ax.plot([float(best.get("rotation_deg", 0.0))], [float(best.get("effectiveness_score", max(scores) if scores else 0.0))], "r*", ms=15, label="selected ideal angle")
    ax.set_title("Orientation Angle Effectiveness (backfilled)")
    ax.set_xlabel("rotation angle [deg]")
    ax.set_ylabel("effectiveness score")
    ax.grid(alpha=0.25)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.86)
    cbar.set_label(cbar_label)
    for a, s, m in zip(angles, scores, mean_r):
        ax.text(a, s + 0.005, f"{m:.3f}", fontsize=7, ha="center", va="bottom", alpha=0.7)
    ax.text(0.995, 0.015, "red ring = temp ceiling violated", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#b22222")
    ax.legend(loc="best", fontsize=8)
    context_lines: list[str] = []
    if exposure_text:
        context_lines.append(str(exposure_text))
    if temp_ceiling_c is not None:
        exp_txt_l = str(exposure_text or "").lower()
        if "ceiling" not in exp_txt_l:
            context_lines.append(f"temperature ceiling: {float(temp_ceiling_c):.1f} C")
    if context_lines:
        ax.text(
            0.01,
            0.99,
            "\n".join(context_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#666666", alpha=0.9),
        )
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def _infer_context_from_run(out_dir: Path, candidates: list[dict]) -> tuple[float | None, str | None]:
    temp_ceiling_c: float | None = None
    for row in candidates:
        if isinstance(row, dict) and row.get("temp_ceiling_c", None) is not None:
            try:
                temp_ceiling_c = float(row.get("temp_ceiling_c"))
                break
            except Exception:
                pass

    cfg = {}
    cfg_path = out_dir / "used_config.yaml"
    if cfg_path.exists():
        try:
            cfg_raw = yaml.safe_load(cfg_path.read_text())
            if isinstance(cfg_raw, dict):
                cfg = cfg_raw
        except Exception:
            cfg = {}

    oo = cfg.get("orientation_optimizer", {}) if isinstance(cfg.get("orientation_optimizer", {}), dict) else {}
    cst = oo.get("constraints", {}) if isinstance(oo.get("constraints", {}), dict) else {}
    if temp_ceiling_c is None and cst.get("temp_ceiling_c", None) is not None:
        try:
            temp_ceiling_c = float(cst.get("temp_ceiling_c"))
        except Exception:
            temp_ceiling_c = None

    # Legacy backfill support: infer ceiling from candidate metrics when explicit
    # constraint fields were not persisted.
    if temp_ceiling_c is None:
        inferred_vals: list[float] = []
        for row in candidates:
            if not isinstance(row, dict):
                continue
            if row.get("max_temp_c", None) is None or row.get("temp_violation", None) is None:
                continue
            try:
                inferred_vals.append(float(row.get("max_temp_c")) - float(row.get("temp_violation")))
            except Exception:
                continue
        if inferred_vals:
            # Median is robust if a few rows are noisy/rounded.
            temp_ceiling_c = float(np.median(np.asarray(inferred_vals, dtype=float)))

    exposures = sorted({float(r.get("exposure_s")) for r in candidates if isinstance(r, dict) and r.get("exposure_s", None) is not None})
    exposure_text: str | None = None
    if exposures:
        if len(exposures) == 1:
            exposure_text = f"fixed exposure: {exposures[0]:.0f} s"
        else:
            diffs = [exposures[i + 1] - exposures[i] for i in range(len(exposures) - 1)]
            step = min([d for d in diffs if d > 1e-9], default=0.0)
            if step > 0:
                exposure_text = f"exposure sweep: {exposures[0]:.0f}-{exposures[-1]:.0f} s (step {step:.0f} s)"
            else:
                exposure_text = f"exposure sweep: {exposures[0]:.0f}-{exposures[-1]:.0f} s"
    elif isinstance(oo, dict) and oo:
        try:
            e_min = float(oo.get("exposure_min_s", 0.0))
            e_max = float(oo.get("exposure_max_s", 0.0))
            e_step = float(oo.get("exposure_step_s", 0.0))
            if e_max > 0 and e_max >= e_min:
                if e_step > 0:
                    exposure_text = f"exposure sweep: {e_min:.0f}-{e_max:.0f} s (step {e_step:.0f} s)"
                else:
                    exposure_text = f"exposure sweep: {e_min:.0f}-{e_max:.0f} s"
        except Exception:
            exposure_text = None
    else:
        therm = cfg.get("thermal", {}) if isinstance(cfg.get("thermal", {}), dict) else {}
        try:
            dt = float(therm.get("dt_s", 0.5))
            n_steps = int(therm.get("n_steps", 0))
            if dt > 0 and n_steps > 0:
                exposure_text = f"fixed exposure: {dt * n_steps:.0f} s"
        except Exception:
            exposure_text = None

    if temp_ceiling_c is None:
        # Last-resort default used throughout solver constraints.
        temp_ceiling_c = 250.0
    if temp_ceiling_c is not None:
        if exposure_text:
            exposure_text = f"{exposure_text}\ntemperature ceiling: {temp_ceiling_c:.1f} C"
        else:
            exposure_text = f"temperature ceiling: {temp_ceiling_c:.1f} C"
    return temp_ceiling_c, exposure_text


def _out_path(out_dir: Path, base_name: str, suffix: str) -> Path:
    p = Path(base_name)
    if not suffix:
        return out_dir / base_name
    return out_dir / f"{p.stem}.{suffix}{p.suffix}"


def _set_exposure_seconds(cfg: dict, exposure_s: float) -> None:
    therm = cfg.setdefault("thermal", {})
    dt_s = float(therm.get("dt_s", 0.5))
    n_steps = max(1, int(round(float(exposure_s) / max(dt_s, 1e-9))))
    therm["n_steps"] = n_steps


def _orientation_frame_from_state(state, row: dict) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    T = np.asarray(state.T, dtype=float)
    tmin = float(np.nanmin(T))
    tmax = float(np.nanmax(T))
    if math.isclose(tmin, tmax):
        tmax = tmin + 1e-6
    norm = np.clip((T - tmin) / max(tmax - tmin, 1e-9), 0.0, 1.0)
    cmap = plt.get_cmap("inferno")
    rgba = cmap(norm)
    rgb = (255.0 * np.asarray(rgba[..., :3], dtype=float)).astype(np.uint8)
    mask = np.asarray(state.part_mask, dtype=bool)
    rgb[~mask] = (0.30 * rgb[~mask]).astype(np.uint8)
    img = Image.fromarray(rgb).resize((280, 280), resample=Image.Resampling.BILINEAR)
    bar_h = 46
    canvas = Image.new("RGB", (280, 280 + bar_h), color=(15, 15, 18))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    txt1 = f"angle {float(row['rotation_deg']):.1f} deg | exposure {float(row['best_exposure_s']):.0f} s"
    txt2 = f"mean {float(row['best_mean_rho']):.3f} | min {float(row['best_min_rho']):.3f} | dTceil {float(row['best_temp_violation']):.1f} C"
    draw.text((8, 286), txt1, fill=(235, 235, 235))
    draw.text((8, 302), txt2, fill=(190, 190, 190))
    return np.asarray(canvas, dtype=np.uint8)


def _base_cfg_for_orientation_gif(out_dir: Path) -> dict:
    cfg_path = out_dir / "used_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path.name}; cannot regenerate GIF.")
    cfg = yaml.safe_load(cfg_path.read_text())
    if not isinstance(cfg, dict):
        raise RuntimeError("Invalid used_config.yaml; expected mapping.")
    base = copy.deepcopy(cfg)
    base.pop("orientation_optimizer", None)
    base.pop("optimizer", None)
    base.pop("turntable", None)
    base.pop("placement_optimizer", None)
    return base


def _write_orientation_angle_gif(
    out_dir: Path,
    angle_rows: list[dict],
    output_gif: Path,
    frame_duration_s: float,
) -> None:
    if not angle_rows:
        return
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required for GIF generation") from exc
    from rfam_eqs_coupled import run_sim

    base = _base_cfg_for_orientation_gif(out_dir)
    frames: list[np.ndarray] = []
    for row in sorted(angle_rows, key=lambda r: float(r.get("rotation_deg", 0.0))):
        c = copy.deepcopy(base)
        if isinstance(c.get("geometry", {}).get("part", {}), dict):
            c["geometry"]["part"]["rotation_deg"] = float(row["rotation_deg"])
        elif isinstance(c.get("geometry", {}).get("parts", []), list):
            for part in c["geometry"]["parts"]:
                if isinstance(part, dict):
                    part["rotation_deg"] = float(row["rotation_deg"])
        _set_exposure_seconds(c, float(row.get("best_exposure_s", 0.0)))
        state, _, _, _, _ = run_sim(c)
        frames.append(_orientation_frame_from_state(state, row))
    if frames:
        imageio.mimsave(output_gif, frames, duration=max(0.1, float(frame_duration_s)), loop=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill orientation diagnostics from existing run outputs.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Run output folder containing orientation JSON files.")
    ap.add_argument("--output-suffix", type=str, default="", help="Optional suffix inserted before file extension.")
    ap.add_argument("--gif-only", action="store_true", help="Generate orientation_angle_gallery GIF only.")
    ap.add_argument("--gif-duration-s", type=float, default=2.0, help="GIF frame duration in seconds.")
    args = ap.parse_args()

    out_dir = args.output_dir.resolve()
    cand_path = out_dir / "orientation_candidates.json"
    pareto_path = out_dir / "orientation_pareto.json"
    best_path = out_dir / "orientation_best.json"

    if cand_path.exists():
        candidates = _load_json(cand_path)
    elif pareto_path.exists():
        candidates = _load_json(pareto_path)
    else:
        raise SystemExit("Missing orientation_candidates.json and orientation_pareto.json; cannot backfill.")
    if not candidates:
        raise SystemExit("No orientation candidates found in source files.")

    scores = _knee_scores(candidates)
    for i, r in enumerate(candidates):
        r["effectiveness_score"] = float(scores[i]) if i < len(scores) else 0.0
        if "is_pareto" not in r:
            r["is_pareto"] = bool(pareto_path.exists())

    best = json.loads(best_path.read_text()) if best_path.exists() else max(candidates, key=lambda rr: float(rr.get("effectiveness_score", 0.0)))
    if "effectiveness_score" not in best:
        best["effectiveness_score"] = float(max(scores)) if len(scores) else 0.0

    angle_map: dict[float, list[dict]] = {}
    for r in candidates:
        angle_map.setdefault(float(r.get("rotation_deg", 0.0)), []).append(r)
    angle_rows: list[dict] = []
    for ang in sorted(angle_map.keys()):
        rows_ang = angle_map[ang]
        best_ang = max(rows_ang, key=lambda rr: float(rr.get("effectiveness_score", 0.0)))
        pareto_exposures = sorted(float(rr.get("exposure_s", 0.0)) for rr in rows_ang if bool(rr.get("is_pareto", False)))
        angle_rows.append({
            "rotation_deg": float(ang),
            "n_candidates": int(len(rows_ang)),
            "best_exposure_s": float(best_ang.get("exposure_s", 0.0)),
            "best_effectiveness_score": float(best_ang.get("effectiveness_score", 0.0)),
            "best_mean_rho": float(best_ang.get("mean_rho", 0.0)),
            "best_min_rho": float(best_ang.get("min_rho", 0.0)),
            "best_std_rho": float(best_ang.get("std_rho", 0.0)),
            "best_temp_violation": float(best_ang.get("temp_violation", 0.0)),
            "is_pareto_angle": bool(len(pareto_exposures) > 0),
            "pareto_exposures_s": pareto_exposures,
            "effective_reason": ("on pareto front" if pareto_exposures else "dominated or unavailable in source"),
        })

    suffix = str(args.output_suffix or "").strip()
    if bool(args.gif_only):
        out_gif_path = _out_path(out_dir, "orientation_angle_gallery.gif", suffix)
        _write_orientation_angle_gif(
            out_dir=out_dir,
            angle_rows=angle_rows,
            output_gif=out_gif_path,
            frame_duration_s=float(args.gif_duration_s),
        )
        print(f"Backfilled orientation GIF in: {out_gif_path}")
        if suffix:
            print(f"Output suffix: {suffix}")
        return

    cand_json_path = _out_path(out_dir, "orientation_candidates.json", suffix)
    cand_csv_path = _out_path(out_dir, "orientation_candidates.csv", suffix)
    ang_json_path = _out_path(out_dir, "orientation_angle_summary.json", suffix)
    ang_csv_path = _out_path(out_dir, "orientation_angle_summary.csv", suffix)
    ang_png_path = _out_path(out_dir, "orientation_angle_effectiveness.png", suffix)

    cand_json_path.write_text(json.dumps(candidates, indent=2))
    with cand_csv_path.open("w", newline="", encoding="utf-8") as f:
        keys = [
            "rotation_deg", "exposure_s", "source", "effectiveness_score", "is_pareto",
            "mean_rho", "min_rho", "std_rho", "max_temp_c", "temp_violation", "min_rho_violation",
        ]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in candidates:
            w.writerow({k: r.get(k, "") for k in keys})

    ang_json_path.write_text(json.dumps(angle_rows, indent=2))
    with ang_csv_path.open("w", newline="", encoding="utf-8") as f:
        keys = [
            "rotation_deg", "n_candidates", "best_exposure_s", "best_effectiveness_score",
            "best_mean_rho", "best_min_rho", "best_std_rho", "best_temp_violation",
            "is_pareto_angle", "pareto_exposures_s", "effective_reason",
        ]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in angle_rows:
            row_out = dict(r)
            row_out["pareto_exposures_s"] = ",".join(f"{v:.1f}" for v in r.get("pareto_exposures_s", []))
            w.writerow({k: row_out.get(k, "") for k in keys})

    temp_ceiling_c, context_text = _infer_context_from_run(out_dir, candidates)
    # Backfill defaults to density-colored view for interpretability.
    _render_orientation_angle_report(
        angle_rows,
        best,
        ang_png_path,
        color_metric="mean_rho",
        temp_ceiling_c=temp_ceiling_c,
        exposure_text=context_text,
    )
    print(f"Backfilled orientation diagnostics in: {out_dir}")
    if suffix:
        print(f"Output suffix: {suffix}")
    print("Note: GIF gallery skipped by default in diagnostics mode. Use --gif-only to rerun just the GIF.")


if __name__ == "__main__":
    main()
