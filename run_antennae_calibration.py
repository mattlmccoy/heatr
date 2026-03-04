#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", str(Path("./.mplconfig").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from rfam_eqs_coupled import (
    _pareto_front,
    _recommend_knee,
    _resolve_antennae_instances,
    generate_antennae_preview,
    run_sim,
    save_outputs,
)


def _parse_sizes(raw: str) -> list[float]:
    out: list[float] = []
    for tok in str(raw or "").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            v = float(t)
        except Exception:
            continue
        if v > 0:
            out.append(v)
    seen: set[float] = set()
    dedup: list[float] = []
    for v in out:
        key = round(v, 6)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(v)
    return dedup


def _score_proxy(cfg_case: dict) -> dict:
    resolved = _resolve_antennae_instances(cfg_case)
    rows = resolved.get("instances", []) if isinstance(resolved, dict) else []
    if not isinstance(rows, list):
        rows = []
    deficits = np.array([float(r.get("qrf_deficit", 0.0)) for r in rows], dtype=float) if rows else np.zeros((0,), dtype=float)
    sizes = np.array([float(r.get("size_mm", 0.0)) for r in rows], dtype=float) if rows else np.zeros((0,), dtype=float)
    mean_deficit = float(np.mean(deficits)) if deficits.size else 0.0
    mean_size = float(np.mean(sizes)) if sizes.size else 0.0
    n_inst = int(len(rows))
    score = 0.65 * mean_deficit + 0.20 * min(1.0, n_inst / 4.0) + 0.15 * min(1.0, mean_size / 2.0)
    return {
        "proxy_score": float(score),
        "n_instances": n_inst,
        "mean_qrf_deficit": mean_deficit,
        "mean_size_mm": mean_size,
        "instances": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline antennae calibration runner (EQS proxy + coupled rerank).")
    ap.add_argument("--config", type=Path, required=True, help="Base config yaml")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--sizes-mm", type=str, default="", help="Comma-separated global-size ladder (mm)")
    ap.add_argument("--include-auto", action="store_true", help="Also evaluate auto-size mode")
    ap.add_argument("--top-k", type=int, default=3, help="Top proxy candidates to run with coupled model")
    ap.add_argument("--use-turntable", action="store_true", help="Enable turntable during coupled validation")
    ap.add_argument("--job-id", type=str, default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if not isinstance(cfg, dict):
        raise SystemExit("invalid config")
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    ant = cfg.get("antennae", {}) if isinstance(cfg.get("antennae", {}), dict) else {}
    ant["enabled"] = True
    cfg["antennae"] = ant
    spk = cfg.get("spike", {}) if isinstance(cfg.get("spike", {}), dict) else {}
    spk["enabled"] = True
    cfg["spike"] = spk

    if args.use_turntable:
        tt = cfg.get("turntable", {}) if isinstance(cfg.get("turntable", {}), dict) else {}
        tt["enabled"] = True
        tt["rotation_deg"] = float(tt.get("rotation_deg", 90.0))
        tt["total_rotations"] = int(tt.get("total_rotations", 1))
        cfg["turntable"] = tt

    manual_sizes = _parse_sizes(args.sizes_mm)
    if not manual_sizes:
        g0 = float(ant.get("global_size_mm", 1.0))
        manual_sizes = sorted(set([max(0.2, g0 - 0.5), g0, g0 + 0.5, g0 + 1.0]))

    candidates: list[dict] = []
    cid = 0
    for s_mm in manual_sizes:
        case = copy.deepcopy(cfg)
        case_ant = case.get("antennae", {}) if isinstance(case.get("antennae", {}), dict) else {}
        case_ant["enabled"] = True
        case_ant["size_mode"] = "global"
        case_ant["global_size_mm"] = float(s_mm)
        case["antennae"] = case_ant
        proxy = _score_proxy(case)
        candidates.append({
            "candidate_id": cid,
            "size_mode": "global",
            "global_size_mm": float(s_mm),
            **proxy,
        })
        cid += 1

    if args.include_auto or str(ant.get("size_mode", "global")).strip().lower() == "auto":
        case = copy.deepcopy(cfg)
        case_ant = case.get("antennae", {}) if isinstance(case.get("antennae", {}), dict) else {}
        case_ant["enabled"] = True
        case_ant["size_mode"] = "auto"
        case["antennae"] = case_ant
        proxy = _score_proxy(case)
        candidates.append({
            "candidate_id": cid,
            "size_mode": "auto",
            "global_size_mm": None,
            **proxy,
        })

    if not candidates:
        raise SystemExit("no calibration candidates produced")

    candidates.sort(key=lambda r: float(r.get("proxy_score", 0.0)), reverse=True)
    (out_dir / "antennae_calibration_candidates.json").write_text(json.dumps(candidates, indent=2))

    top_k = max(1, int(args.top_k))
    top = candidates[:top_k]
    coupled_rows: list[dict] = []
    for rank, row in enumerate(top, start=1):
        case = copy.deepcopy(cfg)
        case_ant = case.get("antennae", {}) if isinstance(case.get("antennae", {}), dict) else {}
        case_ant["enabled"] = True
        case_ant["size_mode"] = str(row.get("size_mode", "global"))
        if row.get("global_size_mm", None) is not None:
            case_ant["global_size_mm"] = float(row["global_size_mm"])
        case["antennae"] = case_ant
        case_dir = cases_dir / f"case_{rank:02d}_{case_ant['size_mode']}"
        case_dir.mkdir(parents=True, exist_ok=True)
        generate_antennae_preview(case, case_dir / "antennae_anchor_preview.png")
        state, summary, hist, tt_rotation_steps, opt_data = run_sim(case)
        save_outputs(case, state, summary, hist, case_dir, tt_rotation_steps=tt_rotation_steps, opt_data=opt_data)
        coupled_rows.append({
            "candidate_id": int(row.get("candidate_id", rank)),
            "rank_proxy": rank,
            "size_mode": case_ant["size_mode"],
            "global_size_mm": row.get("global_size_mm"),
            "proxy_score": float(row.get("proxy_score", 0.0)),
            "mean_rho": float(summary.get("mean_rho_rel_part_final", 0.0)),
            "min_rho": float(summary.get("min_rho_rel_part_final", 0.0)),
            "std_rho": float(summary.get("std_rho_rel_part_final", 0.0)),
            "max_temp_c": float(summary.get("max_T_part_final_c", 0.0)),
            "temp_violation": max(0.0, float(summary.get("max_T_part_final_c", 0.0)) - 250.0),
            "n_instances": int(summary.get("antennae_count", 0)),
            "output_dir": case_dir.relative_to(out_dir).as_posix(),
        })

    coupled_rows.sort(
        key=lambda r: (
            float(r.get("temp_violation", 0.0)) > 0.0,
            -float(r.get("mean_rho", 0.0)),
            -float(r.get("min_rho", 0.0)),
            float(r.get("std_rho", 0.0)),
        )
    )
    (out_dir / "antennae_calibration_topk_coupled.json").write_text(json.dumps(coupled_rows, indent=2))
    pareto = _pareto_front(coupled_rows)
    (out_dir / "antennae_calibration_pareto.json").write_text(json.dumps(pareto, indent=2))
    best = _recommend_knee(pareto if pareto else coupled_rows)
    (out_dir / "antennae_calibration_best.json").write_text(json.dumps(best, indent=2))

    # Summary report
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=180)
    means = [float(r.get("mean_rho", 0.0)) for r in coupled_rows]
    mins = [float(r.get("min_rho", 0.0)) for r in coupled_rows]
    stds = [float(r.get("std_rho", 0.0)) for r in coupled_rows]
    tv = [float(r.get("temp_violation", 0.0)) for r in coupled_rows]
    s0 = ax[0].scatter(means, mins, c=stds, cmap="viridis", s=34, alpha=0.9)
    ax[0].set_title("Densification tradeoff")
    ax[0].set_xlabel("mean rho")
    ax[0].set_ylabel("min rho")
    ax[0].grid(alpha=0.25)
    plt.colorbar(s0, ax=ax[0], shrink=0.86).set_label("std rho")

    s1 = ax[1].scatter(stds, means, c=tv, cmap="magma_r", s=34, alpha=0.9)
    ax[1].set_title("Uniformity vs mean")
    ax[1].set_xlabel("std rho")
    ax[1].set_ylabel("mean rho")
    ax[1].grid(alpha=0.25)
    plt.colorbar(s1, ax=ax[1], shrink=0.86).set_label("temp ceiling violation [C]")

    if best:
        ax[0].plot([float(best.get("mean_rho", 0.0))], [float(best.get("min_rho", 0.0))], "r*", ms=13)
        ax[1].plot([float(best.get("std_rho", 0.0))], [float(best.get("mean_rho", 0.0))], "r*", ms=13)
    fig.suptitle("Antennae Calibration", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "antennae_calibration_report.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote antennae calibration outputs to: {out_dir}")


if __name__ == "__main__":
    main()
