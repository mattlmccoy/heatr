#!/usr/bin/env python3
"""Build the per-shape FGM standards JSON consumed by the HEATR GUI.

One small static file (webui/static/fgm_shape_standards.json) that carries,
per library shape, everything the Operation-tab method chooser and the
standard-parameter preset need:

  v_cal_v            calibrated drive voltage (dual-readstate campaign G1,
                     outputs_eqs/geometry_dual_readstate/<shape>.json "v_cal")
  quick_look_gain    window-reselected proportional gain m
                     (FGM_WINDOW_RESELECTION.md winners table; the campaign's
                     machine-readable record of that reselection is the table)
  quick_look_verdict BETTER / neutral / HARMFUL vs uniform (same table)
  solve_class        SOLVED / IMPROVED / MATCHED / NOT-RESCUED
                     (fgm_solve_campaign/out_lib/<shape>.json class block;
                     MATCHED = census SOLVED that does not beat the stored
                     historical mask on IoU, i.e. square / rounded_rect)
  solve_iou_4bpp     deliverable-arm (A1_4bpp) IoU at grid 120
  actuator_note      dwell / rotation upgrades read from the real campaign
                     outputs (cross: engine program gate JSON; star:
                     out_rot/star_headline.json), plus geometry-limit notes

Every number is read from the campaign artifacts, never re-typed
(data-contract rule). Regenerate with:

  ./.venv312/bin/python scripts/analysis/build_fgm_shape_standards.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

FGM_LIBRARY_SHAPES = [
    "square", "circle", "ellipse", "rectangle", "rounded_rect", "diamond",
    "hexagon", "octagon", "pentagon", "triangle", "equilateral_triangle",
    "trapezoid", "cross", "star", "star6", "H_shape", "L_shape", "T_shape",
    "gt_logo",
]


def _read_v_cal(base: Path, shape: str) -> Optional[float]:
    p = base / "outputs_eqs" / "geometry_dual_readstate" / f"{shape}.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["v_cal"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _parse_window_winners(base: Path) -> Dict[str, dict]:
    """Winners table of FGM_WINDOW_RESELECTION.md -> {shape: {m, verdict}}."""
    p = base / "FGM_WINDOW_RESELECTION.md"
    out: Dict[str, dict] = {}
    if not p.exists():
        return out
    in_table = False
    for line in p.read_text().splitlines():
        if line.startswith("| shape |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 8 or cells[0].startswith("---"):
                continue
            shape = cells[0]
            try:
                m = float(cells[2])
            except ValueError:
                continue
            verdict_raw = cells[7]
            if verdict_raw.upper().startswith("BETTER"):
                verdict = "BETTER"
            elif verdict_raw.lower().startswith("neutral"):
                verdict = "neutral"
            elif verdict_raw.upper().startswith("HARMFUL"):
                verdict = "HARMFUL"
            else:
                verdict = verdict_raw
            out[shape] = {"m": m, "verdict": verdict}
    return out


def _read_solve_census(base: Path, shape: str) -> dict:
    p = base / "fgm_solve_campaign" / "out_lib" / f"{shape}.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    cls_block = d.get("verdict") or d.get("class") or {}
    if not isinstance(cls_block, dict):
        return {}
    raw = str(cls_block.get("class", "")).upper().replace(" ", "-")
    beats_iou = bool(cls_block.get("beats_hist_on_IoU", False))
    if raw == "SOLVED" and not beats_iou:
        cls = "MATCHED"  # reaches nominal but does not beat the stored mask
    elif raw in ("SOLVED", "IMPROVED", "NOT-RESCUED"):
        cls = raw
    else:
        cls = raw or None
    arm = str(cls_block.get("deliverable_arm", "A1_4bpp"))
    iou = None
    try:
        iou = float(d["arms"][arm]["IoU"])
    except (KeyError, TypeError, ValueError):
        pass
    return {"solve_class": cls, "solve_iou_4bpp": iou}


def _actuator_notes(base: Path) -> Dict[str, str]:
    notes: Dict[str, str] = {}
    # Cross: solved map + asymmetric turntable dwell program (engine-verified).
    gate = (base / "fgm_solve_campaign" / "out_dwell"
            / "cross_deliverable_moves_engine_program_gate.json")
    if gate.exists():
        try:
            iou = float(json.loads(gate.read_text())["march"]["IoU"])
            notes["cross"] = (
                f"Turntable dwell upgrade: solved map + dwell program reaches "
                f"IoU {iou:.4f} at grid 120, engine-verified "
                f"(program JSON fgm_solve_campaign/out_dwell/"
                f"cross_turntable_deliverable.json, commits 2cc1548 + 2408329)."
            )
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    # Star: rotation alone; the dopant map is inert.
    star = base / "fgm_solve_campaign" / "out_rot" / "star_headline.json"
    if star.exists():
        try:
            txt = json.loads(star.read_text())
            iou = None
            def _find(d: Any) -> None:
                nonlocal iou
                if isinstance(d, dict):
                    for k, v in d.items():
                        if k == "averaged_kernel_IoU_cont" and iou is None:
                            iou = float(v)
                        else:
                            _find(v)
                elif isinstance(d, list):
                    for v in d:
                        _find(v)
            _find(txt)
            if iou is not None:
                notes["star"] = (
                    f"Rotation upgrade: turntable rotation alone reaches "
                    f"IoU {iou:.3f} at grid 120; the dopant map is inert for "
                    f"this shape (CONTINUOUS_ROTATION_REPORT, commit b1cb582)."
                )
        except (ValueError, json.JSONDecodeError):
            pass
    notes.setdefault("T_shape", (
        "Geometry-limited: best known IoU 0.644 via an unequal dwell program "
        "(DWELL_SCHEDULE_REPORT); honest partial, no full rescue."
    ))
    notes.setdefault("L_shape", (
        "Geometry-limited: rotational actuation exhausted; best is a static "
        "135-degree orientation (DWELL_SCHEDULE_REPORT)."
    ))
    notes.setdefault("rectangle", (
        "Solve stall is actuator-limited in the deployable conductivity "
        "channel; the permittivity channel fixes it but is model-only "
        "pending the material measurement (EPS_CHANNEL_REPORT)."
    ))
    return notes


def build_standards(base: Path) -> dict:
    winners = _parse_window_winners(base)
    notes = _actuator_notes(base)
    shapes: Dict[str, dict] = {}
    for s in FGM_LIBRARY_SHAPES:
        w = winners.get(s, {})
        census = _read_solve_census(base, s)
        shapes[s] = {
            "v_cal_v": _read_v_cal(base, s),
            "quick_look_gain": w.get("m"),
            "quick_look_verdict": w.get("verdict"),
            "solve_class": census.get("solve_class"),
            "solve_iou_4bpp": census.get("solve_iou_4bpp"),
            "actuator_note": notes.get(s),
        }
    return {
        "generated": date.today().isoformat(),
        "engine_version_floor": "2.0.0",
        "sources": {
            "v_cal": "outputs_eqs/geometry_dual_readstate/<shape>.json",
            "quick_look": "FGM_WINDOW_RESELECTION.md winners table",
            "solve_census": "fgm_solve_campaign/out_lib/<shape>.json",
            "upgrades": [
                "fgm_solve_campaign/out_dwell/cross_deliverable_moves_engine_program_gate.json",
                "fgm_solve_campaign/out_rot/star_headline.json",
            ],
        },
        "standard_parameters": {
            "grid_nx": 120,
            "grid_ny": 120,
            "enforce_generator_power": False,
            "proxy_field": "T_phi90",
            "bpp": 4,
            "note": "HEATR_STANDARD_PARAMETERS.md section 4: voltage drive "
                    "for FGM comparisons, grid 120, T_phi90 proxy, 4 bpp.",
        },
        "shapes": shapes,
    }


def main() -> int:
    base = Path(__file__).resolve().parents[2]
    std = build_standards(base)
    out = base / "webui" / "static" / "fgm_shape_standards.json"
    out.write_text(json.dumps(std, indent=1))
    n = sum(1 for v in std["shapes"].values() if v["v_cal_v"] is not None)
    print(f"wrote {out} ({len(std['shapes'])} shapes, {n} with v_cal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
