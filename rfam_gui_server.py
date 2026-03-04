#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import math
import fnmatch
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
import warnings
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import yaml
import numpy as np
from shapes import make_shapes_from_svg_paths

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "webui" / "static"
CONFIG_DIR = ROOT / "configs"
GUI_CONFIG_DIR = CONFIG_DIR / "_gui_generated"
OUTPUTS_DIR = ROOT / "outputs_eqs"
LOGS_DIR = OUTPUTS_DIR / "_logs"
PREVIEW_DIR = OUTPUTS_DIR / "_preview_cache"
RUN_STARS_FILE = OUTPUTS_DIR / "_run_stars.json"
MPLCONFIG_DIR = ROOT / ".mplconfig"
EXPERIMENTAL_DSC_PROFILE = CONFIG_DIR / "experimental_pa12_dsc_profile.yaml"
EXPERIMENTAL_PROVENANCE = CONFIG_DIR / "experimental_pa12_provenance.yaml"

GUI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

SUPPORTED_SHAPES = [
    "square",
    "rectangle",
    "circle",
    "ellipse",
    "triangle",
    "equilateral_triangle",
    "diamond",
    "hexagon",
    "octagon",
    "pentagon",
    "star",
    "star6",
    "star8",
    "gt_logo",
    "spacex_logo",
    "cross",
    "rounded_rect",
    "L_shape",
    "T_shape",
    "trapezoid",
]

DEFAULT_OPT_PHI = [0.50, 0.75, 0.90, 0.95]

JOBS_LOCK = threading.Lock()
JOBS: dict[str, dict[str, Any]] = {}
QUEUE_LOCK = threading.Lock()
JOB_QUEUE: list[str] = []
ACTIVE_JOB_ID: str | None = None
PROC_LOCK = threading.Lock()
RUNNING_PROCS: dict[str, subprocess.Popen] = {}
RUN_META_LOCK = threading.Lock()


class JobCancelled(Exception):
    pass


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _is_valid_output_name(name: str) -> bool:
    if not name:
        return False
    return all(ch.isalnum() or ch in "-_" for ch in name)


def _safe_rel_path(raw: str) -> Path | None:
    p = Path(unquote(raw)).as_posix().lstrip("/")
    if ".." in p.split("/"):
        return None
    full = (ROOT / p).resolve()
    try:
        full.relative_to(ROOT)
    except ValueError:
        return None
    return full


def _safe_workspace_path(raw: str) -> Path | None:
    rel = Path(unquote(raw or "")).as_posix()
    if rel in {"", ".", "/"}:
        rel = "."
    rel = rel.lstrip("/")
    if ".." in rel.split("/"):
        return None
    full = (ROOT / rel).resolve()
    try:
        full.relative_to(ROOT)
    except ValueError:
        return None
    return full


def _load_run_stars() -> dict[str, bool]:
    with RUN_META_LOCK:
        if not RUN_STARS_FILE.exists():
            return {}
        try:
            data = json.loads(RUN_STARS_FILE.read_text())
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[str, bool] = {}
        for k, v in data.items():
            name = str(k).strip().strip("/")
            if not name:
                continue
            out[name] = bool(v)
        return out


def _save_run_stars(stars: dict[str, bool]) -> None:
    with RUN_META_LOCK:
        clean = {str(k).strip().strip("/"): bool(v) for k, v in stars.items() if str(k).strip().strip("/")}
        RUN_STARS_FILE.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")


def _set_run_star(run_name: str, starred: bool) -> None:
    name = str(run_name).strip().strip("/")
    if not name:
        raise ValueError("missing run_name")
    stars = _load_run_stars()
    if starred:
        stars[name] = True
    else:
        stars.pop(name, None)
    _save_run_stars(stars)


def _to_float_list(vals: Any) -> list[float]:
    if not isinstance(vals, list):
        return []
    out: list[float] = []
    for v in vals:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


def _to_str_list(vals: Any) -> list[str]:
    if isinstance(vals, list):
        return [str(v).strip() for v in vals if str(v).strip()]
    if isinstance(vals, str):
        return [s.strip() for s in vals.split(",") if s.strip()]
    return []


def _r4(v: float) -> float:
    return round(float(v), 4)


def _base_config_paths() -> list[Path]:
    paths: list[Path] = []
    for p in CONFIG_DIR.glob("*.yaml"):
        n = p.name.lower()
        if "prewarp" in n:
            continue
        if n.startswith("_gui_"):
            continue
        paths.append(p)
    return sorted(paths)


def _list_base_configs() -> list[str]:
    return [p.name for p in _base_config_paths()]


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return data


def _infer_mode_from_cfg(cfg: dict[str, Any]) -> str:
    oo = cfg.get("orientation_optimizer", {})
    if isinstance(oo, dict) and bool(oo.get("enabled", False)):
        return "orientation_optimizer"
    po = cfg.get("placement_optimizer", {})
    if isinstance(po, dict) and bool(po.get("enabled", False)):
        return "placement_optimizer"
    opt = cfg.get("optimizer", {})
    if isinstance(opt, dict) and bool(opt.get("enabled", False)):
        return "optimizer"
    tt = cfg.get("turntable", {})
    if isinstance(tt, dict) and bool(tt.get("enabled", False)):
        return "turntable"
    return "single"


def _signature_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    mode = _infer_mode_from_cfg(cfg)
    part = cfg.get("geometry", {}).get("part", {})
    thermal = cfg.get("thermal", {})
    physics = cfg.get("physics_model", {}) if isinstance(cfg.get("physics_model", {}), dict) else {}
    dt_s = float(thermal.get("dt_s", 0.5))
    n_steps = float(thermal.get("n_steps", 0.0))
    exposure_minutes = _r4((dt_s * n_steps) / 60.0) if dt_s > 0 else 0.0

    sig: dict[str, Any] = {
        "mode": mode,
        "shape": str(part.get("shape", "")).strip(),
        "exposure_minutes": exposure_minutes,
        "model_family": str(physics.get("family", "baseline")).strip().lower() or "baseline",
    }

    if mode == "optimizer":
        opt = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer", {}), dict) else {}
        sig["phi_snapshots"] = [_r4(v) for v in _to_float_list(opt.get("phi_snapshots", []))]
        sig["temp_ceiling_c"] = _r4(float(opt.get("temp_ceiling_c", 250.0)))
        sig["highlight_phi"] = _r4(float(opt.get("highlight_phi", 0.90)))
    elif mode == "orientation_optimizer":
        oo = cfg.get("orientation_optimizer", {}) if isinstance(cfg.get("orientation_optimizer", {}), dict) else {}
        cst = oo.get("constraints", {}) if isinstance(oo.get("constraints", {}), dict) else {}
        sig["angle_min_deg"] = _r4(float(oo.get("angle_min_deg", 0.0)))
        sig["angle_max_deg"] = _r4(float(oo.get("angle_max_deg", 180.0)))
        sig["angle_step_deg"] = _r4(float(oo.get("angle_step_deg", 15.0)))
        sig["exposure_min_s"] = _r4(float(oo.get("exposure_min_s", 240.0)))
        sig["exposure_max_s"] = _r4(float(oo.get("exposure_max_s", 720.0)))
        sig["exposure_step_s"] = _r4(float(oo.get("exposure_step_s", 60.0)))
        sig["temp_ceiling_c"] = _r4(float(cst.get("temp_ceiling_c", 250.0)))
        sig["min_rho_floor"] = _r4(float(cst.get("min_rho_floor", 0.55)))
        sig["angle_gif_enabled"] = bool(oo.get("angle_gif_enabled", False))
        sig["angle_gif_frame_duration_s"] = _r4(float(oo.get("angle_gif_frame_duration_s", 2.0)))
        sig["color_metric"] = str(oo.get("color_metric", "mean_rho"))
    elif mode == "placement_optimizer":
        po = cfg.get("placement_optimizer", {}) if isinstance(cfg.get("placement_optimizer", {}), dict) else {}
        cst = po.get("constraints", {}) if isinstance(po.get("constraints", {}), dict) else {}
        ga = po.get("ga", {}) if isinstance(po.get("ga", {}), dict) else {}
        sig["algorithm"] = str(po.get("algorithm", "ga"))
        sig["n_parts"] = int(po.get("n_parts", 4))
        sig["part_width_mm"] = _r4(float(po.get("part_width_mm", 20.0)))
        sig["part_height_mm"] = _r4(float(po.get("part_height_mm", po.get("part_width_mm", 20.0))))
        sig["clearance_mm"] = _r4(float(po.get("clearance_mm", 1.0)))
        sig["search_domain_margin_mm"] = _r4(float(po.get("search_domain_margin_mm", 2.0)))
        sig["proxy_top_k"] = int(po.get("proxy_top_k", 8))
        sig["ga_population"] = int(ga.get("population", po.get("proxy_population", 32)))
        sig["ga_generations"] = int(ga.get("generations", po.get("proxy_iters", 8)))
        sig["ga_mutation_rate"] = _r4(float(ga.get("mutation_rate", 0.18)))
        sig["ga_crossover_rate"] = _r4(float(ga.get("crossover_rate", 0.75)))
        sig["ga_elitism"] = int(ga.get("elitism", 2))
        sig["ga_tournament_k"] = int(ga.get("tournament_k", 3))
        sig["ga_seed"] = int(ga.get("seed", po.get("seed", 7)))
        sig["proxy_eval_budget"] = int(po.get("proxy_eval_budget", 96))
        sig["use_turntable"] = bool(po.get("use_turntable", False))
        if sig["use_turntable"]:
            tt = cfg.get("turntable", {}) if isinstance(cfg.get("turntable", {}), dict) else {}
            sig["turntable_rotation_deg"] = _r4(float(tt.get("rotation_deg", 90.0)))
            sig["turntable_total_rotations"] = int(tt.get("total_rotations", 1))
            if tt.get("rotation_interval_s", None) is not None:
                sig["turntable_interval_s"] = _r4(float(tt.get("rotation_interval_s")))
            else:
                sig["turntable_interval_s"] = None
        sig["temp_ceiling_c"] = _r4(float(cst.get("temp_ceiling_c", 250.0)))
        sig["min_rho_floor"] = _r4(float(cst.get("min_rho_floor", 0.55)))
    elif mode == "turntable":
        tt = cfg.get("turntable", {}) if isinstance(cfg.get("turntable", {}), dict) else {}
        if "rotation_deg" in tt or "total_rotations" in tt or "rotation_interval_s" in tt:
            sig["turntable_rotation_deg"] = _r4(float(tt.get("rotation_deg", 90.0)))
            sig["turntable_total_rotations"] = int(tt.get("total_rotations", 1))
            if tt.get("rotation_interval_s", None) is not None:
                sig["turntable_interval_s"] = _r4(float(tt.get("rotation_interval_s")))
            else:
                sig["turntable_interval_s"] = None
        else:
            # Backward-compatible mapping from legacy phases to the new time-based signature.
            phases = tt.get("phases", []) if isinstance(tt.get("phases", []), list) else []
            phase_angles: list[float] = []
            for ph in phases:
                if not isinstance(ph, dict):
                    continue
                if "angle_deg" in ph:
                    phase_angles.append(float(ph["angle_deg"]))
            event_angles = phase_angles[1:]
            deltas: list[float] = []
            prev = 0.0
            for ang in event_angles:
                deltas.append(float(ang) - prev)
                prev = float(ang)
            if deltas:
                sig["turntable_rotation_deg"] = _r4(float(deltas[0]))
                sig["turntable_total_rotations"] = int(len(deltas))
                total_s = n_steps * dt_s
                sig["turntable_interval_s"] = _r4(total_s / max(len(deltas), 1))
            else:
                sig["turntable_rotation_deg"] = 90.0
                sig["turntable_total_rotations"] = 1
                sig["turntable_interval_s"] = None

    return sig


def _signature_for_request(payload: dict[str, Any], mode: str, exposure_minutes: float) -> dict[str, Any]:
    pm = payload.get("physics_model", {}) if isinstance(payload.get("physics_model", {}), dict) else {}
    sig: dict[str, Any] = {
        "mode": mode,
        "shape": str(payload.get("shape", "")).strip(),
        "exposure_minutes": _r4(exposure_minutes),
        "model_family": str(pm.get("family", "baseline")).strip().lower() or "baseline",
    }
    if bool(payload.get("geometry_size_enabled", False)):
        sig["geometry_size_mm"] = _r4(float(payload.get("geometry_size_mm", 0.0)))
        sig["geometry_nominal_mm"] = _r4(float(payload.get("geometry_nominal_mm", 0.0)))
        sig["geometry_lock_aspect"] = bool(payload.get("geometry_lock_aspect", True))
    if mode == "optimizer":
        phi = _to_float_list(payload.get("phi_snapshots", DEFAULT_OPT_PHI))
        if not phi:
            phi = list(DEFAULT_OPT_PHI)
        sig["phi_snapshots"] = [_r4(v) for v in phi]
        sig["temp_ceiling_c"] = _r4(float(payload.get("temp_ceiling_c", 250.0)))
        sig["highlight_phi"] = _r4(float(payload.get("highlight_phi", 0.90)))
    elif mode == "orientation_optimizer":
        sig["angle_min_deg"] = _r4(float(payload.get("orientation_angle_min_deg", 0.0)))
        sig["angle_max_deg"] = _r4(float(payload.get("orientation_angle_max_deg", 180.0)))
        sig["angle_step_deg"] = _r4(float(payload.get("orientation_angle_step_deg", 15.0)))
        sig["exposure_min_s"] = _r4(float(payload.get("orientation_exposure_min_s", 240.0)))
        sig["exposure_max_s"] = _r4(float(payload.get("orientation_exposure_max_s", 720.0)))
        sig["exposure_step_s"] = _r4(float(payload.get("orientation_exposure_step_s", 60.0)))
        sig["temp_ceiling_c"] = _r4(float(payload.get("orientation_temp_ceiling_c", 250.0)))
        sig["min_rho_floor"] = _r4(float(payload.get("orientation_min_rho_floor", 0.55)))
        sig["angle_gif_enabled"] = bool(payload.get("orientation_angle_gif_enabled", False))
        sig["angle_gif_frame_duration_s"] = _r4(float(payload.get("orientation_angle_gif_frame_duration_s", 2.0)))
        sig["color_metric"] = str(payload.get("orientation_color_metric", "mean_rho"))
    elif mode == "placement_optimizer":
        sig["algorithm"] = str(payload.get("placement_algorithm", "ga"))
        sig["n_parts"] = int(payload.get("placement_n_parts", 4))
        sig["part_width_mm"] = _r4(float(payload.get("placement_part_width_mm", 20.0)))
        sig["part_height_mm"] = _r4(float(payload.get("placement_part_height_mm", payload.get("placement_part_width_mm", 20.0))))
        sig["clearance_mm"] = _r4(float(payload.get("placement_clearance_mm", 1.0)))
        sig["search_domain_margin_mm"] = _r4(float(payload.get("placement_search_domain_margin_mm", 2.0)))
        sig["proxy_top_k"] = int(payload.get("placement_proxy_top_k", 8))
        sig["ga_population"] = int(payload.get("placement_ga_population", payload.get("placement_proxy_population", 32)))
        sig["ga_generations"] = int(payload.get("placement_ga_generations", payload.get("placement_proxy_iters", 8)))
        sig["ga_mutation_rate"] = _r4(float(payload.get("placement_ga_mutation_rate", 0.18)))
        sig["ga_crossover_rate"] = _r4(float(payload.get("placement_ga_crossover_rate", 0.75)))
        sig["ga_elitism"] = int(payload.get("placement_ga_elitism", 2))
        sig["ga_tournament_k"] = int(payload.get("placement_ga_tournament_k", 3))
        sig["ga_seed"] = int(payload.get("placement_ga_seed", payload.get("placement_seed", 7)))
        sig["proxy_eval_budget"] = int(payload.get("placement_proxy_eval_budget", 96))
        sig["use_turntable"] = bool(payload.get("placement_use_turntable", False))
        if sig["use_turntable"]:
            sig["turntable_rotation_deg"] = _r4(float(payload.get("placement_turntable_rotation_deg", 90.0)))
            sig["turntable_total_rotations"] = int(payload.get("placement_turntable_total_rotations", 1))
            interval_raw = payload.get("placement_turntable_interval_s", None)
            if interval_raw in ("", None):
                sig["turntable_interval_s"] = None
            else:
                sig["turntable_interval_s"] = _r4(float(interval_raw))
        sig["temp_ceiling_c"] = _r4(float(payload.get("placement_temp_ceiling_c", 250.0)))
        sig["min_rho_floor"] = _r4(float(payload.get("placement_min_rho_floor", 0.55)))
    elif mode == "turntable":
        sig["turntable_rotation_deg"] = _r4(float(payload.get("turntable_rotation_deg", 90.0)))
        sig["turntable_total_rotations"] = int(payload.get("turntable_total_rotations", 1))
        interval_raw = payload.get("turntable_interval_s", None)
        if interval_raw in ("", None):
            sig["turntable_interval_s"] = None
        else:
            sig["turntable_interval_s"] = _r4(float(interval_raw))
    return sig


def _signatures_match(cfg_sig: dict[str, Any], req_sig: dict[str, Any]) -> bool:
    if cfg_sig.get("mode") != req_sig.get("mode"):
        return False
    if str(cfg_sig.get("shape")) != str(req_sig.get("shape")):
        return False
    if abs(float(cfg_sig.get("exposure_minutes", 0.0)) - float(req_sig.get("exposure_minutes", 0.0))) > 1e-3:
        return False
    if str(cfg_sig.get("model_family", "baseline")) != str(req_sig.get("model_family", "baseline")):
        return False

    mode = str(req_sig.get("mode"))
    if mode == "optimizer":
        if cfg_sig.get("phi_snapshots", []) != req_sig.get("phi_snapshots", []):
            return False
        if abs(float(cfg_sig.get("temp_ceiling_c", 0.0)) - float(req_sig.get("temp_ceiling_c", 0.0))) > 1e-3:
            return False
        if abs(float(cfg_sig.get("highlight_phi", 0.0)) - float(req_sig.get("highlight_phi", 0.0))) > 1e-3:
            return False
    elif mode in {"orientation_optimizer", "placement_optimizer"}:
        for key, val in req_sig.items():
            if key in {"mode", "shape", "exposure_minutes", "model_family"}:
                continue
            if cfg_sig.get(key) != val:
                return False
    elif mode == "turntable":
        if abs(float(cfg_sig.get("turntable_rotation_deg", 0.0)) - float(req_sig.get("turntable_rotation_deg", 0.0))) > 1e-3:
            return False
        if int(cfg_sig.get("turntable_total_rotations", 0)) != int(req_sig.get("turntable_total_rotations", 0)):
            return False
        cti = cfg_sig.get("turntable_interval_s", None)
        rti = req_sig.get("turntable_interval_s", None)
        # If interval is omitted in request, treat it as auto and do not constrain matching.
        if rti is not None:
            if cti is None:
                return False
            if abs(float(cti) - float(rti)) > 1e-3:
                return False

    for key in ("geometry_size_mm", "geometry_nominal_mm", "geometry_lock_aspect"):
        if key in req_sig and cfg_sig.get(key) != req_sig.get(key):
            return False

    return True


def _preferred_fallback_name(shape: str, mode: str) -> list[str]:
    names: list[str] = []
    s = str(shape).strip().lower()
    if s in {"gt_logo", "spacex_logo"}:
        # Logo imports should inherit the canonical 20 mm baseline unless a user
        # explicitly selects a logo-specific config.
        names.append("rfam_eqs_xz_uniform_500w.yaml")
        names.append("rfam_eqs_comsol_qrf_driven.yaml")
        return names
    if mode == "optimizer":
        names.append(f"shape_{shape}_12min_optimizer.yaml")
        names.append(f"shape_{shape}_6min.yaml")
    elif mode == "turntable":
        names.append(f"shape_{shape}_6min_turntable.yaml")
        names.append(f"shape_{shape}_6min.yaml")
    elif mode in {"orientation_optimizer", "placement_optimizer"}:
        names.append(f"shape_{shape}_6min.yaml")
    else:
        names.append(f"shape_{shape}_6min.yaml")

    names.append("rfam_eqs_xz_uniform_500w.yaml")
    names.append("rfam_eqs_comsol_qrf_driven.yaml")
    return names


def _resolve_base_config_for_signature(req_sig: dict[str, Any], forced_base_config: str | None = None) -> dict[str, Any]:
    candidates = _base_config_paths()

    if forced_base_config:
        forced = (CONFIG_DIR / forced_base_config).resolve()
        if not forced.exists() or forced.parent != CONFIG_DIR:
            raise ValueError("invalid base_config")
        if "prewarp" in forced.name.lower():
            raise ValueError("prewarp configs are disabled in this GUI")
        return {"match_type": "forced", "config_name": forced.name, "config_path": str(forced)}

    for p in candidates:
        cfg = _load_yaml(p)
        cfg_sig = _signature_from_cfg(cfg)
        if _signatures_match(cfg_sig, req_sig):
            return {"match_type": "matched_existing", "config_name": p.name, "config_path": str(p)}

    pref = _preferred_fallback_name(str(req_sig.get("shape", "")), str(req_sig.get("mode", "single")))
    by_name = {p.name: p for p in candidates}
    for n in pref:
        if n in by_name:
            p = by_name[n]
            return {"match_type": "new_from_fallback", "config_name": p.name, "config_path": str(p)}

    if not candidates:
        raise ValueError("no base configs found")
    p = candidates[0]
    return {"match_type": "new_from_fallback", "config_name": p.name, "config_path": str(p)}


def _config_preview(name: str) -> dict[str, Any]:
    cfg_path = (CONFIG_DIR / name).resolve()
    if not cfg_path.exists() or cfg_path.parent != CONFIG_DIR or cfg_path.suffix.lower() not in {".yaml", ".yml"}:
        raise FileNotFoundError(name)
    if "prewarp" in cfg_path.name.lower():
        raise FileNotFoundError(name)

    text = cfg_path.read_text()
    cfg = _load_yaml(cfg_path)
    part = cfg.get("geometry", {}).get("part", {})
    thermal = cfg.get("thermal", {})
    electric = cfg.get("electric", {})
    optimizer = cfg.get("optimizer", {})
    turntable = cfg.get("turntable", {})
    orientation_optimizer = cfg.get("orientation_optimizer", {})
    placement_optimizer = cfg.get("placement_optimizer", {})
    antennae = cfg.get("antennae", {}) if isinstance(cfg.get("antennae", {}), dict) else {}
    spike = cfg.get("spike", {}) if isinstance(cfg.get("spike", {}), dict) else {}
    shell = cfg.get("geometry", {}).get("shell", {}) if isinstance(cfg.get("geometry", {}).get("shell", {}), dict) else {}
    materials = cfg.get("materials", {}) if isinstance(cfg.get("materials", {}), dict) else {}
    physics = cfg.get("physics_model", {}) if isinstance(cfg.get("physics_model", {}), dict) else {}
    phase_model = cfg.get("phase_model", {}) if isinstance(cfg.get("phase_model", {}), dict) else {}
    dens_model = cfg.get("dens_model", {}) if isinstance(cfg.get("dens_model", {}), dict) else {}
    crystallization_model = cfg.get("crystallization_model", {}) if isinstance(cfg.get("crystallization_model", {}), dict) else {}
    virgin = materials.get("virgin", {}) if isinstance(materials.get("virgin", {}), dict) else {}
    powder = materials.get("powder", {}) if isinstance(materials.get("powder", {}), dict) else {}
    doped = cfg.get("materials", {}).get("doped", {})
    dt_s = float(thermal.get("dt_s", 0.5))
    n_steps = float(thermal.get("n_steps", 0.0))
    exposure_minutes = (dt_s * n_steps / 60.0) if dt_s > 0 else None

    important = {
        "shape": part.get("shape"),
        "grid_nx": cfg.get("geometry", {}).get("grid_nx"),
        "grid_ny": cfg.get("geometry", {}).get("grid_ny"),
        "dt_s": thermal.get("dt_s"),
        "n_steps": thermal.get("n_steps"),
        "exposure_minutes": round(exposure_minutes, 3) if exposure_minutes is not None else None,
        "voltage_v": electric.get("voltage_v"),
        "frequency_hz": electric.get("frequency_hz"),
        "enforce_generator_power": electric.get("enforce_generator_power"),
        "generator_power_w": electric.get("generator_power_w"),
        "generator_transfer_efficiency": electric.get("generator_transfer_efficiency"),
        "ambient_c": thermal.get("ambient_c"),
        "sigma_s_per_m": doped.get("sigma_s_per_m"),
        "virgin_sigma_s_per_m": virgin.get("sigma_s_per_m"),
        "virgin_eps_r": virgin.get("eps_r"),
        "powder_rho_solid_kg_per_m3": powder.get("rho_solid_kg_per_m3"),
        "doped_eps_r": doped.get("eps_r"),
        "optimizer_enabled": bool(optimizer.get("enabled", False)) if isinstance(optimizer, dict) else False,
        "orientation_optimizer_enabled": bool(orientation_optimizer.get("enabled", False)) if isinstance(orientation_optimizer, dict) else False,
        "placement_optimizer_enabled": bool(placement_optimizer.get("enabled", False)) if isinstance(placement_optimizer, dict) else False,
        "antennae_enabled": bool(antennae.get("enabled", spike.get("enabled", False))),
        "shell_enabled": bool(shell.get("enabled", False)) if isinstance(shell, dict) else False,
        "shell_wall_thickness_mm": shell.get("wall_thickness_mm") if isinstance(shell, dict) else None,
        "shell_method": shell.get("method") if isinstance(shell, dict) else None,
        "turntable_enabled": bool(turntable.get("enabled", False)) if isinstance(turntable, dict) else False,
        "turntable_rotation_deg": turntable.get("rotation_deg") if isinstance(turntable, dict) else None,
        "turntable_total_rotations": turntable.get("total_rotations") if isinstance(turntable, dict) else None,
        "turntable_interval_s": turntable.get("rotation_interval_s") if isinstance(turntable, dict) else None,
        "model_family": physics.get("family", "baseline"),
        "experimental_enabled": physics.get("experimental_enabled", False),
        "provenance_tag": physics.get("provenance_tag", ""),
        "phase_model_type": phase_model.get("type"),
        "dens_model_type": dens_model.get("type"),
        "crystallization_enabled": crystallization_model.get("enabled", False),
    }
    return {"name": cfg_path.name, "text": text, "important": important}


def _set_exposure_minutes(cfg: dict[str, Any], minutes: float) -> None:
    thermal = cfg.setdefault("thermal", {})
    dt_s = float(thermal.get("dt_s", 0.5))
    if dt_s <= 0:
        raise ValueError("thermal.dt_s must be > 0")
    n_steps = max(1, int(round((minutes * 60.0) / dt_s)))
    thermal["n_steps"] = n_steps


def _apply_geometry_size(cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    if not bool(payload.get("geometry_size_enabled", False)):
        return
    size_mm = float(payload.get("geometry_size_mm", 0.0))
    nominal_mm = float(payload.get("geometry_nominal_mm", 0.0))
    lock_aspect = bool(payload.get("geometry_lock_aspect", True))
    if size_mm <= 0.0:
        raise ValueError("geometry_size_mm must be > 0 when geometry_size_enabled is true")

    geom = cfg.setdefault("geometry", {})
    part = geom.setdefault("part", {})
    w0 = float(part.get("width", 0.02))
    h0 = float(part.get("height", w0))
    if w0 <= 0.0 or h0 <= 0.0:
        w0, h0 = 0.02, 0.02

    if nominal_mm > 0.0:
        scale = size_mm / nominal_mm
        w_new = w0 * scale
    else:
        w_new = size_mm * 1e-3
        scale = w_new / max(w0, 1e-12)

    if lock_aspect:
        h_new = h0 * scale
    else:
        h_new = w_new

    part["width"] = float(w_new)
    part["height"] = float(h_new)


def _set_shape(cfg: dict[str, Any], shape: str) -> None:
    geom = cfg.setdefault("geometry", {})
    part = geom.setdefault("part", {})
    s = str(shape).strip().lower()
    if s in {"gt_logo", "spacex_logo"}:
        svg_name = "Georgia_Tech_Yellow_Jackets_logo.svg" if s == "gt_logo" else "spacex-svg.svg"
        target_w = float(part.get("width", 0.020))
        target_h = float(part.get("height", target_w))
        cx = float(part.get("center_x", 0.0))
        cy = float(part.get("center_y", 0.0))
        rot = float(part.get("rotation_deg", 0.0))
        polys = make_shapes_from_svg_paths(
            str((ROOT / svg_name).resolve()),
            target_width_m=target_w,
            target_height_m=target_h,
            n_pts_each=2200,
            curve_pts=96,
            min_span_frac=0.0005,
        )

        # Logo strokes alias heavily on 120x120 (0.5 mm/cell over 60 mm chamber).
        # Promote to a finer default grid so curves remain visible in field plots.
        try:
            gx = int(geom.get("grid_nx", 120))
            gy = int(geom.get("grid_ny", 120))
        except Exception:
            gx, gy = 120, 120
        min_logo_grid = 280
        if gx < min_logo_grid:
            geom["grid_nx"] = min_logo_grid
        if gy < min_logo_grid:
            geom["grid_ny"] = min_logo_grid

        parts: list[dict[str, Any]] = []
        for p in polys:
            parts.append(
                {
                    "shape": "polygon",
                    "width": target_w,
                    "height": target_h,
                    "center_x": cx,
                    "center_y": cy,
                    "rotation_deg": rot,
                    "n_circle_pts": int(part.get("n_circle_pts", 360)),
                    "polygon_points": [[float(x), float(y)] for x, y in np.asarray(p, dtype=float)],
                }
            )
        geom["parts"] = parts
        geom["boolean"] = {"mode": "union"}
        part["shape"] = s
        part["width"] = target_w
        part["height"] = target_h
        part["center_x"] = cx
        part["center_y"] = cy
        part["rotation_deg"] = rot
        return
    geom.pop("parts", None)
    geom.pop("boolean", None)
    part["shape"] = s


def _disable_prewarp_like_blocks(cfg: dict[str, Any]) -> None:
    cfg.pop("prewarp", None)
    cfg.pop("evaluation", None)


def _advanced_payload(payload: dict[str, Any]) -> dict[str, Any]:
    adv = payload.get("advanced", {})
    return adv if isinstance(adv, dict) else {}


def _has_advanced_overrides(payload: dict[str, Any]) -> bool:
    return any(str(v).strip() != "" for v in _advanced_payload(payload).values())


def _has_shell_overrides(payload: dict[str, Any]) -> bool:
    if str(payload.get("mode", "single")).strip() == "shell_sweep":
        return True
    if not bool(payload.get("shell_enabled", False)):
        return False
    try:
        return float(payload.get("shell_wall_thickness_mm", 0.0)) > 0.0
    except (TypeError, ValueError):
        return False


def _has_geometry_size_overrides(payload: dict[str, Any]) -> bool:
    return bool(payload.get("geometry_size_enabled", False))


def _has_antennae_overrides(payload: dict[str, Any]) -> bool:
    keys = {
        "antennae_enabled",
        "spike_enabled",
        "antennae_size_mode",
        "antennae_global_size_mm",
        "antennae_auto_min_mm",
        "antennae_auto_max_mm",
        "antennae_auto_qrf_percentile_low",
        "antennae_auto_qrf_percentile_high",
        "antennae_max_per_part",
        "antennae_min_spacing_mm",
        "antennae_edge_margin_mm",
    }
    return any(k in payload for k in keys)


def _experimental_payload(payload: dict[str, Any]) -> dict[str, Any]:
    pm = payload.get("physics_model", {})
    return pm if isinstance(pm, dict) else {}


def _is_experimental_request(payload: dict[str, Any]) -> bool:
    pm = _experimental_payload(payload)
    fam = str(pm.get("family", "baseline")).strip().lower() or "baseline"
    if fam in {"dsc_pa12_hybrid", "dsc_calibrated_pa12"}:
        fam = "experimental_pa12_hybrid"
    enabled = bool(pm.get("experimental_enabled", False))
    return fam != "baseline" and enabled


def _apply_experimental_bucket(cfg: dict[str, Any], payload: dict[str, Any]) -> bool:
    pm = _experimental_payload(payload)
    family = str(pm.get("family", "baseline")).strip().lower() or "baseline"
    if family in {"dsc_pa12_hybrid", "dsc_calibrated_pa12"}:
        family = "experimental_pa12_hybrid"
    enabled = bool(pm.get("experimental_enabled", False))

    if family == "baseline":
        cfg["physics_model"] = {
            "family": "baseline",
            "experimental_enabled": False,
            "provenance_tag": str(pm.get("provenance_tag", "")).strip(),
            "parameter_source": str(pm.get("parameter_source", "baseline")).strip() or "baseline",
            "calibration_version": str(pm.get("calibration_version", "baseline")).strip() or "baseline",
            "ab_bucket_id": str(pm.get("ab_bucket_id", "")).strip(),
            "literature_refs": [],
        }
        cfg.pop("phase_model", None)
        cfg.pop("dens_model", None)
        cfg.pop("crystallization_model", None)
        return False

    if not enabled:
        raise ValueError("experimental model family requires experimental_enabled=true")

    phase_model = payload.get("phase_model", {})
    dens_model = payload.get("dens_model", {})
    crystallization_model = payload.get("crystallization_model", {})
    if not isinstance(phase_model, dict):
        phase_model = {}
    if not isinstance(dens_model, dict):
        dens_model = {}
    if not isinstance(crystallization_model, dict):
        crystallization_model = {}

    cfg["physics_model"] = {
        "family": family,
        "experimental_enabled": True,
        "provenance_tag": str(pm.get("provenance_tag", "")).strip(),
        "parameter_source": str(pm.get("parameter_source", "literature+calibrated")).strip() or "literature+calibrated",
        "calibration_version": str(pm.get("calibration_version", "exp-pa12-v1")).strip() or "exp-pa12-v1",
        "ab_bucket_id": str(pm.get("ab_bucket_id", "")).strip(),
        "provenance_file": str(pm.get("provenance_file", "configs/experimental_pa12_provenance.yaml")).strip()
        or "configs/experimental_pa12_provenance.yaml",
        "dsc_profile_file": str(pm.get("dsc_profile_file", "configs/experimental_pa12_dsc_profile.yaml")).strip()
        or "configs/experimental_pa12_dsc_profile.yaml",
        "literature_refs": pm.get("literature_refs", []),
    }
    cfg["phase_model"] = phase_model
    cfg["dens_model"] = dens_model
    cfg["crystallization_model"] = crystallization_model
    return True


def _effective_config_for_request(payload: dict[str, Any], mode: str, exposure_minutes: float) -> dict[str, Any]:
    forced = str(payload.get("base_config", "")).strip() or None
    req_sig = _signature_for_request(payload, mode, exposure_minutes)
    resolved = _resolve_base_config_for_signature(req_sig, forced_base_config=forced)
    base_path = Path(resolved["config_path"])
    cfg = _load_yaml(base_path)
    _disable_prewarp_like_blocks(cfg)
    _apply_geometry_size(cfg, payload)
    _set_shape(cfg, str(payload.get("shape", "")).strip())
    _apply_advanced_overrides(cfg, payload)
    _apply_antennae_config(cfg, payload)
    _apply_shell_config(cfg, payload)
    experimental_active = _apply_experimental_bucket(cfg, payload)
    _set_exposure_minutes(cfg, exposure_minutes)
    if mode == "optimizer":
        _configure_optimizer(cfg, payload)
    elif mode == "orientation_optimizer":
        _configure_orientation_optimizer(cfg, payload)
    elif mode == "placement_optimizer":
        _configure_placement_optimizer(cfg, payload)
    elif mode == "turntable":
        _configure_turntable(cfg, payload)
    else:
        _configure_single(cfg)

    important = _config_preview(base_path.name).get("important", {})
    physics = cfg.get("physics_model", {}) if isinstance(cfg.get("physics_model", {}), dict) else {}
    phase_model = cfg.get("phase_model", {}) if isinstance(cfg.get("phase_model", {}), dict) else {}
    important.update({
        "shape": cfg.get("geometry", {}).get("part", {}).get("shape"),
        "exposure_minutes": exposure_minutes,
        "model_family": physics.get("family", "baseline"),
        "experimental_enabled": physics.get("experimental_enabled", False),
        "phase_model_type": phase_model.get("type", cfg.get("thermal", {}).get("phase_change", {}).get("model")),
        "melt_onset_c": phase_model.get("melt_onset_c"),
        "melt_peak_c": phase_model.get("melt_peak_c"),
        "melt_end_c": phase_model.get("melt_end_c"),
        "lat_heat_j_per_kg": phase_model.get("lat_heat_j_per_kg"),
    })

    return {
        "mode": mode,
        "requested": req_sig,
        "resolved": resolved,
        "experimental_active": experimental_active,
        "effective_config_text": yaml.safe_dump(cfg, sort_keys=False),
        "important": important,
    }


def _apply_advanced_overrides(cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    adv = _advanced_payload(payload)

    def _num(key: str) -> float | None:
        v = adv.get(key, None)
        if v is None or str(v).strip() == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _intval(key: str) -> int | None:
        v = _num(key)
        if v is None:
            return None
        return int(round(v))

    def _bool(key: str) -> bool | None:
        v = adv.get(key, None)
        if v is None:
            return None
        s = str(v).strip().lower()
        if s == "":
            return None
        if s in {"true", "1", "yes", "on"}:
            return True
        if s in {"false", "0", "no", "off"}:
            return False
        return None

    def _text(key: str) -> str | None:
        v = adv.get(key, None)
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    g = cfg.setdefault("geometry", {})
    e = cfg.setdefault("electric", {})
    t = cfg.setdefault("thermal", {})
    mats = cfg.setdefault("materials", {})
    virgin = mats.setdefault("virgin", {})
    powder = mats.setdefault("powder", {})
    doped = mats.setdefault("doped", {})

    if (v := _intval("grid_nx")) and v > 8:
        g["grid_nx"] = v
    if (v := _intval("grid_ny")) and v > 8:
        g["grid_ny"] = v

    if (v := _num("frequency_hz")) and v > 0:
        e["frequency_hz"] = v
    if (v := _num("voltage_v")) and v > 0:
        e["voltage_v"] = v
    if (v := _num("generator_power_w")) and v > 0:
        e["generator_power_w"] = v
    if (v := _bool("enforce_generator_power")) is not None:
        e["enforce_generator_power"] = v
    if (v := _num("generator_transfer_efficiency")) is not None and 0 < v <= 1:
        e["generator_transfer_efficiency"] = v
    if (v := _num("effective_depth_m")) and v > 0:
        e["effective_depth_m"] = v
    if (v := _num("max_qrf_w_per_m3")) and v > 0:
        e["max_qrf_w_per_m3"] = v

    if (v := _num("dt_s")) and v > 0:
        t["dt_s"] = v
    if (v := _num("ambient_c")) is not None:
        t["ambient_c"] = v
    if (v := _num("convection_h_w_per_m2k")) and v >= 0:
        t["convection_h_w_per_m2k"] = v
    if (v := _num("max_temp_c")) and v > 0:
        t["max_temp_c"] = v

    if (v := _num("virgin_sigma_s_per_m")) is not None and v >= 0:
        virgin["sigma_s_per_m"] = v
    if (v := _num("virgin_eps_r")) is not None and v >= 0:
        virgin["eps_r"] = v

    if (v := _num("powder_rho_solid_kg_per_m3")) is not None and v > 0:
        powder["rho_solid_kg_per_m3"] = v
    if (v := _num("powder_k_solid_w_per_mk")) is not None and v >= 0:
        powder["k_solid_w_per_mk"] = v
    if (v := _num("powder_cp_solid_j_per_kgk")) is not None and v > 0:
        powder["cp_solid_j_per_kgk"] = v

    if (v := _num("sigma_s_per_m")) is not None and v >= 0:
        doped["sigma_s_per_m"] = v
    if (v := _text("sigma_profile")) is not None:
        doped["sigma_profile"] = v
    if (v := _num("doped_eps_r")) is not None and v >= 0:
        doped["eps_r"] = v
    if (v := _num("doped_sigma_temp_coeff_per_K")) is not None:
        doped["sigma_temp_coeff_per_K"] = v
    if (v := _num("doped_sigma_density_coeff")) is not None:
        doped["sigma_density_coeff"] = v
    if (v := _num("doped_sigma_ref_temp_c")) is not None:
        doped["sigma_ref_temp_c"] = v
    if (v := _num("doped_rho_solid_kg_per_m3")) is not None and v > 0:
        doped["rho_solid_kg_per_m3"] = v
    if (v := _num("doped_rho_liquid_kg_per_m3")) is not None and v > 0:
        doped["rho_liquid_kg_per_m3"] = v
    if (v := _num("doped_k_solid_w_per_mk")) is not None and v >= 0:
        doped["k_solid_w_per_mk"] = v
    if (v := _num("doped_k_liquid_w_per_mk")) is not None and v >= 0:
        doped["k_liquid_w_per_mk"] = v
    if (v := _num("doped_cp_solid_j_per_kgk")) is not None and v > 0:
        doped["cp_solid_j_per_kgk"] = v
    if (v := _num("doped_cp_liquid_j_per_kgk")) is not None and v > 0:
        doped["cp_liquid_j_per_kgk"] = v


def _configure_optimizer(cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    phi = _to_float_list(payload.get("phi_snapshots", DEFAULT_OPT_PHI))
    if not phi:
        phi = list(DEFAULT_OPT_PHI)
    cfg["optimizer"] = {
        "enabled": True,
        "phi_snapshots": phi,
        "temp_ceiling_c": float(payload.get("temp_ceiling_c", 250.0)),
        "highlight_phi": float(payload.get("highlight_phi", 0.90)),
    }
    cfg.pop("turntable", None)
    cfg.pop("orientation_optimizer", None)
    cfg.pop("placement_optimizer", None)


def _configure_orientation_optimizer(cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    cfg["orientation_optimizer"] = {
        "enabled": True,
        "angle_min_deg": float(payload.get("orientation_angle_min_deg", 0.0)),
        "angle_max_deg": float(payload.get("orientation_angle_max_deg", 180.0)),
        "angle_step_deg": float(payload.get("orientation_angle_step_deg", 15.0)),
        "refine_window_deg": float(payload.get("orientation_refine_window_deg", 10.0)),
        "refine_step_deg": float(payload.get("orientation_refine_step_deg", 5.0)),
        "exposure_min_s": float(payload.get("orientation_exposure_min_s", 240.0)),
        "exposure_max_s": float(payload.get("orientation_exposure_max_s", 720.0)),
        "exposure_step_s": float(payload.get("orientation_exposure_step_s", 60.0)),
        "angle_gif_enabled": bool(payload.get("orientation_angle_gif_enabled", False)),
        "angle_gif_frame_duration_s": float(payload.get("orientation_angle_gif_frame_duration_s", 2.0)),
        "color_metric": str(payload.get("orientation_color_metric", "mean_rho")),
        "objective": "pareto",
        "constraints": {
            "temp_ceiling_c": float(payload.get("orientation_temp_ceiling_c", 250.0)),
            "min_rho_floor": float(payload.get("orientation_min_rho_floor", 0.55)),
        },
    }
    cfg.pop("turntable", None)
    cfg.pop("optimizer", None)
    cfg.pop("placement_optimizer", None)


def _configure_placement_optimizer(cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    part_w_mm = float(payload.get("placement_part_width_mm", 20.0))
    part_h_mm = float(payload.get("placement_part_height_mm", part_w_mm))
    use_tt = bool(payload.get("placement_use_turntable", False))
    ga = {
        "population": int(payload.get("placement_ga_population", payload.get("placement_proxy_population", 32))),
        "generations": int(payload.get("placement_ga_generations", payload.get("placement_proxy_iters", 8))),
        "crossover_rate": float(payload.get("placement_ga_crossover_rate", 0.75)),
        "mutation_rate": float(payload.get("placement_ga_mutation_rate", 0.18)),
        "elitism": int(payload.get("placement_ga_elitism", 2)),
        "tournament_k": int(payload.get("placement_ga_tournament_k", 3)),
        "seed": int(payload.get("placement_ga_seed", payload.get("placement_seed", 7))),
    }
    cfg["placement_optimizer"] = {
        "enabled": True,
        "algorithm": str(payload.get("placement_algorithm", "ga")),
        "n_parts": int(payload.get("placement_n_parts", 4)),
        "part_width_mm": float(part_w_mm),
        "part_height_mm": float(part_h_mm),
        "clearance_mm": float(payload.get("placement_clearance_mm", 1.0)),
        "search_domain_margin_mm": float(payload.get("placement_search_domain_margin_mm", 2.0)),
        "proxy_top_k": int(payload.get("placement_proxy_top_k", 8)),
        "proxy_population": int(ga["population"]),
        "proxy_iters": int(ga["generations"]),
        "proxy_eval_budget": int(payload.get("placement_proxy_eval_budget", 96)),
        "ga": ga,
        "use_turntable": use_tt,
        "objective": "fairness_first",
        "constraints": {
            "temp_ceiling_c": float(payload.get("placement_temp_ceiling_c", 250.0)),
            "min_rho_floor": float(payload.get("placement_min_rho_floor", 0.55)),
        },
    }
    if use_tt:
        rot_deg = float(payload.get("placement_turntable_rotation_deg", 90.0))
        total_rot = int(payload.get("placement_turntable_total_rotations", 1))
        if total_rot < 1:
            raise ValueError("placement_turntable_total_rotations must be >= 1")
        interval_raw = payload.get("placement_turntable_interval_s", None)
        tt: dict[str, Any] = {
            "enabled": True,
            "rotation_deg": rot_deg,
            "total_rotations": total_rot,
        }
        if interval_raw not in ("", None):
            interval_s = float(interval_raw)
            if interval_s <= 0:
                raise ValueError("placement_turntable_interval_s must be > 0")
            tt["rotation_interval_s"] = interval_s
        cfg["turntable"] = tt
    else:
        cfg.pop("turntable", None)
    cfg.pop("optimizer", None)
    cfg.pop("orientation_optimizer", None)


def _configure_turntable(cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    rot_deg = float(payload.get("turntable_rotation_deg", 90.0))
    total_rot = int(payload.get("turntable_total_rotations", 1))
    if total_rot < 1:
        raise ValueError("turntable_total_rotations must be >= 1")
    interval_raw = payload.get("turntable_interval_s", None)

    tt: dict[str, Any] = {
        "enabled": True,
        "rotation_deg": rot_deg,
        "total_rotations": total_rot,
    }
    if interval_raw not in ("", None):
        interval_s = float(interval_raw)
        if interval_s <= 0:
            raise ValueError("turntable_interval_s must be > 0")
        tt["rotation_interval_s"] = interval_s

    cfg["turntable"] = tt
    cfg.pop("optimizer", None)
    cfg.pop("orientation_optimizer", None)
    cfg.pop("placement_optimizer", None)


def _configure_single(cfg: dict[str, Any]) -> None:
    cfg.pop("turntable", None)
    cfg.pop("optimizer", None)
    cfg.pop("orientation_optimizer", None)
    cfg.pop("placement_optimizer", None)


def _apply_antennae_config(cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    ant_raw = payload.get("antennae_enabled", None)
    spike_raw = payload.get("spike_enabled", None)
    has_cfg_fields = any(
        k in payload
        for k in (
            "antennae_size_mode",
            "antennae_global_size_mm",
            "antennae_auto_min_mm",
            "antennae_auto_max_mm",
            "antennae_auto_qrf_percentile_low",
            "antennae_auto_qrf_percentile_high",
            "antennae_max_per_part",
            "antennae_min_spacing_mm",
            "antennae_edge_margin_mm",
        )
    )
    if ant_raw is None and spike_raw is None and not has_cfg_fields:
        return

    if ant_raw is None and spike_raw is not None:
        warnings.warn(
            "Payload key 'spike_enabled' is deprecated; use 'antennae_enabled'.",
            DeprecationWarning,
            stacklevel=2,
        )
        ant_raw = spike_raw

    if ant_raw is None:
        ant_existing = cfg.get("antennae", {})
        spk_existing = cfg.get("spike", {})
        enabled = bool(
            ant_existing.get("enabled", False) if isinstance(ant_existing, dict)
            else spk_existing.get("enabled", False) if isinstance(spk_existing, dict)
            else False
        )
    else:
        enabled = bool(ant_raw)
    ant_cfg = cfg.get("antennae", {})
    if not isinstance(ant_cfg, dict):
        ant_cfg = {}
    ant_cfg["enabled"] = enabled
    size_mode = str(payload.get("antennae_size_mode", ant_cfg.get("size_mode", "global"))).strip().lower()
    if size_mode not in {"global", "auto"}:
        size_mode = "global"
    ant_cfg["size_mode"] = size_mode
    ant_cfg["global_size_mm"] = float(payload.get("antennae_global_size_mm", ant_cfg.get("global_size_mm", 1.0)))
    ant_cfg["auto_size"] = {
        "min_mm": float(payload.get("antennae_auto_min_mm", ant_cfg.get("auto_size", {}).get("min_mm", 0.5) if isinstance(ant_cfg.get("auto_size", {}), dict) else 0.5)),
        "max_mm": float(payload.get("antennae_auto_max_mm", ant_cfg.get("auto_size", {}).get("max_mm", 2.0) if isinstance(ant_cfg.get("auto_size", {}), dict) else 2.0)),
        "qrf_percentile_low": float(payload.get("antennae_auto_qrf_percentile_low", ant_cfg.get("auto_size", {}).get("qrf_percentile_low", 10.0) if isinstance(ant_cfg.get("auto_size", {}), dict) else 10.0)),
        "qrf_percentile_high": float(payload.get("antennae_auto_qrf_percentile_high", ant_cfg.get("auto_size", {}).get("qrf_percentile_high", 40.0) if isinstance(ant_cfg.get("auto_size", {}), dict) else 40.0)),
    }
    ant_cfg["placement"] = {
        "source": "eqs_qrf_underheated",
        "max_antennae_per_part": int(payload.get("antennae_max_per_part", ant_cfg.get("placement", {}).get("max_antennae_per_part", 4) if isinstance(ant_cfg.get("placement", {}), dict) else 4)),
        "min_anchor_spacing_mm": float(payload.get("antennae_min_spacing_mm", ant_cfg.get("placement", {}).get("min_anchor_spacing_mm", 2.0) if isinstance(ant_cfg.get("placement", {}), dict) else 2.0)),
        "edge_margin_mm": float(payload.get("antennae_edge_margin_mm", ant_cfg.get("placement", {}).get("edge_margin_mm", 1.0) if isinstance(ant_cfg.get("placement", {}), dict) else 1.0)),
    }
    cfg["antennae"] = ant_cfg

    # Keep legacy alias in sync for older tooling that still checks spike.enabled.
    spike_cfg = cfg.get("spike", {})
    if not isinstance(spike_cfg, dict):
        spike_cfg = {}
    spike_cfg["enabled"] = enabled
    cfg["spike"] = spike_cfg


def _apply_shell_config(cfg: dict[str, Any], payload: dict[str, Any], wall_thickness_mm: float | None = None) -> None:
    shell_enabled = bool(payload.get("shell_enabled", False))
    shell_method = str(payload.get("shell_method", "offset_inward")).strip().lower() or "offset_inward"
    if wall_thickness_mm is None:
        try:
            wall_thickness_mm = float(payload.get("shell_wall_thickness_mm", 0.0))
        except (TypeError, ValueError):
            wall_thickness_mm = 0.0

    g = cfg.setdefault("geometry", {})
    if not shell_enabled:
        g.pop("shell", None)
        return
    if wall_thickness_mm <= 0.0:
        raise ValueError("shell_wall_thickness_mm must be > 0 when shell_enabled is true")

    g["shell"] = {
        "enabled": True,
        "wall_thickness_mm": float(wall_thickness_mm),
        "method": shell_method,
    }


def _make_job(mode: str, output_name: str | None = None) -> dict[str, Any]:
    jid = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    log_path = LOGS_DIR / f"gui_job_{jid}.log"
    job = {
        "id": jid,
        "mode": mode,
        "status": "queued",
        "created_at": _now_iso(),
        "started_at": None,
        "ended_at": None,
        "output_name": output_name,
        "log_path": str(log_path),
        "log_url": f"/files/outputs_eqs/_logs/{log_path.name}",
        "error": None,
        "config_resolution": [],
        "output_dirs": [],
        "artifacts": [],
        "total_runs": 1,
        "completed_runs": 0,
        "progress_pct": 0.0,
        "progress_label": "Queued",
        "queue_position": None,
    }
    with JOBS_LOCK:
        JOBS[jid] = job
    return job


def _sync_queue_positions() -> None:
    with JOBS_LOCK:
        for job in JOBS.values():
            job["queue_position"] = None
    with QUEUE_LOCK:
        queued = list(JOB_QUEUE)
    with JOBS_LOCK:
        for idx, jid in enumerate(queued, start=1):
            if jid in JOBS:
                JOBS[jid]["queue_position"] = idx


def _enqueue_job(job_id: str) -> None:
    with QUEUE_LOCK:
        JOB_QUEUE.append(job_id)
    _sync_queue_positions()


def _maybe_start_next_job() -> None:
    global ACTIVE_JOB_ID
    next_id: str | None = None
    with QUEUE_LOCK:
        if ACTIVE_JOB_ID is not None:
            return
        if JOB_QUEUE:
            next_id = JOB_QUEUE.pop(0)
            ACTIVE_JOB_ID = next_id
    if next_id is None:
        return

    with JOBS_LOCK:
        job = JOBS.get(next_id)
        if job is not None:
            job["status"] = "running"
            job["started_at"] = _now_iso()
            job["progress_label"] = "Starting..."

    _sync_queue_positions()
    with JOBS_LOCK:
        payload = dict(JOBS[next_id].get("_payload", {}))
    t = threading.Thread(target=_job_worker, args=(next_id, payload), daemon=True)
    t.start()


def _reorder_queued_job(job_id: str, target_index: int) -> bool:
    with QUEUE_LOCK:
        if job_id not in JOB_QUEUE:
            return False
        cur = JOB_QUEUE.index(job_id)
        tgt = max(0, min(int(target_index), len(JOB_QUEUE) - 1))
        if tgt == cur:
            return True
        JOB_QUEUE.pop(cur)
        JOB_QUEUE.insert(tgt, job_id)
    _sync_queue_positions()
    return True


def _remove_from_queue(job_id: str) -> bool:
    removed = False
    with QUEUE_LOCK:
        if job_id in JOB_QUEUE:
            JOB_QUEUE.remove(job_id)
            removed = True
    if removed:
        _sync_queue_positions()
    return removed


def _get_running_proc(job_id: str) -> subprocess.Popen | None:
    with PROC_LOCK:
        return RUNNING_PROCS.get(job_id)


def _set_cancel_requested(job_id: str, flag: bool = True) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["cancel_requested"] = bool(flag)


def _is_cancel_requested(job_id: str) -> bool:
    with JOBS_LOCK:
        return bool(JOBS.get(job_id, {}).get("cancel_requested", False))


def _pause_running_job(job_id: str) -> tuple[bool, str]:
    proc = _get_running_proc(job_id)
    if proc is None:
        return False, "job process not found"
    if proc.poll() is not None:
        return False, "job already finished"
    if not hasattr(signal, "SIGSTOP"):
        return False, "pause is unsupported on this platform"
    try:
        os.kill(proc.pid, signal.SIGSTOP)
    except Exception as exc:
        return False, f"failed to pause process: {exc}"
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["status"] = "paused"
            JOBS[job_id]["pause_state"] = "running"
            JOBS[job_id]["progress_label"] = "Paused (running process stopped)"
    return True, "paused"


def _resume_paused_running_job(job_id: str) -> tuple[bool, str]:
    proc = _get_running_proc(job_id)
    if proc is None:
        return False, "job process not found"
    if proc.poll() is not None:
        return False, "job already finished"
    if not hasattr(signal, "SIGCONT"):
        return False, "resume is unsupported on this platform"
    try:
        os.kill(proc.pid, signal.SIGCONT)
    except Exception as exc:
        return False, f"failed to resume process: {exc}"
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["pause_state"] = None
            JOBS[job_id]["progress_label"] = "Resumed"
    return True, "resumed"


def _cancel_running_job(job_id: str) -> tuple[bool, str]:
    proc = _get_running_proc(job_id)
    _set_cancel_requested(job_id, True)
    if proc is None:
        return False, "job process not found"
    if proc.poll() is not None:
        return False, "job already finished"
    # If paused, continue before terminate so process can handle signal.
    if hasattr(signal, "SIGCONT"):
        try:
            os.kill(proc.pid, signal.SIGCONT)
        except Exception:
            pass
    try:
        proc.terminate()
    except Exception as exc:
        return False, f"failed to terminate process: {exc}"
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["status"] = "cancelling"
            JOBS[job_id]["progress_label"] = "Cancelling..."
    return True, "cancelling"


def _set_job_progress(
    job_id: str,
    *,
    total_runs: int | None = None,
    completed_runs: int | None = None,
    progress_pct: float | None = None,
    progress_label: str | None = None,
) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        if total_runs is not None:
            job["total_runs"] = max(1, int(total_runs))
        if completed_runs is not None:
            job["completed_runs"] = max(0, int(completed_runs))
        if progress_pct is not None:
            job["progress_pct"] = float(max(0.0, min(100.0, progress_pct)))
        elif completed_runs is not None:
            tot = max(1, int(job.get("total_runs", 1)))
            job["progress_pct"] = float(max(0.0, min(100.0, (100.0 * int(job.get("completed_runs", 0))) / tot)))
        if progress_label is not None:
            job["progress_label"] = str(progress_label)


def _finish_job(job_id: str, status: str, error: str | None = None) -> None:
    global ACTIVE_JOB_ID
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["status"] = status
        job["ended_at"] = _now_iso()
        job["error"] = error
        if status == "completed":
            total = max(1, int(job.get("total_runs", 1)))
            job["completed_runs"] = total
            job["progress_pct"] = 100.0
            job["progress_label"] = "Completed"
        elif status == "failed":
            if not job.get("progress_label"):
                job["progress_label"] = "Failed"
        elif status == "cancelled":
            job["progress_label"] = "Cancelled"
    with QUEUE_LOCK:
        if ACTIVE_JOB_ID == job_id:
            ACTIVE_JOB_ID = None
    _sync_queue_positions()
    _maybe_start_next_job()


def _write_job_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _url_for_output_rel(rel_path: str) -> str:
    return "/files/outputs_eqs/" + quote(rel_path, safe="/")


def _should_skip_result_path(path: Path) -> bool:
    rel = path.relative_to(OUTPUTS_DIR).as_posix().lower()
    if rel.startswith("_archive"):
        return True
    if "prewarp" in rel:
        return True
    return False


def _collect_images_recursive(root: Path) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    for pattern in ("*.png", "*.gif", "*.svg"):
        for img in sorted(root.rglob(pattern)):
            rel_to_root = img.relative_to(root).as_posix()
            rel_to_outputs = img.relative_to(OUTPUTS_DIR).as_posix()
            images.append({
                "path": rel_to_root,
                "url": _url_for_output_rel(rel_to_outputs),
            })
    images.sort(key=lambda it: it["path"])
    return images


def _build_artifact(output_dir: Path) -> dict[str, Any]:
    rel = output_dir.relative_to(OUTPUTS_DIR).as_posix()
    summary = {}
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {"error": "Failed to parse summary.json"}
    images = _collect_images_recursive(output_dir)
    return {
        "output_dir": rel,
        "summary": summary,
        "images": images,
    }


def _register_job_output(job: dict[str, Any], out_dir: Path) -> None:
    rel = out_dir.relative_to(OUTPUTS_DIR).as_posix()
    if rel not in job["output_dirs"]:
        job["output_dirs"].append(rel)


def _output_dir_for_request(output_name: str, payload: dict[str, Any]) -> Path:
    shape = str(payload.get("shape", "unknown")).strip().lower() or "unknown"
    mode = str(payload.get("mode", "single")).strip().lower() or "single"
    family = "experimental" if _is_experimental_request(payload) else "baseline"
    safe_shape = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in shape).strip("_") or "unknown"
    safe_mode = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in mode).strip("_") or "single"
    return OUTPUTS_DIR / "runs" / safe_shape / safe_mode / family / output_name


def _refresh_job_artifacts(job: dict[str, Any]) -> None:
    artifacts: list[dict[str, Any]] = []
    for rel in job.get("output_dirs", []):
        out_dir = OUTPUTS_DIR / rel
        if out_dir.exists() and out_dir.is_dir():
            artifacts.append(_build_artifact(out_dir))
    job["artifacts"] = artifacts


def _run_command(cmd: list[str], log_path: Path, job_id: str | None = None) -> int:
    _write_job_log(log_path, f"$ {' '.join(cmd)}")
    with log_path.open("a", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        if job_id is not None:
            with PROC_LOCK:
                RUNNING_PROCS[job_id] = proc
        try:
            while True:
                rc = proc.poll()
                if rc is not None:
                    return int(rc)
                if job_id is not None and _is_cancel_requested(job_id):
                    try:
                        proc.terminate()
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2.0)
                    raise JobCancelled("cancelled by user")
                time.sleep(0.2)
        finally:
            if job_id is not None:
                with PROC_LOCK:
                    RUNNING_PROCS.pop(job_id, None)


def _orientation_eval_estimate(payload: dict[str, Any]) -> int:
    a_min = float(payload.get("orientation_angle_min_deg", 0.0))
    a_max = float(payload.get("orientation_angle_max_deg", 180.0))
    a_step = max(float(payload.get("orientation_angle_step_deg", 15.0)), 1e-9)
    e_min = float(payload.get("orientation_exposure_min_s", 240.0))
    e_max = float(payload.get("orientation_exposure_max_s", 720.0))
    e_step = max(float(payload.get("orientation_exposure_step_s", 60.0)), 1e-9)
    r_win = float(payload.get("orientation_refine_window_deg", 10.0))
    r_step = max(float(payload.get("orientation_refine_step_deg", 5.0)), 1e-9)
    n_angles = len(np.arange(a_min, a_max + 1e-9, a_step, dtype=float))
    n_exp = len(np.arange(e_min, e_max + 1e-9, e_step, dtype=float))
    n_ref_angles = len(np.arange(-r_win, r_win + 1e-9, r_step, dtype=float))
    return max(1, n_angles * n_exp + n_ref_angles * n_exp)


def _prepare_config(mode: str, payload: dict[str, Any], exposure_minutes: float, output_tag: str, job: dict[str, Any]) -> Path:
    forced = str(payload.get("base_config", "")).strip() or None
    req_sig = _signature_for_request(payload, mode, exposure_minutes)
    resolved = _resolve_base_config_for_signature(req_sig, forced_base_config=forced)
    base_path = Path(resolved["config_path"])

    if (
        resolved["match_type"] == "matched_existing"
        and not forced
        and not _has_advanced_overrides(payload)
        and not _has_antennae_overrides(payload)
        and not _has_shell_overrides(payload)
        and not _has_geometry_size_overrides(payload)
        and not _is_experimental_request(payload)
    ):
        job["config_resolution"].append({
            "requested": req_sig,
            "resolved": resolved,
            "generated_config": None,
        })
        return base_path

    cfg = _load_yaml(base_path)
    _disable_prewarp_like_blocks(cfg)
    _apply_geometry_size(cfg, payload)
    _set_shape(cfg, str(payload.get("shape", "")).strip())
    _apply_advanced_overrides(cfg, payload)
    _apply_antennae_config(cfg, payload)
    _apply_shell_config(cfg, payload)
    experimental_active = _apply_experimental_bucket(cfg, payload)
    _set_exposure_minutes(cfg, exposure_minutes)

    if mode == "optimizer":
        _configure_optimizer(cfg, payload)
    elif mode == "orientation_optimizer":
        _configure_orientation_optimizer(cfg, payload)
    elif mode == "placement_optimizer":
        _configure_placement_optimizer(cfg, payload)
    elif mode == "turntable":
        _configure_turntable(cfg, payload)
    else:
        _configure_single(cfg)

    gen_name = f"{job['id']}_{output_tag}_{mode}.yaml"
    gen_path = GUI_CONFIG_DIR / gen_name
    gen_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    job["config_resolution"].append({
        "requested": req_sig,
        "resolved": resolved,
        "generated_config": str(gen_path),
        "experimental_active": experimental_active,
    })
    return gen_path


def _write_shell_sweep_report(rows: list[dict[str, Any]], report_dir: Path) -> None:
    if not rows:
        return
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_shell_sweep_report_svg(rows, report_dir)
    _write_shell_sweep_report_png(rows, report_dir)


def _write_shell_sweep_report_svg(rows: list[dict[str, Any]], report_dir: Path) -> None:
    by_shape: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_shape.setdefault(str(r.get("shape", "shape")), []).append(r)
    for shape_rows in by_shape.values():
        shape_rows.sort(key=lambda it: float(it.get("wall_thickness_mm", 0.0)))

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
    (report_dir / "shell_sweep_report.svg").write_text("\n".join(lines))


def _write_shell_sweep_report_png(rows: list[dict[str, Any]], report_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    by_shape: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_shape.setdefault(str(r.get("shape", "shape")), []).append(r)
    for shape_rows in by_shape.values():
        shape_rows.sort(key=lambda it: float(it.get("wall_thickness_mm", 0.0)))

    fig, ax = plt.subplots(1, 1, figsize=(9.6, 5.6), dpi=140)
    for shape in sorted(by_shape.keys()):
        shape_rows = by_shape[shape]
        x = [float(it.get("wall_thickness_mm", 0.0)) for it in shape_rows]
        y = [float(it.get("mean_rho_rel_part_final", 0.0)) for it in shape_rows]
        ax.plot(x, y, marker="o", lw=1.8, ms=4.5, label=shape)
    ax.set_title("Shell Sweep: mean density vs wall thickness")
    ax.set_xlabel("Wall thickness (mm)")
    ax.set_ylabel("Mean relative density, rho_rel (-)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(report_dir / "shell_sweep_report.png", bbox_inches="tight")
    plt.close(fig)


def _launch_single_mode(payload: dict[str, Any], job: dict[str, Any]) -> None:
    output_name = str(payload.get("output_name", "")).strip()
    shape = str(payload.get("shape", "")).strip()
    mode = str(payload.get("mode", "single")).strip()

    if not _is_valid_output_name(output_name):
        raise ValueError("output_name must use only letters, numbers, '_' or '-'")
    if shape not in SUPPORTED_SHAPES:
        raise ValueError(f"unsupported shape: {shape}")

    exposure_minutes = float(payload.get("exposure_minutes", 6.0))
    if exposure_minutes <= 0:
        raise ValueError("exposure_minutes must be > 0")

    job_id = str(job["id"])
    run_hint = "Running simulation"
    if mode == "orientation_optimizer":
        run_hint = f"Orientation search running (~{_orientation_eval_estimate(payload)} evaluations)"
    elif mode == "placement_optimizer":
        run_hint = "Placement search running (proxy + coupled rerank)"
    elif mode == "optimizer":
        run_hint = "Optimizer run in progress"
    elif mode == "turntable":
        run_hint = "Turntable run in progress"
    _set_job_progress(job_id, total_runs=1, completed_runs=0, progress_pct=5.0, progress_label=run_hint)

    cfg_path = _prepare_config(mode=mode, payload=payload, exposure_minutes=exposure_minutes, output_tag=output_name, job=job)
    out_dir = _output_dir_for_request(output_name, payload)
    _register_job_output(job, out_dir)
    cmd = [sys.executable, "rfam_eqs_coupled.py", "--config", str(cfg_path), "--output-dir", str(out_dir)]
    rc = _run_command(cmd, Path(job["log_path"]), job_id=job_id)
    if rc != 0:
        _set_job_progress(job_id, progress_label="Run failed")
        raise RuntimeError(f"simulation exited with code {rc}")
    _set_job_progress(job_id, completed_runs=1, progress_pct=100.0, progress_label="Run complete")


def _launch_sweep_mode(payload: dict[str, Any], job: dict[str, Any]) -> None:
    output_prefix = str(payload.get("output_name", "")).strip()
    shape = str(payload.get("shape", "")).strip()
    sweep_minutes = _to_float_list(payload.get("sweep_minutes", [6, 7, 8, 9, 10]))

    if not _is_valid_output_name(output_prefix):
        raise ValueError("output_name must use only letters, numbers, '_' or '-'")
    if shape not in SUPPORTED_SHAPES:
        raise ValueError(f"unsupported shape: {shape}")
    if not sweep_minutes:
        raise ValueError("sweep_minutes is empty")

    total = len(sweep_minutes)
    job_id = str(job["id"])
    _set_job_progress(job_id, total_runs=total, completed_runs=0, progress_pct=0.0, progress_label=f"Sweep queued (0/{total})")

    for idx, minutes in enumerate(sweep_minutes, start=1):
        if minutes <= 0:
            raise ValueError("sweep minutes must be > 0")
        _set_job_progress(
            job_id,
            progress_pct=100.0 * float(idx - 1) / float(total),
            progress_label=f"Sweep run {idx}/{total} ({minutes} min)",
        )
        suffix = str(minutes).replace(".", "p")
        cfg_path = _prepare_config(
            mode="single",
            payload=payload,
            exposure_minutes=minutes,
            output_tag=f"{output_prefix}_{suffix}min_sweep",
            job=job,
        )
        out_name = f"{output_prefix}_{suffix}min"
        out_dir = _output_dir_for_request(out_name, payload)
        _register_job_output(job, out_dir)
        cmd = [sys.executable, "rfam_eqs_coupled.py", "--config", str(cfg_path), "--output-dir", str(out_dir)]
        rc = _run_command(cmd, Path(job["log_path"]), job_id=job_id)
        if rc != 0:
            _set_job_progress(job_id, progress_label=f"Sweep failed at run {idx}/{total}")
            raise RuntimeError(f"sweep run ({minutes} min) exited with code {rc}")
        _set_job_progress(
            job_id,
            completed_runs=idx,
            progress_pct=100.0 * float(idx) / float(total),
            progress_label=f"Sweep run {idx}/{total} complete",
        )


def _launch_shell_sweep_mode(payload: dict[str, Any], job: dict[str, Any]) -> None:
    output_prefix = str(payload.get("output_name", "")).strip()
    selected_shape = str(payload.get("shape", "")).strip()
    shape_tokens = _to_str_list(payload.get("shell_sweep_shapes", []))
    if shape_tokens:
        shapes = [s for s in shape_tokens if s in {"circle", "square"}]
    else:
        shapes = [selected_shape] if selected_shape in {"circle", "square"} else []
    if not shapes:
        raise ValueError("shell sweep supports shapes circle and square (use shape selector or shell_sweep_shapes)")

    if not _is_valid_output_name(output_prefix):
        raise ValueError("output_name must use only letters, numbers, '_' or '-'")

    thicknesses = _to_float_list(payload.get("shell_sweep_thicknesses_mm", [2, 3, 4, 5, 6, 7]))
    thicknesses = [float(t) for t in thicknesses if float(t) > 0.0]
    if not thicknesses:
        raise ValueError("shell_sweep_thicknesses_mm is empty")

    rows: list[dict[str, Any]] = []
    summary_dir = _output_dir_for_request(output_prefix, payload)
    summary_dir.mkdir(parents=True, exist_ok=True)
    _register_job_output(job, summary_dir)

    total = len(shapes) * len(thicknesses)
    done = 0
    job_id = str(job["id"])
    _set_job_progress(job_id, total_runs=total, completed_runs=0, progress_pct=0.0, progress_label=f"Shell sweep queued (0/{total})")

    for shape in shapes:
        for t_mm in thicknesses:
            _set_job_progress(
                job_id,
                progress_pct=100.0 * float(done) / float(max(total, 1)),
                progress_label=f"Shell sweep {done + 1}/{total} ({shape}, {t_mm:g} mm)",
            )
            shell_payload = dict(payload)
            shell_payload["shape"] = shape
            shell_payload["shell_enabled"] = True
            shell_payload["shell_wall_thickness_mm"] = float(t_mm)
            shell_payload["shell_method"] = str(payload.get("shell_method", "offset_inward"))
            suffix = str(t_mm).replace(".", "p")
            run_name = f"{output_prefix}_{shape}_t{suffix}mm"
            cfg_path = _prepare_config(
                mode="single",
                payload=shell_payload,
                exposure_minutes=float(payload.get("exposure_minutes", 6.0)),
                output_tag=run_name,
                job=job,
            )
            out_dir = _output_dir_for_request(run_name, payload)
            _register_job_output(job, out_dir)
            cmd = [sys.executable, "rfam_eqs_coupled.py", "--config", str(cfg_path), "--output-dir", str(out_dir)]
            rc = _run_command(cmd, Path(job["log_path"]), job_id=job_id)
            if rc != 0:
                _set_job_progress(job_id, progress_label=f"Shell sweep failed at {done + 1}/{total}")
                raise RuntimeError(f"shell_sweep run ({shape}, {t_mm} mm) exited with code {rc}")
            done += 1
            _set_job_progress(
                job_id,
                completed_runs=done,
                progress_pct=100.0 * float(done) / float(max(total, 1)),
                progress_label=f"Shell sweep {done}/{total} complete",
            )

            summary = {}
            summary_path = out_dir / "summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                except Exception:
                    summary = {}
            rows.append({
                "shape": shape,
                "wall_thickness_mm": float(t_mm),
                "output_dir": out_dir.relative_to(OUTPUTS_DIR).as_posix(),
                "mean_rho_rel_part_final": float(summary.get("mean_rho_rel_part_final", 0.0)),
                "min_rho_rel_part_final": float(summary.get("min_rho_rel_part_final", 0.0)),
                "max_T_part_final_c": float(summary.get("max_T_part_final_c", 0.0)),
            })

    if rows:
        temp_ceiling = float(payload.get("shell_temp_ceiling_c", 250.0))
        eligible = [r for r in rows if float(r.get("max_T_part_final_c", math.inf)) <= temp_ceiling]
        ranked = sorted(eligible if eligible else rows, key=lambda r: float(r.get("mean_rho_rel_part_final", 0.0)), reverse=True)
        best = ranked[0] if ranked else None
        sweep_summary = {
            "mode": "shell_sweep",
            "temp_ceiling_c": temp_ceiling,
            "best": best,
            "rows": rows,
        }
        (summary_dir / "shell_sweep_summary.json").write_text(json.dumps(sweep_summary, indent=2))
        _write_shell_sweep_report(rows, summary_dir)
        _load_or_build_manifest(summary_dir)


def _launch_antennae_calibration_mode(payload: dict[str, Any], job: dict[str, Any]) -> None:
    output_name = str(payload.get("output_name", "")).strip()
    if not _is_valid_output_name(output_name):
        raise ValueError("output_name must use only letters, numbers, '_' or '-'")
    shape = str(payload.get("shape", "")).strip()
    if shape and shape not in SUPPORTED_SHAPES:
        raise ValueError(f"unsupported shape: {shape}")
    exposure_minutes = float(payload.get("exposure_minutes", 6.0))
    if exposure_minutes <= 0:
        raise ValueError("exposure_minutes must be > 0")

    job_id = str(job["id"])
    _set_job_progress(job_id, total_runs=1, completed_runs=0, progress_pct=5.0, progress_label="Antennae calibration setup")
    cfg_path = _prepare_config(
        mode="single",
        payload=payload,
        exposure_minutes=exposure_minutes,
        output_tag=f"{output_name}_antennae_calibration",
        job=job,
    )
    out_dir = _output_dir_for_request(output_name, payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    _register_job_output(job, out_dir)
    cmd = [
        sys.executable,
        "run_antennae_calibration.py",
        "--config",
        str(cfg_path),
        "--output-dir",
        str(out_dir),
        "--job-id",
        str(job["id"]),
    ]
    sizes = _to_float_list(payload.get("antennae_calibration_sizes_mm", []))
    if sizes:
        cmd.extend(["--sizes-mm", ",".join(f"{float(v):g}" for v in sizes)])
    if bool(payload.get("antennae_calibration_include_auto", False)):
        cmd.append("--include-auto")
    if "antennae_calibration_top_k" in payload:
        try:
            cmd.extend(["--top-k", str(int(payload.get("antennae_calibration_top_k", 3)))])
        except Exception:
            pass
    if bool(payload.get("antennae_calibration_use_turntable", False)):
        cmd.append("--use-turntable")
    _set_job_progress(job_id, progress_pct=20.0, progress_label="Antennae calibration running")
    rc = _run_command(cmd, Path(job["log_path"]), job_id=job_id)
    if rc != 0:
        _set_job_progress(job_id, progress_label="Antennae calibration failed")
        raise RuntimeError(f"antennae calibration exited with code {rc}")
    _load_or_build_manifest(out_dir)
    _set_job_progress(job_id, completed_runs=1, progress_pct=100.0, progress_label="Antennae calibration complete")


def _job_worker(job_id: str, payload: dict[str, Any]) -> None:
    job = JOBS[job_id]
    try:
        mode = payload.get("mode", "single")
        if mode == "backfill_reports":
            _run_backfill_reports_job(payload, job)
        elif mode == "antennae_calibrate":
            _launch_antennae_calibration_mode(payload, job)
        elif mode == "sweep":
            _launch_sweep_mode(payload, job)
        elif mode == "shell_sweep":
            _launch_shell_sweep_mode(payload, job)
        else:
            _launch_single_mode(payload, job)
        _refresh_job_artifacts(job)
        _finish_job(job_id, "completed")
    except JobCancelled:
        _write_job_log(Path(job["log_path"]), "[cancelled] cancelled by user")
        _finish_job(job_id, "cancelled", "cancelled by user")
    except Exception as exc:
        _write_job_log(Path(job["log_path"]), f"[error] {exc}")
        _finish_job(job_id, "failed", str(exc))


def _collect_results() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for p in sorted(OUTPUTS_DIR.rglob("*"), key=lambda q: q.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue
        if _should_skip_result_path(p):
            continue
        has_summary = (p / "summary.json").exists()
        has_media = any(p.glob("*.png")) or any(p.glob("*.gif")) or any(p.glob("*.svg"))
        if not (has_summary or has_media):
            continue
        rel = p.relative_to(OUTPUTS_DIR).as_posix()
        items.append({
            "name": rel,
            "run_created_at": _infer_run_created_at(p),
            "updated_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
        })
    return items


def _is_backfill_target_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if (path / "report_manifest.json").exists():
        return True
    if (path / "summary.json").exists():
        return True
    if (path / "shell_sweep_summary.json").exists():
        return True
    if (path / "orientation_candidates.json").exists() or (path / "orientation_pareto.json").exists():
        return True
    if (path / "placement_topk_coupled.json").exists() or (path / "placement_candidates_proxy.json").exists():
        return True
    if any(path.glob("*.png")) or any(path.glob("*.gif")) or any(path.glob("*.svg")):
        return True
    return False


def _fs_list(path_raw: str) -> dict[str, Any]:
    target = _safe_workspace_path(path_raw)
    if target is None or not target.exists() or not target.is_dir():
        raise FileNotFoundError(path_raw)
    items: list[dict[str, Any]] = []
    for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            st = entry.stat()
        except OSError:
            continue
        rel = entry.relative_to(ROOT).as_posix()
        manifest = _load_or_build_manifest(entry) if entry.is_dir() and entry.relative_to(ROOT).as_posix().startswith("outputs_eqs/") and _is_backfill_target_dir(entry) else {}
        run_type = str(manifest.get("run_type", _get_run_type(entry))) if entry.is_dir() else "unknown"
        caps = [str(v) for v in manifest.get("backfill_capabilities", [])] if manifest else (_resolve_backfill_capabilities(entry, run_type) if entry.is_dir() else [])
        is_orientation_run = bool("orientation_diagnostics_v1" in caps)
        items.append({
            "name": entry.name,
            "path": rel,
            "is_dir": entry.is_dir(),
            "is_orientation_run": is_orientation_run,
            "run_type": run_type,
            "backfill_capabilities": caps,
            "has_report_manifest": bool(entry.is_dir() and (entry / "report_manifest.json").exists()),
            "size_bytes": int(st.st_size),
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        })
    rel_cur = target.relative_to(ROOT).as_posix()
    cwd_manifest = _load_or_build_manifest(target) if rel_cur.startswith("outputs_eqs") and _is_backfill_target_dir(target) else {}
    cwd_run_type = str(cwd_manifest.get("run_type", _get_run_type(target)))
    cwd_caps = [str(v) for v in cwd_manifest.get("backfill_capabilities", [])] if cwd_manifest else _resolve_backfill_capabilities(target, cwd_run_type)
    return {
        "cwd": rel_cur if rel_cur else ".",
        "cwd_is_orientation_run": bool("orientation_diagnostics_v1" in cwd_caps),
        "cwd_run_type": cwd_run_type,
        "cwd_backfill_capabilities": cwd_caps,
        "cwd_has_report_manifest": bool((target / "report_manifest.json").exists()),
        "items": items,
    }


def _iter_rel_files(run_dir: Path) -> list[str]:
    files: list[str] = []
    if not run_dir.exists() or not run_dir.is_dir():
        return files
    for p in run_dir.rglob("*"):
        if p.is_file():
            files.append(p.relative_to(run_dir).as_posix())
    return sorted(files)


def _is_orientation_run_dir(path: Path) -> bool:
    return _get_run_type(path) == "orientation_optimizer"


def _detect_run_type_from_summary(summary: dict[str, Any]) -> str:
    if not isinstance(summary, dict):
        return "unknown"
    if isinstance(summary.get("orientation_optimizer", None), dict):
        return "orientation_optimizer"
    if isinstance(summary.get("placement_optimizer", None), dict):
        return "placement_optimizer"
    if isinstance(summary.get("turntable_rotations", None), list):
        return "turntable"
    if isinstance(summary.get("shell_sweep", None), dict) or ("shell_sweep_summary" in str(summary.get("mode", ""))):
        return "shell_sweep"
    return "single"


def _get_run_type(path: Path) -> str:
    if not path.exists() or not path.is_dir():
        return "unknown"
    manifest_rt = ""
    manifest_path = path / "report_manifest.json"
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            manifest_rt = str(m.get("run_type", "")).strip()
        except Exception:
            pass
    detected_rt = "unknown"
    summary_path = path / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            rt = _detect_run_type_from_summary(summary)
            if rt != "unknown":
                detected_rt = rt
        except Exception:
            pass
    if detected_rt == "unknown":
        if (
            (path / "orientation_pareto.json").exists()
            or (path / "orientation_best.json").exists()
            or (path / "orientation_candidates.json").exists()
            or (path / "orientation_angle_summary.json").exists()
        ):
            detected_rt = "orientation_optimizer"
        elif (path / "placement_topk_coupled.json").exists() or (path / "placement_best.json").exists():
            detected_rt = "placement_optimizer"
        elif (path / "turntable_density.gif").exists() or (path / "turntable_thermal.gif").exists():
            detected_rt = "turntable"
        elif (path / "shell_sweep_summary.json").exists():
            detected_rt = "shell_sweep"
        elif (path / "optimizer_report.png").exists():
            detected_rt = "optimizer"
        elif (path / "summary.json").exists():
            detected_rt = "single"

    if manifest_rt and manifest_rt not in {"single", "unknown"}:
        return manifest_rt
    if detected_rt not in {"single", "unknown"}:
        return detected_rt
    if manifest_rt:
        return manifest_rt
    return detected_rt


def _module_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "manifest_refresh_v1",
            "version": "1",
            "supported_run_types": ["single", "sweep", "optimizer", "turntable", "orientation_optimizer", "placement_optimizer", "shell_sweep", "unknown"],
            "required_any": [],
            "required_all": [],
            "produced_patterns": ["report_manifest.json"],
        },
        {
            "id": "rf_summary_v5_v1",
            "version": "1",
            "supported_run_types": ["single", "sweep", "optimizer", "turntable", "orientation_optimizer", "placement_optimizer", "shell_sweep", "unknown"],
            "required_any": [],
            "required_all": ["summary.json", "fields.npz"],
            "produced_patterns": ["rf_summary_v5.backfill.rf_summary_v5_v1.*.png"],
        },
        {
            "id": "core_reports_v1",
            "version": "1",
            "supported_run_types": ["single", "sweep", "optimizer", "turntable", "orientation_optimizer", "placement_optimizer", "unknown"],
            "required_any": [],
            "required_all": ["fields.npz"],
            "produced_patterns": [
                "paper_style_report.backfill.core_reports_v1.*.png",
                "thermal_fields_final.backfill.core_reports_v1.*.png",
                "electric_fields.backfill.core_reports_v1.*.png",
                "time_series.backfill.core_reports_v1.*.png",
                "paper_style_report.png",
                "thermal_fields_final.png",
                "electric_fields.png",
                "time_series.png",
            ],
        },
        {
            "id": "orientation_diagnostics_v1",
            "version": "1",
            "supported_run_types": ["orientation_optimizer"],
            "required_any": ["orientation_candidates.json", "orientation_pareto.json"],
            "required_all": [],
            "produced_patterns": [
                "orientation_candidates.backfill.orientation_diagnostics_v1.*.json",
                "orientation_candidates.backfill.orientation_diagnostics_v1.*.csv",
                "orientation_angle_summary.backfill.orientation_diagnostics_v1.*.json",
                "orientation_angle_summary.backfill.orientation_diagnostics_v1.*.csv",
                "orientation_angle_effectiveness.backfill.orientation_diagnostics_v1.*.png",
            ],
        },
        {
            "id": "orientation_gif_v1",
            "version": "1",
            "supported_run_types": ["orientation_optimizer"],
            "required_any": ["orientation_angle_summary*.json", "orientation_candidates*.json"],
            "required_all": ["used_config.yaml"],
            "produced_patterns": ["orientation_angle_gallery.backfill.orientation_gif_v1.*.gif"],
        },
        {
            "id": "placement_reports_v1",
            "version": "1",
            "supported_run_types": ["placement_optimizer"],
            "required_any": [],
            "required_all": ["placement_topk_coupled.json"],
            "produced_patterns": ["placement_report.backfill.placement_reports_v1.*.png"],
        },
        {
            "id": "shell_sweep_report_v1",
            "version": "1",
            "supported_run_types": ["shell_sweep"],
            "required_any": [],
            "required_all": ["shell_sweep_summary.json"],
            "produced_patterns": [
                "shell_sweep_report.backfill.shell_sweep_report_v1.*.png",
                "shell_sweep_report.backfill.shell_sweep_report_v1.*.svg",
            ],
        },
        {
            "id": "antennae_reports_v1",
            "version": "1",
            "supported_run_types": ["single", "sweep", "optimizer", "turntable", "orientation_optimizer", "placement_optimizer", "unknown"],
            "required_any": ["antennae_summary.json", "summary.json"],
            "required_all": ["summary.json", "fields.npz"],
            "produced_patterns": [
                "antennae_report.backfill.antennae_reports_v1.*.png",
                "antennae_anchors.backfill.antennae_reports_v1.*.json",
            ],
        },
    ]


def _module_spec_by_id(module_id: str) -> dict[str, Any] | None:
    for m in _module_specs():
        if str(m.get("id", "")) == str(module_id):
            return m
    return None


def _module_requirements_met(spec: dict[str, Any], rel_files: list[str]) -> bool:
    rel_set = set(rel_files)
    req_all = [str(v) for v in spec.get("required_all", [])]
    req_any = [str(v) for v in spec.get("required_any", [])]
    for req in req_all:
        if not any(fnmatch.fnmatch(f, req) for f in rel_set):
            return False
    if req_any:
        if not any(any(fnmatch.fnmatch(f, req) for f in rel_set) for req in req_any):
            return False
    return True


def _resolve_backfill_capabilities(path: Path, run_type: str | None = None) -> list[str]:
    if run_type is None:
        run_type = _get_run_type(path)
    if not path.exists() or not path.is_dir():
        return []
    rel_files = _iter_rel_files(path)
    caps: list[str] = []
    for spec in _module_specs():
        sup = [str(s) for s in spec.get("supported_run_types", [])]
        if run_type not in sup and "unknown" not in sup:
            continue
        if _module_requirements_met(spec, rel_files):
            if str(spec.get("id", "")) == "antennae_reports_v1":
                summary_path = path / "summary.json"
                ant_ok = False
                if (path / "antennae_summary.json").exists():
                    ant_ok = True
                elif summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text())
                        if isinstance(summary, dict):
                            ant_ok = bool(int(summary.get("antennae_count", 0)) > 0 or len(summary.get("antennae_instances", []) or []) > 0)
                    except Exception:
                        ant_ok = False
                if not ant_ok:
                    continue
            caps.append(str(spec.get("id", "")))
    return sorted([c for c in caps if c])


def _collect_available_reports(run_dir: Path, rel_files: list[str]) -> list[str]:
    out: list[str] = []
    for r in rel_files:
        rr = str(r).lower()
        if rr.endswith((".png", ".gif", ".svg", ".json", ".csv")) and (
            "report" in rr
            or rr.endswith("rf_summary_v5.png")
            or rr.startswith("orientation_")
            or rr.startswith("placement_")
            or rr.startswith("shell_sweep_")
        ):
            out.append(r)
    return sorted(set(out))


def _load_or_build_manifest(run_dir: Path) -> dict[str, Any]:
    run_type = _get_run_type(run_dir)
    rel_files = _iter_rel_files(run_dir)
    caps = _resolve_backfill_capabilities(run_dir, run_type=run_type)
    manifest_path = run_dir / "report_manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            raw = json.loads(manifest_path.read_text())
            if isinstance(raw, dict):
                existing = raw
        except Exception:
            existing = {}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_type": run_type,
        "created_at": existing.get("created_at", _now_iso()),
        "last_backfill_at": existing.get("last_backfill_at", None),
        "available_inputs": rel_files,
        "available_reports": _collect_available_reports(run_dir, rel_files),
        "backfill_capabilities": caps,
        "module_history": existing.get("module_history", []),
        "latest_by_module": existing.get("latest_by_module", {}),
    }
    prev_raw = None
    if manifest_path.exists():
        try:
            prev_raw = json.dumps(existing, sort_keys=True)
        except Exception:
            prev_raw = None
    next_raw = json.dumps(manifest, sort_keys=True)
    # Avoid touching file mtimes when manifest content is unchanged.
    if (not manifest_path.exists()) or (prev_raw != next_raw):
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _iso_from_epoch(epoch: float | int | None) -> str:
    try:
        ts = float(epoch) if epoch is not None else 0.0
    except Exception:
        ts = 0.0
    if ts <= 0.0:
        ts = time.time()
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _parse_iso_epoch(text: Any) -> float | None:
    try:
        s = str(text or "").strip()
        if not s:
            return None
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _infer_run_created_at(run_dir: Path) -> str:
    epochs: list[float] = []
    try:
        st = run_dir.stat()
        bt = getattr(st, "st_birthtime", None)
        if isinstance(bt, (int, float)) and bt > 0:
            epochs.append(float(bt))
    except Exception:
        pass

    manifest_path = run_dir / "report_manifest.json"
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            ep = _parse_iso_epoch(m.get("created_at"))
            if ep is not None:
                epochs.append(ep)
        except Exception:
            pass

    key_files = [
        "used_config.yaml",
        "summary.json",
        "fields.npz",
        "shell_sweep_summary.json",
        "orientation_candidates.json",
        "placement_topk_coupled.json",
    ]
    for name in key_files:
        fp = run_dir / name
        if fp.exists() and fp.is_file():
            try:
                epochs.append(float(fp.stat().st_mtime))
            except Exception:
                pass

    if not epochs:
        try:
            for fp in run_dir.rglob("*"):
                if not fp.is_file():
                    continue
                try:
                    epochs.append(float(fp.stat().st_mtime))
                except Exception:
                    continue
        except Exception:
            pass

    if epochs:
        return _iso_from_epoch(min(epochs))
    try:
        return _iso_from_epoch(run_dir.stat().st_mtime)
    except Exception:
        return _iso_from_epoch(None)


def _stamp_tag() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _append_manifest_history(
    run_dir: Path,
    *,
    module_id: str,
    started_at: str,
    finished_at: str,
    status: str,
    produced_files: list[str],
    log_excerpt: str,
) -> None:
    manifest = _load_or_build_manifest(run_dir)
    hist = manifest.get("module_history", [])
    if not isinstance(hist, list):
        hist = []
    hist.append(
        {
            "module_id": str(module_id),
            "started_at": str(started_at),
            "finished_at": str(finished_at),
            "status": str(status),
            "produced_files": [str(p) for p in produced_files],
            "log_excerpt": str(log_excerpt)[:500],
        }
    )
    manifest["module_history"] = hist
    manifest["last_backfill_at"] = str(finished_at)
    lbm = manifest.get("latest_by_module", {})
    if not isinstance(lbm, dict):
        lbm = {}
    if produced_files:
        lbm[str(module_id)] = str(produced_files[-1])
    manifest["latest_by_module"] = lbm
    manifest["backfill_capabilities"] = _resolve_backfill_capabilities(run_dir, run_type=str(manifest.get("run_type", "unknown")))
    manifest["available_inputs"] = _iter_rel_files(run_dir)
    manifest["available_reports"] = _collect_available_reports(run_dir, manifest["available_inputs"])
    (run_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2))


def _execute_backfill_module(module_id: str, run_dir: Path, stamp: str, log_path: Path, job_id: str) -> list[str]:
    suffix = f"backfill.{module_id}.{stamp}"
    produced_rel: list[str] = []
    if module_id == "manifest_refresh_v1":
        _load_or_build_manifest(run_dir)
        return ["report_manifest.json"]
    if module_id == "rf_summary_v5_v1":
        out_name = f"rf_summary_v5.{suffix}.png"
        cmd = [sys.executable, "make_rf_summary_v5.py", str(run_dir), "--out", out_name]
        rc = _run_command(cmd, log_path, job_id=job_id)
        if rc != 0:
            raise RuntimeError(f"module {module_id} failed with code {rc}")
        p = run_dir / out_name
        if p.exists():
            produced_rel.append(p.relative_to(run_dir).as_posix())
        return produced_rel
    if module_id == "core_reports_v1":
        cmd = [
            sys.executable,
            "backfill_core_reports.py",
            "--output-dir",
            str(run_dir),
            "--output-suffix",
            suffix,
            "--write-canonical",
        ]
        rc = _run_command(cmd, log_path, job_id=job_id)
        if rc != 0:
            raise RuntimeError(f"module {module_id} failed with code {rc}")
        for name in [
            f"paper_style_report.{suffix}.png",
            f"thermal_fields_final.{suffix}.png",
            f"electric_fields.{suffix}.png",
            f"time_series.{suffix}.png",
            "paper_style_report.png",
            "thermal_fields_final.png",
            "electric_fields.png",
            "time_series.png",
        ]:
            if (run_dir / name).exists():
                produced_rel.append(name)
        return produced_rel
    if module_id == "orientation_diagnostics_v1":
        cmd = [
            sys.executable,
            "backfill_orientation_reports.py",
            "--output-dir",
            str(run_dir),
            "--output-suffix",
            suffix,
        ]
        rc = _run_command(cmd, log_path, job_id=job_id)
        if rc != 0:
            raise RuntimeError(f"module {module_id} failed with code {rc}")
        for name in [
            f"orientation_candidates.{suffix}.json",
            f"orientation_candidates.{suffix}.csv",
            f"orientation_angle_summary.{suffix}.json",
            f"orientation_angle_summary.{suffix}.csv",
            f"orientation_angle_effectiveness.{suffix}.png",
        ]:
            p = run_dir / name
            if p.exists():
                produced_rel.append(name)
        return produced_rel
    if module_id == "orientation_gif_v1":
        cmd = [
            sys.executable,
            "backfill_orientation_reports.py",
            "--output-dir",
            str(run_dir),
            "--output-suffix",
            suffix,
            "--gif-only",
            "--gif-duration-s",
            "2.0",
        ]
        rc = _run_command(cmd, log_path, job_id=job_id)
        if rc != 0:
            raise RuntimeError(f"module {module_id} failed with code {rc}")
        name = f"orientation_angle_gallery.{suffix}.gif"
        if (run_dir / name).exists():
            produced_rel.append(name)
        return produced_rel
    if module_id == "placement_reports_v1":
        cmd = [
            sys.executable,
            "backfill_placement_reports.py",
            "--output-dir",
            str(run_dir),
            "--output-suffix",
            suffix,
        ]
        rc = _run_command(cmd, log_path, job_id=job_id)
        if rc != 0:
            raise RuntimeError(f"module {module_id} failed with code {rc}")
        name = f"placement_report.{suffix}.png"
        if (run_dir / name).exists():
            produced_rel.append(name)
        return produced_rel
    if module_id == "shell_sweep_report_v1":
        cmd = [
            sys.executable,
            "backfill_shell_sweep_reports.py",
            "--output-dir",
            str(run_dir),
            "--output-suffix",
            suffix,
        ]
        rc = _run_command(cmd, log_path, job_id=job_id)
        if rc != 0:
            raise RuntimeError(f"module {module_id} failed with code {rc}")
        for name in [f"shell_sweep_report.{suffix}.png", f"shell_sweep_report.{suffix}.svg"]:
            if (run_dir / name).exists():
                produced_rel.append(name)
        return produced_rel
    if module_id == "antennae_reports_v1":
        cmd = [
            sys.executable,
            "backfill_antennae_reports.py",
            "--output-dir",
            str(run_dir),
            "--output-suffix",
            suffix,
        ]
        rc = _run_command(cmd, log_path, job_id=job_id)
        if rc != 0:
            raise RuntimeError(f"module {module_id} failed with code {rc}")
        for name in [f"antennae_report.{suffix}.png", f"antennae_anchors.{suffix}.json"]:
            if (run_dir / name).exists():
                produced_rel.append(name)
        return produced_rel
    raise ValueError(f"unknown backfill module: {module_id}")


def _enqueue_antennae_calibration_request(payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode", "single")).strip()
    if mode not in {"single", "optimizer", "turntable", "orientation_optimizer", "placement_optimizer"}:
        mode = "single"
    shape = str(payload.get("shape", "")).strip()
    if shape and shape not in SUPPORTED_SHAPES:
        raise ValueError("invalid shape")
    output_name = str(payload.get("output_name", "")).strip() or f"antennae_cal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not _is_valid_output_name(output_name):
        raise ValueError("output_name must use only letters, numbers, '_' or '-'")
    if payload.get("antennae_enabled", None) is None:
        payload = dict(payload)
        payload["antennae_enabled"] = True
    job_payload = dict(payload)
    job_payload["mode"] = "antennae_calibrate"
    job_payload["output_name"] = output_name
    job = _make_job(mode="antennae_calibrate", output_name=output_name)
    with JOBS_LOCK:
        JOBS[job["id"]]["_payload"] = job_payload
        JOBS[job["id"]]["progress_label"] = "Queued antennae calibration"
    _enqueue_job(job["id"])
    _maybe_start_next_job()
    with JOBS_LOCK:
        status = str(JOBS[job["id"]]["status"])
        qpos = JOBS[job["id"]].get("queue_position", None)
    return {
        "ok": True,
        "job_id": job["id"],
        "status": status,
        "queue_position": qpos,
        "output_dir": _output_dir_for_request(output_name, payload).relative_to(ROOT).as_posix(),
    }


def _run_backfill_reports_job(payload: dict[str, Any], job: dict[str, Any]) -> None:
    output_dir_rel = str(payload.get("output_dir", "")).strip()
    target = _safe_workspace_path(output_dir_rel)
    if target is None or not target.exists() or not target.is_dir():
        raise FileNotFoundError(output_dir_rel or "(missing output_dir)")
    try:
        target.relative_to(OUTPUTS_DIR.resolve())
    except ValueError as exc:
        raise ValueError("output_dir must be under outputs_eqs") from exc

    manifest = _load_or_build_manifest(target)
    run_type = str(manifest.get("run_type", _get_run_type(target)))
    capabilities = [str(v) for v in manifest.get("backfill_capabilities", []) if str(v)]
    req_modules = payload.get("modules", [])
    requested = [str(v).strip() for v in req_modules] if isinstance(req_modules, list) else []
    requested = [m for m in requested if m]
    if requested:
        modules = [m for m in requested if m in capabilities]
    else:
        modules = list(capabilities)
    if not modules:
        raise ValueError("no compatible backfill modules for target output_dir")

    with JOBS_LOCK:
        job["run_type"] = run_type
        job["requested_modules"] = requested
        job["resolved_modules"] = modules
        job["backfill_output_dir"] = target.relative_to(ROOT).as_posix()
    _set_job_progress(str(job["id"]), total_runs=len(modules), completed_runs=0, progress_pct=0.0, progress_label="Backfill queued")

    produced_all: list[str] = []
    failed: list[dict[str, Any]] = []
    for idx, module_id in enumerate(modules, start=1):
        if _is_cancel_requested(str(job["id"])):
            raise JobCancelled("cancelled by user")
        _set_job_progress(
            str(job["id"]),
            progress_pct=100.0 * float(idx - 1) / float(max(len(modules), 1)),
            progress_label=f"Backfill {idx}/{len(modules)}: {module_id}",
        )
        started = _now_iso()
        stamp = _stamp_tag()
        try:
            produced_rel = _execute_backfill_module(module_id, target, stamp, Path(job["log_path"]), str(job["id"]))
            status = "completed"
            log_excerpt = f"module {module_id} completed"
            produced_all.extend([f"outputs_eqs/{target.relative_to(OUTPUTS_DIR).as_posix()}/{p}" for p in produced_rel])
        except JobCancelled:
            raise
        except Exception as exc:
            produced_rel = []
            status = "failed"
            log_excerpt = str(exc)
            failed.append({"module_id": module_id, "error": str(exc)})
            _write_job_log(Path(job["log_path"]), f"[backfill:{module_id}] failed: {exc}")
        _append_manifest_history(
            target,
            module_id=module_id,
            started_at=started,
            finished_at=_now_iso(),
            status=status,
            produced_files=produced_rel,
            log_excerpt=log_excerpt,
        )
        _set_job_progress(str(job["id"]), completed_runs=idx, progress_pct=100.0 * float(idx) / float(max(len(modules), 1)))

    with JOBS_LOCK:
        job["backfill_summary"] = {
            "run_type": run_type,
            "requested_modules": requested,
            "resolved_modules": modules,
            "failed_modules": failed,
            "created_files": produced_all,
        }
    if len(failed) == len(modules):
        raise RuntimeError("all backfill modules failed")


def _enqueue_backfill_job_request(payload: dict[str, Any]) -> dict[str, Any]:
    output_dir = str(payload.get("output_dir", "")).strip()
    if not output_dir:
        raise ValueError("missing output_dir")
    target = _safe_workspace_path(output_dir)
    if target is None or not target.exists() or not target.is_dir():
        raise FileNotFoundError(output_dir)
    try:
        target.relative_to(OUTPUTS_DIR.resolve())
    except ValueError as exc:
        raise ValueError("output_dir must be under outputs_eqs") from exc
    if not _is_backfill_target_dir(target):
        raise ValueError("output_dir is not a recognized run/report directory")
    run_type = _get_run_type(target)
    caps = _resolve_backfill_capabilities(target, run_type=run_type)
    req_modules = payload.get("modules", [])
    requested_modules = [str(v).strip() for v in req_modules] if isinstance(req_modules, list) else []
    requested_modules = [m for m in requested_modules if m]
    queued_modules = [m for m in requested_modules if m in caps] if requested_modules else list(caps)
    if not queued_modules:
        raise ValueError("no compatible backfill modules")
    rel_out = target.relative_to(ROOT).as_posix()
    job_payload = {
        "mode": "backfill_reports",
        "output_dir": rel_out,
        "modules": requested_modules,
        "requested_mode": str(payload.get("mode", "quick")),
    }
    job = _make_job(mode="backfill_reports", output_name=target.name)
    with JOBS_LOCK:
        JOBS[job["id"]]["_payload"] = job_payload
        JOBS[job["id"]]["progress_label"] = "Queued backfill job"
    _enqueue_job(job["id"])
    _maybe_start_next_job()
    with JOBS_LOCK:
        status = str(JOBS[job["id"]]["status"])
        qpos = JOBS[job["id"]].get("queue_position", None)
    return {
        "ok": True,
        "job_id": job["id"],
        "status": status,
        "queue_position": qpos,
        "run_type": run_type,
        "requested_modules": requested_modules,
        "queued_modules": queued_modules,
    }


def _result_detail(name: str) -> dict[str, Any]:
    target = (OUTPUTS_DIR / name).resolve()
    try:
        target.relative_to(OUTPUTS_DIR.resolve())
    except ValueError as exc:
        raise FileNotFoundError(name) from exc
    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(name)
    if _should_skip_result_path(target):
        raise FileNotFoundError(name)

    summary = {}
    summary_path = target / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {"error": "Failed to parse summary.json"}

    images = _collect_images_recursive(target)
    subdirs = []
    for d in sorted([p for p in target.iterdir() if p.is_dir()]):
        if _should_skip_result_path(d):
            continue
        rel = d.relative_to(OUTPUTS_DIR).as_posix()
        subdirs.append(rel)
    manifest = _load_or_build_manifest(target)
    return {
        "name": name,
        "summary": summary,
        "images": images,
        "subdirs": subdirs,
        "run_type": manifest.get("run_type", _get_run_type(target)),
        "backfill_capabilities": manifest.get("backfill_capabilities", []),
        "has_report_manifest": bool((target / "report_manifest.json").exists()),
    }


def _summary_excerpt(summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def _pick(*names: str) -> float | None:
        for n in names:
            if n in summary:
                try:
                    return float(summary[n])
                except (TypeError, ValueError):
                    return None
        return None

    # Normalize to stable frontend keys.
    if (v := _pick("max_T_part_final_c", "max_T_part_c")) is not None:
        out["max_T_part_c"] = v
    if (v := _pick("mean_T_part_final_c", "mean_T_part_c")) is not None:
        out["mean_T_part_c"] = v
    if (v := _pick("mean_phi_part_final", "mean_phi_part", "frac_part_ge_melt_ref")) is not None:
        out["mean_phi_part"] = v
    if (v := _pick("mean_rho_rel_part_final", "mean_rho_rel_part")) is not None:
        out["mean_rho_rel_part"] = v
    if (v := _pick("exposure_time_s", "t_final_s")) is not None:
        out["t_final_s"] = v
    fam = summary.get("model_family", None)
    if fam is not None:
        out["model_family"] = str(fam)
    cal = summary.get("calibration_version", None)
    if cal is not None:
        out["calibration_version"] = str(cal)
    ab = summary.get("ab_bucket_id", None)
    if ab is not None:
        out["ab_bucket_id"] = str(ab)

    return out


def _preferred_images(root: Path, limit: int = 3) -> list[dict[str, str]]:
    preferred = [
        "turntable_electric.gif",
        "turntable_thermal.gif",
        "turntable_density.gif",
        "electric_field_evolution.gif",
        "thermal_evolution.gif",
        "density_evolution.gif",
        "optimizer_report.png",
        "shell_sweep_report.png",
        "shell_sweep_report.svg",
        "paper_style_report.png",
        "rf_summary_v5.png",
        "time_series.png",
        "thermal_fields_final.png",
        "electric_fields.png",
    ]
    found: list[dict[str, str]] = []
    for name in preferred:
        p = root / name
        if p.exists():
            rel_to_outputs = p.relative_to(OUTPUTS_DIR).as_posix()
            found.append({"path": p.name, "url": _url_for_output_rel(rel_to_outputs)})
        if len(found) >= limit:
            return found
    for pattern in ("*.gif", "*.png"):
        for img in sorted(root.rglob(pattern)):
            rel = img.relative_to(root).as_posix()
            if any(it["path"] == rel for it in found):
                continue
            rel_out = img.relative_to(OUTPUTS_DIR).as_posix()
            found.append({"path": rel, "url": _url_for_output_rel(rel_out)})
            if len(found) >= limit:
                return found
    return found


def _collect_run_cards() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    stars = _load_run_stars()
    for p in sorted(OUTPUTS_DIR.rglob("*"), key=lambda q: q.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue
        if _should_skip_result_path(p):
            continue
        has_summary = (p / "summary.json").exists()
        has_media = any(p.glob("*.png")) or any(p.glob("*.gif")) or any(p.glob("*.svg"))
        if not (has_summary or has_media):
            continue

        rel = p.relative_to(OUTPUTS_DIR).as_posix()
        summary: dict[str, Any] = {}
        if has_summary:
            try:
                summary = json.loads((p / "summary.json").read_text())
            except Exception:
                summary = {}
        images = _collect_images_recursive(p)
        hero = _preferred_images(p, limit=3)
        if rel.startswith("runs/"):
            toks = rel.split("/")
            group = toks[1] if len(toks) > 1 else "runs"
        else:
            group = rel.split("/")[0] if "/" in rel else "runs"
        manifest = _load_or_build_manifest(p)
        run_type = str(manifest.get("run_type", _get_run_type(p)))
        caps = [str(v) for v in manifest.get("backfill_capabilities", []) if str(v)]
        cards.append({
            "name": rel,
            "group": group,
            "run_created_at": _infer_run_created_at(p),
            "updated_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            "summary_excerpt": _summary_excerpt(summary),
            "run_type": run_type,
            "backfill_capabilities": caps,
            "has_report_manifest": bool((p / "report_manifest.json").exists()),
            "can_backfill_orientation": "orientation_diagnostics_v1" in caps,
            "starred": bool(stars.get(rel, False)),
            "hero_images": hero,
            "image_count": len(images),
            "images": images,
        })
    return cards


def _pick_preview_image(folder: Path, preferred: list[str]) -> str | None:
    for n in preferred:
        p = folder / n
        if p.exists():
            return n
    candidates = sorted([p.name for p in folder.glob("*.png")])
    return candidates[0] if candidates else None


def _examples_payload() -> dict[str, Any]:
    master_sweep = None
    master = OUTPUTS_DIR / "_sweep_time_sweep.png"
    if master.exists():
        master_sweep = "/files/outputs_eqs/_sweep_time_sweep.png"

    def _reorg_mapped_dirs(prefixes: list[str]) -> list[Path]:
        out: list[Path] = []
        reports = sorted(OUTPUTS_DIR.glob("_reorg_report_*.json")) + sorted(OUTPUTS_DIR.glob("_reorg_topdirs_*.json"))
        wants = tuple(prefixes)
        for rp in reports:
            try:
                data = json.loads(rp.read_text())
            except Exception:
                continue
            moved = data.get("moved", [])
            if not isinstance(moved, list):
                continue
            for row in moved:
                frm = ""
                to = ""
                if isinstance(row, dict):
                    frm = str(row.get("from", ""))
                    to = str(row.get("to", ""))
                elif isinstance(row, (list, tuple)) and len(row) >= 2:
                    frm = str(row[0])
                    to = str(row[1])
                if not frm or not to:
                    continue
                if not frm.startswith(wants):
                    continue
                p = (OUTPUTS_DIR / to).resolve()
                try:
                    p.relative_to(OUTPUTS_DIR.resolve())
                except ValueError:
                    continue
                if p.exists() and p.is_dir():
                    out.append(p)
        return out

    def _collect_examples(roots: list[Path], preferred: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen_dirs: set[str] = set()
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for d in sorted([p for p in root.iterdir() if p.is_dir()]):
                try:
                    rel_dir = d.relative_to(OUTPUTS_DIR).as_posix()
                except Exception:
                    continue
                if rel_dir in seen_dirs:
                    continue
                seen_dirs.add(rel_dir)
                preview = _pick_preview_image(d, preferred)
                images = [p.name for p in sorted(d.glob("*.png"))]
                rows.append({
                    "name": d.name,
                    "rel_dir": rel_dir,
                    "preview": preview,
                    "images": images,
                })
        return rows

    sweep_roots = [OUTPUTS_DIR / "sweeps"] + sorted((OUTPUTS_DIR / "runs").rglob("sweeps")) + _reorg_mapped_dirs(["sweeps/"])
    shape_roots = [OUTPUTS_DIR / "shapes"] + sorted((OUTPUTS_DIR / "runs").rglob("shapes")) + _reorg_mapped_dirs(["shapes/"])
    sweeps = _collect_examples(sweep_roots, ["paper_style_report.png", "time_series.png", "thermal_fields_final.png"])
    shapes = _collect_examples(shape_roots, ["rf_summary_v5.png", "paper_style_report.png", "time_series.png"])

    return {
        "master_sweep": master_sweep,
        "sweeps": sweeps,
        "shapes": shapes,
    }


def _experimental_model_info() -> dict[str, Any]:
    dsc: dict[str, Any] = {}
    prov: dict[str, Any] = {}
    try:
        dsc = _load_yaml(EXPERIMENTAL_DSC_PROFILE)
    except Exception:
        dsc = {}
    try:
        prov = _load_yaml(EXPERIMENTAL_PROVENANCE)
    except Exception:
        prov = {}

    dsc_sources = dsc.get("sources", []) if isinstance(dsc.get("sources", []), list) else []
    prov_sources = prov.get("sources", {}) if isinstance(prov.get("sources", {}), dict) else {}
    prov_ref_list: list[dict[str, str]] = []
    for key, val in prov_sources.items():
        if not isinstance(val, dict):
            continue
        prov_ref_list.append({
            "key": str(key),
            "citation": str(val.get("citation", "")),
            "url": str(val.get("url", "")),
            "path": str(val.get("path", "")),
        })

    return {
        "default_family": "experimental_pa12_hybrid",
        "baseline_family": "baseline",
        "dsc_profile_file": "configs/experimental_pa12_dsc_profile.yaml",
        "provenance_file": "configs/experimental_pa12_provenance.yaml",
        "dsc_profile": {
            "version": dsc.get("version", ""),
            "melt_onset_c": dsc.get("melt_onset_c"),
            "melt_peak_c": dsc.get("melt_peak_c"),
            "melt_end_c": dsc.get("melt_end_c"),
            "lat_heat_j_per_kg": dsc.get("lat_heat_j_per_kg"),
            "notes": dsc.get("notes", []),
            "sources": dsc_sources,
        },
        "provenance": {
            "version": prov.get("version", ""),
            "calibration_version": prov.get("calibration_version", ""),
            "sources": prov_ref_list,
        },
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "rfam-gui/0.2"

    def _json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _text(self, text: str, status: int = 200, ctype: str = "text/plain; charset=utf-8") -> None:
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        if path == "/" or path == "/index.html":
            return self._serve_static("index.html")
        if path == "/files" or path == "/files.html":
            return self._serve_static("files.html")
        if path == "/results" or path == "/results.html":
            return self._serve_static("results.html")
        if path == "/theory" or path == "/theory.html":
            return self._serve_static("theory.html")
        if path == "/examples" or path == "/examples.html":
            return self._serve_static("examples.html")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if path == "/api/meta":
            model_info = _experimental_model_info()
            return self._json({
                "shape_options": SUPPORTED_SHAPES,
                "base_configs": _list_base_configs(),
                "model_families": ["baseline", "experimental_pa12_hybrid"],
                "default_model_family": model_info.get("default_family", "experimental_pa12_hybrid"),
                "default_provenance_file": model_info.get("provenance_file", "configs/experimental_pa12_provenance.yaml"),
                "default_dsc_profile_file": model_info.get("dsc_profile_file", "configs/experimental_pa12_dsc_profile.yaml"),
                "model_info": model_info,
            })
        if path == "/api/config-preview":
            qs = parse_qs(urlparse(self.path).query)
            name = (qs.get("name", [""])[0] or "").strip()
            if not name:
                return self._json({"error": "missing name"}, status=400)
            try:
                return self._json(_config_preview(name))
            except FileNotFoundError:
                return self._json({"error": "not found"}, status=404)
        if path == "/api/jobs":
            with JOBS_LOCK:
                jobs = []
                for j in JOBS.values():
                    out = {k: v for k, v in j.items() if not str(k).startswith("_")}
                    jobs.append(out)
            for j in jobs:
                _refresh_job_artifacts(j)
            def _job_sort_key(j: dict[str, Any]) -> tuple[int, str]:
                status = str(j.get("status", ""))
                if status == "running":
                    return (0, str(j.get("started_at") or j.get("created_at") or ""))
                if status == "paused":
                    # Paused running job stays ahead of queued jobs.
                    if str(j.get("pause_state", "")) == "running":
                        return (1, str(j.get("started_at") or j.get("created_at") or ""))
                    pos = int(j.get("queue_position") or 999999)
                    return (2, f"{pos:06d}")
                if status == "queued":
                    pos = int(j.get("queue_position") or 999999)
                    return (3, f"{pos:06d}")
                ended = str(j.get("ended_at") or "")
                return (4, ended)
            jobs.sort(key=_job_sort_key)
            return self._json(jobs)
        if path == "/api/results":
            return self._json(_collect_results())
        if path == "/api/fs/list":
            qs = parse_qs(urlparse(self.path).query)
            rel = (qs.get("path", ["."])[0] or ".").strip()
            try:
                return self._json(_fs_list(rel))
            except FileNotFoundError:
                return self._json({"error": "path not found"}, status=404)
        if path == "/api/results-runview":
            return self._json(_collect_run_cards())
        if path == "/api/examples":
            return self._json(_examples_payload())
        if path.startswith("/api/results/"):
            name = unquote(path[len("/api/results/"):])
            try:
                return self._json(_result_detail(name))
            except FileNotFoundError:
                return self._json({"error": "not found"}, status=404)
        if path.startswith("/files/"):
            rel = path[len("/files/"):]
            safe = _safe_rel_path(rel)
            if safe is None or not safe.exists() or not safe.is_file():
                return self._text("not found", status=404)
            return self._serve_file(safe)

        self._text("not found", status=404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {
            "/api/run",
            "/api/match-config",
            "/api/effective-config",
            "/api/queue/reorder",
            "/api/job/control",
            "/api/results/delete",
            "/api/results/star",
            "/api/fs/delete",
            "/api/fs/move",
            "/api/tools/backfill-reports",
            "/api/tools/backfill-orientation",
            "/api/tools/antennae-preview",
            "/api/tools/antennae-calibrate",
        }:
            return self._json({"error": "not found"}, status=404)

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._json({"error": "invalid json"}, status=400)

        if path == "/api/queue/reorder":
            job_id = str(payload.get("job_id", "")).strip()
            direction = str(payload.get("direction", "")).strip().lower()
            target_index_raw = payload.get("target_index", None)
            if not job_id:
                return self._json({"error": "missing job_id"}, status=400)

            with QUEUE_LOCK:
                if job_id not in JOB_QUEUE:
                    return self._json({"error": "job is not queued"}, status=400)
                cur = JOB_QUEUE.index(job_id)
                if target_index_raw is not None:
                    try:
                        target_index = int(target_index_raw)
                    except (TypeError, ValueError):
                        return self._json({"error": "invalid target_index"}, status=400)
                else:
                    if direction == "up":
                        target_index = cur - 1
                    elif direction == "down":
                        target_index = cur + 1
                    else:
                        return self._json({"error": "provide direction up/down or target_index"}, status=400)

            ok = _reorder_queued_job(job_id, target_index)
            if not ok:
                return self._json({"error": "job is not queued"}, status=400)
            with QUEUE_LOCK:
                queue_ids = list(JOB_QUEUE)
            return self._json({"ok": True, "queue": queue_ids})

        if path == "/api/job/control":
            job_id = str(payload.get("job_id", "")).strip()
            action = str(payload.get("action", "")).strip().lower()
            if not job_id:
                return self._json({"error": "missing job_id"}, status=400)
            if action not in {"pause", "resume", "cancel"}:
                return self._json({"error": "invalid action"}, status=400)

            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None:
                    return self._json({"error": "job not found"}, status=404)
                status = str(job.get("status", ""))
                pause_state = str(job.get("pause_state", "") or "")

            if action == "pause":
                if status == "queued":
                    removed = _remove_from_queue(job_id)
                    if not removed:
                        return self._json({"error": "job is not queued"}, status=400)
                    with JOBS_LOCK:
                        JOBS[job_id]["status"] = "paused"
                        JOBS[job_id]["pause_state"] = "queued"
                        JOBS[job_id]["progress_label"] = "Paused in queue"
                    return self._json({"ok": True, "status": "paused"})
                if status == "running":
                    ok, msg = _pause_running_job(job_id)
                    if not ok:
                        return self._json({"error": msg}, status=400)
                    return self._json({"ok": True, "status": "paused"})
                return self._json({"error": f"cannot pause job in status '{status}'"}, status=400)

            if action == "resume":
                if status != "paused":
                    return self._json({"error": f"cannot resume job in status '{status}'"}, status=400)
                if pause_state == "running":
                    ok, msg = _resume_paused_running_job(job_id)
                    if not ok:
                        return self._json({"error": msg}, status=400)
                    return self._json({"ok": True, "status": "running"})
                # paused while queued -> enqueue again
                with JOBS_LOCK:
                    JOBS[job_id]["status"] = "queued"
                    JOBS[job_id]["pause_state"] = None
                    JOBS[job_id]["progress_label"] = "Queued (resumed)"
                _enqueue_job(job_id)
                _maybe_start_next_job()
                with JOBS_LOCK:
                    cur_status = str(JOBS[job_id]["status"])
                return self._json({"ok": True, "status": cur_status})

            # cancel
            if status == "queued":
                _remove_from_queue(job_id)
                _finish_job(job_id, "cancelled", "cancelled in queue")
                return self._json({"ok": True, "status": "cancelled"})
            if status == "paused" and pause_state == "queued":
                _finish_job(job_id, "cancelled", "cancelled while paused in queue")
                return self._json({"ok": True, "status": "cancelled"})
            if status in {"running", "paused", "cancelling"}:
                ok, msg = _cancel_running_job(job_id)
                if not ok:
                    # If process already ended, let worker mark final state soon.
                    with JOBS_LOCK:
                        if JOBS.get(job_id, {}).get("status") not in {"completed", "failed", "cancelled"}:
                            JOBS[job_id]["status"] = "cancelling"
                            JOBS[job_id]["progress_label"] = "Cancelling..."
                    return self._json({"ok": True, "status": "cancelling", "note": msg})
                return self._json({"ok": True, "status": "cancelling"})
            return self._json({"error": f"cannot cancel job in status '{status}'"}, status=400)

        if path == "/api/results/delete":
            run_name = str(payload.get("run_name", "")).strip().strip("/")
            if not run_name:
                return self._json({"error": "missing run_name"}, status=400)
            target = (OUTPUTS_DIR / run_name).resolve()
            try:
                target.relative_to(OUTPUTS_DIR.resolve())
            except ValueError:
                return self._json({"error": "invalid run_name"}, status=400)
            if not target.exists() or not target.is_dir():
                return self._json({"error": "run not found"}, status=404)
            if target == OUTPUTS_DIR:
                return self._json({"error": "cannot delete outputs root"}, status=400)
            if _should_skip_result_path(target):
                return self._json({"error": "cannot delete protected directory"}, status=400)
            try:
                shutil.rmtree(target)
            except Exception as exc:
                return self._json({"error": str(exc)}, status=400)
            try:
                _set_run_star(run_name, False)
            except Exception:
                pass
            return self._json({"ok": True, "deleted": run_name})

        if path == "/api/results/star":
            run_name = str(payload.get("run_name", "")).strip().strip("/")
            if not run_name:
                return self._json({"error": "missing run_name"}, status=400)
            starred = bool(payload.get("starred", True))
            target = (OUTPUTS_DIR / run_name).resolve()
            try:
                target.relative_to(OUTPUTS_DIR.resolve())
            except ValueError:
                return self._json({"error": "invalid run_name"}, status=400)
            if not target.exists() or not target.is_dir():
                return self._json({"error": "run not found"}, status=404)
            if _should_skip_result_path(target):
                return self._json({"error": "cannot star protected directory"}, status=400)
            try:
                _set_run_star(run_name, starred)
            except Exception as exc:
                return self._json({"error": str(exc)}, status=400)
            return self._json({"ok": True, "run_name": run_name, "starred": starred})

        if path == "/api/fs/delete":
            rel = str(payload.get("path", "")).strip()
            recursive = bool(payload.get("recursive", False))
            target = _safe_workspace_path(rel)
            if target is None or not target.exists():
                return self._json({"error": "path not found"}, status=404)
            if target == ROOT:
                return self._json({"error": "cannot delete workspace root"}, status=400)
            try:
                if target.is_dir():
                    if not recursive:
                        return self._json({"error": "directory delete requires recursive=true"}, status=400)
                    shutil.rmtree(target)
                else:
                    target.unlink()
            except Exception as exc:
                return self._json({"error": str(exc)}, status=400)
            return self._json({"ok": True})

        if path == "/api/fs/move":
            src_rel = str(payload.get("src", "")).strip()
            dst_rel = str(payload.get("dst", "")).strip()
            src = _safe_workspace_path(src_rel)
            dst = _safe_workspace_path(dst_rel)
            if src is None or dst is None:
                return self._json({"error": "invalid src or dst"}, status=400)
            if not src.exists():
                return self._json({"error": "src not found"}, status=404)
            if src == ROOT or dst == ROOT:
                return self._json({"error": "cannot move workspace root"}, status=400)
            if dst.exists():
                return self._json({"error": "dst already exists"}, status=400)
            if not dst.parent.exists():
                return self._json({"error": "dst parent does not exist"}, status=400)
            try:
                shutil.move(str(src), str(dst))
            except Exception as exc:
                return self._json({"error": str(exc)}, status=400)
            return self._json({"ok": True})

        if path == "/api/tools/backfill-reports":
            try:
                resp = _enqueue_backfill_job_request(payload)
                return self._json(resp, status=202)
            except FileNotFoundError:
                return self._json({"error": "output_dir not found"}, status=404)
            except Exception as exc:
                return self._json({"error": str(exc)}, status=400)

        if path == "/api/tools/backfill-orientation":
            output_dir = str(payload.get("output_dir", "")).strip()
            if not output_dir:
                return self._json({"error": "missing output_dir"}, status=400)
            try:
                alias_payload = {
                    "output_dir": output_dir,
                    "modules": ["orientation_diagnostics_v1"],
                    "mode": "quick",
                }
                resp = _enqueue_backfill_job_request(alias_payload)
                return self._json({**resp, "deprecated_endpoint": True}, status=202)
            except FileNotFoundError:
                return self._json({"error": "output_dir not found"}, status=404)
            except Exception as exc:
                return self._json({"error": str(exc)}, status=400)

        if path == "/api/tools/antennae-preview":
            try:
                mode = str(payload.get("mode", "single")).strip()
                if mode not in {"single", "optimizer", "turntable", "orientation_optimizer", "placement_optimizer"}:
                    mode = "single"
                exposure_minutes = float(payload.get("exposure_minutes", 6.0))
                if exposure_minutes <= 0:
                    exposure_minutes = 6.0
                effective = _effective_config_for_request(payload, mode, exposure_minutes)
                cfg_text = str(effective.get("effective_config_text", "")).strip()
                cfg = yaml.safe_load(cfg_text) if cfg_text else {}
                if not isinstance(cfg, dict):
                    cfg = {}
                if not bool(cfg.get("antennae", {}).get("enabled", False)):
                    return self._json({"ok": False, "error": "antennae must be enabled for preview"}, status=400)
                from rfam_eqs_coupled import generate_antennae_preview  # local import keeps server startup fast
                stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                out_png = PREVIEW_DIR / f"antennae_preview.{stamp}.{uuid.uuid4().hex[:6]}.png"
                out = generate_antennae_preview(cfg, out_png)
                if not bool(out.get("ok", False)):
                    return self._json({"ok": False, "error": str(out.get("error", "preview failed"))}, status=400)
                try:
                    rel_out = out_png.resolve().relative_to(OUTPUTS_DIR.resolve()).as_posix()
                    image_url = _url_for_output_rel(rel_out)
                except Exception:
                    rel = out_png.resolve().relative_to(ROOT.resolve()).as_posix()
                    image_url = "/files/" + quote(rel, safe="/")
                rows = out.get("instances", [])
                if not isinstance(rows, list):
                    rows = []
                part_counts: dict[str, int] = {}
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    pid = int(r.get("part_idx", 0))
                    k = f"part_{pid}"
                    part_counts[k] = int(part_counts.get(k, 0)) + 1
                ant_cfg = cfg.get("antennae", {}) if isinstance(cfg.get("antennae", {}), dict) else {}
                return self._json({
                    "ok": True,
                    "image_url": image_url,
                    "n_instances": int(out.get("n_instances", len(rows))),
                    "instances": rows,
                    "part_counts": part_counts,
                    "size_mode": str(ant_cfg.get("size_mode", "")),
                })
            except Exception as exc:
                return self._json({"ok": False, "error": str(exc)}, status=400)

        if path == "/api/tools/antennae-calibrate":
            try:
                resp = _enqueue_antennae_calibration_request(payload)
                return self._json(resp, status=202)
            except Exception as exc:
                return self._json({"error": str(exc)}, status=400)

        mode = str(payload.get("mode", "single"))
        if mode not in {"single", "sweep", "optimizer", "turntable", "orientation_optimizer", "placement_optimizer", "shell_sweep"}:
            return self._json({"error": "invalid mode"}, status=400)

        shape = str(payload.get("shape", "")).strip()
        if shape and shape not in SUPPORTED_SHAPES:
            return self._json({"error": "invalid shape"}, status=400)

        if path == "/api/match-config":
            try:
                if mode == "sweep":
                    mins = _to_float_list(payload.get("sweep_minutes", [6, 7, 8, 9, 10]))
                    if not mins:
                        raise ValueError("empty sweep")
                    rows = []
                    for m in mins:
                        req_sig = _signature_for_request(payload, "single", float(m))
                        resolved = _resolve_base_config_for_signature(req_sig, forced_base_config=str(payload.get("base_config", "")).strip() or None)
                        if _has_advanced_overrides(payload):
                            resolved = dict(resolved)
                            resolved["match_type"] = "new_from_overrides"
                        if _is_experimental_request(payload):
                            resolved = dict(resolved)
                            resolved["match_type"] = "new_from_dsc_model"
                        rows.append({"minutes": float(m), "requested": req_sig, "resolved": resolved})
                    return self._json({"mode": mode, "matches": rows})
                req_mode = "single" if mode == "shell_sweep" else mode
                minutes = float(payload.get("exposure_minutes", 6.0))
                req_sig = _signature_for_request(payload, req_mode, minutes)
                resolved = _resolve_base_config_for_signature(req_sig, forced_base_config=str(payload.get("base_config", "")).strip() or None)
                if _has_advanced_overrides(payload):
                    resolved = dict(resolved)
                    resolved["match_type"] = "new_from_overrides"
                if _is_experimental_request(payload):
                    resolved = dict(resolved)
                    resolved["match_type"] = "new_from_dsc_model"
                return self._json({"mode": mode, "requested": req_sig, "resolved": resolved})
            except Exception as exc:
                return self._json({"error": str(exc)}, status=400)

        if path == "/api/effective-config":
            try:
                if mode == "sweep":
                    mins = _to_float_list(payload.get("sweep_minutes", [6, 7, 8, 9, 10]))
                    if not mins:
                        raise ValueError("empty sweep")
                    minutes = float(mins[0])
                else:
                    minutes = float(payload.get("exposure_minutes", 6.0))
                if minutes <= 0:
                    raise ValueError("exposure_minutes must be > 0")
                eff_mode = mode
                if mode in {"sweep", "shell_sweep"}:
                    eff_mode = "single"
                return self._json(_effective_config_for_request(payload, eff_mode, minutes))
            except Exception as exc:
                return self._json({"error": str(exc)}, status=400)

        output_name = str(payload.get("output_name", "")).strip()
        job = _make_job(mode=mode, output_name=output_name)
        with JOBS_LOCK:
            JOBS[job["id"]]["_payload"] = payload
        _enqueue_job(job["id"])
        _maybe_start_next_job()
        with JOBS_LOCK:
            status = str(JOBS[job["id"]]["status"])
            qpos = JOBS[job["id"]].get("queue_position", None)
        self._json({"job_id": job["id"], "status": status, "queue_position": qpos}, status=202)

    def _serve_static(self, rel_path: str) -> None:
        rel = Path(rel_path)
        if rel.is_absolute() or ".." in rel.parts:
            return self._text("not found", status=404)
        target = (STATIC_DIR / rel).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return self._text("not found", status=404)
        if not target.exists() or not target.is_file():
            return self._text("not found", status=404)
        self._serve_file(target)

    def _serve_file(self, target: Path) -> None:
        suffix = target.suffix.lower()
        if suffix == ".html":
            ctype = "text/html; charset=utf-8"
        elif suffix == ".css":
            ctype = "text/css; charset=utf-8"
        elif suffix == ".js":
            ctype = "application/javascript; charset=utf-8"
        elif suffix == ".json":
            ctype = "application/json; charset=utf-8"
        elif suffix in {".txt", ".md", ".csv", ".tsv"}:
            ctype = "text/plain; charset=utf-8"
        elif suffix == ".pdf":
            ctype = "application/pdf"
        elif suffix == ".png":
            ctype = "image/png"
        elif suffix == ".gif":
            ctype = "image/gif"
        elif suffix in {".jpg", ".jpeg"}:
            ctype = "image/jpeg"
        elif suffix == ".svg":
            ctype = "image/svg+xml"
        elif suffix == ".yaml":
            ctype = "application/x-yaml; charset=utf-8"
        elif suffix == ".log":
            ctype = "text/plain; charset=utf-8"
        else:
            ctype = "application/octet-stream"

        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        sys.stderr.write(f"[{stamp}] {self.address_string()} {fmt % args}\n")


def main() -> None:
    host = os.environ.get("RFAM_GUI_HOST", "127.0.0.1")
    port = int(os.environ.get("RFAM_GUI_PORT", "8080"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"RFAM GUI running at http://{host}:{port}")
    print("Prewarp flows are disabled in this interface.")
    server.serve_forever()


if __name__ == "__main__":
    main()
