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
from shapes import make_shape_from_svg

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
    if s == "gt_logo":
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
    if s == "gt_logo":
        svg_name = "Georgia_Tech_Yellow_Jackets_logo.svg"
        target_w = float(part.get("width", 0.020))
        target_h = float(part.get("height", target_w))
        cx = float(part.get("center_x", 0.0))
        cy = float(part.get("center_y", 0.0))
        rot = float(part.get("rotation_deg", 0.0))

        # Raster-contour approach: rasterise all SVG paths onto a bitmap and
        # extract the single outer contour.  This produces the same kind of
        # polygon that every other shape uses (circle, star, L-shape, …) and
        # avoids the multi-piece boolean-union issues of the vector-native path.
        # The resulting boundary has natural pixel-level roughness ("lumpy") at
        # the raster resolution, which is physically appropriate.
        poly = make_shape_from_svg(
            str((ROOT / svg_name).resolve()),
            target_width_m=target_w,
            target_height_m=target_h,
            n_pts=800,
            curve_pts=20,
            raster_res=1024,
        )

        geom.pop("parts", None)
        geom.pop("boolean", None)
        part["shape"] = "polygon"
        part["width"] = target_w
        part["height"] = target_h
        part["center_x"] = cx
        part["center_y"] = cy
        part["rotation_deg"] = rot
        part["n_circle_pts"] = int(part.get("n_circle_pts", 800))
        part["polygon_points"] = [[float(x), float(y)] for x, y in poly]
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
    if (v := _num("part_rotation_deg")) is not None:
        g.setdefault("part", {})["rotation_deg"] = v

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
        "max_antennae_per_part": int(payload.get("antennae_max_per_part", ant_cfg.get("placement", {}).get("max_antennae_per_part", 1) if isinstance(ant_cfg.get("placement", {}), dict) else 1)),
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

    # Handle explicitly placed antenna instances from the interactive workshop.
    # When provided, these bypass the EQS auto-placement solve entirely.
    explicit = payload.get("antennae_explicit_instances", None)
    if explicit is not None and isinstance(explicit, list):
        from rfam_eqs_coupled import _parts_from_geometry
        geom = cfg.get("geometry", {})
        if isinstance(geom, dict):
            parts = _parts_from_geometry(geom)
            for p in parts:
                if isinstance(p, dict):
                    p["antennae_instances"] = []
            for inst in explicit:
                if not isinstance(inst, dict):
                    continue
                pid = int(inst.get("part_id", 1)) - 1
                if pid < 0:
                    pid = 0
                if pid < len(parts) and isinstance(parts[pid], dict):
                    if not isinstance(parts[pid].get("antennae_instances"), list):
                        parts[pid]["antennae_instances"] = []
                    size_fallback = float(inst.get("size_mm", 1.0))
                    parts[pid]["antennae_instances"].append({
                        "center_x": float(inst.get("center_x", 0.0)),
                        "center_y": float(inst.get("center_y", 0.0)),
                        "size_mm": size_fallback,
                        "size_x_mm": float(inst.get("size_x_mm", size_fallback)),
                        "size_y_mm": float(inst.get("size_y_mm", size_fallback)),
                        "anchor_x": float(inst.get("anchor_x", inst.get("center_x", 0.0))),
                        "anchor_y": float(inst.get("anchor_y", inst.get("center_y", 0.0))),
                    })
            geom["_antennae_resolved"] = True


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
    # Skip individual fgm_iterate child dirs (iterN and opt_probe) — exposed via parent dashboard
    # e.g. runs/square/fgm_iterate/square_run_20260430/square_run_20260430_iter2  → skip
    # e.g. runs/square/fgm_iterate/square_run_20260430/square_run_20260430_opt_probe  → skip
    import re as _re
    if "/fgm_iterate/" in path.as_posix() and _re.search(r"(_iter\d+|_opt_probe)$", path.name):
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
    placement = None
    placement_path = output_dir / "antennae_placement.json"
    if placement_path.exists():
        try:
            placement = json.loads(placement_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    artifact = {
        "output_dir": rel,
        "summary": summary,
        "images": images,
    }
    if placement is not None:
        artifact["antennae_placement"] = placement
    return artifact


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


def _parse_heatr_progress(line: str, job_id: str) -> None:
    """Parse a HEATR_PROGRESS line from rfam_eqs_coupled.py and update job progress.

    Expected format (space-separated key=value pairs after the prefix):
        HEATR_PROGRESS pct=58.33 step=420 total=720 T=165.2 Tmax=196.1 phi=0.0210 rho=0.5510 err=0.0001 eta=11.3
    """
    parts: dict[str, str] = {}
    for tok in line.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            parts[k] = v
    try:
        pct = float(parts["pct"])
        step = int(parts["step"])
        total = int(parts["total"])
        T_val     = float(parts.get("T",    0.0))
        T_max_val = float(parts.get("Tmax", T_val))
        dT_val    = float(parts.get("dT",   max(0.0, T_max_val - T_val)))
        phi_val   = float(parts.get("phi",  0.0))
        rho_val   = float(parts.get("rho",  0.0))
        err_val   = float(parts.get("err",  0.0))
    except (KeyError, ValueError):
        return
    label = (
        f"Step {step}/{total}"
        f" \u2022 T\u0305={T_val:.1f}\u00b0C"
        f" \u2022 T\u2191={T_max_val:.1f}\u00b0C"
        f" \u2022 \u0394T={dT_val:.1f}\u00b0C"
        f" \u2022 \u03c6\u0305={phi_val:.3f}"
        f" \u2022 \u03c1\u0305={rho_val:.3f}"
        f" \u2022 err={err_val:.3f}%"
    )
    _set_job_progress(job_id, progress_pct=pct, progress_label=label)
    # Persist physics snapshot — survives job completion so the final state
    # remains visible in the webui after the run finishes.
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            job["physics_snapshot"] = {
                "step":     step,
                "total":    total,
                "T_mean_c": T_val,
                "T_max_c":  T_max_val,
                "dT_c":     dT_val,
                "phi_mean": phi_val,
                "rho_mean": rho_val,
                "err_pct":  err_val,
            }


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
        import matplotlib
        matplotlib.use("Agg")
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
    out_dir.mkdir(parents=True, exist_ok=True)
    _register_job_output(job, out_dir)
    # Persist antenna placement so the results tab can offer to reload the workshop
    explicit_instances = payload.get("antennae_explicit_instances")
    if explicit_instances and isinstance(explicit_instances, list):
        placement_rec = {
            "antennae_enabled": True,
            "instances": explicit_instances,
        }
        (out_dir / "antennae_placement.json").write_text(
            json.dumps(placement_rec, indent=2), encoding="utf-8"
        )
    cmd = [sys.executable, "rfam_eqs_coupled.py", "--config", str(cfg_path), "--output-dir", str(out_dir)]
    rc = _run_command(cmd, Path(job["log_path"]), job_id=job_id)
    if rc != 0:
        _set_job_progress(job_id, progress_label="Run failed")
        raise RuntimeError(f"simulation exited with code {rc}")
    _set_job_progress(job_id, completed_runs=1, progress_pct=100.0, progress_label="Run complete")


def _launch_fgm_resimulate_mode(payload: dict[str, Any], job: dict[str, Any]) -> None:
    """Re-run a HEATR simulation with an FGM saturation map injected as spatially-varying conductivity.

    Expected payload keys:
        source_run_dir   — relative path (under OUTPUTS_DIR) of the original run
        fgm_npz_path     — relative path (under ROOT) to the FGM .npz file
        output_name      — name for the new run output directory
        magnitude        — float, FGM magnitude override (optional; default keeps value from NPZ metadata)
        baseline_saturation — float (optional, default 0.5)
        iterate          — bool, whether to use iterative feedback (default false)
    """
    job_id = str(job["id"])
    source_rel = str(payload.get("source_run_dir", "")).strip()
    fgm_rel    = str(payload.get("fgm_npz_path",   "")).strip()
    output_name = str(payload.get("output_name",   "")).strip()

    if not source_rel:
        raise ValueError("source_run_dir is required")
    if not fgm_rel:
        raise ValueError("fgm_npz_path is required")
    if not _is_valid_output_name(output_name):
        raise ValueError("output_name must use only letters, numbers, '_' or '-'")

    source_dir = (OUTPUTS_DIR / source_rel).resolve()
    try:
        source_dir.relative_to(OUTPUTS_DIR)
    except ValueError:
        raise ValueError("source_run_dir must be inside outputs_eqs/")

    fgm_path = (ROOT / fgm_rel).resolve()
    try:
        fgm_path.relative_to(ROOT)
    except ValueError:
        raise ValueError("fgm_npz_path must be inside workspace")

    used_cfg_path = source_dir / "used_config.yaml"
    if not used_cfg_path.exists():
        # The source dir may be a fgm_iterate parent directory (no used_config.yaml there).
        # Strategy 1: look for used_config.yaml in the iter subdir that owns the FGM NPZ.
        #   The NPZ path is like: outputs_eqs/.../parent_iter2/fgm_...npz
        #   → try the NPZ's parent dir.
        # Strategy 2: walk iter subdirs (parent_iter0, parent_iter1, …) and use the last one
        #   that has a used_config.yaml.
        _found_cfg = None
        if fgm_path.exists():
            _npz_parent_cfg = fgm_path.parent / "used_config.yaml"
            if _npz_parent_cfg.exists():
                _found_cfg = _npz_parent_cfg
        if _found_cfg is None:
            # Walk all iter subdirs in descending order and grab the first found
            _iter_dirs = sorted(
                source_dir.glob("*_iter[0-9]*"),
                key=lambda p: p.name,
                reverse=True,
            )
            for _d in _iter_dirs:
                _c = _d / "used_config.yaml"
                if _c.exists():
                    _found_cfg = _c
                    break
        if _found_cfg is None:
            raise FileNotFoundError(
                f"used_config.yaml not found in {source_rel} or any iter subdirectory — "
                f"re-simulation requires a previous run"
            )
        used_cfg_path = _found_cfg
    if not fgm_path.exists():
        raise FileNotFoundError(f"FGM file not found: {fgm_rel}")

    # Load and patch the original run's config
    with open(used_cfg_path, "r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh) or {}

    # Inject the fgm_feedback block
    magnitude          = float(payload.get("magnitude",           1.0))
    baseline_saturation = float(payload.get("baseline_saturation", 0.5))
    iterate            = bool(payload.get("iterate",              False))
    base_cfg["fgm_feedback"] = {
        "enabled":              True,
        "saturation_map_npz":   str(fgm_path),
        "magnitude":            magnitude,
        "baseline_saturation":  baseline_saturation,
        "iterate":              iterate,
        "iterate_interval_steps": int(payload.get("iterate_interval_steps", 50)),
    }

    # Write patched config into GUI_CONFIG_DIR
    GUI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_filename = f"fgm_resim_{job_id}.yaml"
    cfg_path = GUI_CONFIG_DIR / cfg_filename
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.dump(base_cfg, fh, default_flow_style=False, allow_unicode=True)

    # Output dir: sibling of original run, named by output_name
    # Infer the parent dir pattern from the source
    parent_dir = source_dir.parent
    out_dir = parent_dir / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _register_job_output(job, out_dir)

    _set_job_progress(job_id, total_runs=1, completed_runs=0, progress_pct=5.0,
                      progress_label="FGM re-simulation running")

    cmd = [sys.executable, "rfam_eqs_coupled.py", "--config", str(cfg_path), "--output-dir", str(out_dir)]
    rc = _run_command(cmd, Path(job["log_path"]), job_id=job_id)
    if rc != 0:
        _set_job_progress(job_id, progress_label="Re-simulation failed")
        raise RuntimeError(f"FGM re-simulation exited with code {rc}")
    _set_job_progress(job_id, completed_runs=1, progress_pct=100.0, progress_label="Re-simulation complete")


def _fgm_iter_extract_optimal_time(log_path: Path, fallback_s: float, target_phi: float = 0.90) -> tuple[float, str, str]:
    """Parse a completed optimizer-mode sim log for the recommended exposure time.

    The optimizer's generate_optimizer_report() already implements the correct
    recommendation logic (melt criterion vs. max-density criterion, checking T_max safety).
    We parse its output from stdout/log rather than reimplementing it here.

    Returns:
        (time_s, criterion, reason_str)
        - time_s: chosen time in seconds
        - criterion: "melt" | "max_density" | "phi_crossing" | "fallback"
        - reason_str: human-readable explanation
    """
    import re
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return fallback_s, "fallback", "log not readable"

    # 1. Parse the recommendation line (highest priority — already incorporates T_max safety)
    # The solver prints: "[optimizer] ★ Recommendation: melt criterion — 317.1 s ..."
    # or:                "[optimizer] ★ Recommendation: max density criterion — 350.2 s ..."
    # Note: recommendation.replace('_',' ') is applied in the solver, so "max_density" → "max density".
    # We capture everything before " criterion" as the name, then normalise spaces→underscores.
    rec_pat = re.compile(
        r"\[optimizer\]\s+★\s+Recommendation:\s+([\w][\w ]*?)\s+criterion\s*[—\-]+\s*([\d.]+)\s+s"
    )
    m_rec = rec_pat.search(text)
    if m_rec:
        criterion = m_rec.group(1).strip().replace(" ", "_")  # "melt" or "max_density"
        t_s = float(m_rec.group(2))
        # Try to also grab the reason line (the line immediately following the recommendation)
        rec_pos = m_rec.end()
        rest = text[rec_pos:]
        reason_line = rest.strip().splitlines()[0].strip() if rest.strip() else ""
        # Clean up the reason line (may start with whitespace or "  ")
        reason = reason_line.lstrip() or f"{criterion} criterion selected by optimizer"
        return t_s, criterion, reason

    # 2. Fall back to φ crossing scan (no recommendation found — older log format)
    phi_pat = re.compile(
        r"\[optimizer\]\s+Snapshot\s+@\s+φ_mean≥"
        + re.escape(f"{target_phi:.2f}")
        + r":\s+t=([\d.]+)s"
    )
    m_phi = phi_pat.search(text)
    if m_phi:
        t_s = float(m_phi.group(1))
        return t_s, "phi_crossing", f"φ={target_phi:.2f} crossing at t={t_s:.1f}s (T_max not checked)"

    # 3. Ultimate fallback
    return fallback_s, "fallback", f"no optimizer signal found — using input time {fallback_s/60:.1f} min"


def _select_best_iter(
    convergence_log: list[dict],
    use_optimizer: bool,
    damage_t_c: float,
) -> dict:
    """Select the best iteration from a convergence log.

    Rules (applied in order):

    1. **Exclude iter-0 when use_optimizer=True.**
       When the optimizer probe IS iter-0, that run is intentionally over-exposed
       and is never a valid production recommendation — it exists only to find the
       optimal exposure time for subsequent iterations.

    2. **Hard-disqualify thermally damaged or high-bleed iterations.**
       - T_max > damage_t_c  → material degradation; cannot recommend this.
       - frac_bleed > 0.02   → surrounding powder is sintering; geometric fidelity
         lost. An iteration that "looks uniform" because EVERYTHING heated up is
         not a good result.
       If all remaining candidates fail both filters, we fall back to all eligible
       entries and pick the least-bad one (lowest bleed first, then lowest score).

    3. **Prefer near-final over active regime** within the qualified pool.
       near-final (ρ̄ ≥ 0.82) and active (ρ̄ < 0.82) scores use different formulas
       and are numerically incomparable. A genuinely near-final iteration represents
       more physical progress than an active one. But we ONLY apply this preference
       among disqualification-passing candidates, so an over-sintered/bleedy
       near-final entry does not override a clean active-regime one.

    4. Within each regime, lowest score wins.

    5. If the log is empty, return an empty dict (caller should guard).
    """
    if not convergence_log:
        return {}

    BLEED_DISQUALIFY = 0.02   # 2% of outside-part pixels above melt point

    # Step 1: exclude optimizer probe
    eligible = [
        e for e in convergence_log
        if not (use_optimizer and e.get("iter") == 0)
    ]
    if not eligible:
        eligible = list(convergence_log)   # nothing left → use everything

    # Step 2: hard disqualifiers
    safe = [
        e for e in eligible
        if e.get("T_max_c", 0.0) <= damage_t_c
        and e.get("frac_bleed", 1.0) <= BLEED_DISQUALIFY
    ]

    if safe:
        pool = safe
    else:
        # All candidates failed — pick least bad: lowest bleed first, then score
        pool = sorted(
            eligible,
            key=lambda e: (e.get("frac_bleed", 1.0), e.get("score", 999.0)),
        )
        # Return the top of that sorted list directly (no regime preference when
        # everything is disqualified — just minimise damage)
        return pool[0]

    # Step 3 & 4: regime preference among safe candidates
    # Near-final is PREFERRED over active because the score formulas are incomparable
    # across regimes AND a near-final run (ρ̄ ≥ 0.82) has done more physical work.
    # EXCEPTION: if the best-active σ_T is substantially better than the best
    # near-final σ_T, the active candidate wins — this guards against the case where
    # the regime crossed into near-final due to a hot over-exposure but uniformity
    # actually *regressed* (e.g. circle: iter-1 active σ_T=8.76°C is better than
    # iter-2 near-final σ_T=14.78°C).
    nf = [e for e in pool if e.get("score_regime") == "near-final"]
    ac = [e for e in pool if e.get("score_regime") == "active"]

    if nf and ac:
        best_nf = min(nf, key=lambda e: e.get("score", 999.0))
        best_ac = min(ac, key=lambda e: e.get("score", 999.0))
        nf_sigma = best_nf.get("sigma_T", 999.0)
        ac_sigma = best_ac.get("sigma_T", 999.0)
        # Prefer active if its σ_T is meaningfully better (>20% lower than near-final best).
        # The 20% threshold prevents flipping on noise; a 5°C vs 4.8°C difference is not
        # significant, but 14.78°C vs 8.76°C is clearly a regression in uniformity.
        if ac_sigma < nf_sigma * 0.80:
            return best_ac
        return best_nf

    if nf:
        return min(nf, key=lambda e: e.get("score", 999.0))
    if ac:
        return min(ac, key=lambda e: e.get("score", 999.0))
    return min(pool, key=lambda e: e.get("score", 999.0))


def _launch_fgm_iterate_mode(payload: dict[str, Any], job: dict[str, Any]) -> None:
    """Outer-loop iterative FGM optimization starting from a fresh geometry.

    Flow:
      iter-0  — vanilla simulation (no FGM); optionally runs optimizer mode first
                to auto-select exposure time.
      iter-1+ — each run seeded with the FGM derived from the previous iteration's
                temperature field:  high-T → low saturation, low-T → high saturation.

    Convergence metric: ΔT = T_max − T_mean (inside the part).
    The "ideal" FGM is the one where ΔT is minimised, i.e. the saturation map
    that makes RF power absorption spatially uniform regardless of E-field variation.

    Payload keys:
        shape                 — geometry shape (required, same as single mode)
        output_name           — base name; iter-0 dir = <name>_iter0, etc.
        exposure_minutes      — float, used directly when use_optimizer=false
        use_optimizer         — bool (default false); if true, runs optimizer mode
                                for iter-0 and extracts the time when phi≥0.90
        n_iterations          — total iterations INCLUDING iter-0 (default 4)
        magnitude             — float (default 1.0)
        baseline_saturation   — float (default 0.5)
        bpp                   — 2 or 4 (default 2)
        proxy_field           — "T"|"Qrf"|"rho_rel" (default "T")
        invert                — bool (default true)
        convergence_delta_T   — early-stop when ΔT < this °C (default 5.0)
        iterate_inner         — bool, intra-run inner-loop (default false)
        iterate_interval_steps— int (default 50)
        iterate_damping       — float 0-1 (default 0.5)
    """
    job_id = str(job["id"])
    output_name     = str(payload.get("output_name", "")).strip()
    n_iterations    = max(2, int(payload.get("n_iterations", 12)))  # 12 covers iter-9 optimum + 2-iter early-stop buffer
    use_optimizer   = bool(payload.get("use_optimizer", False))
    exposure_min    = float(payload.get("exposure_minutes", 8.0))

    if not _is_valid_output_name(output_name):
        raise ValueError("output_name must use only letters, numbers, '_' or '-'")

    shape = str(payload.get("shape", "")).strip()
    if not shape or shape not in SUPPORTED_SHAPES:
        raise ValueError(f"invalid shape: {shape!r}")

    # ── FGM parameters ────────────────────────────────────────────────────────
    # Defaults reflect the empirically-optimal INTEGRAL mode (circle 4bpp I-series):
    #   magnitude=0.7, decay=1.0, bpp=4, dead_band=0.05, n_iter=12 (catches iter-9
    #   optimum for circle and allows 2-iter buffer for early-stop detection).
    # Users should increase magnitude toward 0.8–0.9 for geometries with very large
    # σ_0 (>30°C), or reduce toward 0.5 for geometries with small σ_0 (10–15°C).
    magnitude           = float(payload.get("magnitude",           0.7))
    magnitude_decay     = float(payload.get("magnitude_decay",     1.0))   # 1.0 = no decay (integral mode)
    baseline_saturation = float(payload.get("baseline_saturation", 0.5))
    bpp                 = int(payload.get("bpp",                   4))      # 4bpp for maximum resolution
    proxy_field         = str(payload.get("proxy_field",           "T_phi90")).strip()
    invert              = bool(payload.get("invert",               True))
    conv_threshold      = float(payload.get("convergence_sigma_T", 3.0))   # σ_T primary
    dead_band           = float(payload.get("dead_band",           0.05))
    fgm_momentum        = float(payload.get("fgm_momentum",        0.0))    # 0.0 = off (not used in integral mode)
    iterate_inner       = bool(payload.get("iterate_inner",        False))
    iterate_interval    = int(payload.get("iterate_interval_steps", 50))
    iterate_damping     = float(payload.get("iterate_damping",     0.5))
    melt_abort_frac     = float(payload.get("melt_abort_frac",     0.15))  # pre-check threshold

    # ── Stagnation / anti-rut parameters ─────────────────────────────────────
    # "thorough" mode: run a SEQUENCE of proxy fields in order.  When one proxy
    # stagnates (σ_T improvement < stagnation_eps for N consecutive iters), the
    # algorithm switches to the next proxy in the schedule and resets magnitude.
    # Default schedule: ["T_phi90", "Qrf"] — start with thermal field, switch to
    # direct RF coupling when thermal-based corrections stop helping.
    # With thorough=False, only the single proxy_field is used (legacy behaviour).
    thorough            = bool(payload.get("thorough",             False))
    _default_schedule   = (["T_phi90", "Qrf", "rho_rel"]
                            if thorough else [proxy_field])
    proxy_schedule      = list(payload.get("proxy_schedule",  _default_schedule))
    # How many consecutive iters without σ_T improving by at least stagnation_eps
    # before switching proxy (ignored when thorough=False).
    stagnation_window   = int(payload.get("stagnation_window",    5))
    stagnation_eps      = float(payload.get("stagnation_eps",     0.5))   # °C
    # regime-adaptive proxy: in active sintering use Qrf; in near-final use T_phi90
    # overrides proxy_schedule selection when True
    regime_adaptive     = bool(payload.get("regime_adaptive",     False))

    # ── Integral control mode ─────────────────────────────────────────────────
    # When True, each iteration accumulates a delta correction onto the prior
    # saturation map rather than recomputing from scratch (proportional mode).
    # Also uses fixed iter-0 normalization bounds to prevent residual re-amplification.
    use_delta_correction = bool(payload.get("use_delta_correction", True))   # integral mode is now default
    # TO parameters — only used when use_delta_correction=True in OC-TO mode (move_limit>0)
    # In pure integral mode (move_limit=0, sensitivity_filter_sigma=0) these are unused.
    move_limit               = float(payload.get("move_limit",               0.0))   # 0 = pure delta (integral mode)
    sensitivity_filter_sigma = float(payload.get("sensitivity_filter_sigma", 0.0))   # 0 = off (integral mode)
    # Stochastic perturbation for local-minimum escape:
    # When > 0, applied on stagnation events to kick the design out of local minima.
    # perturbation_amplitude = magnitude of the perturbation (0.05–0.10 recommended).
    # perturbation_on_stagnation: only perturb when stagnation is detected (default True).
    perturbation_amplitude        = float(payload.get("perturbation_amplitude",        0.0))
    perturbation_on_stagnation    = bool( payload.get("perturbation_on_stagnation",    True))
    # Regression abort sensitivity: score must exceed best-ever × this multiplier
    # for consecutive_bad to increment.  1.30 = abort if 30% above best.
    # Increase to 2.0+ for exhaustive / academic runs where we want all iterations.
    regression_threshold_multiplier = float(payload.get("regression_threshold_multiplier", 10.0))
    # After a stochastic perturbation, suppress regression-abort for this many iters
    # so the new seed has time to show whether it found a better basin.
    post_perturbation_grace         = int(  payload.get("post_perturbation_grace",         4))
    # Cold-zone overprinting: extend the FGM saturation map outward past the part
    # boundary in underheated regions only (sat >= overprint_cold_thresh).
    # 0 = disabled; typical value 2–4 mm for diamond corners.
    overprint_cold_mm     = float(payload.get("overprint_cold_mm",     0.0))
    overprint_cold_thresh = float(payload.get("overprint_cold_thresh", 0.6))

    # ── Hybrid mode: proportional iter-1 + OC-TO refinement ──────────────────
    # When use_hybrid=True:
    #   Phase 1 (iter-1): proportional correction — full magnitude, no accumulation.
    #     Provides the large single-step improvement that proportional mode excels at.
    #   Phase 2 (iter-2+): OC-TO mode with bounded gradient projection.
    #     Starts from the good iter-1 FGM and refines with small, stable corrections.
    # Combines proportional's 54-65% single-step reduction with OC-TO's stability.
    # Note: use_delta_correction is forced True for phase 2; any explicit
    #       use_delta_correction=True setting simply means the run starts in OC-TO
    #       from iter-1 (i.e., no hybrid / standard OC-TO mode).
    use_hybrid                      = bool( payload.get("use_hybrid",                      False))
    hybrid_phase2_magnitude         = float(payload.get("hybrid_phase2_magnitude",         0.4))
    hybrid_phase2_move_limit        = float(payload.get("hybrid_phase2_move_limit",        0.15))
    hybrid_phase2_sensitivity_sigma = float(payload.get("hybrid_phase2_sensitivity_sigma", 0.5))
    # Minimum magnitude floor — 0.0 for integral mode (decay=1.0 means this is never hit).
    # For proportional/hybrid runs with decay<1.0, set to 0.001–0.005 to prevent full decay.
    min_magnitude                   = float(payload.get("min_magnitude",                   0.0))

    # ── EBC — Energy-Budget-Conserving exposure adjustment (M3) ──────────────
    # When use_ebc=True, each FGM iteration computes the mean saturation inside
    # the part mask and adjusts t★ proportionally to maintain constant total RF
    # energy absorbed:
    #   σ(x,y) = σ₀ × sat_map(x,y)
    #   P_total ≈ σ₀ × mean(sat_map) × ∫|∇φ|² dA
    #   → t★ᵢ = t★_base × (mean_sat_iter0 / mean_sat_i)
    #
    # This first-order approximation is exact when the FGM doesn't change the
    # EQS field topology (small perturbations) and approximately correct otherwise.
    # Solves the hexagon t★ confound: FGM reduces hot-spot saturation (κ=32.7×),
    # removing disproportionate total power, causing progressive mean_rho drift.
    # EBC recalibrates t★ so total energy matches the iter-0 baseline every time.
    #
    # Parameters:
    #   ebc_clamp_lo (float, default 0.5): minimum EBC factor (t★ never drops below
    #       50% of base — prevents runaway under-exposure if FGM over-corrects)
    #   ebc_clamp_hi (float, default 2.5): maximum EBC factor (t★ never exceeds
    #       2.5× base — prevents runaway over-exposure if FGM under-corrects)
    use_ebc       = bool( payload.get("use_ebc",       False))
    ebc_clamp_lo  = float(payload.get("ebc_clamp_lo",  0.50))
    ebc_clamp_hi  = float(payload.get("ebc_clamp_hi",  2.50))

    # ── Output layout ─────────────────────────────────────────────────────────
    # All iteration dirs live under outputs_eqs/runs/<shape>/fgm_iterate/<output_name>/
    safe_shape   = shape.replace(" ", "_").lower()
    parent_dir   = OUTPUTS_DIR / "runs" / safe_shape / "fgm_iterate" / output_name
    parent_dir.mkdir(parents=True, exist_ok=True)
    _register_job_output(job, parent_dir)

    total_runs = n_iterations  # iter-0 through iter-(n_iterations-1)

    # ── Phase 0a (optional): run time optimizer to find ideal exposure ────────
    # When use_optimizer=True we first run a dedicated optimizer-mode sim in a
    # temporary "opt_probe" subdir, extract the optimal time from its log, then
    # throw away that dir and run a clean single-mode iter-0 at the optimal time.
    # iter-1+ must run as plain single-mode sims; optimizer key is stripped from base_cfg below.
    optimizer_criterion: str | None = None   # populated when use_optimizer=True
    optimizer_reason:    str | None = None
    optimizer_max_min:   float      = exposure_min   # user's stated upper bound (for reporting)

    # ── iter-0 ───────────────────────────────────────────────────────────────
    # Two paths:
    #   a) Resume: a source_run_dir was supplied — skip the simulation, point iter0_dir
    #      at the existing run so FGM generation proceeds from it.
    #   b) Fresh: run iter-0 now.
    #        • use_optimizer=False → single-mode at exposure_min (user's chosen time).
    #        • use_optimizer=True  → optimizer-mode at exposure_min (user's MAXIMUM time).
    #          iter-0 IS the diagnostic probe: it runs the full over-exposed duration,
    #          collects snapshots at each φ crossing and T_max ceiling, then the
    #          optimizer's recommendation logic picks the ideal exposure time.
    #          T_max > DAMAGE_T_C in iter-0 is EXPECTED when the user intentionally
    #          ran long — it is NOT an abort condition in this path.
    #          All subsequent iterations (1…N) run at the optimizer-chosen time.
    source_rel = str(payload.get("source_run_dir", "")).strip()
    _resuming  = bool(source_rel)

    if _resuming:
        # Validate and resolve the supplied source directory
        source_iter_dir = (OUTPUTS_DIR / source_rel).resolve()
        try:
            source_iter_dir.relative_to(OUTPUTS_DIR)
        except ValueError:
            raise ValueError("source_run_dir must be inside outputs_eqs/")
        if not (source_iter_dir / "used_config.yaml").exists():
            raise FileNotFoundError(
                f"used_config.yaml not found in {source_rel} — "
                f"source_run_dir must be a completed single-mode or fgm_iterate iter dir"
            )
        if not (source_iter_dir / "fields.npz").exists():
            raise FileNotFoundError(f"fields.npz not found in {source_rel}")
        iter0_dir = source_iter_dir
        _set_job_progress(
            job_id, total_runs=total_runs, completed_runs=1, progress_pct=5.0,
            progress_label=f"FGM continue: using {Path(source_rel).name} as baseline — generating FGM…",
        )
    else:
        # ── Two-phase iter-0 when use_optimizer=True ─────────────────────────
        # Phase A: optimizer PROBE — runs at the ceiling time to find t★.
        #          Stored in {output_name}_opt_probe (not counted as iter-0).
        # Phase B: clean single-mode BASELINE at t★ — stored in {output_name}_iter0.
        #          This is the true iter-0: correct sigma_T, T_max, rho at t★.
        #          FGM and all metrics are computed from this clean baseline.
        #
        # When use_optimizer=False, only Phase B runs (at the user's fixed time).
        # ─────────────────────────────────────────────────────────────────────

        if use_optimizer:
            # ── Phase A: optimizer probe ──────────────────────────────────────
            _set_job_progress(
                job_id, total_runs=total_runs, completed_runs=0, progress_pct=3.0,
                progress_label=f"FGM opt probe: running to {exposure_min:.2f} min ceiling to find t★…",
            )
            probe_dir = parent_dir / f"{output_name}_opt_probe"
            probe_dir.mkdir(parents=True, exist_ok=True)
            _register_job_output(job, probe_dir)

            probe_payload = dict(payload)
            probe_payload["mode"] = "optimizer"
            probe_cfg = _prepare_config(
                mode="optimizer", payload=probe_payload,
                exposure_minutes=exposure_min,
                output_tag=f"{output_name}_opt_probe",
                job=job,
            )
            cmd_probe = [sys.executable, "rfam_eqs_coupled.py",
                         "--config", str(probe_cfg), "--output-dir", str(probe_dir)]
            rc_probe = _run_command(cmd_probe, Path(job["log_path"]), job_id=job_id)
            if rc_probe != 0:
                _set_job_progress(job_id, progress_label="optimizer probe failed")
                raise RuntimeError(f"FGM optimizer probe exited with code {rc_probe}")

            # Extract t★ from probe log
            fallback_s = exposure_min * 60.0
            opt_t_s, optimizer_criterion, optimizer_reason = _fgm_iter_extract_optimal_time(
                Path(job["log_path"]), fallback_s=fallback_s, target_phi=0.90
            )
            _CRIT_LABELS = {
                "melt":         "φ=0.90 criterion",
                "max_density":  "max-density criterion (extended past φ=0.90)",
                "phi_crossing": "φ=0.90 crossing",
                "fallback":     "fallback — no snapshot found",
            }
            crit_label = _CRIT_LABELS.get(optimizer_criterion, optimizer_criterion)
            if optimizer_criterion != "fallback":
                exposure_min = opt_t_s / 60.0
                _write_job_log(
                    Path(job["log_path"]),
                    f"[fgm_iterate] probe → ideal time = {exposure_min:.2f} min "
                    f"({optimizer_criterion}): {optimizer_reason}",
                )
            else:
                _write_job_log(
                    Path(job["log_path"]),
                    f"[fgm_iterate] WARNING: optimizer probe found no usable snapshot in "
                    f"{optimizer_max_min:.1f} min. Falling back to max time. "
                    f"If ρ̄ is also low the part is not sintering — check power/material.",
                )

            # ── Phase B: clean single-mode baseline at t★ ────────────────────
            _set_job_progress(
                job_id, total_runs=total_runs, completed_runs=0, progress_pct=5.0,
                progress_label=(
                    f"FGM iter-0: clean baseline at t★={exposure_min:.2f} min "
                    f"({crit_label})…"
                ),
            )
            iter0_dir = parent_dir / f"{output_name}_iter0"
            iter0_dir.mkdir(parents=True, exist_ok=True)
            _register_job_output(job, iter0_dir)

            iter0_payload = dict(payload)
            iter0_payload["mode"] = "single"
            cfg0_path = _prepare_config(
                mode="single", payload=iter0_payload,
                exposure_minutes=exposure_min,
                output_tag=f"{output_name}_iter0",
                job=job,
            )
            cmd0 = [sys.executable, "rfam_eqs_coupled.py",
                    "--config", str(cfg0_path), "--output-dir", str(iter0_dir)]
            rc0 = _run_command(cmd0, Path(job["log_path"]), job_id=job_id)
            if rc0 != 0:
                _set_job_progress(job_id, progress_label="iter-0 baseline failed")
                raise RuntimeError(f"FGM iter-0 baseline exited with code {rc0}")

            _set_job_progress(
                job_id, completed_runs=1, progress_pct=100.0 / total_runs,
                progress_label=(
                    f"iter-0 done at t★={exposure_min:.2f} min — generating FGM…"
                ),
            )

        else:
            # ── No optimizer: single-phase iter-0 at user's fixed time ───────
            _set_job_progress(
                job_id, total_runs=total_runs, completed_runs=0, progress_pct=5.0,
                progress_label=f"FGM iter-0: baseline single run at {exposure_min:.2f} min…",
            )
            iter0_dir = parent_dir / f"{output_name}_iter0"
            iter0_dir.mkdir(parents=True, exist_ok=True)
            _register_job_output(job, iter0_dir)

            iter0_payload = dict(payload)
            iter0_payload["mode"] = "single"
            cfg0_path = _prepare_config(
                mode="single", payload=iter0_payload,
                exposure_minutes=exposure_min,
                output_tag=f"{output_name}_iter0",
                job=job,
            )
            cmd0 = [sys.executable, "rfam_eqs_coupled.py",
                    "--config", str(cfg0_path), "--output-dir", str(iter0_dir)]
            rc0 = _run_command(cmd0, Path(job["log_path"]), job_id=job_id)
            if rc0 != 0:
                _set_job_progress(job_id, progress_label="iter-0 simulation failed")
                raise RuntimeError(f"FGM iter-0 exited with code {rc0}")
            _set_job_progress(
                job_id, completed_runs=1, progress_pct=100.0 / total_runs,
                progress_label=f"iter-0 complete ({exposure_min:.2f} min) — generating FGM…",
            )

    # Load base config from iter-0 for patching in subsequent iterations.
    # IMPORTANT: strip all optimizer/turntable/orientation modes — iter-1+ must run as
    # plain single-mode sims at the extracted optimal exposure time.
    iter0_cfg_file = iter0_dir / "used_config.yaml"
    if not iter0_cfg_file.exists():
        raise FileNotFoundError("iter-0 did not write used_config.yaml — cannot continue")
    with open(iter0_cfg_file, "r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh) or {}
    # Disable any optimizer/sweep modes that were active in iter-0 so subsequent iters
    # run as straightforward single-mode sims at the (possibly optimizer-extracted) time.
    for _opt_key in ("optimizer", "orientation_optimizer", "placement_optimizer", "turntable"):
        if _opt_key in base_cfg:
            if isinstance(base_cfg[_opt_key], dict):
                base_cfg[_opt_key]["enabled"] = False
            else:
                base_cfg.pop(_opt_key, None)

    # ── Generate FGM from iter-0 fields ──────────────────────────────────────
    from fgm_generator import generate_fgm  # local import
    if not (iter0_dir / "fields.npz").exists():
        raise FileNotFoundError("iter-0 did not write fields.npz — cannot generate FGM")

    # ── Helper: extract full metrics from a completed iteration dir ───────────
    # Thermal damage threshold: PA12 begins to decompose above ~245°C.
    # Under-sintering threshold: ρ̄ < 0.60 means the part barely melted.
    DAMAGE_T_C      = 245.0   # °C — thermal damage onset
    RHO_TARGET      = 0.92    # target relative density for score computation
    MIN_RHO_VALID   = 0.60    # below this → under-sintered, abort
    T_AMBIENT_C     = 25.0    # °C — ambient/initial powder temperature
    SEL_TARGET      = 2.5     # minimum acceptable thermal selectivity

    def _iter_metrics(it_dir: Path) -> dict[str, Any]:
        """Read fields.npz + summary.json from an iteration dir and return all metrics.

        Score formula (lower = better):
          score = 1.0×σ_T  +  20×σ_ρ  +  0.5×|ρ̄ − rho_target|
                           +  0.5×max(0, T_max − DAMAGE_T_C)    [damage]
                           +  3.0×max(0, SEL_TARGET − selectivity)  [bleed risk]
                           +  15×frac_bleed                     [actual bleed]

        Selectivity = (T_mean_inside − T_ambient) / (T_mean_outside − T_ambient)
          A value >> 1.0 means the part heats selectively vs. surrounding powder.
          Values near 1.0 indicate bulk heating — the powder bed is also sintering,
          which destroys geometric fidelity (the part blurs into the surroundings).

        frac_bleed = fraction of outside-mask powder cells above melt_reference_c.
          Any non-zero value means surrounding loose powder has reached sintering
          temperature and will fuse to the part, degrading dimensional accuracy.
        """
        metrics: dict[str, Any] = {}
        melt_ref_c = 186.0   # PA12 default; overridden from summary if available
        sfile = it_dir / "summary.json"
        if sfile.exists():
            try:
                s = json.loads(sfile.read_text(encoding="utf-8"))
                metrics["frac_melt"]    = float(s.get("frac_part_ge_melt_ref", 0.0))
                metrics["mean_rho"]     = float(s.get("mean_rho_rel_part_final", 0.0))
                metrics["T_mean_c"]     = float(s.get("mean_T_part_final_c", 0.0))
                metrics["T_max_c"]      = float(s.get("max_T_part_final_c", 0.0))
                metrics["T_p95_c"]      = float(s.get("p95_T_part_final_c", 0.0))
                metrics["mean_phi"]     = float(s.get("mean_phi_part_final", 0.0))
                melt_ref_c              = float(s.get("melt_reference_c", melt_ref_c))
            except Exception:
                pass
        ffile = it_dir / "fields.npz"
        if ffile.exists():
            try:
                f     = np.load(ffile)
                mask  = f["part_mask"].astype(bool)
                T     = f["T"]
                T_in  = T[mask]
                T_out = T[~mask]
                rho_in= f["rho_rel"][mask]

                metrics["sigma_T"]   = round(float(np.std(T_in)),  3)
                metrics["sigma_rho"] = round(float(np.std(rho_in)), 4)
                metrics["dT_c"]      = round(float(T_in.max() - T_in.mean()), 3)

                # Selectivity: how much hotter is inside vs. outside (relative to ambient)
                dT_in  = float(T_in.mean())  - T_AMBIENT_C
                dT_out = float(T_out.mean()) - T_AMBIENT_C
                thermal_sel = dT_in / max(dT_out, 0.1)
                metrics["T_out_mean_c"]      = round(float(T_out.mean()), 2)
                metrics["thermal_selectivity"] = round(thermal_sel, 3)

                # Bleed: fraction of outside-mask pixels at or above melt reference
                frac_bleed = float((T_out >= melt_ref_c).sum()) / max(len(T_out), 1)
                metrics["frac_bleed"]        = round(frac_bleed, 4)

                # ── Composite score (lower = better) ─────────────────────────
                #
                # Two-regime design based on mean density:
                #
                # ACTIVE-SINTERING regime (ρ̄ < 0.82):
                #   The part is still densifying. σ_T fluctuations are expected and
                #   do NOT indicate failure — they reflect active sintering gradients.
                #   Primary objectives: (a) keep heat inside the part (zero bleed),
                #   (b) move ρ̄ toward target, (c) avoid thermal damage.
                #   σ_rho counts but at reduced weight; σ_T is mostly ignored.
                #
                # NEAR-FINAL regime (ρ̄ ≥ 0.82):
                #   Most sintering is complete. Now uniformity matters: σ_T and σ_rho
                #   determine whether the final print will have spatial density gradients.
                #   σ_T is the primary uniformity signal here.
                #
                # This prevents the algorithm from rewarding an under-sintered, low-σ_T
                # state (which looks "uniform" only because nothing has happened yet)
                # over a well-sintered, slightly higher-σ_T state.
                #
                sm   = metrics.get("sigma_T",  0.0)
                srho = metrics.get("sigma_rho", 0.0)
                rm   = metrics.get("mean_rho",  0.0)
                tmax = metrics.get("T_max_c",   0.0)
                dmg      = max(0.0, tmax - DAMAGE_T_C) * 0.5
                bleed_pen = frac_bleed * 15.0   # zero bleed is critical
                sel_pen   = max(0.0, SEL_TARGET - thermal_sel) * 2.0

                RHO_ACTIVE_THRESH = 0.82   # boundary between regimes
                if rm < RHO_ACTIVE_THRESH:
                    # Active-sintering regime: drive density up, keep bleed zero.
                    # σ_T not penalised (it will naturally be non-zero during sintering).
                    # σ_rho weight kept LOW (2.0) so that a density gain (rho_pen decrease of
                    #   15×Δρ̄) beats a σ_rho increase whenever Δρ̄ > (2/15)×Δσ_rho — i.e.
                    #   small density improvements win even when variance ticks up slightly,
                    #   which is the physically correct ordering for active sintering.
                    #   (With 5.0× weight the cross-over was too easy to hit, causing iter-1 to
                    #   beat iter-2/3 even when iter-2/3 had higher density AND zero bleed.)
                    rho_pen  = (RHO_ACTIVE_THRESH - rm) * 15.0
                    # 0.3×σ_T added as tiebreaker: when density gain is identical,
                    # prefer the iteration with lower temperature non-uniformity.
                    metrics["score"] = round(
                        0.3 * sm + 2.0 * srho + rho_pen + bleed_pen + sel_pen + dmg,
                        3
                    )
                    metrics["score_regime"] = "active"
                else:
                    # Near-final regime: uniformity is the primary objective.
                    # σ_T is the dominant term; σ_rho adds density-uniformity signal.
                    # Light penalty for deviation from target density.
                    rho_pen = abs(rm - RHO_TARGET) * 1.0
                    metrics["score"] = round(
                        2.0 * sm + 25.0 * srho + rho_pen + bleed_pen + sel_pen + dmg,
                        3
                    )
                    metrics["score_regime"] = "near-final"
            except Exception:
                pass
        return metrics

    # ── Pre-check iter-0: abort only on REAL problems ─────────────────────────
    #
    # Manual mode (use_optimizer=False):
    #   Abort only if the part is under-sintered OR T_max exceeds the PA12 hard
    #   decomposition limit (~350°C).  T_max between DAMAGE_T_C (245°C) and
    #   HARD_DECOMP_T_C (350°C) means the part is in the melt/densification
    #   regime — the hot spots are exactly what FGM iteration is designed to fix.
    #   Aborting here would prevent FGM from ever running on geometries (e.g.
    #   L-shape) whose hot corners naturally exceed 245°C even at short exposures.
    #
    # Optimizer mode (use_optimizer=True):
    #   iter-0 was intentionally run LONGER than the ideal time so the optimizer could
    #   collect intermediate snapshots.  T_max > DAMAGE_T_C at the END of iter-0 is
    #   completely expected — the ideal time chosen by the optimizer is BEFORE that point.
    #   → Do NOT abort on T_max.
    #   → DO abort if ρ̄ < MIN_RHO_VALID — this means sintering never even started
    #     (under-powered, wrong material, or probe too short to see any densification).
    #   → DO abort if optimizer gave "fallback" AND ρ̄ is also low (doubly bad signal).
    HARD_DECOMP_T_C = 350.0   # PA12 decomposition onset; abort if exceeded even at iter-0

    iter0_metrics  = _iter_metrics(iter0_dir)
    iter0_rho      = iter0_metrics.get("mean_rho",  0.0)
    iter0_T_max    = iter0_metrics.get("T_max_c",   0.0)

    abort_reason: str | None = None
    iter0_hot_warning: str | None = None   # warn but do NOT abort
    if use_optimizer:
        # Optimizer path: T_max check suppressed (over-exposure is by design)
        if iter0_rho < MIN_RHO_VALID:
            abort_reason = (
                f"ρ̄={iter0_rho:.3f} < {MIN_RHO_VALID} even after {optimizer_max_min:.1f} min — "
                f"part is not sintering at all. Check RF power and material properties."
            )
    else:
        # Manual path: hard decomposition check; soft check is warning only
        if iter0_rho < MIN_RHO_VALID:
            abort_reason = (
                f"ρ̄={iter0_rho:.3f} < {MIN_RHO_VALID} — part is severely under-sintered. "
                f"Increase exposure time and re-run."
            )
        elif iter0_T_max > HARD_DECOMP_T_C:
            abort_reason = (
                f"T_max={iter0_T_max:.1f}°C > {HARD_DECOMP_T_C:.0f}°C — thermal decomposition "
                f"risk (PA12 hard limit). Reduce exposure time significantly."
            )
        elif iter0_T_max > DAMAGE_T_C:
            # Hot but not decomposing: the part is in the melt regime.
            # FGM will redistribute energy to cool the hot spots — continue.
            iter0_hot_warning = (
                f"⚠ iter-0 T_max={iter0_T_max:.1f}°C > {DAMAGE_T_C:.0f}°C (melt/damage onset) "
                f"but < {HARD_DECOMP_T_C:.0f}°C (decomposition limit) — proceeding with FGM "
                f"iteration to correct hot spots."
            )

    if abort_reason:
        warn_label = f"⚠ {abort_reason} FGM cannot fix operating-point errors."
        _write_job_log(Path(job["log_path"]), warn_label)
        _set_job_progress(job_id, progress_label=warn_label)
        (parent_dir / "convergence.json").write_text(json.dumps({
            "shape": shape, "exposure_minutes": round(exposure_min, 3),
            "proxy_field": proxy_field, "warning": warn_label, "aborted": True,
            "abort_reason": abort_reason,
            "optimizer_chosen_time_min": round(exposure_min, 3) if use_optimizer else None,
            "optimizer_criterion": optimizer_criterion,
            "optimizer_reason": optimizer_reason,
            "iterations": [{"iter": 0, **iter0_metrics,
                             "output_dir": iter0_dir.relative_to(OUTPUTS_DIR).as_posix(),
                             "note": f"baseline (no FGM) — ABORTED: {abort_reason}"}],
        }, indent=2), encoding="utf-8")
        return

    # Log soft warning (T_max in melt regime but below decomposition limit) without aborting
    if iter0_hot_warning:
        _write_job_log(Path(job["log_path"]), iter0_hot_warning)
        print(f"  [fgm_iterate] {iter0_hot_warning}", flush=True)

    # Generate iter-0 FGM at full magnitude (always proportional — no prior exists yet)
    init_result = generate_fgm(
        run_output_dir      = iter0_dir,
        bpp                 = bpp,
        proxy_field         = proxy_field,
        invert              = invert,
        magnitude           = magnitude,
        baseline_saturation = baseline_saturation,
        dead_band           = dead_band,
        emit_formats        = ("npz", "png"),  # include PNG so dashboard can show it
        overprint_cold_mm     = overprint_cold_mm,
        overprint_cold_thresh = overprint_cold_thresh,
    )
    current_fgm_npz = Path(init_result["npz_path"])

    # Capture iter-0 normalization bounds for fixed-reference normalization in
    # integral control mode.  These bounds define the scale of the original problem;
    # subsequent iterations normalize against this fixed range so that a smaller
    # residual field stays proportionally small rather than being re-amplified.
    iter0_ref_lo: float | None = init_result.get("ref_lo")
    iter0_ref_hi: float | None = init_result.get("ref_hi")

    # ── EBC: capture iter-0 mean saturation as the power-budget reference ─────
    # mean_sat = mean(sat_map[part_mask]) — the effective mean σ-scaling factor.
    # Saved once here; subsequent iterations compute their own mean_sat and adjust
    # t★ to keep total absorbed power ≈ constant.
    exposure_min_base: float = exposure_min   # save t★ (possibly optimizer-chosen)
    iter0_mean_sat:    float | None = None
    if use_ebc:
        try:
            _s0_data  = np.load(str(current_fgm_npz))
            _s0_map   = _s0_data["sat_map"].astype(np.float32)
            _f0_path  = iter0_dir / "fields.npz"
            if _f0_path.exists():
                _mask0 = np.load(str(_f0_path))["part_mask"].astype(bool)
                iter0_mean_sat = float(_s0_map[_mask0].mean())
            else:
                _inside0 = _s0_map[_s0_map > 1e-6]
                iter0_mean_sat = float(_inside0.mean()) if len(_inside0) > 0 else None
            if iter0_mean_sat is not None:
                print(
                    f"  [fgm_iterate] EBC enabled — iter-0 mean_sat={iter0_mean_sat:.4f}  "
                    f"t★_base={exposure_min_base:.3f} min",
                    flush=True,
                )
            else:
                print("  [fgm_iterate] EBC: sat_map empty — disabling EBC", flush=True)
                use_ebc = False
        except Exception as _ebc_init_exc:
            print(
                f"  [fgm_iterate] EBC init failed ({_ebc_init_exc}); "
                f"EBC disabled for this run",
                flush=True,
            )
            use_ebc = False

    convergence_log: list[dict[str, Any]] = []
    iter0_sigma_T = iter0_metrics.get("sigma_T", iter0_metrics.get("dT_c", 999.0))

    # ── Early-stop: part already sufficiently uniform at baseline ─────────────
    # If σ_T < MIN_SIGMA_T_FOR_FGM at iter-0, the spatial non-uniformity is too
    # small to correct reliably — any FGM will mostly amplify noise and degrade
    # uniformity.  In this case, skip all FGM iterations and return iter-0 as best.
    MIN_SIGMA_T_FOR_FGM = 4.0   # °C — below this, FGM makes things worse
    if iter0_sigma_T < MIN_SIGMA_T_FOR_FGM and not _resuming:
        early_abort_msg = (
            f"Baseline σ_T={iter0_sigma_T:.2f}°C < {MIN_SIGMA_T_FOR_FGM}°C threshold — "
            f"part is already highly uniform. FGM iterations would amplify noise "
            f"and degrade uniformity. Iter-0 IS the best result."
        )
        print(f"  [fgm_iterate] {early_abort_msg}", flush=True)
        conv_data_early: dict[str, Any] = {
            "shape": shape, "exposure_minutes": round(exposure_min, 3),
            "proxy_field": proxy_field, "magnitude_base": magnitude,
            "bpp": bpp, "n_iterations_run": 1,
            "iterations": [{
                "iter": 0,
                "output_dir": iter0_dir.relative_to(OUTPUTS_DIR).as_posix(),
                "note": "baseline — part already uniform, FGM skipped",
                **iter0_metrics,
            }],
            "best_iter": 0, "best_score": iter0_metrics.get("score"),
            "best_sigma_T": iter0_sigma_T, "best_score_regime": iter0_metrics.get("score_regime"),
            "converged": True, "convergence_type": "already_uniform",
            "convergence_note": early_abort_msg,
            "aborted": False,
        }
        (parent_dir / "convergence.json").write_text(
            json.dumps(conv_data_early, indent=2), encoding="utf-8")
        _set_job_progress(
            job_id, completed_runs=1, progress_pct=100.0,
            progress_label=f"FGM skipped — part already uniform (σ_T={iter0_sigma_T:.2f}°C < {MIN_SIGMA_T_FOR_FGM}°C)",
        )
        return

    # ── Auto-scale initial magnitude by severity of baseline non-uniformity ──
    # Rationale: magnitude=1.0 on an already-uniform field (σ_T≈4°C) creates huge
    # saturation swings → frac_melt explosion.  Full magnitude is only warranted when
    # σ_T ≥ 15°C.  For smaller gradients we scale down proportionally.
    severity_scale = min(1.0, max(0.1, iter0_sigma_T / 15.0))
    current_mag    = magnitude * severity_scale
    print(
        f"  [fgm_iterate] baseline σ_T={iter0_sigma_T:.2f}°C "
        f"→ severity_scale={severity_scale:.3f} → initial mag={current_mag:.3f}",
        flush=True,
    )

    # Build iter-0 note: describe what kind of run it was
    if use_optimizer and optimizer_criterion and optimizer_criterion != "fallback":
        _crit_str = {"melt": "φ=0.90", "max_density": "max-density (extended)",
                     "phi_crossing": "φ=0.90"}.get(optimizer_criterion, optimizer_criterion)
        iter0_note = (
            f"optimizer probe at {optimizer_max_min:.1f} min — "
            f"ideal time = {exposure_min:.2f} min ({_crit_str}); "
            f"iter-1+ use the ideal time"
        )
    elif use_optimizer:
        iter0_note = (
            f"optimizer probe at {optimizer_max_min:.1f} min — no ideal time found; "
            f"iter-1+ use max time {exposure_min:.1f} min"
        )
    else:
        iter0_note = "baseline (no FGM applied) — FGM generated for iter-1"

    convergence_log.append({
        "iter":       0,
        "output_dir": iter0_dir.relative_to(OUTPUTS_DIR).as_posix(),
        "fgm_npz":    str(current_fgm_npz.relative_to(ROOT)),
        "fgm_png":    (str(Path(init_result.get("png_path","")).relative_to(ROOT))
                       if init_result.get("png_path") else None),
        "note":       iter0_note,
        "magnitude_used": round(current_mag, 4),
        "optimizer_probe_min": round(optimizer_max_min, 3) if use_optimizer else None,
        **iter0_metrics,
    })
    prev_sigma_T    = iter0_sigma_T
    prev_rho        = iter0_metrics.get("mean_rho", 0.0)
    prev_score      = iter0_metrics.get("score", 999.0)
    iter0_regime    = iter0_metrics.get("score_regime", "active")
    # ── Per-regime best tracking ──────────────────────────────────────────────
    # Active-regime and near-final scores use DIFFERENT formulas and are
    # INCOMPARABLE.  If best_score_ever is set from an active-regime iteration
    # (score ≈ 0.3) and then a near-final iteration (score ≈ 20) is compared
    # against it, the near-final run appears to "regress" catastrophically —
    # even when it is physically BETTER (higher ρ̄, lower bleed, more sintered).
    # Fix: track best scores PER REGIME.  Reset best_score_ever each time the
    # regime transitions; the final best_entry prefers near-final over active.
    best_score_active     = prev_score if iter0_regime == "active"     else 999.0
    best_score_near_final = prev_score if iter0_regime == "near-final" else 999.0
    best_score_ever       = prev_score   # kept for backward compatibility, updated per regime
    prev_regime           = iter0_regime
    consecutive_bad = 0   # count iterations where score is significantly worse than best-ever
    # Active-regime convergence: ρ̄ change below threshold for N consecutive iters
    RHO_PLATEAU_DELTA  = 0.002   # ρ̄ change smaller than this is considered "plateau"
    consecutive_plateau = 0     # consecutive iters where |Δρ̄| < RHO_PLATEAU_DELTA

    # ── Anti-rut / proxy-switching state ─────────────────────────────────────
    # proxy_schedule is a list of proxy fields to try in sequence.
    # When thorough=True (or regime_adaptive=True) the active proxy can change
    # mid-run.  proxy_schedule_idx tracks which schedule entry is current.
    proxy_schedule_idx  = 0
    proxy_field         = proxy_schedule[0]       # active proxy (may change per iter)
    stagnation_counter  = 0                       # iters without meaningful σ_T improvement
    best_sigma_T_this_proxy = iter0_sigma_T       # reset each time proxy switches
    proxy_switch_log    = []                      # record of when/why proxy switched
    # How many iters each proxy has been active (for reporting)
    proxy_iter_count    = {p: 0 for p in proxy_schedule}

    # ── Stochastic perturbation state ────────────────────────────────────────
    # _gen_stagnation_counter tracks how many consecutive iterations have failed
    # to improve σ_T by at least stagnation_eps.  When it reaches stagnation_window,
    # the NEXT generate_fgm() call will include random noise to escape local minima.
    # This is independent of the proxy-switching logic (both can fire simultaneously).
    _gen_stagnation_counter   = 0
    _gen_best_sigma_T         = iter0_sigma_T
    _next_perturbation_amp    = 0.0          # perturbation to apply at the NEXT FGM gen
    _post_perturb_grace       = 0            # counts down after each perturbation fires
    _min_sigma_T_ever         = iter0_sigma_T  # true minimum σ_T across all iterations
    _min_sigma_T_iter         = 0            # iteration that achieved it

    # ── Hybrid mode phase state ───────────────────────────────────────────────
    # Phase 1 (proportional) runs for exactly iter-1; phase 2 (OC-TO) for all
    # subsequent iterations.  _hybrid_phase2_active is set True at the first
    # FGM generation after iter-1 completes and stays True for the remainder.
    _hybrid_phase2_active           = False
    _hybrid_phase2_baseline_sigma_T = iter0_sigma_T  # updated when phase switches

    # ── Outer loop: iterations 1 … n_iterations-1 ────────────────────────────
    fgm_iters     = n_iterations - 1

    for i in range(1, fgm_iters + 1):
        abs_iter  = i
        iter_label = f"FGM iter-{abs_iter}/{fgm_iters} | σ_T was {prev_sigma_T:.1f}°C"
        _set_job_progress(
            job_id,
            completed_runs=i,
            progress_pct=max(5.0, 100.0 * i / total_runs),
            progress_label=iter_label,
        )

        # Build iter config: base + fgm_feedback block + exposure time
        iter_cfg = dict(base_cfg)
        iter_cfg["fgm_feedback"] = {
            "enabled":                True,
            "saturation_map_npz":     str(current_fgm_npz),
            "magnitude":              current_mag,
            "baseline_saturation":    baseline_saturation,
            "proxy_field":            proxy_field,
            "invert":                 invert,
            "iterate":                iterate_inner,
            "iterate_interval_steps": iterate_interval,
            "iterate_damping":        iterate_damping,
        }

        # ── EBC: adjust t★ to conserve total absorbed energy ─────────────────
        # σ(x,y) = σ₀ × sat_map(x,y)  →  P ∝ mean(sat_map)
        # t★ᵢ = t★_base × (mean_sat₀ / mean_satᵢ)   [first-order EBC]
        exposure_for_iter = exposure_min   # default: unchanged
        ebc_ratio_used: float | None = None
        if use_ebc and iter0_mean_sat is not None and iter0_mean_sat > 1e-6:
            try:
                _si_data  = np.load(str(current_fgm_npz))
                _si_map   = _si_data["sat_map"].astype(np.float32)
                _fi_path  = current_fgm_npz.parent / "fields.npz"
                if _fi_path.exists():
                    _maski = np.load(str(_fi_path))["part_mask"].astype(bool)
                    iter_mean_sat_i = float(_si_map[_maski].mean())
                else:
                    _insi = _si_map[_si_map > 1e-6]
                    iter_mean_sat_i = float(_insi.mean()) if len(_insi) > 0 else iter0_mean_sat
                if iter_mean_sat_i > 1e-6:
                    raw_ratio = iter0_mean_sat / iter_mean_sat_i
                    ebc_ratio_used = max(ebc_clamp_lo, min(ebc_clamp_hi, raw_ratio))
                    exposure_for_iter = exposure_min_base * ebc_ratio_used
                    print(
                        f"  [fgm_iterate] EBC iter-{i}: "
                        f"mean_sat {iter0_mean_sat:.4f}→{iter_mean_sat_i:.4f}  "
                        f"raw_ratio={raw_ratio:.3f}  clamped={ebc_ratio_used:.3f}  "
                        f"t★={exposure_min_base:.3f}→{exposure_for_iter:.3f} min",
                        flush=True,
                    )
            except Exception as _ebc_exc:
                print(
                    f"  [fgm_iterate] EBC iter-{i} failed ({_ebc_exc}); "
                    f"using base exposure",
                    flush=True,
                )

        _set_exposure_minutes(iter_cfg, exposure_for_iter)

        GUI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cfg_path = GUI_CONFIG_DIR / f"fgm_iter_{job_id}_i{i}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as fh:
            yaml.dump(iter_cfg, fh, default_flow_style=False, allow_unicode=True)

        iter_dir = parent_dir / f"{output_name}_iter{i}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        _register_job_output(job, iter_dir)

        cmd = [sys.executable, "rfam_eqs_coupled.py",
               "--config", str(cfg_path), "--output-dir", str(iter_dir)]
        rc = _run_command(cmd, Path(job["log_path"]), job_id=job_id)
        if rc != 0:
            _set_job_progress(job_id, progress_label=f"{iter_label} — simulation failed")
            raise RuntimeError(f"FGM iteration {i} exited with code {rc}")

        # Compute full metrics from this iter's output
        metrics = _iter_metrics(iter_dir)
        sigma_T    = metrics.get("sigma_T", metrics.get("dT_c", 999.0))
        this_score = metrics.get("score", 999.0)

        convergence_log.append({
            "iter":           abs_iter,
            "output_dir":     iter_dir.relative_to(OUTPUTS_DIR).as_posix(),
            "fgm_npz":        str(current_fgm_npz.relative_to(ROOT)),
            "fgm_png":        None,   # filled in below after next FGM generated
            "magnitude_used": round(current_mag, 4),
            "proxy_field":    proxy_field,   # record active proxy for this iter
            # TO parameters used for this iteration's FGM correction
            "move_limit":               round(move_limit, 4) if use_delta_correction else None,
            "sensitivity_filter_sigma": round(sensitivity_filter_sigma, 3) if use_delta_correction else None,
            # EBC fields: actual exposure used and the power-ratio that drove it
            "exposure_min_used": round(exposure_for_iter, 4),
            "ebc_ratio":         round(ebc_ratio_used, 4) if ebc_ratio_used is not None else None,
            **metrics,
        })

        # ── Track true minimum σ_T across all iterations ─────────────────────
        _st_this = metrics.get("sigma_T")
        if isinstance(_st_this, (int, float)) and _st_this < _min_sigma_T_ever:
            _min_sigma_T_ever = _st_this
            _min_sigma_T_iter = abs_iter

        # ── Check for divergence / thermal damage before generating next FGM ──
        early_stop_reason: str | None = None

        # (a) Thermal damage abort: T_max crossed the decomposition threshold.
        # Threshold is adaptive: max(DAMAGE_T_C, iter0_T_max * 1.15).
        # Geometries that naturally run hot at their optimal exposure (e.g. L-shape
        # with inner concavity) have iter0_T_max >> DAMAGE_T_C even without FGM.
        # In those cases we only abort if the FGM-corrected run RAISES T_max by
        # more than 15% above the baseline — a true FGM-driven runaway, not a
        # pre-existing geometry characteristic.
        this_T_max     = metrics.get("T_max_c", 0.0)
        damage_ceiling = max(DAMAGE_T_C, iter0_T_max * 1.15)
        if this_T_max > damage_ceiling:
            early_stop_reason = (
                f"iter-{abs_iter} T_max={this_T_max:.1f}°C exceeded "
                f"thermal damage ceiling {damage_ceiling:.0f}°C "
                f"(max({DAMAGE_T_C:.0f}, iter0={iter0_T_max:.0f}°C × 1.15))"
            )
            convergence_log[-1]["note"] = f"thermal damage abort at iter-{abs_iter}"

        # (b) Regime-aware regression abort.
        #
        # The two scoring regimes have different expectations:
        #
        #   ACTIVE (ρ̄ < 0.82): score CAN rise while ρ̄ rises — the FGM is concentrating
        #   heat in under-sintered regions, which naturally increases σ_T temporarily.
        #   This is NOT a regression — it's expected physics.  We only abort if ρ̄ is
        #   actually declining (FGM pushing a well-sintered region into the under-sintered
        #   zone) or bleed is growing (FGM over-correcting and burning surrounding powder).
        #
        #   NEAR-FINAL (ρ̄ ≥ 0.82): sintering is mostly complete.  Score should now be
        #   trending downward (lower σ_T, lower bleed).  Two consecutive score rises
        #   > 30% above best-ever → abort.
        #
        this_regime   = metrics.get("score_regime", "near-final")
        this_rho      = metrics.get("mean_rho", 0.0)
        this_bleed    = metrics.get("frac_bleed", 0.0)
        prev_entry    = convergence_log[-2] if len(convergence_log) >= 2 else {}
        prev_rho      = prev_entry.get("mean_rho", this_rho)
        prev_bleed    = prev_entry.get("frac_bleed", 0.0)

        # ── Update per-regime best tracking ─────────────────────────────────────
        # When the regime transitions from active → near-final, reset consecutive_bad
        # and best_score_ever to the current near-final score so the regression check
        # doesn't compare near-final scores (typically 15–25) against active-regime
        # scores (typically 0.3–2.0).  The two formulas are dimensionally incomparable.
        regime_just_crossed = (prev_regime == "active" and this_regime == "near-final")
        if regime_just_crossed:
            best_score_ever       = this_score   # fresh start for near-final tracking
            best_score_near_final = this_score
            consecutive_bad       = 0            # prior active-regime streak is irrelevant
            print(
                f"  [fgm_iterate] regime transition active→near-final at iter-{abs_iter} "
                f"(ρ̄={this_rho:.3f}) — resetting best_score_ever to {this_score:.3f}",
                flush=True,
            )
        elif this_score < best_score_ever:
            best_score_ever = this_score

        if this_regime == "active":
            if this_score < best_score_active:
                best_score_active = this_score
            # In active regime: abort only if density dropped or bleed grew 2× in a row
            rho_declined  = this_rho < prev_rho - 0.02        # meaningful drop
            bleed_grew    = this_bleed > prev_bleed + 0.005   # >0.5pp new bleed
            if rho_declined and bleed_grew:
                consecutive_bad += 1
            elif rho_declined and not bleed_grew and consecutive_bad > 0:
                # partial recovery — don't reset fully but slow count
                consecutive_bad = max(0, consecutive_bad - 1)
            else:
                consecutive_bad = 0
            regression_threshold = None   # not used in active regime
        else:
            if this_score < best_score_near_final:
                best_score_near_final = this_score
            # Near-final regime: score-based regression check within same regime only.
            # Grace period suppresses regression counting immediately after a perturbation
            # so the new seed gets post_perturbation_grace iters to show improvement.
            regression_threshold = best_score_ever * regression_threshold_multiplier
            _in_grace = _post_perturb_grace > 0
            if _post_perturb_grace > 0:
                _post_perturb_grace -= 1
            if this_score > regression_threshold and not regime_just_crossed and not _in_grace:
                consecutive_bad += 1
            elif not _in_grace:
                consecutive_bad = 0

        if consecutive_bad >= 2 and early_stop_reason is None:
            if this_regime == "active":
                early_stop_reason = (
                    f"iter-{abs_iter} ρ̄ declined and bleed grew for "
                    f"{consecutive_bad} consecutive iterations "
                    f"(ρ̄ {prev_rho:.3f}→{this_rho:.3f}, "
                    f"bleed {prev_bleed*100:.2f}%→{this_bleed*100:.2f}%)"
                )
            else:
                early_stop_reason = (
                    f"score regressed for {consecutive_bad} consecutive iterations "
                    f"({this_score:.2f} vs best-ever {best_score_ever:.2f}, "
                    f"threshold {regression_threshold:.2f})"
                )
            convergence_log[-1]["note"] = f"regression abort at iter-{abs_iter}"

        # ── Best iteration selection ──────────────────────────────────────────────
        best_entry = _select_best_iter(convergence_log, use_optimizer, DAMAGE_T_C)
        conv_data: dict[str, Any] = {
            "shape":            shape,
            "exposure_minutes": round(exposure_min, 3),
            "proxy_field":      proxy_field,        # current active proxy (may have switched)
            "proxy_schedule":   proxy_schedule,     # full schedule used
            "thorough":         thorough,
            "regime_adaptive":  regime_adaptive,
            "proxy_switch_log": proxy_switch_log,
            "proxy_iter_count": {k: v for k, v in proxy_iter_count.items() if v > 0},
            "magnitude_base":   magnitude,
            "severity_scale":   round(severity_scale, 3),
            "magnitude_decay":  magnitude_decay,
            "dead_band":        dead_band,
            "fgm_momentum":     fgm_momentum,
            "bpp":              bpp,
            "n_iterations_run": abs_iter + 1,
            "iterations":       convergence_log,
            "best_iter":         best_entry["iter"],
            "best_score":        best_entry.get("score"),
            "best_sigma_T":      best_entry.get("sigma_T"),
            "best_score_regime": best_entry.get("score_regime", "near-final"),
            # True minimum σ_T observed (may differ from best_iter when active-regime
            # score formula prioritises density over uniformity).
            "min_sigma_T":       round(_min_sigma_T_ever, 3),
            "min_sigma_T_iter":  _min_sigma_T_iter,
            # Converged = either σ_T convergence (near-final) or ρ̄ plateau (active).
            # Filled in after break below; during the loop it stays False.
            "converged":         False,
            "convergence_type":  None,
            "optimizer_chosen_time_min": round(exposure_min, 3) if use_optimizer else None,
            "optimizer_criterion": optimizer_criterion,
            "optimizer_reason": optimizer_reason,
            # OC-TO integral mode metadata
            "use_delta_correction":     use_delta_correction,
            "iter0_ref_lo":             iter0_ref_lo,
            "iter0_ref_hi":             iter0_ref_hi,
            "move_limit":               move_limit,
            "sensitivity_filter_sigma": sensitivity_filter_sigma,
            # EBC metadata (M3 — Energy-Budget-Conserving exposure adjustment)
            "use_ebc":              use_ebc,
            "iter0_mean_sat":       round(iter0_mean_sat, 4) if iter0_mean_sat is not None else None,
            "exposure_min_base":    round(exposure_min_base, 3),
            "ebc_clamp_lo":         ebc_clamp_lo,
            "ebc_clamp_hi":         ebc_clamp_hi,
            # Hybrid mode metadata
            "use_hybrid":               use_hybrid,
            "hybrid_phase2_active":     _hybrid_phase2_active,
            "hybrid_phase2_magnitude":  hybrid_phase2_magnitude   if use_hybrid else None,
            "hybrid_phase2_move_limit": hybrid_phase2_move_limit  if use_hybrid else None,
            "hybrid_phase2_sensitivity_sigma": hybrid_phase2_sensitivity_sigma if use_hybrid else None,
            "hybrid_phase2_baseline_sigma_T":  round(_hybrid_phase2_baseline_sigma_T, 3) if use_hybrid else None,
            # Cold-zone overprinting
            "overprint_cold_mm":     overprint_cold_mm,
            "overprint_cold_thresh": overprint_cold_thresh,
            "min_magnitude":         min_magnitude,
        }
        if early_stop_reason:
            conv_data["aborted"]      = True
            conv_data["abort_reason"] = early_stop_reason
        (parent_dir / "convergence.json").write_text(
            json.dumps(conv_data, indent=2), encoding="utf-8"
        )

        direction = "↓" if sigma_T < prev_sigma_T else "↑"
        progress_label = (f"FGM iter-{abs_iter}/{fgm_iters} done | "
                          f"σ_T: {prev_sigma_T:.1f}°C → {sigma_T:.1f}°C {direction}  "
                          f"mag={current_mag:.2f}")
        _set_job_progress(job_id, completed_runs=i + 1, progress_label=progress_label)

        # ── FGM generation — ALWAYS runs, even for aborted iterations ────────
        # Generating the FGM before the abort break means every iteration has a
        # diagnostic saturation map visible in the convergence dashboard, even when
        # the run stopped early.  The FGM from an aborted iter is NOT used to seed
        # the next run (there is no next run), but it is useful for understanding
        # what correction was being attempted.
        if (iter_dir / "fields.npz").exists():
            # ── Hybrid phase transition ────────────────────────────────────────
            # When use_hybrid=True and we just completed iter-1 (abs_iter == 1):
            # switch from proportional to OC-TO mode.  This fires exactly once.
            _switching_to_octo = use_hybrid and abs_iter == 1 and not _hybrid_phase2_active
            if _switching_to_octo:
                _hybrid_phase2_active           = True
                _hybrid_phase2_baseline_sigma_T = sigma_T   # iter-1 σ_T is new baseline
                # Reset regression tracking so phase-2 scores don't compare against
                # the favorable active/proportional iter-1 score (dimensionally incomparable)
                best_score_ever       = this_score
                best_score_near_final = this_score
                consecutive_bad       = 0
                print(
                    f"  [fgm_iterate] HYBRID → switching to OC-TO phase after iter-{abs_iter} "
                    f"(σ_T={sigma_T:.2f}°C — new phase-2 baseline); "
                    f"phase-2 mag={hybrid_phase2_magnitude:.3f}, "
                    f"move_limit={hybrid_phase2_move_limit:.3f}",
                    flush=True,
                )

            # ── Adaptive magnitude schedule ────────────────────────────────────
            # Three cases:
            #   (A) Hybrid phase 2 just starting (switching_to_octo): reset to
            #       hybrid_phase2_magnitude, scaled by σ_T severity within phase 2.
            #   (B) Already in hybrid phase 2: decay from current_mag, cap by σ_T
            #       ratio relative to phase-2 baseline (not iter-0 baseline).
            #   (C) Standard proportional or non-hybrid OC-TO: existing schedule.
            if _switching_to_octo:
                # Fresh start for OC-TO phase — scale starting magnitude by how much
                # σ_T was already reduced in phase-1 so OC-TO doesn't overshoot the
                # now-smaller residual.  E.g. if iter-1 got σ_T from 19→8.8°C (55%
                # reduction), we scale: m_phase2 × (8.8/19) ≈ 0.4 × 0.46 ≈ 0.18.
                _residual_scale = sigma_T / max(iter0_sigma_T, 0.1)
                _scaled_phase2_mag = hybrid_phase2_magnitude * _residual_scale
                next_mag = max(min_magnitude, _scaled_phase2_mag)
                print(
                    f"  [fgm_iterate] [hybrid phase-2] mag reset to {next_mag:.3f} "
                    f"(phase2_mag={hybrid_phase2_magnitude:.3f} × residual_scale={_residual_scale:.3f})",
                    flush=True,
                )
            elif _hybrid_phase2_active:
                _geom_decay_mag  = current_mag * magnitude_decay
                _sigma_ratio_cap = sigma_T / max(_hybrid_phase2_baseline_sigma_T, 0.1)
                _phase2_prop_mag = hybrid_phase2_magnitude * _sigma_ratio_cap
                next_mag = max(min_magnitude, min(_geom_decay_mag, _phase2_prop_mag))
                print(
                    f"  [fgm_iterate] [hybrid phase-2] mag: "
                    f"geom={_geom_decay_mag:.3f} σ_T_cap={_phase2_prop_mag:.3f} "
                    f"→ {next_mag:.3f}  "
                    f"(σ_T={sigma_T:.2f}°C vs phase-2 baseline "
                    f"{_hybrid_phase2_baseline_sigma_T:.2f}°C)",
                    flush=True,
                )
            else:
                # Standard magnitude schedule (proportional phase or pure OC-TO mode)
                # Pure geometric decay only — no sigma_T_cap to prevent contrast collapse
                # after a strong iter-1 correction (the cap was permanently locking magnitude
                # at ~37% of initial because sigma_T/iter0_sigma_T < 0.4 after iter-1).
                next_mag = max(min_magnitude, current_mag * magnitude_decay)
                print(
                    f"  [fgm_iterate] mag schedule: geom_decay → next_mag={next_mag:.3f}"
                    f" (current={current_mag:.3f} × decay={magnitude_decay:.3f},"
                    f" σ_T={sigma_T:.2f}°C)",
                    flush=True,
                )

            # ── Effective correction parameters ───────────────────────────────
            # Hybrid phase 2 forces OC-TO with its own move_limit and sensitivity_sigma.
            # Non-hybrid: use the explicit use_delta_correction flag.
            _in_octo_phase   = use_delta_correction or _hybrid_phase2_active
            _eff_move_limit  = (
                hybrid_phase2_move_limit        if _hybrid_phase2_active else
                (move_limit                     if use_delta_correction  else 0.0)
            )
            _eff_sens_sigma  = (
                hybrid_phase2_sensitivity_sigma if _hybrid_phase2_active else
                (sensitivity_filter_sigma       if use_delta_correction  else 0.0)
            )

            try:
                _has_prior = not early_stop_reason and current_fgm_npz is not None
                # Consume the pending perturbation amplitude (set by stagnation detector
                # in the PREVIOUS iteration; reset to 0 immediately so only one iter is
                # perturbed per stagnation event).
                _this_perturbation = _next_perturbation_amp
                _next_perturbation_amp = 0.0
                if _this_perturbation > 0.0:
                    # Give the new seed post_perturbation_grace iters before
                    # regression-abort tracking resumes.
                    consecutive_bad       = 0
                    _post_perturb_grace   = post_perturbation_grace
                    print(
                        f"  [fgm_iterate] Applying stochastic perturbation "
                        f"amplitude={_this_perturbation:.3f} to escape local minimum; "
                        f"regression abort suspended for {post_perturbation_grace} iters",
                        flush=True,
                    )
                next_fgm = generate_fgm(
                    run_output_dir      = iter_dir,
                    bpp                 = bpp,
                    proxy_field         = proxy_field,
                    invert              = invert,
                    magnitude           = next_mag,
                    baseline_saturation = baseline_saturation,
                    dead_band           = dead_band,
                    emit_formats        = ("npz", "png"),
                    # OC-TO integral mode: accumulate corrections with move limits and
                    # sensitivity filter (adaptive normalization, no fixed ref bounds).
                    # In hybrid mode, _in_octo_phase is True from phase-2 onwards.
                    use_delta_correction     = _in_octo_phase,
                    ref_lo                   = None,   # always adaptive normalization
                    ref_hi                   = None,
                    prior_sat_npz            = current_fgm_npz if _has_prior else None,
                    move_limit               = _eff_move_limit,
                    sensitivity_filter_sigma = _eff_sens_sigma,
                    # Stochastic perturbation for local-minimum escape
                    perturbation_amplitude   = _this_perturbation,
                    # Momentum blending (proportional mode only; bypassed in OC-TO).
                    momentum = (0.0 if _in_octo_phase
                                else (fgm_momentum if _has_prior else 0.0)),
                    # Cold-zone overprinting
                    overprint_cold_mm     = overprint_cold_mm,
                    overprint_cold_thresh = overprint_cold_thresh,
                )
                # Advance current_fgm_npz only when continuing (not aborting, not final iter)
                if i < fgm_iters and not early_stop_reason:
                    current_mag     = next_mag
                    current_fgm_npz = Path(next_fgm["npz_path"])
                # Update this iter's log entry with its OUTPUT FGM path + PNG
                if convergence_log:
                    convergence_log[-1]["fgm_npz"] = str(
                        Path(next_fgm["npz_path"]).relative_to(ROOT))
                    convergence_log[-1]["fgm_png"] = (
                        str(Path(next_fgm.get("png_path", "")).relative_to(ROOT))
                        if next_fgm.get("png_path") else None
                    )
                    # Record which correction algorithm generated the FGM OUTPUT
                    # by this iteration (i.e., what seeds the NEXT iteration):
                    # phase-2 = OC-TO (just switched or already active), else proportional.
                    convergence_log[-1]["fgm_algo"] = (
                        "OC-TO"        if (_switching_to_octo or _hybrid_phase2_active) else
                        "proportional"
                    )
                # Patch convergence.json on disk with the updated FGM paths
                if (parent_dir / "convergence.json").exists():
                    try:
                        _cd = json.loads(
                            (parent_dir / "convergence.json").read_text(encoding="utf-8"))
                        for entry in _cd.get("iterations", []):
                            if entry.get("iter") == abs_iter:
                                entry["fgm_npz"] = convergence_log[-1]["fgm_npz"]
                                entry["fgm_png"] = convergence_log[-1]["fgm_png"]
                                break
                        (parent_dir / "convergence.json").write_text(
                            json.dumps(_cd, indent=2), encoding="utf-8")
                    except Exception as _patch_exc:
                        _write_job_log(
                            Path(job["log_path"]),
                            f"[fgm_iterate] WARNING: could not patch convergence.json: {_patch_exc}",
                        )
            except Exception as _fgm_exc:
                _write_job_log(
                    Path(job["log_path"]),
                    f"[fgm_iterate] WARNING: FGM generation failed for iter-{abs_iter}: {_fgm_exc}",
                )
        else:
            _write_job_log(
                Path(job["log_path"]),
                f"[fgm_iterate] WARNING: fields.npz not in iter-{i} output; skipping FGM generation",
            )

        # ── Abort break ───────────────────────────────────────────────────────
        if early_stop_reason:
            _set_job_progress(
                job_id,
                progress_label=f"FGM early stop — {early_stop_reason}",
            )
            break

        prev_sigma_T = sigma_T
        prev_score   = this_score   # for progress label display only (abort uses best_score_ever)
        prev_regime  = this_regime  # track regime transitions
        proxy_iter_count[proxy_field] = proxy_iter_count.get(proxy_field, 0) + 1

        # ── General stagnation tracking (for perturbation, applies in all modes) ─
        # Update BEFORE proxy-switching so the counter captures true σ_T stagnation.
        if sigma_T < _gen_best_sigma_T - stagnation_eps:
            _gen_best_sigma_T       = sigma_T
            _gen_stagnation_counter = 0
        else:
            _gen_stagnation_counter += 1

        # When perturbation is enabled and we've been stuck, flag the NEXT FGM
        # generation to add stochastic noise (mean-zero, so volume-neutral).
        if (perturbation_amplitude > 0.0
                and perturbation_on_stagnation
                and _gen_stagnation_counter >= stagnation_window):
            _next_perturbation_amp = perturbation_amplitude
            _gen_stagnation_counter = 0          # reset so we don't perturb every iter
            print(
                f"  [fgm_iterate] STAGNATION detected — will perturb next FGM "
                f"with amplitude={_next_perturbation_amp:.3f} "
                f"(σ_T stagnant at ~{sigma_T:.1f}°C)",
                flush=True,
            )

        # ── Stagnation detection & proxy switching (thorough / regime_adaptive) ─
        # Check whether the current proxy is still making progress.  If σ_T has
        # not improved by at least stagnation_eps°C for stagnation_window consecutive
        # iterations, try switching to the next proxy in the schedule.
        if thorough or regime_adaptive:
            # Regime-adaptive override: use Qrf during active sintering, T_phi90 when
            # near-final.  This takes priority over the schedule-based rotation when
            # regime_adaptive=True.
            if regime_adaptive:
                if this_regime == "active":
                    desired_proxy = "Qrf"
                else:
                    desired_proxy = "T_phi90"
                if desired_proxy != proxy_field:
                    old_proxy = proxy_field
                    proxy_field = desired_proxy
                    stagnation_counter = 0
                    best_sigma_T_this_proxy = sigma_T
                    current_mag = min(current_mag * 1.5, magnitude)  # small boost on switch
                    switch_note = (
                        f"regime-adaptive switch {old_proxy}→{proxy_field} "
                        f"at iter-{abs_iter} (regime={this_regime}, ρ̄={this_rho:.3f})"
                    )
                    proxy_switch_log.append({"iter": abs_iter, "from": old_proxy,
                                             "to": proxy_field, "reason": "regime"})
                    convergence_log[-1]["proxy_switch"] = switch_note
                    print(f"  [fgm_iterate] {switch_note}", flush=True)

            elif thorough:  # schedule-based rotation
                improved = sigma_T < best_sigma_T_this_proxy - stagnation_eps
                if improved:
                    best_sigma_T_this_proxy = sigma_T
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                if (stagnation_counter >= stagnation_window
                        and proxy_schedule_idx < len(proxy_schedule) - 1):
                    proxy_schedule_idx += 1
                    old_proxy  = proxy_field
                    proxy_field = proxy_schedule[proxy_schedule_idx]
                    stagnation_counter = 0
                    best_sigma_T_this_proxy = sigma_T
                    # Reset magnitude to a moderate value on proxy switch so the new
                    # proxy gets a fair chance with reasonable correction amplitude.
                    current_mag = min(current_mag * 1.5, magnitude * severity_scale)
                    switch_note = (
                        f"stagnation switch {old_proxy}→{proxy_field} "
                        f"at iter-{abs_iter} (σ_T unchanged for "
                        f"{stagnation_window} iters)"
                    )
                    proxy_switch_log.append({"iter": abs_iter, "from": old_proxy,
                                             "to": proxy_field, "reason": "stagnation"})
                    convergence_log[-1]["proxy_switch"] = switch_note
                    print(f"  [fgm_iterate] {switch_note}", flush=True)
                    # Also update the current-proxy label in the progress bar
                    _set_job_progress(
                        job_id,
                        progress_label=(f"FGM iter-{abs_iter}: switched proxy "
                                        f"{old_proxy}→{proxy_field} (stagnation escape)"),
                    )

        # ── Convergence check 1: near-final regime — σ_T below threshold ─────
        if sigma_T < conv_threshold and this_regime == "near-final":
            conv_data["converged"] = True
            (parent_dir / "convergence.json").write_text(
                json.dumps(conv_data, indent=2), encoding="utf-8"
            )
            _set_job_progress(
                job_id,
                progress_label=(
                    f"Converged after {abs_iter} FGM iteration(s) — "
                    f"σ_T={sigma_T:.1f}°C < {conv_threshold}°C threshold"
                ),
            )
            break

        # ── Convergence check 2: active regime — ρ̄ plateau ──────────────────
        # If the FGM can no longer improve density (|Δρ̄| < RHO_PLATEAU_DELTA for
        # 2 consecutive iters) AND bleed is zero, the FGM has done all it can at
        # this operating point.  The part needs longer exposure to reach near-final.
        rho_delta = abs(this_rho - prev_rho)
        if this_regime == "active" and rho_delta < RHO_PLATEAU_DELTA and this_bleed < 0.002:
            consecutive_plateau += 1
        else:
            consecutive_plateau = 0

        if consecutive_plateau >= 2:
            plateau_msg = (
                f"ρ̄ plateau in active regime: |Δρ̄|={rho_delta:.4f} < {RHO_PLATEAU_DELTA} "
                f"for {consecutive_plateau} consecutive iters (ρ̄={this_rho:.3f}). "
                f"FGM has maximally redistributed energy at this exposure time. "
                f"To reach near-final regime (ρ̄≥0.82), increase exposure time."
            )
            conv_data["converged"]         = True
            conv_data["convergence_type"]  = "rho_plateau"
            conv_data["convergence_note"]  = plateau_msg
            (parent_dir / "convergence.json").write_text(
                json.dumps(conv_data, indent=2), encoding="utf-8"
            )
            _set_job_progress(
                job_id,
                progress_label=f"Converged (ρ̄ plateau) after {abs_iter} iters — {plateau_msg[:80]}…",
            )
            break

        prev_rho = this_rho

    best_entry = _select_best_iter(convergence_log, use_optimizer, DAMAGE_T_C)
    # Recompute true min_sigma_T across all finished iterations (catches last iter too)
    for _e in convergence_log:
        _st = _e.get("sigma_T")
        if isinstance(_st, (int, float)) and _st < _min_sigma_T_ever:
            _min_sigma_T_ever = _st
            _min_sigma_T_iter = _e["iter"]

    # Write final convergence.json with all metadata
    _final_conv: dict[str, Any] = {
        k: v for k, v in {
            **conv_data,
            "n_iterations_run": abs_iter + 1 if "abs_iter" in dir() else 1,
            "best_iter":         best_entry["iter"],
            "best_score":        best_entry.get("score"),
            "best_sigma_T":      best_entry.get("sigma_T"),
            "best_score_regime": best_entry.get("score_regime", "near-final"),
            "min_sigma_T":       round(_min_sigma_T_ever, 3),
            "min_sigma_T_iter":  _min_sigma_T_iter,
        }.items()
    }
    if (parent_dir / "convergence.json").exists():
        try:
            _existing = json.loads((parent_dir / "convergence.json").read_text(encoding="utf-8"))
            _existing.update(_final_conv)
            (parent_dir / "convergence.json").write_text(
                json.dumps(_existing, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    _set_job_progress(
        job_id, completed_runs=total_runs, progress_pct=100.0,
        progress_label=(
            f"FGM optimize complete — min σ_T={_min_sigma_T_ever:.1f}°C at iter-{_min_sigma_T_iter} "
            f"(baseline {iter0_sigma_T:.1f}°C, {100*(1-_min_sigma_T_ever/iter0_sigma_T):.0f}% reduction)"
        ),
    )


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


def _launch_antennae_size_sweep_mode(payload: dict[str, Any], job: dict[str, Any]) -> None:
    """Run multiple single simulations with fixed anchor positions but varying antenna sizes."""
    output_prefix = str(payload.get("output_name", "")).strip()
    if not _is_valid_output_name(output_prefix):
        raise ValueError("output_name must use only letters, numbers, '_' or '-'")

    anchors = payload.get("antennae_explicit_instances", [])
    if not isinstance(anchors, list) or not anchors:
        raise ValueError("antennae_explicit_instances must be a non-empty list for antenna size sweep")

    sweep_min = float(payload.get("antennae_sweep_min_mm", 0.5))
    sweep_max = float(payload.get("antennae_sweep_max_mm", 2.0))
    sweep_steps = max(2, int(payload.get("antennae_sweep_steps", 4)))
    exposure_minutes = float(payload.get("exposure_minutes", 6.0))
    if exposure_minutes <= 0:
        exposure_minutes = 6.0

    import numpy as _np
    sizes = list(_np.linspace(sweep_min, sweep_max, sweep_steps))
    total = len(sizes)
    job_id = str(job["id"])
    _set_job_progress(job_id, total_runs=total, completed_runs=0, progress_pct=0.0,
                      progress_label=f"Antenna size sweep queued (0/{total})")

    for idx, size_mm in enumerate(sizes, start=1):
        instances_for_size = [dict(inst, size_mm=float(size_mm)) for inst in anchors if isinstance(inst, dict)]
        step_payload = dict(payload)
        step_payload["antennae_explicit_instances"] = instances_for_size
        step_payload["antennae_enabled"] = True

        suffix = f"{size_mm:.2f}mm".replace(".", "p")
        cfg_path = _prepare_config(
            mode="single",
            payload=step_payload,
            exposure_minutes=exposure_minutes,
            output_tag=f"{output_prefix}_{suffix}_antsize",
            job=job,
        )
        out_name = f"{output_prefix}_{suffix}_antsize"
        out_dir = _output_dir_for_request(out_name, step_payload)
        _register_job_output(job, out_dir)
        _set_job_progress(
            job_id,
            progress_pct=100.0 * float(idx - 1) / float(total),
            progress_label=f"Antenna sweep {idx}/{total} ({size_mm:.2f}mm)",
        )
        cmd = [sys.executable, "rfam_eqs_coupled.py", "--config", str(cfg_path), "--output-dir", str(out_dir)]
        rc = _run_command(cmd, Path(job["log_path"]), job_id=job_id)
        if rc != 0:
            _set_job_progress(job_id, progress_label=f"Antenna sweep failed at step {idx}/{total}")
            raise RuntimeError(f"antenna size sweep step {idx} ({size_mm:.2f}mm) exited with code {rc}")
        _set_job_progress(
            job_id,
            completed_runs=idx,
            progress_pct=100.0 * float(idx) / float(total),
            progress_label=f"Antenna sweep {idx}/{total} complete ({size_mm:.2f}mm)",
        )


def _launch_fgm_gradient_descent(payload: dict[str, Any], job: dict[str, Any]) -> None:
    """Finite-difference gradient descent for FGM optimization.

    Starts from an existing FGM result (typically proportional iter-1 at σ_T≈7°C),
    estimates the true sensitivity dσ_T/ds_k for K spatial zones by running K small
    perturbation simulations, then takes a gradient step to improve σ_T below the
    proportional fixed point.

    Required payload keys:
        source_fgm_dir  — path to existing iter dir (relative to outputs_eqs/, has
                          fields.npz and a .npz FGM file in it).
        output_name     — base name for this run's output directory.
        shape           — part shape (circle, square, …).

    Optional keys:
        n_zones         (default 6)   — number of spatial zones.
        zone_type       (default "radial") — "radial" or "grid".
        perturbation_delta (default 0.10) — sat perturbation per zone (0.05–0.15).
        step_size       (default 0.8)  — gradient step size α.
        n_gradient_steps (default 3)  — outer iterations.
        bpp             (default 4).
    """
    from fgm_generator import (
        define_spatial_zones,
        apply_fd_gradient_step,
        save_fgm_from_sat_map,
    )

    job_id    = job["id"]
    shape     = str(payload.get("shape", "")).strip()
    if not shape or shape not in SUPPORTED_SHAPES:
        raise ValueError(f"invalid shape: {shape!r}")

    output_name = str(payload.get("output_name", "")).strip()
    if not _is_valid_output_name(output_name):
        raise ValueError("output_name must use only letters, numbers, '_' or '-'")

    # ── Source FGM dir ────────────────────────────────────────────────────────
    source_rel = str(payload.get("source_fgm_dir", "")).strip()
    if not source_rel:
        raise ValueError("source_fgm_dir is required")
    source_dir = (OUTPUTS_DIR / source_rel) if not Path(source_rel).is_absolute() else Path(source_rel)
    if not source_dir.exists():
        source_dir = ROOT / source_rel
    if not source_dir.exists():
        raise ValueError(f"source_fgm_dir not found: {source_rel!r}")
    if not (source_dir / "fields.npz").exists():
        raise ValueError(f"source_fgm_dir has no fields.npz: {source_dir}")

    # ── Parameters ────────────────────────────────────────────────────────────
    n_zones             = int(  payload.get("n_zones",             6))
    zone_type           = str(  payload.get("zone_type",           "radial")).strip()
    perturbation_delta  = float(payload.get("perturbation_delta",  0.10))
    step_size           = float(payload.get("step_size",           0.80))
    n_gradient_steps    = int(  payload.get("n_gradient_steps",    3))
    bpp                 = int(  payload.get("bpp",                 4))
    exposure_min        = float(payload.get("exposure_minutes",    8.0))

    safe_shape = shape.replace(" ", "_").lower()
    parent_dir = OUTPUTS_DIR / "runs" / safe_shape / "fgm_iterate" / output_name
    parent_dir.mkdir(parents=True, exist_ok=True)
    _register_job_output(job, parent_dir)

    # ── Load base config from source dir's parent chain ───────────────────────
    # Walk up to find used_config.yaml or convergence.json
    base_cfg: dict[str, Any] | None = None
    _search = source_dir
    for _ in range(6):
        if (_search / "used_config.yaml").exists():
            with open(_search / "used_config.yaml", encoding="utf-8") as fh:
                base_cfg = yaml.safe_load(fh)
            break
        _search = _search.parent
    if base_cfg is None:
        raise ValueError(f"could not find used_config.yaml near {source_dir}")

    # ── Load base FGM sat_map ─────────────────────────────────────────────────
    base_fgm_npz: Path | None = None
    for candidate in sorted(source_dir.glob("fgm_*.npz")):
        if "_preview" not in candidate.name:
            base_fgm_npz = candidate
            break
    if base_fgm_npz is None:
        raise ValueError(f"no fgm_*.npz found in {source_dir}")

    base_data    = np.load(base_fgm_npz, allow_pickle=True)
    sat_map_base = base_data["sat_map"].astype(np.float32)   # (ny_sim, nx_sim)

    # ── Load part_mask from source fields.npz ─────────────────────────────────
    src_fields    = np.load(source_dir / "fields.npz")
    part_mask     = src_fields["part_mask"].astype(bool)
    x_mm          = src_fields["x"] * 1000.0
    y_mm          = src_fields["y"] * 1000.0

    # ── Read base σ_T ─────────────────────────────────────────────────────────
    base_metrics  = _iter_metrics_from_dir(source_dir)
    sigma_T_base  = base_metrics.get("sigma_T", 999.0)

    print(
        f"  [fgm_gd] Starting FD gradient descent from {source_dir.name}  "
        f"σ_T_base={sigma_T_base:.3f}°C  "
        f"n_zones={n_zones}  zone_type={zone_type}  Δ={perturbation_delta}  "
        f"step={step_size}  n_iters={n_gradient_steps}",
        flush=True,
    )

    # ── Define spatial zones ──────────────────────────────────────────────────
    zone_masks = define_spatial_zones(part_mask, n_zones=n_zones, zone_type=zone_type)
    n_zones_actual = len(zone_masks)
    print(f"  [fgm_gd] Defined {n_zones_actual} zones:", flush=True)
    for k, zm in enumerate(zone_masks):
        print(f"    zone-{k}: {zm.sum()} pixels", flush=True)

    # Write base FGM into parent_dir for reference
    base_copy_npz = parent_dir / f"fgm_gd_base.npz"
    import shutil
    shutil.copy2(base_fgm_npz, base_copy_npz)

    convergence_log: list[dict[str, Any]] = [{
        "iter": "base",
        "output_dir": source_dir.relative_to(OUTPUTS_DIR).as_posix(),
        "sigma_T": sigma_T_base,
        "fgm_npz": str(base_fgm_npz.relative_to(ROOT)),
        "note": "starting point (proportional iter-1)",
        **base_metrics,
    }]
    best_sigma_T   = sigma_T_base
    best_sat_map   = sat_map_base.copy()
    current_sat    = sat_map_base.copy()

    total_sims = n_gradient_steps * (n_zones_actual + 1)
    completed  = 0

    _set_job_progress(
        job_id,
        total_runs=total_sims,
        completed_runs=0,
        progress_label=f"FD-GD | base σ_T={sigma_T_base:.2f}°C | {n_zones_actual} zones × {n_gradient_steps} steps",
    )

    for gd_step in range(n_gradient_steps):
        step_label = f"GD step {gd_step + 1}/{n_gradient_steps}"
        print(f"\n  [fgm_gd] === {step_label} ===", flush=True)

        # ── K perturbation simulations ────────────────────────────────────────
        zone_sigma_T: list[float] = []
        for k, zone_mask in enumerate(zone_masks):
            pert_dir = parent_dir / f"gd{gd_step}_zone{k}"
            pert_dir.mkdir(parents=True, exist_ok=True)
            _register_job_output(job, pert_dir)

            # Create perturbed sat_map
            sat_pert = current_sat.copy()
            sat_pert[zone_mask] = np.clip(sat_pert[zone_mask] + perturbation_delta, 0.0, 1.0)
            sat_pert[~part_mask] = 0.0
            pert_npz = pert_dir / f"fgm_pert_zone{k}.npz"
            save_result = save_fgm_from_sat_map(
                sat_pert, bpp=bpp, output_path=pert_npz,
                x_mm=x_mm, y_mm=y_mm, part_mask=part_mask, emit_png=False,
            )
            pert_npz = Path(save_result["npz_path"])

            # Build config
            pert_cfg = dict(base_cfg)
            pert_cfg["fgm_feedback"] = {
                "enabled":             True,
                "saturation_map_npz":  str(pert_npz),
                "magnitude":           1.0,
                "baseline_saturation": 0.5,
            }
            _set_exposure_minutes(pert_cfg, exposure_min)
            cfg_path = GUI_CONFIG_DIR / f"fgm_gd_{job_id}_s{gd_step}_z{k}.yaml"
            with open(cfg_path, "w", encoding="utf-8") as fh:
                yaml.dump(pert_cfg, fh, default_flow_style=False, allow_unicode=True)

            # Run simulation
            cmd = [sys.executable, "rfam_eqs_coupled.py",
                   "--config", str(cfg_path), "--output-dir", str(pert_dir)]
            completed += 1
            _set_job_progress(
                job_id, completed_runs=completed, total_runs=total_sims,
                progress_label=f"{step_label} | perturbing zone {k+1}/{n_zones_actual}",
            )
            rc = _run_command(cmd, Path(job["log_path"]), job_id=job_id)
            if rc != 0:
                raise RuntimeError(f"Perturbation sim gd{gd_step}_zone{k} failed (rc={rc})")

            m = _iter_metrics_from_dir(pert_dir)
            st = m.get("sigma_T", 999.0)
            zone_sigma_T.append(st)
            print(
                f"  [fgm_gd] zone-{k} pert: σ_T={st:.3f}°C  "
                f"(base={sigma_T_base:.3f}°C  Δ={st - sigma_T_base:+.3f}°C)",
                flush=True,
            )

        # ── Compute gradients ─────────────────────────────────────────────────
        gradients = [(st - sigma_T_base) / perturbation_delta for st in zone_sigma_T]
        print(
            f"  [fgm_gd] Gradients: {[f'{g:+.3f}' for g in gradients]}  "
            f"(dσ_T/dsat per zone)",
            flush=True,
        )

        # ── Apply gradient step ───────────────────────────────────────────────
        eval_dir = parent_dir / f"gd{gd_step}_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        _register_job_output(job, eval_dir)

        eval_npz = eval_dir / "fgm_gd_step.npz"
        apply_fd_gradient_step(
            sat_map=current_sat,
            zone_masks=zone_masks,
            zone_gradients=gradients,
            step_size=step_size,
            part_mask=part_mask,
            bpp=bpp,
            output_path=eval_npz,
            emit_png=True,
        )
        eval_npz = Path(str(eval_npz))

        # Build eval config
        eval_cfg = dict(base_cfg)
        eval_cfg["fgm_feedback"] = {
            "enabled":             True,
            "saturation_map_npz":  str(eval_npz),
            "magnitude":           1.0,
            "baseline_saturation": 0.5,
        }
        _set_exposure_minutes(eval_cfg, exposure_min)
        cfg_path = GUI_CONFIG_DIR / f"fgm_gd_{job_id}_s{gd_step}_eval.yaml"
        with open(cfg_path, "w", encoding="utf-8") as fh:
            yaml.dump(eval_cfg, fh, default_flow_style=False, allow_unicode=True)

        # Run evaluation simulation
        cmd = [sys.executable, "rfam_eqs_coupled.py",
               "--config", str(cfg_path), "--output-dir", str(eval_dir)]
        completed += 1
        _set_job_progress(
            job_id, completed_runs=completed, total_runs=total_sims,
            progress_label=f"{step_label} | evaluating gradient step",
        )
        rc = _run_command(cmd, Path(job["log_path"]), job_id=job_id)
        if rc != 0:
            raise RuntimeError(f"Eval sim gd{gd_step}_eval failed (rc={rc})")

        eval_metrics = _iter_metrics_from_dir(eval_dir)
        sigma_T_new  = eval_metrics.get("sigma_T", 999.0)
        improved     = sigma_T_new < sigma_T_base

        print(
            f"  [fgm_gd] {step_label} eval: σ_T={sigma_T_new:.3f}°C  "
            f"(was {sigma_T_base:.3f}°C  Δ={sigma_T_new - sigma_T_base:+.3f}°C  "
            f"{'IMPROVED ✓' if improved else 'DEGRADED ✗'})",
            flush=True,
        )

        convergence_log.append({
            "iter": gd_step + 1,
            "output_dir": eval_dir.relative_to(OUTPUTS_DIR).as_posix(),
            "sigma_T": sigma_T_new,
            "sigma_T_base_this_step": sigma_T_base,
            "zone_sigma_T_perturbed": zone_sigma_T,
            "zone_gradients": gradients,
            "step_size": step_size,
            "improved": improved,
            "fgm_npz": str(eval_npz.relative_to(ROOT)),
            **eval_metrics,
        })

        if improved:
            # Accept and update for next step
            best_sigma_T  = sigma_T_new
            best_sat_map  = np.load(eval_npz)["sat_map"].astype(np.float32)
            current_sat   = best_sat_map.copy()
            sigma_T_base  = sigma_T_new
        else:
            # Halve step size and retry direction next iteration
            step_size *= 0.5
            print(
                f"  [fgm_gd] No improvement → halving step_size to {step_size:.3f} for next step",
                flush=True,
            )
            # Keep current_sat unchanged (don't accept the degraded step)

        # Write convergence.json incrementally
        conv_data: dict[str, Any] = {
            "source_fgm_dir":    source_rel,
            "shape":             shape,
            "n_zones":           n_zones_actual,
            "zone_type":         zone_type,
            "perturbation_delta": perturbation_delta,
            "step_size_initial": float(payload.get("step_size", 0.80)),
            "bpp":               bpp,
            "n_gradient_steps":  n_gradient_steps,
            "iterations":        convergence_log,
            "best_sigma_T":      best_sigma_T,
            "best_step":         (max(i for i, e in enumerate(convergence_log)
                                     if e.get("sigma_T", 999) == best_sigma_T)
                                  if convergence_log else 0),
        }
        (parent_dir / "convergence.json").write_text(
            json.dumps(conv_data, indent=2), encoding="utf-8"
        )

    _set_job_progress(
        job_id,
        progress_pct=100.0,
        progress_label=f"FD-GD done | best σ_T={best_sigma_T:.2f}°C (base={float(payload.get('base_sigma_T', sigma_T_base)):.2f}°C)",
    )


def _iter_metrics_from_dir(it_dir: Path) -> dict[str, Any]:
    """Thin wrapper: compute iteration metrics from a directory.
    Needed because _iter_metrics is defined inside _launch_fgm_iterate_mode.
    """
    metrics: dict[str, Any] = {}
    T_AMBIENT_C = 20.0
    DAMAGE_T_C  = 300.0
    SEL_TARGET  = 2.0
    melt_ref_c  = 186.0
    sfile = it_dir / "summary.json"
    if sfile.exists():
        try:
            s = json.loads(sfile.read_text(encoding="utf-8"))
            metrics["frac_melt"]    = float(s.get("frac_part_ge_melt_ref", 0.0))
            metrics["mean_rho"]     = float(s.get("mean_rho_rel_part_final", 0.0))
            metrics["T_mean_c"]     = float(s.get("mean_T_part_final_c", 0.0))
            metrics["T_max_c"]      = float(s.get("max_T_part_final_c", 0.0))
            melt_ref_c              = float(s.get("melt_reference_c", melt_ref_c))
        except Exception:
            pass
    ffile = it_dir / "fields.npz"
    if ffile.exists():
        try:
            f      = np.load(ffile)
            mask   = f["part_mask"].astype(bool)
            T_in   = f["T"][mask]
            T_out  = f["T"][~mask]
            rho_in = f["rho_rel"][mask]
            metrics["sigma_T"]   = round(float(np.std(T_in)),   3)
            metrics["sigma_rho"] = round(float(np.std(rho_in)), 4)
            metrics["dT_c"]      = round(float(T_in.max() - T_in.mean()), 3)
            dT_in  = float(T_in.mean())  - T_AMBIENT_C
            dT_out = float(T_out.mean()) - T_AMBIENT_C
            metrics["thermal_selectivity"] = round(dT_in / max(dT_out, 0.1), 3)
            frac_bleed = float((T_out >= melt_ref_c).sum()) / max(len(T_out), 1)
            metrics["frac_bleed"] = round(frac_bleed, 4)
        except Exception:
            pass
    return metrics


def _job_worker(job_id: str, payload: dict[str, Any]) -> None:
    job = JOBS[job_id]
    try:
        mode = payload.get("mode", "single")
        if mode == "backfill_reports":
            _run_backfill_reports_job(payload, job)
        elif mode == "antennae_calibrate":
            _launch_antennae_calibration_mode(payload, job)
        elif mode == "antennae_size_sweep":
            _launch_antennae_size_sweep_mode(payload, job)
        elif mode == "sweep":
            _launch_sweep_mode(payload, job)
        elif mode == "shell_sweep":
            _launch_shell_sweep_mode(payload, job)
        elif mode == "fgm_resimulate":
            _launch_fgm_resimulate_mode(payload, job)
        elif mode == "fgm_iterate":
            _launch_fgm_iterate_mode(payload, job)
        elif mode == "fgm_gradient_descent":
            _launch_fgm_gradient_descent(payload, job)
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
        elif (path / "convergence.json").exists():
            detected_rt = "fgm_iterate"
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
        has_convergence = (p / "convergence.json").exists()
        if not (has_summary or has_media or has_convergence):
            continue

        rel = p.relative_to(OUTPUTS_DIR).as_posix()
        summary: dict[str, Any] = {}
        if has_summary:
            try:
                summary = json.loads((p / "summary.json").read_text())
            except Exception:
                summary = {}
        images = _collect_images_recursive(p)
        try:
            disk_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        except Exception:
            disk_bytes = 0
        hero = _preferred_images(p, limit=3)
        if rel.startswith("runs/"):
            toks = rel.split("/")
            group = toks[1] if len(toks) > 1 else "runs"
        else:
            group = rel.split("/")[0] if "/" in rel else "runs"
        manifest = _load_or_build_manifest(p)
        run_type = str(manifest.get("run_type", _get_run_type(p)))
        caps = [str(v) for v in manifest.get("backfill_capabilities", []) if str(v)]

        # For fgm_iterate runs, attach best-iteration metrics for richer card display.
        fgm_best: dict[str, Any] = {}
        if run_type == "fgm_iterate" and has_convergence:
            try:
                _cd = json.loads((p / "convergence.json").read_text(encoding="utf-8"))
                _bi = int(_cd.get("best_iter", -1))
                _iters = _cd.get("iterations", [])
                _best = next((it for it in _iters if int(it.get("iter", -99)) == _bi), None)
                if _best:
                    fgm_best = {
                        "best_iter":    _bi,
                        "n_iters":      len(_iters),
                        "sigma_T":      _best.get("sigma_T"),
                        "sigma_rho":    _best.get("sigma_rho"),
                        "mean_rho":     _best.get("mean_rho"),
                        "frac_melt":    _best.get("frac_melt"),
                        "frac_bleed":   _best.get("frac_bleed"),
                        "score":        _best.get("score"),
                        "score_regime": _best.get("score_regime"),
                        "converged":    bool(_cd.get("converged", False)),
                        "aborted":      bool(_cd.get("aborted", False)),
                        "exposure_minutes": float(_cd.get("exposure_minutes", 0)),
                    }
            except Exception:
                fgm_best = {}

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
            "fgm_best": fgm_best,
            "disk_bytes": disk_bytes,
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


# ── HEATR-3D integration ──────────────────────────────────────────────────────
# The 3-D solver (heatr3d.py) must run under a numpy<2.2 interpreter (system numpy
# 2.2 + py3.14 has a buffer-elision bug). We run heatr3d_job.py as a SUBPROCESS
# under a dedicated venv; the GUI just spawns/monitors it.
import base64 as _b64

_H3D_OUT = OUTPUTS_DIR / "_heatr3d"
_H3D_JOBS: dict[str, dict] = {}


def _h3d_python() -> str:
    """Resolve the interpreter for heatr3d (prefer the local venv)."""
    cands = [
        ROOT / ".venv-heatr3d" / "bin" / "python",
        Path.home() / "GaTech Dropbox" / "Matthew McCoy" / "mattmccoy-research" /
        "research" / "dissertation_materials" / "analysis-3dfgm" / ".venv312" / "bin" / "python",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    return sys.executable


def _h3d_write_config(payload: dict) -> Path:
    out = _H3D_OUT / uuid.uuid4().hex[:12]
    out.mkdir(parents=True, exist_ok=True)
    keys = ("shape", "diam", "zspan", "n", "exposure_s", "densify", "stop_mean_rho", "fgm", "magnitude")
    cfg = {k: payload[k] for k in keys if payload.get(k) is not None}
    if payload.get("src") == "stl" and payload.get("stl_b64"):
        stl = out / "input.stl"
        stl.write_bytes(_b64.b64decode(payload["stl_b64"]))
        cfg["stl"] = str(stl)
        cfg.pop("shape", None)
    cfg["out_dir"] = str(out)
    (out / "config.json").write_text(json.dumps(cfg))
    return out


def _h3d_preview(payload: dict) -> dict:
    out = _h3d_write_config(payload)
    proc = subprocess.run(
        [_h3d_python(), str(ROOT / "heatr3d_job.py"), str(out / "config.json"), "--preview"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    geo = out / "geometry.json"
    if not geo.exists():
        raise RuntimeError((proc.stderr or proc.stdout or "preview failed")[-800:])
    return json.loads(geo.read_text())


def _h3d_spawn(payload: dict) -> str:
    out = _h3d_write_config(payload)
    jid = out.name
    logf = open(out / "job.log", "w")
    proc = subprocess.Popen(
        [_h3d_python(), str(ROOT / "heatr3d_job.py"), str(out / "config.json")],
        cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT, text=True)
    _H3D_JOBS[jid] = {"proc": proc, "out": out, "log": logf}
    return jid


def _h3d_status(jid: str) -> dict:
    job = _H3D_JOBS.get(jid)
    if not job:
        # fall back to disk (survives server restart): a finished job has results.
        disk = _H3D_OUT / jid
        if (disk / "results.json").exists():
            resp = {"progress": 100, "done": True,
                    "results": json.loads((disk / "results.json").read_text())}
            geo = disk / "geometry.json"
            if geo.exists():
                resp["geometry"] = json.loads(geo.read_text())
            return resp
        return {"done": True, "error": "unknown job"}
    out = job["out"]
    prog = 0
    try:
        for line in (out / "job.log").read_text().splitlines():
            if line.startswith("PROGRESS"):
                prog = int(float(line.split()[1]))
    except Exception:
        pass
    rc = job["proc"].poll()
    resp: dict = {"progress": prog, "done": rc is not None}
    geo = out / "geometry.json"
    if geo.exists():
        try:
            resp["geometry"] = json.loads(geo.read_text())
        except Exception:
            pass
    if rc is not None:
        res = out / "results.json"
        if res.exists():
            resp["results"] = json.loads(res.read_text())
            resp["progress"] = 100
        else:
            try:
                resp["error"] = (out / "job.log").read_text()[-800:]
            except Exception:
                resp["error"] = f"job exited rc={rc} with no results"
    return resp


class Handler(BaseHTTPRequestHandler):
    server_version = "rfam-gui/0.2"
    timeout = None   # disable per-connection idle timeout in BaseHTTPRequestHandler

    def _json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "keep-alive")
        self.send_header("Keep-Alive", "timeout=65, max=10000")
        self.end_headers()
        self.wfile.write(raw)

    def _text(self, text: str, status: int = 200, ctype: str = "text/plain; charset=utf-8") -> None:
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "keep-alive")
        self.send_header("Keep-Alive", "timeout=65, max=10000")
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
        if path == "/heatr3d" or path == "/heatr3d.html":
            return self._serve_static("heatr3d.html")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if path == "/api/ping":
            return self._json({"ok": True, "ts": time.time()})

        if path == "/api/heatr3d/status":
            query = urlparse(self.path).query
            jid = ""
            for kv in query.split("&"):
                if kv.startswith("id="):
                    jid = kv[3:]
            try:
                return self._json(_h3d_status(jid))
            except Exception as e:
                return self._json({"done": True, "error": str(e)}, status=500)

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
                # Attach live convergence iteration data for fgm_iterate jobs so the
                # job card can render a real-time convergence sparkline.
                if j.get("mode") == "fgm_iterate" and j.get("status") in ("running", "paused", "completed"):
                    _output_dirs = j.get("output_dirs", [])
                    for _od in _output_dirs:
                        _conv = (OUTPUTS_DIR / _od / "convergence.json").resolve()
                        if not _conv.exists():
                            # Try parent dir (the fgm_iterate parent, not an iter subdir)
                            _conv = (OUTPUTS_DIR / Path(_od).parent / "convergence.json").resolve()
                        if _conv.exists():
                            try:
                                _cd = json.loads(_conv.read_text(encoding="utf-8"))
                                j["convergence_iters"] = [
                                    {k: v for k, v in it.items()
                                     if k in ("iter", "sigma_T", "mean_rho", "score",
                                              "frac_bleed", "score_regime", "magnitude_used")}
                                    for it in _cd.get("iterations", [])
                                ]
                                break
                            except Exception:
                                pass
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
        if path.startswith("/api/convergence/"):
            # Return convergence.json enriched with per-iter FGM PNG as base64
            rel = unquote(path[len("/api/convergence/"):])
            conv_dir = (OUTPUTS_DIR / rel).resolve()
            try:
                conv_dir.relative_to(OUTPUTS_DIR)
            except ValueError:
                return self._json({"error": "path outside outputs"}, status=400)
            conv_file = conv_dir / "convergence.json"
            if not conv_file.exists():
                return self._json({"error": "convergence.json not found"}, status=404)
            try:
                data = json.loads(conv_file.read_text(encoding="utf-8"))
                # Enrich each iteration entry with FGM PNG base64 if available.
                # Primary: use the stored fgm_png path from convergence.json.
                # Fallback: scan the iter output_dir for any fgm_*.png file.
                import base64 as _b64
                for entry in data.get("iterations", []):
                    png_path_obj: Path | None = None

                    png_rel = entry.get("fgm_png")
                    if png_rel:
                        candidate = (ROOT / png_rel).resolve()
                        if candidate.exists():
                            png_path_obj = candidate

                    # Fallback: look in the iter dir for any fgm_*.png
                    if png_path_obj is None:
                        out_dir_rel = entry.get("output_dir", "")
                        if out_dir_rel:
                            iter_dir = (OUTPUTS_DIR / out_dir_rel).resolve()
                            if iter_dir.is_dir():
                                candidates = sorted(iter_dir.glob("fgm_*.png"),
                                                    key=lambda p: p.stat().st_mtime,
                                                    reverse=True)
                                if candidates:
                                    png_path_obj = candidates[0]

                    if png_path_obj is not None:
                        try:
                            entry["fgm_png_b64"] = _b64.b64encode(
                                png_path_obj.read_bytes()
                            ).decode()
                        except Exception:
                            pass
                        # Also serve the Meteor-import PNG (inverted: black=max ink).
                        # It lives alongside the preview PNG with the _meteor_import suffix.
                        # Naming: fgm_..._preview.png → fgm_..._meteor_import.png
                        meteor_candidate = Path(str(png_path_obj).replace(
                            "_preview.png", "_meteor_import.png"
                        ))
                        if not meteor_candidate.exists():
                            # Fallback: scan same directory for *_meteor_import.png
                            _mdir = png_path_obj.parent
                            _mc_list = sorted(_mdir.glob("fgm_*_meteor_import.png"),
                                              key=lambda p: p.stat().st_mtime, reverse=True)
                            if _mc_list:
                                meteor_candidate = _mc_list[0]
                        if meteor_candidate.exists():
                            try:
                                entry["fgm_meteor_png_b64"] = _b64.b64encode(
                                    meteor_candidate.read_bytes()
                                ).decode()
                            except Exception:
                                pass

                    # Also serve thermal_fields_final.png from the iter output dir
                    out_dir_rel2 = entry.get("output_dir", "")
                    if out_dir_rel2:
                        thermal_p = (OUTPUTS_DIR / out_dir_rel2 / "thermal_fields_final.png").resolve()
                        if thermal_p.exists():
                            try:
                                entry["thermal_png_b64"] = _b64.b64encode(
                                    thermal_p.read_bytes()
                                ).decode()
                            except Exception:
                                pass

                # ── Re-apply best-iter selection using current logic ─────────
                # Historical convergence.json files may have stale best_iter
                # values computed by older code (e.g. before bleed/T_max
                # disqualification was added).  Re-select on every read so the
                # dashboard always reflects the current algorithm.
                try:
                    _iters_for_sel = data.get("iterations", [])
                    if _iters_for_sel:
                        _use_opt = bool(data.get("optimizer_chosen_time_min"))
                        _dmg_t   = float(data.get("damage_t_c", 245.0))
                        _best    = _select_best_iter(
                            _iters_for_sel, _use_opt, _dmg_t
                        )
                        if _best:
                            data["best_iter"]         = _best.get("iter", data.get("best_iter"))
                            data["best_score"]        = _best.get("score", data.get("best_score"))
                            data["best_sigma_T"]      = _best.get("sigma_T", data.get("best_sigma_T"))
                            data["best_score_regime"] = _best.get("score_regime", data.get("best_score_regime"))
                except Exception:
                    pass  # never break serving on a selection error

                return self._json(data)
            except Exception as exc:
                return self._json({"error": str(exc)}, status=500)
        if path.startswith("/api/profile/"):
            # Return horizontal and vertical centerline profiles from fields.npz.
            # GET /api/profile/<rel_run_path>
            rel = unquote(path[len("/api/profile/"):])
            run_dir = (OUTPUTS_DIR / rel).resolve()
            try:
                run_dir.relative_to(OUTPUTS_DIR)
            except ValueError:
                return self._json({"error": "path outside outputs"}, status=400)
            npz_path = run_dir / "fields.npz"
            if not npz_path.exists():
                # For fgm_iterate parent dirs: look in best iter, then any iter dir
                _conv_f = run_dir / "convergence.json"
                if _conv_f.exists():
                    try:
                        _cd = json.loads(_conv_f.read_text(encoding="utf-8"))
                        _bi = int(_cd.get("best_iter", -1))
                        _iters = _cd.get("iterations", [])
                        # Try best iter first, then descending
                        _ordered = sorted(_iters, key=lambda x: -(1 if int(x.get("iter",-1))==_bi else int(x.get("iter",-1))))
                        for _it in _ordered:
                            _od = _it.get("output_dir", "")
                            if _od:
                                _cand = (OUTPUTS_DIR / _od / "fields.npz").resolve()
                                if _cand.exists():
                                    npz_path = _cand
                                    break
                    except Exception:
                        pass
                if not npz_path.exists():
                    return self._json({"error": "fields.npz not found"}, status=404)
            try:
                import numpy as _np
                fdata = _np.load(str(npz_path), allow_pickle=False)
                mask  = fdata["part_mask"].astype(bool) if "part_mask" in fdata.files else None
                x_m   = fdata["x"].tolist() if "x" in fdata.files else None
                y_m   = fdata["y"].tolist() if "y" in fdata.files else None
                ny, nx_sz = (fdata["part_mask"].shape if "part_mask" in fdata.files
                             else (len(y_m or []), len(x_m or [])))

                def _profile(field_name):
                    if field_name not in fdata.files:
                        return None
                    arr = fdata[field_name].astype(float)
                    # Horizontal centerline: row closest to Y center of part bbox
                    rows = _np.any(mask, axis=1) if mask is not None else _np.ones(ny, dtype=bool)
                    row_idxs = _np.where(rows)[0]
                    if len(row_idxs) == 0:
                        return None
                    cy = int(row_idxs[len(row_idxs)//2])
                    h_vals = arr[cy, :].tolist()
                    h_mask = mask[cy, :].tolist() if mask is not None else [True]*nx_sz
                    # Vertical centerline: col closest to X center of part bbox
                    cols = _np.any(mask, axis=0) if mask is not None else _np.ones(nx_sz, dtype=bool)
                    col_idxs = _np.where(cols)[0]
                    if len(col_idxs) == 0:
                        return None
                    cx = int(col_idxs[len(col_idxs)//2])
                    v_vals = arr[:, cx].tolist()
                    v_mask = mask[:, cx].tolist() if mask is not None else [True]*ny
                    return {"h": h_vals, "h_mask": h_mask, "h_row": cy,
                            "v": v_vals, "v_mask": v_mask, "v_col": cx}

                result = {}
                for fname in ["T", "T_phi90", "rho_rel", "Qrf"]:
                    p = _profile(fname)
                    if p is not None:
                        result[fname] = p
                return self._json({
                    "run": rel,
                    "x_m": x_m, "y_m": y_m,
                    "profiles": result,
                    "available": list(result.keys()),
                })
            except Exception as exc:
                return self._json({"error": str(exc)}, status=500)

        if path.startswith("/api/fields/"):
            # Return field data from fields.npz for the interactive field viewer.
            # GET /api/fields/<rel_run_path>?field=T_phi90&maxpx=256
            from urllib.parse import parse_qs
            rel = unquote(path[len("/api/fields/"):])
            qs  = parse_qs(urlparse(self.path).query)
            field_name  = qs.get("field",  ["T_phi90"])[0]
            max_px      = int(qs.get("maxpx", ["256"])[0])
            run_dir = (OUTPUTS_DIR / rel).resolve()
            try:
                run_dir.relative_to(OUTPUTS_DIR)
            except ValueError:
                return self._json({"error": "path outside outputs"}, status=400)
            npz_path = run_dir / "fields.npz"
            if not npz_path.exists():
                # For fgm_iterate parent dirs: look in best iter, then any iter dir
                _conv_f = run_dir / "convergence.json"
                if _conv_f.exists():
                    try:
                        _cd = json.loads(_conv_f.read_text(encoding="utf-8"))
                        _bi = int(_cd.get("best_iter", -1))
                        _iters = _cd.get("iterations", [])
                        # Try best iter first, then descending
                        _ordered = sorted(_iters, key=lambda x: -(1 if int(x.get("iter",-1))==_bi else int(x.get("iter",-1))))
                        for _it in _ordered:
                            _od = _it.get("output_dir", "")
                            if _od:
                                _cand = (OUTPUTS_DIR / _od / "fields.npz").resolve()
                                if _cand.exists():
                                    npz_path = _cand
                                    break
                    except Exception:
                        pass
                if not npz_path.exists():
                    return self._json({"error": "fields.npz not found"}, status=404)
            try:
                import numpy as _np
                fdata = _np.load(str(npz_path), allow_pickle=False)
                available = [k for k in fdata.files if k != "part_mask"]
                if field_name not in fdata.files:
                    field_name = available[0] if available else "T"
                arr = fdata[field_name].astype(float)
                mask = fdata["part_mask"].astype(bool) if "part_mask" in fdata.files else _np.ones(arr.shape, dtype=bool)
                x_m = fdata["x"].tolist() if "x" in fdata.files else list(range(arr.shape[1]))
                y_m = fdata["y"].tolist() if "y" in fdata.files else list(range(arr.shape[0]))
                # Downsample to max_px for fast transfer
                ny, nx = arr.shape
                step_y = max(1, ny // max_px)
                step_x = max(1, nx // max_px)
                arr_ds   = arr[::step_y, ::step_x]
                mask_ds  = mask[::step_y, ::step_x]
                x_ds = x_m[::step_x]
                y_ds = y_m[::step_y]
                inside = arr_ds[mask_ds]
                lo  = float(inside.min())  if inside.size > 0 else 0.0
                hi  = float(inside.max())  if inside.size > 0 else 1.0
                mean_val = float(inside.mean()) if inside.size > 0 else 0.0
                return self._json({
                    "field": field_name,
                    "available_fields": available,
                    "shape": list(arr_ds.shape),
                    "x_m": x_ds,
                    "y_m": y_ds,
                    "data": arr_ds.tolist(),
                    "mask": mask_ds.tolist(),
                    "vmin": lo, "vmax": hi, "vmean": mean_val,
                })
            except Exception as exc:
                return self._json({"error": str(exc)}, status=500)

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

        # ── QUIT endpoint — handled before allowlist check ────────────────────
        if path == "/api/quit":
            def _shutdown():
                time.sleep(0.2)
                _SERVER_REF.shutdown()
            t = threading.Thread(target=_shutdown, daemon=True)
            t.start()
            return self._json({"ok": True, "msg": "HEATR server shutting down"})

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
            "/api/tools/antennae-geometry",
            "/api/tools/antennae-quick-search",
            "/api/tools/generate-fgm",
            "/api/tools/fgm-resimulate",
            "/api/tools/fgm-to-rip",
            "/api/tools/fgm-iterate",
            "/api/tools/fgm-continue",
            "/api/tools/fgm-gradient-descent",
            "/api/heatr3d/preview",
            "/api/heatr3d/run",
        }:
            return self._json({"error": "not found"}, status=404)

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._json({"error": "invalid json"}, status=400)

        if path == "/api/heatr3d/preview":
            try:
                return self._json(_h3d_preview(payload))
            except Exception as e:
                return self._text(str(e), status=500)
        if path == "/api/heatr3d/run":
            try:
                return self._json({"id": _h3d_spawn(payload)})
            except Exception as e:
                return self._text(str(e), status=500)

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

        if path == "/api/tools/generate-fgm":
            # Payload keys (all optional except output_dir):
            #   output_dir         — relative path under ROOT (e.g. "outputs_eqs/runs/.../my_run")
            #   bpp                — 2 or 4 (default 2)
            #   proxy_field        — "Qrf" | "T" | "rho_rel" (default "Qrf")
            #   invert             — bool (default true)
            #   magnitude          — float 0–2 (default 1.0)
            #   baseline_saturation— float 0–1 (default 0.5)
            #   dpi                — int (default 720)
            #   smoothing_sigma    — float (default 1.5)
            output_dir_rel = str(payload.get("output_dir", "")).strip()
            if not output_dir_rel:
                return self._json({"error": "missing output_dir"}, status=400)
            run_dir = (ROOT / output_dir_rel).resolve()
            try:
                run_dir.relative_to(ROOT)
            except ValueError:
                return self._json({"error": "output_dir outside workspace"}, status=400)
            if not run_dir.is_dir():
                return self._json({"error": f"directory not found: {output_dir_rel}"}, status=404)
            # If this is an fgm_iterate parent dir (has convergence.json but no fields.npz),
            # automatically redirect to the best iteration's subdir so the user can generate
            # a fresh FGM from its fields without having to know the internal layout.
            if not (run_dir / "fields.npz").exists():
                conv_path = run_dir / "convergence.json"
                if conv_path.exists():
                    try:
                        _conv = json.loads(conv_path.read_text(encoding="utf-8"))
                        _best_iter = int(_conv.get("best_iter", 0))
                        _iters = _conv.get("iterations", [])
                        _best_entry = next(
                            (e for e in _iters if e.get("iter") == _best_iter), None
                        )
                        if _best_entry and _best_entry.get("output_dir"):
                            _iter_dir = (OUTPUTS_DIR / _best_entry["output_dir"]).resolve()
                            if (_iter_dir / "fields.npz").exists():
                                run_dir = _iter_dir
                    except Exception:
                        pass
            if not (run_dir / "fields.npz").exists():
                return self._json({"error": "fields.npz not found in output_dir"}, status=404)
            try:
                from fgm_generator import generate_fgm  # local import for fast server startup
                import base64 as _b64
                bpp               = int(payload.get("bpp", 2))
                proxy_field       = str(payload.get("proxy_field", "Qrf")).strip()
                invert            = bool(payload.get("invert", True))
                magnitude         = float(payload.get("magnitude", 1.0))
                baseline          = float(payload.get("baseline_saturation", 0.5))
                dpi               = int(payload.get("dpi", 720))
                sigma             = float(payload.get("smoothing_sigma", 1.5))

                result = generate_fgm(
                    run_output_dir      = run_dir,
                    bpp                 = bpp,
                    proxy_field         = proxy_field,
                    invert              = invert,
                    magnitude           = magnitude,
                    baseline_saturation = baseline,
                    dpi                 = dpi,
                    smoothing_sigma     = sigma,
                    emit_formats        = ("npz", "json", "png"),
                )

                # Convert output paths to workspace-relative for the client
                def _rel(p: str | None) -> str | None:
                    if not p:
                        return None
                    try:
                        return Path(p).relative_to(ROOT).as_posix()
                    except ValueError:
                        return p

                # Encode preview PNG as base64 so the client can display it inline
                png_b64: str | None = None
                if result.get("png_path") and Path(result["png_path"]).exists():
                    png_b64 = _b64.b64encode(Path(result["png_path"]).read_bytes()).decode()

                # Level distribution stats for display
                lm = result["level_map"]
                import numpy as _np
                inside = lm[lm > 0]
                level_stats = {
                    "min":    int(inside.min())  if inside.size else 0,
                    "max":    int(inside.max())  if inside.size else 0,
                    "mean":   round(float(inside.mean()), 3) if inside.size else 0.0,
                    "unique": sorted(int(v) for v in _np.unique(inside).tolist()) if inside.size else [],
                    "shape":  list(lm.shape),
                }

                return self._json({
                    "ok":           True,
                    "npz_path":     _rel(result.get("npz_path")),
                    "json_path":    _rel(result.get("json_path")),
                    "png_path":     _rel(result.get("png_path")),
                    "png_b64":      png_b64,
                    "bpp":          result["bpp"],
                    "n_levels":     result["n_levels"],
                    "proxy_field":  result["proxy_field"],
                    "magnitude":    result["magnitude"],
                    "level_stats":  level_stats,
                })
            except (KeyError, ValueError, FileNotFoundError) as exc:
                return self._json({"error": str(exc)}, status=400)
            except Exception as exc:
                import traceback
                return self._json({"error": str(exc), "traceback": traceback.format_exc()}, status=500)

        if path == "/api/tools/import-fgm-png":
            # Import an externally-created FGM PNG and run a single simulation with it.
            # Payload keys:
            #   source_run_dir  — relative path under outputs_eqs/  (provides used_config.yaml)
            #   png_b64         — base64-encoded PNG of the FGM (grayscale, printer-DPI)
            #   bpp             — 2 or 4 (default 2; determines how pixel values map to levels)
            #   output_name     — optional; auto-generated if blank
            #   invert_vis      — bool (default true): if true, treat PNG as visualisation
            #                     where white=no-dopant, black=full-dopant (matches fgm_generator
            #                     preview output); if false, treat white=full-dopant, black=none
            try:
                import base64 as _b64, io as _io
                from PIL import Image as _Image

                source_rel  = str(payload.get("source_run_dir", "")).strip()
                png_b64_str = str(payload.get("png_b64", "")).strip()
                bpp_imp     = int(payload.get("bpp", 2))
                output_name = str(payload.get("output_name", "")).strip()
                invert_vis  = bool(payload.get("invert_vis", True))

                if not source_rel:
                    return self._json({"error": "missing source_run_dir"}, status=400)
                if not png_b64_str:
                    return self._json({"error": "missing png_b64"}, status=400)
                if bpp_imp not in (2, 4):
                    return self._json({"error": "bpp must be 2 or 4"}, status=400)

                source_dir = (OUTPUTS_DIR / source_rel).resolve()
                try:
                    source_dir.relative_to(OUTPUTS_DIR)
                except ValueError:
                    return self._json({"error": "source_run_dir must be inside outputs_eqs/"}, status=400)

                # Decode PNG → level_map
                png_bytes = _b64.b64decode(png_b64_str)
                img = _Image.open(_io.BytesIO(png_bytes)).convert("L")
                pix = np.array(img, dtype=np.uint8)   # (H, W) grayscale

                n_levels_imp = 1 << bpp_imp
                max_val_imp  = n_levels_imp - 1

                # fgm_generator saves: vis = 255 - (level/maxval * 255), so white=no-dopant
                # Invert that to recover level_map:
                if invert_vis:
                    level_map_imp = np.round((255 - pix.astype(np.float32)) / 255.0 * max_val_imp)
                else:
                    level_map_imp = np.round(pix.astype(np.float32) / 255.0 * max_val_imp)
                level_map_imp = np.clip(level_map_imp, 0, max_val_imp).astype(np.uint8)

                # Generate a safe output name
                if not output_name:
                    output_name = Path(source_rel).name + "_imported_fgm"
                if not _is_valid_output_name(output_name):
                    return self._json({"error": "output_name contains invalid characters"}, status=400)

                # Save as NPZ in the source run's directory (or a scratch subdir)
                npz_dir = source_dir
                npz_dir.mkdir(parents=True, exist_ok=True)
                npz_filename = f"imported_fgm_{bpp_imp}bpp_{output_name}.npz"
                npz_path = npz_dir / npz_filename
                np.savez_compressed(
                    npz_path,
                    level_map = level_map_imp,
                    bpp       = np.array(bpp_imp),
                    magnitude = np.array(1.0),
                    proxy_field = np.array("imported"),
                )

                # Relative path for fgm_resimulate payload (relative to ROOT)
                npz_rel = str(npz_path.relative_to(ROOT))

                # Launch fgm_resimulate job
                job_payload = {
                    "mode":           "fgm_resimulate",
                    "source_run_dir": source_rel,
                    "fgm_npz_path":   npz_rel,
                    "output_name":    output_name,
                    "magnitude":      float(payload.get("magnitude", 1.0)),
                    "baseline_saturation": float(payload.get("baseline_saturation", 0.5)),
                    "iterate":        bool(payload.get("iterate", False)),
                }
                job = _make_job(mode="fgm_resimulate", output_name=output_name)
                with JOBS_LOCK:
                    JOBS[job["id"]]["_payload"] = job_payload
                _enqueue_job(job["id"])
                _maybe_start_next_job()
                with JOBS_LOCK:
                    status = str(JOBS[job["id"]]["status"])
                    qpos   = JOBS[job["id"]].get("queue_position", None)
                return self._json({
                    "ok":          True,
                    "job_id":      job["id"],
                    "status":      status,
                    "queue_position": qpos,
                    "output_name": output_name,
                    "npz_path":    npz_rel,
                    "level_map_shape": list(level_map_imp.shape),
                    "bpp":         bpp_imp,
                }, status=202)
            except Exception as exc:
                import traceback
                return self._json({"error": str(exc), "traceback": traceback.format_exc()}, status=500)

        if path == "/api/tools/fgm-resimulate":
            # Payload keys:
            #   source_run_dir   — "runs/square/single/my_run"  (relative to outputs_eqs/)
            #   fgm_npz_path     — "outputs_eqs/runs/…/fgm_…_2bpp_mag1.00.npz" (relative to ROOT)
            #   output_name      — e.g. "my_run_fgm"  (valid identifier chars only)
            #   magnitude        — float 0–2 (default 1.0)
            #   baseline_saturation — float 0–1 (default 0.5)
            #   iterate          — bool (default false)
            source_rel  = str(payload.get("source_run_dir", "")).strip()
            fgm_rel     = str(payload.get("fgm_npz_path",   "")).strip()
            output_name = str(payload.get("output_name",    "")).strip()
            if not source_rel:
                return self._json({"error": "missing source_run_dir"}, status=400)
            if not fgm_rel:
                return self._json({"error": "missing fgm_npz_path"}, status=400)
            if not output_name:
                # Auto-generate: append _fgm to the source leaf name
                output_name = Path(source_rel).name + "_fgm"
            if not _is_valid_output_name(output_name):
                return self._json({"error": "output_name contains invalid characters"}, status=400)
            job_payload = dict(payload)
            job_payload["mode"]        = "fgm_resimulate"
            job_payload["output_name"] = output_name
            job = _make_job(mode="fgm_resimulate", output_name=output_name)
            with JOBS_LOCK:
                JOBS[job["id"]]["_payload"] = job_payload
            _enqueue_job(job["id"])
            _maybe_start_next_job()
            with JOBS_LOCK:
                status = str(JOBS[job["id"]]["status"])
                qpos   = JOBS[job["id"]].get("queue_position", None)
            return self._json({"ok": True, "job_id": job["id"], "status": status,
                               "queue_position": qpos, "output_name": output_name}, status=202)

        if path == "/api/tools/fgm-iterate":
            # Outer-loop iterative FGM optimisation starting from a fresh geometry.
            # Payload keys: shape, output_name, exposure_minutes, use_optimizer,
            #   n_iterations, magnitude, bpp, proxy_field, convergence_delta_T,
            #   iterate_inner, iterate_interval_steps, iterate_damping
            shape       = str(payload.get("shape", "")).strip()
            output_name = str(payload.get("output_name", "")).strip()
            if not shape or shape not in SUPPORTED_SHAPES:
                return self._json({"error": f"invalid shape: {shape!r}"}, status=400)
            if not output_name:
                output_name = f"{shape}_fgmopt"
            if not _is_valid_output_name(output_name):
                return self._json({"error": "output_name contains invalid characters"}, status=400)
            job_payload = dict(payload)
            job_payload["mode"]        = "fgm_iterate"
            job_payload["output_name"] = output_name
            job = _make_job(mode="fgm_iterate", output_name=output_name)
            with JOBS_LOCK:
                JOBS[job["id"]]["_payload"] = job_payload
            _enqueue_job(job["id"])
            _maybe_start_next_job()
            with JOBS_LOCK:
                status = str(JOBS[job["id"]]["status"])
                qpos   = JOBS[job["id"]].get("queue_position", None)
            return self._json({
                "ok": True, "job_id": job["id"], "status": status,
                "queue_position": qpos, "output_name": output_name,
            }, status=202)

        if path == "/api/tools/fgm-gradient-descent":
            # Finite-difference gradient descent starting from an existing FGM result.
            # Payload keys: source_fgm_dir, output_name, shape, n_zones, zone_type,
            #   perturbation_delta, step_size, n_gradient_steps, bpp, exposure_minutes
            shape       = str(payload.get("shape", "")).strip()
            output_name = str(payload.get("output_name", "")).strip()
            if not shape or shape not in SUPPORTED_SHAPES:
                return self._json({"error": f"invalid shape: {shape!r}"}, status=400)
            if not output_name:
                output_name = f"{shape}_fgm_gd"
            if not _is_valid_output_name(output_name):
                return self._json({"error": "output_name contains invalid characters"}, status=400)
            source_fgm_dir = str(payload.get("source_fgm_dir", "")).strip()
            if not source_fgm_dir:
                return self._json({"error": "source_fgm_dir is required"}, status=400)
            job_payload = dict(payload)
            job_payload["mode"]        = "fgm_gradient_descent"
            job_payload["output_name"] = output_name
            job = _make_job(mode="fgm_gradient_descent", output_name=output_name)
            with JOBS_LOCK:
                JOBS[job["id"]]["_payload"] = job_payload
            _enqueue_job(job["id"])
            _maybe_start_next_job()
            with JOBS_LOCK:
                status = str(JOBS[job["id"]]["status"])
                qpos   = JOBS[job["id"]].get("queue_position", None)
            return self._json({
                "ok": True, "job_id": job["id"], "status": status,
                "queue_position": qpos, "output_name": output_name,
            }, status=202)

        if path == "/api/tools/fgm-continue":
            # Resume a non-converged fgm_iterate run from its best iteration.
            # The best iter's used_config.yaml + fields.npz seed a new fgm_iterate run.
            # All FGM params are inherited from the existing convergence.json so the
            # continuation is consistent with the prior run.
            run_name    = str(payload.get("run_name",    "")).strip()
            n_more      = int(payload.get("n_iterations", 4))
            output_name = str(payload.get("output_name", "")).strip()

            if not run_name:
                return self._json({"error": "run_name is required"}, status=400)

            conv_path = (OUTPUTS_DIR / run_name / "convergence.json").resolve()
            if not conv_path.exists():
                return self._json({"error": f"convergence.json not found for {run_name}"}, status=404)

            try:
                conv_data = json.loads(conv_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return self._json({"error": f"could not read convergence.json: {exc}"}, status=500)

            iters = conv_data.get("iterations", [])
            if not iters:
                return self._json({"error": "no iterations found in convergence.json"}, status=400)

            best_entry = min(iters, key=lambda e: e.get("score", 999))
            best_iter_rel = best_entry.get("output_dir", "")
            if not best_iter_rel:
                return self._json({"error": "best iteration has no output_dir"}, status=400)

            best_iter_dir = (OUTPUTS_DIR / best_iter_rel).resolve()
            if not (best_iter_dir / "used_config.yaml").exists():
                return self._json({"error": f"used_config.yaml not found in best iter dir {best_iter_rel}"}, status=400)
            if not (best_iter_dir / "fields.npz").exists():
                return self._json({"error": f"fields.npz not found in best iter dir {best_iter_rel}"}, status=400)

            # Auto-generate output name: original run name + _cont (+ _2, _3 if already taken)
            if not output_name:
                base = Path(run_name).name + "_cont"
                output_name = base
                _suffix = 2
                parent_candidate = OUTPUTS_DIR / "runs" / conv_data.get("shape", "square") / "fgm_iterate"
                while (parent_candidate / output_name).exists():
                    output_name = f"{base}_{_suffix}"
                    _suffix += 1
            if not _is_valid_output_name(output_name):
                return self._json({"error": "output_name contains invalid characters"}, status=400)

            # Build payload that mirrors a fresh fgm_iterate but seeded from best iter dir
            job_payload = {
                "mode":                  "fgm_iterate",
                "shape":                 str(conv_data.get("shape", "")),
                "output_name":           output_name,
                # Use best iter dir as the source so iter-0 runs from its config/fields
                "source_run_dir":        best_iter_rel,
                "resume_from_best_iter": best_entry.get("iter", 0),
                "n_iterations":          n_more,
                # Inherit all FGM params from the prior run
                "proxy_field":           str(conv_data.get("proxy_field",      "T_phi90")),
                "magnitude":             float(conv_data.get("magnitude_base",  1.0)),
                "magnitude_decay":       float(conv_data.get("magnitude_decay", 0.7)),
                "dead_band":             float(conv_data.get("dead_band",        0.05)),
                "fgm_momentum":          float(conv_data.get("fgm_momentum",    0.3)),
                "bpp":                   int(conv_data.get("bpp",               2)),
                "baseline_saturation":   0.5,
                "invert":                True,
                "exposure_minutes":      float(conv_data.get("exposure_minutes", 7.0)),
                "use_optimizer":         False,   # exposure already found; don't re-probe
                "iterate_inner":         False,
                "convergence_delta_T":   float(payload.get("convergence_delta_T", 5.0)),
                "convergence_sigma_T":   float(payload.get("convergence_sigma_T", 3.0)),
            }
            job = _make_job(mode="fgm_iterate", output_name=output_name)
            with JOBS_LOCK:
                JOBS[job["id"]]["_payload"] = job_payload
            _enqueue_job(job["id"])
            _maybe_start_next_job()
            with JOBS_LOCK:
                status = str(JOBS[job["id"]]["status"])
                qpos   = JOBS[job["id"]].get("queue_position", None)
            return self._json({
                "ok":                True,
                "job_id":            job["id"],
                "status":            status,
                "queue_position":    qpos,
                "output_name":       output_name,
                "resumed_from_iter": best_entry.get("iter", 0),
                "best_score":        best_entry.get("score"),
            }, status=202)

        if path == "/api/tools/fgm-to-rip":
            # Synchronous bridge — converts an FGM NPZ into MetPrint TIFF stack immediately.
            # Payload keys:
            #   fgm_npz_path  — relative path (under ROOT) to the FGM .npz file (required)
            #   output_dir    — absolute or relative path for TIFF output (default: sibling "rip_output" next to NPZ)
            #   n_layers      — int (default 1)
            #   job_name      — string prefix for TIFF filenames (default: npz stem)
            #   dpi           — int (default 720)
            #   bpp           — int or null (inferred from NPZ)
            #   rotate_deg    — 0/90/180/270 (default 0)
            #   compression   — "lzw" | "none" (default "lzw")
            fgm_rel = str(payload.get("fgm_npz_path", "")).strip()
            if not fgm_rel:
                return self._json({"error": "missing fgm_npz_path"}, status=400)
            fgm_path = (ROOT / fgm_rel).resolve()
            try:
                fgm_path.relative_to(ROOT)
            except ValueError:
                return self._json({"error": "fgm_npz_path outside workspace"}, status=400)
            if not fgm_path.exists():
                return self._json({"error": f"FGM file not found: {fgm_rel}"}, status=404)

            # Determine output directory
            out_dir_raw = str(payload.get("output_dir", "")).strip()
            if out_dir_raw:
                out_dir = Path(out_dir_raw)
                if not out_dir.is_absolute():
                    out_dir = (ROOT / out_dir_raw).resolve()
            else:
                out_dir = fgm_path.parent / "rip_output"

            try:
                # Add meteor/tools/ to sys.path for this server session so that
                # fgm_to_rip.py can find meteor_rip.py when its lazy import runs.
                # We keep it on the path permanently (no removal) because the server
                # is long-lived and fgm_to_rip uses a deferred `from meteor_rip import`
                # that fires at call time, not import time.
                import sys as _sys
                meteor_tools = ROOT.parent.parent / "software" / "meteor" / "tools"
                if not meteor_tools.is_dir():
                    meteor_tools = ROOT.parent / "software" / "meteor" / "tools"
                if not meteor_tools.is_dir():
                    raise ImportError(f"meteor/tools not found (tried {meteor_tools})")
                _mt_str = str(meteor_tools)
                if _mt_str not in _sys.path:
                    _sys.path.insert(0, _mt_str)

                from fgm_to_rip import fgm_to_tiff_stack

                n_layers    = int(payload.get("n_layers",    1))
                job_name    = str(payload.get("job_name",    fgm_path.stem)).strip() or fgm_path.stem
                dpi         = int(payload.get("dpi",         720))
                bpp         = payload.get("bpp", None)
                if bpp is not None:
                    bpp = int(bpp)
                rotate_deg  = int(payload.get("rotate_deg",  0))
                compression = str(payload.get("compression", "lzw")).strip()

                tiff_paths = fgm_to_tiff_stack(
                    fgm_path    = fgm_path,
                    output_dir  = out_dir,
                    n_layers    = n_layers,
                    job_name    = job_name,
                    dpi         = dpi,
                    bpp         = bpp,
                    compression = compression,
                    rotate_deg  = rotate_deg,
                )

                # Return paths relative to ROOT where possible
                def _rel(p: str) -> str:
                    try:
                        return Path(p).relative_to(ROOT).as_posix()
                    except ValueError:
                        return p

                # Read physical dimensions from NPZ for UI confirmation
                _dim_info = {}
                try:
                    import numpy as _np_dim
                    _npz_check = _np_dim.load(fgm_path, allow_pickle=True)
                    if "width_mm"  in _npz_check: _dim_info["width_mm"]  = float(_npz_check["width_mm"])
                    if "height_mm" in _npz_check: _dim_info["height_mm"] = float(_npz_check["height_mm"])
                    if "dpi"       in _npz_check: _dim_info["dpi_npz"]   = int(_npz_check["dpi"])
                    _npz_check.close()
                except Exception:
                    pass

                return self._json({
                    "ok":         True,
                    "tiff_paths": [_rel(p) for p in tiff_paths],
                    "output_dir": _rel(str(out_dir)),
                    "n_layers":   len(tiff_paths),
                    **_dim_info,   # width_mm, height_mm, dpi_npz if available
                })
            except ImportError as exc:
                return self._json({
                    "error": f"fgm_to_rip module not available: {exc}. "
                             "Ensure meteor/tools/ is present alongside the geo-prewarp directory.",
                }, status=500)
            except Exception as exc:
                import traceback
                return self._json({"error": str(exc), "traceback": traceback.format_exc()}, status=500)

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

        if path == "/api/tools/antennae-geometry":
            try:
                req_mode = str(payload.get("mode", "single")).strip()
                if req_mode not in {"single", "optimizer", "turntable", "orientation_optimizer", "placement_optimizer"}:
                    req_mode = "single"
                exposure_minutes = float(payload.get("exposure_minutes", 6.0))
                if exposure_minutes <= 0:
                    exposure_minutes = 6.0
                effective = _effective_config_for_request(payload, req_mode, exposure_minutes)
                cfg_text = str(effective.get("effective_config_text", "")).strip()
                cfg = yaml.safe_load(cfg_text) if cfg_text else {}
                if not isinstance(cfg, dict):
                    cfg = {}
                geom = cfg.get("geometry", {})
                chamber_x = float(geom.get("chamber_x", 0.06))
                chamber_y = float(geom.get("chamber_y", 0.06))
                from shapes import make_shape, rotate as _rotate_poly
                import math as _math
                from rfam_eqs_coupled import _parts_from_geometry
                parts_out = []
                for p in _parts_from_geometry(geom):
                    if not isinstance(p, dict):
                        continue
                    shape_name = str(p.get("shape", "square"))
                    w = float(p.get("width", 0.02))
                    h = float(p.get("height", w))
                    cx = float(p.get("center_x", 0.0))
                    cy = float(p.get("center_y", 0.0))
                    rot_deg = float(p.get("rotation_deg", 0.0))
                    try:
                        poly = make_shape(shape_name, w, h)
                    except Exception:
                        import numpy as _np
                        poly = _np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
                    if rot_deg:
                        poly = _rotate_poly(poly, _math.radians(rot_deg))
                    poly_trans = (poly + [cx, cy]).tolist()
                    parts_out.append({
                        "polygon_m": poly_trans,
                        "center_x": cx,
                        "center_y": cy,
                        "width": w,
                        "height": h,
                        "shape": shape_name,
                    })
                return self._json({
                    "ok": True,
                    "chamber_x_m": chamber_x,
                    "chamber_y_m": chamber_y,
                    "parts": parts_out,
                })
            except Exception as exc:
                return self._json({"ok": False, "error": str(exc)}, status=400)

        if path == "/api/tools/antennae-quick-search":
            try:
                req_mode = str(payload.get("mode", "single")).strip()
                if req_mode not in {"single", "optimizer", "turntable", "orientation_optimizer", "placement_optimizer"}:
                    req_mode = "single"
                exposure_minutes = float(payload.get("exposure_minutes", 6.0))
                if exposure_minutes <= 0:
                    exposure_minutes = 6.0
                effective = _effective_config_for_request(payload, req_mode, exposure_minutes)
                cfg_text = str(effective.get("effective_config_text", "")).strip()
                cfg = yaml.safe_load(cfg_text) if cfg_text else {}
                if not isinstance(cfg, dict):
                    cfg = {}
                ant_cfg = cfg.get("antennae", {})
                if not isinstance(ant_cfg, dict):
                    ant_cfg = {}
                ant_cfg["enabled"] = True
                cfg["antennae"] = ant_cfg
                from rfam_eqs_coupled import generate_antennae_quick_search
                out = generate_antennae_quick_search(cfg)
                return self._json(out)
            except Exception as exc:
                return self._json({"ok": False, "error": str(exc)}, status=400)

        mode = str(payload.get("mode", "single"))
        if mode not in {"single", "sweep", "optimizer", "turntable", "orientation_optimizer", "placement_optimizer", "shell_sweep", "antennae_size_sweep"}:
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


_SERVER_REF: ThreadingHTTPServer | None = None   # global so /api/quit can reach it


def main() -> None:
    global _SERVER_REF
    import socket as _socket
    host = os.environ.get("RFAM_GUI_HOST", "127.0.0.1")
    port = int(os.environ.get("RFAM_GUI_PORT", "8080"))
    server = ThreadingHTTPServer((host, port), Handler)
    # Enable TCP keepalive so idle connections survive NAT/proxy timeouts
    server.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)
    # Also disable HTTP request timeout — the default BaseHTTPServer uses a 5-minute
    # timeout; set to 0 (disable) so long-poll loops never get disconnected server-side.
    server.timeout = None
    _SERVER_REF = server
    print(f"RFAM GUI running at http://{host}:{port}")
    print("Prewarp flows are disabled in this interface.")
    print("Stop via the ⏻ Quit button in the UI or Ctrl-C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("HEATR server stopped.")


if __name__ == "__main__":
    main()
