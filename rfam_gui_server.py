#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import yaml

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "webui" / "static"
CONFIG_DIR = ROOT / "configs"
GUI_CONFIG_DIR = CONFIG_DIR / "_gui_generated"
OUTPUTS_DIR = ROOT / "outputs_eqs"
LOGS_DIR = OUTPUTS_DIR / "_logs"

GUI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

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
    "cross",
    "rounded_rect",
    "L_shape",
    "T_shape",
    "trapezoid",
]

DEFAULT_OPT_PHI = [0.50, 0.75, 0.90, 0.95]

JOBS_LOCK = threading.Lock()
JOBS: dict[str, dict[str, Any]] = {}


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
    dt_s = float(thermal.get("dt_s", 0.5))
    n_steps = float(thermal.get("n_steps", 0.0))
    exposure_minutes = _r4((dt_s * n_steps) / 60.0) if dt_s > 0 else 0.0

    sig: dict[str, Any] = {
        "mode": mode,
        "shape": str(part.get("shape", "")).strip(),
        "exposure_minutes": exposure_minutes,
    }

    if mode == "optimizer":
        opt = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer", {}), dict) else {}
        sig["phi_snapshots"] = [_r4(v) for v in _to_float_list(opt.get("phi_snapshots", []))]
        sig["temp_ceiling_c"] = _r4(float(opt.get("temp_ceiling_c", 250.0)))
        sig["highlight_phi"] = _r4(float(opt.get("highlight_phi", 0.90)))
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
    sig: dict[str, Any] = {
        "mode": mode,
        "shape": str(payload.get("shape", "")).strip(),
        "exposure_minutes": _r4(exposure_minutes),
    }
    if mode == "optimizer":
        phi = _to_float_list(payload.get("phi_snapshots", DEFAULT_OPT_PHI))
        if not phi:
            phi = list(DEFAULT_OPT_PHI)
        sig["phi_snapshots"] = [_r4(v) for v in phi]
        sig["temp_ceiling_c"] = _r4(float(payload.get("temp_ceiling_c", 250.0)))
        sig["highlight_phi"] = _r4(float(payload.get("highlight_phi", 0.90)))
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

    mode = str(req_sig.get("mode"))
    if mode == "optimizer":
        if cfg_sig.get("phi_snapshots", []) != req_sig.get("phi_snapshots", []):
            return False
        if abs(float(cfg_sig.get("temp_ceiling_c", 0.0)) - float(req_sig.get("temp_ceiling_c", 0.0))) > 1e-3:
            return False
        if abs(float(cfg_sig.get("highlight_phi", 0.0)) - float(req_sig.get("highlight_phi", 0.0))) > 1e-3:
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

    return True


def _preferred_fallback_name(shape: str, mode: str) -> list[str]:
    names: list[str] = []
    if mode == "optimizer":
        names.append(f"shape_{shape}_12min_optimizer.yaml")
        names.append(f"shape_{shape}_6min.yaml")
    elif mode == "turntable":
        names.append(f"shape_{shape}_6min_turntable.yaml")
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
    materials = cfg.get("materials", {}) if isinstance(cfg.get("materials", {}), dict) else {}
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
        "turntable_enabled": bool(turntable.get("enabled", False)) if isinstance(turntable, dict) else False,
        "turntable_rotation_deg": turntable.get("rotation_deg") if isinstance(turntable, dict) else None,
        "turntable_total_rotations": turntable.get("total_rotations") if isinstance(turntable, dict) else None,
        "turntable_interval_s": turntable.get("rotation_interval_s") if isinstance(turntable, dict) else None,
    }
    return {"name": cfg_path.name, "text": text, "important": important}


def _set_exposure_minutes(cfg: dict[str, Any], minutes: float) -> None:
    thermal = cfg.setdefault("thermal", {})
    dt_s = float(thermal.get("dt_s", 0.5))
    if dt_s <= 0:
        raise ValueError("thermal.dt_s must be > 0")
    n_steps = max(1, int(round((minutes * 60.0) / dt_s)))
    thermal["n_steps"] = n_steps


def _set_shape(cfg: dict[str, Any], shape: str) -> None:
    part = cfg.setdefault("geometry", {}).setdefault("part", {})
    part["shape"] = shape


def _disable_prewarp_like_blocks(cfg: dict[str, Any]) -> None:
    cfg.pop("prewarp", None)
    cfg.pop("evaluation", None)


def _advanced_payload(payload: dict[str, Any]) -> dict[str, Any]:
    adv = payload.get("advanced", {})
    return adv if isinstance(adv, dict) else {}


def _has_advanced_overrides(payload: dict[str, Any]) -> bool:
    return any(str(v).strip() != "" for v in _advanced_payload(payload).values())


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


def _configure_single(cfg: dict[str, Any]) -> None:
    cfg.pop("turntable", None)
    cfg.pop("optimizer", None)


def _make_job(mode: str, output_name: str | None = None) -> dict[str, Any]:
    jid = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    log_path = LOGS_DIR / f"gui_job_{jid}.log"
    job = {
        "id": jid,
        "mode": mode,
        "status": "running",
        "started_at": _now_iso(),
        "ended_at": None,
        "output_name": output_name,
        "log_path": str(log_path),
        "log_url": f"/files/outputs_eqs/_logs/{log_path.name}",
        "error": None,
        "config_resolution": [],
        "output_dirs": [],
        "artifacts": [],
    }
    with JOBS_LOCK:
        JOBS[jid] = job
    return job


def _finish_job(job_id: str, status: str, error: str | None = None) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["status"] = status
        job["ended_at"] = _now_iso()
        job["error"] = error


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
    for pattern in ("*.png", "*.gif"):
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


def _refresh_job_artifacts(job: dict[str, Any]) -> None:
    artifacts: list[dict[str, Any]] = []
    for rel in job.get("output_dirs", []):
        out_dir = OUTPUTS_DIR / rel
        if out_dir.exists() and out_dir.is_dir():
            artifacts.append(_build_artifact(out_dir))
    job["artifacts"] = artifacts


def _run_command(cmd: list[str], log_path: Path) -> int:
    _write_job_log(log_path, f"$ {' '.join(cmd)}")
    with log_path.open("a", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        return proc.wait()


def _prepare_config(mode: str, payload: dict[str, Any], exposure_minutes: float, output_tag: str, job: dict[str, Any]) -> Path:
    forced = str(payload.get("base_config", "")).strip() or None
    req_sig = _signature_for_request(payload, mode, exposure_minutes)
    resolved = _resolve_base_config_for_signature(req_sig, forced_base_config=forced)
    base_path = Path(resolved["config_path"])

    if resolved["match_type"] == "matched_existing" and not forced and not _has_advanced_overrides(payload):
        job["config_resolution"].append({
            "requested": req_sig,
            "resolved": resolved,
            "generated_config": None,
        })
        return base_path

    cfg = _load_yaml(base_path)
    _disable_prewarp_like_blocks(cfg)
    _set_shape(cfg, str(payload.get("shape", "")).strip())
    _apply_advanced_overrides(cfg, payload)
    _set_exposure_minutes(cfg, exposure_minutes)

    if mode == "optimizer":
        _configure_optimizer(cfg, payload)
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
    })
    return gen_path


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

    cfg_path = _prepare_config(mode=mode, payload=payload, exposure_minutes=exposure_minutes, output_tag=output_name, job=job)
    out_dir = OUTPUTS_DIR / output_name
    _register_job_output(job, out_dir)
    cmd = [sys.executable, "rfam_eqs_coupled.py", "--config", str(cfg_path), "--output-dir", str(out_dir)]
    rc = _run_command(cmd, Path(job["log_path"]))
    if rc != 0:
        raise RuntimeError(f"simulation exited with code {rc}")


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

    for minutes in sweep_minutes:
        if minutes <= 0:
            raise ValueError("sweep minutes must be > 0")
        suffix = str(minutes).replace(".", "p")
        cfg_path = _prepare_config(
            mode="single",
            payload=payload,
            exposure_minutes=minutes,
            output_tag=f"{output_prefix}_{suffix}min_sweep",
            job=job,
        )
        out_name = f"{output_prefix}_{suffix}min"
        out_dir = OUTPUTS_DIR / out_name
        _register_job_output(job, out_dir)
        cmd = [sys.executable, "rfam_eqs_coupled.py", "--config", str(cfg_path), "--output-dir", str(out_dir)]
        rc = _run_command(cmd, Path(job["log_path"]))
        if rc != 0:
            raise RuntimeError(f"sweep run ({minutes} min) exited with code {rc}")


def _job_worker(job_id: str, payload: dict[str, Any]) -> None:
    job = JOBS[job_id]
    try:
        mode = payload.get("mode", "single")
        if mode == "sweep":
            _launch_sweep_mode(payload, job)
        else:
            _launch_single_mode(payload, job)
        _refresh_job_artifacts(job)
        _finish_job(job_id, "completed")
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
        has_media = any(p.glob("*.png")) or any(p.glob("*.gif"))
        if not (has_summary or has_media):
            continue
        rel = p.relative_to(OUTPUTS_DIR).as_posix()
        items.append({
            "name": rel,
            "updated_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
        })
    return items


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
    return {"name": name, "summary": summary, "images": images, "subdirs": subdirs}


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
    for p in sorted(OUTPUTS_DIR.rglob("*"), key=lambda q: q.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue
        if _should_skip_result_path(p):
            continue
        has_summary = (p / "summary.json").exists()
        has_media = any(p.glob("*.png")) or any(p.glob("*.gif"))
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
        group = rel.split("/")[0] if "/" in rel else "runs"
        cards.append({
            "name": rel,
            "group": group,
            "updated_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            "summary_excerpt": _summary_excerpt(summary),
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

    sweeps: list[dict[str, Any]] = []
    sweeps_root = OUTPUTS_DIR / "sweeps"
    if sweeps_root.exists():
        for d in sorted([p for p in sweeps_root.iterdir() if p.is_dir()]):
            preview = _pick_preview_image(d, ["paper_style_report.png", "time_series.png", "thermal_fields_final.png"])
            images = [p.name for p in sorted(d.glob("*.png"))]
            sweeps.append({
                "name": d.name,
                "preview": preview,
                "images": images,
            })

    shapes: list[dict[str, Any]] = []
    shapes_root = OUTPUTS_DIR / "shapes"
    if shapes_root.exists():
        for d in sorted([p for p in shapes_root.iterdir() if p.is_dir()]):
            preview = _pick_preview_image(d, ["rf_summary_v5.png", "paper_style_report.png", "time_series.png"])
            images = [p.name for p in sorted(d.glob("*.png"))]
            shapes.append({
                "name": d.name,
                "preview": preview,
                "images": images,
            })

    return {
        "master_sweep": master_sweep,
        "sweeps": sweeps,
        "shapes": shapes,
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
        if path == "/results" or path == "/results.html":
            return self._serve_static("results.html")
        if path == "/theory" or path == "/theory.html":
            return self._serve_static("theory.html")
        if path == "/examples" or path == "/examples.html":
            return self._serve_static("examples.html")
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if path == "/api/meta":
            return self._json({
                "shape_options": SUPPORTED_SHAPES,
                "base_configs": _list_base_configs(),
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
                jobs = list(JOBS.values())
            for j in jobs:
                _refresh_job_artifacts(j)
            jobs.sort(key=lambda j: j["started_at"])
            return self._json(jobs)
        if path == "/api/results":
            return self._json(_collect_results())
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
        if path not in {"/api/run", "/api/match-config"}:
            return self._json({"error": "not found"}, status=404)

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._json({"error": "invalid json"}, status=400)

        mode = str(payload.get("mode", "single"))
        if mode not in {"single", "sweep", "optimizer", "turntable"}:
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
                        rows.append({"minutes": float(m), "requested": req_sig, "resolved": resolved})
                    return self._json({"mode": mode, "matches": rows})
                minutes = float(payload.get("exposure_minutes", 6.0))
                req_sig = _signature_for_request(payload, mode, minutes)
                resolved = _resolve_base_config_for_signature(req_sig, forced_base_config=str(payload.get("base_config", "")).strip() or None)
                if _has_advanced_overrides(payload):
                    resolved = dict(resolved)
                    resolved["match_type"] = "new_from_overrides"
                return self._json({"mode": mode, "requested": req_sig, "resolved": resolved})
            except Exception as exc:
                return self._json({"error": str(exc)}, status=400)

        output_name = str(payload.get("output_name", "")).strip()
        job = _make_job(mode=mode, output_name=output_name)
        t = threading.Thread(target=_job_worker, args=(job["id"], payload), daemon=True)
        t.start()
        self._json({"job_id": job["id"], "status": job["status"]}, status=202)

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
