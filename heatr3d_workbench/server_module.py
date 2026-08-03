"""Workbench server module, mounted into rfam_gui_server (STDLIB ONLY).

The GUI server interpreter has an unreliable numpy (Appendix B constraint 2),
so this module never imports numpy/trimesh/matplotlib; anything numeric runs
in the workbench_job subprocess under the solver venv.

Endpoints served (dispatched from rfam_gui_server's handler):
  GET  /api/heatr3d/wb/library   - 14-shape gallery source (meta JSON)
  GET  /api/heatr3d/wb/queue     - queue state + live progress per running job
  GET  /api/heatr3d/wb/run?id=   - run detail: results, shape metrics, flags,
                                   badges, march series, fieldmeta, snapshots
  GET  /api/heatr3d/wb/solved    - solved-map cards from solve3d/results
  POST /api/heatr3d/wb/enqueue   - validate + enqueue (+ campaign)
  POST /api/heatr3d/wb/cancel    - cancel a queued/running job
  POST /api/heatr3d/wb/intake    - STL gate verdict (sync subprocess)

Existing /api/heatr3d/* routes are untouched (parity: Appendix A).
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs

from heatr3d_workbench import badges as B
from heatr3d_workbench import jobqueue as JQ
from heatr3d_workbench import march as M

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
H3D_OUT = ROOT / "outputs_eqs" / "_heatr3d"
LIB_META = ROOT / "shape_library_3d" / "meta"
SOLVE_RESULTS = ROOT / "solve3d" / "results"
UPLOADS = H3D_OUT / "_uploads"

GRID_CEILING_FULL = 96
_LOCK = threading.Lock()
_QUEUE: Optional[JQ.Queue] = None


def _queue() -> JQ.Queue:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = JQ.Queue(H3D_OUT)
    return _QUEUE


def _queue_for_tests(store: Path) -> JQ.Queue:
    """Isolated queue instance for tests (does not touch the module singleton)."""
    return JQ.Queue(Path(store))


def _h3d_python() -> str:
    """Solver interpreter resolution (mirrors rfam_gui_server._h3d_python;
    duplicated by design - Appendix B forbids modifying the shared helper)."""
    cands = [
        ROOT / ".venv-heatr3d" / "bin" / "python",
        ROOT / ".venv312" / "bin" / "python",
        Path.home() / "GaTech Dropbox" / "Matthew McCoy" / "mattmccoy-research" /
        "research" / "dissertation_materials" / "analysis-3dfgm" / ".venv312" /
        "bin" / "python",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    return sys.executable


_SPAWN_ENV_PINS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


# --------------------------------------------------------------------------- #
# LIBRARY
# --------------------------------------------------------------------------- #
def library_shapes() -> List[Dict[str, Any]]:
    shapes: List[Dict[str, Any]] = []
    if not LIB_META.is_dir():
        return shapes
    for p in sorted(LIB_META.glob("*.json")):
        if p.name == "library_manifest.json":
            continue
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("library meta %s unreadable: %s", p.name, e)
            continue
        shapes.append({
            "name": m.get("name", p.stem),
            "tier": m.get("tier"),
            "role": m.get("role"),
            "rf_characteristic": m.get("rf_characteristic", ""),
            "numerical_characteristic": m.get("numerical_characteristic", ""),
            "volume_mm3": m.get("actual_volume_mm3"),
            "bbox_mm": m.get("bbox_mm"),
            "loadable": m.get("tier") in (1, 2),
        })
    order = {1: 0, 2: 1, 3: 2}
    shapes.sort(key=lambda s: (order.get(s["tier"], 9), s["name"]))
    return shapes


def _loadable_names() -> set:
    return {s["name"] for s in library_shapes() if s["loadable"]}


# --------------------------------------------------------------------------- #
# RUN DETAIL
# --------------------------------------------------------------------------- #
def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _slice_axes(run: Path, fieldmeta: Optional[Dict[str, Any]]) -> List[str]:
    """Axes with pre-rendered slices. Legacy runs are z-only; the new writer
    advertises axes in fieldmeta and writes <field>_x_/_y_ PNGs."""
    if fieldmeta and "axes" in fieldmeta:
        sl = run / "slices"
        axes = []
        for ax in ("x", "y", "z"):
            if any(sl.glob(f"*_{ax}_*.png")):
                axes.append(ax)
        return axes or ["z"]
    return ["z"]


def run_detail(jid: str) -> Dict[str, Any]:
    if not jid or "/" in jid or ".." in jid:
        return {"error": "bad run id"}
    run = H3D_OUT / jid
    if not run.is_dir():
        return {"error": f"run {jid} not found"}
    results = _read_json(run / "results.json")
    cfg = _read_json(run / "config.json") or {}
    fieldmeta = _read_json(run / "fieldmeta.json")
    detail: Dict[str, Any] = {"id": jid, "config": cfg}
    if results is None:
        detail["error"] = "run has no results.json (incomplete or failed)"
        job = _queue().get(jid)
        if job:
            detail["state"] = job["state"]
        return detail
    flags = B.run_flags(results)
    detail.update({
        "results": results,
        "shape_metrics": _read_json(run / "shape_metrics.json"),
        "flags": flags,
        "banner": B.banner_state(flags),
        "badges": {c: B.badge_for(c) for c in
                   ("thermal", "eqs", "shape", "shrinkage")},
        "fieldmeta": fieldmeta,
        "slice_axes": _slice_axes(run, fieldmeta),
        "snapshots": _read_json(run / "snapshots" / "index.json"),
        "has_warp": (run / "warped_geometry.json").exists(),
    })
    log = run / "job.log"
    if log.exists():
        try:
            expo = float(cfg.get("exposure_s") or 0) or None
            detail["march"] = M.parse_log(log.read_text(errors="replace"),
                                          max_time_s=expo)
        except OSError:
            pass
    return detail


# --------------------------------------------------------------------------- #
# QUEUE
# --------------------------------------------------------------------------- #
def validate_cfg(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    src = str(cfg.get("source", "parametric"))
    if src not in ("parametric", "library", "stl"):
        return False, f"unknown geometry source '{src}'"
    try:
        n = int(cfg.get("n", 64))
    except (TypeError, ValueError):
        return False, "grid n is not an integer"
    if n > GRID_CEILING_FULL:
        return False, (f"grid n={n} exceeds the full-physics ceiling "
                       f"n<={GRID_CEILING_FULL} (EQS-01)")
    if n < 8:
        return False, f"grid n={n} is below the minimum (8)"
    if src == "library":
        name = str(cfg.get("library_shape", ""))
        if name not in _loadable_names():
            return False, (f"library shape '{name}' is not a loadable part "
                           f"(Tier-3 rejection fixture or unknown)")
    if src == "stl" and not cfg.get("stl"):
        return False, "source=stl requires an accepted intake upload first"
    if src == "parametric" and not cfg.get("shape"):
        return False, "source=parametric requires a shape"
    return True, ""


_CFG_KEYS = ("source", "shape", "library_shape", "stl", "stl_name", "diam",
             "zspan", "n", "fgm", "magnitude", "densify", "exposure_s",
             "stop_mean_rho", "phase_update", "power_density_w_per_m3",
             "eqs_update_interval_s", "sigma_temp_coeff_per_K",
             "sigma_density_coeff", "snapshots")


def enqueue(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {k: payload[k] for k in _CFG_KEYS if payload.get(k) is not None}
    ok, err = validate_cfg(cfg)
    if not ok:
        return {"error": err}
    with _LOCK:
        jid = _queue().enqueue(cfg, campaign=payload.get("campaign"))
        _tick_locked()
    return {"id": jid}


def cancel(payload: Dict[str, Any]) -> Dict[str, Any]:
    jid = str(payload.get("id", ""))
    with _LOCK:
        ok = _queue().cancel(jid)
        _tick_locked()
    return {"ok": ok} if ok else {"error": f"job {jid} is not active"}


def _spawn(jid: str) -> None:
    q = _queue()
    d = q.run_dir(jid)
    logf = open(d / "job.log", "a")
    env = dict(os.environ)
    env.update(_SPAWN_ENV_PINS)
    proc = subprocess.Popen(
        [_h3d_python(), str(ROOT / "heatr3d_workbench" / "workbench_job.py"),
         str(d / "config.json")],
        cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT, text=True, env=env)
    q.mark_running(jid, pid=proc.pid)
    logger.info("workbench run %s spawned pid=%s", jid, proc.pid)


def _tick_locked() -> None:
    q = _queue()
    q.refresh()
    while q.claimable():
        nxt = q.next_queued()
        if nxt is None:
            break
        _spawn(nxt)


def queue_state() -> Dict[str, Any]:
    with _LOCK:
        _tick_locked()
        jobs = _queue().jobs()
    out = []
    for j in sorted(jobs, key=lambda x: x.get("created", 0), reverse=True)[:80]:
        item = dict(j)
        if j["state"] == "running":
            log = _queue().run_dir(j["id"]) / "job.log"
            if log.exists():
                try:
                    cfg = _read_json(_queue().run_dir(j["id"]) / "config.json") or {}
                    expo = float(cfg.get("exposure_s") or 0) or None
                    parsed = M.parse_log(log.read_text(errors="replace"),
                                         max_time_s=expo)
                    item["progress"] = parsed["progress"]
                    item["phase"] = parsed["phase"]
                except OSError:
                    pass
        out.append(item)
    return {"jobs": out}


# --------------------------------------------------------------------------- #
# INTAKE (sync subprocess: trimesh runs solver-side only)
# --------------------------------------------------------------------------- #
def intake(payload: Dict[str, Any]) -> Dict[str, Any]:
    b64 = payload.get("stl_b64")
    name = str(payload.get("stl_name", "upload.stl"))
    if not b64:
        return {"error": "no STL payload"}
    try:
        raw = base64.b64decode(b64)
    except (ValueError, TypeError):
        return {"error": "invalid base64 STL payload"}
    UPLOADS.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(raw).hexdigest()[:12]
    stl = UPLOADS / f"{digest}.stl"
    if not stl.exists():
        stl.write_bytes(raw)
    verdict_path = UPLOADS / f"{digest}.verdict.json"
    try:
        proc = subprocess.run(
            [_h3d_python(), str(ROOT / "heatr3d_workbench" / "workbench_job.py"),
             "--intake", str(stl), str(verdict_path)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120,
            env={**os.environ, **_SPAWN_ENV_PINS})
    except subprocess.TimeoutExpired:
        return {"error": "intake timed out (120 s)"}
    verdict = _read_json(verdict_path)
    if verdict is None:
        tail = (proc.stderr or proc.stdout or "no output")[-600:]
        return {"error": f"intake subprocess failed: {tail}"}
    verdict["stl"] = str(stl)
    verdict["stl_name"] = name
    return verdict


# --------------------------------------------------------------------------- #
# SOLVED MAPS (read-only truth surface over solve3d/results)
# --------------------------------------------------------------------------- #
def solved_cards() -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    solves = _read_json(SOLVE_RESULTS / "phase_c_solves.json") or {"arms": {}}
    baselines = _read_json(SOLVE_RESULTS / "phase_c_baselines.json") or {"arms": {}}
    gate = _read_json(SOLVE_RESULTS / "phase_c_gate.json") or {"arms": {}}
    prereg = _read_json(SOLVE_RESULTS / "phase_c_preregistration.json") or {}
    badge = B.badge_for("solved_map")

    def _card(arm: str, rec: Dict[str, Any], kind: str) -> Dict[str, Any]:
        g = gate.get("arms", {}).get(arm, {})
        status = str(rec.get("status", rec.get("note", "scored")))
        c: Dict[str, Any] = {
            "arm": arm, "kind": kind, "shape": "cylinder (extruded circle)",
            "campaign": "phase_c", "status": status,
            "solved_label": bool(g.get("solved_label", False)),
            "deviation": "scaled" in arm,
            "badge": badge,
        }
        for k in ("J_asymmetric", "J_symmetric", "J_out_of_bounds",
                  "J_in_bounds_deficit", "sigma_T_c", "in_part_melt_frac_phi09",
                  "bed_melt_frac_phi09", "gradient_evaluations_used",
                  "map_mean", "map_min", "map_max"):
            if k in rec:
                c[k] = rec[k]
        ms = rec.get("map_stats")
        if isinstance(ms, dict):
            for k in ("mean", "min", "max"):
                if k in ms:
                    c[f"map_{k}"] = ms[k]
        if g:
            hold = g.get("mesh_holdout", {})
            sm = g.get("smoothing_robustness", {})
            c["gates"] = {
                "mesh_holdout_pass": hold.get("all_pass", hold.get("pass")),
                "smoothing_pass": sm.get("pass"),
                "in_grid_margin_rel": g.get("in_grid_margin_rel"),
            }
        return c

    for arm, rec in baselines.get("arms", {}).items():
        kind = "baseline" if "uniform" in arm else "heuristic"
        card = _card(arm, rec, kind)
        if arm == "inversion_map":
            card["status"] = "DROPPED"
            card["status_detail"] = ("pre-registered transfer condition violated "
                                     "(in-part dopant moved 7.15% > 2%); provenance "
                                     "proven before the drop")
        cards.append(card)
    for arm, rec in solves.get("arms", {}).items():
        cards.append(_card(arm, rec, "solve"))
    for arm in ("solve_projection_beta_continuation", "solve_filter_only_w3"):
        if not any(c["arm"] == arm for c in cards):
            cards.append({"arm": arm, "kind": "solve", "campaign": "phase_c",
                          "shape": "cylinder (extruded circle)",
                          "status": "NOT_RUN",
                          "status_detail": "session compute exhausted; "
                                           "pre-registered unrun_arms_policy applies",
                          "solved_label": False, "deviation": False,
                          "badge": badge})
    cards.append({"arm": "_report", "kind": "report", "campaign": "phase_c",
                  "report": "solve3d/PHASE_C_REPORT.md",
                  "prereg_commit": prereg.get("commit", "08e3b97"),
                  "badge": badge})
    return cards


# --------------------------------------------------------------------------- #
# Dispatch (called from rfam_gui_server's handler)
# --------------------------------------------------------------------------- #
GET_PATHS = ("/api/heatr3d/wb/library", "/api/heatr3d/wb/queue",
             "/api/heatr3d/wb/run", "/api/heatr3d/wb/solved")
POST_PATHS = ("/api/heatr3d/wb/enqueue", "/api/heatr3d/wb/cancel",
              "/api/heatr3d/wb/intake")


def handle_get(path: str, query: str) -> Optional[Tuple[int, Dict[str, Any]]]:
    if path == "/api/heatr3d/wb/library":
        return 200, {"shapes": library_shapes()}
    if path == "/api/heatr3d/wb/queue":
        return 200, queue_state()
    if path == "/api/heatr3d/wb/run":
        q = parse_qs(query or "")
        d = run_detail((q.get("id") or [""])[0])
        return (404 if d.get("error") and "results" not in d else 200), d
    if path == "/api/heatr3d/wb/solved":
        return 200, {"cards": solved_cards()}
    return None


def handle_post(path: str, payload: Dict[str, Any]) -> Optional[Tuple[int, Dict[str, Any]]]:
    if path == "/api/heatr3d/wb/enqueue":
        r = enqueue(payload)
        return (400 if "error" in r else 200), r
    if path == "/api/heatr3d/wb/cancel":
        r = cancel(payload)
        return (400 if "error" in r else 200), r
    if path == "/api/heatr3d/wb/intake":
        r = intake(payload)
        return (400 if "error" in r else 200), r
    return None
