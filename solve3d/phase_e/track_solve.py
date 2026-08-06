"""Status snapshotter for a long detached phase_e solve.

Reads a checkpoint npz + pid + budget and emits one honest status object:
progress, J trajectory, gradient health (finite / nan / zero), memory,
per-eval wall, ETA, and stall detection. No dependency on the solver
process; safe to call any time, including while the checkpoint is being
written (a torn read is caught and reported, not crashed on).

Usage:
    python -m solve3d.phase_e.track_solve <checkpoint.npz> <pid> <budget> [--json]

Exit code is 0 when the read succeeded, 2 on a torn/absent checkpoint, so
a watcher can distinguish "no data yet" from "data says X".
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return pid_has_perm_err(pid)
    except Exception:
        return False


def pid_has_perm_err(pid: int) -> bool:
    # PermissionError means the pid exists but is not ours; still alive.
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _rss_gb(pid: int) -> float | None:
    try:
        import subprocess

        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return round(int(out) / 1024 / 1024, 2) if out else None
    except Exception:
        return None


def snapshot(ckpt: Path, pid: int, budget: int) -> dict:
    import numpy as np

    now = time.time()
    alive = _pid_alive(pid)
    out: dict = {
        "checkpoint": str(ckpt),
        "pid": pid,
        "pid_alive": alive,
        "budget_evals": budget,
        "rss_gb": _rss_gb(pid) if alive else None,
    }
    if not ckpt.exists():
        out["state"] = "no_checkpoint_yet"
        out["read_ok"] = False
        return out
    out["ckpt_age_s"] = round(now - ckpt.stat().st_mtime)
    try:
        d = np.load(ckpt, allow_pickle=True)
        hist = json.loads(str(d["hist"]))
        n = int(d["n"])
        best_j = float(d["best_j"]) if "best_j" in d.files else float(d["best_J"])
    except Exception as e:  # torn write or schema drift
        out["state"] = "checkpoint_unreadable"
        out["read_ok"] = False
        out["error"] = repr(e)
        return out

    out["read_ok"] = True
    j0 = hist[0]["J"] if hist else None
    grads = [e.get("grad_norm") for e in hist]
    last_g = grads[-1] if grads else None

    def _bad(g):
        if g is None:
            return "missing"
        try:
            gf = float(g)
        except Exception:
            return "nonnumeric"
        if gf != gf:
            return "nan"
        if gf in (float("inf"), float("-inf")):
            return "inf"
        if gf == 0.0:
            return "zero"
        return "finite"

    grad_health = _bad(last_g)
    # wall_s resets on warm restart (resume), so cross-eval deltas can go
    # negative and must never drive timing. Use only a positive monotonic
    # delta; otherwise timing this run is unknown (report it, do not fake).
    walls = [e.get("wall_s", 0.0) for e in hist]
    per_eval = None
    if len(walls) >= 2 and walls[-1] > walls[0]:
        per_eval = round((walls[-1] - walls[0]) / (len(walls) - 1))
    elif len(walls) == 1 and walls[0] > 0:
        per_eval = round(walls[0])
    out["timing_note"] = None if per_eval else "unknown (resumed run; wall clock reset)"
    remaining = max(0, budget - n)
    eta_s = per_eval * remaining if per_eval else None

    j_traj = [e.get("J") for e in hist]
    # Trend over the last few evals: are we still descending, or flat?
    trend = "unknown"
    if len(j_traj) >= 2 and j0:
        recent = j_traj[-min(3, len(j_traj)):]
        drop = (recent[0] - recent[-1]) / j0
        if drop > 1e-3:
            trend = "descending"
        elif drop > 1e-5:
            trend = "slow"
        elif drop >= -1e-5:
            trend = "flat"
        else:
            trend = "rising"
    out.update({
        "state": "running" if alive else "process_gone",
        "evals_done": n,
        "evals_total": budget,
        "J0": j0,
        "best_J": best_j,
        "J_trajectory": [round(j, 10) if j is not None else None for j in j_traj],
        "trend": trend,
        "design_cells_moved": None,
        "improvement_pct": round(100 * (1 - best_j / j0), 3) if j0 else None,
        "last_grad_norm": last_g,
        "grad_health": grad_health,
        "all_grad_health": [_bad(g) for g in grads],
        "per_eval_wall_s": per_eval,
        "remaining_evals": remaining,
        "eta_s": eta_s,
        "eta_h": round(eta_s / 3600, 1) if eta_s else None,
    })

    # Stall: alive but the checkpoint has not advanced in too long. Use an
    # ABSOLUTE cap (not per_eval, which is unknown across resumes). One eval
    # is ~80 min on this class of run; 2.5 h is a safe wedge threshold.
    STALL_CAP_S = 9000
    if alive and out["ckpt_age_s"] > STALL_CAP_S:
        out["stall_suspected"] = True
        out["stall_reason"] = f"ckpt age {out['ckpt_age_s']}s > cap {STALL_CAP_S}s"
    else:
        out["stall_suspected"] = False

    # Health verdict a watcher can branch on.
    if not alive:
        out["verdict"] = "DIED" if remaining > 0 else "DONE"
    elif grad_health in ("nan", "inf", "zero"):
        out["verdict"] = f"BAD_GRADIENT_{grad_health}"
    elif out["stall_suspected"]:
        out["verdict"] = "STALLED"
    elif remaining == 0:
        out["verdict"] = "DONE"
    else:
        out["verdict"] = "HEALTHY"
    return out


def _pretty(s: dict) -> str:
    if not s.get("read_ok"):
        return f"[{s['state']}] pid {s['pid']} alive={s.get('pid_alive')}"
    timing = f"{s['per_eval_wall_s']}s/eval ETA {s['eta_h']}h" if s.get("eta_h") else "timing unknown (resumed)"
    return (
        f"[{s['verdict']}] eval {s['evals_done']}/{s['evals_total']} "
        f"| J {s['best_J']:.4e} ({s['improvement_pct']:+.2f}%, {s.get('trend')}) "
        f"| grad {s['grad_health']} | {timing} "
        f"| RSS {s['rss_gb']}GB | ckpt {s['ckpt_age_s']}s ago"
    )


def main() -> int:
    args = sys.argv[1:]
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    ckpt = Path(args[0])
    pid = int(args[1])
    budget = int(args[2])
    s = snapshot(ckpt, pid, budget)
    print(json.dumps(s, indent=2) if as_json else _pretty(s))
    return 0 if s.get("read_ok") or s["state"] == "no_checkpoint_yet" else 2


if __name__ == "__main__":
    raise SystemExit(main())
