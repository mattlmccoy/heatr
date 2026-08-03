#!/usr/bin/env python3
"""Schedule co-solve wrapper: dopant map + turntable program from one job.

Commissioned by the graphical user interface (mode "Schedule co-solve"); can
also run standalone. It SHELLS the existing campaign drivers read-only (their
modules are imported and their output directory constant is redirected into
this run's folder; no campaign artifact is ever overwritten):

  indexed     -> scripts/analysis/run_rot_avg_solve.py  (matched N-position
                 averaged kernel; the campaign's best actuator on the
                 four-fold shapes, CONTINUOUS_ROTATION_REPORT.md section 7)
  asym_dwell  -> scripts/analysis/run_dwell_solve.py    (joint dopant map +
                 asymmetric dwell fractions, DWELL_SCHEDULE_REPORT.md)
  sequential  -> scripts/analysis/run_seq_arms.py       (time-varying
                 sequential schedule; L_shape and T_shape only)

Outputs, landed in --output-dir (the Results tab picks the folder up):
  summary.json                       engine version stamped, honest labels
  schedule_maps.npz                  the co-solved 4 bits-per-pixel map
  turntable_program_deliverable.json the machine-readable program (dwell
                                     campaign format: ordered moves of
                                     {position_deg, dwell_s, move_at_s})
  turntable_program_equal_dwell_control.json  the equal-dwell control
  j_trace.json                       the per-evaluation objective trace
  fig_schedule_cosolve.png           map + melt-versus-nominal + J trace
  campaign_raw/                      the driver's own artifacts, untouched

Honesty note, carried into summary.json: schedules execute on the part-frame
march at solve time; the engine turntable program mode
(rfam_eqs_coupled tt_program_mode) is the execution path for verification.

Run:
  ./.venv312/bin/python scripts/solve_schedule.py --shape cross --mode indexed \
      --positions 4 --budget 10 --output-dir outputs_eqs/runs/cross/schedule_cosolve/x
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]

FE_PER_EVAL = 2.5          # one gradient evaluation = 2.5 forward-equivalents
CYCLE_TIME_S = 20.0        # the dwell campaign's production cycle time
TOTAL_S = 750.0            # 1500-step horizon at dt 0.5 s
DT_S = 0.5
MODES = ("indexed", "asym_dwell", "sequential")
# Supported shapes per driver (mirrors run_dwell_solve.SHAPES,
# run_rot_avg_solve.SHAPES and run_seq_arms.CFG; validated again at run time).
DWELL_SHAPES = ("cross", "square", "T_shape", "L_shape")
ROT_SHAPES = ("T_shape", "L_shape", "cross", "star", "square")
SEQ_SHAPES = ("L_shape", "T_shape")
VALID_POSITIONS = (2, 3, 4, 6, 8, 12, 24)


def budget_to_evals(budget_fe: float) -> int:
    """Forward-equivalents to gradient evaluations (campaign convention:
    40 forward-equivalents = 16 evaluations)."""
    return max(1, int(round(float(budget_fe) / FE_PER_EVAL)))


def positions_to_step_deg(n: int) -> float:
    if int(n) not in VALID_POSITIONS:
        raise ValueError(f"positions must be one of {VALID_POSITIONS}, got {n}")
    return 360.0 / int(n)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shape", required=True)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--positions", type=int, default=4,
                    help="indexed mode: number of turntable positions")
    ap.add_argument("--budget", type=float, required=True,
                    help="forward-equivalents per solve stage")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    if a.budget <= 0:
        ap.error("budget must be > 0")
    if a.mode == "indexed":
        if a.shape not in ROT_SHAPES:
            ap.error(f"indexed mode supports {ROT_SHAPES}, got {a.shape!r}")
        if a.positions not in VALID_POSITIONS:
            ap.error(f"positions must be one of {VALID_POSITIONS}")
    elif a.mode == "asym_dwell":
        if a.shape not in DWELL_SHAPES:
            ap.error(f"asym_dwell mode supports {DWELL_SHAPES}, got {a.shape!r}")
    elif a.mode == "sequential":
        if a.shape not in SEQ_SHAPES:
            ap.error(f"sequential mode supports {SEQ_SHAPES} only "
                     f"(run_seq_arms.CFG), got {a.shape!r}")
    return a


# ---------------------------------------------------------------------------
# driver-log progress watch: campaign log lines -> SCHEDULE_PROGRESS lines
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(
    r"block\s+(\w+)\s*:\s*(\d+)\s+evals,\s*J\s+([0-9.eE+-]+)\s*->\s*([0-9.eE+-]+)")
_START_RE = re.compile(
    r"start/(\w+):\s*(\d+)\s+evaluations,\s*J\s+([0-9.eE+-]+)\s*->\s*([0-9.eE+-]+)")
_ARM_RE = re.compile(r"\sJ\s+([0-9.eE+-]+)\s+IoU\s+([0-9.]+)")


class DriverWatch:
    """Stateful parser over the campaign drivers' own log lines.

    Block-completion and start-completion lines advance the evaluation
    counter; scored-arm rows update the current objective J and the best
    intersection over union seen so far. feed() returns a progress dict when
    the line carried progress, else None.
    """

    def __init__(self, evals_total: Optional[int]) -> None:
        self.evals_total = int(evals_total) if evals_total else None
        self.evals_done = 0
        self.current_j: Optional[float] = None
        self.best_iou: Optional[float] = None

    def _progress(self) -> Dict[str, Any]:
        return {
            "evals_done": self.evals_done,
            "evals_total": self.evals_total,
            "fe_spent": self.evals_done * FE_PER_EVAL,
            "fe_budget": (self.evals_total * FE_PER_EVAL
                          if self.evals_total else None),
            "J": self.current_j,
            "IoU": self.best_iou,
        }

    def feed(self, line: str) -> Optional[Dict[str, Any]]:
        m = _BLOCK_RE.search(line) or _START_RE.search(line)
        if m:
            self.evals_done += int(m.group(2))
            self.current_j = float(m.group(4))
            return self._progress()
        m = _ARM_RE.search(line)
        if m:
            self.current_j = float(m.group(1))
            iou = float(m.group(2))
            self.best_iou = iou if self.best_iou is None else max(self.best_iou, iou)
            return self._progress()
        return None


def format_progress_line(p: Dict[str, Any]) -> str:
    toks = ["SCHEDULE_PROGRESS"]
    for k in ("evals_done", "evals_total", "fe_spent", "fe_budget", "J", "IoU"):
        v = p.get(k)
        if v is None:
            continue
        toks.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    return " ".join(toks)


class _TeeWatch:
    """stdout tee that feeds every completed line to a DriverWatch and emits
    SCHEDULE_PROGRESS lines alongside the original output."""

    def __init__(self, real: Any, watch: DriverWatch) -> None:
        self._real = real
        self._watch = watch
        self._buf = ""

    def write(self, s: str) -> int:
        self._real.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            p = self._watch.feed(line)
            if p is not None:
                self._real.write(format_progress_line(p) + "\n")
                self._real.flush()
        return len(s)

    def flush(self) -> None:
        self._real.flush()


# ---------------------------------------------------------------------------
# shared assembly helpers (heavy imports stay inside functions)
# ---------------------------------------------------------------------------

def _campaign_paths() -> None:
    for p in (str(REPO), str(REPO / "fgm_solve_campaign"),
              str(REPO / "scripts" / "analysis")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _equal_dwell_program(shape: str, positions_deg: List[float],
                         voltage_v: float, stop_s: float,
                         map_key: str, arm: str) -> Dict[str, Any]:
    """Equal-dwell cycle program in the dwell campaign JSON format."""
    import numpy as np
    from adjoint2d import dwell
    n = len(positions_deg)
    w = np.full(n, 1.0 / n)
    pr = dwell.cycle_program(w, np.asarray(positions_deg, dtype=float),
                             cycle_time_s=CYCLE_TIME_S, total_s=TOTAL_S,
                             dt_s=DT_S)
    j = pr.as_json()
    j.update({
        "shape": shape, "arm": arm,
        "candidate_positions_deg": [float(a) for a in positions_deg],
        "recommended_stop_s": float(stop_s),
        "dopant_map_npz": "schedule_maps.npz",
        "dopant_map_key": map_key,
        "rf_program": {"mode": "constant", "relative_power": 1.0,
                       "voltage_v": float(voltage_v)},
    })
    return j


def _write_summary(out_dir: Path, args: argparse.Namespace, *,
                   n_evals: int, metrics: Dict[str, Any],
                   program_files: List[str], wall_s: float,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from rfam_eqs_coupled import ENGINE_VERSION, ENGINE_VERSION_NAME
    smoke = float(args.budget) < 20.0
    summary: Dict[str, Any] = {
        "run_type": "schedule_cosolve",
        "engine_version": str(ENGINE_VERSION),
        "engine_version_name": str(ENGINE_VERSION_NAME),
        "shape": args.shape,
        "schedule_mode": args.mode,
        "budget_forward_equivalents": float(args.budget),
        "n_gradient_evals_per_stage": int(n_evals),
        "label": (args.label or ("smoke" if smoke else "standard")),
        "quality_class": ("smoke class, not a quality solve (budget < 20 "
                          "forward-equivalents)" if smoke else "standard"),
        "program_files": program_files,
        "execution_note": (
            "Schedules execute on the part-frame march at solve time; the "
            "engine turntable program mode (rfam_eqs_coupled "
            "tt_program_mode) is the execution path for verification."),
        "grid_note": ("grid 120 only; SOLVE_ROBUSTNESS_VALIDATION.md: "
                      "grid-120 fidelity does not transfer to grid 160"),
        "solved_at": datetime.now().isoformat(timespec="seconds"),
        "wall_s": float(wall_s),
    }
    summary.update(metrics)
    if extra:
        summary.update(extra)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


def _figure(out_dir: Path, title: str, sat: Any, phi: Any, part_mask: Any,
            trace: List[Dict[str, Any]]) -> None:
    """Map + melt-versus-nominal + objective trace, one readable row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=180)
    ax = axes[0]
    im = ax.imshow(np.asarray(sat), origin="lower", cmap="viridis",
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.contour(np.asarray(part_mask), levels=[0.5], colors="w", linewidths=0.8)
    ax.set_title("co-solved dopant map (4 bits per pixel)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1]
    phi_a = np.asarray(phi)
    ax.imshow(phi_a >= 0.5, origin="lower", cmap="Reds", interpolation="nearest")
    ax.contour(np.asarray(part_mask), levels=[0.5], colors="k", linewidths=1.0)
    ax.set_title("melt at stop (red) vs nominal outline (black)")
    ax = axes[2]
    if trace:
        xs = list(range(1, len(trace) + 1))
        ax.plot(xs, [t["J"] for t in trace], "o-", ms=3, lw=1)
        ax.set_yscale("log")
    ax.set_xlabel("gradient evaluation")
    ax.set_ylabel("shape objective J")
    ax.set_title("objective trace")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "fig_schedule_cosolve.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# the three modes
# ---------------------------------------------------------------------------

def run_indexed(args: argparse.Namespace, out_dir: Path, raw: Path) -> None:
    _campaign_paths()
    import numpy as np
    import run_rot_avg_solve as rot
    t0 = time.perf_counter()
    n_evals = budget_to_evals(args.budget)
    step = positions_to_step_deg(args.positions)
    watch = DriverWatch(evals_total=2 * n_evals)   # two starts (cold + warm)
    real = sys.stdout
    sys.stdout = _TeeWatch(real, watch)            # type: ignore[assignment]
    try:
        rot.OUT_ROOT = raw                         # redirect, never overwrite
        result = rot.main(args.shape, step, n_evals)
    finally:
        sys.stdout = real
    sfx = "" if abs(step - rot.STEP_DEG) < 1e-9 else f"_step{int(round(step))}"
    npz_src = raw / f"{args.shape}_rotavg{sfx}_maps.npz"
    shutil.copy2(npz_src, out_dir / "schedule_maps.npz")
    arms = result["arms"]
    positions = [float(i * step) for i in range(args.positions)]
    stop_s = float(arms["AVG_4bpp"]["t_stop_s"])
    volt = float(result["voltage_v"])
    prog = _equal_dwell_program(args.shape, positions, volt, stop_s,
                                "sat_4bpp", "deliverable")
    prog["note"] = ("indexed mode: the deliverable IS the equal-dwell "
                    "program over the matched positions")
    ctrl = dict(prog)
    ctrl["arm"] = "equal_dwell_control"
    (out_dir / "turntable_program_deliverable.json").write_text(
        json.dumps(prog, indent=1, default=float), encoding="utf-8")
    (out_dir / "turntable_program_equal_dwell_control.json").write_text(
        json.dumps(ctrl, indent=1, default=float), encoding="utf-8")
    trace = [r for rows in result["rows_by_start"].values() for r in rows]
    (out_dir / "j_trace.json").write_text(
        json.dumps({"rows_by_start": result["rows_by_start"]}, indent=1,
                   default=float), encoding="utf-8")
    z = np.load(npz_src)
    _figure(out_dir,
            f"{args.shape} schedule co-solve, indexed {args.positions} "
            f"positions, budget {args.budget:g} forward-equivalents",
            z["sat_4bpp"], z["phi_4bpp"], z["part_mask"], trace)
    metrics = {
        "J": float(arms["AVG_4bpp"]["J"]), "IoU": float(arms["AVG_4bpp"]["IoU"]),
        "uniform_J": float(arms["AVG_uniform"]["J"]),
        "uniform_IoU": float(arms["AVG_uniform"]["IoU"]),
        "positions_deg": positions, "step_deg": step,
        "dwell_fractions": [1.0 / args.positions] * args.positions,
        "recommended_stop_s": stop_s, "voltage_v": volt,
        "winner_start": result["winner_start"],
    }
    _write_summary(out_dir, args, n_evals=n_evals, metrics=metrics,
                   program_files=["turntable_program_deliverable.json",
                                  "turntable_program_equal_dwell_control.json"],
                   wall_s=time.perf_counter() - t0)


def run_asym_dwell(args: argparse.Namespace, out_dir: Path, raw: Path) -> None:
    _campaign_paths()
    import numpy as np
    import run_dwell_solve as dws
    t0 = time.perf_counter()
    n_evals = budget_to_evals(args.budget)
    watch = DriverWatch(evals_total=3 * n_evals)   # control + joint cold + warm
    real = sys.stdout
    sys.stdout = _TeeWatch(real, watch)            # type: ignore[assignment]
    try:
        dws.OUT = raw                              # redirect, never overwrite
        result = dws.main(args.shape, n_evals)
    finally:
        sys.stdout = real
    shutil.copy2(raw / f"{args.shape}_dwell_maps.npz",
                 out_dir / "schedule_maps.npz")
    for tag in ("deliverable", "equal_dwell_control"):
        src = raw / f"{args.shape}_turntable_{tag}.json"
        prog = json.loads(src.read_text(encoding="utf-8"))
        prog["dopant_map_npz"] = "schedule_maps.npz"
        (out_dir / f"turntable_program_{tag}.json").write_text(
            json.dumps(prog, indent=1, default=float), encoding="utf-8")
    trace = [r for rows in result["solve_rows"].values() for r in rows]
    (out_dir / "j_trace.json").write_text(
        json.dumps({"solve_rows": result["solve_rows"]}, indent=1,
                   default=float), encoding="utf-8")
    arms = result["arms"]
    z = np.load(out_dir / "schedule_maps.npz")
    _figure(out_dir,
            f"{args.shape} schedule co-solve, asymmetric dwell, budget "
            f"{args.budget:g} forward-equivalents per stage",
            z["sat_D_joint_4bpp"], z["phi_D_timeresolved"], z["part_mask"],
            trace)
    deliv = arms.get("D_timeresolved", arms["D_joint_4bpp"])
    ctrl = arms.get("D_timeresolved_equal", arms["D_map_equal_4bpp"])
    metrics = {
        "J": float(deliv["J"]), "IoU": float(deliv["IoU"]),
        "control_J": float(ctrl["J"]), "control_IoU": float(ctrl["IoU"]),
        "dwell_weights": deliv.get("dwell_weights"),
        "candidate_positions_deg": result["candidate_angles_deg"],
        "recommended_stop_s": float(deliv["t_stop_s"]),
        "voltage_v": float(result["voltage_v"]),
        "deliverable_source_arm": result["deliverable_source_arm"],
    }
    _write_summary(out_dir, args, n_evals=n_evals, metrics=metrics,
                   program_files=["turntable_program_deliverable.json",
                                  "turntable_program_equal_dwell_control.json"],
                   wall_s=time.perf_counter() - t0)


def run_sequential(args: argparse.Namespace, out_dir: Path, raw: Path) -> None:
    _campaign_paths()
    import numpy as np
    import run_seq_arms as seq
    t0 = time.perf_counter()
    n_evals = budget_to_evals(args.budget)
    # The sequential driver READS its screen results from its output
    # directory; seed the redirected directory with copies so the campaign
    # originals stay untouched.
    src_dir = REPO / "fgm_solve_campaign" / "out_seq"
    for name in (f"{args.shape}_screen.json", f"{args.shape}_screen2.json"):
        if (src_dir / name).exists():
            shutil.copy2(src_dir / name, raw / name)
    watch = DriverWatch(evals_total=None)          # driver-internal staging
    real = sys.stdout
    sys.stdout = _TeeWatch(real, watch)            # type: ignore[assignment]
    try:
        seq.OUT = raw                              # redirect, never overwrite
        result = seq.main(args.shape, n_evals)
    finally:
        sys.stdout = real
    shutil.copy2(raw / f"{args.shape}_seq_maps.npz",
                 out_dir / "schedule_maps.npz")
    arms = result["arms"]
    deliv_name = ("S_seq_cosolved_4bpp" if "S_seq_cosolved_4bpp" in arms
                  else max((n for n in result.get("programs", {})),
                           default=None, key=lambda n: n or ""))
    program_files: List[str] = []
    if deliv_name and deliv_name in result.get("programs", {}):
        prog = dict(result["programs"][deliv_name])
        prog["dopant_map_npz"] = "schedule_maps.npz"
        (out_dir / "turntable_program_deliverable.json").write_text(
            json.dumps(prog, indent=1, default=float), encoding="utf-8")
        program_files.append("turntable_program_deliverable.json")
    ctrl_arm = arms.get("S_cycled_equal_uniform", {})
    ctrl = _equal_dwell_program(
        args.shape, [float(a) for a in np.arange(8) * 45.0],
        0.0, float(ctrl_arm.get("t_stop_s", TOTAL_S)),
        "sat_S_static_best_uniform", "equal_dwell_control")
    (out_dir / "turntable_program_equal_dwell_control.json").write_text(
        json.dumps(ctrl, indent=1, default=float), encoding="utf-8")
    program_files.append("turntable_program_equal_dwell_control.json")
    (out_dir / "j_trace.json").write_text(
        json.dumps({"screen_best": result.get("screen_best", {})}, indent=1,
                   default=float), encoding="utf-8")
    deliv = arms.get(deliv_name or "", {})
    z = np.load(out_dir / "schedule_maps.npz")
    sat_key = f"sat_{deliv_name}" if f"sat_{deliv_name}" in z else "part_mask"
    phi_key = f"phi_{deliv_name}" if f"phi_{deliv_name}" in z else "part_mask"
    _figure(out_dir,
            f"{args.shape} schedule co-solve, sequential, budget "
            f"{args.budget:g} forward-equivalents",
            z[sat_key], z[phi_key], z["part_mask"], [])
    metrics = {
        "J": float(deliv.get("J", float("nan"))),
        "IoU": float(deliv.get("IoU", float("nan"))),
        "control_J": float(ctrl_arm.get("J", float("nan"))),
        "control_IoU": float(ctrl_arm.get("IoU", float("nan"))),
        "deliverable_arm": deliv_name,
        "sequential_caveat": ("sequential programs are grid-120, part-frame "
                              "results; see run_seq_arms.py"),
    }
    _write_summary(out_dir, args, n_evals=n_evals, metrics=metrics,
                   program_files=program_files,
                   wall_s=time.perf_counter() - t0)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "campaign_raw"
    raw.mkdir(parents=True, exist_ok=True)
    print(f"[solve_schedule] shape={args.shape} mode={args.mode} "
          f"budget={args.budget:g} forward-equivalents "
          f"(= {budget_to_evals(args.budget)} gradient evaluations per stage)",
          flush=True)
    if args.mode == "indexed":
        run_indexed(args, out_dir, raw)
    elif args.mode == "asym_dwell":
        run_asym_dwell(args, out_dir, raw)
    else:
        run_sequential(args, out_dir, raw)
    print(f"[solve_schedule] DONE -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
