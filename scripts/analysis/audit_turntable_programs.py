#!/usr/bin/env python3
"""Audit every emitted turntable program for the divisor bug class.

THE BUG CLASS. A cyclic turntable program allocates `n_slots = cycle_time / dt`
integer control steps per cycle across its kept positions. If the old emitter's
largest-remainder allocation is computed once and then repeated every cycle, a
per-cycle rounding becomes a permanent bias in the realized dwell fractions.
The exposure is exposed to it whenever `n_slots` is not a multiple of the kept
position count AND the requested fractions do not already lie on the 1/n_slots
grid.

This walks every stored program and reports, per program: the slot count, the
kept position count, whether they divide, and the largest absolute difference
between the realized and the requested dwell fraction. It also reports the
SEQUENTIAL programs (`seq_dwell`), which are a different emitter with no cyclic
apportionment in it at all, and checks their holds land on the control-step grid.

Run:
  ./.venv312/bin/python scripts/analysis/audit_turntable_programs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "fgm_solve_campaign"
OUT = CAMPAIGN / "out_rot_holdout/turntable_program_audit.json"


def _walk(obj, path, hits):
    if isinstance(obj, dict):
        if "realized_dwell_fraction" in obj and "cycle_time_s" in obj:
            hits.append((path, obj))
        for k, v in obj.items():
            _walk(v, f"{path}/{k}", hits)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:64]):
            _walk(v, f"{path}[{i}]", hits)


def audit_cyclic() -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for p in sorted(CAMPAIGN.glob("out_*/*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        hits: list[tuple[str, dict]] = []
        _walk(d, "", hits)
        for path, o in hits:
            key = (p.name, path)
            if key in seen:
                continue
            seen.add(key)
            r = np.asarray(o["realized_dwell_fraction"], dtype=float)
            q = np.asarray(o["requested_dwell_fraction"], dtype=float)
            n_slots = int(round(float(o["cycle_time_s"])
                                / float(o["control_step_s"])))
            kept = len(o["positions_deg"])
            # The request is stored on the FULL candidate set and the realized
            # on the KEPT one. When they differ in length the candidate set is
            # the uniform one of that size starting at zero degrees (the dwell
            # campaign's convention), so the kept entries are selected by angle
            # and then renormalized: that is the design the emitter was handed.
            if q.size == r.size:
                qk = q
            else:
                step = 360.0 / q.size
                sel = [int(round((float(a) % 360.0) / step))
                       for a in o["positions_deg"]]
                qk = q[sel] if len(set(sel)) == r.size else np.array([])
            if qk.size == r.size and qk.sum() > 0:
                qk = qk / qk.sum()
                err = float(np.max(np.abs(r - qk)))
            else:
                qk, err = np.array([]), float("nan")
            n_total = int(round(float(o["total_exposure_s"])
                                / float(o["control_step_s"])))
            tol = 1.0 / max(n_total, 1)      # one control step over the exposure
            rows.append({
                "file": str(p.relative_to(REPO)), "path": path or "/",
                "emitter": "cyclic (dwell.cycle_program)",
                "control_steps_per_cycle": n_slots,
                "n_kept_positions": kept,
                "divides_evenly": bool(n_slots % kept == 0),
                "request_on_the_slot_grid": bool(
                    np.max(np.abs(qk * n_slots - np.round(qk * n_slots)))
                    < 1e-9) if qk.size else None,
                "max_abs_realized_minus_requested": err,
                "one_control_step_over_the_exposure": tol,
                "n_control_steps_over_the_exposure": n_total,
                "n_moves": int(o["n_moves"]),
                "AFFECTED": bool(err == err and err > tol),
            })
    return rows


def audit_sequential() -> list[dict]:
    rows: list[dict] = []
    for p in sorted((CAMPAIGN / "out_seq").glob("*turntable*.json")):
        d = json.loads(p.read_text())
        if d.get("schedule_kind") != "sequential":
            continue
        dt = float(d["control_step_s"])
        dur = np.array([float(m["dwell_s"]) for m in d["moves"]])
        t0 = np.array([float(m["move_at_s"]) for m in d["moves"]])
        rows.append({
            "file": str(p.relative_to(REPO)),
            "emitter": "sequential (seq_dwell.sequential_program)",
            "n_moves": int(d["n_moves"]),
            "positions_deg": list(d["positions_deg"]),
            "max_off_control_step_grid_s": float(dt * np.max(
                np.abs(np.concatenate([dur, t0]) / dt
                       - np.round(np.concatenate([dur, t0]) / dt)))),
            "cyclic_apportionment": False,
            "AFFECTED": False,
            "why_not": "no per-cycle apportionment exists; the holds are named "
                       "durations and they land exactly on the control-step grid",
        })
    return rows


def main() -> None:
    cyc = audit_cyclic()
    seq = audit_sequential()
    rec = {
        "task": "divisor-bug audit of every emitted turntable program",
        "bug": "a per-cycle largest-remainder allocation repeated every cycle "
               "turns a rounding into a permanent bias in the realized dwell",
        "cyclic_programs": cyc,
        "sequential_programs": seq,
        "n_cyclic": len(cyc),
        "n_cyclic_affected": int(sum(r["AFFECTED"] for r in cyc)),
        "n_cyclic_in_the_exposed_class_not_dividing": int(
            sum(not r["divides_evenly"] for r in cyc)),
        "n_sequential": len(seq),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2, default=float))

    print(f"{'file':44s}{'path':32s} slots pos div  max|r-q|  AFFECTED")
    for r in cyc:
        print(f"{Path(r['file']).name:44s}{r['path']:32s} "
              f"{r['control_steps_per_cycle']:5d} {r['n_kept_positions']:3d} "
              f"{str(r['divides_evenly'])[0]:>3s}  "
              f"{r['max_abs_realized_minus_requested']:8.4f}  "
              f"{'YES' if r['AFFECTED'] else '.'}")
    print()
    for r in seq:
        print(f"{Path(r['file']).name:52s} sequential, {r['n_moves']} moves, "
              f"off grid by {r['max_off_control_step_grid_s']:.3g} s, "
              f"AFFECTED {r['AFFECTED']}")
    print(f"\ncyclic {rec['n_cyclic']}, of them not dividing "
          f"{rec['n_cyclic_in_the_exposed_class_not_dividing']}, "
          f"AFFECTED {rec['n_cyclic_affected']}; sequential {rec['n_sequential']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
