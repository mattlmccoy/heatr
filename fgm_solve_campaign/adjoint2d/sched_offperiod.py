"""Diagnosis of the OFF period found in the cross schedule.

WHAT IS BEING EXPLAINED. The warm-started cross deliverable commands full power
for the first eleven segments, then drops to zero for three consecutive
segments, then returns to full power for the last one. The objective it is
minimizing penalizes melted BED cells exactly as hard as unmelted PART cells,
and the library baseline it starts from has 17 percent bed growth. The
hypothesis this script tests is DIFFERENTIAL COOLING: with the generator off,
the melted powder outside the part falls back below the melt threshold FASTER
than the part interior does, so the OFF period is a selective eraser. The final
burst then re-melts the part without putting the bed back.

The test is a measurement, not a story. Along the deliverable trajectory, per
outer step, this records

    part_under_melt_pct    part cells with melt fraction below 0.5
    bed_melt_pct_of_part   bed cells above 0.5, as a percent of part cell count
    mean part temperature and mean temperature of the bed ring that melted

and reports the change in each ACROSS THE OFF PERIOD. If bed melt falls much
faster than part melt rises, the mechanism is confirmed. If both fall together,
it is not differential and the honest answer is that the OFF period simply
lowers the dose.

Output: <shape>_offperiod.json and fig_sched_offperiod_<shape>.png, a
temperature field pair at the first and last step of the OFF period with the
melt front and the nominal outline drawn on both.

Run:
  ./.venv312/bin/python -m adjoint2d.sched_offperiod <shape> <outdir> <figs> [arm] [stem]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import schedule as sch, shape_objective as so
from .library_solve import shape_config
from .make_figures import crop_box
from .pins import build_case, load_cfg
from .sched_solve import run_forward

DPI = 180
OFF_LEVEL = 1e-6


def off_runs(p_seg, window: int, n_seg: int) -> list[tuple[int, int]]:
    """Contiguous outer-step ranges where the commanded level is zero."""
    p = np.asarray(p_seg, dtype=float).ravel()
    bounds = sch.segment_bounds(window, n_seg)
    runs, cur = [], None
    for k, (lo, hi) in enumerate(bounds):
        if p[k] <= OFF_LEVEL:
            cur = (cur[0], hi) if cur else (lo, hi)
        elif cur:
            runs.append(cur); cur = None
    if cur:
        runs.append(cur)
    return runs


def main(shape: str, outdir: str, figdir: str, arm: str = "WARM_CO_4bpp",
         stem: str = "warm") -> dict:
    out, figs = Path(outdir).resolve(), Path(figdir).resolve()
    figs.mkdir(parents=True, exist_ok=True)
    res = json.loads((out / f"{shape}_{stem}.json").read_text())
    m = res["arms"][arm]
    case = build_case(load_cfg(shape_config(shape)))
    pm = case.part_mask
    maps = np.load(out / f"{shape}_{stem}_maps.npz")
    s = np.asarray(maps[f"map_{arm}"], dtype=float)
    p = np.asarray(m["p_seg"], dtype=float)
    n_seg, horizon = res["n_seg"], res["horizon_steps"]
    window = res["schedule_window_steps"]
    dt = res["dt_s"]

    runs = off_runs(p, window, n_seg)
    doc = {"shape": shape, "arm": arm, "p_seg": [float(v) for v in p],
           "schedule_window_steps": window, "horizon_steps": horizon,
           "dt_s": dt, "t_stop_index": m["t_stop_index"], "t_stop_s": m["t_stop_s"],
           "off_runs_steps": [list(r) for r in runs],
           "off_runs_s": [[r[0] * dt, r[1] * dt] for r in runs],
           "n_off_runs": len(runs)}
    if not runs:
        doc["mechanism"] = "NO OFF PERIOD"
        doc["verdict"] = "NO OFF PERIOD: the optimized schedule never commands zero"
        (out / f"{shape}_{stem}_{arm}_offperiod.json").write_text(
            json.dumps(doc, indent=2))
        print(f"[{shape}/{arm}] {doc['verdict']}")
        return doc

    tr = run_forward(case, s, p, n_seg, horizon, window=window)
    n = tr.n_outer
    under, grow, tp, tb = [], [], [], []
    # the bed ring that ever melted, fixed once so the two ends are comparable
    ever = np.zeros_like(pm, dtype=bool)
    fields = {}
    for it in range(n):
        T = tr.T_at_end(it)
        phi, _ = so.phi_field(T, case)
        melted = phi >= so.MELT_LEVEL
        ever |= melted & ~pm
        under.append(100.0 * np.sum(pm & ~melted) / max(pm.sum(), 1))
        grow.append(100.0 * np.sum(melted & ~pm) / max(pm.sum(), 1))
        tp.append(float(np.mean(T[pm])))
        tb.append(T)
    tb_ring = [float(np.mean(t[ever])) if ever.any() else float("nan") for t in tb]
    # The LONGEST contiguous off run, not first-start to last-end: a schedule
    # with two separate off runs would otherwise have on stretches counted
    # inside the labelled off period.
    longest = max(runs, key=lambda r: r[1] - r[0])
    lo, hi = longest[0], min(longest[1], n - 1)
    fields = {"start": tr.T_at_end(lo), "end": tr.T_at_end(hi),
              "stop": tr.T_at_end(m["t_stop_index"])}

    doc["across_off_period"] = {
        "step_lo": int(lo), "step_hi": int(hi),
        "t_lo_s": lo * dt, "t_hi_s": hi * dt, "duration_s": (hi - lo) * dt,
        "bed_melt_pct_start": grow[lo], "bed_melt_pct_end": grow[hi],
        "bed_melt_pct_change": grow[hi] - grow[lo],
        "part_under_melt_pct_start": under[lo], "part_under_melt_pct_end": under[hi],
        "part_under_melt_pct_change": under[hi] - under[lo],
        "mean_T_part_start_c": tp[lo], "mean_T_part_end_c": tp[hi],
        "mean_T_melted_bed_ring_start_c": tb_ring[lo],
        "mean_T_melted_bed_ring_end_c": tb_ring[hi],
        "part_cooling_c": tp[lo] - tp[hi],
        "bed_ring_cooling_c": tb_ring[lo] - tb_ring[hi],
    }
    a = doc["across_off_period"]
    bed_removed = -a["bed_melt_pct_change"]
    part_lost = a["part_under_melt_pct_change"]
    a["selectivity_bed_removed_per_part_lost"] = (
        bed_removed / part_lost if abs(part_lost) > 1e-9 else float("inf"))
    a["differential_cooling"] = bool(bed_removed > 1.15 * max(part_lost, 0.0))

    # The OFF period is only half the cycle. The leg AFTER it is where the
    # dopant can act, because the part is doped and the bed is not, so the
    # re-heat has a selectivity the passive cooling cannot have. Both legs are
    # measured, and the verdict names whichever one is actually selective.
    stop_i = int(m["t_stop_index"])
    reheat = None
    if stop_i > hi:
        d_bed = grow[stop_i] - grow[hi]
        d_part = under[hi] - under[stop_i]
        reheat = {"step_lo": int(hi), "step_hi": stop_i,
                  "t_lo_s": hi * dt, "t_hi_s": stop_i * dt,
                  "duration_s": (stop_i - hi) * dt,
                  "bed_melt_pct_change": d_bed,
                  "part_under_melt_pct_change": -d_part,
                  "selectivity_part_recovered_per_bed_added":
                      (d_part / d_bed) if abs(d_bed) > 1e-9 else float("inf")}
    doc["reheat_leg"] = reheat
    doc["on_leg"] = {"step_lo": 0, "step_hi": int(lo), "t_hi_s": lo * dt,
                     "bed_melt_pct_at_end": grow[lo],
                     "part_under_melt_pct_at_end": under[lo]}
    # Guard against a spurious mechanism reading. If nothing had melted yet
    # when the generator went off, there was nothing to erase and the off
    # period is a DELAY, not an erase step. Reading a re-heat selectivity in
    # that case would be measuring the first melt, not a recovery.
    a["anything_to_erase"] = bool(grow[lo] > 1.0 or under[lo] < 99.0)
    if not a["anything_to_erase"]:
        doc["mechanism"] = "DELAY, NOT AN ERASE CYCLE"
        doc["verdict"] = (
            "the off period falls BEFORE any melting has happened "
            f"(bed melt {grow[lo]:.1f}%, part unmelted {under[lo]:.1f}% at its "
            "start), so it removes nothing and only postpones the process; no "
            "selectivity can be read from it")
    elif a["differential_cooling"]:
        doc["mechanism"] = "DIFFERENTIAL COOLING"
        doc["verdict"] = ("the off period itself is selective: it removes more "
                          "bed melt than it costs in part melt")
    elif reheat and reheat["selectivity_part_recovered_per_bed_added"] > 1.15:
        doc["mechanism"] = "NON SELECTIVE COOLING FOLLOWED BY SELECTIVE RE-HEAT"
        doc["verdict"] = (
            "NOT differential cooling. Across the off period bed melt and part "
            "melt fall together at "
            f"{a['selectivity_bed_removed_per_part_lost']:.2f} to 1, which is "
            "not selective. The selectivity is in the RE-HEAT leg, where the "
            "part recovers "
            f"{reheat['selectivity_part_recovered_per_bed_added']:.2f} points "
            "of melt for every point the bed gains, because the dopant is in "
            "the part and the bed only heats by conduction. The cycle is a "
            "melt, erase and selectively re-melt sequence.")
    else:
        doc["mechanism"] = "NONE FOUND"
        doc["verdict"] = ("neither the off period nor the re-heat is selective; "
                          "the off period only lowers the delivered dose")
    doc["final_at_stop"] = {"IoU": m["IoU"], "bed_melt_pct": grow[m["t_stop_index"]],
                            "part_under_melt_pct": under[m["t_stop_index"]]}
    doc["curves"] = {"part_under_melt_pct": under, "bed_melt_pct_of_part": grow,
                     "mean_T_part_c": tp, "mean_T_melted_bed_ring_c": tb_ring}
    (out / f"{shape}_{stem}_{arm}_offperiod.json").write_text(
        json.dumps(doc, indent=2, default=float))

    # ---- figure: temperature field pair + the two curves --------------------
    extent = (case.x[0] * 1e3, case.x[-1] * 1e3, case.y[0] * 1e3, case.y[-1] * 1e3)
    box = crop_box(pm, case.x, case.y)
    xs = np.linspace(extent[0], extent[1], pm.shape[1])
    ys = np.linspace(extent[2], extent[3], pm.shape[0])
    vmax = max(float(fields["start"].max()), float(fields["end"].max()))
    fig = plt.figure(figsize=(13.8, 4.9), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.30)
    for j, (key, when) in enumerate((("start", lo), ("end", hi))):
        ax = fig.add_subplot(gs[0, j])
        T = fields[key]
        im = ax.imshow(T, origin="lower", extent=extent, vmin=case.pins.ambient_c,
                       vmax=vmax, cmap="magma", interpolation="bilinear")
        phi, _ = so.phi_field(T, case)
        ax.contour(xs, ys, pm.astype(float), levels=[0.5], colors="#00e5ff", linewidths=1.4)
        ax.contour(xs, ys, phi, levels=[0.5], colors="#ffffff", linewidths=1.1, linestyles="--")
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
        ax.set_xticks([]); ax.set_yticks([])
        lab = "generator OFF begins" if key == "start" else "generator OFF ends"
        ax.set_title(f"{lab}, t = {when * dt:.0f} s\n"
                     f"bed melt {grow[when]:.1f}% of part, part unmelted {under[when]:.1f}%",
                     fontsize=8, linespacing=1.4)
        fig.colorbar(im, ax=ax, fraction=0.042, pad=0.012, label="temperature, C")
    ax = fig.add_subplot(gs[0, 2])
    t = np.arange(n) * dt
    ax.plot(t, grow, color="#d62728", lw=1.5, label="bed melt, percent of part cells")
    ax.plot(t, under, color="#1f77b4", lw=1.5, label="part unmelted, percent")
    ax.axvspan(lo * dt, hi * dt, color="#888888", alpha=0.22, lw=0)
    ax.axvline(m["t_stop_s"], color="#000000", lw=1.4)
    ax.text(m["t_stop_s"], ax.get_ylim()[1], "  stop", fontsize=7, va="top")
    ax.text((lo + hi) * dt / 2, ax.get_ylim()[1] * 0.62, "generator\nOFF",
            fontsize=7.5, ha="center", va="top")
    ax.set_xlabel("time, s", fontsize=8)
    ax.set_ylabel("percent of part cells", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7); ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.95)
    rt = ("" if reheat is None or not a["anything_to_erase"] else
          f"\nre-heat: {reheat['selectivity_part_recovered_per_bed_added']:.2f} part "
          f"points recovered per bed point added")
    ax.set_title(f"off period: bed melt {a['bed_melt_pct_change']:+.1f} points,\n"
                 f"part unmelted {a['part_under_melt_pct_change']:+.1f} points "
                 f"({a['selectivity_bed_removed_per_part_lost']:.2f} to 1, not "
                 f"selective){rt}" if a["anything_to_erase"] else
                 f"nothing had melted when the generator went off:\n"
                 f"a delay, not an erase cycle", fontsize=7.4, linespacing=1.45)
    fig.suptitle(f"{shape}, arm {arm}: mechanism = {doc['mechanism']}", fontsize=9.2)

    fig.subplots_adjust(left=0.02, right=0.965, top=0.74, bottom=0.13)
    q = figs / f"fig_sched_offperiod_{shape}_{arm}.png"
    fig.savefig(q, dpi=DPI); plt.close(fig)
    print(f"[{shape}/{arm}] off {lo}-{hi} ({(hi-lo)*dt:.0f} s) bed "
          f"{grow[lo]:.1f} -> {grow[hi]:.1f}%, part unmelted {under[lo]:.1f} -> "
          f"{under[hi]:.1f}%, part cools {a['part_cooling_c']:.1f} C, bed ring cools "
          f"{a['bed_ring_cooling_c']:.1f} C\n  MECHANISM {doc['mechanism']}\n"
          f"  {doc['verdict']}\n  {q}")
    return doc


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         sys.argv[4] if len(sys.argv) > 4 else "WARM_CO_4bpp",
         sys.argv[5] if len(sys.argv) > 5 else "warm")
