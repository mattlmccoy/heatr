"""Build the v2 (unseeded) comparison tables. No transcription: every number is read."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gain_calibration as gc  # noqa: E402
ORDER = ["triangle", "H_shape", "trapezoid", "T_shape", "L_shape",
         "square", "circle", "hexagon"]


def fmt(v: object, n: int = 3) -> str:
    if v is None:
        return "NOT_REACHED"
    return f"{v:.{n}f}" if isinstance(v, float) else str(v)


def pct(new: float | None, ref: float | None) -> str:
    if new is None or ref in (None, 0.0):
        return "n/a"
    return f"{(new - ref) / ref * 100:+.1f} %"


def union_selection(v1: dict, v2: dict) -> tuple[gc.Selection, int]:
    """Apply the SAME hold-out rule to the union of every gain ever solved.

    Zero extra solves: this is only a better use of data already on disk. Selection
    is still on the heating-peak fit metric with the same feasibility rule, so the
    melt-onset hold-out never enters the choice.
    """
    merged: dict[float, dict] = {}
    for src in (v1["arms"], v2["arms"]):
        for k, a in src.items():
            merged[round(float(k), 4)] = a
    cands = [gc.GainCandidate(gain=m,
                              fit_score=a["heating_peak_sigma_T"],
                              holdout_score=a["melt_onset_sigma_T"],
                              melt_reached=bool(a["melt_reached"]))
             for m, a in sorted(merged.items())]
    return gc.select_on_holdout(cands), len(merged)


def main() -> None:
    pairs = []
    for p in sorted(HERE.glob("*.v2.json")):
        shape = p.stem[:-3]
        v1p = HERE / f"{shape}.json"
        if not v1p.exists():
            continue
        v2 = json.loads(p.read_text())
        v1 = json.loads(v1p.read_text())
        if v2.get("complete") and v1.get("complete"):
            pairs.append((shape, v1, v2))
    pairs.sort(key=lambda t: (ORDER.index(t[0]) if t[0] in ORDER else 99, t[0]))

    out: list[str] = []
    out.append("### Table 5. Seeded (v1) versus unseeded (v2) calibration, "
               "melt-onset sigma_T (HOLD-OUT read state), deg C\n")
    out.append("| shape | (a) uniform | (b) best-of-four | (c1) v1 seeded | (c2) v2 unseeded "
               "| c2 vs a | c2 vs c1 | m (v1) | m (v2) | new solves (v2) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    n_beat_b = n_harm_v1 = n_harm_v2 = n_rescued = 0
    for shape, v1, v2 in pairs:
        base = v1["baseline"]
        a = base["melt_onset_sigma_T"] if base["melt_reached"] else None
        barm = v1.get("best_of_four_in_sample")
        b = barm["melt_onset_sigma_T"] if barm else None
        c1 = v1["selected_holdout_melt_onset"]
        c2 = v2["selected_holdout_melt_onset"]
        out.append(
            f"| {shape} | {fmt(a)} | {fmt(b)} | {fmt(c1)} | {fmt(c2)} | {pct(c2, a)} "
            f"| {pct(c2, c1)} | {fmt(v1['selected_gain'], 4)} | {fmt(v2['selected_gain'], 4)} "
            f"| {v2['n_new_solves']} |")
        if c1 is not None and a is not None and c1 > a:
            n_harm_v1 += 1
        if c2 is not None and a is not None and c2 > a:
            n_harm_v2 += 1
        if c1 is not None and c2 is not None and a is not None and c1 > a >= c2:
            n_rescued += 1
        if b is not None and c2 is not None and c2 < b:
            n_beat_b += 1

    out.append(f"\nOver {len(pairs)} shapes: v2 beats best-of-four on the hold-out in "
               f"**{n_beat_b}**; harmful versus uniform at melt-onset, **{n_harm_v1}** for v1 "
               f"and **{n_harm_v2}** for v2; **{n_rescued}** shapes rescued by going unseeded.")

    out.append("\n### Table 5b. Union arm (same hold-out rule, all gains already solved), "
               "melt-onset sigma_T, deg C\n")
    out.append("| shape | (a) uniform | (c1) v1 seeded | (c2) v2 unseeded "
               "| (c3) union | c3 vs a | m (union) | evaluations in union |")
    out.append("|---|---|---|---|---|---|---|")
    n_harm_u = n_beat_b_u = 0
    for shape, v1, v2 in pairs:
        base = v1["baseline"]
        a = base["melt_onset_sigma_T"] if base["melt_reached"] else None
        barm = v1.get("best_of_four_in_sample")
        b = barm["melt_onset_sigma_T"] if barm else None
        sel, n = union_selection(v1, v2)
        c3 = sel.reported_holdout
        out.append(f"| {shape} | {fmt(a)} | {fmt(v1['selected_holdout_melt_onset'])} "
                   f"| {fmt(v2['selected_holdout_melt_onset'])} | {fmt(c3)} | {pct(c3, a)} "
                   f"| {fmt(sel.selected.gain, 4) if sel.selected else 'none'} | {n} |")
        if c3 is not None and a is not None and c3 > a:
            n_harm_u += 1
        if b is not None and c3 is not None and c3 < b:
            n_beat_b_u += 1
    out.append(f"\nUnion arm over {len(pairs)} shapes: harmful versus uniform at melt-onset in "
               f"**{n_harm_u}**; beats best-of-four on the hold-out in **{n_beat_b_u}**.")

    out.append("\n### Table 6. v2 search diagnostics and gates\n")
    out.append("| shape | selected m | evaluations (new / cached) | stop reason | status "
               "| infeasible gains | P_abs at melt, W/m (uniform / v2) | residual frac (v2) "
               "| dT clip frac (v2) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for shape, v1, v2 in pairs:
        sel = v2["selected_gain"]
        arm = v2["arms"].get(f"{sel:.4f}") if sel is not None else None
        pa = v1["baseline"].get("power_melt_W_per_m")
        pc = arm.get("power_melt_W_per_m") if arm else None
        rc = arm.get("residual_frac_melt") if arm else None
        cl = arm.get("frac_dT_clipped_final") if arm else None
        ninf = len(v2["infeasible_gains"])
        inf = (f"{ninf} of {v2['n_evaluations']}" if ninf else "none")
        out.append(
            f"| {shape} | {fmt(sel, 4)} | {v2['n_new_solves']} / {v2['n_cache_hits']} "
            f"| {v2['stop_reason']} | {v2['status']} | {inf} "
            f"| {fmt(pa, 1)} / {fmt(pc, 1)} "
            f"| {fmt(rc, 6) if rc is not None else 'n/a'} "
            f"| {fmt(cl, 4) if cl is not None else 'n/a'} |")

    out.append("\n### Table 7. v2 heating-peak sigma_T (the FIT read state), deg C\n")
    out.append("| shape | (a) uniform | (c1) v1 seeded | (c2) v2 unseeded | c2 vs a |")
    out.append("|---|---|---|---|---|")
    for shape, v1, v2 in pairs:
        a = v1["baseline"]["heating_peak_sigma_T"]
        out.append(f"| {shape} | {fmt(a)} | {fmt(v1['selected_fit_heating_peak'])} "
                   f"| {fmt(v2['selected_fit_heating_peak'])} "
                   f"| {pct(v2['selected_fit_heating_peak'], a)} |")

    txt = "\n".join(out) + "\n"
    (HERE / "tables_v2.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
