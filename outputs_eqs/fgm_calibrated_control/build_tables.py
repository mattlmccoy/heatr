"""Build the report tables from the per-shape JSON artifacts. No transcription."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP = {"cache_verification", "summary"}


def fmt(v: object, n: int = 3) -> str:
    if v is None:
        return "NOT_REACHED"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def pct(new: float | None, ref: float | None) -> str:
    if new is None or ref in (None, 0.0):
        return "n/a"
    return f"{(new - ref) / ref * 100:+.1f} %"


def load_all() -> list[dict]:
    out = []
    for p in sorted(HERE.glob("*.json")):
        if p.stem in SKIP:
            continue
        d = json.loads(p.read_text())
        if d.get("complete"):
            out.append(d)
    return out


def main() -> None:
    recs = load_all()
    order = ["square", "circle", "hexagon", "triangle", "L_shape"]
    recs.sort(key=lambda d: (order.index(d["shape"]) if d["shape"] in order
                             else 99, d["shape"]))

    lines: list[str] = []

    lines.append("### Table 1. Melt-onset sigma_T (hold-out read state), deg C\n")
    lines.append("| shape | (a) uniform | (b) best-of-four in-sample | (c) calibrated gain "
                 "| c vs a | c vs b | b gain m | c gain m | new solves |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for d in recs:
        a = d["baseline"]["melt_onset_sigma_T"] if d["baseline"]["melt_reached"] else None
        b_arm = d.get("best_of_four_in_sample")
        b = b_arm["melt_onset_sigma_T"] if b_arm else None
        c = d["selected_holdout_melt_onset"]
        lines.append(
            f"| {d['shape']} | {fmt(a)} | {fmt(b)} | {fmt(c)} | {pct(c, a)} | {pct(c, b)} "
            f"| {fmt(b_arm['m'], 2) if b_arm else 'none'} | {fmt(d['selected_gain'], 4)} "
            f"| {d['n_new_solves']} |"
        )

    lines.append("\n### Table 2. Heating-peak sigma_T (fit read state), deg C\n")
    lines.append("| shape | (a) uniform | (b) best-of-four in-sample | (c) calibrated gain "
                 "| c vs a | c vs b |")
    lines.append("|---|---|---|---|---|---|")
    for d in recs:
        a = d["baseline"]["heating_peak_sigma_T"]
        b_arm = d.get("best_of_four_in_sample")
        b = b_arm["heating_peak_sigma_T"] if b_arm else None
        c = d["selected_fit_heating_peak"]
        lines.append(f"| {d['shape']} | {fmt(a)} | {fmt(b)} | {fmt(c)} "
                     f"| {pct(c, a)} | {pct(c, b)} |")

    lines.append("\n### Table 3. Search diagnostics and gates\n")
    lines.append("| shape | gains evaluated (new) | stop reason | status | infeasible gains "
                 "| P_abs at melt, W/m (a / c) | energy residual frac (c) | dT clip frac (c) "
                 "| wall s (new solves) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for d in recs:
        sel = d["selected_gain"]
        arm = d["arms"].get(f"{sel:.4f}") if sel is not None else None
        pa = d["baseline"].get("power_melt_W_per_m")
        pc = arm.get("power_melt_W_per_m") if arm else None
        rc = arm.get("residual_frac_melt") if arm else None
        cl = arm.get("frac_dT_clipped_final") if arm else None
        newg = ", ".join(f"{g:.4f}" for g in d["new_gains"]) or "none"
        inf = ", ".join(f"{g:.4f}" for g in d["infeasible_gains"]) or "none"
        wall = sum(d["timings_s"].values())
        lines.append(
            f"| {d['shape']} | {newg} | {d['stop_reason']} | {d['status']} | {inf} "
            f"| {fmt(pa, 1)} / {fmt(pc, 1)} | {fmt(rc, 6) if rc is not None else 'n/a'} "
            f"| {fmt(cl, 4) if cl is not None else 'n/a'} | {wall:.0f} |"
        )

    lines.append("\n### Table 4. Verdict per shape (pre-registered criteria)\n")
    lines.append("| shape | (b) harmful vs uniform? | (c) harmful vs uniform? "
                 "| calibration beats (b) on hold-out? | rescued? |")
    lines.append("|---|---|---|---|---|")
    n_beat = n_resc = n_harm_b = n_harm_c = 0
    for d in recs:
        a = d["baseline"]["melt_onset_sigma_T"] if d["baseline"]["melt_reached"] else None
        b_arm = d.get("best_of_four_in_sample")
        b = b_arm["melt_onset_sigma_T"] if b_arm else None
        c = d["selected_holdout_melt_onset"]
        hb = "n/a (no arm melts)" if b is None else ("YES" if b > a else "no")
        hc = "NOT_REACHED" if c is None else ("YES" if c > a else "no")
        beat = "n/a" if (b is None or c is None) else ("YES" if c < b else "no")
        resc = "n/a"
        if b is not None and c is not None and a is not None:
            resc = "YES" if (b > a and c <= a) else "no"
        n_harm_b += 1 if (b is not None and b > a) else 0
        n_harm_c += 1 if (c is not None and c > a) else 0
        n_beat += 1 if beat == "YES" else 0
        n_resc += 1 if resc == "YES" else 0
        lines.append(f"| {d['shape']} | {hb} | {hc} | {beat} | {resc} |")
    lines.append(f"\nTotals over {len(recs)} shapes: calibration beats best-of-four on the "
                 f"hold-out in **{n_beat}**; shapes harmful versus uniform at melt-onset: "
                 f"**{n_harm_b}** for (b), **{n_harm_c}** for (c); rescued: **{n_resc}**.")

    txt = "\n".join(lines) + "\n"
    (HERE / "tables.md").write_text(txt)
    sys.stdout.write(txt)


if __name__ == "__main__":
    main()
