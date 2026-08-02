#!/usr/bin/env python3
"""Tables for the geometry generalization pass -> out_intake/_tables.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

from adjoint2d import library_solve as lib          # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_intake"
LIBJ = REPO / "fgm_solve_campaign/out_lib"


def static_class(shape: str) -> str:
    p = LIBJ / f"{shape}.json"
    if not p.exists():
        return "-"
    j = json.loads(p.read_text())
    return str(j.get("verdict", {}).get("class", "-"))


def main() -> None:
    rows = []
    for s in lib.SHAPES:
        p = OUT / f"{s}_intake.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))

    L: list[str] = []
    L.append("# Geometry intake, all tables\n")
    L.append("Grid 120 throughout. Anisotropy is the occupancy-weighted azimuthal")
    L.append("anisotropy of `adjoint2d.geometry_actuator`, measured against the")
    L.append("24-angle reference rotation group, uniform dopant map, electrical")
    L.append("state B.\n")

    L.append("\n## Table A. Intake fidelity against the stored library configurations\n")
    L.append("| shape | part mask matches | cells differing | voltage calibrated (V) "
             "| voltage stored (V) | relative error | part cells | raster minus "
             "area-fill area |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['shape']} | {r['part_mask_matches_stored_config']} | "
                 f"{r['n_cells_differing']} | {r['voltage_calibrated_v']:.2f} | "
                 f"{r['voltage_stored_v']:.2f} | {r['voltage_rel_error']*100:.6f} pct | "
                 f"{r['intake']['n_part_cells']} | "
                 f"{r['raster_vs_area']['area_rel_delta']*100:+.2f} pct |")

    L.append("\n## Table B. Symmetry detection\n")
    L.append("| shape | detected order | point group | mirror axes | order the "
             "library defines | agrees |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        exp = r.get("expected_order")
        got = r["symmetry"]["rotational_order"]
        ok = "-" if exp is None else ("yes" if int(exp) == int(got) else "NO")
        L.append(f"| {r['shape']} | {got} | {r['symmetry']['point_group']} | "
                 f"{len(r['symmetry']['mirror_axes_deg'])} | "
                 f"{'-' if exp is None else exp} | {ok} |")

    L.append("\n## Table C. Anisotropy spectrum and actuator recommendation\n")
    L.append("| shape | static | continuous | best indexing | recommended mode | "
             "residual | reduction | class | measured rotation outcome | static-solve "
             "class (out_lib) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        a = r["anisotropy"]
        idx = {k: v for k, v in a.items() if k.startswith("index")}
        best_idx = (f"{min(idx, key=lambda k: idx[k])} {min(idx.values()):.4f}"
                    if idx else "-")
        rec = r["recommendation"]
        L.append(f"| {r['shape']} | {a['static']:.4f} | {a['continuous']:.4f} | "
                 f"{best_idx} | {rec['mode']} | {rec['residual_anisotropy']:.4f} | "
                 f"{rec['reduction_factor']:.2f} | {rec['actuator_class']} | "
                 f"{r.get('known_rotation_outcome') or '-'} | "
                 f"{static_class(r['shape'])} |")

    L.append("\n## Table D. Classifier against the five MEASURED rotation outcomes\n")
    L.append("| shape | residual | predicted class | predicted rotation helps | "
             "measured outcome | agrees |")
    L.append("|---|---|---|---|---|---|")
    n_ok = n_tot = 0
    for r in rows:
        k = r.get("known_rotation_outcome")
        if not k:
            continue
        rec = r["recommendation"]
        pred_helps = rec["actuator_class"] in ("MODE_SUFFICES", "MAP_PLUS_MODE")
        ok = pred_helps == (k == "rotation_wins")
        n_tot += 1
        n_ok += int(ok)
        L.append(f"| {r['shape']} | {rec['residual_anisotropy']:.4f} | "
                 f"{rec['actuator_class']} | {pred_helps} | {k} | "
                 f"{'yes' if ok else 'NO'} |")
    L.append(f"\nAgreement: **{n_ok} of {n_tot}**. These five are the points the "
             "bands were CALIBRATED on, so this is threshold reproduction, not "
             "out-of-sample validation.\n")

    L.append("\n## Table E. Novel geometries, not in the library\n")
    L.append("| shape | vertices | part cells | calibrated V | order | mirrors | "
             "static | best mode | residual | class |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(OUT.glob("*_novel.json")):
        r = json.loads(p.read_text())
        rec = r["recommendation"]
        L.append(f"| {r['shape']} | {r['intake']['n_vertices']} | "
                 f"{r['intake']['n_part_cells']} | {r['intake']['voltage_v']:.1f} | "
                 f"{r['symmetry']['rotational_order']} | "
                 f"{len(r['symmetry']['mirror_axes_deg'])} | "
                 f"{r['anisotropy']['static']:.4f} | {rec['mode']} | "
                 f"{rec['residual_anisotropy']:.4f} | {rec['actuator_class']} |")

    for p in sorted(OUT.glob("*_novel.json")):
        r = json.loads(p.read_text())
        if "arms" not in r or not r["arms"]:
            continue
        L.append(f"\n## Table F. Solve arms, {r['shape']} (grid 120, area-fill chi)\n")
        L.append("| arm | J | J vs raster chi | IoU | IoU_area | growth pct | "
                 "under-melt pct | stop s | horizon | P_abs W/m | max T C | "
                 "energy residual pct |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, m in r["arms"].items():
            L.append(f"| {name} | {m['J']:.2f} | {m['J_raster_chi']:.2f} | "
                     f"{m['IoU']:.4f} | {m['IoU_area']:.4f} | "
                     f"{m['bed_melt_pct_of_part']:.2f} | "
                     f"{m['part_under_melt_pct']:.2f} | {m['t_stop_s']:.1f} | "
                     f"{'HORIZON' if m['t_stop_at_horizon'] else '-'} | "
                     f"{m['P_abs_W_per_m']:.1f} | {m['max_T_at_stop_c']:.1f} | "
                     f"{m['energy_gate']['rel_residual_at_index']*100:.2f} |")

    (OUT / "_tables.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT / '_tables.md'} ({len(L)} lines); classifier {n_ok}/{n_tot}")


if __name__ == "__main__":
    main()
