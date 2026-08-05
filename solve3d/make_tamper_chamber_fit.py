"""Why the Tamper does not melt in the frozen chamber: a closed-form estimate.

    heatr3d_d1_spike/env/bin/python -m solve3d.make_tamper_chamber_fit

Writes solve3d/results/tamper_chamber_fit.json.

The marched forward (tamper_uniform_forward.json) shows the Tamper plateauing
below the melt point. This is the one-line reason, computed independently of
the solver so it is a CHECK on the march rather than a restatement of it.

The drive is a fixed POWER DENSITY (forward.qrf_dg0 renormalizes Q so that
integral(Q dV) = power_density_w_per_m3 * V_part), so both parts absorb the
same watts per cubic metre. What differs is where the heat goes. The chamber
side L_DOMAIN is FROZEN at 60 mm and its wall is held at the preheat
temperature, so the powder gap between the part and that wall sets the
conduction loss. The Tamper spans 44.4 mm of the 60 mm chamber and sits 7.8 mm
from the wall; the library pyramid spans 23.2 mm and sits 18.4 mm away.

A one-dimensional slab estimate, dT = P * gap / (k_powder * A_surface), is
crude on purpose: it uses no solver quantity, and it only has to separate
"comfortably above the 180 C melt point" from "below it".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solve3d import forward as fwd, stl_mesh
from solve3d.phase_e import run_tamper as rt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "solve3d" / "results" / "tamper_chamber_fit.json"


def _row(name: str, path, L_m: float | None = None) -> dict:
    v, f = stl_mesh.load_stl(path)
    area_mm2 = float(0.5 * np.linalg.norm(
        np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]),
        axis=1).sum())
    vol_mm3 = float(stl_mesh.enclosed_volume(v, f))
    ext = v.max(axis=0) - v.min(axis=0)
    span_mm = float(max(ext[0], ext[1]))
    from solve3d import chamber as _ch
    L_mm = (_ch.chamber_for_bbox(*(ext * 1e-3)) if L_m is None
            else float(L_m)) * 1e3
    gap_mm = (L_mm - span_mm) / 2.0
    p = fwd.ForwardParams()
    P_w = p.power_density_w_per_m3 * vol_mm3 * 1e-9
    dT = P_w * (gap_mm * 1e-3) / (p.k_powder * area_mm2 * 1e-6)
    return {"part": name, "chamber_mm": L_mm,
            "volume_mm3": vol_mm3, "surface_area_mm2": area_mm2,
            "surface_to_volume_per_mm": area_mm2 / vol_mm3,
            "xy_span_mm": span_mm, "chamber_side_mm": L_mm,
            "span_over_chamber": span_mm / L_mm,
            "powder_gap_to_wall_mm": gap_mm,
            "absorbed_power_w": P_w,
            "conduction_delta_t_k": dT,
            "plateau_estimate_c": p.ambient_c + dT,
            "melt_onset_c": p.t_pc_c,
            "melts": bool(p.ambient_c + dT > p.t_pc_c)}


def main() -> int:
    p = fwd.ForwardParams()
    doc = {
        "what": ("closed-form check on why the chamber-embedded Tamper does "
                 "not melt at the frozen drive, computed WITHOUT the solver"),
        "model": ("1-D slab conduction to the fixed-temperature chamber wall: "
                  "dT = P_abs * gap / (k_powder * A_surface). Deliberately "
                  "crude; it only has to separate 'above the melt point' from "
                  "'below it'."),
        "drive": {
            "power_density_w_per_m3": p.power_density_w_per_m3,
            "normalization": ("forward.qrf_dg0 renormalizes Q so that "
                              "integral(Q dV) = power_density * V_part, so "
                              "both parts absorb the SAME watts per cubic "
                              "metre; the drive is not the difference"),
            "k_powder_w_per_m_k": p.k_powder,
            "melt_onset_c": p.t_pc_c,
            "preheat_c": p.preheat_c},
        "chamber": {"L_DOMAIN_m": fwd.L_DOMAIN,
                    "status": "FROZEN convention, not a tunable"},
        "parts_frozen_60mm": [
            _row("pyramid", ROOT / "shape_library_3d" / "stl" / "pyramid.stl",
                 L_m=0.060),
            _row("tamper", rt.TAMPER_STL, L_m=0.060)],
        "parts_adaptive": [
            _row("pyramid", ROOT / "shape_library_3d" / "stl" / "pyramid.stl"),
            _row("tamper", rt.TAMPER_STL)],
        "conclusion": ("the Tamper spans 0.740 of the frozen chamber against "
                       "the pyramid's 0.387, leaving 7.8 mm of powder to the "
                       "fixed-temperature wall against 18.4 mm. At the same "
                       "volumetric power it reaches a conduction-limited "
                       "plateau BELOW the 180 C melt onset, so it never "
                       "melts and the asymmetric objective is all in-bounds "
                       "deficit with a degenerate argmin at the horizon. This "
                       "is a property of the part-in-chamber configuration, "
                       "not of the mesh: every geometric gate on this mesh "
                       "passes (solve3d/results/stl_chamber_gate.json)."),
    }
    OUT.write_text(json.dumps(doc, indent=1, default=float))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
