"""Verification of the heatr3d Q_rf conventions coded in eqs_common.

Not part of the Task 1 plan gate (that is run_plate.py), but qrf_dg0 is used by
Tasks 2-5, and an unverified Q convention would silently corrupt every later
comparison. The check is analytic: a FULLY doped chamber has the uniform field
|E| = 860 / 0.060 = 14333.33 V/m, so before renormalization

    Q_raw = 0.5 * sigma_doped * |E|^2
          = 0.5 * 0.04 * (860/0.060)^2 = 4.10889e6 W/m^3

and after heatr3d's renormalization (integral Q dV = power_density * V_doped,
with V_doped = the whole chamber here) every cell must equal exactly

    Q_norm = power_density_w_per_m3 = 1.59155e6 W/m^3

so the returned scale must be power_density / Q_raw = 0.387345.

Run: ./env/bin/python test_eqs_common.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import eqs_common as ec

N = 8
TOL = 1e-9


def main() -> int:
    msh = ec.box_mesh(N)
    mats = ec.materials(msh, in_part=lambda p: np.ones(p.shape[1], bool))  # all doped
    Vr, Vi = ec.solve_eqs(msh, mats)
    q, scale, p_target = ec.qrf_dg0(msh, Vr, Vi, mats)

    e_mag = (ec.V_LO - ec.V_HI) / ec.L_DOMAIN
    q_raw = 0.5 * ec.SIGMA_DOPED * e_mag ** 2
    scale_exp = ec.POWER_DENSITY_W_PER_M3 / q_raw
    q_arr = np.real(q.x.array)

    checks = {
        "scale": (scale, scale_exp),
        "q_uniform_min": (float(q_arr.min()), ec.POWER_DENSITY_W_PER_M3),
        "q_uniform_max": (float(q_arr.max()), ec.POWER_DENSITY_W_PER_M3),
        "p_target_W": (p_target, ec.POWER_DENSITY_W_PER_M3 * ec.L_DOMAIN ** 3),
    }
    ok = True
    for name, (got, exp) in checks.items():
        rel = abs(got - exp) / abs(exp)
        ok &= rel < TOL
        print(f"{name:16s} got={got:.10g} expected={exp:.10g} rel={rel:.3e}")
    print("Q_raw analytic =", q_raw, "W/m^3")
    out = {"task1_qrf_convention_check": {
        "case": "fully doped chamber, uniform |E| = 860/0.060 V/m",
        "q_raw_analytic_w_per_m3": q_raw,
        "checks": {k: {"got": float(g), "expected": float(e),
                       "rel": float(abs(g - e) / abs(e))}
                   for k, (g, e) in checks.items()},
        "tol": TOL, "gate_ok": bool(ok)}}
    p = Path(__file__).parent / "results.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d.update(out)
    p.write_text(json.dumps(d, indent=1))
    assert ok, "Q_rf convention check failed"
    print("PASS: qrf_dg0 reproduces heatr3d's Q definition and power-renorm basis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
