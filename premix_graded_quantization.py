"""How much of the graded uniformity win survives printable quantization?

The graded map is continuous; RFAM prints 2/4 bpp. Quantize the adjoint A15 map to
4 bpp (16 levels) and 2 bpp (4 levels), inject each into run_sim, drive to the 250 C
ceiling, and compare part uniformity/fusion to the continuous graded and to uniform.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

import rfam_eqs_coupled as rc
from premix_study_metrics import peak_to_mean
from quantize_sat import quantize_sat
from premix_vs_graded import _cfg_uniform, _cfg_graded, _run, drive_to_ceiling, ADJ_MAPS, GRADED_SAT

OUTDIR = Path("results/premix_vs_graded")


def _cfg_graded_from(sat_path, sat_max, gen_power_w):
    c = _cfg_uniform(gen_power_w)
    c["fgm_feedback"] = {"enabled": True, "sat_map_npz_direct": str(Path(sat_path).resolve()),
                         "sat_max": float(sat_max)}
    return c


def main():
    base = rc.make_domain(_cfg_uniform(500.0))
    part_mask = base[3]
    d = np.load(ADJ_MAPS, allow_pickle=True)
    s_cont = np.asarray(d["A15"], dtype=np.float32)
    sat_max = 1.5
    results = {}
    for n_bits, tag in [(4, "graded_4bpp"), (2, "graded_2bpp")]:
        sat_q = quantize_sat(s_cont, n_bits=n_bits, sat_max=sat_max)
        qpath = OUTDIR / f"{tag}_sat.npz"
        np.savez(qpath, sat_map=sat_q)
        n_levels_used = int(len(np.unique(sat_q[part_mask])))
        cfg_fn = lambda p, _qp=qpath: _cfg_graded_from(_qp, sat_max, p)
        drive = drive_to_ceiling(cfg_fn, part_mask)
        st = _run(cfg_fn(drive))
        results[tag] = dict(n_bits=n_bits, n_levels_in_part=n_levels_used,
                            drive_to_ceiling_W=drive,
                            part_peak_T_C=float(st.T[part_mask].max()),
                            part_mean_phi=float(st.phi[part_mask].mean()),
                            part_peak_to_mean=peak_to_mean(st.T, part_mask))
        print(f"[{tag}] levels={n_levels_used} drive={drive:.0f}W "
              f"peakT={results[tag]['part_peak_T_C']:.1f} phi={results[tag]['part_mean_phi']:.3f} "
              f"peak/mean={results[tag]['part_peak_to_mean']:.3f}")
    (OUTDIR / "premix_graded_quantization.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUTDIR/'premix_graded_quantization.json'}")


if __name__ == "__main__":
    main()
