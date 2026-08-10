"""CHEAP hot-spot predictor: build the Tamper case, ONE EQS solve (no march),
inspect deposited-power density Q_rf and cell geometry.

Answers: where does RF power concentrate, and is the peak on a tiny sliver cell
(coarse/degenerate-mesh over-prediction signature) or a resolved bulk cell?
No transient march -> light. Single-threaded.
"""
import numpy as np
from solve3d.phase_e import run_tamper as rt

RESULTS = rt.RESULTS


def peakinfo(tag, qpart, vpart, cpart):
    """All args already restricted to part cells, same order as s_map."""
    size_mm = (vpart ** (1.0 / 3.0)) * 1e3
    qbar = np.average(qpart, weights=vpart)
    order = np.argsort(qpart)[::-1]
    print(f"\n=== {tag}: Q_rf on {qpart.size} part cells ===")
    print(f"  qbar(vol-wt)={qbar:.3e} W/m^3   q_max={qpart.max():.3e}   "
          f"peak/mean={qpart.max()/qbar:.1f}x")
    print(f"  cell size mm: min {size_mm.min():.3f}  median {np.median(size_mm):.3f}"
          f"  max {size_mm.max():.3f}")
    n_sliver = int((size_mm < 1.0).sum())
    print(f"  sub-1mm cells: {n_sliver} ({100*vpart[size_mm<1.0].sum()/vpart.sum():.2f}% of vol)")
    print("  --- top 8 Q cells (transient-spike candidates) ---")
    print("   q/qbar  size_mm     x      y      z   (mm)")
    for r in order[:8]:
        c = cpart[r] * 1e3
        print(f"   {qpart[r]/qbar:7.1f}  {size_mm[r]:6.3f}  "
              f"{c[0]:6.1f} {c[1]:6.1f} {c[2]:6.1f}")
    top = order[:max(1, int(0.01 * qpart.size))]
    print(f"  hottest 1% cells carry {100*(qpart[top]*vpart[top]).sum()/(qpart*vpart).sum():.1f}%"
          f" of power; their median size {np.median(size_mm[top]):.3f} mm")
    # is the peak cell an outlier in size (sliver) vs the bulk?
    print(f"  PEAK cell: q/qbar={qpart[order[0]]/qbar:.1f}, size={size_mm[order[0]]:.3f} mm, "
          f"z={cpart[order[0]][2]*1e3:.1f} mm  (median cell {np.median(size_mm):.3f} mm)")
    return qbar, size_mm, order


def main():
    print("building Tamper case (mesh + materials, no march)...")
    tc, info = rt.build_case()
    part = tc.eqs.part                    # index array
    vol = tc.eqs.vol[part]                # part-ordered
    cent = rt._part_centroids(tc)         # part-ordered, matches s_map
    n = part.size
    print(f"n_part_cells={n}  n_cells_total={tc.eqs.vol.size}  "
          f"lc_part={info.lc_part*1e3:.2f}mm  L_chamber={info.L_chamber_m*1e3:.1f}mm")

    su = np.ones(n)
    st_u = tc.eqs.forward(tc.design_to_sigma(su))
    peakinfo("UNIFORM s=1", st_u.q[part], vol, cent)

    m = np.load(RESULTS / "map_tamper_solve_filter_only.npz")
    s_solved = m["s_map"]
    assert s_solved.shape[0] == n, (s_solved.shape, n)
    st_s = tc.eqs.forward(tc.design_to_sigma(s_solved))
    peakinfo("SOLVED dopant", st_s.q[part], vol, cent)

    # cross-check s_map centroids match build centroids (data contract)
    dmax = np.abs(m["centroids"] - cent).max()
    print(f"\n[contract] saved-map vs rebuilt centroid max abs diff: {dmax:.2e} m "
          f"({'MATCH' if dmax < 1e-9 else 'MISMATCH -- ordering differs'})")


if __name__ == "__main__":
    main()
