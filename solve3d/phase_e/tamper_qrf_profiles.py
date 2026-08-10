"""Spatial structure of Q_rf: radial, azimuthal, z profiles. Decides whether the
hot spot is a full RING (rotation cannot help) or an azimuthal spot (rotation
helps), and where the cold under-dense bulk sits. Saves fields for figures."""
import numpy as np
from solve3d.phase_e import run_tamper as rt

OUT = "/private/tmp/claude-501/-Users-mattmccoy-GaTech-Dropbox-Matthew-McCoy-mattmccoy-research-research-binderjet-code-geo-prewarp/632fc5f5-58d6-4ceb-a720-2930f7f099ff/scratchpad/tamper_qrf.npz"


def prof(name, x, q, vol, edges, unit=""):
    print(f"\n  {name} profile:")
    print(f"    {'bin':>10}  qbar/global  ncells  volfrac")
    gbar = np.average(q, weights=vol)
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1] + (1e-15 if i == len(edges) - 2 else 0))
        if m.sum() == 0:
            continue
        qb = np.average(q[m], weights=vol[m])
        print(f"    {0.5*(edges[i]+edges[i+1]):9.1f}{unit}  {qb/gbar:9.2f}  "
              f"{m.sum():6d}  {vol[m].sum()/vol.sum():.3f}")


def main():
    tc, info = rt.build_case()
    part = tc.eqs.part
    vol = tc.eqs.vol[part]
    cent = rt._part_centroids(tc)
    n = part.size
    s_solved = np.load(rt.RESULTS / "map_tamper_solve_filter_only.npz")["s_map"]

    qs = {}
    qs["uniform"] = tc.eqs.forward(tc.design_to_sigma(np.ones(n))).q[part]
    qs["solved"] = tc.eqs.forward(tc.design_to_sigma(s_solved)).q[part]

    x, y, z = cent[:, 0] * 1e3, cent[:, 1] * 1e3, cent[:, 2] * 1e3
    r = np.hypot(x, y)
    th = np.degrees(np.arctan2(y, x))

    for key in ("uniform", "solved"):
        q = qs[key]
        print(f"\n================ {key} Q_rf ================")
        prof("RADIAL r(mm)", r, q, vol, np.linspace(0, r.max(), 9), "mm")
        prof("Z (mm)", z, q, vol, np.linspace(z.min(), z.max(), 9), "mm")
        prof("AZIMUTH th(deg)", th, q, vol, np.linspace(-180, 180, 13), "d")

    # cold-bulk locator (uniform): where is Q lowest = under-dense risk
    q = qs["uniform"]
    lo = q < np.percentile(q, 25)
    print("\n=== COLDEST 25% cells (under-dense bulk) location ===")
    print(f"  r: mean {np.average(r[lo],weights=vol[lo]):.1f}  "
          f"z: mean {np.average(z[lo],weights=vol[lo]):.1f}  "
          f"volfrac {vol[lo].sum()/vol.sum():.2f}")
    hi = q > np.percentile(q, 99)
    print("=== HOTTEST 1% cells (ceiling risk) location ===")
    print(f"  r: mean {np.average(r[hi],weights=vol[hi]):.1f} (range {r[hi].min():.1f}-{r[hi].max():.1f})  "
          f"z: mean {np.average(z[hi],weights=vol[hi]):.1f}  "
          f"volfrac {vol[hi].sum()/vol.sum():.3f}")
    # azimuthal spread of the hottest 1%: ring or spot?
    hth = th[hi]
    hist, _ = np.histogram(hth, bins=np.linspace(-180, 180, 13))
    print(f"  hottest-1% azimuth histogram (12 x 30deg bins): {hist}")
    occ = int((hist > 0).sum())
    print(f"  -> occupies {occ}/12 azimuth bins  "
          f"({'FULL RING (rotation cannot help)' if occ >= 10 else 'AZIMUTHAL SPOT (rotation may help)'})")

    np.savez_compressed(OUT, x=x, y=y, z=z, r=r, th=th, vol=vol,
                        q_uniform=qs["uniform"], q_solved=qs["solved"],
                        s_solved=s_solved)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
