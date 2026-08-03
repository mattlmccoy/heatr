"""Fresh-process demonstration of the per-job EQS disk solution store.

Runs the SAME solve in two separate interpreters against a shared store
directory and records the wall time of each. Process 2 must report a disk hit,
must return a bit-identical field, and must be dramatically faster.

This is the claim the engine lane asked to see demonstrated: a package-verify
in a fresh process no longer pays for the EQS solve. Factorizations are NOT on
disk (SuperLU objects are not picklable), so a fresh process still rebuilds the
ILU on a genuine miss -- only exact repeats are free.

Load-check per the machine convention before running: n=48 does a real EQS
solve (~9 s), n=96 would be minutes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]

_CHILD = r'''
import json, sys, time
import numpy as np
import heatr3d as h3
from engine_speed.eqs_cache import EqsCache, solve_eqs_3d_cached

store, n, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
g = h3.Grid(n=n)
part = h3.make_geometry(g, "square", diam=0.020, zspan=0.020)
p = h3.Params()
gamma = h3.build_gamma(part, p, None)
c = EqsCache(store_dir=store)
t0 = time.perf_counter()
V = solve_eqs_3d_cached(gamma, g, p, cache=c)
dt = time.perf_counter() - t0
np.save(outp, V)
print("RESULT " + json.dumps({"stats": c.stats, "wall_s": dt}))
'''


def _child(store: Path, n: int, out_npy: Path) -> dict[str, Any]:
    script = store.parent / "_child.py"
    script.write_text(_CHILD)
    env = {**os.environ, "PYTHONPATH": str(_REPO),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    r = subprocess.run([sys.executable, str(script), str(store), str(n),
                        str(out_npy)],
                       capture_output=True, text=True, env=env, cwd=str(_REPO))
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")][0]
    return json.loads(line[len("RESULT "):])


def demo(n: int = 48) -> dict[str, Any]:
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        store = td / "eqs_store"
        a = _child(store, n, td / "a.npy")
        b = _child(store, n, td / "b.npy")
        va = np.load(td / "a.npy")
        vb = np.load(td / "b.npy")
        payloads = sorted(store.glob("*.npy"))
        store_bytes = sum(p.stat().st_size for p in store.iterdir())
        return {
            "n": n,
            "process1_wall_s": a["wall_s"],
            "process2_wall_s": b["wall_s"],
            "saved_s": a["wall_s"] - b["wall_s"],
            "speedup": a["wall_s"] / max(b["wall_s"], 1e-12),
            "process1_stats": a["stats"],
            "process2_stats": b["stats"],
            "process2_disk_hit": bool(b["stats"]["disk_hits"] == 1),
            "process2_solved_nothing": bool(b["stats"]["solution_misses"] == 0),
            "bit_identical_across_processes": bool(np.array_equal(va, vb)),
            "store_entries": len(payloads),
            "store_bytes": int(store_bytes),
        }


def main(out_path: str | Path | None = None, sizes=(48,)) -> dict[str, Any]:
    recs = {f"n{n}": demo(n) for n in sizes}
    recs["env"] = {"uptime": os.popen("uptime").read().strip()}
    if out_path is not None:
        Path(out_path).write_text(json.dumps(recs, indent=2, default=float))
    return recs


if __name__ == "__main__":
    sizes = tuple(int(a) for a in sys.argv[1:] if a.isdigit()) or (48,)
    here = Path(__file__).resolve().parent
    r = main(here / "fresh_process_demo.json", sizes=sizes)
    for k, d in r.items():
        if k == "env":
            continue
        print(f"{k}: process1 {d['process1_wall_s']:8.3f} s (cold solve) -> "
              f"process2 {d['process2_wall_s']:8.4f} s (disk hit)  "
              f"SAVED {d['saved_s']:.3f} s  ({d['speedup']:.0f}x)")
        print(f"   disk hit={d['process2_disk_hit']}  "
              f"solved nothing={d['process2_solved_nothing']}  "
              f"bit-identical across processes={d['bit_identical_across_processes']}  "
              f"store={d['store_bytes'] / 1e6:.1f} MB")
