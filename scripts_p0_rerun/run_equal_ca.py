#!/usr/bin/env python3
"""Sequential equal-cross-sectional-area FGM driver (GAP 1 + GAP 2).

For one shape: submit an INTEGRAL fgm_iterate job to the running GUI server with
the equal-area geometry scale (from equal_ca_scale.json), poll to completion,
then print the gate report.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCALE = json.loads((ROOT / "scripts_p0_rerun" / "equal_ca_scale.json").read_text())
PORT = 8080


def _jobs() -> list:
    with urllib.request.urlopen(f"http://localhost:{PORT}/api/jobs", timeout=30) as r:
        return json.loads(r.read())


def main() -> None:
    shape = sys.argv[1]
    mag = float(sys.argv[2])
    out_name = sys.argv[3]
    sc = SCALE[shape]
    cmd = [
        "python3", str(ROOT / "scripts_p0_rerun" / "submit_fgm_iterate.py"),
        "--shape", shape, "--output-name", out_name,
        "--magnitude", str(mag), "--exposure-minutes", "12",
        "--n-iterations", "12",
        "--geometry-size-mm", f"{sc['new_w_mm']:.4f}",
        "--geometry-nominal-mm", f"{sc['base_w_mm']:.4f}",
    ]
    print(f"[submit] {shape} mag={mag} w={sc['new_w_mm']:.3f}mm A={sc['new_A_mm2']:.1f}mm2",
          flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("SUBMIT FAILED:", res.stderr, flush=True)
        sys.exit(1)
    body = json.loads(res.stdout.strip().splitlines()[-1])
    job_id = str(body.get("job_id") or body.get("id"))
    print(f"[job] id={job_id}", flush=True)

    t0 = time.time()
    last_label = ""
    while True:
        time.sleep(15)
        jobs = _jobs()
        j = next((x for x in jobs if str(x.get("id")) == job_id), None)
        if j is None:
            print("[poll] job vanished", flush=True); break
        st = str(j.get("status"))
        label = str(j.get("progress_label", ""))
        if label != last_label:
            print(f"[poll] {int(time.time()-t0)}s status={st} pct={j.get('progress_pct')} {label}",
                  flush=True)
            last_label = label
        if st in ("completed", "failed", "cancelled", "error"):
            print(f"[done] status={st} after {int(time.time()-t0)}s", flush=True)
            break

    rel = f"runs/{shape}/fgm_iterate/{out_name}"
    print(f"[report] {rel}", flush=True)
    rep = subprocess.run(
        ["python3", str(ROOT / "scripts_p0_rerun" / "report_fgm_run.py"), rel],
        capture_output=True, text=True)
    print(rep.stdout, flush=True)
    if rep.stderr:
        print(rep.stderr, flush=True)


if __name__ == "__main__":
    main()
