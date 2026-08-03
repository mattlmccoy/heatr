"""March-log parsing: live progress + scalar series from workbench_job job.log.

The solver's verbose line (heatr3d.py:1237, captured format):
  `  t= 123.4s  Tmax= 190.1  phi=0.453  rho=0.550`
plus the wrapper's own `PROGRESS <pct>` and `PHASE <name>` lines. The parser
runs in the SERVER interpreter, so it must be stdlib-only.
"""
from __future__ import annotations

from heatr3d_workbench import march

LOG = """PHASE build
PROGRESS 5
PHASE march
  t=   0.0s  Tmax=  23.0  phi=0.000  rho=0.550
  t=  10.0s  Tmax=  61.2  phi=0.000  rho=0.550
  t=  20.0s  Tmax= 101.7  phi=0.012  rho=0.551
garbage line that must be ignored
  t=  30.0s  Tmax= 143.9  phi=0.180  rho=0.560
PROGRESS 90
PHASE post
"""


def test_series_parsed():
    s = march.parse_log(LOG)
    assert s["series"]["t_s"] == [0.0, 10.0, 20.0, 30.0]
    assert s["series"]["T_max_c"][1] == 61.2
    assert s["series"]["phi_bar"][-1] == 0.180
    assert s["series"]["rho_bar"][-1] == 0.560


def test_progress_prefers_march_time_over_coarse_steps():
    # Mid-march (log truncated before the post-march PROGRESS mark): at t=30
    # of max_time 60 s, march-derived progress should land mid-window, above
    # the wrapper's coarse pre-march mark.
    mid = LOG.split("PROGRESS 90")[0]
    s = march.parse_log(mid, max_time_s=60.0)
    assert 40 <= s["progress"] <= 80
    # A LATER wrapper mark supersedes march-derived progress.
    assert march.parse_log(LOG, max_time_s=60.0)["progress"] == 90.0
    # with no march lines yet, falls back to the last PROGRESS mark
    s2 = march.parse_log("PROGRESS 5\n", max_time_s=60.0)
    assert s2["progress"] == 5


def test_phase_reported():
    assert march.parse_log(LOG)["phase"] == "post"
    assert march.parse_log("PHASE build\n")["phase"] == "build"
    assert march.parse_log("")["phase"] == "queued"


def test_progress_capped_at_99_until_done():
    long = "PHASE march\n" + "\n".join(
        f"  t={t:6.1f}s  Tmax= 100.0  phi=0.900  rho=0.550" for t in (0, 3600))
    s = march.parse_log(long, max_time_s=60.0)
    assert s["progress"] <= 99
