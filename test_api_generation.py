#!/usr/bin/env python3
"""RED/GREEN tests for the server/page API-generation handshake.

Why this exists: the project's known GUI failure mode is a long-running
rfam_gui_server.py process serving STALE python routes underneath NEW static
files read from disk per request. The 2026-08-01 "Launch Run queues nothing"
report was exactly this: a pre-solve-mode server process (no
/api/tools/fgm-solve route) under the v2 front end, whose promoted launch
path posts to that route. The handshake makes the skew announce itself:
the server stamps an integer API_GENERATION into /api/meta and
/api/engine-version; the front end pins the generation it was written
against and shows a loud banner on mismatch or absence.

Contract:
  1. rfam_gui_server.API_GENERATION is an int (bump it whenever a route or
     launch-payload shape changes).
  2. _engine_version_info() carries api_generation == API_GENERATION, so
     both /api/meta and /api/engine-version serve it.
  3. webui/static/app.js pins EXPECTED_API_GENERATION to the same value
     (client/server agreement check, parsed from the source).

Run: ./.venv312/bin/python -m pytest test_api_generation.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import rfam_gui_server as srv  # noqa: E402


def test_api_generation_constant_is_int() -> None:
    assert isinstance(srv.API_GENERATION, int)
    assert srv.API_GENERATION >= 20260801  # date-stamped generation


def test_engine_version_info_carries_api_generation() -> None:
    info = srv._engine_version_info()
    assert info["api_generation"] == srv.API_GENERATION


def test_app_js_pins_matching_generation() -> None:
    src = (BASE / "webui" / "static" / "app.js").read_text(encoding="utf-8")
    m = re.search(r"EXPECTED_API_GENERATION\s*=\s*(\d+)", src)
    assert m, "app.js must pin EXPECTED_API_GENERATION"
    assert int(m.group(1)) == srv.API_GENERATION
