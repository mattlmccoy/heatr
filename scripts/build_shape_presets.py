#!/usr/bin/env python3
"""Regenerate the server-side per-shape preset YAML files.

Deferred item (b) from HEATR_V2_ROLLOUT_NOTES.md. The single source of the
per-shape one-click standards is webui/static/fgm_shape_standards.json (itself
generated from the campaign artifacts by
scripts/analysis/build_fgm_shape_standards.py). This script converts that JSON
into one YAML preset per shape under presets/, which the graphical user
interface server reads and serves at GET /api/presets.

Never edit presets/*.yaml by hand; edit the standards JSON generator and
re-run this script.

Run:
  ./.venv312/bin/python scripts/build_shape_presets.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
STANDARDS_JSON = REPO / "webui" / "static" / "fgm_shape_standards.json"
PRESETS_DIR = REPO / "presets"
PRESET_VERSION = 1


def standards_to_presets(standards: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """One standalone preset dict per shape from the standards JSON dict."""
    shared = dict(standards.get("standard_parameters", {}) or {})
    presets: Dict[str, Dict[str, Any]] = {}
    for shape, std in (standards.get("shapes", {}) or {}).items():
        presets[str(shape)] = {
            "preset_version": PRESET_VERSION,
            "shape": str(shape),
            "shape_standard": dict(std or {}),
            "standard_parameters": dict(shared),
            "generated_from": "webui/static/fgm_shape_standards.json",
            "sources": list(standards.get("sources", []) or []),
            "engine_version_floor": standards.get("engine_version_floor"),
        }
    return presets


def write_presets(presets: Dict[str, Dict[str, Any]], out_dir: Path) -> List[Path]:
    """Write one <shape>.yaml per preset; returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for shape in sorted(presets):
        p = out_dir / f"{shape}.yaml"
        with open(p, "w", encoding="utf-8") as fh:
            yaml.safe_dump(presets[shape], fh, default_flow_style=False,
                           allow_unicode=True, sort_keys=True)
        written.append(p)
    return written


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    standards = json.loads(STANDARDS_JSON.read_text(encoding="utf-8"))
    presets = standards_to_presets(standards)
    written = write_presets(presets, PRESETS_DIR)
    logger.info("wrote %d preset YAML files to %s",
                len(written), PRESETS_DIR.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
