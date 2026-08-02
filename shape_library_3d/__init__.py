"""3-D primitive geometry test library for heatr3d / solve3d.

Curated watertight STL primitives, each stressing one RF-heating or numerical
behavior, normalized to equal solid volume V* = 4188.79 mm^3 (the 20 mm sphere).

Public API (stable handoff for the Grade-and-Print tool and any consumer):
  - SHAPES:              curated registry (name -> ShapeSpec)
  - iter_parts():        iterate loadable parts (tiers 1-2 only)
  - load_part_stl(name): load a normalized part mesh from the library's stl/ dir
  - build (build_library.build): (re)generate stl/ + meta/ + manifest
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import trimesh

from shape_library_3d.registry import SHAPES, ShapeSpec

_STL_DIR = Path(__file__).resolve().parent / "stl"


def iter_parts() -> "Iterator[Tuple[str, ShapeSpec]]":
    """Yield (name, ShapeSpec) for loadable parts only (Tier-1 and Tier-2).

    Tier-3 rejection fixtures are intentionally excluded -- they are not parts.
    """
    for name, spec in SHAPES.items():
        if spec.tier in (1, 2):
            yield name, spec


def load_part_stl(name: str, stl_dir: Optional[Path] = None) -> trimesh.Trimesh:
    """Load a generated, normalized part mesh from ``stl/<name>.stl``."""
    directory = Path(stl_dir) if stl_dir is not None else _STL_DIR
    path = directory / f"{name}.stl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python -m shape_library_3d.build_library` first")
    return trimesh.load(path, force="mesh")


__all__ = ["SHAPES", "ShapeSpec", "iter_parts", "load_part_stl"]
