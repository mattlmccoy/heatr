"""3-D primitive geometry test library for heatr3d / solve3d.

Curated watertight STL primitives, each stressing one RF-heating or numerical
behavior, normalized to equal solid volume V* = 4188.79 mm^3 (the 20 mm sphere).

The stable public API (SHAPES, iter_parts, load_part_stl) is wired in Task 7
(build_library). Submodules (constants, normalize, validate, generators) are
importable directly and independently during the TDD build.
"""
from __future__ import annotations
