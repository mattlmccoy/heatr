"""Level 0 affine pre-compensation for MATERIAL shrinkage.

Spec: docs/superpowers/specs/2026-08-04-shrinkage-prewarp-v2-design.md sec 2.
Coefficients: SHRINKAGE_COEFFICIENTS_MEMO.md sec 3, carried in the shared
repo-root file `shrinkage_precomp.json` (schema 1.0, mirrored by the Studio
lane). Nothing here hardcodes a coefficient.

WHAT THIS IS. The nominal CAD is scaled UP by 1/(1 - s) per axis before chi
and the mesh are built, so that the part which forms and then shrinks by s
lands on nominal. The scale is anisotropic: s_xy in plane, s_z_mat in build
height.

WHAT THIS IS NOT, and the guard that keeps it that way. Two physically
separate effects, and Matt's recorded distinction between them is the reason
this module exists as its own layer:

  MATERIAL SHRINKAGE      nylon 12 contracting on melt and recrystallisation,
                          roughly affine at part scale. THIS module's job.
  DENSIFICATION           powder consolidating to solid, rho 0.55 -> ~1.0,
                          about 45 percent by volume, dominantly vertical.
                          The DENSIFY MARCH's job, already in the model.

Folding a consolidation term into the pre-scale would count that collapse
twice: once here as geometry and again in the marched density field. So the
schema carries material coefficients ONLY, and `from_mapping` REFUSES a
mapping that offers a densification or total-shrinkage term instead of
quietly ignoring it. That refusal is the named double-counting guard.

DEFAULT IS ON. Authority: the APPROVED spec
docs/superpowers/specs/2026-08-04-shrinkage-prewarp-v2-design.md section 2
(Level 0), verbatim: "Off by default until the memo lands; then default ON
with the coefficients displayed." SHRINKAGE_COEFFICIENTS_MEMO.md is that memo
and it has landed, so ON is the approved state, not a new default decision by
this lane.

Consequence for reproduction, and it is not subtle: every campaign artifact in
solve3d/ predates L0 and was built with NO pre-compensation. Re-running any of
them reproduces the recorded numbers only if the coefficients are pinned to
zero explicitly, `build_case(shape, precomp_coeffs=ShrinkageL0(0.0, 0.0))`.
The flip governs new runs; it does not retroactively describe old ones.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SCHEMA = ("1.0",)

# Substrings that mark a coefficient as belonging to consolidation, or as a
# total that already contains it. Matching keys are refused outright.
_BANNED_SUBSTRINGS = ("densif", "consolidat", "collapse", "rho", "density",
                      "total", "porosity", "shrink_total")


class DoubleCountingError(ValueError):
    """Raised when a mapping tries to put consolidation into the pre-scale."""


class SchemaVersionError(ValueError):
    """Raised on an unrecognised shared-config schema version."""


def config_path() -> Path:
    """The ONE shared coefficients file, cross-lane (Studio mirrors it)."""
    return ROOT / "shrinkage_precomp.json"


@dataclass(frozen=True)
class ShrinkageL0:
    """Material-shrinkage coefficients and the affine map they induce.

    `s_xy` and `s_z_mat` are the ONLY coefficient fields; see `field_names`.
    Everything else on this object is provenance, not physics.
    """

    s_xy: float
    s_z_mat: float
    enabled: bool = True
    s_xy_band: tuple[float, float] | None = None
    s_z_mat_band: tuple[float, float] | None = None
    source: str = ""
    schema_version: str = ""
    material_only: bool = True
    applicability: str = ""
    _extra: dict = field(default_factory=dict, repr=False, compare=False)

    # -- the coefficient schema, asserted by the double-counting guard ----- #
    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """The MATERIAL coefficient schema. Deliberately just these two: any
        densification parameter appearing here would be double counted."""
        return ("s_xy", "s_z_mat")

    # -- the affine map ---------------------------------------------------- #
    @property
    def xy_scale(self) -> float:
        return 1.0 / (1.0 - float(self.s_xy))

    @property
    def z_scale(self) -> float:
        return 1.0 / (1.0 - float(self.s_z_mat))

    @property
    def is_identity(self) -> bool:
        return (not self.enabled) or (self.s_xy == 0.0 and self.s_z_mat == 0.0)

    @property
    def factors(self) -> np.ndarray:
        """Per-axis scale (x, y, z); exactly ones when the map is identity."""
        if self.is_identity:
            return np.ones(3, dtype=float)
        return np.array([self.xy_scale, self.xy_scale, self.z_scale], float)

    def xy_scale_band(self) -> tuple[float, float]:
        lo, hi = self._band("s_xy_band", self.s_xy)
        return 1.0 / (1.0 - lo), 1.0 / (1.0 - hi)

    def z_scale_band(self) -> tuple[float, float]:
        lo, hi = self._band("s_z_mat_band", self.s_z_mat)
        return 1.0 / (1.0 - lo), 1.0 / (1.0 - hi)

    def _band(self, name: str, fallback: float) -> tuple[float, float]:
        b = getattr(self, name)
        return (float(fallback), float(fallback)) if b is None else (
            float(b[0]), float(b[1]))

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_mapping(cls, m: dict, *, enabled: bool = True) -> "ShrinkageL0":
        """Build from a mapping, REFUSING any consolidation-flavoured key.

        The refusal is loud and names the offending key, because silently
        dropping it would leave the caller believing consolidation had been
        compensated when it had not.
        """
        offenders = [k for k in m
                     if k not in ("s_xy", "s_z_mat")
                     and any(b in str(k).lower() for b in _BANNED_SUBSTRINGS)]
        if offenders:
            raise DoubleCountingError(
                "Level 0 pre-compensation is MATERIAL shrinkage only; "
                f"refusing consolidation-flavoured key(s) {sorted(offenders)}. "
                "Densification is marched by the model (spec section 1); "
                "putting it here counts the collapse twice.")
        if "s_xy" not in m or "s_z_mat" not in m:
            raise ValueError("both s_xy and s_z_mat are required")
        band_xy = m.get("s_xy_band")
        band_z = m.get("s_z_mat_band")
        return cls(
            s_xy=float(m["s_xy"]), s_z_mat=float(m["s_z_mat"]),
            enabled=bool(enabled),
            s_xy_band=None if band_xy is None else (float(band_xy[0]),
                                                    float(band_xy[1])),
            s_z_mat_band=None if band_z is None else (float(band_z[0]),
                                                      float(band_z[1])),
            source=str(m.get("source", "")),
            schema_version=str(m.get("schema_version", "")),
            material_only=bool(m.get("material_only", True)),
            applicability=str(m.get("applicability", "")),
            _extra={k: v for k, v in m.items() if k == "note"})

    # -- provenance -------------------------------------------------------- #
    def provenance(self) -> dict:
        """What a run records. Bands travel with the values so that any
        dimensional claim downstream can carry its labeled uncertainty."""
        p = {
            "level": 0,
            "enabled": bool(self.enabled),
            "s_xy": float(self.s_xy),
            "s_z_mat": float(self.s_z_mat),
            "s_xy_band": list(self._band("s_xy_band", self.s_xy)),
            "s_z_mat_band": list(self._band("s_z_mat_band", self.s_z_mat)),
            "xy_scale": float(self.xy_scale),
            "z_scale": float(self.z_scale),
            "xy_scale_band": list(self.xy_scale_band()),
            "z_scale_band": list(self.z_scale_band()),
            "source": self.source or "SHRINKAGE_COEFFICIENTS_MEMO.md",
            "schema_version": self.schema_version,
            "material_only": bool(self.material_only),
            "applicability": self.applicability,
            "scope": "MATERIAL shrinkage of nylon 12 only",
            "excludes": ("densification consolidation, which the densify "
                         "march owns (double-counting guard)"),
            "config_file": str(config_path().name),
        }
        p.update(self._extra)
        return p


def from_mapping_checked(m: dict, *, enabled: bool = True) -> ShrinkageL0:
    """`from_mapping` plus the two shared-file contract checks.

    The schema version is refused rather than read optimistically, and
    `material_only` must be asserted TRUE by the file: a coefficients file
    that does not claim to be material-only is not trusted to be.
    """
    ver = str(m.get("schema_version", ""))
    if ver not in SUPPORTED_SCHEMA:
        raise SchemaVersionError(
            f"unsupported shrinkage_precomp schema_version {ver!r}; "
            f"this lane supports {SUPPORTED_SCHEMA}")
    if m.get("material_only") is not True:
        raise DoubleCountingError(
            "shared coefficients file must assert material_only = true; "
            f"got {m.get('material_only')!r}. Without that assertion the "
            "values may already contain consolidation (double-counting "
            "guard, spec section 1).")
    return ShrinkageL0.from_mapping(m, enabled=enabled)


def load_defaults(*, enabled: bool = True) -> ShrinkageL0:
    """Read the shared root config. Default ON; see the module docstring for
    the spec line that authorises it and for the reproduction consequence."""
    return from_mapping_checked(json.loads(config_path().read_text()),
                                enabled=enabled)


def scale_points(points: np.ndarray, coeffs: ShrinkageL0) -> np.ndarray:
    """Apply the pre-compensation to (N, 3) points in metres.

    Identity coefficients return a bit-identical copy, so that switching L0
    off reproduces existing campaigns exactly rather than approximately.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {p.shape}")
    if coeffs.is_identity:
        return p.copy()
    return p * coeffs.factors


def unscale_points(points: np.ndarray, coeffs: ShrinkageL0) -> np.ndarray:
    """Inverse map: pre-compensated frame back to nominal."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {p.shape}")
    if coeffs.is_identity:
        return p.copy()
    return p / coeffs.factors
