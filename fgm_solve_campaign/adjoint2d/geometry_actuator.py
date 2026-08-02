"""Heating-kernel anisotropy spectrum, candidate angles, actuator recommendation.

THE QUESTION THIS ANSWERS. Given an imported geometry, before any solve is run:
is a dopant map enough, does the part need a turntable, and if so which
schedule; or is the part beyond what rotational actuation can do, so that the
money should go into a sequential dwell instead? `CONTINUOUS_ROTATION_REPORT.md`
Section 4 measured that the residual azimuthal anisotropy of the part-frame
heating kernel RANKS the outcomes exactly: the shapes whose kernel the actuator
drove close to annular (star, square, cross) are the shapes it rescued, and the
two it left strongly anisotropic (T_shape, L_shape) are the two it did not.
Section 1 gives the mechanism in one line: a radial kernel melts a rounded blob.

THE METRIC, stated exactly because a second implementation must not drift.
Let `A_k` be the part-frame rotation by `theta_k` over the REFERENCE group (24
angles at 15 degrees, the near-continuous turntable step of
`configs/diamond_tt_15deg_48rot_nearcont.yaml`, and the same set the rotation
campaign averaged over). For a part-frame kernel `K` and part mask `P`:

    w    = mean_k R_k P                 the annular OCCUPANCY of the part
    Kbar = mean_k R_k K                 the rotationally averaged (annular) part
    A(K) = sum(w |K - Kbar|) / sum(w Kbar)

A is zero exactly when K is invariant under the sampled rotation group, it is
invariant to the drive scale (so the recommendation cannot be changed by the
voltage calibration), and the occupancy weight w is what makes it defined for
EVERY geometry: it down-weights the annuli the part barely occupies instead of
requiring an inscribed disc, which does not exist for an L-shaped part whose
material does not cover the rotation centre.

WHY THE METRIC IS A RE-DERIVATION AND NOT THE STORED ONE. The five numbers the
rotation report quotes survive in `out_rot/kernel_anisotropy.json`, but the
script that produced them does NOT survive anywhere in the repository, and four
reconstructions of the definition its prose gives ("mean over 23 rotations of
the mean absolute difference between the kernel and its own rotation,
normalized by the kernel's mean, on the largest disc inscribed in the part")
all failed to reproduce the stored values; the same definition also returns an
undefined value on the L_shape, whose part does not cover the rotation centre,
so it cannot be the definition that produced the stored L_shape number. The
thresholds below are therefore calibrated against the campaign's OUTCOMES, which
are ground truth, and not against its numbers, which cannot be re-derived. Both
sets are reported side by side wherever this classifier is used.

MODES. `static` (no rotation), `continuous` (the 24-angle average), and
`index<N>` for every N that DIVIDES the part's rotational order, which are the
indexing schedules whose averaged kernel carries the part's own symmetry.
`CONTINUOUS_ROTATION_REPORT.md` Section 7 measured that such a matched indexing
beats the finest continuous rotation by factors of 2.8 (cross) and 2.1 (square),
and that more averaging is NOT better.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .rot_frame import averaging_angles, rotation_operator

__all__ = ["azimuthal_anisotropy", "reference_angles", "mode_names",
           "mode_angles", "classify_residual", "recommend", "Recommendation",
           "spectrum_from_cfg", "CALIBRATION", "A_MODE_SUFFICES",
           "A_PHYSICAL_LIMIT", "MIN_REDUCTION"]

REFERENCE_STEP_DEG = 15.0

# Bands, calibrated on the five campaign shapes below. The gap between the
# worst rotation winner (star, 0.462) and the best rotation failure (T_shape,
# 0.837) is a factor of 1.81; the two thresholds are placed inside it. The
# star's margin to the first threshold is 8 percent, and that thinness is the
# honest limit of this classifier.
A_MODE_SUFFICES = 0.50
A_PHYSICAL_LIMIT = 0.80
# A mode has to reduce the anisotropy by at least this factor before rotation
# is worth recommending; below it the static arm is kept. Set from the L_shape,
# whose continuous mode buys 1.14 and whose measured campaign result was
# "effectively a tie" (CONTINUOUS_ROTATION_REPORT Section 1).
MIN_REDUCTION = 1.15

# The calibration set. `A_measured` is the CONTINUOUS-mode residual measured by
# this module in this pass, at grid 120, uniform dopant map, electrical state B.
# `outcome` is the campaign's own verdict, quoted with its source.
CALIBRATION = (
    {"shape": "square", "A_measured": 0.277, "outcome": "rotation_wins",
     "source": "CONTINUOUS_ROTATION_REPORT Section 1: J 25.56 -> 13.93, "
               "IoU 0.9804 -> 1.0000 at 90-degree indexing"},
    {"shape": "cross", "A_measured": 0.419, "outcome": "rotation_wins",
     "source": "CONTINUOUS_ROTATION_REPORT Section 1: J 147.03 -> 34.82, "
               "IoU 0.8514 -> 0.9866 at 90-degree indexing"},
    {"shape": "star", "A_measured": 0.462, "outcome": "rotation_wins",
     "source": "CONTINUOUS_ROTATION_REPORT Section 1: J 106.28 -> 24.46, "
               "IoU 0.7870 -> 0.9527; the win is the ROTATION, the map is inert"},
    {"shape": "T_shape", "A_measured": 0.837, "outcome": "rotation_fails",
     "source": "CONTINUOUS_ROTATION_REPORT Section 1: only -10.5 percent J and "
               "+0.016 IoU, 42.9 percent of the part still unmelted; the best "
               "T_shape result on record is the SEQUENTIAL dwell "
               "(SEQUENTIAL_DWELL_REPORT: 343.02 / 0.6661)"},
    {"shape": "L_shape", "A_measured": 1.098, "outcome": "rotation_fails",
     "source": "CONTINUOUS_ROTATION_REPORT Section 1: -6.4 percent J and "
               "-0.011 IoU, effectively a tie; the best L_shape result on "
               "record is the SEQUENTIAL dwell (SEQUENTIAL_DWELL_REPORT: "
               "276.39 / 0.7161)"},
)


# ---------------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------------

def reference_angles(step_deg: float = REFERENCE_STEP_DEG) -> np.ndarray:
    """The reference rotation group every residual is measured against."""
    return averaging_angles(float(step_deg))


def _ops(shape: tuple[int, int], angles: np.ndarray, cache: dict | None):
    key = (shape, tuple(np.round(np.asarray(angles, dtype=float), 9)))
    if cache is not None and key in cache:
        return cache[key]
    ops = [rotation_operator(shape, float(a)) for a in np.asarray(angles, dtype=float)]
    if cache is not None:
        cache[key] = ops
    return ops


def rotational_average(field_: np.ndarray, angles_deg, cache: dict | None = None
                       ) -> np.ndarray:
    """The annular (rotationally averaged) part of a field."""
    f = np.asarray(field_, dtype=float)
    ops = _ops(f.shape, angles_deg, cache)
    out = np.zeros_like(f)
    for R in ops:
        out += R.apply(f, outside=0.0)
    return out / float(len(ops))


def azimuthal_anisotropy(K: np.ndarray, part_mask: np.ndarray, angles_deg,
                         cache: dict | None = None) -> float:
    """Occupancy-weighted deviation of a kernel from rotational invariance."""
    k = np.asarray(K, dtype=float)
    p = np.asarray(part_mask, dtype=float)
    if p.shape != k.shape:
        raise ValueError(f"mask shape {p.shape} does not match kernel {k.shape}")
    w = rotational_average(p, angles_deg, cache)
    kb = rotational_average(k, angles_deg, cache)
    den = float(np.sum(w * kb))
    if den <= 0.0:
        raise ValueError("the kernel carries no weight inside the part")
    return float(np.sum(w * np.abs(k - kb)) / den)


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def mode_names(rotational_order: int) -> tuple[str, ...]:
    from .geometry_symmetry import indexing_orders
    return ("static", "continuous") + tuple(
        f"index{n}" for n in indexing_orders(int(rotational_order)))


def mode_angles(mode: str, step_deg: float = REFERENCE_STEP_DEG) -> np.ndarray:
    m = str(mode)
    if m == "static":
        return np.array([0.0])
    if m == "continuous":
        return averaging_angles(float(step_deg))
    if m.startswith("index"):
        n = int(m[len("index"):])
        if n < 2:
            raise ValueError(f"indexing order must be at least 2, got {n}")
        return np.arange(n, dtype=float) * (360.0 / n)
    raise ValueError(f"unknown mode {mode!r}")


def spectrum_from_cfg(cfg: dict, rotational_order: int,
                      step_deg: float = REFERENCE_STEP_DEG,
                      log=None) -> dict:
    """Run the electro-quasi-static (EQS) solves and measure every mode.

    One `AveragedKernel` per mode, uniform dopant map, electrical state B (the
    state the march runs in from the first `update_interval` tick onwards, so
    the state most of the exposure sees). Returns the residual per mode plus
    the kernels themselves for the figures.
    """
    from .rot_kernel import AveragedKernel

    ref = reference_angles(step_deg)
    cache: dict = {}
    out: dict = {"residual": {}, "kernels": {}, "reference_angles_deg": list(ref),
                 "metric": "occupancy-weighted azimuthal anisotropy, "
                           "adjoint2d.geometry_actuator.azimuthal_anisotropy"}
    pm = None
    for mode in mode_names(int(rotational_order)):
        ang = mode_angles(mode, step_deg)
        kern = AveragedKernel.build(copy.deepcopy(cfg), angles=ang)
        pm = kern.case0.part_mask
        _Qa, Qb = kern.averaged_Q(np.ones(pm.shape))
        a = azimuthal_anisotropy(Qb, pm, ref, cache)
        out["residual"][mode] = float(a)
        out["kernels"][mode] = np.array(Qb, dtype=float)
        if log is not None:
            log(f"  mode {mode:11s} angles {len(ang):2d}  residual anisotropy "
                f"{a:.4f}")
        del kern
    out["part_mask"] = pm
    return out


# ---------------------------------------------------------------------------
# the recommendation
# ---------------------------------------------------------------------------

def classify_residual(a: float) -> str:
    v = float(a)
    if v <= A_MODE_SUFFICES:
        return "MODE_SUFFICES"
    if v <= A_PHYSICAL_LIMIT:
        return "MAP_PLUS_MODE"
    return "PHYSICAL_LIMIT"


@dataclass(frozen=True)
class Recommendation:
    mode: str
    actuator_class: str
    residual: float
    static_residual: float
    reduction_factor: float
    rotation_recommended: bool
    advice: str
    candidate_angles_deg: tuple[float, ...] = ()
    spectrum: dict = field(default_factory=dict)

    def as_json(self) -> dict:
        return {"mode": self.mode, "actuator_class": self.actuator_class,
                "residual_anisotropy": float(self.residual),
                "static_residual_anisotropy": float(self.static_residual),
                "reduction_factor": float(self.reduction_factor),
                "rotation_recommended": bool(self.rotation_recommended),
                "advice": self.advice,
                "candidate_angles_deg": [float(a) for a in self.candidate_angles_deg],
                "spectrum": {k: float(v) for k, v in self.spectrum.items()}}


_ADVICE = {
    "MODE_SUFFICES": (
        "The residual anisotropy after this mode is inside the band where the "
        "rotation campaign's three winners sat (square 0.277, cross 0.419, "
        "star 0.462), so the ACTUATOR alone is expected to carry most of the "
        "gain. Solve the dopant map against this mode's averaged kernel for "
        "the remainder; on the star that map was measurably inert, so quote "
        "the win as an orientation result unless the map earns it."),
    "MAP_PLUS_MODE": (
        "The mode helps but leaves real anisotropy, between the winners' band "
        "and the failures' band. Solve the map against this mode's averaged "
        "kernel and expect the map to be load bearing rather than a finish."),
    "PHYSICAL_LIMIT": (
        "The residual anisotropy is in the band of the two shapes rotation "
        "did NOT rescue (T_shape 0.837, L_shape 1.098). Expect a physical "
        "limit: a radial kernel melts a rounded blob and this part is not "
        "close to radially symmetric. The best results on record for that "
        "class came from SEQUENTIAL DWELL, an ordered hold that melts one limb "
        "and then the other (SEQUENTIAL_DWELL_REPORT: L_shape 396.20 -> "
        "276.39 J, IoU 0.6484 -> 0.7161; T_shape 422.28 -> 343.02 J). Neither "
        "reached the SOLVED class, so this is a partial rescue, not a fix."),
}


def recommend(spectrum: dict, rotational_order: int,
              candidate_step_deg: float = 15.0) -> Recommendation:
    """Pick the actuator mode and say what the campaign says to expect from it."""
    spec = {str(k): float(v) for k, v in dict(spectrum).items()}
    if "static" not in spec:
        raise ValueError("the spectrum must contain the 'static' arm; every "
                         "reduction factor is measured against it")
    static = spec["static"]
    others = {k: v for k, v in spec.items() if k != "static"}
    best_mode, best_a = ("static", static)
    if others:
        m = min(others, key=lambda k: others[k])
        if others[m] < static:
            best_mode, best_a = m, others[m]
    reduction = static / max(best_a, 1e-30)
    recommended = (best_mode != "static") and (reduction >= MIN_REDUCTION)
    if not recommended:
        best_mode, best_a = "static", static
    cls = classify_residual(best_a)
    advice = _ADVICE[cls]
    if not recommended:
        advice = ("No rotation mode reduces the anisotropy by the "
                  f"{MIN_REDUCTION:.2f} factor that makes a turntable worth its "
                  "cost here, so the static arm is kept. " + advice)
    from .geometry_symmetry import candidate_angles
    cand = candidate_angles(order=int(rotational_order),
                            step_deg=float(candidate_step_deg))
    return Recommendation(
        mode=best_mode, actuator_class=cls, residual=best_a,
        static_residual=static, reduction_factor=reduction,
        rotation_recommended=bool(recommended), advice=advice,
        candidate_angles_deg=tuple(float(a) for a in cand), spectrum=spec)
