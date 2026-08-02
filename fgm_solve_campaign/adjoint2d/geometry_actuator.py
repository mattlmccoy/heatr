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
           "A_PHYSICAL_LIMIT", "MIN_REDUCTION",
           "prop_inverse_stand_in", "spectrum_v2_from_cfg",
           "PROP_INVERSE_MAGNITUDE", "PROP_INVERSE_BASELINE",
           "recommend_v2", "classify_v2", "CALIBRATION_V2",
           "A2_MODE_SUFFICES", "A2_PHYSICAL_LIMIT", "MIN_REDUCTION_V2",
           "DEFAULT_V2_BASIS"]

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
    basis: str = "uniform"
    version: int = 1

    def as_json(self) -> dict:
        return {"mode": self.mode, "actuator_class": self.actuator_class,
                "basis": self.basis, "classifier_version": int(self.version),
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


# ---------------------------------------------------------------------------
# CLASSIFIER VERSION 2: predict against the SOLVED arm, not the uniform arm
# ---------------------------------------------------------------------------
#
# THE MISS THAT MOTIVATES IT. `GEOMETRY_GENERALIZATION_REPORT.md` Section 6.2
# measured, on a novel eight-tooth gear: version 1 called MODE_SUFFICES via
# continuous rotation; rotation did exactly what the anisotropy predicted (the
# uniform arm's J fell 207.89 -> 138.59, a 33 percent cut at 8 percent less
# absorbed power); and the SOLVED STATIC MAP won anyway, 132.36 / IoU 0.8458
# against 138.59 / 0.8273. Version 1 compares an actuator against NO actuator.
# The pipeline's real competitor is a solved dopant map, and on a nearly
# radially symmetric part a solved static map is already very good.
#
# THE CHANGE, in one line: measure each mode's residual anisotropy with a MAP
# INJECTED instead of with the uniform map s = 1. Two variants of the injected
# map, both implemented and both measured, because they differ by roughly ten
# minutes of latency per geometry:
#
#   "solved"       the static SOLVED map, from the pipeline's own static arm.
#                  Costs one filtered gradient solve (about 40 forward
#                  equivalents) before the classifier can run at all.
#   "prop_inverse" the FREE variant: the percentile-normalized proportional
#                  inverse of the STATIC kernel, which the version-1 spectrum
#                  already computed, so it costs ZERO additional gradient
#                  solves. It is a stand-in for what a solved map does to the
#                  field, not a claim that it equals one.
#
# Both variants still cost one extra averaged-kernel evaluation per mode (the
# "one forward per mode" of the report's named fix), because the kernel is an
# OPERATOR in the design map and not a fixed array.

# The stand-in has NO free parameter that was tuned on the outcome: full swing
# about the mid box, which is the maximum-contrast member of the
# proportional-inverse family, with the production smoothing and dead band of
# `control.proportional_inverse_map`. Sensitivity to the magnitude is measured
# and reported rather than fitted.
PROP_INVERSE_MAGNITUDE = 1.0
PROP_INVERSE_BASELINE = 0.5


def prop_inverse_stand_in(Q_static: np.ndarray, part_mask: np.ndarray,
                          magnitude: float = PROP_INVERSE_MAGNITUDE,
                          baseline: float = PROP_INVERSE_BASELINE) -> np.ndarray:
    """The zero-gradient-solve stand-in for a solved static dopant map.

    Delegates to the production `control.proportional_inverse_map` rather than
    re-deriving the rule: two copies of a control law drift. Outside the part
    the saturation is held at the nominal 1, which is the convention every arm
    in the campaign uses, so the stand-in cannot silently change the geometry.
    """
    from .control import proportional_inverse_map
    return proportional_inverse_map(np.asarray(Q_static, dtype=float),
                                    np.asarray(part_mask, dtype=bool),
                                    magnitude=float(magnitude),
                                    baseline=float(baseline))


def spectrum_v2_from_cfg(cfg: dict, rotational_order: int,
                         solved_static_map: np.ndarray | None = None,
                         step_deg: float = REFERENCE_STEP_DEG,
                         prop_magnitude: float = PROP_INVERSE_MAGNITUDE,
                         log=None) -> dict:
    """Every mode's residual anisotropy under the uniform map AND under a map.

    Returns, under `residual`, three sub-dictionaries keyed by the injected
    design map:

      "uniform"      s = 1, which is exactly the version-1 spectrum;
      "prop_inverse" the free stand-in built from THIS geometry's static kernel;
      "solved"       `solved_static_map`, present only when one is supplied.

    The static kernel is evaluated once up front to build the stand-in, then
    every mode is built once and evaluated against each available map, so the
    electro-quasi-static solve count is (number of maps) times the version-1
    count and the kernel ASSEMBLY count is unchanged.
    """
    from .rot_kernel import AveragedKernel

    ref = reference_angles(step_deg)
    cache: dict = {}
    modes = mode_names(int(rotational_order))

    # pass 1: the static kernel under the uniform map, which is what the free
    # stand-in is built from. One angle, so this is the cheapest kernel there is.
    k0 = AveragedKernel.build(copy.deepcopy(cfg), angles=mode_angles("static"))
    pm = k0.case0.part_mask
    _Qa0, Qb0 = k0.averaged_Q(np.ones(pm.shape))
    s_pi = prop_inverse_stand_in(Qb0, pm, magnitude=float(prop_magnitude))
    del k0

    maps: dict[str, np.ndarray] = {"uniform": np.ones(pm.shape),
                                   "prop_inverse": s_pi}
    if solved_static_map is not None:
        sm = np.asarray(solved_static_map, dtype=float)
        if sm.shape != pm.shape:
            raise ValueError(f"solved map shape {sm.shape} does not match the "
                             f"part {pm.shape}")
        maps["solved"] = sm

    out: dict = {"residual": {k: {} for k in maps}, "kernels": {},
                 "reference_angles_deg": list(ref),
                 "prop_inverse_map": s_pi,
                 "prop_inverse_magnitude": float(prop_magnitude),
                 "metric": "occupancy-weighted azimuthal anisotropy, "
                           "adjoint2d.geometry_actuator.azimuthal_anisotropy, "
                           "evaluated with a design map injected (classifier v2)"}
    for mode in modes:
        ang = mode_angles(mode, step_deg)
        kern = AveragedKernel.build(copy.deepcopy(cfg), angles=ang)
        for mname, s in maps.items():
            _Qa, Qb = kern.averaged_Q(s)
            a = azimuthal_anisotropy(Qb, pm, ref, cache)
            out["residual"][mname][mode] = float(a)
            if mname == "uniform":
                out["kernels"][mode] = np.array(Qb, dtype=float)
            elif mname == "solved":
                out["kernels"][f"{mode}__solved"] = np.array(Qb, dtype=float)
        if log is not None:
            log(f"  mode {mode:11s} angles {len(ang):2d}  " + "  ".join(
                f"{m} {out['residual'][m][mode]:.4f}" for m in maps))
        del kern
    out["part_mask"] = pm
    return out


# ---------------------------------------------------------------------------
# the version-2 bands, recalibrated on the SAME evidence set
# ---------------------------------------------------------------------------
#
# THE EVIDENCE SET, seven measured points, each one a comparison of the best
# ROTATING arm against a SOLVED STATIC arm (see CALIBRATION_V2 for the sources).
# It is the five rotation-campaign shapes plus the two novel geometries the
# generalization pass ran end to end, which are the only out-of-sample points
# that exist.
#
# WHAT THE MEASUREMENT SHOWED, and it is the finding of this layer. Under the
# FREE stand-in the reduction factor separates the two outcome classes:
#
#     rotation fails   gear8 1.115   L_shape 1.115   T_shape 1.000
#     rotation wins    keyhole 1.137   star 1.328   square 1.424   cross 1.610
#
# so a threshold anywhere in (1.115, 1.137) is 7 of 7. Under the EXPENSIVE
# one-forward variant (the report's literal named fix, the static SOLVED map
# injected) it does NOT separate at any threshold:
#
#     T_shape 1.000 < keyhole 1.118 < L_shape 1.151 < gear8 1.365
#                   < star 1.433 < cross 1.497 < square 1.873
#
# the keyhole, whose rotation win is the largest in the campaign (J 168.35 ->
# 8.08), sits BELOW two failures, and the best any single threshold can do is 6
# of 7. Version 1, for reference, cannot separate at all: gear8's uniform-map
# reduction is 1.947, above three of the four winners. MEASURED, this pass,
# `out_intake/*_v2.json`.
#
# THE MARGIN IS THIN AND IT IS NAMED. 1.137 against 1.115 is 2.0 percent, which
# is thinner than version 1's 8 percent margin on the star. It widens to 5.0
# percent at stand-in magnitude 1.5 and it DISAPPEARS at magnitude 0.5 (gear8
# 1.416 against keyhole 1.155, the wrong order), so the stand-in must have at
# least full swing. That is not a fitted choice: every solved static map in the
# library spans the whole [0, 1] box (measured in-part minima 0.000 to 0.229,
# maxima 1.000), and a magnitude-0.5 map only moves saturation over [0.25,
# 0.75], so it under-represents the authority a real solved map has.
A2_MODE_SUFFICES = 0.50
A2_PHYSICAL_LIMIT = 0.80
MIN_REDUCTION_V2 = 1.125
DEFAULT_V2_BASIS = "prop_inverse"

CALIBRATION_V2 = (
    {"shape": "square", "rotational_order": 4, "outcome": "rotation_wins",
     "A2_static": 0.3650, "A2_best": 0.2564, "best_mode": "continuous",
     "source": "CONTINUOUS_ROTATION_REPORT S1: solved static 25.56 / 0.9804 -> "
               "rotating 13.93 / 1.0000, -45.5 percent J"},
    {"shape": "cross", "rotational_order": 4, "outcome": "rotation_wins",
     "A2_static": 0.6632, "A2_best": 0.4120, "best_mode": "continuous",
     "source": "CONTINUOUS_ROTATION_REPORT S1: 147.03 / 0.8514 -> 34.82 / "
               "0.9866, -76.3 percent J"},
    {"shape": "star", "rotational_order": 5, "outcome": "rotation_wins",
     "A2_static": 0.5818, "A2_best": 0.4380, "best_mode": "continuous",
     "source": "CONTINUOUS_ROTATION_REPORT S1: 106.28 / 0.7870 -> 24.46 / "
               "0.9527, -77.0 percent J; the win is the ROTATION, the map is "
               "measurably inert"},
    {"shape": "keyhole", "rotational_order": 1, "outcome": "rotation_wins",
     "A2_static": 0.5412, "A2_best": 0.4759, "best_mode": "continuous",
     "source": "GEOMETRY_GENERALIZATION_REPORT S6.1: solved static 4 bpp "
               "168.35 / 0.8163 -> solved continuous 4 bpp 8.08 / 0.9753"},
    {"shape": "gear8", "rotational_order": 8, "outcome": "rotation_fails",
     "A2_static": 0.3149, "A2_best": 0.2823, "best_mode": "continuous",
     "source": "GEOMETRY_GENERALIZATION_REPORT S6.2: solved static 4 bpp "
               "132.36 / 0.8458 BEATS solved continuous 4 bpp 138.59 / 0.8273. "
               "Rotation cut the UNIFORM arm's J by 33 percent and still lost "
               "to the map; this is the point version 1 got wrong"},
    {"shape": "T_shape", "rotational_order": 1, "outcome": "rotation_fails",
     "A2_static": 1.1245, "A2_best": 1.1245, "best_mode": "continuous",
     "source": "CONTINUOUS_ROTATION_REPORT S1: 522.28 (HORIZON) / 0.5356 -> "
               "467.33 / 0.5516, -10.5 percent J with 42.9 percent of the part "
               "still unmelted; the perpendicular-limb penalty is NOT erased"},
    {"shape": "L_shape", "rotational_order": 1, "outcome": "rotation_fails",
     "A2_static": 1.2083, "A2_best": 1.0834, "best_mode": "continuous",
     "source": "CONTINUOUS_ROTATION_REPORT S1: 376.94 / 0.6693 -> 352.97 / "
               "0.6581, better J and worse IoU, effectively a tie"},
)

_ADVICE_V2 = {
    "MAP_SUFFICES": (
        "THE MAP SUFFICES: no turntable. After a dopant map has flattened the "
        "static heating, no rotation mode reduces the residual anisotropy by "
        f"the {MIN_REDUCTION_V2:.3f} factor that would make a turntable worth "
        "its cost, and what is left is already inside the band the rotation "
        "campaign's winners sat in. Spend the budget on the map, not on the "
        "actuator. This is the class version 1 could not express: it compared "
        "the actuator against NO actuator, so it recommended a turntable for "
        "the eight-tooth gear, whose solved static map then beat every rotating "
        "arm (132.36 against 138.59 J)."),
    "MODE_SUFFICES": (
        "The mode still helps AFTER the map has done its work, and the residual "
        "it leaves is inside the band the rotation campaign's winners sat in "
        "(square 0.256, cross 0.412, star 0.438, keyhole 0.476 on this "
        "measurement). Expect the actuator to carry most of the remaining gain; "
        "solve the dopant map against that mode's averaged kernel for the rest."),
    "MAP_PLUS_MODE": (
        "The mode still helps after the map, but it leaves real anisotropy, "
        "between the winners' band and the failures' band. Solve the map "
        "against this mode's averaged kernel and expect the map to be load "
        "bearing rather than a finish."),
    "PHYSICAL_LIMIT": (
        "The residual anisotropy after a map is in the band of the two shapes "
        "rotation did NOT rescue (T_shape 1.124, L_shape 1.208 on this "
        "measurement). Expect a physical limit: a radial kernel melts a rounded "
        "blob and this part is not close to radially symmetric, and a dopant "
        "map cannot fix it either. The best results on record for that class "
        "came from SEQUENTIAL DWELL, an ordered hold that melts one limb and "
        "then the other (SEQUENTIAL_DWELL_REPORT: L_shape 396.20 -> 276.39 J, "
        "IoU 0.6484 -> 0.7161; T_shape 422.28 -> 343.02 J). Neither reached the "
        "SOLVED class, so this is a partial rescue, not a fix."),
}


def classify_v2(a: float, rotation_recommended: bool) -> str:
    """The four version-2 classes.

    The extra class over version 1 is MAP_SUFFICES, which is the answer to the
    question version 1 could not ask: the heating is fine once a map has acted,
    so do not buy a turntable. `a` is the residual the recommendation rests on:
    the best mode's residual when rotation is recommended, and the STATIC
    residual (what the map alone leaves) when it is not.
    """
    v = float(a)
    if v > A2_PHYSICAL_LIMIT:
        return "PHYSICAL_LIMIT"
    if not rotation_recommended:
        return "MAP_SUFFICES"
    return "MODE_SUFFICES" if v <= A2_MODE_SUFFICES else "MAP_PLUS_MODE"


def recommend_v2(spectrum: dict, rotational_order: int,
                 candidate_step_deg: float = 15.0,
                 basis: str = DEFAULT_V2_BASIS) -> Recommendation:
    """Pick the actuator mode against a MAP, not against the uniform arm.

    `spectrum` is `{mode: residual anisotropy}` measured with a design map
    INJECTED into each mode's averaged kernel, which is what
    `spectrum_v2_from_cfg` returns under `residual[basis]`. `basis` names which
    injected map it was measured on and is carried into the result so two
    bases can never be silently mixed; the default is the FREE stand-in, which
    is the variant the seven-point evidence set says to use.
    """
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
    recommended = (best_mode != "static") and (reduction >= MIN_REDUCTION_V2)
    if not recommended:
        best_mode, best_a = "static", static
    cls = classify_v2(best_a, recommended)
    from .geometry_symmetry import candidate_angles
    cand = candidate_angles(order=int(rotational_order),
                            step_deg=float(candidate_step_deg))
    return Recommendation(
        mode=best_mode, actuator_class=cls, residual=best_a,
        static_residual=static, reduction_factor=reduction,
        rotation_recommended=bool(recommended), advice=_ADVICE_V2[cls],
        candidate_angles_deg=tuple(float(a) for a in cand), spectrum=spec,
        basis=str(basis), version=2)
