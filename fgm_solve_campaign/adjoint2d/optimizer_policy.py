"""Which optimizer runs, as a function of the OBJECTIVE CLASS.

THE MEASUREMENT THIS ENCODES, and its two halves.

  CONSTRAINED class (the hinge / asymmetric dense-if-and-only-if-in-bounds
  objective). At a matched 40 forward-equivalents, single cold start, the same
  filtered design variable and the same box, the method of moving asymptotes
  beat L-BFGS-B on 4 of 5 shapes: triangle by 21.4 percent, cross 10.4, hexagon
  2.1, L_shape 0.5, losing on the square by 2.1 percent
  (DENSE_IFF_INBOUNDS_REPORT.md Section 8). The mechanism measured there: the
  method of moving asymptotes takes one design update per evaluation with no
  line search, so a 20-evaluation pool buys 20 design updates, while L-BFGS-B
  shares its pool with its line search.

  SMOOTH class (the melt-region shape-fidelity objective). The same comparison
  under the smoothed-Heaviside projection continuation was 4 of 6 WITH TWO
  CATASTROPHIC FAILURES (MMA_RETEST_REPORT.md Section 1 item 3), and the
  filter-only production recipe improves on 74.0 percent of its follow-up
  evaluations under L-BFGS-B. There is no measured reason to move the smooth
  class, so it stays on L-BFGS-B.

WHAT THIS IS NOT. It is not a claim that the method of moving asymptotes is the
better optimizer in general. It is a per-class default with a per-class
measurement behind it, and an explicit configuration value always wins over it,
in either direction, with the source recorded in the run record.

Acronyms: MMA = the method of moving asymptotes (Svanberg 1987). L-BFGS-B =
limited-memory Broyden-Fletcher-Goldfarb-Shanno with box constraints.
"""
from __future__ import annotations

OPTIMIZERS = ("lbfgsb", "mma")
OPTIMIZER_CHOICES = ("auto", "lbfgsb", "mma")

_SMOOTH_NAMES = frozenset({"j_phi", "melt", "shape", "phi", "melt_region"})
_CONSTRAINED_NAMES = frozenset({"j_asym", "asym", "hinge", "dense_iff_inbounds"})

_CLASS_DEFAULT = {"smooth": "lbfgsb", "constrained": "mma"}

_CLASS_REASON = {
    "constrained": (
        "constrained (hinge / asymmetric) objective class: the method of "
        "moving asymptotes beat L-BFGS-B on 4 of 5 shapes at a matched 40 "
        "forward-equivalents (DENSE_IFF_INBOUNDS_REPORT.md Section 8)"),
    "smooth": (
        "smooth melt-region objective class: the method of moving asymptotes "
        "was not a clean win there (4 of 6 with two catastrophic failures, "
        "MMA_RETEST_REPORT.md Section 1 item 3), so L-BFGS-B stays the "
        "default"),
}


def objective_class(objective: str) -> str:
    """Classify an objective name as "smooth" or "constrained"."""
    name = str(objective).strip().lower()
    if name in _SMOOTH_NAMES:
        return "smooth"
    if name in _CONSTRAINED_NAMES:
        return "constrained"
    raise ValueError(
        f"unknown objective {objective!r}; known smooth names "
        f"{sorted(_SMOOTH_NAMES)}, known constrained names "
        f"{sorted(_CONSTRAINED_NAMES)}. The class is not guessed, because the "
        f"optimizer default depends on it.")


def resolve_optimizer(objective: str, override: str | None = None) -> dict:
    """Pick the optimizer for one objective, recording where the choice came from.

    Args:
        objective: the objective name (see `objective_class`).
        override: a configured optimizer, or "auto" / None to use the policy.

    Returns:
        {"optimizer", "objective_class", "source", "reason"}.

    Raises:
        ValueError: on an unknown objective or an unknown override.
    """
    cls = objective_class(objective)
    if override is None or str(override).strip().lower() == "auto":
        return {"optimizer": _CLASS_DEFAULT[cls], "objective_class": cls,
                "source": "policy", "reason": _CLASS_REASON[cls]}
    ov = str(override).strip().lower()
    if ov not in OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer {override!r}; allowed: "
            f"{list(OPTIMIZER_CHOICES)} ('auto' applies the per-class policy)")
    return {"optimizer": ov, "objective_class": cls, "source": "config",
            "reason": (f"optimizer {ov!r} set explicitly in the configuration; "
                       f"it overrides the {cls} class default "
                       f"{_CLASS_DEFAULT[cls]!r}")}
