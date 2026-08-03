"""Phase C shape-fidelity objective: the symmetric control and the ASYMMETRIC
primary.

Pure numpy on nodal arrays, so it runs in either environment and can be checked
in closed form before it is ever wired to a solve.

WHY TWO OBJECTIVES
------------------
Matt's recorded refinement (spec commit d298c6d, Phase C OBJECTIVE REFINEMENT):
"the goal is dense IF AND ONLY IF in-bounds - out-of-bounds melt (bed growth)
is the hard penalty side; in-bounds under-density is a SOFT trade with a floor
near 80-90% density. The symmetric (phi-chi)^2 does not encode this."

  SYMMETRIC CONTROL   J = sum_i vol_i (phi_i - chi_i)^2
      The Phase B functional and the 2-D lane's. It is run in EVERY arm, not
      because it is right for this goal but because it is the only quantity
      this campaign shares with the 2-D lane. Cross-lane comparability is worth
      one extra scalar per arm.

  ASYMMETRIC PRIMARY  J = W_out sum_i vol_i max(phi_i - chi_i, 0)^2
                        + W_in  sum_i vol_i max(phi_floor chi_i - phi_i, 0)^2

      Two hinges, and each encodes one half of the sentence:
        * out-of-bounds melt is penalized from the first increment, quadratically;
        * in-bounds density is penalized ONLY below phi_floor, so 85-100 %
          density is free and the optimizer is never pushed to over-melt a part
          it has already made dense enough. That is the "soft trade with a
          floor" made concrete.

      phi_floor = 0.85 is the middle of the stated 80-90 % band.
      W_out / W_in = 10 is PRE-REGISTERED, not tuned: the hinge already does the
      "soft" work, so the ratio only has to say "hard versus soft", and one
      order of magnitude is the coarsest non-trivial way to say it without
      implying a calibration nobody has measured. A 3x sensitivity arm is run;
      if the ranking depends on the ratio, that is reported as the finding.

NOT AN OBJECTIVE: sigma_T. It is a reported flatness diagnostic, per the spec
and the Phase A close-out verdict rule.

SUBGRADIENTS. Both hinges are kinks at zero. The derivative takes the
forward's own side (zero exactly at the kink), which is the same rule the 2-D
lane applies to its clips and the same rule Phase B applies to the melt-window
clip. The melt fraction phi is itself a clipped quantity, so the composed
objective is already a subgradient object; that is why the Phase B protocol
carries the 1e-5 subgradient standard alongside the 1e-6 preferred one.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"


@lru_cache(maxsize=1)
def preregistration() -> dict:
    p = RESULTS / "phase_c_preregistration.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Phase C Task 0 pre-registers the objective "
            "weights BEFORE the objective exists; they are never chosen here.")
    return json.loads(p.read_text())


_A = preregistration()["objective"]["asymmetric"]
PHI_FLOOR: float = float(_A["phi_floor"])
W_OUT_OVER_W_IN: float = float(_A["w_out_over_w_in"])
W_SENSITIVITY: float = float(_A["sensitivity_arm_w_out_over_w_in"])


# --------------------------------------------------------------------------- #
# Symmetric control
# --------------------------------------------------------------------------- #
def j_symmetric(phi, chi, vol) -> float:
    phi, chi, vol = (np.asarray(a, dtype=float) for a in (phi, chi, vol))
    return float(np.sum(vol * (phi - chi) ** 2))


def dj_symmetric_dphi(phi, chi, vol) -> np.ndarray:
    phi, chi, vol = (np.asarray(a, dtype=float) for a in (phi, chi, vol))
    return 2.0 * vol * (phi - chi)


# --------------------------------------------------------------------------- #
# Asymmetric primary
# --------------------------------------------------------------------------- #
def _errors(phi, chi, phi_floor: float):
    e_out = np.maximum(phi - chi, 0.0)                    # melt outside bounds
    e_in = np.maximum(phi_floor * chi - phi, 0.0)         # under-dense in bounds
    return e_out, e_in


def j_asymmetric(phi, chi, vol, w_ratio: float | None = None,
                 phi_floor: float | None = None) -> float:
    phi, chi, vol = (np.asarray(a, dtype=float) for a in (phi, chi, vol))
    w = W_OUT_OVER_W_IN if w_ratio is None else float(w_ratio)
    f = PHI_FLOOR if phi_floor is None else float(phi_floor)
    e_out, e_in = _errors(phi, chi, f)
    return float(w * np.sum(vol * e_out ** 2) + np.sum(vol * e_in ** 2))


def dj_asymmetric_dphi(phi, chi, vol, w_ratio: float | None = None,
                       phi_floor: float | None = None) -> np.ndarray:
    phi, chi, vol = (np.asarray(a, dtype=float) for a in (phi, chi, vol))
    w = W_OUT_OVER_W_IN if w_ratio is None else float(w_ratio)
    f = PHI_FLOOR if phi_floor is None else float(phi_floor)
    e_out, e_in = _errors(phi, chi, f)
    # subgradient: exactly zero at each kink, the forward's own side
    return 2.0 * w * vol * e_out - 2.0 * vol * e_in


# --------------------------------------------------------------------------- #
# Diagnostics reported alongside, never optimized
# --------------------------------------------------------------------------- #
def split_asymmetric(phi, chi, vol, w_ratio: float | None = None,
                     phi_floor: float | None = None) -> dict:
    """The two halves separately, so a report can say WHERE the cost sits."""
    phi, chi, vol = (np.asarray(a, dtype=float) for a in (phi, chi, vol))
    w = W_OUT_OVER_W_IN if w_ratio is None else float(w_ratio)
    f = PHI_FLOOR if phi_floor is None else float(phi_floor)
    e_out, e_in = _errors(phi, chi, f)
    j_out = float(w * np.sum(vol * e_out ** 2))
    j_in = float(np.sum(vol * e_in ** 2))
    v_part = float(np.sum(vol * chi))
    return {"J_asym": j_out + j_in, "J_out_of_bounds": j_out,
            "J_in_bounds_deficit": j_in, "w_ratio": w, "phi_floor": f,
            "out_of_bounds_volume_m3": float(np.sum(vol * (e_out > 0.0))),
            "out_of_bounds_melt_fraction_of_part":
                (float(np.sum(vol * e_out)) / v_part) if v_part > 0 else float("nan"),
            "in_bounds_below_floor_fraction":
                (float(np.sum(vol * chi * (e_in > 0.0))) / v_part)
                if v_part > 0 else float("nan")}


OBJECTIVES = {
    "symmetric": (j_symmetric, dj_symmetric_dphi),
    "asymmetric": (j_asymmetric, dj_asymmetric_dphi),
}
