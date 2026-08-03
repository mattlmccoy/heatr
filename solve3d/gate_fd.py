"""The ported finite-difference / subgradient gate machinery.

Pure numpy, so it runs in both environments and can be unit-tested against
analytic functions before it is ever pointed at the solver (checklist item 2,
layered bisect).

Every number here is READ from solve3d/results/phase_b_protocol.json, which was
pre-registered in Task 0 from FROZEN_CONVENTIONS_2D.md section 5. This module
deliberately holds no threshold of its own -- a gate that carries its own copy
of the numbers can drift from the frozen convention without anyone noticing.

Semantic template: fgm_solve_campaign/adjoint2d/gate_rho.py (READ ONLY).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"


@lru_cache(maxsize=1)
def protocol() -> dict:
    p = RESULTS / "phase_b_protocol.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Phase B Task 0 pre-registers the protocol BEFORE "
            "any gradient is gated; thresholds are never invented at gate time.")
    return json.loads(p.read_text())


_PR = protocol()
EPSILONS: tuple[float, ...] = tuple(_PR["fd"]["epsilons"])
PASS_REL_ERR: float = _PR["thresholds"]["pass_rel_err"]
SUBGRADIENT_PASS_REL_ERR: float = _PR["thresholds"]["subgradient_pass_rel_err"]
TRANSPOSE_REL_ERR: float = _PR["thresholds"]["transpose_rel_err"]


# --------------------------------------------------------------------------- #
# Probes (FROZEN_CONVENTIONS_2D.md section 5 item 3; adjoint2d/gate_rho._probe_dirs)
# --------------------------------------------------------------------------- #
def probe_directions(g: np.ndarray, mask: np.ndarray, seed: int = 7) -> list[dict]:
    """The four pre-registered probes, as unit vectors supported on `mask`.

    max_sensitivity_cell -- argmax |g| inside the mask; the probe most likely to
                            expose a wrong term.
    random_cell          -- a fixed pseudo-random in-part cell; guards against
                            the max cell being special.
    random_direction     -- a random unit direction over all in-part cells.
    gradient_direction   -- the analytic gradient itself, normalized; the
                            direction the optimizer will actually move in, and
                            the one with the largest signal-to-floor ratio.

    The filter-smooth probe of the 2-D protocol is deliberately absent: there is
    no filter in the Phase B chain (see the protocol's checklist item 8). It
    returns in Phase C with the filter.
    """
    g = np.asarray(g, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if g.shape != mask.shape:
        raise ValueError("probe_directions: g and mask must have one shape")
    if not mask.any():
        raise ValueError("probe_directions: empty mask")
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask)

    gm = np.where(mask, np.abs(g), -np.inf)
    i_max = int(np.argmax(gm))
    d_max = np.zeros_like(g)
    d_max[i_max] = 1.0

    i_rnd = int(idx[rng.integers(idx.size)])
    d_rnd = np.zeros_like(g)
    d_rnd[i_rnd] = 1.0

    d_dir = np.zeros_like(g)
    u = rng.standard_normal(idx.size)
    d_dir[idx] = u / np.linalg.norm(u)

    d_grad = np.zeros_like(g)
    gin = g[idx]
    n_g = float(np.linalg.norm(gin))
    if n_g > 0.0:
        d_grad[idx] = gin / n_g
    else:                                   # dead gradient: fall back, and say so
        d_grad[idx] = u / np.linalg.norm(u)

    return [
        {"name": "max_sensitivity_cell", "direction": d_max, "index": i_max},
        {"name": "random_cell", "direction": d_rnd, "index": i_rnd},
        {"name": "random_direction", "direction": d_dir, "index": None},
        {"name": "gradient_direction", "direction": d_grad, "index": None,
         "gradient_norm": n_g},
    ]


# --------------------------------------------------------------------------- #
# Central-difference sweep
# --------------------------------------------------------------------------- #
def sweep(J, x0: np.ndarray, direction: np.ndarray, analytic_dir: float,
          x_scale: float, epsilons: tuple[float, ...] = EPSILONS) -> dict:
    """Central differences along `direction`, epsilon swept over the frozen set.

    The step is `eps * x_scale`, i.e. epsilon is RELATIVE to the design
    magnitude, matching the 2-D convention (adjoint_core.fd_central uses
    `rel_eps * abs(s[dof])` for a single dof). `x_scale` is therefore the
    caller's job: |x[i]| for a single-cell probe, a representative in-part
    magnitude for a direction probe.

    Returns the whole V (every epsilon), the best relative error, and the
    MEASURED evaluation floor (checklist item 6): in the roundoff-dominated tail
    the central-difference error is floor/(2 eps), so 2*eps*abs_err at the two
    smallest epsilons estimates the objective's absolute floor.
    """
    x0 = np.asarray(x0, dtype=float)
    direction = np.asarray(direction, dtype=float)
    rows = []
    for eps in epsilons:
        h = float(eps) * float(x_scale)
        if h <= 0.0:
            continue
        jp = float(J(x0 + h * direction))
        jm = float(J(x0 - h * direction))
        fd = (jp - jm) / (2.0 * h)
        abs_err = abs(fd - analytic_dir)
        den = abs(analytic_dir)
        rows.append({"eps": float(eps), "step": h, "fd": fd,
                     "abs_err": abs_err,
                     "rel_err": (abs_err / den) if den > 0 else float("inf"),
                     "J_plus": jp, "J_minus": jm})
    if not rows:
        raise ValueError("sweep: no usable epsilon (x_scale must be > 0)")
    best = min(rows, key=lambda r: r["rel_err"])
    # floor estimate from the two SMALLEST epsilons (the roundoff tail)
    tail = sorted(rows, key=lambda r: r["eps"])[:2]
    floor = float(np.mean([2.0 * r["step"] * r["abs_err"] for r in tail]))
    return {"analytic_directional": float(analytic_dir),
            "best_rel_err": best["rel_err"], "best_abs_err": best["abs_err"],
            "best_eps": best["eps"], "best_fd": best["fd"],
            "evaluation_floor_estimate": floor,
            "x_scale": float(x_scale), "sweep": rows}


def verdict(rel_err: float) -> dict:
    """Label a relative error at BOTH standards (checklist item 5 requires the
    count at both, so neither is reported alone)."""
    r = float(rel_err)
    return {"rel_err": r,
            "pass_preferred": bool(r <= PASS_REL_ERR),
            "pass_subgradient": bool(r <= SUBGRADIENT_PASS_REL_ERR),
            "pass_rel_err": PASS_REL_ERR,
            "subgradient_pass_rel_err": SUBGRADIENT_PASS_REL_ERR}


def run_probes(J, x0: np.ndarray, grad: np.ndarray, mask: np.ndarray,
               seed: int = 7, x_scale_direction: float | None = None) -> dict:
    """The full per-layer gate: four probes, each swept, each labelled.

    Also reports the scatter of ABSOLUTE error against analytic magnitude
    (checklist item 7): a relative-error miss on a probe whose derivative is
    tiny is a floor artifact, not a wrong gradient, and the report must let a
    reader see that rather than take the relative number alone.
    """
    x0 = np.asarray(x0, dtype=float)
    grad = np.asarray(grad, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    scale_dir = (float(x_scale_direction) if x_scale_direction is not None
                 else float(np.mean(np.abs(x0[mask]))))
    out = {"probes": {}, "seed": seed}
    for pr in probe_directions(grad, mask, seed=seed):
        d = pr["direction"]
        i = pr["index"]
        x_scale = abs(float(x0[i])) if i is not None else scale_dir
        an = float(np.dot(grad, d))
        s = sweep(J, x0, d, an, x_scale)
        s.update(verdict(s["best_rel_err"]))
        s["index"] = (int(i) if i is not None else None)
        out["probes"][pr["name"]] = s
    rels = [p["best_rel_err"] for p in out["probes"].values()]
    abss = [p["best_abs_err"] for p in out["probes"].values()]
    ans = [abs(p["analytic_directional"]) for p in out["probes"].values()]
    floors = [p["evaluation_floor_estimate"] for p in out["probes"].values()]
    out.update({
        "worst_best_rel_err": float(max(rels)),
        "n_pass_preferred": int(sum(p["pass_preferred"] for p in out["probes"].values())),
        "n_pass_subgradient": int(sum(p["pass_subgradient"] for p in out["probes"].values())),
        "n_probes": len(rels),
        "abs_err_range": [float(min(abss)), float(max(abss))],
        "analytic_magnitude_range": [float(min(ans)), float(max(ans))],
        "evaluation_floor_range": [float(min(floors)), float(max(floors))],
        "all_pass_subgradient": bool(all(p["pass_subgradient"]
                                         for p in out["probes"].values())),
        "all_pass_preferred": bool(all(p["pass_preferred"]
                                       for p in out["probes"].values())),
    })
    return out


# --------------------------------------------------------------------------- #
# Transpose exactness (checklist item 8)
# --------------------------------------------------------------------------- #
def transpose_residual(forward_op, transpose_op, x: np.ndarray,
                       y: np.ndarray) -> dict:
    """Dot-product identity <A x, y> == <x, A^T y>, the bisect that separates a
    chain-rule error from a property of the forward. Threshold 1e-10 (2-D
    measured 4.19e-16 worst)."""
    lhs = float(np.dot(np.asarray(forward_op(x)).ravel(), np.asarray(y).ravel()))
    rhs = float(np.dot(np.asarray(x).ravel(), np.asarray(transpose_op(y)).ravel()))
    den = max(abs(lhs), abs(rhs), 1e-300)
    rel = abs(lhs - rhs) / den
    return {"lhs": lhs, "rhs": rhs, "rel_err": rel,
            "tolerance": TRANSPOSE_REL_ERR,
            "pass": bool(rel <= TRANSPOSE_REL_ERR)}
