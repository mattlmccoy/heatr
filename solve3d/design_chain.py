"""Phase C design chain: explicit normalized-convolution filter + smoothed
Heaviside projection.

    v  --F-->  v_f  --P_beta-->  s  --forward-->  T  --J-->  scalar

the 3-D port of FROZEN_CONVENTIONS_2D.md section 1 (`adjoint2d/topopt.py`,
`adjoint2d/design_filter.py`).

WHY AN EXPLICIT MATRIX AND NOT A HELMHOLTZ PDE FILTER
-----------------------------------------------------
The Helmholtz (Lazarov-Sigmund) filter is the usual unstructured-mesh density
filter and would have been less code. It is also SELF-ADJOINT, which makes its
transpose gate trivially true and therefore worthless as evidence: the frozen
checklist's item 8 exists precisely to separate a chain-rule error from a
property of the forward, and a test that cannot fail does not do that. The
explicit matrix here is deliberately NOT symmetric (row normalization and
volume weighting both break symmetry, asserted in
test_filter_matrix_is_not_symmetric), so its transpose gate is a real check.
It is also the literal 2-D convention: a normalized convolution, not a PDE.

THE KERNEL
----------
    w_ij = vol_j * exp(-d_ij^2 / (2 sigma^2))   for d_ij <= 3 sigma
    W_ij = w_ij / sum_j w_ij

on the part cells only, with the value held at the nominal 1.0 outside
(section 4). sigma = FILTER_RADIUS_M = 1.0e-3 m, a PHYSICAL LENGTH read from
the Phase C pre-registration, converted to no grid quantity anywhere.

Two consequences that the tests pin rather than assume:
  * rows sum to 1, so a constant design passes through EXACTLY (partition of
    unity) and the filter cannot dim the map;
  * every row is a convex combination of in-part values, so v in [0,1] gives
    s in [0,1] with NO clip, and therefore no clip subgradient enters the chain
    rule. That is the same reason the 2-D lane chose a normalized convolution
    (their section 1.2).

DEVIATION from the 2-D form, named: their grid is uniform so every cell has the
same area and the weights need no volume factor. A tetrahedral mesh does not,
so the weights carry vol_j. Without it the filter would over-weight wherever
the mesh happens to be fine, which would make the "physical radius" claim false
on a graded mesh.

THE PROJECTION
--------------
tanh projection of Wang, Lazarov and Sigmund, eta = 0.5, beta continuation
1/2/4/8/16 (section 1.3). beta <= 0 is BIT-IDENTICAL to the filter alone, which
is the flag-off switch the frozen conventions require the port to keep.
MMA_RETEST_REPORT.md settles its role: filter-only is the in-grid production
recipe, projection is the robustness arm. Both are run; filter-only is primary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

RESULTS = Path(__file__).resolve().parent / "results"
TRUNCATION_SIGMAS = 3.0
ETA = 0.5


def _prereg() -> dict:
    p = RESULTS / "phase_c_preregistration.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. The filter radius and beta schedule are "
            "PRE-REGISTERED; they are never chosen here.")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- #
# Projection (FROZEN_CONVENTIONS_2D section 1.3, topopt.project)
# --------------------------------------------------------------------------- #
def project(u, beta: float, eta: float = ETA):
    u = np.asarray(u, dtype=float)
    if beta <= 0.0:
        return u.copy()
    tb = np.tanh(beta * eta)
    den = tb + np.tanh(beta * (1.0 - eta))
    return (tb + np.tanh(beta * (u - eta))) / den


def project_prime(u, beta: float, eta: float = ETA):
    u = np.asarray(u, dtype=float)
    if beta <= 0.0:
        return np.ones_like(u)
    tb = np.tanh(beta * eta)
    den = tb + np.tanh(beta * (1.0 - eta))
    t = np.tanh(beta * (u - eta))
    return beta * (1.0 - t * t) / den


# --------------------------------------------------------------------------- #
class DesignChain:
    """The filter matrix and the projection, on one mesh's part cells."""

    def __init__(self, centroids: np.ndarray, volumes: np.ndarray,
                 radius_m: float, beta_schedule: list[float]):
        self.centroids = np.asarray(centroids, dtype=float)
        self.volumes = np.asarray(volumes, dtype=float)
        self.radius_m = float(radius_m)
        self.beta_schedule = list(beta_schedule)
        self.n_design = int(self.centroids.shape[0])
        self.W = self._build()

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, tc, radius_m: float | None = None):
        pr = _prereg()["design_chain"]
        import dolfinx
        mp = dolfinx.mesh.compute_midpoints(
            tc.msh, tc.msh.topology.dim,
            np.arange(tc.ncells, dtype=np.int32))
        part = tc.eqs.part
        return cls(np.asarray(mp)[part], tc.eqs.vol[part],
                   float(radius_m if radius_m is not None
                         else pr["filter_radius_m"]),
                   [float(b) for b in pr["projection"]["beta_schedule"]])

    def _build(self) -> csr_matrix:
        sig = self.radius_m
        cutoff = TRUNCATION_SIGMAS * sig
        tree = cKDTree(self.centroids)
        pairs = tree.query_ball_point(self.centroids, r=cutoff)
        rows, cols, vals = [], [], []
        for i, nb in enumerate(pairs):
            nb = np.asarray(nb, dtype=np.int64)
            d2 = np.sum((self.centroids[nb] - self.centroids[i]) ** 2, axis=1)
            w = self.volumes[nb] * np.exp(-d2 / (2.0 * sig * sig))
            w = w / w.sum()                       # row-normalize: partition of unity
            rows.append(np.full(nb.size, i, dtype=np.int64))
            cols.append(nb)
            vals.append(w)
        return csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_design, self.n_design))

    # ------------------------------------------------------------------ #
    def filter_apply(self, v: np.ndarray) -> np.ndarray:
        return self.W @ np.asarray(v, dtype=float)

    def filter_transpose(self, g: np.ndarray) -> np.ndarray:
        return self.W.T @ np.asarray(g, dtype=float)

    def design_to_map(self, v: np.ndarray, beta: float) -> np.ndarray:
        vf = self.filter_apply(v)
        return vf if beta <= 0.0 else project(vf, beta)

    def design_jvp(self, v: np.ndarray, d: np.ndarray, beta: float) -> np.ndarray:
        wd = self.filter_apply(d)
        if beta <= 0.0:
            return wd
        return project_prime(self.filter_apply(v), beta) * wd

    def design_vjp(self, v: np.ndarray, g: np.ndarray, beta: float) -> np.ndarray:
        g = np.asarray(g, dtype=float)
        if beta > 0.0:
            g = project_prime(self.filter_apply(v), beta) * g
        return self.filter_transpose(g)

    # ------------------------------------------------------------------ #
    def kernel_report(self) -> dict:
        """Measured properties of the discrete kernel, so 'physical radius' is
        a measurement rather than a claim."""
        W = self.W.tocoo()
        d = np.linalg.norm(self.centroids[W.row] - self.centroids[W.col], axis=1)
        # isotropic 3-D Gaussian: E[d^2] = 3 sigma^2
        e_d2 = float(np.sum(W.data * d * d) / self.n_design)
        counts = np.diff(self.W.indptr)
        return {"radius_m": self.radius_m,
                "truncation_sigmas": TRUNCATION_SIGMAS,
                "measured_std_m": float(np.sqrt(e_d2 / 3.0)),
                "mean_neighbours": float(counts.mean()),
                "min_neighbours": int(counts.min()),
                "max_neighbours": int(counts.max()),
                "n_design": self.n_design,
                "mean_cell_size_m": float(np.mean(self.volumes) ** (1.0 / 3.0)),
                "row_sum_max_dev_from_one":
                    float(np.max(np.abs(np.asarray(self.W.sum(axis=1)).ravel() - 1.0))),
                "asymmetry_max": float(abs(self.W - self.W.T).max())}
