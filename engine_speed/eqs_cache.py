"""Content-addressed caching of the heatr3d EQS solve.

heatr3d.py is NOT modified. The assembly below is a VERBATIM port of
``heatr3d.solve_eqs_3d`` (read-only), pinned by
``test_assembly_reproduces_heatr3d_solution_exactly``, which solves the ported
system and requires the result to equal ``heatr3d.solve_eqs_3d`` bit-for-bit.

-----------------------------------------------------------------------------
WHAT ACTUALLY ENTERS THE ASSEMBLY (enumerated from heatr3d.solve_eqs_3d)
-----------------------------------------------------------------------------
The matrix A is a function of EXACTLY two things:

  * ``gamma``  -- through ``_harmonic(gamma, roll(gamma))`` on all six faces,
  * ``grid.h`` -- through ``h2 = grid.h**2``,

plus the hard-coded electrode geometry (Dirichlet on the y_min and y_max
planes, Neumann elsewhere) and the fixed ``+ 1e-18 * I`` regularisation.

The right-hand side b is a function of those PLUS ``p.v_lo`` and ``p.v_hi``.

Nothing else reaches the linear system. In particular the part mask, ``sat``,
sigma/eps parameters, frequency, edge_width and premix settings all reach it
ONLY through ``gamma``, so hashing ``gamma`` itself captures every one of them
with no risk of forgetting an input. (Hashing the mask and sat separately would
be strictly worse: it would miss an h change at fixed n, which
``test_changed_grid_spacing_at_same_n_misses`` pins.)

The key therefore folds in: gamma bytes, gamma shape+dtype, h, v_lo, v_hi, the
resolved solver path (direct vs iterative), and a fingerprint of heatr3d's own
EQS source (see ``solver_fingerprint``) so a future edit to ``solve_eqs_3d`` or
``_harmonic`` can never be served from a stale entry.

Keys are BYTE-exact, which is strictly stronger than value equality: +0.0 and
-0.0 hash differently and would MISS. That direction is safe -- it can waste a
solve, it can never return the wrong field.

-----------------------------------------------------------------------------
WHAT IS CACHED, AND WHAT IS NOT
-----------------------------------------------------------------------------
Two levels, both bit-identity-verified:

  * SOLUTION cache (key = matrix key + voltages). A hit returns V with no
    linear algebra at all, so bit-identity is trivial. This is the level the
    Express before/after and package-verify scenarios hit.

  * FACTORIZATION cache (key = matrix key alone), ITERATIVE PATH ONLY. heatr3d
    uses ``spilu(A, drop_tol=1e-4, fill_factor=12)`` + BiCGSTAB whenever
    N > 50_000 (n >= 37 on a cubic grid), and that ILU build is the dominant
    cost. ``spilu`` is deterministic on a given matrix (probed), so reusing a
    cached ILU is bit-identical to rebuilding it. This is what makes a
    fixed-geometry voltage sweep cheap.

  * The DIRECT path (N <= 50_000) is NOT factorization-cached. heatr3d calls
    ``spla.spsolve``, and ``splu(A).solve(b)`` does NOT agree with it
    bit-for-bit (measured 5.0e-15 relative on an n=20 system: different SuperLU
    driver, different pivoting path). Substituting it would break the gate, so
    the direct path falls back to a full ``spsolve`` on a solution miss. That
    costs nothing in practice -- the direct path is the small-grid path.

FACTORIZATION CACHE ENTRIES ARE LARGE. An ILU at n=48 with fill_factor=12 is
order 10^7 complex nonzeros (~250 MB); at n=96 heatr3d's own note records
2.21 GB for the solve. ``max_factorizations`` therefore defaults to 1. Raise it
only if you know the memory is there.

  * Optional PER-JOB DISK STORE of solutions only (``DiskSolutionStore``), so a
    package-verify in a FRESH PROCESS hits instead of re-solving. Factorizations
    stay in memory (SuperLU objects are not picklable), so a fresh process still
    rebuilds the ILU on a genuine miss. Every stored payload carries a blake2b
    hash that is re-verified on read: a truncated or bit-flipped file MISSES
    loudly and is never loaded. Off by default; a campaign-level SHARED store is
    deliberately not built in (point ``store_dir`` at a shared directory to get
    one explicitly).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import heatr3d as h3

logger = logging.getLogger(__name__)

__all__ = ["EqsCache", "DiskSolutionStore", "assemble_eqs",
           "solve_eqs_3d_cached", "solver_fingerprint",
           "scale_invariance_deviation"]

# Bump when the ASSEMBLY PORT below changes in a way that could alter results.
_PORT_VERSION = "engine_speed.eqs_cache/1"

# Test hook: set to a string to simulate a changed heatr3d EQS source.
_FINGERPRINT_OVERRIDE: str | None = None
_FINGERPRINT_CACHE: str | None = None


def solver_fingerprint() -> str:
    """Hash of the heatr3d source this cache is pinned to.

    Folded into every key so that editing ``solve_eqs_3d``, ``_harmonic`` or
    the EQS constants invalidates the whole cache instead of silently serving
    fields computed by the old code.
    """
    global _FINGERPRINT_CACHE
    if _FINGERPRINT_OVERRIDE is not None:
        return _FINGERPRINT_OVERRIDE
    if _FINGERPRINT_CACHE is None:
        src = "".join([
            inspect.getsource(h3.solve_eqs_3d),
            inspect.getsource(h3._harmonic),
            repr(h3.EQS_DIRECT_MAX_UNKNOWNS),
            _PORT_VERSION,
        ])
        _FINGERPRINT_CACHE = hashlib.blake2b(src.encode(), digest_size=16).hexdigest()
    return _FINGERPRINT_CACHE


# --------------------------------------------------------------------------- #
# assembly: verbatim port of heatr3d.solve_eqs_3d's system construction
# --------------------------------------------------------------------------- #
def assemble_eqs(gamma: np.ndarray, grid: h3.Grid, p: h3.Params):
    """Return (A, b) exactly as heatr3d.solve_eqs_3d builds them.

    Uses ``heatr3d._harmonic`` itself rather than a re-derivation, so the face
    conductances cannot drift from the reference implementation.
    """
    nx, ny, nz = gamma.shape
    h2 = grid.h ** 2
    N = nx * ny * nz
    idx = np.arange(N).reshape(nx, ny, nz)
    elec_lo = np.zeros(gamma.shape, bool); elec_lo[:, 0, :] = True
    elec_hi = np.zeros(gamma.shape, bool); elec_hi[:, -1, :] = True
    dirich = elec_lo | elec_hi

    rows = [idx[dirich]]
    cols = [idx[dirich]]
    vals = [np.ones(int(dirich.sum()), np.complex128)]
    b = np.zeros(N, np.complex128)
    b[idx[elec_lo]] = p.v_lo
    b[idx[elec_hi]] = p.v_hi

    diag = np.zeros(gamma.shape, np.complex128)
    for ax in range(3):
        for s in (-1, +1):
            g_nb = np.roll(gamma, -s, axis=ax)
            gf = h3._harmonic(gamma, g_nb) / h2
            valid = np.ones(gamma.shape, bool)
            sl = [slice(None)] * 3
            sl[ax] = (-1 if s == +1 else 0)
            valid[tuple(sl)] = False
            w = np.where(valid, gf, 0.0)
            interior = ~dirich
            src = interior & valid
            i_lin = idx[src]
            nb_lin = np.roll(idx, -s, axis=ax)[src]
            diag[src] += w[src]
            nb_dir = np.roll(dirich, -s, axis=ax)[src]
            wv = w[src]
            on = ~nb_dir
            rows.append(i_lin[on]); cols.append(nb_lin[on]); vals.append(-wv[on])
            bd = np.where(np.roll(elec_lo, -s, axis=ax)[src], p.v_lo, p.v_hi)
            np.add.at(b, i_lin[nb_dir], (wv * bd)[nb_dir])
    interior = ~dirich
    rows.append(idx[interior]); cols.append(idx[interior]); vals.append(diag[interior])

    A = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    A = A + sp.eye(N, format="csr", dtype=np.complex128) * 1e-18
    return A, b


def _use_iterative(gamma: np.ndarray, iterative: bool | None) -> bool:
    """heatr3d's own path resolution: N > 50_000 unless overridden."""
    N = int(np.prod(gamma.shape))
    return (N > 50_000) if iterative is None else bool(iterative)


def _matrix_key(gamma: np.ndarray, grid: h3.Grid, use_iter: bool) -> str:
    g = np.ascontiguousarray(gamma)
    hsh = hashlib.blake2b(digest_size=32)
    hsh.update(g.tobytes())
    hsh.update(repr((g.shape, str(g.dtype))).encode())
    # grid.h is a derived float; hash its exact bits, not L and n separately
    hsh.update(np.float64(grid.h).tobytes())
    hsh.update(repr(bool(use_iter)).encode())
    hsh.update(solver_fingerprint().encode())
    return hsh.hexdigest()


def _solution_key(matrix_key: str, p: h3.Params) -> str:
    hsh = hashlib.blake2b(digest_size=32)
    hsh.update(matrix_key.encode())
    hsh.update(np.array([p.v_lo, p.v_hi], np.float64).tobytes())
    return hsh.hexdigest()


class DiskSolutionStore:
    """Per-job on-disk store of EQS SOLUTIONS (V vectors).

    ONLY solutions go to disk. SuperLU/ILU objects are not picklable, so the
    factorization cache is in-memory only and a fresh process re-builds it.

    Layout, one pair of files per key:
        <key>.npy   -- the raw complex128 V array
        <key>.json  -- {format, key, payload_blake2b, shape, dtype, ...}

    WHY NOT .npz. A .npy payload has no checksum of its own, which makes the
    sidecar hash below genuinely load-bearing: a flipped bit inside the data
    region loads without complaint and would otherwise be marched with. A .npz
    would hide that behind the zip CRC and this gate would never actually be
    exercised -- an untested corruption gate is not a corruption gate.

    CORRUPTION POLICY: any anomaly -- missing sidecar, key mismatch, shape or
    dtype mismatch, payload-hash mismatch, unreadable file -- is logged at
    ERROR and treated as a MISS. The store never returns a field it cannot
    prove is the one it wrote. A miss costs a solve; a false hit corrupts a
    result.

    Writes are atomic (tmp + os.replace), payload BEFORE sidecar, so a torn
    write leaves a sidecar-less payload, which misses.
    """

    FORMAT = 1

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.npy", self.root / f"{key}.json"

    @staticmethod
    def _payload_hash(V: np.ndarray) -> str:
        return hashlib.blake2b(np.ascontiguousarray(V).tobytes(),
                               digest_size=16).hexdigest()

    def get(self, key: str) -> tuple[np.ndarray | None, bool]:
        """Return (V, corrupt). V is None on any miss; ``corrupt`` separates
        'not present' from 'present but REFUSED'."""
        pay, side = self._paths(key)
        if not pay.exists() and not side.exists():
            return None, False
        try:
            if not side.exists():
                raise ValueError("sidecar missing (torn write?)")
            if not pay.exists():
                raise ValueError("payload missing")
            meta = json.loads(side.read_text())
            if meta.get("format") != self.FORMAT:
                raise ValueError(f"unknown store format {meta.get('format')!r}")
            if meta.get("key") != key:
                raise ValueError("sidecar key does not match the requested key")
            V = np.load(pay, allow_pickle=False)
            if list(V.shape) != list(meta["shape"]):
                raise ValueError(f"shape {V.shape} != recorded {meta['shape']}")
            if str(V.dtype) != meta["dtype"]:
                raise ValueError(f"dtype {V.dtype} != recorded {meta['dtype']}")
            got = self._payload_hash(V)
            if got != meta["payload_blake2b"]:
                raise ValueError(
                    f"payload hash {got} != recorded {meta['payload_blake2b']}")
        except Exception as exc:
            logger.error(
                "EQS disk store: REFUSING a CORRUPT or unreadable entry for "
                "key %s... (%s). Falling back to a full solve; the files are "
                "left in place under %s for inspection.",
                key[:16], exc, self.root)
            return None, True
        return V, False

    def put(self, key: str, V: np.ndarray) -> None:
        pay, side = self._paths(key)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            V = np.ascontiguousarray(V)
            tmp_pay = pay.with_suffix(".npy.tmp")
            with open(tmp_pay, "wb") as fh:
                np.save(fh, V, allow_pickle=False)
            os.replace(tmp_pay, pay)                       # payload first
            meta = {"format": self.FORMAT, "key": key,
                    "payload_blake2b": self._payload_hash(V),
                    "shape": list(V.shape), "dtype": str(V.dtype),
                    "engine_fingerprint": solver_fingerprint()}
            tmp_side = side.with_suffix(".json.tmp")
            tmp_side.write_text(json.dumps(meta, indent=2))
            os.replace(tmp_side, side)                     # sidecar last
        except OSError as exc:
            # A read-only or full disk must degrade to memory-only, not kill a
            # run that was going to succeed anyway.
            logger.warning("EQS disk store: could not write %s (%s); "
                           "continuing memory-only", pay, exc)


class EqsCache:
    """In-memory LRU cache of EQS solutions and ILU factorizations, with an
    optional per-job disk store for the solutions.

    ``max_factorizations`` defaults to 1 because a single ILU is hundreds of MB
    at production grid sizes (see the module docstring).

    store_dir: optional directory for the disk SOLUTION store (the Studio
    passes ``<grade_dir>/heatr3d/eqs_store``). None = memory only. A
    campaign-level SHARED store is deliberately not built in; a caller that
    wants one points store_dir at a shared directory explicitly.
    """

    def __init__(self, max_solutions: int = 8, max_factorizations: int = 1,
                 store_dir: str | Path | None = None):
        self.max_solutions = int(max_solutions)
        self.max_factorizations = int(max_factorizations)
        self.store_dir = Path(store_dir) if store_dir is not None else None
        self._store = (DiskSolutionStore(self.store_dir)
                       if self.store_dir is not None else None)
        self._sol: OrderedDict[str, np.ndarray] = OrderedDict()
        self._fac: OrderedDict[str, Any] = OrderedDict()
        self.stats = {"solution_hits": 0, "solution_misses": 0,
                      "disk_hits": 0, "disk_corrupt": 0,
                      "factorization_hits": 0, "factorization_misses": 0,
                      "direct_solves": 0, "iterative_solves": 0}

    # -- solutions -------------------------------------------------------- #
    def lookup_solution(self, key: str) -> np.ndarray | None:
        """Memory tier, then disk tier. Increments EXACTLY ONE of
        solution_hits (memory), disk_hits (disk) or solution_misses (neither,
        i.e. a real solve is about to happen), so the recorded counts add up."""
        v = self._sol.get(key)
        if v is not None:
            self._sol.move_to_end(key)
            self.stats["solution_hits"] += 1
            return v.copy()      # callers must not be able to poison the cache
        if self._store is not None:
            v, corrupt = self._store.get(key)
            if corrupt:
                self.stats["disk_corrupt"] += 1
            if v is not None:
                self._promote(key, v)
                self.stats["disk_hits"] += 1
                return v.copy()
        self.stats["solution_misses"] += 1
        return None

    def _promote(self, key: str, V: np.ndarray) -> None:
        self._sol[key] = V
        self._sol.move_to_end(key)
        while len(self._sol) > self.max_solutions:
            self._sol.popitem(last=False)

    def put_solution(self, key: str, V: np.ndarray) -> None:
        self._promote(key, V.copy())
        if self._store is not None:
            self._store.put(key, V)

    # -- factorizations --------------------------------------------------- #
    def get_factorization(self, key: str):
        f = self._fac.get(key)
        if f is None:
            self.stats["factorization_misses"] += 1
            return None
        self._fac.move_to_end(key)
        self.stats["factorization_hits"] += 1
        return f

    def put_factorization(self, key: str, ilu) -> None:
        if self.max_factorizations <= 0:
            return
        self._fac[key] = ilu
        self._fac.move_to_end(key)
        while len(self._fac) > self.max_factorizations:
            self._fac.popitem(last=False)

    def clear(self) -> None:
        self._sol.clear()
        self._fac.clear()


def solve_eqs_3d_cached(gamma: np.ndarray, grid: h3.Grid, p: h3.Params,
                        iterative: bool | None = None,
                        cache: EqsCache | None = None) -> np.ndarray:
    """Cached drop-in for ``heatr3d.solve_eqs_3d``.

    cache=None disables caching entirely and simply delegates to heatr3d, so
    the call site can be left in place unconditionally.
    """
    if cache is None:
        return h3.solve_eqs_3d(gamma, grid, p, iterative=iterative)

    use_iter = _use_iterative(gamma, iterative)
    mkey = _matrix_key(gamma, grid, use_iter)
    skey = _solution_key(mkey, p)

    V = cache.lookup_solution(skey)
    if V is not None:
        return V

    if not use_iter:
        # DIRECT PATH: no factorization reuse. heatr3d calls spsolve, and
        # splu().solve() does not match it bit-for-bit (5.0e-15 measured), so
        # we delegate to heatr3d verbatim rather than risk the gate.
        cache.stats["direct_solves"] += 1
        V = h3.solve_eqs_3d(gamma, grid, p, iterative=iterative)
        cache.put_solution(skey, V)
        return V

    # ITERATIVE PATH: the ILU build is the expensive part and is reusable.
    cache.stats["iterative_solves"] += 1
    A, b = assemble_eqs(gamma, grid, p)
    Acsc = A.tocsc()
    ilu = cache.get_factorization(mkey)
    if ilu is None:
        ilu = spla.spilu(Acsc, drop_tol=1e-4, fill_factor=12)
        cache.put_factorization(mkey, ilu)
    M = spla.LinearOperator(A.shape, ilu.solve, dtype=np.complex128)
    Vf, info = spla.bicgstab(A, b, rtol=1e-8, atol=0.0, maxiter=2000, M=M)
    if info != 0 or not np.all(np.isfinite(Vf)):
        # Mirror heatr3d's fallback exactly, including its EQS-01 guard, by
        # handing the whole solve back to heatr3d rather than re-deriving it.
        logger.warning("EQS cache: BiCGSTAB failed (info=%s); delegating the "
                       "fallback to heatr3d.solve_eqs_3d", info)
        V = h3.solve_eqs_3d(gamma, grid, p, iterative=iterative)
    else:
        V = Vf.reshape(gamma.shape)
    cache.put_solution(skey, V)
    return V


# --------------------------------------------------------------------------- #
# THE RENORM QUESTION (engine lane's ask), as an executable measurement
# --------------------------------------------------------------------------- #
def scale_invariance_deviation(gamma: np.ndarray, grid: h3.Grid, p: h3.Params,
                               part: np.ndarray, c: float) -> dict[str, Any]:
    """Measure how far ``gamma -> c*gamma`` is from leaving V and Qrf unchanged.

    THE EXACT-ARITHMETIC ARGUMENT. Under gamma -> c*gamma (real c > 0):
      * every harmonic face conductance scales by c, because
        harmonic(ca, cb) = 2(ca)(cb)/(ca+cb) = c * harmonic(a, b);
      * Dirichlet rows are untouched (their matrix entry is 1 and their RHS is
        v_lo / v_hi);
      * every interior row has both its matrix entries AND its RHS contribution
        scaled by c, so c cancels and V is unchanged;
      * Qrf = 0.5*Re(gamma |E|^2) then scales by exactly c, and the fixed-power
        renormalisation in compute_qrf_3d divides that constant straight back
        out.
    So in exact arithmetic the renormalised drive IS invariant, and an
    S4 re-solve that differed only by a scalar factor could be served from the
    cache.

    WHY WE DO NOT DO IT. In floating point the cancellation is inexact -- the
    ``+ 1e-18 * I`` regularisation does not scale with c, and every product and
    sum rounds differently -- so V and Qrf are NOT bit-identical, and the
    deviation grows with c. The gate forbids trading that away, so the cache
    keys on exact bytes only.

    SEPARATELY, the S4 coupling is not a scalar rescale anyway:
    ``apply_sigma_coupling`` multiplies only Re(gamma), by the spatially varying
    factor (1 + a(T-Tref))(1 + b(rho-rho_ref)). With a = b = 0 it returns gamma
    bit-for-bit, so those re-solves are exact cache HITS with no trick needed;
    with a or b nonzero the field genuinely changes and MUST be re-solved.
    """
    V0 = h3.solve_eqs_3d(gamma, grid, p)
    Vc = h3.solve_eqs_3d(gamma * c, grid, p)
    Q0 = h3.compute_qrf_3d(V0, gamma, grid, p, part)
    Qc = h3.compute_qrf_3d(Vc, gamma * c, grid, p, part)

    def _rel(a, b):
        d = np.max(np.abs(a - b))
        return float(d / max(float(np.max(np.abs(a))), 1e-300))

    return {
        "c": float(c),
        "V_bit_identical": bool(np.array_equal(V0, Vc)),
        "V_max_rel_dev": _rel(V0, Vc),
        "Qrf_bit_identical": bool(np.array_equal(Q0, Qc)),
        "Qrf_max_rel_dev": _rel(Q0, Qc),
    }
