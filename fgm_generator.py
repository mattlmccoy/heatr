#!/usr/bin/env python3
"""
fgm_generator.py — Functionally Graded Map generator for HEATR run outputs.

Reads fields.npz produced by rfam_eqs_coupled.py and converts the RF power
deposition field (Qrf), temperature (T), or relative density (rho_rel) into a
spatially graded binder-saturation map quantized to 2 bpp (4 levels) or
4 bpp (16 levels), suitable for import into the Meteor MetPrint RIP.

Default inversion rule:
    HIGH Qrf  →  LOW saturation  (overheated regions get less carbon-black dopant)
    LOW  Qrf  →  HIGH saturation (underheated regions get more dopant)

This compensates for uneven RF coupling and drives more uniform sintering.

Usage (module):
    from fgm_generator import generate_fgm
    result = generate_fgm("outputs_eqs/runs/square/single/experimental/my_run",
                           bpp=2, proxy_field="Qrf", magnitude=1.0)
    print(result["npz_path"], result["png_path"])

Usage (CLI):
    python fgm_generator.py <run_output_dir> [--bpp 2|4] [--proxy Qrf|T|rho_rel]
                            [--magnitude 1.0] [--dpi 720] [--no-json] [--no-png]

Dependencies: numpy, scipy, Pillow
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Which direction "higher value" maps for each proxy field.
# True  → higher value means MORE energy deposited (overheated) → we want LESS saturation.
# False → higher value means MORE density (well-sintered); invert=True still makes sense
#         (low density → add more dopant to boost energy absorption next cycle).
_PROXY_HIGH_MEANS_HOT: dict[str, bool] = {
    "Qrf":     True,
    "T":       True,
    "T_phi90": True,   # T at first phi=0.90 crossing — better than final T for FGM (see plan §C)
    "rho_rel": True,   # low rho_rel = under-sintered = needs more saturation = invert still correct
}

SUPPORTED_PROXY_FIELDS = tuple(_PROXY_HIGH_MEANS_HOT.keys())


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_fgm(
    run_output_dir: str | Path,
    bpp: int = 2,
    proxy_field: str = "Qrf",
    invert: bool = True,
    magnitude: float = 1.0,
    baseline_saturation: float = 0.5,
    dpi: int = 720,
    smoothing_sigma: float = 1.5,
    clip_percentile: tuple[float, float] = (2.0, 98.0),
    dead_band: float = 0.0,
    emit_formats: Sequence[str] = ("npz", "json", "png"),
    output_dir: Optional[str | Path] = None,
    prior_sat_npz: Optional[str | Path] = None,
    momentum: float = 0.0,
    use_delta_correction: bool = False,
    ref_lo: Optional[float] = None,
    ref_hi: Optional[float] = None,
    # ── TO parameters (used in integral/OC mode) ──────────────────────────────
    move_limit: float = 0.0,
    sensitivity_filter_sigma: float = 0.0,
    # ── Stochastic perturbation (local-minimum escape) ─────────────────────────
    # When > 0, adds mean-zero uniform noise in [-amp, +amp] to sat_scaled after
    # the main update.  Applied regardless of mode (proportional or OC-TO).
    # Preserves volume approximately (mean-zero inside the part).
    # Recommended for stagnation-escape calls only (amp ≈ 0.05–0.10).
    perturbation_amplitude: float = 0.0,
    # ── Cold-zone overprinting ─────────────────────────────────────────────────
    # Extends the saturation map outward past the part boundary in cold/underheated
    # zones only.  Hot zones (low sat at boundary) get no extension.
    # overprint_cold_mm     : dilation radius in mm (0 = disabled)
    # overprint_cold_thresh : sat threshold above which a boundary pixel is "cold"
    #                         and eligible for overprint extension (default 0.6)
    overprint_cold_mm: float = 0.0,
    overprint_cold_thresh: float = 0.6,
) -> dict:
    """Generate a functionally graded binder-saturation map from a HEATR run.

    Parameters
    ----------
    run_output_dir:
        Path to a HEATR run output directory containing ``fields.npz``.
    bpp:
        Bits per pixel for the output map.  2 → 4 levels (0–3);  4 → 16 levels (0–15).
    proxy_field:
        Which field from ``fields.npz`` to use as the grading proxy.
        ``"Qrf"`` (default) — RF volumetric power deposition [W/m³].
        ``"T"``             — final temperature field [K or °C].
        ``"rho_rel"``       — relative density [0–1].
    invert:
        If True (default), HIGH proxy values → LOW saturation level.
        This is correct for ``Qrf`` and ``T``: overheated regions get less dopant.
        For ``rho_rel`` with invert=True: low-density regions get more dopant.
    magnitude:
        Gradient strength.
        0.0 → flat map at ``baseline_saturation`` (no FGM effect).
        1.0 → full computed contrast (default).
        >1.0 → exaggerated gradient (clamped to [0, 1]).
    dead_band:
        Fraction of normalised proxy range (0–1) around the mean within which NO
        saturation correction is applied — those pixels stay at baseline_saturation.
        Prevents noise in a nearly-uniform T field from being amplified into a
        large gradient.  0.0 = no dead-band (default).  A value of 0.1 means
        pixels whose normalised proxy is within ±0.1 of the mean are not adjusted.
    baseline_saturation:
        Saturation value (0–1) applied uniformly when magnitude=0.
        Default 0.5 = mid-scale (level 1 of 3 for 2bpp, level 7 of 15 for 4bpp).
    dpi:
        Target printer resolution in dots per inch (default 720, Meteor native).
    smoothing_sigma:
        Gaussian smoothing applied to the raw proxy field before quantization
        to suppress simulation grid noise (in simulation pixels, default 1.5).
    clip_percentile:
        (lo_pct, hi_pct) — contrast-stretch range computed over inside-part pixels.
        Values outside this range are clamped.  Default (2, 98) removes outliers.
    emit_formats:
        Subset of ("npz", "json", "png") to write.  All three are written by default.
    output_dir:
        Where to write output files.  Defaults to ``run_output_dir``.
    prior_sat_npz:
        Path to the NPZ file from the *previous* FGM iteration.  When set together
        with ``momentum > 0``, the new saturation map is blended with the prior:

            sat_final = (1 - momentum) * sat_new + momentum * sat_prior

        This damps oscillation: if the new FGM is a mirror-image of the prior (the
        classical proportional-control overshoot pattern), blending pulls the result
        back toward the midpoint rather than flipping fully.  Momentum ∈ [0, 1].
        momentum=0 → no blending (default), momentum=0.5 → equal weight.
    momentum:
        Blend weight for the prior FGM (see ``prior_sat_npz``).  0.0 = no blending.
        Bypassed when ``use_delta_correction=True`` (delta mode has its own accumulation).
    use_delta_correction:
        **Integral control mode.**  If True, instead of recomputing the full saturation map
        from scratch, compute a *delta correction* and accumulate it onto the prior map:

            error      = norm − 0.5          # deviation from center, in [−0.5, +0.5]
            correction = −magnitude × error  # hot → reduce sat, cool → increase sat
            sat_new    = clip(sat_prior + correction, 0, 1)

        This prevents two failure modes of proportional mode:
          1. **Normalization re-amplification** — with fixed ``ref_lo``/``ref_hi`` the
             residual error stays proportionally small rather than being re-stretched to [0,1].
          2. **Anti-correlation oscillation** — accumulating a small delta never produces
             a full pattern reversal; corrections converge rather than oscillate.
        Requires ``prior_sat_npz``.  Falls back to proportional mode if prior is unavailable.
    ref_lo:
        Fixed reference lower bound for normalization (iter-0 percentile lower bound).
        When provided together with ``ref_hi`` and ``use_delta_correction=True``, the
        proxy field is normalized against the *original* iter-0 range rather than the
        current run's own percentiles.  This means a small residual variation (after a
        good correction) stays proportionally small rather than being amplified to [0,1].
        If None, falls back to computing percentiles from the current run (default behaviour).
    ref_hi:
        Fixed reference upper bound for normalization (iter-0 percentile upper bound).
        See ``ref_lo``.
    move_limit:
        **TO move limit.**  Maximum absolute change |Δs_i| allowed per pixel per
        iteration (used in OC integral mode only).  0.0 = no limit (default, backward
        compatible).  Typical values: 0.10–0.20 for early iterations, decreasing as
        iterations converge.  Move limits prevent individual pixels from jumping to the
        opposite extreme in a single step — this is the primary mechanism preventing
        the proportional-control oscillation.  Applied as a symmetric box constraint:

            s_i^new ∈ [s_i^prior − move_limit,  s_i^prior + move_limit]  ∩  [0, 1]
    sensitivity_filter_sigma:
        **TO sensitivity filter radius** (in simulation pixels).  0.0 = off (default).
        A spatial Gaussian filter applied to the sensitivity field g_i = (norm_i − 0.5)
        *before* computing the OC correction.  This suppresses simulation grid noise and
        prevents numerical checkerboarding (neighbouring pixels being pushed to opposite
        extremes by noise).  Recommended starting value: 1.5–2.0 sim pixels.
        Note: distinct from ``smoothing_sigma`` (which filters the raw proxy field).
        The sensitivity filter acts on the *derived* per-pixel gradient, not the input.

    Returns
    -------
    dict with keys:
        ``"level_map"``               — (ny_dpi, nx_dpi) uint8 array, values in [0, 2^bpp-1]
        ``"sat_map"``                 — (ny_sim, nx_sim) float32 saturation map, values in [0,1]
        ``"grid_x_mm"``               — 1-D float64 simulation grid X coords [mm]
        ``"grid_y_mm"``               — 1-D float64 simulation grid Y coords [mm]
        ``"width_mm"``                — float  sim-domain physical width [mm]
        ``"height_mm"``               — float  sim-domain physical height [mm]
        ``"bpp"``                     — int
        ``"n_levels"``                — int  (= 2**bpp)
        ``"proxy_field"``             — str
        ``"magnitude"``               — float
        ``"invert"``                  — bool
        ``"baseline_saturation"``     — float
        ``"ref_lo"``                  — float  normalization lower bound actually used
        ``"ref_hi"``                  — float  normalization upper bound actually used
        ``"use_delta_correction"``    — bool
        ``"move_limit"``              — float
        ``"sensitivity_filter_sigma"``— float
        ``"oc_alpha"``                — float | None  Lagrange multiplier from OC bisection
        ``"volume_error"``            — float | None  residual volume error after OC step
        ``"npz_path"``                — str  (if "npz" in emit_formats)
        ``"json_path"``               — str  (if "json" in emit_formats)
        ``"png_path"``                — str  preview PNG: white=max ink (if "png" in emit_formats)
        ``"meteor_png_path"``         — str  Meteor-import PNG: black=max ink (if "png" in emit_formats)
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    run_output_dir = Path(run_output_dir)
    if not run_output_dir.is_dir():
        raise FileNotFoundError(f"Run output directory not found: {run_output_dir}")

    fields_path = run_output_dir / "fields.npz"
    if not fields_path.exists():
        raise FileNotFoundError(f"fields.npz not found in: {run_output_dir}")

    if bpp not in (2, 4):
        raise ValueError(f"bpp must be 2 or 4, got {bpp!r}")

    if proxy_field not in SUPPORTED_PROXY_FIELDS:
        raise ValueError(
            f"proxy_field must be one of {SUPPORTED_PROXY_FIELDS}, got {proxy_field!r}"
        )

    if not (0.0 <= baseline_saturation <= 1.0):
        raise ValueError(f"baseline_saturation must be in [0, 1], got {baseline_saturation}")

    output_dir = Path(output_dir) if output_dir else run_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load fields ───────────────────────────────────────────────────────────
    fields    = np.load(fields_path)
    part_mask = fields["part_mask"].astype(bool)   # (ny, nx)
    x = np.asarray(fields["x"], dtype=np.float64)  # 1-D, metres
    y = np.asarray(fields["y"], dtype=np.float64)  # 1-D, metres

    if proxy_field not in fields:
        available = [k for k in fields.keys()]
        raise KeyError(
            f"proxy_field='{proxy_field}' not found in fields.npz. "
            f"Available keys: {available}"
        )
    raw = fields[proxy_field].astype(np.float32)   # (ny, nx)

    # ── 1. Smooth to suppress simulation grid noise ───────────────────────────
    raw_smooth = gaussian_filter(raw, sigma=float(smoothing_sigma))

    # ── 2. Contrast-stretch using percentiles inside the part only ────────────
    inside_vals = raw_smooth[part_mask]
    if inside_vals.size == 0:
        raise ValueError(
            "part_mask contains no True pixels — no part found in this simulation output."
        )
    lo_pct, hi_pct = clip_percentile
    # Always use adaptive percentile normalization.
    # (Fixed iter-0 reference bounds are NOT used in OC-TO mode: move_limit naturally
    # caps the correction amplitude, so re-amplification of small residuals is harmless
    # and adaptive normalization correctly tracks pattern flips across iterations.)
    lo = float(np.percentile(inside_vals, lo_pct))
    hi = float(np.percentile(inside_vals, hi_pct))
    span = max(hi - lo, 1e-12)
    norm = np.clip((raw_smooth - lo) / span, 0.0, 1.0)   # 0=coolest, 1=hottest

    # ── 3. Invert: high proxy value → low saturation (default for Qrf / T) ───
    #
    #   invert=True:
    #     norm=1.0 (hottest / most-coupled pixel)  →  sat_raw=0.0  (no dopant)
    #     norm=0.0 (coolest / least-coupled pixel)  →  sat_raw=1.0  (full dopant)
    #
    #   invert=False (rarely needed):
    #     norm=1.0  →  sat_raw=1.0
    sat_raw = (1.0 - norm) if invert else norm

    # ── 4. Magnitude scaling around baseline ─────────────────────────────────
    #
    #   sat_scaled = baseline + magnitude * (sat_raw - baseline)
    #
    #   magnitude=0.0  →  sat_scaled = baseline_saturation everywhere (flat)
    #   magnitude=1.0  →  sat_scaled = sat_raw (full gradient)
    #   magnitude>1.0  →  gradient amplified (clamped to [0, 1])
    sat_scaled = baseline_saturation + float(magnitude) * (sat_raw - baseline_saturation)
    sat_scaled = np.clip(sat_scaled, 0.0, 1.0).astype(np.float32)

    # ── 4b. Dead-band: suppress correction near the mean (prevents noise amplification)
    #
    #   Pixels whose *normalised* proxy value is within ±dead_band of the
    #   inside-part mean are held at baseline_saturation — no correction applied.
    #   This is most important when the field is already nearly uniform (later
    #   iterations), where small residual ΔT would otherwise be amplified to a
    #   large gradient.
    if float(dead_band) > 0.0:
        mean_norm_inside = float(np.mean(norm[part_mask]))
        in_dead_band     = (np.abs(norm - mean_norm_inside) <= float(dead_band)) & part_mask
        sat_scaled[in_dead_band] = float(baseline_saturation)

    # ── 4c. OC-TO INTEGRAL MODE ───────────────────────────────────────────────
    #
    #   Topology Optimisation update (Optimality Criteria formulation) that
    #   accumulates corrections onto the prior saturation map with three key
    #   mechanisms missing from basic feedback control:
    #
    #   1. SENSITIVITY FILTER — spatial Gaussian on g_i before update prevents
    #      checkerboarding from simulation grid noise.
    #
    #   2. MOVE LIMITS — |Δs_i| ≤ move_limit per pixel per iteration prevents
    #      single-step overshot and guarantees monotone convergence.
    #
    #   3. OC BISECTION — bisect on the Lagrange multiplier α to satisfy the
    #      volume constraint Σ s_i^new[part] = Σ s_i^prior[part] exactly,
    #      even when pixels are clamped at their box bounds [0,1] or move limits.
    #      Mean-subtraction only satisfies the constraint when no pixel is clamped;
    #      bisection is exact in all cases.
    #
    #   Update rule:
    #       g_i   = norm_i − 0.5                    raw sensitivity (≈ ∂J/∂s_i / C)
    #       ĝ_i   = GaussFilter(g_i, σ_f)           filtered sensitivity
    #       ĝ_i   = 0  where |ĝ_i| < dead_band      dead-band suppression
    #       sens_i = −ĝ_i  (invert=True)            sign: hot→reduce s_i
    #       Find α via bisection:
    #           Σ_i clamp(s_i^prior − α·mag·sens_i,
    #                      s_i^prior − m, s_i^prior + m, 0, 1)[part] = V_target
    #       s_i^new = clamp(s_i^prior − α·mag·sens_i, …)
    #
    _skip_momentum = False
    _oc_alpha      = None   # reported in results dict for diagnostics
    _vol_error     = None
    if use_delta_correction and prior_sat_npz is not None:
        _prior_path = Path(prior_sat_npz)
        _delta_applied = False
        if _prior_path.exists():
            try:
                _pd      = np.load(_prior_path, allow_pickle=True)
                _prior_s = _pd["sat_map"].astype(np.float32)
                if _prior_s.shape == sat_scaled.shape:

                    # ── Sensitivity field g_i ────────────────────────────────
                    # g_i = norm_i − 0.5 : positive = too hot → should decrease s_i
                    _g = (norm - 0.5).astype(np.float64)   # range ≈ [−0.5, +0.5]

                    # ── Sensitivity filter (checkerboard prevention) ─────────
                    _sf = float(sensitivity_filter_sigma)
                    if _sf > 0.0:
                        _g = gaussian_filter(_g, sigma=_sf)
                        print(f"  [fgm] OC-TO: sensitivity filter σ={_sf} applied", flush=True)

                    # ── Dead-band suppression ────────────────────────────────
                    if float(dead_band) > 0.0:
                        _g[np.abs(_g) < float(dead_band)] = 0.0

                    # ── Gradient-projection update with volume conservation ───
                    #
                    # Update rule (gradient descent on σ_T²):
                    #   Δs_i = -magnitude × g_i           (unconstrained step)
                    #   Δs_i = clip(Δs_i, -move_limit, +move_limit)  (move limit)
                    #   Δs_i -= mean(Δs_i[part])           (zero-mean → volume neutral)
                    #   s_i^new = clip(s_i^prior + Δs_i, 0, 1)
                    #   final volume drift correction via additive shift then re-clip
                    #
                    # Sign: invert=True  → hot pixels have g>0, need s decreased → delta = -g
                    #       invert=False → high proxy needs s increased → delta = +g
                    #
                    # Why not bisection? When move limits cause many pixels to saturate at
                    # their box/move bounds, the volume-as-function-of-Lagrange-multiplier
                    # becomes flat and the bracket may not straddle V0. The projection
                    # approach below is always well-defined and volume-accurate.
                    _mag  = float(magnitude)
                    _ml   = float(move_limit)
                    _V0   = float(_prior_s[part_mask].sum())   # volume target
                    _ps64 = _prior_s.astype(np.float64)

                    # Raw gradient direction (sens > 0 → decrease s_i)
                    # invert=True: hot (g>0) → sens = g → step = -mag×g (decrease)
                    # invert=False: high proxy (g>0) → sens = -g → step = +mag×g (increase)
                    _sens = (_g if invert else -_g).astype(np.float64)

                    # Unconstrained step, then move-limit clamp
                    _delta = -_mag * _sens               # shape (ny, nx)
                    if _ml > 0.0:
                        _delta = np.clip(_delta, -_ml, _ml)

                    # Make the inside-part mean exactly zero (volume neutral before clipping)
                    _delta[part_mask] -= float(_delta[part_mask].mean())

                    # Apply and box-clip
                    _s_new = np.clip(_ps64 + _delta, 0.0, 1.0)
                    _s_new[~part_mask] = 0.0

                    # Iterative volume correction: additive shift then re-clip, repeated
                    # until volume error < 0.5 pixels or max 5 iterations.
                    # Simple one-shot shift + clip accumulates error when many pixels
                    # hit the box bounds (0 or 1) and eat into the correction.
                    for _vc_iter in range(5):
                        _vol_now  = float(_s_new[part_mask].sum())
                        _vol_err  = _V0 - _vol_now
                        if abs(_vol_err) < 0.5:   # < 0.5 pixels of volume drift → done
                            break
                        _shift = _vol_err / max(1, int(part_mask.sum()))
                        _s_new[part_mask] = np.clip(
                            _s_new[part_mask] + _shift, 0.0, 1.0
                        )
                    _s_new = _s_new.astype(np.float32)
                    _s_new[~part_mask] = 0.0

                    sat_scaled  = _s_new
                    _oc_alpha   = None     # no Lagrange multiplier in projection form
                    _vol_actual = float(sat_scaled[part_mask].sum())
                    _vol_error  = abs(_vol_actual - _V0)
                    _delta_applied = True
                    _skip_momentum = True
                    _sat_inside    = sat_scaled[part_mask]

                    # Diagnostic: fraction of inside pixels that changed by >= move_limit
                    _n_part = int(part_mask.sum())
                    _n_at_limit = (
                        int(np.sum(np.abs(_delta[part_mask]) >= _ml - 1e-6))
                        if _ml > 0.0 else 0
                    )
                    print(
                        f"  [fgm] OC-TO (proj): mag={_mag:.3f}  move_limit={_ml:.3f}  "
                        f"sens_filter={float(sensitivity_filter_sigma):.1f}  "
                        f"ΔV_err={_vol_error:.5f}  "
                        f"at_limit={_n_at_limit}/{_n_part} "
                        f"({100*_n_at_limit/_n_part:.1f}%)  "
                        f"sat_inside: mean={_sat_inside.mean():.3f}  "
                        f"std={_sat_inside.std():.3f}  "
                        f"[{_sat_inside.min():.3f}, {_sat_inside.max():.3f}]",
                        flush=True,
                    )
                else:
                    print(
                        f"  [fgm] WARNING: OC-TO prior shape {_prior_s.shape} ≠ "
                        f"{sat_scaled.shape}; falling back to proportional",
                        flush=True,
                    )
            except Exception as _de:
                print(f"  [fgm] WARNING: OC-TO prior load failed: {_de}; "
                      "falling back to proportional", flush=True)
        if not _delta_applied:
            print("  [fgm] WARNING: OC-TO mode requested but prior unavailable; "
                  "using proportional correction this iteration", flush=True)

    # ── 4d. Momentum blending: mix with prior FGM to damp oscillation ────────
    #
    #   sat_final = (1-momentum) * sat_new  +  momentum * sat_prior
    #
    #   When the FGM is oscillating (iter N and N+2 are nearly identical mirrors),
    #   blending with momentum ∈ (0, 0.5] pulls the correction back toward the
    #   midpoint and prevents the proportional-control overshoot.
    #   Bypassed when use_delta_correction=True (delta mode has its own accumulation).
    if not _skip_momentum and float(momentum) > 0.0 and prior_sat_npz is not None:
        prior_path = Path(prior_sat_npz)
        if prior_path.exists():
            try:
                _prior_data = np.load(prior_path, allow_pickle=True)
                _prior_sat  = _prior_data["sat_map"].astype(np.float32)
                if _prior_sat.shape == sat_scaled.shape:
                    alpha     = float(np.clip(momentum, 0.0, 1.0))
                    sat_scaled = ((1.0 - alpha) * sat_scaled + alpha * _prior_sat).astype(np.float32)
                    sat_scaled = np.clip(sat_scaled, 0.0, 1.0)
                    print(f"  [fgm] momentum blend α={alpha:.2f}  prior={prior_path.name}",
                          flush=True)
                else:
                    print(f"  [fgm] WARNING: prior_sat shape {_prior_sat.shape} ≠ "
                          f"sat_scaled shape {sat_scaled.shape}; skipping momentum blend",
                          flush=True)
            except Exception as _me:
                print(f"  [fgm] WARNING: could not load prior FGM for blending: {_me}",
                      flush=True)

    # ── 4d. STOCHASTIC PERTURBATION (local-minimum escape) ───────────────────
    # Add mean-zero uniform random noise inside the part to perturb the
    # saturation map away from a local minimum.  Only applied when
    # perturbation_amplitude > 0 (set by the server on stagnation detection).
    # Mean-subtraction preserves Σ sat_map[part] approximately.
    if float(perturbation_amplitude) > 0.0 and part_mask.any():
        _pamp  = float(perturbation_amplitude)
        _noise = np.random.uniform(-_pamp, _pamp, sat_scaled.shape).astype(np.float32)
        _noise[~part_mask] = 0.0
        _noise[part_mask] -= float(_noise[part_mask].mean())  # zero-mean → volume neutral
        sat_scaled = np.clip(sat_scaled + _noise, 0.0, 1.0).astype(np.float32)
        _ni = sat_scaled[part_mask]
        print(
            f"  [fgm] stochastic perturbation: amp={_pamp:.3f}  "
            f"noise [{_noise[part_mask].min():.3f}, {_noise[part_mask].max():.3f}]  "
            f"sat_mean={_ni.mean():.3f}  sat_std={_ni.std():.3f}",
            flush=True,
        )

    # ── 5. Zero outside part ──────────────────────────────────────────────────
    sat_scaled[~part_mask] = 0.0

    # ── 5b. Selective cold-zone overprint ─────────────────────────────────────
    #
    # Extends the saturation map outward past the nominal part boundary, but ONLY
    # in underheated regions.  Hot edge zones (high proxy → low sat) get no extension.
    #
    # Mechanism:
    #   1. Find the "cold threshold" = sat value above which an edge pixel is
    #      considered underheated (default: above median interior saturation).
    #   2. Morphologically dilate the part mask by `overprint_cold_mm` pixels.
    #   3. In the dilated fringe (outside original mask), assign each pixel the
    #      saturation value of its nearest interior cold-edge neighbour.
    #   4. Hot-edge fringe pixels (nearest interior sat below threshold) stay at 0.
    #
    # This produces a lobe of high-saturation ink just outside cold boundaries —
    # more material where energy is needed — while leaving hot zones untouched.
    #
    # Parameters (must be passed to generate_fgm — defaults are 0 / disabled):
    #   overprint_cold_mm     : dilation radius in mm (0 = disabled)
    #   overprint_cold_thresh : sat threshold above which an edge is "cold"
    #                           (default 0.6 — edge pixels with sat > 0.6 get overprinted)
    if overprint_cold_mm > 0.0:
        dx_m_op = float(x[1] - x[0]) if len(x) > 1 else 1e-4
        dy_m_op = float(y[1] - y[0]) if len(y) > 1 else 1e-4
        _dx_mm  = dx_m_op * 1e3
        _dy_mm  = dy_m_op * 1e3
        # Pixels radius (use average of x/y pitch)
        _avg_pitch_mm = 0.5 * (_dx_mm + _dy_mm)
        _rad_px = max(1, int(round(overprint_cold_mm / _avg_pitch_mm)))

        from scipy.ndimage import binary_dilation as _bdil, distance_transform_edt as _dist_edt

        # 1. Dilated mask
        _struct = np.ones((2 * _rad_px + 1, 2 * _rad_px + 1), dtype=bool)
        _dilated_mask = _bdil(part_mask, structure=_struct)

        # 2. Fringe = new pixels just outside the original mask
        _fringe = _dilated_mask & ~part_mask

        if np.any(_fringe):
            # 3. For each fringe pixel, find its nearest original mask pixel via
            #    distance_transform_edt with indices=True
            _dist, _nearest_idx = _dist_edt(~part_mask, return_indices=True)
            # _nearest_idx[0] = row of nearest mask pixel, [1] = col

            _fringe_rows, _fringe_cols = np.where(_fringe)
            _nn_rows = _nearest_idx[0][_fringe_rows, _fringe_cols]
            _nn_cols = _nearest_idx[1][_fringe_rows, _fringe_cols]
            _nn_sat  = sat_scaled[_nn_rows, _nn_cols]  # sat of nearest interior pixel

            # 4. Only overprint where the nearest interior pixel is "cold"
            _cold = _nn_sat >= float(overprint_cold_thresh)
            if np.any(_cold):
                _fr_cold = _fringe_rows[_cold]
                _fc_cold = _fringe_cols[_cold]
                sat_scaled[_fr_cold, _fc_cold] = _nn_sat[_cold]
                print(
                    f"  [fgm] overprint: radius={overprint_cold_mm:.1f}mm "
                    f"({_rad_px}px)  thresh={overprint_cold_thresh:.2f}  "
                    f"cold_fringe_px={int(_cold.sum())}  "
                    f"sat_range=[{_nn_sat[_cold].min():.3f},{_nn_sat[_cold].max():.3f}]",
                    flush=True,
                )

    # ── 6. Quantize to N-bpp discrete levels ─────────────────────────────────
    n_levels = 1 << bpp       # 4 (2bpp) or 16 (4bpp)
    max_val  = n_levels - 1   # 3 or 15
    level_map_sim = np.round(sat_scaled * max_val).astype(np.uint8)

    # ── 7. Resample from simulation grid to printer DPI ───────────────────────
    #
    #   sim pixel pitch [m] → zoom factor to reach printer pixel pitch (25.4e-3 / dpi)
    #   We zoom the *float* saturation map (not the quantized integer map) to avoid
    #   staircase artefacts at edges, then re-quantize after zoom.
    px_m  = 25.4e-3 / float(dpi)    # printer pixel size in metres
    dx_m  = float(x[1] - x[0]) if len(x) > 1 else px_m
    dy_m  = float(y[1] - y[0]) if len(y) > 1 else px_m
    zx    = dx_m / px_m
    zy    = dy_m / px_m

    if abs(zx - 1.0) > 0.01 or abs(zy - 1.0) > 0.01:
        sat_dpi = zoom(sat_scaled, (zy, zx), order=1)
        sat_dpi = np.clip(sat_dpi, 0.0, 1.0)
    else:
        sat_dpi = sat_scaled  # already at target resolution

    level_map_dpi = np.round(sat_dpi * max_val).astype(np.uint8)
    level_map_dpi = np.clip(level_map_dpi, 0, max_val)

    # ── 8. Build result dict ──────────────────────────────────────────────────
    x_mm = x * 1000.0
    y_mm = y * 1000.0
    run_name = run_output_dir.name

    # Physical dimensions of the FGM in mm — derived from simulation grid extents.
    # These are stored in the NPZ so downstream tools (fgm_to_rip, convergence dashboard)
    # can verify physical size without recomputing from x/y coordinate arrays.
    # The formula is: (last coord - first coord) = grid span, which is the part domain width.
    width_mm  = float(x_mm[-1] - x_mm[0]) if len(x_mm) > 1 else float(dx_m * 1e3 * sat_scaled.shape[1])
    height_mm = float(y_mm[-1] - y_mm[0]) if len(y_mm) > 1 else float(dy_m * 1e3 * sat_scaled.shape[0])

    # Pixel dimensions at printer DPI
    _dpi_w = level_map_dpi.shape[1]  # width in printer pixels
    _dpi_h = level_map_dpi.shape[0]  # height in printer pixels
    # Cross-check: pixels / DPI × 25.4 should equal width_mm / height_mm
    _check_w = _dpi_w / float(dpi) * 25.4
    _check_h = _dpi_h / float(dpi) * 25.4
    print(
        f"  [fgm] Part size: {width_mm:.2f} × {height_mm:.2f} mm  "
        f"→ {_dpi_w} × {_dpi_h} px @ {dpi} DPI  "
        f"({_check_w:.2f} × {_check_h:.2f} mm encoded in TIFF)",
        flush=True,
    )
    if abs(_check_w - width_mm) > 1.0 or abs(_check_h - height_mm) > 1.0:
        print(
            f"  [fgm] WARNING: encoded size ({_check_w:.2f}×{_check_h:.2f} mm) differs "
            f"from sim domain ({width_mm:.2f}×{height_mm:.2f} mm) by >1 mm — "
            f"check that dpi={dpi} matches the simulation grid spacing.",
            flush=True,
        )

    results: dict = {
        "level_map":                  level_map_dpi,
        "sat_map":                    sat_scaled,
        "grid_x_mm":                  x_mm,
        "grid_y_mm":                  y_mm,
        "width_mm":                   width_mm,
        "height_mm":                  height_mm,
        "bpp":                        bpp,
        "n_levels":                   n_levels,
        "proxy_field":                proxy_field,
        "magnitude":                  float(magnitude),
        "invert":                     bool(invert),
        "baseline_saturation":        float(baseline_saturation),
        "dead_band":                  float(dead_band),
        "dpi":                        dpi,
        "run_name":                   run_name,
        # Normalization reference — always returned so caller can store iter-0 bounds
        # and pass them to subsequent iterations for fixed-reference normalization.
        "ref_lo":                     lo,
        "ref_hi":                     hi,
        "use_delta_correction":       bool(use_delta_correction),
        # TO parameters
        "move_limit":                 float(move_limit),
        "sensitivity_filter_sigma":   float(sensitivity_filter_sigma),
        "oc_alpha":                   _oc_alpha,          # None if not in OC mode
        "volume_error":               _vol_error,         # None if not in OC mode
        "perturbation_amplitude":     float(perturbation_amplitude),
    }

    # ── 9. Write outputs ──────────────────────────────────────────────────────
    # Include proxy_field and magnitude in filename so multiple variants coexist.
    mag_str  = f"{magnitude:.2f}".replace(".", "p")
    stem     = f"fgm_{run_name}_{proxy_field}_{bpp}bpp_mag{mag_str}"

    if "npz" in emit_formats:
        npz_path = output_dir / f"{stem}.npz"
        np.savez_compressed(
            npz_path,
            level_map          = level_map_dpi,
            sat_map            = sat_scaled,
            x_mm               = x_mm,
            y_mm               = y_mm,
            # Physical dimensions (sim domain span in mm) — for downstream dimension verification
            width_mm           = np.array(width_mm,           dtype=np.float64),
            height_mm          = np.array(height_mm,          dtype=np.float64),
            bpp                = np.array(bpp,                dtype=np.int32),
            n_levels           = np.array(n_levels,           dtype=np.int32),
            magnitude          = np.array(magnitude,          dtype=np.float32),
            baseline_saturation= np.array(baseline_saturation,dtype=np.float32),
            dead_band          = np.array(dead_band,          dtype=np.float32),
            proxy_field        = np.array(proxy_field),
            invert             = np.array(invert),
            dpi                = np.array(dpi,                dtype=np.int32),
        )
        results["npz_path"] = str(npz_path)
        print(f"  [fgm] NPZ  → {npz_path.name}")

    if "png" in emit_formats:
        from PIL import Image

        _scale = 255.0 / max(max_val, 1)
        _lm_f  = level_map_dpi.astype(np.float32)

        # ── Preview PNG — white = max ink, black = no ink ─────────────────────
        # Convention: level 0 → pixel 0 (black), level max → pixel 255 (white).
        # Visually intuitive — bright regions = more ink deposited.
        # Use for inspection, figures, and the convergence dashboard thumbnail.
        # NOT for direct Meteor RIP import (see meteor_import PNG below).
        #
        # COORDINATE NOTE: fields.npz stores arrays with row 0 = y[0] = physical
        # BOTTOM of the chamber (y ascending upward).  PIL/PNG treats row 0 as the
        # top of the image.  np.flipud() corrects the visual orientation so that the
        # top of the physical part appears at the top of the preview image.
        # The sat_map and level_map stored in the NPZ are intentionally NOT flipped
        # because the simulation feedback loop reads them in the same coordinate
        # system as the T field (row 0 = physical bottom).
        vis_preview = np.flipud(_lm_f * _scale).astype(np.uint8)
        png_path = output_dir / f"{stem}_preview.png"
        Image.fromarray(vis_preview, "L").save(str(png_path))
        results["png_path"] = str(png_path)

        # ── Meteor-import PNG — black = max ink, white = no ink ──────────────
        # Convention: level 0 → pixel 255 (white/no ink),
        #             level max → pixel 0   (black/full ink).
        # This matches Meteor RIP's quantize() pathway which uses
        # PHOTOMETRIC_WHITEISZERO: dark pixel → high binder level.
        # Use this file directly when importing a grayscale FGM image into Meteor.
        # The TIFF stack from fgm_to_rip.py is always correct; this PNG is the
        # equivalent alternative for manual Meteor import.
        # Also flipped vertically so physical top appears at image top.
        vis_meteor = (255 - vis_preview)   # exact pixel inversion (vis_preview already flipped)
        meteor_png_path = output_dir / f"{stem}_meteor_import.png"
        Image.fromarray(vis_meteor, "L").save(str(meteor_png_path))
        results["meteor_png_path"] = str(meteor_png_path)

        _px_str = f"{level_map_dpi.shape[1]}×{level_map_dpi.shape[0]} px"
        print(f"  [fgm] PNG (preview)       → {png_path.name}  ({_px_str})  [white=max ink]")
        print(f"  [fgm] PNG (meteor import) → {meteor_png_path.name}  ({_px_str})  "
              f"[black=max ink — use THIS file to import into Meteor RIP]")

    if "json" in emit_formats:
        meta = {
            "run_name":            run_name,
            "bpp":                 bpp,
            "n_levels":            n_levels,
            "proxy_field":         proxy_field,
            "invert":              bool(invert),
            "magnitude":           float(magnitude),
            "baseline_saturation": float(baseline_saturation),
            "dpi":                 dpi,
            "shape_hw":            list(level_map_dpi.shape),
            "x_mm":                x_mm.tolist(),
            "y_mm":                y_mm.tolist(),
            # Raw level map as base64-encoded row-major uint8 bytes
            "level_map_b64":       base64.b64encode(level_map_dpi.tobytes()).decode(),
        }
        json_path = output_dir / f"{stem}.json"
        json_path.write_text(json.dumps(meta, indent=2))
        results["json_path"] = str(json_path)
        print(f"  [fgm] JSON → {json_path.name}")

    # ── 10. Summary stats ─────────────────────────────────────────────────────
    inside_levels = level_map_dpi[level_map_dpi > 0]  # rough inside-part mask at DPI res
    if inside_levels.size:
        print(
            f"  [fgm] level distribution  "
            f"min={inside_levels.min()}  max={inside_levels.max()}  "
            f"mean={inside_levels.mean():.2f}  "
            f"unique={sorted(np.unique(inside_levels).tolist())}"
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a functionally graded map from a HEATR run output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "run_output_dir",
        help="Path to HEATR run output directory (must contain fields.npz).",
    )
    p.add_argument(
        "--bpp", type=int, choices=[2, 4], default=2,
        help="Bits per pixel (default: 2 → 4 levels).",
    )
    p.add_argument(
        "--proxy", dest="proxy_field", default="Qrf",
        choices=list(SUPPORTED_PROXY_FIELDS),
        help="Proxy field to use for grading (default: Qrf).",
    )
    p.add_argument(
        "--no-invert", dest="invert", action="store_false", default=True,
        help="Disable inversion (high proxy → high saturation instead of low).",
    )
    p.add_argument(
        "--magnitude", type=float, default=1.0,
        help="Gradient magnitude: 0=flat, 1=full, >1=exaggerated (default: 1.0).",
    )
    p.add_argument(
        "--baseline", dest="baseline_saturation", type=float, default=0.5,
        help="Saturation at magnitude=0 (default: 0.5).",
    )
    p.add_argument(
        "--dpi", type=int, default=720,
        help="Target printer DPI (default: 720).",
    )
    p.add_argument(
        "--sigma", dest="smoothing_sigma", type=float, default=1.5,
        help="Gaussian smoothing sigma in sim pixels (default: 1.5).",
    )
    p.add_argument(
        "--clip-lo", type=float, default=2.0,
        help="Low percentile for contrast stretch (default: 2).",
    )
    p.add_argument(
        "--clip-hi", type=float, default=98.0,
        help="High percentile for contrast stretch (default: 98).",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: same as run_output_dir).",
    )
    p.add_argument(
        "--no-json", action="store_true", help="Skip JSON output.",
    )
    p.add_argument(
        "--no-png", action="store_true", help="Skip PNG preview output.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    emit_formats = ["npz"]
    if not args.no_json:
        emit_formats.append("json")
    if not args.no_png:
        emit_formats.append("png")

    print(f"[fgm] Run dir   : {args.run_output_dir}")
    print(f"[fgm] Proxy     : {args.proxy_field}  invert={args.invert}")
    print(f"[fgm] bpp={args.bpp}  magnitude={args.magnitude:.2f}  baseline={args.baseline_saturation:.2f}")
    print(f"[fgm] DPI={args.dpi}  sigma={args.smoothing_sigma}  clip=[{args.clip_lo},{args.clip_hi}]")

    result = generate_fgm(
        run_output_dir     = args.run_output_dir,
        bpp                = args.bpp,
        proxy_field        = args.proxy_field,
        invert             = args.invert,
        magnitude          = args.magnitude,
        baseline_saturation= args.baseline_saturation,
        dpi                = args.dpi,
        smoothing_sigma    = args.smoothing_sigma,
        clip_percentile    = (args.clip_lo, args.clip_hi),
        emit_formats       = emit_formats,
        output_dir         = args.output_dir,
    )

    print(f"[fgm] Done.  level_map shape: {result['level_map'].shape}  bpp={result['bpp']}")
    for key in ("npz_path", "png_path", "json_path"):
        if key in result:
            print(f"  {key}: {result[key]}")


# ---------------------------------------------------------------------------
# Finite-difference gradient descent helpers
# ---------------------------------------------------------------------------

def define_spatial_zones(
    part_mask: np.ndarray,
    n_zones: int = 6,
    zone_type: str = "radial",
) -> list[np.ndarray]:
    """Divide the part into K spatial zones for finite-difference sensitivity.

    Returns a list of boolean masks (each shape (ny, nx)) with True where that
    zone occupies the part interior.  Zones are non-overlapping and their union
    equals part_mask.

    zone_type:
        "radial" — equal-count annuli from centroid outward (best for circles).
        "grid"   — N×N rectangular grid cells clipped to part_mask (general shapes).
    """
    ny, nx = part_mask.shape
    rows, cols = np.where(part_mask)
    if len(rows) == 0:
        return []

    if zone_type == "radial":
        cy = float(np.mean(rows))
        cx = float(np.mean(cols))
        ii, jj = np.indices((ny, nx))
        r = np.sqrt((ii - cy) ** 2 + (jj - cx) ** 2)
        r_inside = r[part_mask]

        # Equal-count thresholds — each zone has approximately the same pixel count
        pcts = np.linspace(0.0, 100.0, n_zones + 1)
        thresholds = np.percentile(r_inside, pcts)
        # Force strict monotone to avoid duplicates at flat percentiles
        for k in range(1, len(thresholds)):
            if thresholds[k] <= thresholds[k - 1]:
                thresholds[k] = thresholds[k - 1] + 1e-6

        zones = []
        for k in range(n_zones):
            lo, hi = thresholds[k], thresholds[k + 1]
            if k < n_zones - 1:
                zone_mask = part_mask & (r >= lo) & (r < hi)
            else:
                zone_mask = part_mask & (r >= lo) & (r <= hi)
            zones.append(zone_mask)

    else:  # "grid"
        n_side = max(1, int(round(n_zones ** 0.5)))
        r_min, r_max = int(rows.min()), int(rows.max())
        c_min, c_max = int(cols.min()), int(cols.max())
        r_edges = np.linspace(r_min, r_max + 1, n_side + 1)
        c_edges = np.linspace(c_min, c_max + 1, n_side + 1)
        ii, jj = np.indices((ny, nx))
        zones = []
        for i in range(n_side):
            for j in range(n_side):
                zone_mask = (
                    part_mask
                    & (ii >= r_edges[i]) & (ii < r_edges[i + 1])
                    & (jj >= c_edges[j]) & (jj < c_edges[j + 1])
                )
                if zone_mask.any():
                    zones.append(zone_mask)

    # Guarantee full coverage: assign any unassigned part pixels to nearest zone
    assigned = np.zeros((ny, nx), dtype=bool)
    for z in zones:
        assigned |= z
    unassigned = part_mask & ~assigned
    if unassigned.any() and zones:
        zones[-1] = zones[-1] | unassigned  # append leftovers to last zone

    return zones


def save_fgm_from_sat_map(
    sat_map: np.ndarray,
    bpp: int,
    output_path: str | Path,
    x_mm: np.ndarray | None = None,
    y_mm: np.ndarray | None = None,
    part_mask: np.ndarray | None = None,
    emit_png: bool = True,
) -> dict:
    """Quantize a float sat_map and save as NPZ (+ optional PNG preview).

    sat_map: (ny, nx) float32 in [0, 1] at simulation-grid resolution.
    Returns dict with 'npz_path' and optionally 'png_path'.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sat = np.clip(sat_map.astype(np.float32), 0.0, 1.0)
    if part_mask is not None:
        sat[~part_mask] = 0.0

    n_levels = 1 << bpp
    max_val = n_levels - 1
    level_map = np.round(sat * max_val).astype(np.uint8)

    npz_path = output_path if str(output_path).endswith(".npz") else Path(str(output_path) + ".npz")
    np.savez_compressed(
        npz_path,
        sat_map=sat,
        level_map=level_map,
        bpp=np.array(bpp),
    )
    if x_mm is not None:
        data = dict(np.load(npz_path, allow_pickle=True))
        data["x_mm"] = np.asarray(x_mm)
        data["y_mm"] = np.asarray(y_mm) if y_mm is not None else np.array([])
        np.savez_compressed(npz_path, **data)

    result: dict = {"npz_path": str(npz_path)}

    if emit_png:
        try:
            from PIL import Image  # type: ignore
            # np.flipud: row 0 = physical bottom → flip so physical top is at image top.
            # sat_map/level_map in the NPZ remain unflipped (sim coordinate system).
            vis = np.flipud(255 - (level_map.astype(np.float32) / max_val * 255)).astype(np.uint8)
            png_path = Path(str(npz_path).replace(".npz", "_preview.png"))
            Image.fromarray(vis, "L").save(png_path)
            result["png_path"] = str(png_path)
        except ImportError:
            pass

    return result


def apply_fd_gradient_step(
    sat_map: np.ndarray,
    zone_masks: list[np.ndarray],
    zone_gradients: list[float],
    step_size: float,
    part_mask: np.ndarray,
    bpp: int,
    output_path: str | Path,
    emit_png: bool = True,
) -> dict:
    """Apply a volume-conserving finite-difference gradient step.

    zone_gradients[k] = (sigma_T_perturbed_k - sigma_T_base) / perturbation_delta
    Negative gradient = increasing sat in zone k reduces sigma_T → we WANT that direction.

    Update rule:
        G_pixel[zone_k] = zone_gradients[k]   (scalar → pixel field)
        G_mean = mean(G_pixel[part_mask])      (volume conservation offset)
        sat_new = sat_base - step_size * (G_pixel - G_mean)
        sat_new = clip(sat_new, 0, 1); outside part = 0

    Returns dict from save_fgm_from_sat_map with 'npz_path' + 'png_path'.
    """
    ny, nx = sat_map.shape
    G = np.zeros((ny, nx), dtype=np.float32)
    for zone_mask, g in zip(zone_masks, zone_gradients):
        G[zone_mask] = float(g)

    # Volume conservation: zero-mean within part so total saturation is preserved
    G_mean = float(G[part_mask].mean()) if part_mask.any() else 0.0
    G[part_mask] -= G_mean
    G[~part_mask] = 0.0

    sat_new = sat_map.astype(np.float32) - step_size * G
    sat_new = np.clip(sat_new, 0.0, 1.0)
    sat_new[~part_mask] = 0.0

    print(
        f"  [fd_gradient] step_size={step_size:.3f}  "
        f"G range=[{G[part_mask].min():.4f}, {G[part_mask].max():.4f}]  "
        f"G_mean_offset={G_mean:.4f}  "
        f"sat_change=[{(sat_new - sat_map)[part_mask].min():.4f}, "
        f"{(sat_new - sat_map)[part_mask].max():.4f}]",
        flush=True,
    )

    return save_fgm_from_sat_map(
        sat_new, bpp=bpp, output_path=output_path,
        part_mask=part_mask, emit_png=emit_png,
    )


if __name__ == "__main__":
    main()
