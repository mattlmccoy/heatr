# Thermal-Ceiling Stage A: Densify Forward + Best-Part Drive Under Ceiling

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.

**Goal:** Deliver Stage A of the APPROVED thermal-ceiling spec (docs/superpowers/specs/2026-08-06-thermal-ceiling-joint-solve-design.md): a solve3d densify FORWARD (shrinkage-v2 L2 forward half), the KS peak-temperature observable with true-max reporting, and a drive-selection that returns the recommended per-part power for the BEST PART under the degradation ceiling, with the dopant shape-solve run at that drive. No dwell/rotation (Stage C), no rho adjoint (that is L2's second half / Stage B+).

**Architecture:** `solve3d/densify_forward.py` (enthalpy densify march to rho_target on the FEM mesh, semantics ported from heatr3d densify), `solve3d/ceiling.py` (KS peak aggregate + true max, per-material T config), `solve3d/stage_a.py` (drive sweep -> best-part selection -> dopant solve at chosen drive -> power_settings output). Sits on the Phase A forward + Phase C/E solve stack.

**Ground rules:** frozen conventions stand; densify default OFF and bit-identical when off; every gate JSON-quoted; no threshold widening; per-material T_config (melt ~185 C completeness, degradation ~250 C ceiling) read from config, provisional until POLYMER_AM_DENSITY_THERMAL_MEMO.md finalizes; rho_target config with provisional 0.90 floor / 1.0 ideal; heavy runs follow the compute-schedule convention (the Tamper solve is running - announce, load<20, one heavy slot); atomic solve3d/-only commits; no push without protocol; dissertation_materials READ-ONLY; OMP/OPENBLAS=1.

### Task 0: Pre-registration
- [ ] `solve3d/results/stage_a_preregistration.json`: T_config (melt/degradation + margins, cited-provisional), rho_target (0.90 floor/1.0 ideal), the drive sweep range + resolution, the "best part" quality metric (weighted density-completeness + shape-fidelity, weights stated), the tie-break (cooler drive wins), and the acceptance bands. Commit before any solve code.

### Task 1: Densify forward in solve3d (L2 forward half)
- [ ] Red test: `solve3d/densify_forward.py::march_densify` reproduces heatr3d's densify march (rho trajectory, T_final, T_end_max) on the extruded-circle anchor within the measured cross-family tolerance (reuse the Phase A band machinery); densify OFF is bit-identical to the current Phase A forward.
- [ ] Implement (port enthalpy densify semantics; march to rho_target stop). Green. Commit. Mutation check: a dropped densification-rate term must fail the equivalence gate.

### Task 2: Ceiling observable (KS peak + true max)
- [ ] Red test: `solve3d/ceiling.py::peak_temp` returns (KS_aggregate, true_max) over the full densify trajectory; KS tracks true max within a pre-registered band on a synthetic field AND a real march; T_ceiling_ok computed on the true max (never the surrogate); melt-completeness check (min in-part peak >= melt onset) reported separately.
- [ ] Implement. Green. Commit.

### Task 3: Stage A drive selection (best part, not speed)
- [ ] Red test: on a small case, `stage_a.py::select_drive` sweeps drive over the feasible range, rejects any drive whose true-max peak > degradation ceiling, scores the rest by the pre-registered quality metric (density completeness + shape), returns the BEST-quality feasible drive, and on a quality tie returns the COOLER one. Test the drive-limited-null path: if no feasible drive reaches rho_target under ceiling, it returns an honest "cannot make this part under this ceiling/chamber" verdict, not a best-effort over-ceiling map.
- [ ] Implement (forward drive sweep on a fixed/uniform map - the ceiling is nearly dopant-independent, so no dopant solve inside the sweep). Green. Commit. Schedule the sweep's forwards per the compute convention.

### Task 4: Dopant shape-solve at the chosen drive
- [ ] Run the Phase C/E dopant solve stack at the selected drive, reading the objective at the densify end-state (not melt-onset). Emit the solved map + the recommended drive. Acceptance: end-state peak <= ceiling on a mesh hold-out (not just solve mesh); the solved map + drive reproduce their claimed peak under heatr3d verify. JSON-quoted. Commit.

### Task 5: Output + report
- [ ] Populate power_settings.power_density_w_per_m3 with the recommended drive in the Studio contract shape (route the exact field write to the Studio lane for confirmation - no schema change, 2.0.0). Write STAGE_A_REPORT.md: densify equivalence, KS-vs-true-max band, the drive-selection result on a real part (pyramid or cube - both known over-ceiling at 1.0x, so a clean demonstration), the honest-null behavior, and the not-covered list (Stage B exposure, Stage C schedule, the rho adjoint). Commit. Notify the Studio lane with the power_settings field write and the T_config values used.
