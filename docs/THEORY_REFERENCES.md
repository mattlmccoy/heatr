# HEATR / RFAM Theory and References

**What this is.** The authoritative, sourced reference list for the HEATR and
heatr3d / solve3d models, from the governing physics down to the numerical
methods used to solve the PDEs. Every literature entry traces to
`rfam_paper_overleaf/references.bib` (the canonical BibTeX, 28 entries) or to an
already-sourced project memo (listed in Part III). Nothing here is invented; each
method is anchored to the code that implements it (`file:line`).

This file is the single content source for the HEATR GUI Theory page
(`webui/static/theory.html`, section 9 "Data Sources and References" and the
numerical-methods section). When this file changes, the page is re-rendered from
it, so the page and the record cannot drift.

**Provenance honesty.** Values are tagged so a measured/cited number is never
displayed as if it were assumed, and vice versa:
- **[CITED]** — traceable to a published source (DOI/patent/thesis below).
- **[MEASURED-INTERNAL]** — from a project dataset (DSC profile, FLIR capture).
- **[ASSUMED]** — a model choice not yet traced to a measurement; an open
  de-risking item, flagged as such on the page too.

---

## Part I — Governing equations (the physics), with citations

### I.1 Electro-quasi-static (EQS) field

$$\nabla\cdot\left[(\sigma + j\omega\epsilon_0\epsilon_r)\nabla V\right] = 0,\qquad \mathbf{E} = -\nabla V$$

The RF drive is quasi-static because the free-space wavelength (~22 m at
13.56 MHz, ~11 m at 27.12 MHz) is three orders of magnitude larger than the
~20 mm part, so retardation is negligible and the problem reduces to a complex
Laplace/continuity equation in the potential. Operating frequency in the model:
$f = 27.12$ MHz (`solve3d/forward.py:86`, `freq_hz`), the second ISM harmonic;
$\omega = 2\pi f$ (`solve3d/forward.py:152`).

- **[CITED]** Allison et al. 2022, *Computational design strategy to improve RF
  heating uniformity*, Rapid Prototyping J. — `allison2022computational` (eqs
  1–5). The RFAM EQS formulation this model reproduces.
- **[CITED]** Ferdous et al. 2017, an experimental **13.56 MHz** RF heating
  system — `ferdous2017rfheating`. Anchors the ISM-band RF-heating regime.
- **[CITED]** Meixner 1972, *The behavior of electromagnetic fields at edges* —
  `Meixner1972singular`. Governs the field singularity at conductive corners
  that the masked-gradient stencil (EQS-02) is built to handle.

### I.2 RF power deposition

$$Q_{\mathrm{rf}}(\mathbf{x}) = \tfrac{1}{2}\,\sigma(\mathbf{x})\,|\mathbf{E}(\mathbf{x})|^2$$

Time-average for sinusoidal fields; in generator-enforced mode the pattern is
rescaled to a target absorbed power (`solve3d/adjoint.py:97`,
`p_target = power_density * v_doped`). Consequence used in Stage A: the total
absorbed power is set by the drive knob, so **frequency reshapes the spatial
coupling pattern (the $\sigma$ vs $\omega\epsilon_0\epsilon_r$ balance), not the
heating rate.**

- **[CITED]** `allison2022computational` (eqs 6–7, $Q_{rh}=\mathrm{Re}\{J\cdot E\}$).

### I.3 Transient thermal transport with phase change

$$\rho c_p\frac{\partial T}{\partial t} = \nabla\cdot(k\nabla T) + Q_{\mathrm{rf}} - Q_{\mathrm{conv}},\qquad -k\nabla T\cdot\hat n = h\,(T-T_\infty)$$

Latent heat enters through an apparent-heat-capacity / enthalpy formulation over
the melt window; melt fraction $\phi(T)\in[0,1]$.

- **[CITED]** `allison2022computational` (eqs 8–13, apparent-heat-capacity phase
  change).
- **[CITED]** Lakraimi et al. 2023, thermal modeling of PA12 powder (SLS) —
  `Lakraimi2023`. Independent PA12 thermal model for the property set.
- **[MEASURED-INTERNAL]** PA12 DSC profile
  (`configs/experimental_pa12_dsc_profile.yaml`): $T_{\mathrm{onset}}=171.0$ °C,
  $T_{\mathrm{peak}}=180.8$ °C, $T_{\mathrm{end}}=186.0$ °C, $L=101.7$ kJ/kg.

### I.4 Densification (sintering) kinetics

$$\frac{d\rho_{\mathrm{rel}}}{dt} = R_{ss}(T,\phi,\rho_{\mathrm{rel}}) + R_{\mathrm{liq}}(T,\phi,\rho_{\mathrm{rel}})$$

Two additive branches, gated by porosity: solid-state Arrhenius creep, and
liquid-regime viscous-capillary flow with an Arrhenius (or WLF) viscosity law.

- **[CITED]** Frenkel 1945, viscous flow under surface tension — `Frenkel1945`.
  The viscous-capillary sintering rate's physical basis.
- **[CITED]** Zhao, Wudy & Drummer 2018, PA12 crystallization kinetics —
  `zhao2018crystallization`. The optional crystallization branch (Nakamura/Avrami
  form, default-disabled).
- **[CITED]** Verbelen et al. 2016 — `verbelen2016characterization`; Zarringhalam
  et al. 2009 — `Zarringhalam2009`; Peyre et al. 2015 — `peyre2015sls`;
  Mokrane et al. 2018 — `mokrane2018process`. PA12 melt/coalescence/densification
  data and SLS process models the densification family is consistent with.

---

## Part II — How HEATR solves the PDEs (numerical methods)

The two engines are intentionally different discretizations of the same physics
(so their agreement is cross-validation, not a shared bug): **solve3d** is a
finite-element (dolfinx) engine built for differentiable inverse design;
**heatr3d** is a voxel finite-volume engine, the fast forward validated against
Allison's data.

### II.1 EQS field solve

**solve3d (FEM).** Galerkin weak form of $\nabla\cdot(\gamma\nabla V)=0$ with
complex $\gamma=\sigma+j\omega\epsilon_0\epsilon_r$, complex P1 Lagrange elements,
Dirichlet electrodes. Solved with **GMRES + GAMG (algebraic multigrid)**
preconditioner, rtol $10^{-10}$; direct complex LU fallback for small systems.
- Code: `solve3d/forward.py:282` (`solve_eqs`), assembly `:295`, solver config
  `:166` (`KSP_ITER = gmres/gamg`).
- Method basis: Galerkin FEM for elliptic PDEs; GAMG algebraic multigrid.

**heatr3d (finite volume).** Cell-centred 7-point (6-neighbour) stencil with
**harmonic-mean face averaging** of $\gamma$, so the ~$4\times10^6$ conductivity
contrast at the doped/virgin interface is captured without smearing. Direct
complex LU (SuperLU) for small grids, else **BiCGSTAB + ILU**, rtol $10^{-8}$.
- Code: `heatr3d.py:314` (`solve_eqs_3d`), harmonic mean `:278`, solver `:376`.
- Method basis: Patankar-style finite-volume for discontinuous coefficients
  (harmonic face conductance is the standard treatment).

### II.2 Thermal + phase march

Mass-lumped **explicit Forward-Euler enthalpy method**: the state is volumetric
enthalpy $H$; temperature is recovered by exact piecewise-linear inversion, so
latent heat is handled without iterating on an apparent $c_p$. Stability by CFL
sub-stepping, $\Delta t < h^2/(6\,\alpha_{\max})$, safety 0.9, with a per-step
$\Delta T$ clamp.
- Code (solve3d): `solve3d/forward.py:540` (`march_enthalpy`), enthalpy inversion
  `:458`/`:468`, update `:789`. Code (heatr3d): CFL limit `heatr3d.py:200`.
- Method basis: enthalpy method for Stefan/phase-change problems
  (Voller–Prakash class); explicit conservative time integration.

### II.3 Densification rate

Solid-state creep $k_{ss}=k_0\exp(-E_a/RT)(1-\phi)^m$ plus liquid viscous-capillary
$k_{\mathrm{liq}}=\text{geom}\cdot\gamma_s/(\eta\,r_{\mathrm{particle}})\,\phi_{\mathrm{act}}^p$
with Arrhenius viscosity $\eta=\eta_{\mathrm{ref}}\exp[(E_{a,\eta}/R)(1/T-1/T_{\mathrm{ref}})]$;
gated by porosity, $R=8.314$.
- Code: `solve3d/forward.py:431` (`densify_rate`, numpy port of
  `heatr3d.py:707`). Bit-for-bit cross-engine (Stage A Task 1 gate).
- Method basis: Frenkel 1945 (viscous-capillary); Arrhenius creep sintering.

### II.4 Adjoint / co-state and the optimizer

**Discrete adjoint** of the EQS solve: the co-state system $A^{H}\lambda=\rho$
reuses the forward LU factorization (the operator is used self-adjointly), with 2
sweeps of iterative refinement for finite-difference-gate headroom; returns
$dJ/d\sigma$ from $dJ/dQ$. Every gradient is finite-difference-gated before it is
trusted.
- Code: `solve3d/adjoint.py:232` (`vjp_q`), factorize `:158`, refinement `:198`.
- Method basis: adjoint-state / PDE-constrained optimization; the ILT
  inverse-design lineage (`poonawala2007mask`, `pang2021inverse`,
  `chan2008initialization`) is the methodological analogy the whole prewarp line
  descends from.

**Optimizer: L-BFGS-B** (SciPy), `jac=True`, box $[0,1]$, objective rescaled by
$1/|g_0|$ so the first trial step is $O(1)$ (a pure reparameterization; the
minimizer is unchanged). MMA was considered and not implemented; L-BFGS-B is the
frozen primary.
- Code: `solve3d/phase_c_run.py:322` (`minimize(..., method="L-BFGS-B")`),
  rescale `:297`.
- Method basis: **[CITED]** Byrd, Lu, Nocedal & Zhu, L-BFGS-B (limited-memory
  bound-constrained quasi-Newton). *[to add to references.bib — see Part V open
  items]*

**Density filter + Heaviside projection** (topology-optimization regularization):
an explicit normalized-convolution (Gaussian) density filter (partition-of-unity
weights via a k-d tree), then a smoothed-Heaviside tanh projection
$s=[\tanh(\beta\eta)+\tanh(\beta(u-\eta))]/[\tanh(\beta\eta)+\tanh(\beta(1-\eta))]$,
$\eta=0.5$, with $\beta$-continuation $1\to2\to4\to8\to16$.
- Code: `solve3d/design_chain.py:124` (filter), `:78` (projection).
- Method basis: **[CITED]** Wang, Lazarov & Sigmund (tanh projection, named in
  the code); Bruns–Tortorelli / Bourdin density filter. *[to add to
  references.bib — Part V]*

### II.5 Energy-conservation validation

Every run tracks $E_{\mathrm{in}}=E_{\mathrm{stored}}+E_{\mathrm{out}}+\varepsilon$
with an incremental accumulator consistent with the Forward-Euler scheme (BOS
property evaluation); acceptance $|\varepsilon|/E_{\mathrm{in}}<1\%$. This is the
`validation_report.png` six-panel gate already documented on the Theory page §8.

---

## Part III — Material properties, sourced values

Each row cites the project memo that sourced it; the memos carry the primary
literature. Provenance tags as above.

| Quantity | Value | Tag | Memo → primary source |
|---|---|---|---|
| In-plane shrinkage $s_{xy}$ | 0.030 (band 0.02–0.04) | [CITED] | `SHRINKAGE_COEFFICIENTS_MEMO.md` → Wang et al., Polymers 12(6):1373 (2020) |
| Z shrinkage $s_z$ (material) | 0.020 (band 0.01–0.03) | [ASSUMED] | `SHRINKAGE_COEFFICIENTS_MEMO.md` (no clean PA12 material-only Z measurement) |
| Crystallization share of shrinkage | ~4.6% (~60% of total) | [CITED] | Benedetti et al., Mater. & Design 180:107906 (2019) |
| Fully-dense PA12 | ~1.01 g/cm³ | [CITED] | `POLYMER_AM_DENSITY_THERMAL_MEMO.md` → Xometry 2023 |
| SLS PA12 relative density (typical) | 0.95–0.975 | [CITED] | Morano & Pagnotta, Polymers 15(22):4446 (2023) |
| $\rho_{\mathrm{target}}$ floor / good / ideal | 0.90 / 0.95 / 0.98 | [CITED]-band | `POLYMER_AM_DENSITY_THERMAL_MEMO.md` (mechanical knee) |
| Degradation ceiling $T_{\mathrm{ceiling}}$ | 250 °C | [CITED] | Vendittoli et al., Sci. Rep. (2025); Vasquez et al., J. Loss Prev. (2013) |
| Melt onset | ~171–184 °C | [CITED]/[MEASURED-INTERNAL] | DSC profile + EOS PA2200 datasheet |
| $\sigma_{\mathrm{doped}}$ (27.12 MHz) | ~0.04 S/m | [CITED] | Allison RPJ 2022 Table 1 (`allison2022volumetric`) |
| $\epsilon_r$ virgin / doped (published) | 2 / 13.8 | [CITED] | Allison RPJ 2022 Table 1 |
| $\epsilon_r$ used in solver | 20 | **[ASSUMED]** | `INK_DOPANT_CALIBRATION_BRIEF.md` — **does NOT trace to Allison's 13.8**; open de-risking item (`rfam_eqs_coupled.py:290`) |
| Carbon-black percolation vs graphite | CB likely percolates ≪ 30–35 wt% graphite optimum | [ASSUMED] | `INK_DOPANT_CALIBRATION_BRIEF.md` — untested, flagged for measurement |
| Latent heat $L$ | 96.7–101.7 kJ/kg | [CITED]/[MEASURED-INTERNAL] | `RFAM_physics_from_literature.md` (Allison) + DSC profile |

**Open measurements (named, not hidden):** dielectric spectroscopy $\epsilon_r(f)$,
$\sigma(f)$ of the doped nylon (would replace the [ASSUMED] $\epsilon_r=20$ and
answer the 13.56-vs-27.12 MHz coupling question); carbon-black percolation
threshold in the printed matrix; RF coupling efficiency $\eta$ (the P-gate);
material-only PA12 Z-shrinkage.

---

## Part V — Full bibliography (categorized, with links)

DOI links resolve at `https://doi.org/<doi>`. Keys match `references.bib`.

### RFAM — core process
- **Allison 2020** — *Radio Frequency Additive Manufacturing: A Volumetric
  Approach to Polymer Powder Bed Fusion.* PhD thesis, UT Austin.
  `allisonDissertation`
- **Allison, Pearce, Beaman & Seepersad 2022** — *Computational design strategy
  to improve RF heating uniformity.* Rapid Prototyping J. 28(8):1476–1491.
  doi:10.1108/RPJ-08-2021-0193. `allison2022computational`
- **Allison, Pearce, Beaman & Seepersad 2022** — *Volumetric fusion of
  graphite-doped nylon 12 powder with RF radiation.* Rapid Prototyping J.
  28(2):317–329. doi:10.1108/RPJ-09-2020-0218. `allison2022volumetric`
- **Song, Sohaib, Allison et al. 2025** — *Enhancing Heating Uniformity of RFAM
  via Functional Grading.* J. Manuf. Processes. doi:10.1016/j.jmapro.2025.07.013.
  `Song2025RFAMUniformity`

### RF / microwave heating
- **Wroe & Rowley 1998** — RF/microwave-assisted processing of materials.
  Canadian Patent CA2261995C. `wroe1998rfmicrowave`
- **Patil et al. 2023** — RF/microwave heating of preceramic polymer
  nanocomposites. Adv. Eng. Mater. doi:10.1002/adem.201900276. `RFceramic2023`
- **Ferdous et al. 2017** — Experimental 13.56 MHz RF heating system. Prog.
  Electromagn. Res. B 79:83–101. doi:10.2528/PIERB17091409. `ferdous2017rfheating`
- **Li & Zhou 2024** — COMSOL microwave heating of Al₂O₃/SiC composites.
  Symmetry 16(10):1254. doi:10.3390/sym16101254. `DielectricCeramicsPolymers`

### Field theory
- **Meixner 1972** — The behavior of electromagnetic fields at edges. IEEE Trans.
  Antennas Propag. 20(4):442–446. doi:10.1109/TAP.1972.1140243.
  `Meixner1972singular`

### Sintering / densification physics
- **Frenkel 1945** — Viscous flow of crystalline bodies under surface tension. J.
  Phys. (USSR) 9(5):385–391. `Frenkel1945`
- **Zarringhalam et al. 2009** — Degree of particle melt in Nylon-12 SLS parts.
  Rapid Prototyping J. 15(3):126–132. doi:10.1108/13552540910943423.
  `Zarringhalam2009`
- **Zarringhalam et al. 2006** — Processing effects on SLS Nylon 12. Mater. Sci.
  Eng. A 435–436:172–180. doi:10.1016/j.msea.2006.07.084. `zarringhalam2006effects`
- **Zhao, Wudy & Drummer 2018** — Crystallization kinetics of PA12 during SLS.
  Polymers 10(2):168. doi:10.3390/polym10020168. `zhao2018crystallization`
- **Verbelen et al. 2016** — Characterization of polyamide powders for laser
  sintering. Eur. Polym. J. 75:163–174. doi:10.1016/j.eurpolymj.2015.12.014.
  `verbelen2016characterization`
- **Peyre et al. 2015** — SLS of PA12 and PEKK semi-crystalline polymers. J.
  Mater. Process. Technol. 225:326–336. doi:10.1016/j.jmatprotec.2015.04.030.
  `peyre2015sls`
- **Mokrane, Boutaous & Xin 2018** — SLS of polymer powders: modeling,
  simulation, validation. C. R. Méc. 346(12):1087–1103.
  doi:10.1016/j.crme.2018.08.002. `mokrane2018process`
- **Lupone et al. 2021** — Process phenomena and material properties in SLS of
  polymers (review). Materials 15(1):183. doi:10.3390/ma15010183. `lupone2021process`
- **Yang et al. 2019** — 3D non-isothermal phase-field simulation of SLS
  microstructure. Addit. Manuf. 29:100783. doi:10.1016/j.addma.2019.100783.
  `Yang2019`
- **Lakraimi et al. 2023** — Thermal modeling of PA12 powder in SLS via DEM.
  Materials 16(2):753. doi:10.3390/ma16020753. `Lakraimi2023`
- **Bourell et al. 2014** — Performance limitations in polymer laser sintering.
  Phys. Procedia 56:147–156. doi:10.1016/j.phpro.2014.08.157. `bourell2014performance`
- **Wudy et al. 2016** — SLS of filled polymer systems. Phys. Procedia
  83:991–1002. doi:10.1016/j.phpro.2016.08.104. `wudy2016filled`

### Inverse design (ILT analogy — the prewarp methodological lineage)
- **Pang 2021** — Inverse lithography technology: 30 years. J. Micro/Nanopattern.
  Mater. Metrol. 20(3):030901. doi:10.1117/1.JMM.20.3.030901. `pang2021inverse`
- **Poonawala & Milanfar 2007** — Mask design for optical microlithography. IEEE
  Trans. Image Process. 16(3):774–788. doi:10.1109/TIP.2006.891332.
  `poonawala2007mask`
- **Chan, Wong & Lam 2008** — Initialization for robust inverse mask synthesis.
  Opt. Express 16(19):14746–14760. doi:10.1364/OE.16.014746. `chan2008initialization`
- **Zhang, Ma & Zhang 2023** — Fast inverse lithography via a model-driven GCN.
  Opt. Express 31(22):36451–36467. doi:10.1364/OE.493178. `zhang2023fast`
- **Zhu et al. 2024** — L2O-ILT: Learning to Optimize ILT. IEEE TCAD
  43(3):944–955. doi:10.1109/TCAD.2023.3323164. `zhu2024l2o`

### Additive manufacturing — background
- **Gibson, Rosen & Stucker 2015** — *Additive Manufacturing Technologies.*
  Springer. doi:10.1007/978-1-4939-2113-3. `gibson2015additive`
- **Beaman et al. 2020** — Additive manufacturing review: early past to current
  practice. J. Manuf. Sci. Eng. 142(11):110812. doi:10.1115/1.4048193.
  `Beaman2020AdditiveMR`
- **Deckard 1989** — Method and apparatus for producing parts by selective
  sintering. U.S. Patent US4863538A. `deckard1989method`

### Open items to add to references.bib (used in the code, not yet in the .bib)
- **Byrd, Lu, Nocedal & Zhu 1995** — L-BFGS-B (limited-memory bound-constrained
  quasi-Newton). *SIAM J. Sci. Comput.* 16(5):1190–1208. — the optimizer.
- **Wang, Lazarov & Sigmund 2011** — On projection methods, convergence and
  robust formulations in topology optimization. *Struct. Multidiscip. Optim.*
  43:767–784. — the tanh Heaviside projection (named in `design_chain.py`).
- **Bruns & Tortorelli 2001** (or **Bourdin 2001**) — density filtering in
  topology optimization. — the density filter.
- A finite-volume / heterogeneous-elliptic reference (Patankar 1980) for the
  harmonic-face heatr3d discretization.

*These four are cited in Part II by name but are not yet BibTeX entries; adding
them makes `docs/heatr_simulation_methods.tex` fully citeable.*
