const state = {
  meta: null,
  jobs: [],
  matchedConfigName: "",
  autoOutputPrefix: "",
  outputNameTouched: false,
};
const byId = (id) => document.getElementById(id);

// ── Job ETA tracker ────────────────────────────────────────────────────────
// Maps job_id → { startTime, startPct, samples: [{t, pct}] }
const JOB_ETA = new Map();
const OPEN_LOGS = new Set(); // job IDs whose inline log panel is currently open

function _updateEta(jobId, pct) {
  if (!Number.isFinite(pct) || pct <= 0) return;
  const now = Date.now();
  if (!JOB_ETA.has(jobId)) {
    JOB_ETA.set(jobId, { startTime: now, startPct: pct, samples: [{ t: now, pct }] });
    return;
  }
  const entry = JOB_ETA.get(jobId);
  entry.samples.push({ t: now, pct });
  // Keep only last 10 samples for rolling window
  if (entry.samples.length > 10) entry.samples.shift();
}

function _getEtaLabel(jobId, pct) {
  if (!Number.isFinite(pct) || pct >= 100) return "";
  const entry = JOB_ETA.get(jobId);
  if (!entry || entry.samples.length < 2) return "";
  const first = entry.samples[0];
  const last  = entry.samples[entry.samples.length - 1];
  const dPct  = last.pct - first.pct;
  const dT    = (last.t  - first.t) / 1000; // seconds
  if (dPct <= 0 || dT <= 0) return "";
  const ratePctPerSec = dPct / dT;
  const remainingSec  = (100 - pct) / ratePctPerSec;
  if (remainingSec > 7200 || remainingSec < 5) return "";
  const mins = Math.floor(remainingSec / 60);
  const secs = Math.round(remainingSec % 60);
  return `ETA: ~${mins}m ${secs}s`;
}

const viewer = {
  items: [],
  index: 0,
};

const SHAPE_NOMINAL_MM = {
  square: 20.0,
  rectangle: 20.0,
  circle: 20.0,
  ellipse: 20.0,
  triangle: 20.0,
  equilateral_triangle: 20.0,
  diamond: 20.0,
  hexagon: 20.0,
  octagon: 20.0,
  pentagon: 20.0,
  star: 22.0,
  star6: 20.0,
  star8: 20.0,
  cross: 22.0,
  rounded_rect: 20.0,
  L_shape: 20.0,
  T_shape: 24.0,
  trapezoid: 20.0,
  gt_logo: 20.0,
};

const FREQ_PROFILES = {
  rf_27_cal: {
    label: "27.12 MHz calibrated",
    frequency_hz: 27120000,
    generator_transfer_efficiency: 0.02,
    effective_depth_m: 0.02,
    warning: "Using calibrated 27.12 MHz defaults.",
  },
  rf_13_untested: {
    label: "13.56 MHz untested approximation",
    frequency_hz: 13560000,
    // Best current approximation: keep calibrated coupling/depth, change frequency only.
    generator_transfer_efficiency: 0.02,
    effective_depth_m: 0.02,
    warning:
      "13.56 MHz is untested in this model. This preset changes frequency only (keeps calibrated coupling/depth). Validate before drawing conclusions.",
  },
  custom: {
    label: "Custom",
    frequency_hz: null,
    generator_transfer_efficiency: null,
    effective_depth_m: null,
    warning: "Custom profile: set frequency and coupling parameters manually.",
  },
};

const EPS0 = 8.8541878128e-12;
const PARAM_HELP = {
  mode: "Selects the run workflow (single, sweep, optimizers, turntable, shell sweep).",
  shape: "Sets the primary geometry shape used for the part.",
  geometrySizeEnabled: "Enable to override shape nominal size for this run.",
  geometryNominalMm: "Reference nominal size used for this shape in millimeters.",
  geometrySizeMm: "Desired geometry size in millimeters. Applied by scaling from nominal.",
  geometryLockAspect: "When true, width and height scale together using the nominal aspect ratio.",
  baseConfigSelect: "Optional starting YAML template. Auto-select matches current settings.",
  outputName: "Output folder name for this run under outputs_eqs.",
  exposureMinutes: "Total exposure duration for the run in minutes.",
  sweepMinutes: "Comma-separated exposure durations to batch in sweep mode.",
  phiSnapshots: "Target phi levels used by optimizer report snapshots.",
  tempCeiling: "Thermal ceiling constraint used by optimizer ranking.",
  highlightPhi: "Phi contour emphasized in optimizer summary plots.",
  turntableRotationDeg: "Rotation increment applied at each turntable event.",
  turntableTotalRotations: "How many turntable rotation events are applied.",
  turntableIntervalS: "Optional fixed time between turntable events.",
  orientationAngleMinDeg: "Minimum orientation angle sampled in coarse search.",
  orientationAngleMaxDeg: "Maximum orientation angle sampled in coarse search.",
  orientationAngleStepDeg: "Coarse angular increment for orientation search.",
  orientationRefineWindowDeg: "Local refinement window around best coarse angle.",
  orientationRefineStepDeg: "Angular increment used during local refinement.",
  orientationExposureMinS: "Minimum exposure tested by orientation optimizer (seconds).",
  orientationExposureMaxS: "Maximum exposure tested by orientation optimizer (seconds).",
  orientationExposureStepS: "Exposure increment in orientation optimizer (seconds).",
  orientationTempCeilingC: "Temperature ceiling constraint for orientation ranking.",
  orientationMinRhoFloor: "Minimum acceptable density floor for orientation ranking.",
  orientationAngleGifEnabled: "If true, generates an angle-by-angle final-state GIF (adds runtime).",
  orientationAngleGifFrameDurationS: "GIF playback speed in seconds per frame. Larger values play slower.",
  orientationColorMetric: "Color metric used in orientation-angle effectiveness plot. mean_rho is usually the most intuitive quality indicator.",
  placementAlgorithm: "Search algorithm used for part placement optimization.",
  placementNParts: "Number of parts in placement optimization. Higher counts increase runtime and make feasible packing harder.",
  placementPartWidthMm: "Nominal part width in millimeters used for each candidate sample.",
  placementPartHeightMm: "Nominal part height in millimeters used for each candidate sample.",
  placementUseTurntable: "If true, placement candidates are validated with turntable rotation enabled (much slower).",
  placementTurntableRotationDeg: "Turntable rotation increment applied at each event for placement mode.",
  placementTurntableTotalRotations: "Number of turntable events used during placement mode validation.",
  placementTurntableIntervalS: "Optional fixed spacing between turntable events in placement mode.",
  placementClearanceMm: "Minimum spacing between parts in millimeters.",
  placementSearchDomainMarginMm: "Margin from chamber boundary reserved in placement search.",
  placementProxyTopK: "Top proxy-ranked layouts to validate with full coupled runs.",
  placementProxyPopulation: "Population size for each proxy optimization generation.",
  placementProxyIters: "Number of proxy optimization generations.",
  placementProxyEvalBudget: "Maximum proxy layout evaluations before reranking.",
  placementTempCeilingC: "Temperature ceiling constraint for placement ranking.",
  placementMinRhoFloor: "Minimum density floor for placement ranking.",
  placementGaPopulation: "GA population size for placement search.",
  placementGaGenerations: "GA generation count.",
  placementGaCrossoverRate: "Probability of crossover for selected parent layouts.",
  placementGaMutationRate: "Probability of mutation per part coordinate.",
  placementGaElitism: "Number of top layouts carried unchanged to next generation.",
  placementGaTournamentK: "Tournament selection size for GA parent picks.",
  placementGaSeed: "Random seed for reproducible GA initialization.",
  antennaeEnabled: "When checked, Launch Run opens the interactive Antenna Workshop where you place antennas visually, run a Quick Search, and choose single or size-sweep mode before launching.",
  shellEnabled: "Enables shell geometry generation for selected part(s).",
  shellWallThicknessMm: "Shell wall thickness in millimeters.",
  shellMethod: "Shell generation method (v1 uses inward offset).",
  shellSweepThicknessesMm: "Wall thickness values to test for shell sweep mode.",
  shellSweepShapes: "Comma-separated shapes for shell sweep (circle,square).",
  shellTempCeilingC: "Temperature ceiling used when selecting best shell thickness.",
  freqProfile: "Frequency preset. Calibrated 27.12 MHz is recommended; 13.56 MHz is an approximation and should be validated.",
  advGridNx: "Number of grid cells along chamber X. Higher values improve spatial fidelity but increase runtime.",
  advGridNy: "Number of grid cells along chamber Y. Higher values improve spatial fidelity but increase runtime.",
  advFreqHz: "RF frequency used in EQS solve. Changing this affects electric field distribution and deposited power pattern.",
  advVoltage: "Applied electrode voltage magnitude used by the EQS solver.",
  advPower: "Generator power target used when enforce_generator_power is enabled.",
  advEnforceGen: "When true, scales Qrf so absorbed part power matches generator target and transfer efficiency.",
  advEff: "Transfer efficiency from generator power to absorbed part power. Lower values reduce absorbed energy.",
  advDepth: "Effective out-of-plane depth used to convert 2D fields into per-meter absorbed power/energy terms.",
  advDt: "Thermal timestep in seconds. Smaller values are more stable but slower.",
  advAmbient: "Ambient/environment temperature boundary reference in C.",
  advConv: "Convective heat transfer coefficient in W/m^2-K for selected boundaries.",
  advMaxQrf: "Upper cap on volumetric RF heating Qrf to prevent numerical blow-up.",
  advMaxTemp: "Hard thermal clamp for safety/stability in solver updates.",
  advSigma: "Doped-region electrical conductivity in S/m; primary driver of RF absorption strength.",
  advDopedSigmaProfile: "Conductivity spatial profile mode for doped region (for example uniform).",
  advVirginSigma: "Background conductivity outside doped region.",
  advVirginEps: "Background relative permittivity outside doped region.",
  advPowderRho: "Powder bulk solid density used by thermal and densification calculations.",
  advPowderK: "Powder thermal conductivity used in heat diffusion.",
  advPowderCp: "Powder heat capacity used in thermal transients.",
  advDopedEps: "Doped-region relative permittivity used in EQS solve.",
  advSigmaTempCoeff: "Temperature sensitivity coefficient for doped conductivity model.",
  advSigmaDensityCoeff: "Density sensitivity coefficient for doped conductivity model.",
  advSigmaRefTemp: "Reference temperature for conductivity-temperature correction.",
  advDopedRhoSolid: "Solid-state density for doped material.",
  advDopedRhoLiquid: "Liquid-state density for doped material.",
  advDopedKSolid: "Solid-state thermal conductivity for doped material.",
  advDopedKLiquid: "Liquid-state thermal conductivity for doped material.",
  advDopedCpSolid: "Solid-state heat capacity for doped material.",
  advDopedCpLiquid: "Liquid-state heat capacity for doped material.",
  physicsModelFamily: "Physics model family. Baseline reproduces legacy behavior; experimental PA12 hybrid enables DSC-calibrated phase+density models.",
  physicsExperimentalEnabled: "Master gate for experimental calibrated model blocks.",
  physicsProvenanceTag: "Freeform provenance label stored in outputs for traceability.",
  physicsParameterSource: "Source descriptor for parameter set (for example literature+calibrated).",
  physicsCalibrationVersion: "Calibration version tag written into output summary.",
  physicsAbBucketId: "Optional A/B bucket identifier for experiment tracking.",
  physicsProvenanceFile: "Path to provenance YAML listing references and calibration notes.",
  physicsDscProfileFile: "Path to DSC profile YAML used by experimental phase model.",
  expPhaseType: "Phase model implementation type. Experimental default uses apparent heat capacity calibrated to DSC melt interval.",
  expMeltOnsetC: "Temperature where melting begins in DSC-calibrated phase model.",
  expMeltPeakC: "Peak melt temperature in DSC profile.",
  expMeltEndC: "Temperature where melting interval ends.",
  expLatHeat: "Latent heat of fusion in J/kg.",
  expCpSmoothing: "Smoothing strategy across phase transition window.",
  expDensType: "Densification model type used in experimental family.",
  expViscosityModel: "Viscosity law used by densification model (Arrhenius or WLF).",
  expEtaRef: "Reference viscosity magnitude in Pa·s.",
  expEtaRefTempK: "Reference temperature for viscosity in Kelvin.",
  expEtaEa: "Activation energy for Arrhenius viscosity model.",
  expWlfC1: "WLF constant C1 for WLF viscosity model.",
  expWlfC2: "WLF constant C2 in Kelvin for WLF viscosity model.",
  expSurfaceTension: "Surface tension term for viscous-capillary densification.",
  expParticleRadius: "Effective particle radius in meters for densification kinetics.",
  expPhiExponent: "Melt-fraction exponent used in densification rate law.",
  expPhiThreshold: "Minimum melt fraction threshold before rapid liquid densification activates.",
  expPorosityExponent: "Porosity coupling exponent in densification model.",
  expGeomFactor: "Geometric scaling factor in densification kinetics.",
  expCrEnabled: "Enable crystallization submodel coupling.",
  expCrType: "Crystallization model type.",
  expCrK0: "Pre-exponential crystallization rate factor.",
  expCrEa: "Activation energy for crystallization kinetics.",
  expCrExponent: "Model exponent in crystallization kinetics.",
  expCrSuppression: "Liquid-phase suppression factor for crystallization.",
};

function _finitePositive(v) {
  return Number.isFinite(v) && v > 0;
}

function computeSuggestedEffectiveDepthFromInputs() {
  const f = Number(byId("advFreqHz")?.value);
  const sigma = Number(byId("advSigma")?.value || 0.04);
  const epsR = Number(byId("advDopedEps")?.value || 20.0);
  if (!_finitePositive(f) || !Number.isFinite(sigma) || sigma < 0 || !_finitePositive(epsR)) return null;
  const omega = 2 * Math.PI * f;
  const eps = EPS0 * epsR;
  const tanDelta = sigma / Math.max(omega * eps, 1e-30);
  const alpha = omega * Math.sqrt((4e-7 * Math.PI * eps) / 2.0) * Math.sqrt(Math.max(Math.sqrt(1 + tanDelta * tanDelta) - 1.0, 0.0));
  if (!(alpha > 0)) return null;
  const depth = 1.0 / alpha;
  return Number.isFinite(depth) && depth > 0 ? depth : null;
}

function updateDerivedFromFrequency() {
  const depthEl = byId("advDepth");
  if (!depthEl) return;
  const d = computeSuggestedEffectiveDepthFromInputs();
  if (d !== null) depthEl.value = String(d);
}

function parseNumberList(raw) {
  return String(raw || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number)
    .filter((n) => Number.isFinite(n));
}

function parseIsoMs(raw) {
  const s = String(raw || "").trim();
  if (!s) return null;
  const ms = Date.parse(s);
  return Number.isFinite(ms) ? ms : null;
}

function formatDuration(totalSec) {
  const s = Math.max(0, Math.floor(Number(totalSec) || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m ${String(sec).padStart(2, "0")}s`;
  return `${m}m ${String(sec).padStart(2, "0")}s`;
}

function modeTag(mode) {
  const m = String(mode || "").toLowerCase();
  if (m === "single") return "single";
  if (m === "sweep") return "sweep";
  if (m === "shell_sweep") return "shellsweep";
  if (m === "optimizer") return "opt";
  if (m === "orientation_optimizer") return "orient";
  if (m === "placement_optimizer") return "place";
  if (m === "turntable") return "tt";
  return "run";
}

function shapeTag(shape) {
  const s = String(shape || "")
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/^_+|_+$/g, "");
  if (!s) return "shape";
  const compact = s.replaceAll("_", "");
  return compact.slice(0, 10) || "shape";
}

function dateTag() {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}${m}${day}`;
}

function computeAutoOutputPrefix() {
  const shape = shapeTag(byId("shape")?.value);
  const mode = modeTag(byId("mode")?.value);
  return `${shape}_${mode}_${dateTag()}`;
}

function refreshOutputNamePrefix() {
  const el = byId("outputName");
  if (!el) return;
  const nextPrefix = computeAutoOutputPrefix();
  const prevPrefix = state.autoOutputPrefix;
  const current = String(el.value || "");
  const canRewrite =
    !state.outputNameTouched ||
    current.trim() === "" ||
    (prevPrefix && current.startsWith(prevPrefix));
  if (canRewrite) {
    const suffix = prevPrefix && current.startsWith(prevPrefix) ? current.slice(prevPrefix.length) : "";
    el.value = `${nextPrefix}${suffix}`;
  }
  state.autoOutputPrefix = nextPrefix;
}

function refreshTurntableInfo() {
  const info = byId("turntableFullRotInfo");
  if (!info) return;
  const stepDeg = Number(byId("turntableRotationDeg")?.value);
  const nEvents = Number(byId("turntableTotalRotations")?.value);
  const deg = Number.isFinite(stepDeg) ? stepDeg : 0;
  const n = Number.isFinite(nEvents) ? nEvents : 0;
  const totalDeg = deg * n;
  const fullTurns = totalDeg / 360.0;
  info.textContent = `Total planned rotation: ${totalDeg.toFixed(1)}° (${fullTurns.toFixed(2)} full 360° turns)`;
}

function nominalSizeMmForShape(shapeName) {
  const s = String(shapeName || "").trim();
  const v = Number(SHAPE_NOMINAL_MM[s]);
  return Number.isFinite(v) && v > 0 ? v : 20.0;
}

function refreshGeometrySizeUI() {
  const shape = byId("shape")?.value || "square";
  const nominal = nominalSizeMmForShape(shape);
  const nominalEl = byId("geometryNominalMm");
  const desiredEl = byId("geometrySizeMm");
  const infoEl = byId("geometryNominalInfo");
  const enabled = String(byId("geometrySizeEnabled")?.value || "false").toLowerCase() === "true";
  if (nominalEl) nominalEl.value = nominal.toFixed(1);
  if (desiredEl && (!enabled || !Number.isFinite(Number(desiredEl.value)) || Number(desiredEl.value) <= 0)) {
    desiredEl.value = nominal.toFixed(1);
  }
  if (infoEl) infoEl.textContent = `Nominal shape size: ${nominal.toFixed(1)} mm.`;
  _setVisibleById("geometrySizeMm", enabled);
  _setVisibleById("geometryLockAspect", enabled);
}

function syncExperimentalControls() {
  const fam = byId("physicsModelFamily")?.value || "experimental_pa12_hybrid";
  const en = byId("physicsExperimentalEnabled");
  if (!en) return;
  if (fam !== "baseline" && String(en.value).toLowerCase() !== "true") en.value = "true";
  if (fam === "baseline" && String(en.value).toLowerCase() !== "false") en.value = "false";
}

function _fieldContainerById(id) {
  const el = byId(id);
  if (!el) return null;
  return el.closest("label, .adv-section") || el;
}

function _setVisibleById(id, visible) {
  const container = _fieldContainerById(id);
  if (!container) return;
  container.classList.toggle("hidden", !visible);
}

function _setVisibleElement(el, visible) {
  if (!el) return;
  el.classList.toggle("hidden", !visible);
}

function updateDependentVisibility() {
  const geomSizeEnabled = String(byId("geometrySizeEnabled")?.value || "false").toLowerCase() === "true";
  _setVisibleById("geometrySizeMm", geomSizeEnabled);
  _setVisibleById("geometryLockAspect", geomSizeEnabled);

  const placementUsesTurntable = String(byId("placementUseTurntable")?.value || "false").toLowerCase() === "true";
  ["placementTurntableRotationDeg", "placementTurntableTotalRotations", "placementTurntableIntervalS"]
    .forEach((id) => _setVisibleById(id, placementUsesTurntable));

  const shellEnabled = String(byId("shellEnabled")?.value || "false").toLowerCase() === "true";
  ["shellWallThicknessMm", "shellMethod"].forEach((id) => _setVisibleById(id, shellEnabled));

  // Antennae UI is now in the Antenna Workshop modal; nothing to toggle here.

  const family = String(byId("physicsModelFamily")?.value || "experimental_pa12_hybrid").toLowerCase();
  const expEnabled = String(byId("physicsExperimentalEnabled")?.value || "true").toLowerCase() === "true";
  const experimentalActive = family !== "baseline" && expEnabled;

  [
    "physicsProvenanceTag", "physicsParameterSource", "physicsCalibrationVersion", "physicsAbBucketId",
    "physicsProvenanceFile", "physicsDscProfileFile",
    "expPhaseType", "expMeltOnsetC", "expMeltPeakC", "expMeltEndC", "expLatHeat", "expCpSmoothing",
    "expDensType", "expViscosityModel", "expEtaRef", "expEtaRefTempK", "expSurfaceTension",
    "expParticleRadius", "expPhiExponent", "expPhiThreshold", "expPorosityExponent", "expGeomFactor",
    "expCrEnabled",
  ].forEach((id) => _setVisibleById(id, experimentalActive));
  _setVisibleElement(byId("expPhaseSection"), experimentalActive);
  _setVisibleElement(byId("expDensSection"), experimentalActive);
  _setVisibleElement(byId("expCrSection"), experimentalActive);

  const crystallizationEnabled =
    experimentalActive && (String(byId("expCrEnabled")?.value || "false").toLowerCase() === "true");
  ["expCrType", "expCrK0", "expCrEa", "expCrExponent", "expCrSuppression"]
    .forEach((id) => _setVisibleById(id, crystallizationEnabled));

  const viscosityModel = String(byId("expViscosityModel")?.value || "arrhenius").toLowerCase();
  const isWlf = viscosityModel === "wlf";
  _setVisibleById("expEtaEa", experimentalActive && !isWlf);
  _setVisibleById("expWlfC1", experimentalActive && isWlf);
  _setVisibleById("expWlfC2", experimentalActive && isWlf);
}

function attachParamInfoIcons() {
  const form = byId("runForm");
  if (!form) return;
  form.querySelectorAll("label").forEach((label) => {
    if (label.querySelector(".param-help-badge")) return;
    const input = label.querySelector("input,select,textarea");
    const key = input?.id || "";
    const textNodes = [];
    for (const node of label.childNodes) {
      if (node === input) break;
      if (node.nodeType === Node.TEXT_NODE) textNodes.push(node);
      if (node.nodeType === Node.ELEMENT_NODE && node.tagName.toLowerCase() !== "span") break;
    }
    const firstText = textNodes.map((n) => n.textContent || "").join(" ").replace(/\s+/g, " ").trim();
    const fallback = firstText ? `Controls ${firstText}.` : "Parameter help.";
    const tip = PARAM_HELP[key] || fallback;
    const title = document.createElement("span");
    title.className = "label-with-help";
    title.textContent = firstText || "Parameter";
    const help = document.createElement("span");
    help.className = "param-help-badge";
    help.setAttribute("title", tip);
    help.setAttribute("data-help", tip);
    help.setAttribute("aria-label", tip);
    help.setAttribute("tabindex", "0");
    help.textContent = "i";
    title.appendChild(help);
    textNodes.forEach((n) => n.remove());
    label.insertBefore(title, input || null);
  });
}

function buildAdvanced() {
  return {
    grid_nx: byId("advGridNx")?.value || "",
    grid_ny: byId("advGridNy")?.value || "",
    frequency_hz: byId("advFreqHz")?.value || "",
    voltage_v: byId("advVoltage")?.value || "",
    generator_power_w: byId("advPower")?.value || "",
    enforce_generator_power: byId("advEnforceGen")?.value || "",
    generator_transfer_efficiency: byId("advEff")?.value || "",
    effective_depth_m: byId("advDepth")?.value || "",
    dt_s: byId("advDt")?.value || "",
    ambient_c: byId("advAmbient")?.value || "",
    convection_h_w_per_m2k: byId("advConv")?.value || "",
    sigma_s_per_m: byId("advSigma")?.value || "",
    sigma_profile: byId("advDopedSigmaProfile")?.value || "",
    virgin_sigma_s_per_m: byId("advVirginSigma")?.value || "",
    virgin_eps_r: byId("advVirginEps")?.value || "",
    powder_rho_solid_kg_per_m3: byId("advPowderRho")?.value || "",
    powder_k_solid_w_per_mk: byId("advPowderK")?.value || "",
    powder_cp_solid_j_per_kgk: byId("advPowderCp")?.value || "",
    doped_eps_r: byId("advDopedEps")?.value || "",
    doped_sigma_temp_coeff_per_K: byId("advSigmaTempCoeff")?.value || "",
    doped_sigma_density_coeff: byId("advSigmaDensityCoeff")?.value || "",
    doped_sigma_ref_temp_c: byId("advSigmaRefTemp")?.value || "",
    doped_rho_solid_kg_per_m3: byId("advDopedRhoSolid")?.value || "",
    doped_rho_liquid_kg_per_m3: byId("advDopedRhoLiquid")?.value || "",
    doped_k_solid_w_per_mk: byId("advDopedKSolid")?.value || "",
    doped_k_liquid_w_per_mk: byId("advDopedKLiquid")?.value || "",
    doped_cp_solid_j_per_kgk: byId("advDopedCpSolid")?.value || "",
    doped_cp_liquid_j_per_kgk: byId("advDopedCpLiquid")?.value || "",
    max_qrf_w_per_m3: byId("advMaxQrf")?.value || "",
    max_temp_c: byId("advMaxTemp")?.value || "",
  };
}

function buildExperimental() {
  const fam = byId("physicsModelFamily")?.value || "experimental_pa12_hybrid";
  const en = (byId("physicsExperimentalEnabled")?.value || "true").toLowerCase() === "true";
  const physicsModel = {
    family: fam,
    experimental_enabled: en,
    provenance_tag: byId("physicsProvenanceTag")?.value || "",
    parameter_source: byId("physicsParameterSource")?.value || "",
    calibration_version: byId("physicsCalibrationVersion")?.value || "",
    ab_bucket_id: byId("physicsAbBucketId")?.value || "",
    provenance_file: byId("physicsProvenanceFile")?.value || "",
    dsc_profile_file: byId("physicsDscProfileFile")?.value || "",
  };
  const phaseModel = {
    type: byId("expPhaseType")?.value || "apparent_heat_capacity_dsc",
    melt_onset_c: Number(byId("expMeltOnsetC")?.value || 171.0),
    melt_peak_c: Number(byId("expMeltPeakC")?.value || 180.8),
    melt_end_c: Number(byId("expMeltEndC")?.value || 186.0),
    lat_heat_j_per_kg: Number(byId("expLatHeat")?.value || 101700),
    cp_smoothing_strategy: byId("expCpSmoothing")?.value || "linear",
  };
  const densModel = {
    type: byId("expDensType")?.value || "viscous_capillary_pa12",
    viscosity_model: byId("expViscosityModel")?.value || "arrhenius",
    viscosity_params: {
      eta_ref_pa_s: Number(byId("expEtaRef")?.value || 8000),
      eta_ref_temp_k: Number(byId("expEtaRefTempK")?.value || 458.15),
      activation_energy_j_per_mol: Number(byId("expEtaEa")?.value || 60000),
      wlf_c1: Number(byId("expWlfC1")?.value || 8.86),
      wlf_c2_k: Number(byId("expWlfC2")?.value || 101.6),
    },
    surface_tension_n_per_m: Number(byId("expSurfaceTension")?.value || 0.03),
    particle_radius_m: Number(byId("expParticleRadius")?.value || 3.5e-5),
    phi_exponent: Number(byId("expPhiExponent")?.value || 1.0),
    phi_threshold: Number(byId("expPhiThreshold")?.value || 0.02),
    porosity_coupling_params: {
      porosity_exponent: Number(byId("expPorosityExponent")?.value || 1.0),
    },
    geom_factor: Number(byId("expGeomFactor")?.value || 0.05),
  };
  const crystallizationModel = {
    enabled: (byId("expCrEnabled")?.value || "false").toLowerCase() === "true",
    type: byId("expCrType")?.value || "nakamura",
    params: {
      k0_per_s: Number(byId("expCrK0")?.value || 0.05),
      ea_j_per_mol: Number(byId("expCrEa")?.value || 42000),
      exponent: Number(byId("expCrExponent")?.value || 2.0),
      liquid_suppression: Number(byId("expCrSuppression")?.value || 0.25),
    },
  };
  return { physicsModel, phaseModel, densModel, crystallizationModel };
}

function renderModelInfo(meta) {
  const card = byId("modelInfoCard");
  if (!card) return;
  const modelInfo = meta?.model_info || {};
  const dsc = modelInfo.dsc_profile || {};
  const dscSrc = Array.isArray(dsc.sources) ? dsc.sources : [];
  const prov = modelInfo.provenance || {};
  const provSrc = Array.isArray(prov.sources) ? prov.sources : [];

  const dscRefs = dscSrc
    .map((s) => {
      const c = s?.citation || "";
      const u = s?.url || "";
      const p = s?.path || "";
      const cite = u ? `<a href="${u}" target="_blank" rel="noopener">${c || u}</a>` : (c || "source");
      const pathLine = p ? `<div class="muted small">local: ${p}</div>` : "";
      return `<li>${cite}${pathLine}</li>`;
    })
    .join("");

  const paperRefs = provSrc
    .map((s) => {
      const key = s?.key || "";
      const c = s?.citation || "";
      const u = s?.url || "";
      const p = s?.path || "";
      const link = u ? `<a href="${u}" target="_blank" rel="noopener">${c || key}</a>` : `${c || key}`;
      const pathLine = p ? `<div class="muted small">local: ${p}</div>` : "";
      return `<li>${link}${pathLine}</li>`;
    })
    .join("");

  card.innerHTML = `
    <h3>Model Comparison</h3>
    <div class="model-grid">
      <article>
        <h4>Default: dsc_calibrated_pa12 (DSC-calibrated AHC)</h4>
        <p>Uses a melt interval from DSC data and latent heat in apparent heat capacity, plus viscous-capillary densification.</p>
        <p class="muted">Active profile: onset=${dsc.melt_onset_c ?? "?"} C, peak=${dsc.melt_peak_c ?? "?"} C, end=${dsc.melt_end_c ?? "?"} C, latent=${dsc.lat_heat_j_per_kg ?? "?"} J/kg.</p>
      </article>
      <article>
        <h4>Optional: baseline (simplified)</h4>
        <p>Uses the legacy smoothed-Heaviside apparent heat-capacity transition and baseline densification path for continuity with older runs.</p>
        <p class="muted">Select baseline when reproducing historical datasets or debugging against legacy outputs.</p>
      </article>
    </div>
    <details>
      <summary>DSC and model references used by HEATR</summary>
      <div class="refs-grid">
        <div>
          <strong>DSC profile sources</strong>
          <ul>${dscRefs || "<li>No DSC sources loaded.</li>"}</ul>
        </div>
        <div>
          <strong>Prior-model provenance sources</strong>
          <ul>${paperRefs || "<li>No provenance sources loaded.</li>"}</ul>
        </div>
      </div>
    </details>
  `;
}

function buildPayload(includeOutput = true) {
  const mode = byId("mode")?.value;
  const payload = {
    mode,
    shape: byId("shape")?.value,
    advanced: buildAdvanced(),
  };
  const exp = buildExperimental();
  payload.physics_model = exp.physicsModel;
  payload.phase_model = exp.phaseModel;
  payload.dens_model = exp.densModel;
  payload.crystallization_model = exp.crystallizationModel;
  payload.geometry_size_enabled = String(byId("geometrySizeEnabled")?.value || "false").toLowerCase() === "true";
  payload.geometry_nominal_mm = Number(byId("geometryNominalMm")?.value);
  payload.geometry_size_mm = Number(byId("geometrySizeMm")?.value);
  payload.geometry_lock_aspect = String(byId("geometryLockAspect")?.value || "true").toLowerCase() === "true";
  const forcedBase = byId("baseConfigSelect")?.value || "";
  if (forcedBase) payload.base_config = forcedBase;

  if (includeOutput) payload.output_name = byId("outputName")?.value?.trim();

  if (["single", "optimizer", "turntable", "orientation_optimizer", "placement_optimizer"].includes(mode)) {
    payload.exposure_minutes = Number(byId("exposureMinutes")?.value);
  }
  if (mode === "shell_sweep") {
    payload.exposure_minutes = Number(byId("exposureMinutes")?.value);
    payload.shell_sweep_thicknesses_mm = parseNumberList(byId("shellSweepThicknessesMm")?.value);
    payload.shell_sweep_shapes = String(byId("shellSweepShapes")?.value || "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    payload.shell_temp_ceiling_c = Number(byId("shellTempCeilingC")?.value);
  }
  if (mode === "sweep") payload.sweep_minutes = parseNumberList(byId("sweepMinutes")?.value);
  if (mode === "optimizer") {
    payload.phi_snapshots = parseNumberList(byId("phiSnapshots")?.value);
    payload.temp_ceiling_c = Number(byId("tempCeiling")?.value);
    payload.highlight_phi = Number(byId("highlightPhi")?.value);
  }
  if (mode === "turntable") {
    payload.turntable_rotation_deg = Number(byId("turntableRotationDeg")?.value);
    payload.turntable_total_rotations = Number(byId("turntableTotalRotations")?.value);
    const intervalRaw = byId("turntableIntervalS")?.value;
    if (String(intervalRaw || "").trim() !== "") {
      payload.turntable_interval_s = Number(intervalRaw);
    }
  }
  if (mode === "orientation_optimizer") {
    const targetExposureS = Number(byId("exposureMinutes")?.value) * 60.0;
    let orientationExposureMinS = Number(byId("orientationExposureMinS")?.value);
    let orientationExposureMaxS = Number(byId("orientationExposureMaxS")?.value);
    let orientationExposureStepS = Number(byId("orientationExposureStepS")?.value);
    // Preserve advanced behavior when user customizes this range; otherwise, if the
    // stock broad sweep is still present, lock to the requested main exposure.
    if (
      Number.isFinite(targetExposureS) && targetExposureS > 0 &&
      orientationExposureMinS === 240 &&
      orientationExposureMaxS === 720 &&
      orientationExposureStepS === 60
    ) {
      orientationExposureMinS = targetExposureS;
      orientationExposureMaxS = targetExposureS;
      orientationExposureStepS = Math.max(targetExposureS, 1);
      if (byId("orientationExposureMinS")) byId("orientationExposureMinS").value = String(orientationExposureMinS);
      if (byId("orientationExposureMaxS")) byId("orientationExposureMaxS").value = String(orientationExposureMaxS);
      if (byId("orientationExposureStepS")) byId("orientationExposureStepS").value = String(orientationExposureStepS);
    }
    payload.orientation_angle_min_deg = Number(byId("orientationAngleMinDeg")?.value);
    payload.orientation_angle_max_deg = Number(byId("orientationAngleMaxDeg")?.value);
    payload.orientation_angle_step_deg = Number(byId("orientationAngleStepDeg")?.value);
    payload.orientation_refine_window_deg = Number(byId("orientationRefineWindowDeg")?.value);
    payload.orientation_refine_step_deg = Number(byId("orientationRefineStepDeg")?.value);
    payload.orientation_exposure_min_s = orientationExposureMinS;
    payload.orientation_exposure_max_s = orientationExposureMaxS;
    payload.orientation_exposure_step_s = orientationExposureStepS;
    payload.orientation_temp_ceiling_c = Number(byId("orientationTempCeilingC")?.value);
    payload.orientation_min_rho_floor = Number(byId("orientationMinRhoFloor")?.value);
    payload.orientation_angle_gif_enabled = String(byId("orientationAngleGifEnabled")?.value || "false").toLowerCase() === "true";
    payload.orientation_angle_gif_frame_duration_s = Number(byId("orientationAngleGifFrameDurationS")?.value || 2.0);
    payload.orientation_color_metric = String(byId("orientationColorMetric")?.value || "mean_rho");
  }
  if (mode === "placement_optimizer") {
    payload.placement_algorithm = String(byId("placementAlgorithm")?.value || "ga");
    payload.placement_n_parts = Number(byId("placementNParts")?.value);
    payload.placement_part_width_mm = Number(byId("placementPartWidthMm")?.value);
    payload.placement_part_height_mm = Number(byId("placementPartHeightMm")?.value);
    payload.placement_use_turntable = String(byId("placementUseTurntable")?.value || "false").toLowerCase() === "true";
    payload.placement_turntable_rotation_deg = Number(byId("placementTurntableRotationDeg")?.value);
    payload.placement_turntable_total_rotations = Number(byId("placementTurntableTotalRotations")?.value);
    const pttIntervalRaw = byId("placementTurntableIntervalS")?.value;
    if (String(pttIntervalRaw || "").trim() !== "") {
      payload.placement_turntable_interval_s = Number(pttIntervalRaw);
    }
    payload.placement_clearance_mm = Number(byId("placementClearanceMm")?.value);
    payload.placement_search_domain_margin_mm = Number(byId("placementSearchDomainMarginMm")?.value);
    payload.placement_proxy_top_k = Number(byId("placementProxyTopK")?.value);
    payload.placement_proxy_population = Number(byId("placementProxyPopulation")?.value);
    payload.placement_proxy_iters = Number(byId("placementProxyIters")?.value);
    payload.placement_proxy_eval_budget = Number(byId("placementProxyEvalBudget")?.value);
    payload.placement_ga_population = Number(byId("placementGaPopulation")?.value);
    payload.placement_ga_generations = Number(byId("placementGaGenerations")?.value);
    payload.placement_ga_crossover_rate = Number(byId("placementGaCrossoverRate")?.value);
    payload.placement_ga_mutation_rate = Number(byId("placementGaMutationRate")?.value);
    payload.placement_ga_elitism = Number(byId("placementGaElitism")?.value);
    payload.placement_ga_tournament_k = Number(byId("placementGaTournamentK")?.value);
    payload.placement_ga_seed = Number(byId("placementGaSeed")?.value);
    payload.placement_temp_ceiling_c = Number(byId("placementTempCeilingC")?.value);
    payload.placement_min_rho_floor = Number(byId("placementMinRhoFloor")?.value);
  }

  if (mode === "fgm_iterate") {
    // Geometry + exposure come from the shared form fields (shape, exposureMinutes, etc.)
    // already in payload above. Add iterative-specific params:
    const timeSource = byId("fgmIterTimeSource")?.value || "manual";
    const rawProxy   = byId("fgmIterProxy")?.value || "T_phi90";

    payload.use_optimizer           = timeSource === "optimizer";
    payload.exposure_minutes        = Number(byId("exposureMinutes")?.value) || 8;
    payload.n_iterations            = parseInt(byId("fgmIterN")?.value) || 7;
    payload.magnitude               = parseFloat(byId("fgmIterMagnitude")?.value) || 1.0;
    payload.magnitude_decay         = parseFloat(byId("fgmIterMagDecay")?.value) || 0.7;
    payload.min_magnitude           = parseFloat(byId("fgmIterMinMag")?.value   ?? 0.005);
    payload.dead_band               = parseFloat(byId("fgmIterDeadBand")?.value) || 0.05;
    payload.fgm_momentum            = parseFloat(byId("fgmIterMomentum")?.value ?? 0.3);
    payload.bpp                     = parseInt(byId("fgmIterBpp")?.value) || 2;
    const _corrMode = byId("fgmIterCorrMode")?.value || "proportional";
    payload.use_delta_correction    = (_corrMode === "integral");
    payload.use_hybrid              = (_corrMode === "hybrid");
    // OC-TO parameters (used in 'integral' and 'hybrid' modes)
    if (payload.use_delta_correction || payload.use_hybrid) {
      payload.move_limit               = parseFloat(byId("fgmIterMoveLimit")?.value ?? 0.15);
      payload.sensitivity_filter_sigma = parseFloat(byId("fgmIterSensFilter")?.value ?? 0.5);
    }
    // Hybrid-only phase-2 parameters
    if (payload.use_hybrid) {
      payload.hybrid_phase2_magnitude      = parseFloat(byId("fgmIterHybridMag")?.value ?? 0.4);
      payload.hybrid_phase2_move_limit     = parseFloat(byId("fgmIterHybridMoveLimit")?.value ?? 0.15);
      payload.hybrid_phase2_sensitivity_sigma = parseFloat(byId("fgmIterSensFilter")?.value ?? 0.5);
    }
    payload.convergence_sigma_T     = parseFloat(byId("fgmIterConvThresh")?.value) || 3.0;
    payload.melt_abort_frac         = parseFloat(byId("fgmIterMeltAbort")?.value) || 0.15;
    payload.iterate_inner           = String(byId("fgmIterInner")?.value || "false").toLowerCase() === "true";
    payload.iterate_interval_steps  = parseInt(byId("fgmIterInterval")?.value) || 50;
    payload.iterate_damping         = parseFloat(byId("fgmIterDamping")?.value) || 0.5;
    payload.overprint_cold_mm       = parseFloat(byId("fgmIterOverprintMm")?.value    || "0") || 0.0;
    payload.overprint_cold_thresh   = parseFloat(byId("fgmIterOverprintThresh")?.value || "0.6") || 0.6;

    // Thorough / regime-adaptive proxy modes
    if (rawProxy === "__thorough__") {
      payload.thorough    = true;
      payload.proxy_field = "T_phi90";   // schedule starts here; server auto-rotates
      payload.proxy_schedule = ["T_phi90", "Qrf", "rho_rel"];
      payload.stagnation_window = parseInt(byId("fgmIterStagnWindow")?.value) || 2;
      payload.stagnation_eps    = parseFloat(byId("fgmIterStagnEps")?.value)  || 0.5;
    } else if (rawProxy === "__regime_adaptive__") {
      payload.regime_adaptive = true;
      payload.proxy_field = "Qrf";  // starting proxy; server switches on regime change
    } else {
      payload.proxy_field = rawProxy;
    }
  }

  // Antennae enable state is read from the simple checkbox.
  // Placement details are configured in the Antenna Workshop modal, not here.

  payload.shell_enabled = (String(byId("shellEnabled")?.value || "false").toLowerCase() === "true");
  payload.shell_wall_thickness_mm = Number(byId("shellWallThicknessMm")?.value);
  payload.shell_method = String(byId("shellMethod")?.value || "offset_inward");

  if (mode === "prewarp") {
    // Geometry Pre-Warp (level-set ILT). Geometry comes from the shared "shape" field.
    payload.prewarp_grid      = parseInt(byId("prewarpGrid")?.value)   || 56;
    payload.prewarp_iters     = parseInt(byId("prewarpIters")?.value)  || 400;
    payload.prewarp_melt_frac = parseFloat(byId("prewarpMeltFrac")?.value) || 0.95;
    payload.prewarp_gate      = String(byId("prewarpGate")?.value || "true").toLowerCase() === "true";
  }

  return payload;
}

function setModeSections() {
  const mode = byId("mode")?.value;
  document.querySelectorAll(".mode-section").forEach((el) => {
    const modes = el.dataset.mode.split(/\s+/);
    el.classList.toggle("hidden", !modes.includes(mode));
  });
  // Hide the main Launch Run button for fgm_import (has its own dedicated button)
  const runBtn = byId("runBtnBottom");
  if (runBtn) runBtn.style.display = (mode === "fgm_import") ? "none" : "";
  updateDependentVisibility();
  updateFgmIterTimeSourceHelp();
}

function updateFgmProxyHelp() {
  const val  = byId("fgmIterProxy")?.value || "T_phi90";
  const help = byId("fgmProxyHelp");
  const thoroughOpts = byId("fgmThoroughOpts");
  const msgs = {
    "T_phi90":            "T_phi90 captures heating at the optimal sintering moment (φ=0.90), not the cooled final state. Best all-round proxy.",
    "T":                  "Final temperature field. Blurred by diffusion — use T_phi90 instead if available.",
    "Qrf":                "Direct RF power deposition (σ|E|²). Not confounded by latent heat or diffusion. Best proxy during active sintering (ρ̄ < 0.82).",
    "rho_rel":            "Relative density. Corrects under-dense regions directly. Useful when density uniformity (not temperature) is the primary goal.",
    "__thorough__":       "⟳ Thorough: runs T_phi90 first, then automatically switches to Qrf and rho_rel when the current proxy stagnates. Most thorough — needs more iterations (10–15).",
    "__regime_adaptive__":"⚡ Regime-adaptive: uses Qrf during active sintering (fast, direct) and switches to T_phi90 when near-final (ρ̄ ≥ 0.82). Automatically adjusts to the sintering phase.",
  };
  if (help) help.textContent = msgs[val] || "";
  if (thoroughOpts) {
    thoroughOpts.style.display = (val === "__thorough__") ? "" : "none";
  }
}

function updateFgmIterCorrModeHelp() {
  const el          = byId("fgmIterCorrModeHelp");
  const octoPanel   = byId("fgmIterToParams");
  const hybridPanel = byId("fgmIterHybridParams");
  if (!el) return;
  const v = byId("fgmIterCorrMode")?.value || "integral";
  if (v === "integral") {
    el.innerHTML =
      "<strong>Integral (recommended)</strong> — Δs = −m × (T̃ − 0.5) is accumulated " +
      "onto the prior map each iteration. The normalization reference is <em>fixed</em> at " +
      "iter-0 bounds, so small residuals stay small rather than being re-amplified to full range. " +
      "Best empirical result: σ_T = 4.43°C at iter-9 for circle 4bpp (m=0.7, 77% reduction). " +
      "Best iter depends on geometry — early-stop detects it automatically. " +
      "Use m=0.7, decay=1.0, n=12 (defaults). " +
      "No move limit (set Move Limit to 0) and no sensitivity filter needed.";
    if (octoPanel)   octoPanel.style.display   = "none";   // OC-TO params not needed for pure integral
    if (hybridPanel) hybridPanel.style.display = "none";
  } else if (v === "hybrid") {
    el.innerHTML =
      "<strong>Hybrid</strong> — Phase 1: proportional correction at iter-1 (full m=1.0, " +
      "strongest single-step). Phase 2: OC-TO bounded delta refinement from the iter-1 baseline. " +
      "Empirically, best result is still at iter-1; OC-TO phase-2 consistently converges to " +
      "σ_T ≈ 15–16°C attractor above iter-1. " +
      "Use as a comparison baseline or when you need phase-2 move limits for stability.";
    if (octoPanel)   octoPanel.style.display   = "";
    if (hybridPanel) hybridPanel.style.display = "";
  } else {
    el.innerHTML =
      "<strong>Proportional</strong> — each iteration recomputes the full saturation map " +
      "from scratch (no accumulation). Strong single-step improvement at iter-1 " +
      "(54–67% σ_T reduction), then diverges to glass-ceiling attractor at σ_T ≈ 15–16°C " +
      "from iter-2 onward. Best result is <em>always</em> at iter-1. " +
      "Use only for single-iteration corrections or baseline comparison. " +
      "Set n=2 and read iter-1 result.";
    if (octoPanel)   octoPanel.style.display   = "none";
    if (hybridPanel) hybridPanel.style.display = "none";
  }
}

function updateFgmIterTimeSourceHelp() {
  const src  = byId("fgmIterTimeSource")?.value || "manual";
  const help = byId("fgmIterTimeSourceHelp");
  const expRow = byId("exposureMinutes")?.closest("label") ||
                 byId("exposureMinutes")?.closest(".mode-section");
  if (help) {
    if (src === "optimizer") {
      help.innerHTML =
        '<strong>Iter-0 runs in optimizer mode</strong> to the "Exposure (minutes)" limit — ' +
        'intentionally over-exposed so the optimizer can collect snapshots across the full ' +
        'sintering curve. The ideal safe time is extracted from those snapshots ' +
        '(T_max ceiling check included). Iter-1+ all run at that ideal time. ' +
        '<em>T_max &gt; 245°C in iter-0 is expected and will not abort the run.</em>';
    } else {
      help.textContent = 'Set "Exposure (minutes)" above — this exact time is used for every FGM iteration.';
    }
  }
  // Dim the manual exposure field when optimizer is selected
  const expLabel = byId("exposureMinutes")?.closest("label");
  if (expLabel) expLabel.style.opacity = src === "optimizer" ? "0.4" : "";
}

async function populateFgmIterRunList() {
  const sel = byId("fgmIterSourceRun");
  if (!sel) return;
  try {
    const runs = await fetchJson("/api/results-runview");
    const list = Array.isArray(runs) ? runs : [];
    _buildExpCalibFromRuns(list);  // calibrate exposure calculator from run history
    if (list.length === 0) {
      sel.innerHTML = '<option value="">— no completed runs found —</option>';
      return;
    }
    // Prefer runs that have fields.npz (completed sims); fall back to all runs
    const current = sel.value;
    sel.innerHTML =
      '<option value="">— select a run —</option>' +
      list
        .map((r) => {
          const name = r.name || r;
          return `<option value="${name}">${name}</option>`;
        })
        .join("");
    // Restore previous selection if still valid
    if (current && [...sel.options].some((o) => o.value === current)) {
      sel.value = current;
    }
  } catch (err) {
    sel.innerHTML = `<option value="">— error loading runs: ${err.message} —</option>`;
  }
}

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

async function reorderQueuedJob(jobId, direction) {
  await fetchJson("/api/queue/reorder", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: jobId, direction }),
  });
  await loadJobs();
}

async function controlJob(jobId, action) {
  await fetchJson("/api/job/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: jobId, action }),
  });
  await loadJobs();
}

function renderMeta(meta) {
  const shape = byId("shape");
  if (!shape) return;
  shape.innerHTML = "";
  const shapeOptions = Array.isArray(meta.shape_options) ? [...meta.shape_options] : [];
  if (!shapeOptions.includes("gt_logo")) shapeOptions.push("gt_logo");
  shapeOptions.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = shapeOptionLabel(name);
    shape.appendChild(opt);
  });

  const baseSel = byId("baseConfigSelect");
  if (baseSel) {
    baseSel.innerHTML = "";
    const auto = document.createElement("option");
    auto.value = "";
    auto.textContent = "Auto-select from match";
    baseSel.appendChild(auto);
    (meta.base_configs || []).forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      baseSel.appendChild(opt);
    });
  }

  const yamlSel = byId("yamlPreviewSelect");
  if (yamlSel) {
    yamlSel.innerHTML = "";
    const effective = document.createElement("option");
    effective.value = "__effective__";
    effective.textContent = "Effective run config (merged)";
    yamlSel.appendChild(effective);
    const matched = document.createElement("option");
    matched.value = "__matched__";
    matched.textContent = "Matched base config only";
    yamlSel.appendChild(matched);
    (meta.base_configs || []).forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      yamlSel.appendChild(opt);
    });
  }
}

function shapeOptionLabel(shapeName) {
  const s = String(shapeName || "").trim();
  const icon = {
    square: "◻",
    rectangle: "▭",
    circle: "◯",
    ellipse: "⬭",
    triangle: "△",
    equilateral_triangle: "△",
    diamond: "◇",
    hexagon: "⬡",
    octagon: "8",
    pentagon: "⬠",
    star: "★",
    star6: "✶",
    star8: "✴",
    cross: "✚",
    rounded_rect: "▢",
    L_shape: "└",
    T_shape: "┬",
    trapezoid: "⏢",
    gt_logo: "GT",
  }[s] || "◻";
  return `${icon} │ ${s}`;
}

function renderShapePreview(shapeName) {
  const glyph = byId("shapePreviewGlyph");
  const label = byId("shapePreviewLabel");
  if (!glyph) return;
  const s = String(shapeName || "square").trim();
  const stroke = "#8ab5ff";
  const fill = "rgba(138,181,255,0.18)";
  const common = `stroke='${stroke}' stroke-width='3' fill='${fill}'`;
  let inner = "";
  if (s === "circle") inner = `<circle cx='50' cy='50' r='28' ${common} />`;
  else if (s === "ellipse") inner = `<ellipse cx='50' cy='50' rx='30' ry='22' ${common} />`;
  else if (s === "triangle" || s === "equilateral_triangle") inner = `<polygon points='50,18 81,76 19,76' ${common} />`;
  else if (s === "diamond") inner = `<polygon points='50,16 84,50 50,84 16,50' ${common} />`;
  else if (s === "hexagon") inner = `<polygon points='30,20 70,20 86,50 70,80 30,80 14,50' ${common} />`;
  else if (s === "octagon") inner = `<polygon points='34,14 66,14 86,34 86,66 66,86 34,86 14,66 14,34' ${common} />`;
  else if (s === "pentagon") inner = `<polygon points='50,14 82,38 70,82 30,82 18,38' ${common} />`;
  else if (s === "rectangle") inner = `<rect x='18' y='30' width='64' height='40' rx='4' ${common} />`;
  else if (s === "rounded_rect") inner = `<rect x='18' y='24' width='64' height='52' rx='10' ${common} />`;
  else if (s === "cross") inner = `<path d='M40 18h20v22h22v20H60v22H40V60H18V40h22z' ${common} />`;
  else if (s === "L_shape") inner = `<path d='M22 22h22v34h34v22H22z' ${common} />`;
  else if (s === "T_shape") inner = `<path d='M18 22h64v20H60v40H40V42H18z' ${common} />`;
  else if (s === "trapezoid") inner = `<polygon points='25,24 75,24 86,76 14,76' ${common} />`;
  else if (s === "star" || s === "star6" || s === "star8") {
    const n = (s === "star6") ? 6 : (s === "star8" ? 8 : 5);
    const ratio = (s === "star6") ? 0.5 : (s === "star8" ? 0.414 : 0.382);
    const phaseDeg = (s === "star8") ? 22.5 : 90.0;
    const phase = (Math.PI / 180) * phaseDeg;
    const outer = 34;
    const innerR = outer * ratio;
    const pts = [];
    for (let i = 0; i < n; i += 1) {
      const a0 = phase + (2 * Math.PI * i) / n;
      const a1 = a0 + Math.PI / n;
      pts.push(`${(50 + outer * Math.cos(a0)).toFixed(2)},${(50 - outer * Math.sin(a0)).toFixed(2)}`);
      pts.push(`${(50 + innerR * Math.cos(a1)).toFixed(2)},${(50 - innerR * Math.sin(a1)).toFixed(2)}`);
    }
    inner = `<polygon points='${pts.join(" ")}' ${common} />`;
  }
  else if (s === "gt_logo") {
    inner = `<text x='50' y='58' text-anchor='middle' font-size='34' font-weight='700' fill='${stroke}' font-family='Avenir Next, Inter, sans-serif'>GT</text>`;
  }
  else inner = `<rect x='20' y='20' width='60' height='60' rx='4' ${common} />`;
  glyph.innerHTML = inner;
  if (label) label.textContent = s;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function highlightYaml(text) {
  const highlightKeys = [
    "shape:", "grid_nx:", "grid_ny:", "dt_s:", "n_steps:",
    "frequency_hz:", "voltage_v:", "generator_power_w:", "enforce_generator_power:",
    "generator_transfer_efficiency:", "ambient_c:", "materials:", "virgin:", "powder:", "doped:",
    "sigma_s_per_m:", "sigma_profile:", "eps_r:", "rho_solid_kg_per_m3:", "rho_liquid_kg_per_m3:",
    "k_solid_w_per_mk:", "k_liquid_w_per_mk:", "cp_solid_j_per_kgk:", "cp_liquid_j_per_kgk:",
    "physics_model:", "family:", "experimental_enabled:", "provenance_tag:", "parameter_source:",
    "calibration_version:", "ab_bucket_id:", "phase_model:", "dens_model:", "crystallization_model:",
    "melt_onset_c:", "melt_peak_c:", "melt_end_c:", "lat_heat_j_per_kg:", "cp_smoothing_strategy:",
    "viscosity_model:", "viscosity_params:", "porosity_coupling_params:",
    "shell:", "wall_thickness_mm:", "method:",
    "antennae:", "spike:",
    "optimizer:", "turntable:", "orientation_optimizer:", "placement_optimizer:", "enabled:",
    "rotation_deg:", "total_rotations:", "rotation_interval_s:"
  ];
  const lines = String(text).split("\n");
  const out = lines.map((line) => {
    const esc = escapeHtml(line);
    const hit = highlightKeys.some((k) => line.includes(k));
    return hit ? `<span class="yaml-hit">${esc}</span>` : esc;
  });
  return out.join("\n");
}

function renderImportant(obj) {
  const root = byId("yamlImportant");
  if (!root) return;
  root.innerHTML = "";
  const entries = Object.entries(obj || {}).filter(([, v]) => v !== null && v !== undefined && v !== "");
  if (!entries.length) {
    root.textContent = "No extracted highlights for this config.";
    return;
  }
  entries.forEach(([k, v]) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${k}: ${v}`;
    root.appendChild(chip);
  });
}

function applyFrequencyProfile() {
  const key = byId("freqProfile")?.value || "rf_27_cal";
  const p = FREQ_PROFILES[key] || FREQ_PROFILES.rf_27_cal;
  const freq = byId("advFreqHz");
  const eff = byId("advEff");
  const depth = byId("advDepth");
  if (p.frequency_hz !== null && freq) freq.value = String(p.frequency_hz);
  if (p.generator_transfer_efficiency !== null && eff) eff.value = String(p.generator_transfer_efficiency);
  if (p.effective_depth_m !== null && depth) depth.value = String(p.effective_depth_m);
  if (key === "custom") updateDerivedFromFrequency();
  const warn = byId("freqProfileWarning");
  if (warn) warn.textContent = p.warning;
  refreshMatch();
}

function syncFrequencyProfileFromManual() {
  const sel = byId("freqProfile");
  if (!sel) return;
  if (sel.value === "custom") return;
  const freq = Number(byId("advFreqHz")?.value);
  const eff = Number(byId("advEff")?.value);
  const depth = Number(byId("advDepth")?.value);
  const p = FREQ_PROFILES[sel.value];
  if (!p) return;
  const sameFreq = Number.isFinite(freq) && p.frequency_hz !== null && Math.abs(freq - p.frequency_hz) < 1e-6;
  const sameEff =
    Number.isFinite(eff) &&
    p.generator_transfer_efficiency !== null &&
    Math.abs(eff - p.generator_transfer_efficiency) < 1e-9;
  const sameDepth =
    Number.isFinite(depth) &&
    p.effective_depth_m !== null &&
    Math.abs(depth - p.effective_depth_m) < 1e-9;
  if (!sameFreq || !sameEff || !sameDepth) {
    sel.value = "custom";
    const warn = byId("freqProfileWarning");
    if (warn) warn.textContent = FREQ_PROFILES.custom.warning;
  }
}

async function refreshYamlPreview() {
  const pre = byId("yamlPreview");
  if (!pre) return;

  let selected = byId("yamlPreviewSelect")?.value || "__effective__";
  if (selected === "__effective__") {
    try {
      const payload = buildPayload(false);
      const data = await fetchJson("/api/effective-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      renderImportant(data.important || {});
      pre.innerHTML = highlightYaml(data.effective_config_text || "");
      return;
    } catch (err) {
      pre.textContent = `Effective preview error: ${err.message}`;
      renderImportant({});
      return;
    }
  }
  if (selected === "__matched__") selected = state.matchedConfigName || "";
  if (!selected) {
    pre.textContent = "No matched config available yet. Adjust parameters or choose a config manually.";
    renderImportant({});
    return;
  }

  try {
    const data = await fetchJson(`/api/config-preview?name=${encodeURIComponent(selected)}`);
    renderImportant(data.important || {});
    pre.innerHTML = highlightYaml(data.text || "");
  } catch (err) {
    pre.textContent = `Preview error: ${err.message}`;
    renderImportant({});
  }
}

function openViewer(items, index) {
  viewer.items = items;
  viewer.index = index;
  const dlg = byId("imageModal");
  if (!dlg) return;

  const render = () => {
    const cur = viewer.items[viewer.index];
    byId("imageModalImg").src = cur.url;
    byId("imageModalTitle").textContent = cur.title || cur.url;
  };

  byId("imagePrev").onclick = () => {
    viewer.index = (viewer.index - 1 + viewer.items.length) % viewer.items.length;
    render();
  };
  byId("imageNext").onclick = () => {
    viewer.index = (viewer.index + 1) % viewer.items.length;
    render();
  };
  byId("imageModalClose").onclick = () => dlg.close();

  render();
  if (!dlg.open) dlg.showModal();
}

// ── Persistent animation state for job cards ─────────────────────────────
// Keyed by job ID.  Each entry holds references to animated DOM elements so
// they survive the innerHTML rebuild and CSS transitions keep firing.
const _jobAnimState = new Map();

function _getJobAnim(jobId) {
  if (!_jobAnimState.has(jobId)) {
    _jobAnimState.set(jobId, {
      progressFill: null,
      tempFill: null,
      tmaxFill: null,
      meltFill: null,
      densFill: null,
      errFill: null,
      dTFill: null,
      physicsOpen: null,   // null = use default; true/false = user chose
      wasRunning: false,
    });
  }
  return _jobAnimState.get(jobId);
}

// Build the physics-snapshot <details> panel HTML.
// All gauge fill widths start at 0% — they are set live via rAF after paint.
function _buildPhysicsPanel(snap, isOpen) {
  if (!snap) return "";
  const T    = Number(snap.T_mean_c || 0);
  const Tmax = Number(snap.T_max_c  || snap.T_mean_c || 0);
  const dT   = Number(snap.dT_c != null ? snap.dT_c : Math.max(0, Tmax - T));
  const phi  = Number(snap.phi_mean || 0);
  const rho  = Number(snap.rho_mean || 0);
  const err  = Number(snap.err_pct  || 0);
  const step  = Number(snap.step  || 0);
  const total = Number(snap.total || 1);
  const summary =
    `T\u0305\u00a0${T.toFixed(1)}\u00b0C` +
    `\u2002\u00b7\u2002T\u2191\u00a0${Tmax.toFixed(1)}\u00b0C` +
    `\u2002\u00b7\u2002\u0394T\u00a0${dT.toFixed(1)}\u00b0C` +
    `\u2002\u00b7\u2002\u03c6\u0305\u00a0${phi.toFixed(3)}` +
    `\u2002\u00b7\u2002\u03c1\u0305\u00a0${rho.toFixed(3)}` +
    `\u2002\u00b7\u2002err\u00a0${err.toFixed(3)}%`;
  return `<details class="job-physics"${isOpen ? " open" : ""}>
  <summary class="job-physics-summary">${summary}</summary>
  <div class="physics-gauges">
    <div class="gauge-row">
      <span class="gauge-label">T\u2191</span>
      <div class="gauge-track"><div class="gauge-fill gauge-tmax-fill" style="width:0%"></div></div>
      <span class="gauge-val">${Tmax.toFixed(1)}\u00b0C</span>
    </div>
    <div class="gauge-row">
      <span class="gauge-label">T\u0305</span>
      <div class="gauge-track"><div class="gauge-fill gauge-temp-fill" style="width:0%"></div></div>
      <span class="gauge-val">${T.toFixed(1)}\u00b0C</span>
    </div>
    <div class="gauge-row">
      <span class="gauge-label">\u0394T</span>
      <div class="gauge-track"><div class="gauge-fill gauge-dT-fill" style="width:0%"></div></div>
      <span class="gauge-val">${dT.toFixed(1)}\u00b0C</span>
    </div>
    <div class="gauge-row">
      <span class="gauge-label">\u03c6\u0305</span>
      <div class="gauge-track"><div class="gauge-fill gauge-melt-fill" style="width:0%"></div></div>
      <span class="gauge-val">${phi.toFixed(3)}</span>
    </div>
    <div class="gauge-row">
      <span class="gauge-label">\u03c1\u0305</span>
      <div class="gauge-track"><div class="gauge-fill gauge-dens-fill" style="width:0%"></div></div>
      <span class="gauge-val">${rho.toFixed(3)}</span>
    </div>
    <div class="gauge-row">
      <span class="gauge-label">err</span>
      <div class="gauge-track"><div class="gauge-fill gauge-err-fill" style="width:0%"></div></div>
      <span class="gauge-val">${err.toFixed(3)}%</span>
    </div>
  </div>
  <div class="job-physics-step">Step ${step}\u2009/\u2009${total}</div>
</details>`;
}

/**
 * Build an inline SVG sparkline chart for FGM iterate convergence data.
 * Shows σ_T (primary), ρ̄, and the ρ̄=0.82 regime boundary as reference lines.
 * Used live in the job card while fgm_iterate is running.
 */
function _buildFgmConvergenceSparkline(iters) {
  if (!iters || iters.length < 1) return "";
  const W = 240, H = 72, padL = 32, padR = 8, padT = 8, padB = 20;
  const cW = W - padL - padR;
  const cH = H - padT - padB;
  const n  = iters.length;

  const sigTs = iters.map((it) => Number(it.sigma_T ?? 0));
  const rhos  = iters.map((it) => Number(it.mean_rho ?? 0));
  const iNums = iters.map((it) => Number(it.iter ?? 0));

  const sigTMax = Math.max(...sigTs, 1);
  const sigTMin = 0;
  const rhoMax  = Math.max(...rhos, 0.82);
  const rhoMin  = Math.min(...rhos, 0.55);
  const rhoSpan = Math.max(rhoMax - rhoMin, 0.05);
  const iterMax = Math.max(...iNums, n - 1);

  function ix(iter)  { return padL + (iter / Math.max(iterMax, 1)) * cW; }
  function iyT(v)    { return padT + (1 - (v - sigTMin) / (sigTMax - sigTMin || 1)) * cH; }
  function iyR(v)    { return padT + (1 - (v - rhoMin) / rhoSpan) * cH; }

  // Polylines
  const ptT = iters.map((it) => `${ix(it.iter).toFixed(1)},${iyT(Number(it.sigma_T??0)).toFixed(1)}`).join(" ");
  const ptR = iters.map((it) => `${ix(it.iter).toFixed(1)},${iyR(Number(it.mean_rho??0)).toFixed(1)}`).join(" ");

  // ρ̄=0.82 threshold line
  const thresh82Y = iyR(0.82).toFixed(1);
  const threshLine = rhoMin <= 0.82 && 0.82 <= rhoMax
    ? `<line x1="${padL}" y1="${thresh82Y}" x2="${W - padR}" y2="${thresh82Y}"
            stroke="#6050a0" stroke-width="1" stroke-dasharray="3,3" opacity="0.6"/>`
    : "";

  // Dots — gold on best iter
  const bestIter = iters.reduce((b, it) => (it.sigma_T??999) < (b.sigma_T??999) ? it : b, iters[0]);
  const dotsT = iters.map((it) => {
    const isBest = it.iter === bestIter.iter;
    return `<circle cx="${ix(it.iter).toFixed(1)}" cy="${iyT(Number(it.sigma_T??0)).toFixed(1)}"
                    r="${isBest?3.5:2}" fill="${isBest?'#f0c060':'#5090e0'}" opacity="0.9"/>`;
  }).join("");

  // Y-axis labels (σ_T)
  const yLabels = [0, sigTMax / 2, sigTMax].map((v) =>
    `<text x="${padL - 3}" y="${(iyT(v) + 4).toFixed(1)}" text-anchor="end"
            font-size="8" fill="#607080">${v.toFixed(0)}</text>`
  ).join("");

  // X-axis iter labels
  const xLabels = iters.map((it) =>
    `<text x="${ix(it.iter).toFixed(1)}" y="${(H - 5).toFixed(1)}" text-anchor="middle"
            font-size="8" fill="#607080">${it.iter}</text>`
  ).join("");

  const lastT = sigTs[sigTs.length - 1];
  const lastR = rhos[rhos.length - 1];

  return `
    <div style="margin-top:6px;padding:6px 8px;background:#0a0f1a;border-radius:6px;
                border:1px solid #1e2d44;">
      <div style="font-size:10px;color:#7090b0;margin-bottom:4px;display:flex;justify-content:space-between;">
        <span>FGM Convergence — ${n} iter${n!==1?'s':''}</span>
        <span style="color:#5090e0;">σ_T=${lastT.toFixed(1)}°C</span>
        <span style="color:#60c880;">ρ̄=${lastR.toFixed(3)}</span>
      </div>
      <svg width="${W}" height="${H}" style="display:block;">
        <!-- background -->
        <rect x="${padL}" y="${padT}" width="${cW}" height="${cH}"
              fill="#0d1520" rx="2"/>
        <!-- ρ̄=0.82 threshold -->
        ${threshLine}
        <!-- ρ̄ line (right axis, green) -->
        ${ptR.includes(" ") ? `<polyline points="${ptR}" fill="none" stroke="#40a060" stroke-width="1.5" opacity="0.7"/>` : ""}
        <!-- σ_T line (blue) -->
        <polyline points="${ptT}" fill="none" stroke="#5090e0" stroke-width="2"/>
        ${dotsT}
        ${yLabels}
        ${xLabels}
        <!-- axis labels -->
        <text x="${padL - 3}" y="${(padT - 1).toFixed(1)}" text-anchor="end"
              font-size="7" fill="#5090e0">σT</text>
        <text x="${(W - padR + 2).toFixed(1)}" y="${(padT + 4).toFixed(1)}" text-anchor="start"
              font-size="7" fill="#40a060">ρ̄</text>
        <!-- convergence target zone hint if no near-final reached -->
        <text x="${(padL + cW / 2).toFixed(1)}" y="${(H - 5).toFixed(1)}" text-anchor="middle"
              font-size="7" fill="#405060">iter →</text>
      </svg>
    </div>`;
}

function _buildJobCardHTML(j, pct, snap, physicsOpen) {
  const statusCls = j.status;
  const cfg = j.config_resolution?.[0]?.resolved;
  const cfgLine = cfg?.config_name ? `${cfg.match_type}: ${cfg.config_name}` : "";
  const outputs = (j.output_dirs || []).join(", ");
  const totalRuns = Number.isFinite(Number(j.total_runs)) ? Math.max(1, Number(j.total_runs)) : 1;
  const doneRuns = Number.isFinite(Number(j.completed_runs)) ? Math.max(0, Number(j.completed_runs)) : 0;
  const progressLabel = String(j.progress_label || (j.status === "completed" ? "Completed" : "Running"));
  const createdMs = parseIsoMs(j.created_at);
  const startedMs = parseIsoMs(j.started_at);
  const endedMs   = parseIsoMs(j.ended_at);
  let timerLabel = "";
  if (startedMs !== null) {
    const endRef = endedMs !== null ? endedMs : Date.now();
    timerLabel = (endedMs !== null || ["completed", "failed", "cancelled"].includes(String(j.status)))
      ? `Duration: ${formatDuration((endRef - startedMs) / 1000)}`
      : `Elapsed: ${formatDuration((endRef - startedMs) / 1000)}`;
  } else if (createdMs !== null && (String(j.status) === "queued" || String(j.status) === "paused")) {
    timerLabel = `Waiting: ${formatDuration((Date.now() - createdMs) / 1000)}`;
  }
  const queuePos = Number.isFinite(Number(j.queue_position)) ? Number(j.queue_position) : null;
  const queueControls = (j.status === "queued")
    ? `<div class="queue-controls">
        <span class="muted">Queue #${queuePos ?? "?"}</span>
        <button class="queue-btn" data-jobid="${j.id}" data-dir="up" type="button">Up</button>
        <button class="queue-btn" data-jobid="${j.id}" data-dir="down" type="button">Down</button>
      </div>`
    : "";
  let controlButtons = "";
  if (j.status === "queued") {
    controlButtons = `<div class="job-controls">
      <button class="job-ctl-btn" data-jobid="${j.id}" data-action="pause" type="button">Pause</button>
      <button class="job-ctl-btn danger" data-jobid="${j.id}" data-action="cancel" type="button">Cancel</button>
    </div>`;
  } else if (j.status === "running") {
    controlButtons = `<div class="job-controls">
      <button class="job-ctl-btn" data-jobid="${j.id}" data-action="pause" type="button">Pause</button>
      <button class="job-ctl-btn danger" data-jobid="${j.id}" data-action="cancel" type="button">Cancel</button>
    </div>`;
  } else if (j.status === "paused") {
    controlButtons = `<div class="job-controls">
      <button class="job-ctl-btn" data-jobid="${j.id}" data-action="resume" type="button">Resume</button>
      <button class="job-ctl-btn danger" data-jobid="${j.id}" data-action="cancel" type="button">Cancel</button>
    </div>`;
  } else if (j.status === "cancelling") {
    controlButtons = `<div class="job-controls"><span class="muted">Cancelling...</span></div>`;
  }
  // ETA (running jobs only)
  const etaLabel = (j.status === "running") ? _getEtaLabel(j.id, pct) : "";
  const etaHTML  = etaLabel
    ? `<span style="color:#80d0c0;font-size:11px;font-family:monospace;margin-left:8px;">${etaLabel}</span>`
    : "";

  return `
    <div class="job-head">
      <strong>${j.id}</strong>
      <span class="badge ${statusCls}">${j.status}</span>
      ${etaHTML}
    </div>
    <div class="job-progress-wrap">
      <div class="job-progress-label">${progressLabel}</div>
      <div class="job-progress-meta">${doneRuns}/${totalRuns} \u2022 ${pct.toFixed(0)}%</div>
    </div>
    <div class="job-progress-track">
      <div class="job-progress-fill ${statusCls}" style="width:0%"></div>
    </div>
    ${_buildPhysicsPanel(snap, physicsOpen)}
    ${j.mode === "fgm_iterate" && Array.isArray(j.convergence_iters) && j.convergence_iters.length
        ? _buildFgmConvergenceSparkline(j.convergence_iters)
        : ""}
    <div>${j.mode} \u2022 ${j.output_name || "(batch)"}</div>
    <div>${j.started_at || j.created_at || ""}</div>
    <div class="muted">${timerLabel}</div>
    <div class="muted">${cfgLine}</div>
    ${queueControls}
    ${controlButtons}
    <div class="muted">${outputs || "No outputs yet"}</div>
    <div style="display:flex;gap:10px;align-items:center;">
      <a href="${j.log_url}" target="_blank" style="font-size:11px;">\ud83d\udcc4 Open log</a>
      <button class="job-log-toggle" data-jobid="${j.id}" data-logurl="${j.log_url}"
              type="button" style="font-size:11px;padding:2px 8px;border-radius:3px;
              background:#0d1a2a;border:1px solid #1e3a50;color:#7090b0;cursor:pointer;">
        \ud83d\udccb Inline log
      </button>
    </div>
    <div class="job-inline-log" style="display:none;"></div>
  `;
}

function _attachJobCardListeners(el, j) {
  el.querySelectorAll(".queue-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id  = btn.getAttribute("data-jobid");
      const dir = btn.getAttribute("data-dir");
      if (!id || !dir) return;
      try { await reorderQueuedJob(id, dir); } catch (err) { console.error(err); }
    });
  });
  el.querySelectorAll(".job-ctl-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id     = btn.getAttribute("data-jobid");
      const action = btn.getAttribute("data-action");
      if (!id || !action) return;
      try { await controlJob(id, action); } catch (err) { console.error(err); }
    });
  });

  // ── Inline log panel ────────────────────────────────────────────────────
  const logToggle = el.querySelector(".job-log-toggle");
  const logPanel  = el.querySelector(".job-inline-log");
  if (logToggle && logPanel) {
    let logPollId = null;

    async function fetchLog() {
      const logUrl = logToggle.getAttribute("data-logurl");
      if (!logUrl) return;
      try {
        const res = await fetch(logUrl);
        if (!res.ok) { logPanel.textContent = "Log unavailable."; return; }
        const text = await res.text();
        const lines = text.split("\n");
        const tail  = lines.slice(-80).join("\n");
        // Preserve scroll position: only auto-scroll if user is near bottom
        const atBottom = logPanel.scrollHeight - logPanel.scrollTop - logPanel.clientHeight < 40;
        logPanel.textContent = tail;
        if (atBottom) logPanel.scrollTop = logPanel.scrollHeight;
      } catch (_) { logPanel.textContent = "Could not fetch log."; }
    }

    function openLog() {
      OPEN_LOGS.add(j.id);
      logToggle.textContent = "📋 Hide log";
      logPanel.style.cssText = "display:block;max-height:200px;overflow-y:auto;" +
        "background:#050a10;border:1px solid #1e2d44;border-radius:4px;" +
        "padding:8px;margin-top:6px;font-family:monospace;font-size:10px;color:#7090b0;";
      if (!logPanel.textContent || logPanel.textContent === "Loading…") {
        logPanel.textContent = "Loading…";
        fetchLog();
      }
      if (!logPollId && j.status === "running") {
        logPollId = setInterval(fetchLog, 3000);
      }
    }

    function closeLog() {
      OPEN_LOGS.delete(j.id);
      logToggle.textContent = "📋 Inline log";
      logPanel.style.display = "none";
      if (logPollId) { clearInterval(logPollId); logPollId = null; }
    }

    logToggle.addEventListener("click", () => {
      if (OPEN_LOGS.has(j.id)) closeLog(); else openLog();
    });

    // Restore open state if this job had its log open before re-render
    if (OPEN_LOGS.has(j.id)) openLog();
  }
}

function renderJobs(jobs) {
  const root = byId("jobs");
  if (!root) return;
  root.innerHTML = "";

  if (!jobs.length) {
    root.textContent = "No jobs yet.";
    _jobAnimState.clear();
    return;
  }

  const hasRunning = jobs.some((j) => j.status === "running");
  const seenIds = new Set(jobs.map((j) => j.id));
  _jobAnimState.forEach((_, id) => { if (!seenIds.has(id)) _jobAnimState.delete(id); });
  // Clean up OPEN_LOGS for jobs no longer in the list
  OPEN_LOGS.forEach((id) => { if (!seenIds.has(id)) OPEN_LOGS.delete(id); });

  jobs.forEach((j) => {
    const rawPct = Number.isFinite(Number(j.progress_pct))
      ? Number(j.progress_pct)
      : (j.status === "completed" ? 100 : 0);
    const pct  = Math.max(0, Math.min(100, rawPct));
    const snap = j.physics_snapshot || null;
    const anim = _getJobAnim(j.id);

    // Default open state: open for running/paused jobs, closed for everything else.
    // Once the user toggles the panel we respect their choice.
    if (anim.physicsOpen === null) {
      anim.physicsOpen = (j.status === "running" || j.status === "paused");
    } else if (!anim.wasRunning && hasRunning && j.status !== "running" && j.status !== "paused") {
      // Auto-collapse queued/completed jobs that were never running when another
      // job starts (only fires once per job, then user controls it).
      anim.physicsOpen = false;
    }
    anim.wasRunning = anim.wasRunning || (j.status === "running");

    // Update ETA tracker for running jobs
    if (j.status === "running") _updateEta(j.id, pct);
    // Clean up ETA state for finished jobs
    if (["completed","failed","cancelled"].includes(j.status)) JOB_ETA.delete(j.id);

    const el = document.createElement("div");
    el.className = "job";
    el.innerHTML = _buildJobCardHTML(j, pct, snap, !!anim.physicsOpen);

    // ── Transplant persistent animated fill elements ──────────────────────
    // By reusing the same DOM node across renders the CSS `transition` fires
    // properly (it sees a change on an existing element instead of a new one).
    function transplant(selector, animKey, newClass) {
      const placeholder = el.querySelector(selector);
      if (!placeholder) return;
      if (anim[animKey]) {
        if (newClass) anim[animKey].className = newClass;
        placeholder.parentNode.replaceChild(anim[animKey], placeholder);
      } else {
        anim[animKey] = placeholder;
      }
    }
    transplant(".job-progress-fill", "progressFill", `job-progress-fill ${j.status}`);
    transplant(".gauge-temp-fill",   "tempFill");
    transplant(".gauge-tmax-fill",   "tmaxFill");
    transplant(".gauge-dT-fill",     "dTFill");
    transplant(".gauge-melt-fill",   "meltFill");
    transplant(".gauge-dens-fill",   "densFill");
    transplant(".gauge-err-fill",    "errFill");

    _attachJobCardListeners(el, j);
    root.appendChild(el);

    // ── Animate fill widths after first paint ─────────────────────────────
    requestAnimationFrame(() => {
      if (anim.progressFill) anim.progressFill.style.width = `${pct.toFixed(1)}%`;
      if (snap) {
        const T    = Number(snap.T_mean_c || 0);
        const Tmax = Number(snap.T_max_c  || snap.T_mean_c || 0);
        const dT   = Number(snap.dT_c != null ? snap.dT_c : Math.max(0, Tmax - T));
        const phi  = Number(snap.phi_mean || 0);
        const rho  = Number(snap.rho_mean || 0);
        const err  = Number(snap.err_pct  || 0);
        // Temperature: 25 °C (ambient) → 240 °C (headroom above 231 °C max observed)
        // DSC ticks at 175 / 180 / 185 °C are rendered via CSS ::after on the track.
        if (anim.tempFill) anim.tempFill.style.width =
          `${Math.min(100, Math.max(0, (T - 25) / (240 - 25) * 100)).toFixed(1)}%`;
        if (anim.tmaxFill) anim.tmaxFill.style.width =
          `${Math.min(100, Math.max(0, (Tmax - 25) / (240 - 25) * 100)).toFixed(1)}%`;
        // ΔT = T_max − T_mean: 0 °C → 50 °C scale; amber bar; lower is better
        if (anim.dTFill) anim.dTFill.style.width =
          `${Math.min(100, Math.max(0, dT / 50.0 * 100)).toFixed(1)}%`;
        // Melt fraction: 0 → 1
        if (anim.meltFill) anim.meltFill.style.width =
          `${Math.min(100, Math.max(0, phi * 100)).toFixed(1)}%`;
        // Relative density: powder (~0.45) → fully dense (1.0)
        if (anim.densFill) anim.densFill.style.width =
          `${Math.min(100, Math.max(0, (rho - 0.45) / 0.55 * 100)).toFixed(1)}%`;
        // Energy error: 0 % → 2 % cap
        if (anim.errFill) anim.errFill.style.width =
          `${Math.min(100, Math.max(0, err / 2.0 * 100)).toFixed(1)}%`;
      }
    });

    // ── Track user open/close toggle ──────────────────────────────────────
    const details = el.querySelector(".job-physics");
    if (details) {
      details.addEventListener("toggle", () => {
        anim.physicsOpen = details.open;
      });
    }
  });
}

function renderLiveArtifacts(jobs) {
  const root = byId("completedArtifacts");
  if (!root) return;
  root.innerHTML = "";

  const entries = [];
  jobs.forEach((job) => {
    (job.artifacts || []).forEach((a) => {
      entries.push({ job, artifact: a });
    });
  });

  if (!entries.length) {
    root.textContent = "Figures appear here as each run writes output files.";
    return;
  }

  entries.slice().reverse().forEach(({ job, artifact }) => {
    const card = document.createElement("article");
    card.className = "artifact-card";
    const imgs = (artifact.images || []).slice(0, 12);

    card.innerHTML = `
      <h4>${artifact.output_dir}</h4>
      <div class="muted">job ${job.id} • ${job.status}</div>
      <div class="artifact-grid"></div>
    `;

    const grid = card.querySelector(".artifact-grid");
    imgs.forEach((img, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "thumb-btn";
      btn.innerHTML = `<img src="${img.url}" alt="${img.path}" loading="lazy" />`;
      btn.addEventListener("click", () => {
        const items = imgs.map((x) => ({ url: x.url, title: `${artifact.output_dir}/${x.path}` }));
        openViewer(items, idx);
      });
      grid.appendChild(btn);
    });

    root.appendChild(card);
  });
}

async function loadMeta() {
  const meta = await fetchJson("/api/meta");
  state.meta = meta;
  renderMeta(meta);
  renderModelInfo(meta);
  const defaultFamily = meta?.default_model_family || "experimental_pa12_hybrid";
  const fam = byId("physicsModelFamily");
  const en = byId("physicsExperimentalEnabled");
  if (fam && [...fam.options].some((o) => o.value === defaultFamily)) {
    fam.value = defaultFamily;
  }
  if (en) {
    en.value = defaultFamily === "baseline" ? "false" : "true";
  }
  if (byId("physicsProvenanceFile") && meta?.default_provenance_file) {
    byId("physicsProvenanceFile").value = meta.default_provenance_file;
  }
  if (byId("physicsDscProfileFile") && meta?.default_dsc_profile_file) {
    byId("physicsDscProfileFile").value = meta.default_dsc_profile_file;
  }
  const dsc = meta?.model_info?.dsc_profile || {};
  if (byId("expMeltOnsetC") && Number.isFinite(Number(dsc.melt_onset_c))) byId("expMeltOnsetC").value = String(dsc.melt_onset_c);
  if (byId("expMeltPeakC") && Number.isFinite(Number(dsc.melt_peak_c))) byId("expMeltPeakC").value = String(dsc.melt_peak_c);
  if (byId("expMeltEndC") && Number.isFinite(Number(dsc.melt_end_c))) byId("expMeltEndC").value = String(dsc.melt_end_c);
  if (byId("expLatHeat") && Number.isFinite(Number(dsc.lat_heat_j_per_kg))) byId("expLatHeat").value = String(dsc.lat_heat_j_per_kg);
  renderShapePreview(byId("shape")?.value || "square");
}

let _loadJobsFailCount = 0;

async function loadJobs() {
  try {
    state.jobs = await fetchJson("/api/jobs");
    renderJobs(state.jobs);
    renderLiveArtifacts(state.jobs);
    if (_loadJobsFailCount > 0) {
      // Reconnected — restore status pill
      _loadJobsFailCount = 0;
      const s = byId("serverStatus");
      if (s) { s.textContent = "HEATR service connected"; s.classList.remove("status-disconnected"); }
    }
  } catch (e) {
    _loadJobsFailCount++;
    const s = byId("serverStatus");
    if (s) {
      s.textContent = _loadJobsFailCount < 5
        ? `Reconnecting… (${_loadJobsFailCount})`
        : "HEATR service unreachable";
      s.classList.add("status-disconnected");
    }
    throw e;
  }
}

function renderMatchInfo(data) {
  const box = byId("configMatch");
  if (!box) return;
  if (data.mode === "sweep") {
    const lines = (data.matches || []).map((m) => `${m.minutes} min -> ${m.resolved.match_type}: ${m.resolved.config_name}`);
    box.textContent = lines.join("\n") || "No match info.";
    const firstCfg = data.matches?.[0]?.resolved?.config_name || "";
    state.matchedConfigName = firstCfg;
    return;
  }
  const resolved = data.resolved || {};
  box.textContent = `${resolved.match_type || "unknown"}: ${resolved.config_name || "n/a"}`;
  state.matchedConfigName = resolved.config_name || "";
}

// ── Live Exposure Calculator ──────────────────────────────────────────────────
// Shows a T_max estimate as the user adjusts exposure time.
// Strategy (priority order):
//   1. If a completed run for the same shape exists in the run list, scale its
//      T_max proportionally: T_rise ∝ t^0.55  (simple thermal diffusion model)
//   2. Otherwise hide the chip — we have no reliable anchor for the heuristic.
//
// Note: match-config returns only a matched YAML, not temperature data, so it
// cannot provide a calibration anchor on its own.

const _EXP_T_AMB  = 25.0;   // °C ambient
const _EXP_T_MELT = 185.0;  // PA12 melt onset
const _EXP_T_CRIT = 220.0;  // over-melt danger threshold

let _expCalcLastMatch  = null;  // match-config response (YAML only — no T data)
let _expCalibCache     = {};    // shape → { refTmax, refExpMin } from run list

// Called once after the run list loads; extracts the most recent completed
// single run for each shape to use as a calibration anchor.
function _buildExpCalibFromRuns(runs) {
  const byShape = {};
  for (const r of runs) {
    if (r.run_type !== "single") continue;
    const shape = r.group || "";
    const ex = r.summary_excerpt || {};
    const tmax = ex.max_T_part_c;
    const tFinalS = ex.t_final_s;
    if (!tmax || !tFinalS) continue;
    const expMin = tFinalS / 60.0;
    if (expMin <= 0 || tmax <= _EXP_T_AMB) continue;
    // Keep the most recent (first encountered, list is newest-first)
    if (!byShape[shape]) {
      byShape[shape] = { refTmax: tmax, refExpMin: expMin };
    }
  }
  _expCalibCache = byShape;
  _updateExpCalc();  // re-render chip with new calibration
}

function _updateExpCalc() {
  const chip = byId("expCalcChip");
  if (!chip) return;

  const expMin = parseFloat(byId("exposureMinutes")?.value);
  const shape  = byId("shape")?.value || "square";
  if (!expMin || expMin <= 0) { chip.classList.add("hidden"); return; }

  const calib = _expCalibCache[shape];
  if (!calib) {
    // No completed run for this shape — hide rather than show a wrong number
    chip.classList.add("hidden");
    return;
  }

  // Scale from calibration anchor: T_rise ∝ t^0.55
  const { refTmax, refExpMin } = calib;
  const refRise = refTmax - _EXP_T_AMB;
  const tMaxEst = _EXP_T_AMB + refRise * Math.pow(expMin / refExpMin, 0.55);
  const deltaT  = tMaxEst - _EXP_T_AMB;

  let tClass = "exp-est-val";
  let tHint  = "";
  if (tMaxEst >= _EXP_T_CRIT) {
    tClass = "exp-est-danger";
    tHint  = " ⚠ over-melt risk";
  } else if (tMaxEst >= _EXP_T_MELT + 10) {
    tClass = "exp-est-warn";
    tHint  = " above melt onset";
  } else if (tMaxEst >= _EXP_T_MELT - 5) {
    tHint  = " near melt onset";
  } else if (tMaxEst < _EXP_T_MELT - 20) {
    tHint  = " below melt";
  }

  chip.classList.remove("hidden");
  chip.innerHTML =
    `<span>Est. T_max</span>` +
    `<span class="${tClass}">${tMaxEst.toFixed(0)}°C${tHint}</span>` +
    `<span>ΔT≈${deltaT.toFixed(0)}°C rise</span>` +
    `<span style="color:#2a4a60;font-size:10px;" title="Scaled from ${shape} run: ` +
    `${refTmax.toFixed(0)}°C @ ${refExpMin.toFixed(1)}min using T_rise∝t^0.55">` +
    `(±30°C · scaled from prior ${shape} run)</span>`;
}

async function refreshMatch() {
  const box = byId("configMatch");
  if (!box) return;
  try {
    const payload = buildPayload(false);
    const data = await fetchJson("/api/match-config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    _expCalcLastMatch = data;
    renderMatchInfo(data);
    await refreshYamlPreview();
  } catch (err) {
    box.textContent = `Match error: ${err.message}`;
  }
}

async function launchRun(ev) {
  ev.preventDefault();
  const payload = buildPayload(true);
  const mode = payload.mode || "single";

  // fgm_import uses a dedicated endpoint (not /api/run)
  if (mode === "fgm_import") {
    // Handled entirely by #fgmImportLaunchBtn — the submit button is hidden for this mode
    return;
  }

  // fgm_iterate uses a dedicated endpoint (not /api/run)
  if (mode === "fgm_iterate") {
    const btn = ev.target?.querySelector('[type="submit"]');
    if (btn) { btn.disabled = true; btn.textContent = "Queuing…"; }
    try {
      const resp = await fetchJson("/api/tools/fgm-iterate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (btn) { btn.disabled = false; btn.textContent = "Launch Run"; }
      await loadJobs();
    } catch (err) {
      if (btn) { btn.disabled = false; btn.textContent = "Launch Run"; }
      alert("FGM Iterate error: " + err.message);
    }
    return;
  }

  // prewarp (Geometry Pre-Warp / level-set ILT) uses a dedicated endpoint
  if (mode === "prewarp") {
    const btn = ev.target?.querySelector('[type="submit"]');
    if (btn) { btn.disabled = true; btn.textContent = "Queuing…"; }
    try {
      await fetchJson("/api/tools/prewarp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (btn) { btn.disabled = false; btn.textContent = "Launch Run"; }
      await loadJobs();
    } catch (err) {
      if (btn) { btn.disabled = false; btn.textContent = "Launch Run"; }
      alert("Geometry Pre-Warp error: " + err.message);
    }
    return;
  }

  const antennaeEnabled = byId("antennaeEnabled")?.checked || false;
  if (antennaeEnabled) {
    payload.antennae_enabled = true;
    AntennaWorkshop.open(payload);
  } else {
    try {
      await fetchJson("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      await loadJobs();
    } catch (err) {
      alert("Launch error: " + err.message);
    }
  }
}

// ── Quick-launch Preset System ────────────────────────────────────────────────
// Captures all named form inputs + selects at save time; restores at load time.
// Stored in localStorage under key "heatr_presets" as JSON array of {name, ts, values}.

const PRESET_KEY = "heatr_presets";

function _getPresets() {
  try { return JSON.parse(localStorage.getItem(PRESET_KEY) || "[]"); } catch (_) { return []; }
}
function _savePresets(ps) {
  localStorage.setItem(PRESET_KEY, JSON.stringify(ps.slice(-20))); // cap at 20
}

function _captureFormValues() {
  const form = byId("runForm");
  if (!form) return {};
  const vals = {};
  form.querySelectorAll("input[id],select[id],textarea[id]").forEach((el) => {
    if (!el.id) return;
    if (el.type === "checkbox") vals[el.id] = el.checked;
    else vals[el.id] = el.value;
  });
  return vals;
}

function _applyFormValues(vals) {
  if (!vals || typeof vals !== "object") return;
  Object.entries(vals).forEach(([id, val]) => {
    const el = byId(id);
    if (!el) return;
    if (el.type === "checkbox") el.checked = Boolean(val);
    else el.value = String(val);
    // Fire change events so dependent UI updates
    el.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

function _renderPresetDropdown() {
  const sel = byId("presetSelect");
  if (!sel) return;
  const ps = _getPresets();
  sel.innerHTML = '<option value="">— Load preset —</option>' +
    ps.map((p, i) => `<option value="${i}">${p.name} (${new Date(p.ts).toLocaleDateString()})</option>`).join("");
}

function _initPresets() {
  _renderPresetDropdown();

  const saveBtn = byId("presetSaveBtn");
  if (saveBtn) {
    saveBtn.addEventListener("click", () => {
      const name = window.prompt("Preset name:", `${byId("shape")?.value || "run"}_${byId("mode")?.value || "single"}`);
      if (!name?.trim()) return;
      const ps = _getPresets();
      ps.push({ name: name.trim(), ts: Date.now(), values: _captureFormValues() });
      _savePresets(ps);
      _renderPresetDropdown();
      saveBtn.textContent = "✓ Saved";
      setTimeout(() => { saveBtn.textContent = "💾 Save Preset"; }, 1500);
    });
  }

  const loadBtn = byId("presetLoadBtn");
  if (loadBtn) {
    loadBtn.addEventListener("click", () => {
      const sel = byId("presetSelect");
      if (!sel || sel.value === "") return;
      const idx = parseInt(sel.value);
      const ps  = _getPresets();
      if (!ps[idx]) return;
      _applyFormValues(ps[idx].values);
      loadBtn.textContent = "✓ Loaded";
      setTimeout(() => { loadBtn.textContent = "📂 Load"; }, 1500);
      // Re-run dependent UI refresh
      void (async () => {
        setModeSections();
        refreshOutputNamePrefix();
        renderShapePreview(byId("shape")?.value || "square");
        await refreshMatch();
      })();
    });
  }

  const deleteBtn = byId("presetDeleteBtn");
  if (deleteBtn) {
    deleteBtn.addEventListener("click", () => {
      const sel = byId("presetSelect");
      if (!sel || sel.value === "") return;
      const idx = parseInt(sel.value);
      const ps  = _getPresets();
      if (!ps[idx]) return;
      if (!window.confirm(`Delete preset "${ps[idx].name}"?`)) return;
      ps.splice(idx, 1);
      _savePresets(ps);
      _renderPresetDropdown();
    });
  }
}

async function init() {
  const form = byId("runForm");
  if (!form) return;

  _initPresets();

  byId("mode")?.addEventListener("change", async () => {
    setModeSections();
    refreshOutputNamePrefix();
    refreshTurntableInfo();
    await refreshMatch();
  });

  form.addEventListener("submit", launchRun);

  byId("shape")?.addEventListener("change", () => {
    refreshOutputNamePrefix();
    renderShapePreview(byId("shape")?.value || "square");
    refreshGeometrySizeUI();
    updateDependentVisibility();
    refreshMatch();
  });
  byId("geometrySizeEnabled")?.addEventListener("change", () => {
    updateDependentVisibility();
    refreshMatch();
  });
  byId("geometrySizeMm")?.addEventListener("input", refreshMatch);
  byId("geometrySizeMm")?.addEventListener("change", refreshMatch);
  byId("exposureMinutes")?.addEventListener("input", _updateExpCalc);
  byId("exposureMinutes")?.addEventListener("change", _updateExpCalc);
  byId("geometryLockAspect")?.addEventListener("change", refreshMatch);
  byId("physicsModelFamily")?.addEventListener("change", () => {
    syncExperimentalControls();
    updateDependentVisibility();
    refreshMatch();
  });
  byId("physicsExperimentalEnabled")?.addEventListener("change", updateDependentVisibility);
  byId("placementUseTurntable")?.addEventListener("change", updateDependentVisibility);
  byId("shellEnabled")?.addEventListener("change", updateDependentVisibility);
  byId("expCrEnabled")?.addEventListener("change", updateDependentVisibility);
  byId("expViscosityModel")?.addEventListener("change", updateDependentVisibility);
  byId("turntableRotationDeg")?.addEventListener("input", refreshTurntableInfo);
  byId("turntableRotationDeg")?.addEventListener("change", refreshTurntableInfo);
  byId("turntableTotalRotations")?.addEventListener("input", refreshTurntableInfo);
  byId("turntableTotalRotations")?.addEventListener("change", refreshTurntableInfo);
  byId("outputName")?.addEventListener("input", () => {
    state.outputNameTouched = true;
  });
  byId("freqProfile")?.addEventListener("change", applyFrequencyProfile);
  byId("advFreqHz")?.addEventListener("input", () => {
    updateDerivedFromFrequency();
    syncFrequencyProfileFromManual();
  });
  byId("advFreqHz")?.addEventListener("change", () => {
    updateDerivedFromFrequency();
    syncFrequencyProfileFromManual();
  });
  byId("advEff")?.addEventListener("input", syncFrequencyProfileFromManual);
  byId("advEff")?.addEventListener("change", syncFrequencyProfileFromManual);

  [
    "shape", "geometrySizeEnabled", "geometryNominalMm", "geometrySizeMm", "geometryLockAspect",
    "exposureMinutes", "sweepMinutes", "phiSnapshots", "tempCeiling", "highlightPhi",
    "freqProfile",
    "turntableRotationDeg", "turntableTotalRotations", "turntableIntervalS",
    "orientationAngleMinDeg", "orientationAngleMaxDeg", "orientationAngleStepDeg",
    "orientationRefineWindowDeg", "orientationRefineStepDeg",
    "orientationExposureMinS", "orientationExposureMaxS", "orientationExposureStepS",
    "orientationTempCeilingC", "orientationMinRhoFloor", "orientationAngleGifEnabled", "orientationAngleGifFrameDurationS", "orientationColorMetric",
    "placementNParts", "placementPartWidthMm", "placementPartHeightMm",
    "placementUseTurntable", "placementTurntableRotationDeg", "placementTurntableTotalRotations", "placementTurntableIntervalS",
    "placementClearanceMm",
    "placementSearchDomainMarginMm", "placementProxyTopK",
    "placementAlgorithm", "placementProxyPopulation", "placementProxyIters", "placementProxyEvalBudget",
    "placementGaPopulation", "placementGaGenerations", "placementGaCrossoverRate", "placementGaMutationRate",
    "placementGaElitism", "placementGaTournamentK", "placementGaSeed",
    "placementTempCeilingC", "placementMinRhoFloor",
    "shellSweepThicknessesMm", "shellSweepShapes", "shellTempCeilingC",
    "shellEnabled", "shellWallThicknessMm", "shellMethod",
    "advGridNx", "advGridNy", "advFreqHz", "advVoltage", "advPower", "advEnforceGen", "advEff",
    "advDepth", "advDt", "advAmbient", "advConv", "advSigma", "advDopedSigmaProfile",
    "advVirginSigma", "advVirginEps", "advPowderRho", "advPowderK", "advPowderCp",
    "advDopedEps", "advSigmaTempCoeff", "advSigmaDensityCoeff", "advSigmaRefTemp",
    "advDopedRhoSolid", "advDopedRhoLiquid", "advDopedKSolid", "advDopedKLiquid",
    "advDopedCpSolid", "advDopedCpLiquid",
    "advMaxQrf", "advMaxTemp", "baseConfigSelect",
    "physicsModelFamily", "physicsExperimentalEnabled", "physicsProvenanceTag",
    "physicsParameterSource", "physicsCalibrationVersion", "physicsAbBucketId", "physicsProvenanceFile",
    "physicsDscProfileFile",
    "expPhaseType", "expMeltOnsetC", "expMeltPeakC", "expMeltEndC", "expLatHeat", "expCpSmoothing",
    "expDensType", "expViscosityModel", "expEtaRef", "expEtaRefTempK", "expEtaEa", "expWlfC1", "expWlfC2",
    "expSurfaceTension", "expParticleRadius", "expPhiExponent", "expPhiThreshold", "expPorosityExponent", "expGeomFactor",
    "expCrEnabled", "expCrType", "expCrK0", "expCrEa", "expCrExponent", "expCrSuppression"
  ].forEach((id) => {
    const el = byId(id);
    if (el) {
      el.addEventListener("change", refreshMatch);
      el.addEventListener("input", refreshMatch);
    }
  });

  byId("yamlPreviewSelect")?.addEventListener("change", refreshYamlPreview);

  AntennaWorkshop.init();
  setModeSections();
  attachParamInfoIcons();
  await loadMeta();
  refreshGeometrySizeUI();
  syncExperimentalControls();
  updateDependentVisibility();
  applyFrequencyProfile();
  refreshOutputNamePrefix();
  refreshTurntableInfo();
  byId("fgmIterTimeSource")?.addEventListener("change", updateFgmIterTimeSourceHelp);
  byId("fgmIterProxy")?.addEventListener("change", updateFgmProxyHelp);
  byId("fgmIterCorrMode")?.addEventListener("change", updateFgmIterCorrModeHelp);
  updateFgmProxyHelp();          // set initial help text and show/hide thorough opts
  updateFgmIterCorrModeHelp();   // set initial correction mode help text

  // ── FGM Import PNG mode ───────────────────────────────────────────────────
  // Populate the source-run datalist for fgm_import
  (async () => {
    try {
      const runs = await fetchJson("/api/results-runview");
      const list = Array.isArray(runs) ? runs : [];
      const dl = byId("fgmImportRunList");
      if (dl) {
        dl.innerHTML = list.map((r) => {
          const name = r.name || r;
          return `<option value="${name}">`;
        }).join("");
      }
    } catch (_) {}
  })();

  byId("fgmImportLaunchBtn")?.addEventListener("click", async () => {
    const statusEl = byId("fgmImportStatus");
    const btn = byId("fgmImportLaunchBtn");
    const sourceRun = (byId("fgmImportSourceRun")?.value || "").trim();
    const fileInput = byId("fgmImportFile");
    const file = fileInput?.files?.[0];
    const bpp = parseInt(byId("fgmImportBpp")?.value || "2", 10);
    const outputName = (byId("fgmImportOutputName")?.value || "").trim();
    const invertVis = (byId("fgmImportInvertVis")?.value || "true") === "true";
    const magnitude = parseFloat(byId("fgmImportMagnitude")?.value || "1.0") || 1.0;

    if (!sourceRun) {
      if (statusEl) statusEl.textContent = "⚠ Please specify a source run.";
      return;
    }
    if (!file) {
      if (statusEl) statusEl.textContent = "⚠ Please select a PNG file.";
      return;
    }

    btn.disabled = true;
    btn.textContent = "Importing…";
    if (statusEl) statusEl.textContent = `Reading ${file.name}…`;

    try {
      // Read PNG as base64
      const png_b64 = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result.split(",")[1]);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      if (statusEl) statusEl.textContent = "Uploading to server…";

      const resp = await fetchJson("/api/tools/import-fgm-png", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_run_dir: sourceRun,
          png_b64,
          bpp,
          output_name: outputName,
          invert_vis:  invertVis,
          magnitude,
        }),
      });

      if (!resp.ok) throw new Error(resp.error || "import-fgm-png failed");
      const msg =
        `✓ Job ${resp.job_id} queued (${resp.status})\n` +
        `  Output: ${resp.output_name}   ${resp.bpp}bpp   ` +
        `${(resp.level_map_shape || []).join("×")} px`;
      if (statusEl) statusEl.textContent = msg;
      await loadJobs();
      // Clear the file input for next use
      if (fileInput) fileInput.value = "";
    } catch (err) {
      if (statusEl) statusEl.textContent = `✗ Error: ${err.message}`;
    } finally {
      btn.disabled = false;
      btn.textContent = "📥 Import & Run";
    }
  });

  await Promise.all([loadJobs(), refreshMatch()]);

  // Poll jobs every second — this also acts as the primary liveness check.
  setInterval(async () => {
    try { await loadJobs(); } catch { /* status updated inside loadJobs */ }
  }, 1000);

  // Dedicated heartbeat every 30 s — keeps the TCP connection alive through
  // NAT/proxy idle-timeout windows that are longer than 1 s but shorter than
  // the browser's own idle-tab throttle (which can slow setInterval to ~1 min).
  setInterval(async () => {
    try { await fetch("/api/ping"); } catch { /* ignore — loadJobs handles status */ }
  }, 30_000);

  // QUIT button — graceful server shutdown
  const quitBtn = byId("quitServerBtn");
  if (quitBtn) {
    quitBtn.addEventListener("click", async () => {
      if (!confirm("Stop the HEATR server?\n\nRunning jobs will be allowed to finish, but queued jobs will be lost.")) return;
      quitBtn.disabled = true;
      quitBtn.textContent = "Stopping…";
      try {
        await fetch("/api/quit", { method: "POST" });
        const s = byId("serverStatus");
        if (s) { s.textContent = "Server stopped"; s.classList.add("status-disconnected"); }
        quitBtn.textContent = "Stopped";
      } catch {
        quitBtn.disabled = false;
        quitBtn.textContent = "⏻ Quit Server";
      }
    });
  }

  setInterval(() => {
    try {
      renderJobs(state.jobs || []);
    } catch {
      // no-op: timer refresh is best-effort.
    }
  }, 1000);
}

// ─── Antenna Workshop ────────────────────────────────────────────────────────
const AntennaWorkshop = {
  // Runtime state
  formPayload: null,
  geometry: null,       // {chamber_x_m, chamber_y_m, parts:[{polygon_m,center_x,center_y,...}]}
  antennas: [],         // [{id, center_x, center_y, size_x_mm, size_y_mm}]  — positions in metres
  selectedIdx: -1,
  drag: null,           // {idx, svgStartX, svgStartY, simStartX, simStartY}
  heatmapUrl: null,
  nextId: 0,
  toolMode: "select",   // "select" | "add"
  _dlg: null,           // cached <dialog> element ref
  savedSession: null,   // {antennas, geometry} preserved after launch for reopen

  SVG_W: 500,
  SVG_H: 500,
  PADDING: 30,

  // ── coordinate helpers ──────────────────────────────────────────────────────
  _scale() {
    const g = this.geometry;
    if (!g) return 1;
    const dw = this.SVG_W - 2 * this.PADDING;
    const dh = this.SVG_H - 2 * this.PADDING;
    return Math.min(dw / (g.chamber_x_m * 1000), dh / (g.chamber_y_m * 1000));
  },
  _offset() {
    const g = this.geometry;
    if (!g) return { x: this.PADDING, y: this.PADDING };
    const sc = this._scale();
    const dw = this.SVG_W - 2 * this.PADDING;
    const dh = this.SVG_H - 2 * this.PADDING;
    return {
      x: this.PADDING + (dw - g.chamber_x_m * 1000 * sc) / 2,
      y: this.PADDING + (dh - g.chamber_y_m * 1000 * sc) / 2,
    };
  },
  simToSvg(sx, sy) {
    const g = this.geometry;
    if (!g) return { x: 0, y: 0 };
    const sc = this._scale();
    const off = this._offset();
    return {
      x: off.x + (sx + g.chamber_x_m / 2) * 1000 * sc,
      y: off.y + (g.chamber_y_m / 2 - sy) * 1000 * sc,
    };
  },
  svgToSim(vx, vy) {
    const g = this.geometry;
    if (!g) return { x: 0, y: 0 };
    const sc = this._scale();
    const off = this._offset();
    return {
      x: (vx - off.x) / (1000 * sc) - g.chamber_x_m / 2,
      y: g.chamber_y_m / 2 - (vy - off.y) / (1000 * sc),
    };
  },
  // Use SVG CTM so coordinates are exact regardless of how CSS scales the element
  getSvgPoint(evt) {
    const svg = byId("antCanvas");
    if (!svg) return { x: 0, y: 0 };
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX;
    pt.y = evt.clientY;
    const svgPt = pt.matrixTransform(svg.getScreenCTM().inverse());
    return { x: svgPt.x, y: svgPt.y };
  },

  // ── snap click to nearest point on any part surface edge ───────────────────
  _snapToPartSurface(sx, sy) {
    const g = this.geometry;
    if (!g || !g.parts || g.parts.length === 0) return { x: sx, y: sy };
    let bestDist = Infinity, bestX = sx, bestY = sy;
    for (const part of g.parts) {
      const poly = part.polygon_m || [];
      const n = poly.length;
      for (let i = 0; i < n; i++) {
        const [ax, ay] = poly[i];
        const [bx, by] = poly[(i + 1) % n];
        const edx = bx - ax, edy = by - ay;
        const len2 = edx * edx + edy * edy;
        if (len2 < 1e-20) continue;
        const t = Math.max(0, Math.min(1, ((sx - ax) * edx + (sy - ay) * edy) / len2));
        const px = ax + t * edx, py = ay + t * edy;
        const d = Math.hypot(px - sx, py - sy);
        if (d < bestDist) { bestDist = d; bestX = px; bestY = py; }
      }
    }
    return { x: bestX, y: bestY };
  },

  // ── open / close ────────────────────────────────────────────────────────────
  async open(formPayload) {
    this.formPayload = formPayload;
    this.antennas = [];
    this.selectedIdx = -1;
    this.drag = null;
    this.heatmapUrl = null;
    this.geometry = null;
    this.nextId = 0;
    this.toolMode = "select";

    this._dlg = byId("antennaWorkshop");
    if (!this._dlg) return;
    this._dlg.showModal();

    this._setStatus("Loading geometry\u2026");
    this._renderPlaceholder();
    try {
      const data = await fetchJson("/api/tools/antennae-geometry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formPayload),
      });
      if (!data?.ok) throw new Error(data?.error || "geometry failed");
      this.geometry = data;
      this.render();
      this.setToolMode("select");
    } catch (err) {
      this._setStatus(`Error loading geometry: ${err.message}`);
    }
  },

  close() {
    const dlg = this._dlg || byId("antennaWorkshop");
    if (dlg && dlg.open) dlg.close();
  },

  _setStatus(text) {
    const el = byId("antCanvasStatus");
    if (el) el.textContent = text;
  },

  setToolMode(mode) {
    this.toolMode = mode;
    const svg = byId("antCanvas");
    if (svg) svg.setAttribute("data-tool", mode);
    byId("antToolSelect")?.classList.toggle("ant-tool-active", mode === "select");
    byId("antToolAdd")?.classList.toggle("ant-tool-active", mode === "add");
    if (mode === "add") {
      this._setStatus("Add mode \u2022 click a part edge to place \u2022 right-click to delete");
    } else {
      this._setStatus("Select mode \u2022 click to select \u2022 drag to move \u2022 scroll to resize \u2022 Delete to remove");
    }
  },

  _renderPlaceholder() {
    const svg = byId("antCanvas");
    if (!svg) return;
    svg.setAttribute("width", this.SVG_W);
    svg.setAttribute("height", this.SVG_H);
    svg.innerHTML = `<rect width="${this.SVG_W}" height="${this.SVG_H}" fill="#0d1117"/>
      <text x="${this.SVG_W / 2}" y="${this.SVG_H / 2}" text-anchor="middle" fill="#4b5563" font-size="14">Loading\u2026</text>`;
  },

  // ── SVG renderer ─────────────────────────────────────────────────────────────
  render() {
    const svg = byId("antCanvas");
    if (!svg || !this.geometry) return;
    const g = this.geometry;
    const ns = "http://www.w3.org/2000/svg";
    const sc = this._scale();

    svg.setAttribute("width", this.SVG_W);
    svg.setAttribute("height", this.SVG_H);
    svg.setAttribute("viewBox", `0 0 ${this.SVG_W} ${this.SVG_H}`);
    svg.innerHTML = "";

    // Background
    const bg = document.createElementNS(ns, "rect");
    bg.setAttribute("width", this.SVG_W);
    bg.setAttribute("height", this.SVG_H);
    bg.setAttribute("fill", "#0d1117");
    svg.appendChild(bg);

    // Heatmap overlay (from Quick Search)
    if (this.heatmapUrl) {
      const tl = this.simToSvg(-g.chamber_x_m / 2, g.chamber_y_m / 2);
      const br = this.simToSvg(g.chamber_x_m / 2, -g.chamber_y_m / 2);
      const img = document.createElementNS(ns, "image");
      img.setAttributeNS("http://www.w3.org/1999/xlink", "href", this.heatmapUrl);
      img.setAttribute("x", tl.x);
      img.setAttribute("y", tl.y);
      img.setAttribute("width", br.x - tl.x);
      img.setAttribute("height", br.y - tl.y);
      img.setAttribute("preserveAspectRatio", "none");
      img.setAttribute("opacity", "0.75");
      svg.appendChild(img);
    }

    // Chamber border
    const tl = this.simToSvg(-g.chamber_x_m / 2, g.chamber_y_m / 2);
    const br = this.simToSvg(g.chamber_x_m / 2, -g.chamber_y_m / 2);
    const chamber = document.createElementNS(ns, "rect");
    chamber.setAttribute("x", tl.x);
    chamber.setAttribute("y", tl.y);
    chamber.setAttribute("width", br.x - tl.x);
    chamber.setAttribute("height", br.y - tl.y);
    chamber.setAttribute("fill", this.heatmapUrl ? "none" : "rgba(15,23,42,0.9)");
    chamber.setAttribute("stroke", "#334155");
    chamber.setAttribute("stroke-width", "1.5");
    svg.appendChild(chamber);

    // Electrode strips (top and bottom of chamber)
    const elecH = 5;
    for (const ey of [tl.y - elecH, br.y]) {
      const el = document.createElementNS(ns, "rect");
      el.setAttribute("x", tl.x);
      el.setAttribute("y", ey);
      el.setAttribute("width", br.x - tl.x);
      el.setAttribute("height", elecH);
      el.setAttribute("fill", "#f59e0b");
      el.setAttribute("opacity", "0.8");
      svg.appendChild(el);
    }

    // Part polygons
    for (const part of g.parts || []) {
      const pts = (part.polygon_m || []).map(([px, py]) => {
        const s = this.simToSvg(px, py);
        return `${s.x.toFixed(2)},${s.y.toFixed(2)}`;
      }).join(" ");
      const poly = document.createElementNS(ns, "polygon");
      poly.setAttribute("points", pts);
      poly.setAttribute("fill", "rgba(59,130,246,0.2)");
      poly.setAttribute("stroke", "#3b82f6");
      poly.setAttribute("stroke-width", "1.5");
      svg.appendChild(poly);
    }

    // Chamber size label
    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", tl.x + 4);
    label.setAttribute("y", br.y - 4);
    label.setAttribute("fill", "#475569");
    label.setAttribute("font-size", "9");
    label.textContent = `${(g.chamber_x_m * 1000).toFixed(0)}\u00d7${(g.chamber_y_m * 1000).toFixed(0)} mm`;
    svg.appendChild(label);

    // Antennas — rectangles with W\u00d7H label
    this.antennas.forEach((ant, idx) => {
      const pos = this.simToSvg(ant.center_x, ant.center_y);
      const wPx = Math.max(ant.size_x_mm * sc, 6);
      const hPx = Math.max(ant.size_y_mm * sc, 6);
      const sel = idx === this.selectedIdx;

      const g_el = document.createElementNS(ns, "g");
      g_el.setAttribute("data-ant-idx", idx);

      const rectEl = document.createElementNS(ns, "rect");
      rectEl.setAttribute("x", pos.x - wPx / 2);
      rectEl.setAttribute("y", pos.y - hPx / 2);
      rectEl.setAttribute("width", wPx);
      rectEl.setAttribute("height", hPx);
      rectEl.setAttribute("rx", "2");
      rectEl.setAttribute("fill", sel ? "rgba(34,197,94,0.45)" : "rgba(34,197,94,0.25)");
      rectEl.setAttribute("stroke", sel ? "#22c55e" : "#16a34a");
      rectEl.setAttribute("stroke-width", sel ? "2.5" : "1.5");
      g_el.appendChild(rectEl);

      // W\u00d7H size label below the rectangle
      const txt = document.createElementNS(ns, "text");
      txt.setAttribute("x", pos.x);
      txt.setAttribute("y", pos.y + hPx / 2 + 10);
      txt.setAttribute("text-anchor", "middle");
      txt.setAttribute("fill", sel ? "#86efac" : "#4ade80");
      txt.setAttribute("font-size", "9");
      txt.textContent = `${ant.size_x_mm.toFixed(1)}\u00d7${ant.size_y_mm.toFixed(1)}`;
      g_el.appendChild(txt);

      // Centre dot
      const dot = document.createElementNS(ns, "circle");
      dot.setAttribute("cx", pos.x);
      dot.setAttribute("cy", pos.y);
      dot.setAttribute("r", "2");
      dot.setAttribute("fill", sel ? "#22c55e" : "#16a34a");
      g_el.appendChild(dot);

      svg.appendChild(g_el);
    });

    this._updateInstanceList();
  },

  // ── canvas interaction ───────────────────────────────────────────────────────
  _hitTest(svgPt) {
    const sc = this._scale();
    const TOL = 5;
    for (let i = this.antennas.length - 1; i >= 0; i--) {
      const ant = this.antennas[i];
      const pos = this.simToSvg(ant.center_x, ant.center_y);
      const hw = Math.max(ant.size_x_mm * sc / 2, 4) + TOL;
      const hh = Math.max(ant.size_y_mm * sc / 2, 4) + TOL;
      if (Math.abs(svgPt.x - pos.x) <= hw && Math.abs(svgPt.y - pos.y) <= hh) return i;
    }
    return -1;
  },

  onMouseDown(evt) {
    if (!this.geometry) return;
    evt.preventDefault();
    const pt = this.getSvgPoint(evt);
    const hit = this._hitTest(pt);

    if (evt.button === 2) {
      // Right-click \u2192 delete in either mode
      if (hit >= 0) {
        this.antennas.splice(hit, 1);
        if (this.selectedIdx === hit) this.selectedIdx = -1;
        else if (this.selectedIdx > hit) this.selectedIdx--;
        this.render();
      }
      return;
    }

    if (hit >= 0) {
      // Hit existing antenna \u2192 select and arm drag (both modes)
      this.selectedIdx = hit;
      const ant = this.antennas[hit];
      this.drag = { idx: hit, svgStartX: pt.x, svgStartY: pt.y, simStartX: ant.center_x, simStartY: ant.center_y };
      this.render();
      this._syncSelectedSizeInput();
      return;
    }

    // Miss \u2192 only place in add mode
    if (this.toolMode !== "add") return;

    const rawSim = this.svgToSim(pt.x, pt.y);
    const g = this.geometry;
    if (Math.abs(rawSim.x) > g.chamber_x_m / 2 || Math.abs(rawSim.y) > g.chamber_y_m / 2) return;

    // Snap to nearest part-surface edge
    const snapped = this._snapToPartSurface(rawSim.x, rawSim.y);
    const sizeXMm = Math.max(Number(byId("antDefaultSizeXMm")?.value) || 1.0, 0.1);
    const sizeYMm = Math.max(Number(byId("antDefaultSizeYMm")?.value) || 1.0, 0.1);
    this.antennas.push({ id: this.nextId++, center_x: snapped.x, center_y: snapped.y, size_x_mm: sizeXMm, size_y_mm: sizeYMm });
    this.selectedIdx = this.antennas.length - 1;
    this.drag = null;
    this.render();
    this._syncSelectedSizeInput();
  },

  onMouseMove(evt) {
    if (!this.drag) return;
    evt.preventDefault();
    const pt = this.getSvgPoint(evt);
    const sc = this._scale();
    const dx = (pt.x - this.drag.svgStartX) / (1000 * sc);
    const dy = -(pt.y - this.drag.svgStartY) / (1000 * sc);
    const g = this.geometry;
    const halfX = g.chamber_x_m / 2;
    const halfY = g.chamber_y_m / 2;
    this.antennas[this.drag.idx].center_x = Math.max(-halfX, Math.min(halfX, this.drag.simStartX + dx));
    this.antennas[this.drag.idx].center_y = Math.max(-halfY, Math.min(halfY, this.drag.simStartY + dy));
    this.render();
  },

  onMouseUp() {
    this.drag = null;
  },

  onContextMenu(evt) {
    evt.preventDefault();
  },

  onWheel(evt) {
    if (!this.geometry) return;
    const pt = this.getSvgPoint(evt);
    const hit = this._hitTest(pt);
    if (hit < 0) return;
    evt.preventDefault();
    const ant = this.antennas[hit];
    const factor = evt.deltaY < 0 ? 1.1 : 0.9;
    ant.size_x_mm = Math.max(0.1, Math.round(ant.size_x_mm * factor * 10) / 10);
    ant.size_y_mm = Math.max(0.1, Math.round(ant.size_y_mm * factor * 10) / 10);
    if (hit === this.selectedIdx) this._syncSelectedSizeInput();
    this.render();
  },

  // ── quick search ─────────────────────────────────────────────────────────────
  async runQuickSearch() {
    const btn = byId("antQuickSearchBtn");
    const status = byId("antSearchStatus");
    if (btn) btn.disabled = true;
    if (status) { status.textContent = "Running EQS\u2026 this takes a few seconds."; status.classList.remove("hidden"); }
    this._setStatus("Running Quick Search EQS pass\u2026");
    try {
      const payload = Object.assign({}, this.formPayload);
      payload.antennae_enabled = true;
      const data = await fetchJson("/api/tools/antennae-quick-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!data?.ok) throw new Error(data?.error || "search failed");
      this.heatmapUrl = data.heatmap_data_url || null;
      const rows = Array.isArray(data.instances) ? data.instances : [];
      this.antennas = rows.map((r) => {
        const szX = Number(r.size_x_mm || r.size_mm) || 1.0;
        const szY = Number(r.size_y_mm || r.size_mm) || 1.0;
        return { id: this.nextId++, center_x: Number(r.center_x), center_y: Number(r.center_y), size_x_mm: szX, size_y_mm: szY };
      });
      this.selectedIdx = -1;
      this.render();
      if (status) status.textContent = `Quick Search found ${this.antennas.length} candidate position${this.antennas.length !== 1 ? "s" : ""}. Switch to Select mode to adjust.`;
      this.setToolMode("select");
    } catch (err) {
      if (status) status.textContent = `Search failed: ${err.message}`;
      this._setStatus("Quick Search failed.");
    } finally {
      if (btn) btn.disabled = false;
    }
  },

  // ── instance list ────────────────────────────────────────────────────────────
  _updateInstanceList() {
    const list = byId("antInstanceList");
    const count = byId("antCountLabel");
    if (count) count.textContent = this.antennas.length;
    if (!list) return;
    if (this.antennas.length === 0) {
      list.innerHTML = '<p class="ant-instance-empty">No antennas placed yet.<br>Switch to Add mode and click a part edge.</p>';
      return;
    }
    list.innerHTML = this.antennas.map((ant, idx) => {
      const xmm = (ant.center_x * 1000).toFixed(1);
      const ymm = (ant.center_y * 1000).toFixed(1);
      const sel = idx === this.selectedIdx;
      return `<div class="ant-instance-item${sel ? " ant-instance-selected" : ""}" data-ant-idx="${idx}">
        <span class="ant-instance-info">Ant ${idx + 1}: (${xmm}, ${ymm}) mm \u2022 ${ant.size_x_mm.toFixed(1)}\u00d7${ant.size_y_mm.toFixed(1)} mm</span>
        <button class="ant-instance-del" data-del-idx="${idx}" title="Remove">&#x2715;</button>
      </div>`;
    }).join("");

    list.querySelectorAll(".ant-instance-item").forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.classList.contains("ant-instance-del")) return;
        const idx = Number(el.dataset.antIdx);
        this.selectedIdx = idx;
        this.render();
        this._syncSelectedSizeInput();
      });
    });
    list.querySelectorAll(".ant-instance-del").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.dataset.delIdx);
        this.antennas.splice(idx, 1);
        if (this.selectedIdx === idx) this.selectedIdx = -1;
        else if (this.selectedIdx > idx) this.selectedIdx--;
        this.render();
      });
    });

    const selRow = byId("antSelectedSizeRow");
    if (selRow) selRow.classList.toggle("hidden", this.selectedIdx < 0);
  },

  _syncSelectedSizeInput() {
    const selRow = byId("antSelectedSizeRow");
    if (!selRow) return;
    if (this.selectedIdx >= 0 && this.selectedIdx < this.antennas.length) {
      selRow.classList.remove("hidden");
      const ant = this.antennas[this.selectedIdx];
      const xIn = byId("antSelectedSizeXMm");
      const yIn = byId("antSelectedSizeYMm");
      if (xIn) xIn.value = ant.size_x_mm.toFixed(1);
      if (yIn) yIn.value = ant.size_y_mm.toFixed(1);
    } else {
      selRow.classList.add("hidden");
    }
  },

  // ── launch ───────────────────────────────────────────────────────────────────
  async _doLaunch(sweepMode) {
    const payload = Object.assign({}, this.formPayload);
    payload.antennae_enabled = true;
    payload.antennae_explicit_instances = this.antennas.map((ant) => ({
      center_x: ant.center_x,
      center_y: ant.center_y,
      size_x_mm: ant.size_x_mm,
      size_y_mm: ant.size_y_mm,
      size_mm: (ant.size_x_mm + ant.size_y_mm) / 2,
      anchor_x: ant.center_x,
      anchor_y: ant.center_y,
      part_id: 1,
    }));

    if (sweepMode) {
      payload.mode = "antennae_size_sweep";
      payload.antennae_sweep_min_mm = Number(byId("antSweepMinMm")?.value || 0.5);
      payload.antennae_sweep_max_mm = Number(byId("antSweepMaxMm")?.value || 2.0);
      payload.antennae_sweep_steps = Number(byId("antSweepSteps")?.value || 4);
    }

    try {
      await fetchJson("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      this.savedSession = { antennas: this.antennas.map(a => Object.assign({}, a)), geometry: this.geometry };
      this._showReopenBtn(true);
      this.close();
      await loadJobs();
    } catch (err) {
      this._setStatus(`Launch failed: ${err.message}`);
    }
  },

  // ── session persistence ──────────────────────────────────────────────────────
  reopen() {
    if (!this.savedSession) return;
    const { antennas, geometry } = this.savedSession;
    // Restore state
    this.geometry = geometry;
    this.antennas = antennas.map(a => Object.assign({}, a));
    this.selectedIdx = -1;
    this.drag = null;
    this.heatmapUrl = null;
    this._dlg = byId("antennaWorkshop");
    if (!this._dlg) return;
    this._dlg.showModal();
    this.render();
    this.setToolMode("select");
  },

  _showReopenBtn(visible) {
    const btn = byId("antReopenBtn");
    if (btn) btn.classList.toggle("hidden", !visible);
  },

  // ── wiring (called from init) ────────────────────────────────────────────────
  init() {
    const svg = byId("antCanvas");
    if (svg) {
      svg.addEventListener("mousedown", (e) => this.onMouseDown(e));
      svg.addEventListener("contextmenu", (e) => this.onContextMenu(e));
      svg.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
      window.addEventListener("mousemove", (e) => { if (this.drag) this.onMouseMove(e); });
      window.addEventListener("mouseup", () => this.onMouseUp());
    }

    // Tool mode buttons
    byId("antToolSelect")?.addEventListener("click", () => this.setToolMode("select"));
    byId("antToolAdd")?.addEventListener("click", () => this.setToolMode("add"));

    // Keyboard shortcuts: S = select, A = add, Delete/Backspace = remove selected
    byId("antennaWorkshop")?.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === "s" || e.key === "S") { e.preventDefault(); this.setToolMode("select"); }
      if (e.key === "a" || e.key === "A") { e.preventDefault(); this.setToolMode("add"); }
      if ((e.key === "Delete" || e.key === "Backspace") && this.selectedIdx >= 0) {
        e.preventDefault();
        this.antennas.splice(this.selectedIdx, 1);
        this.selectedIdx = -1;
        this.render();
      }
    });

    byId("antWorkshopCloseBtn")?.addEventListener("click", () => this.close());
    byId("antCancelBtn")?.addEventListener("click", () => this.close());
    byId("antReopenBtn")?.addEventListener("click", () => this.reopen());
    byId("antQuickSearchBtn")?.addEventListener("click", () => this.runQuickSearch());

    byId("antClearAllBtn")?.addEventListener("click", () => {
      this.antennas = [];
      this.selectedIdx = -1;
      this.heatmapUrl = null;
      this.render();
    });

    // Default size inputs \u2014 immediately apply to any selected antenna
    byId("antDefaultSizeXMm")?.addEventListener("input", () => {
      if (this.selectedIdx >= 0 && this.selectedIdx < this.antennas.length) {
        const v = Math.max(Number(byId("antDefaultSizeXMm")?.value) || 0, 0.1);
        this.antennas[this.selectedIdx].size_x_mm = v;
        const xIn = byId("antSelectedSizeXMm");
        if (xIn) xIn.value = v.toFixed(1);
        this.render();
      }
    });
    byId("antDefaultSizeYMm")?.addEventListener("input", () => {
      if (this.selectedIdx >= 0 && this.selectedIdx < this.antennas.length) {
        const v = Math.max(Number(byId("antDefaultSizeYMm")?.value) || 0, 0.1);
        this.antennas[this.selectedIdx].size_y_mm = v;
        const yIn = byId("antSelectedSizeYMm");
        if (yIn) yIn.value = v.toFixed(1);
        this.render();
      }
    });

    // Selected-antenna size inputs
    byId("antSelectedSizeXMm")?.addEventListener("input", () => {
      if (this.selectedIdx >= 0 && this.selectedIdx < this.antennas.length) {
        const v = Math.max(Number(byId("antSelectedSizeXMm")?.value) || 0, 0.1);
        this.antennas[this.selectedIdx].size_x_mm = v;
        this.render();
      }
    });
    byId("antSelectedSizeYMm")?.addEventListener("input", () => {
      if (this.selectedIdx >= 0 && this.selectedIdx < this.antennas.length) {
        const v = Math.max(Number(byId("antSelectedSizeYMm")?.value) || 0, 0.1);
        this.antennas[this.selectedIdx].size_y_mm = v;
        this.render();
      }
    });

    // Launch mode radio buttons
    document.querySelectorAll('input[name="antLaunchMode"]').forEach((radio) => {
      radio.addEventListener("change", () => {
        const isSweep = byId("antModeSweep")?.checked;
        byId("antSweepControls")?.classList.toggle("hidden", !isSweep);
        byId("antLaunchRunBtn")?.classList.toggle("hidden", isSweep);
        byId("antLaunchSweepBtn")?.classList.toggle("hidden", !isSweep);
      });
    });

    byId("antLaunchRunBtn")?.addEventListener("click", () => this._doLaunch(false));
    byId("antLaunchSweepBtn")?.addEventListener("click", () => this._doLaunch(true));

    // Close on backdrop click
    byId("antennaWorkshop")?.addEventListener("click", (e) => {
      if (e.target === byId("antennaWorkshop")) this.close();
    });
  },
};

init();
