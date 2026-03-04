const state = {
  meta: null,
  jobs: [],
  matchedConfigName: "",
  autoOutputPrefix: "",
  outputNameTouched: false,
};
const byId = (id) => document.getElementById(id);

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
  spacex_logo: 20.0,
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
  antennaeEnabledMode: "Antennae assist toggle. Inherit keeps base config value; enabled/disabled explicitly writes antennae.enabled for this run.",
  antennaeSizeMode: "Antennae size strategy. Global uses one size; auto maps local Qrf deficit to a size range.",
  antennaeGlobalSizeMm: "Global antenna size in millimeters when size mode is global.",
  antennaeAutoMinMm: "Minimum antenna size for auto mode.",
  antennaeAutoMaxMm: "Maximum antenna size for auto mode.",
  antennaeMaxPerPart: "Maximum antennae placed per part from underheated boundary regions.",
  antennaeMinSpacingMm: "Minimum spacing between selected antenna anchors.",
  antennaeEdgeMarginMm: "Minimum chamber-edge margin for antenna placement.",
  antennaeAutoQrfPercentileLow: "Lower Qrf percentile used for auto-size deficit normalization.",
  antennaeAutoQrfPercentileHigh: "Upper Qrf percentile used for auto-size deficit normalization.",
  antennaeCalibrationSizesMm: "Comma-separated global antenna sizes (mm) used during calibration runs.",
  antennaeCalibrationTopK: "Number of proxy-ranked antennae candidates promoted to full coupled validation.",
  antennaeCalibrationIncludeAuto: "If true, includes one auto-size candidate alongside the global-size ladder.",
  antennaeCalibrationUseTurntable: "If true, validates calibration candidates with turntable enabled (slower).",
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

  const antEnabledMode = String(byId("antennaeEnabledMode")?.value || "inherit").toLowerCase();
  const antPanelEnabled = antEnabledMode !== "false";
  [
    "antennaeSizeMode", "antennaeMaxPerPart", "antennaeMinSpacingMm", "antennaeEdgeMarginMm",
    "antennaePreviewBtn", "antennaeCalibrationSizesMm", "antennaeCalibrationTopK",
    "antennaeCalibrationIncludeAuto", "antennaeCalibrationUseTurntable", "antennaeCalibrateBtn",
  ].forEach((id) => _setVisibleById(id, antPanelEnabled));
  const antSizeMode = String(byId("antennaeSizeMode")?.value || "global").toLowerCase();
  _setVisibleById("antennaeGlobalSizeMm", antPanelEnabled && antSizeMode !== "auto");
  ["antennaeAutoMinMm", "antennaeAutoMaxMm", "antennaeAutoQrfPercentileLow", "antennaeAutoQrfPercentileHigh"]
    .forEach((id) => _setVisibleById(id, antPanelEnabled && antSizeMode === "auto"));

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
    payload.orientation_angle_min_deg = Number(byId("orientationAngleMinDeg")?.value);
    payload.orientation_angle_max_deg = Number(byId("orientationAngleMaxDeg")?.value);
    payload.orientation_angle_step_deg = Number(byId("orientationAngleStepDeg")?.value);
    payload.orientation_refine_window_deg = Number(byId("orientationRefineWindowDeg")?.value);
    payload.orientation_refine_step_deg = Number(byId("orientationRefineStepDeg")?.value);
    payload.orientation_exposure_min_s = Number(byId("orientationExposureMinS")?.value);
    payload.orientation_exposure_max_s = Number(byId("orientationExposureMaxS")?.value);
    payload.orientation_exposure_step_s = Number(byId("orientationExposureStepS")?.value);
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

  const antennaeMode = String(byId("antennaeEnabledMode")?.value || "inherit").trim().toLowerCase();
  if (antennaeMode === "true") payload.antennae_enabled = true;
  else if (antennaeMode === "false") payload.antennae_enabled = false;
  payload.antennae_size_mode = String(byId("antennaeSizeMode")?.value || "global").trim().toLowerCase();
  payload.antennae_global_size_mm = Number(byId("antennaeGlobalSizeMm")?.value);
  payload.antennae_auto_min_mm = Number(byId("antennaeAutoMinMm")?.value);
  payload.antennae_auto_max_mm = Number(byId("antennaeAutoMaxMm")?.value);
  payload.antennae_max_per_part = Number(byId("antennaeMaxPerPart")?.value);
  payload.antennae_min_spacing_mm = Number(byId("antennaeMinSpacingMm")?.value);
  payload.antennae_edge_margin_mm = Number(byId("antennaeEdgeMarginMm")?.value);
  payload.antennae_auto_qrf_percentile_low = Number(byId("antennaeAutoQrfPercentileLow")?.value);
  payload.antennae_auto_qrf_percentile_high = Number(byId("antennaeAutoQrfPercentileHigh")?.value);

  payload.shell_enabled = (String(byId("shellEnabled")?.value || "false").toLowerCase() === "true");
  payload.shell_wall_thickness_mm = Number(byId("shellWallThicknessMm")?.value);
  payload.shell_method = String(byId("shellMethod")?.value || "offset_inward");

  return payload;
}

function setModeSections() {
  const mode = byId("mode")?.value;
  document.querySelectorAll(".mode-section").forEach((el) => {
    const modes = el.dataset.mode.split(/\s+/);
    el.classList.toggle("hidden", !modes.includes(mode));
  });
  updateDependentVisibility();
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
    spacex_logo: "SPX",
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
  else if (s === "spacex_logo") {
    inner = `<text x='50' y='58' text-anchor='middle' font-size='20' font-weight='700' fill='${stroke}' font-family='Avenir Next, Inter, sans-serif'>SPX</text>`;
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

function renderJobs(jobs) {
  const root = byId("jobs");
  if (!root) return;
  root.innerHTML = "";
  if (!jobs.length) {
    root.textContent = "No jobs yet.";
    return;
  }

  jobs.forEach((j) => {
    const el = document.createElement("div");
    el.className = "job";
    const statusCls = j.status;
    const cfg = j.config_resolution?.[0]?.resolved;
    const cfgLine = cfg?.config_name ? `${cfg.match_type}: ${cfg.config_name}` : "";
    const outputs = (j.output_dirs || []).join(", ");
    const totalRuns = Number.isFinite(Number(j.total_runs)) ? Math.max(1, Number(j.total_runs)) : 1;
    const doneRuns = Number.isFinite(Number(j.completed_runs)) ? Math.max(0, Number(j.completed_runs)) : 0;
    const rawPct = Number.isFinite(Number(j.progress_pct)) ? Number(j.progress_pct) : (j.status === "completed" ? 100 : 0);
    const pct = Math.max(0, Math.min(100, rawPct));
    const progressLabel = String(j.progress_label || (j.status === "completed" ? "Completed" : "Running"));
    const createdMs = parseIsoMs(j.created_at);
    const startedMs = parseIsoMs(j.started_at);
    const endedMs = parseIsoMs(j.ended_at);
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
    el.innerHTML = `
      <div class="job-head">
        <strong>${j.id}</strong>
        <span class="badge ${statusCls}">${j.status}</span>
      </div>
      <div class="job-progress-wrap">
        <div class="job-progress-label">${progressLabel}</div>
        <div class="job-progress-meta">${doneRuns}/${totalRuns} • ${pct.toFixed(0)}%</div>
      </div>
      <div class="job-progress-track">
        <div class="job-progress-fill ${statusCls}" style="width:${pct.toFixed(1)}%"></div>
      </div>
      <div>${j.mode} • ${j.output_name || "(batch)"}</div>
      <div>${j.started_at || j.created_at || ""}</div>
      <div class="muted">${timerLabel}</div>
      <div class="muted">${cfgLine}</div>
      ${queueControls}
      ${controlButtons}
      <div class="muted">${outputs || "No outputs yet"}</div>
      <div><a href="${j.log_url}" target="_blank">log</a></div>
    `;
    el.querySelectorAll(".queue-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.getAttribute("data-jobid");
        const dir = btn.getAttribute("data-dir");
        if (!id || !dir) return;
        try {
          await reorderQueuedJob(id, dir);
        } catch (err) {
          console.error(err);
        }
      });
    });
    el.querySelectorAll(".job-ctl-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.getAttribute("data-jobid");
        const action = btn.getAttribute("data-action");
        if (!id || !action) return;
        try {
          await controlJob(id, action);
        } catch (err) {
          console.error(err);
        }
      });
    });
    root.appendChild(el);
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

async function loadJobs() {
  state.jobs = await fetchJson("/api/jobs");
  renderJobs(state.jobs);
  renderLiveArtifacts(state.jobs);
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
    renderMatchInfo(data);
    await refreshYamlPreview();
  } catch (err) {
    box.textContent = `Match error: ${err.message}`;
  }
}

async function previewAntennae() {
  const status = byId("serverStatus");
  const img = byId("antennaePreviewImg");
  const txt = byId("antennaePreviewText");
  if (!img || !txt) return;
  try {
    if (status) status.textContent = "Generating antennae preview...";
    const payload = buildPayload(false);
    payload.antennae_preview_request = true;
    const data = await fetchJson("/api/tools/antennae-preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!data?.ok) throw new Error(data?.error || "preview failed");
    img.src = data.image_url || "";
    img.classList.toggle("hidden", !data.image_url);
    const rows = Array.isArray(data.instances) ? data.instances : [];
    const preview = {
      n_instances: Number(data.n_instances || rows.length),
      size_mode: data.size_mode || "",
      part_counts: data.part_counts || {},
      first_instances: rows.slice(0, 8),
    };
    txt.textContent = JSON.stringify(preview, null, 2);
    txt.classList.remove("hidden");
    if (status) status.textContent = "Antennae preview ready";
  } catch (err) {
    txt.textContent = `Preview error: ${err.message}`;
    txt.classList.remove("hidden");
    if (status) status.textContent = "Antennae preview failed";
  }
}

async function queueAntennaeCalibration() {
  const status = byId("serverStatus");
  try {
    if (status) status.textContent = "Queueing antennae calibration...";
    const payload = buildPayload(false);
    const baseName = String(byId("outputName")?.value || "").trim() || "antennae_calibration";
    payload.output_name = `${baseName}_antcal`;
    payload.antennae_enabled = true;
    payload.antennae_calibration_sizes_mm = parseNumberList(byId("antennaeCalibrationSizesMm")?.value);
    payload.antennae_calibration_top_k = Number(byId("antennaeCalibrationTopK")?.value);
    payload.antennae_calibration_include_auto = String(byId("antennaeCalibrationIncludeAuto")?.value || "true").toLowerCase() === "true";
    payload.antennae_calibration_use_turntable = String(byId("antennaeCalibrationUseTurntable")?.value || "false").toLowerCase() === "true";
    await fetchJson("/api/tools/antennae-calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await loadJobs();
    if (status) status.textContent = "Antennae calibration queued";
  } catch (err) {
    if (status) status.textContent = `Antennae calibration error: ${err.message}`;
  }
}

async function launchRun(ev) {
  ev.preventDefault();
  const payload = buildPayload(true);
  await fetchJson("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await loadJobs();
}

async function init() {
  const form = byId("runForm");
  if (!form) return;

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
  byId("geometryLockAspect")?.addEventListener("change", refreshMatch);
  byId("physicsModelFamily")?.addEventListener("change", () => {
    syncExperimentalControls();
    updateDependentVisibility();
    refreshMatch();
  });
  byId("physicsExperimentalEnabled")?.addEventListener("change", updateDependentVisibility);
  byId("placementUseTurntable")?.addEventListener("change", updateDependentVisibility);
  byId("shellEnabled")?.addEventListener("change", updateDependentVisibility);
  byId("antennaeEnabledMode")?.addEventListener("change", updateDependentVisibility);
  byId("antennaeSizeMode")?.addEventListener("change", updateDependentVisibility);
  byId("expCrEnabled")?.addEventListener("change", updateDependentVisibility);
  byId("expViscosityModel")?.addEventListener("change", updateDependentVisibility);
  byId("antennaePreviewBtn")?.addEventListener("click", previewAntennae);
  byId("antennaeCalibrateBtn")?.addEventListener("click", queueAntennaeCalibration);
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
    "antennaeEnabledMode", "antennaeSizeMode", "antennaeGlobalSizeMm", "antennaeAutoMinMm", "antennaeAutoMaxMm",
    "antennaeMaxPerPart", "antennaeMinSpacingMm", "antennaeEdgeMarginMm",
    "antennaeAutoQrfPercentileLow", "antennaeAutoQrfPercentileHigh",
    "antennaeCalibrationSizesMm", "antennaeCalibrationTopK", "antennaeCalibrationIncludeAuto", "antennaeCalibrationUseTurntable",
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

  setModeSections();
  attachParamInfoIcons();
  await loadMeta();
  refreshGeometrySizeUI();
  syncExperimentalControls();
  updateDependentVisibility();
  applyFrequencyProfile();
  refreshOutputNamePrefix();
  refreshTurntableInfo();
  await Promise.all([loadJobs(), refreshMatch()]);

  setInterval(async () => {
    try {
      await loadJobs();
    } catch {
      const s = byId("serverStatus");
      if (s) s.textContent = "HEATR service unreachable";
    }
  }, 3000);

  setInterval(() => {
    try {
      renderJobs(state.jobs || []);
    } catch {
      // no-op: timer refresh is best-effort.
    }
  }, 1000);
}

init();
