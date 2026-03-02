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

function parseNumberList(raw) {
  return String(raw || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number)
    .filter((n) => Number.isFinite(n));
}

function modeTag(mode) {
  const m = String(mode || "").toLowerCase();
  if (m === "single") return "single";
  if (m === "sweep") return "sweep";
  if (m === "optimizer") return "opt";
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

function buildPayload(includeOutput = true) {
  const mode = byId("mode")?.value;
  const payload = {
    mode,
    shape: byId("shape")?.value,
    advanced: buildAdvanced(),
  };
  const forcedBase = byId("baseConfigSelect")?.value || "";
  if (forcedBase) payload.base_config = forcedBase;

  if (includeOutput) payload.output_name = byId("outputName")?.value?.trim();

  if (["single", "optimizer", "turntable"].includes(mode)) {
    payload.exposure_minutes = Number(byId("exposureMinutes")?.value);
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

  return payload;
}

function setModeSections() {
  const mode = byId("mode")?.value;
  document.querySelectorAll(".mode-section").forEach((el) => {
    const modes = el.dataset.mode.split(/\s+/);
    el.classList.toggle("hidden", !modes.includes(mode));
  });
}

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

function renderMeta(meta) {
  const shape = byId("shape");
  if (!shape) return;
  shape.innerHTML = "";
  meta.shape_options.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
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
    const matched = document.createElement("option");
    matched.value = "__matched__";
    matched.textContent = "Auto matched config";
    yamlSel.appendChild(matched);
    (meta.base_configs || []).forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      yamlSel.appendChild(opt);
    });
  }
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
    "optimizer:", "turntable:", "enabled:",
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

async function refreshYamlPreview() {
  const pre = byId("yamlPreview");
  if (!pre) return;

  let selected = byId("yamlPreviewSelect")?.value || "__matched__";
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

  jobs.slice().reverse().forEach((j) => {
    const el = document.createElement("div");
    el.className = "job";
    const statusCls = j.status;
    const cfg = j.config_resolution?.[0]?.resolved;
    const cfgLine = cfg?.config_name ? `${cfg.match_type}: ${cfg.config_name}` : "";
    const outputs = (j.output_dirs || []).join(", ");
    el.innerHTML = `
      <div class="job-head">
        <strong>${j.id}</strong>
        <span class="badge ${statusCls}">${j.status}</span>
      </div>
      <div>${j.mode} • ${j.output_name || "(batch)"}</div>
      <div>${j.started_at || ""}</div>
      <div class="muted">${cfgLine}</div>
      <div class="muted">${outputs || "No outputs yet"}</div>
      <div><a href="${j.log_url}" target="_blank">log</a></div>
    `;
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

  byId("shape")?.addEventListener("change", refreshOutputNamePrefix);
  byId("turntableRotationDeg")?.addEventListener("input", refreshTurntableInfo);
  byId("turntableRotationDeg")?.addEventListener("change", refreshTurntableInfo);
  byId("turntableTotalRotations")?.addEventListener("input", refreshTurntableInfo);
  byId("turntableTotalRotations")?.addEventListener("change", refreshTurntableInfo);
  byId("outputName")?.addEventListener("input", () => {
    state.outputNameTouched = true;
  });

  [
    "shape", "exposureMinutes", "sweepMinutes", "phiSnapshots", "tempCeiling", "highlightPhi",
    "turntableRotationDeg", "turntableTotalRotations", "turntableIntervalS",
    "advGridNx", "advGridNy", "advFreqHz", "advVoltage", "advPower", "advEnforceGen", "advEff",
    "advDepth", "advDt", "advAmbient", "advConv", "advSigma", "advDopedSigmaProfile",
    "advVirginSigma", "advVirginEps", "advPowderRho", "advPowderK", "advPowderCp",
    "advDopedEps", "advSigmaTempCoeff", "advSigmaDensityCoeff", "advSigmaRefTemp",
    "advDopedRhoSolid", "advDopedRhoLiquid", "advDopedKSolid", "advDopedKLiquid",
    "advDopedCpSolid", "advDopedCpLiquid",
    "advMaxQrf", "advMaxTemp", "baseConfigSelect"
  ].forEach((id) => {
    const el = byId(id);
    if (el) {
      el.addEventListener("change", refreshMatch);
      el.addEventListener("input", refreshMatch);
    }
  });

  byId("yamlPreviewSelect")?.addEventListener("change", refreshYamlPreview);

  setModeSections();
  await loadMeta();
  refreshOutputNamePrefix();
  refreshTurntableInfo();
  await Promise.all([loadJobs(), refreshMatch()]);

  setInterval(async () => {
    try {
      await loadJobs();
    } catch {
      const s = byId("serverStatus");
      if (s) s.textContent = "API disconnected";
    }
  }, 3000);
}

init();
