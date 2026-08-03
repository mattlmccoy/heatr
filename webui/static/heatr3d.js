// HEATR-3D Workbench - library launch, densification study viewer, solved maps,
// run comparison. Backend: /api/heatr3d/wb/* (heatr3d_workbench/server_module.py)
// plus the preserved legacy /api/heatr3d/* endpoints (parity Appendix A).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const $ = (id) => document.getElementById(id);
const FILES = (id, rel) => `/files/outputs_eqs/_heatr3d/${encodeURIComponent(id)}/${rel}`;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "not computed"
  : (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(d)) : String(v));

const CMAP_CLASS = { T_phi90: "cb-inferno", rho_final: "cb-density", sat: "cb-viridis", Qrf: "cb-viridis", phi_final: "cb-viridis" };
const FIELD_UNITS = { sat: "dopant fraction", T_phi90: "deg C", phi_final: "melt fraction", rho_final: "rel. density", Qrf: "W/m^3" };
const BADGE = (b) => b ? `<span class="wb-badge" title="${esc(b.detail)} Evidence: ${esc(b.evidence)}">${esc(b.label)}</span>` : "";

const state = {
  screen: "library",
  library: [], selectedShape: null, intake: null,
  runs: [], queue: [], currentRun: null, detail: null,
  studyPollTimer: null,
};

// ── screen switching ────────────────────────────────────────────────────────
function showScreen(name) {
  state.screen = name;
  document.querySelectorAll(".wb-screen").forEach((s) => s.classList.remove("active"));
  document.querySelectorAll(".wb-nav button").forEach((b) => b.classList.toggle("active", b.dataset.screen === name));
  $(`screen-${name}`).classList.add("active");
  if (name === "solved") loadSolved();
  if (name === "compare") fillComparePickers();
}
document.querySelectorAll(".wb-nav button").forEach((b) =>
  b.addEventListener("click", () => showScreen(b.dataset.screen)));

// ── LIBRARY screen ──────────────────────────────────────────────────────────
const STALE_SERVER_MSG =
  "The GUI server process predates the workbench (endpoint 404). " +
  "Restart the HEATR server (python3 rfam_gui_server.py) to load the workbench module.";

async function wbFetch(url) {
  const r = await fetch(url);
  if (r.status === 404) throw new Error(STALE_SERVER_MSG);
  if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
  return r.json();
}

async function loadLibrary() {
  const g = $("libGallery");
  try {
    const r = await wbFetch("/api/heatr3d/wb/library");
    state.library = r.shapes || [];
  } catch (e) {
    state.library = [];
    g.innerHTML = `<span class="wb-error">Shape library unavailable: ${esc(e.message)}</span>`;
    return;
  }
  if (!state.library.length) {
    g.innerHTML = `<span class="wb-error">Shape library empty: shape_library_3d/meta has no shape JSON files on this checkout.</span>`;
    return;
  }
  g.innerHTML = "";
  for (const s of state.library) {
    const el = document.createElement("div");
    el.className = "wb-shape" + (s.loadable ? "" : " rejected");
    el.innerHTML =
      `<div class="nm">${esc(s.name)} <span class="tier-chip tier-${s.tier}">Tier ${s.tier}${s.loadable ? "" : " reject"}</span></div>` +
      `<div class="rf">${esc(s.rf_characteristic)}</div>` +
      `<div class="wb-note">${s.volume_mm3 ? s.volume_mm3.toFixed(0) + " mm^3" : ""} ${s.role ? "&middot; " + esc(s.role) : ""}</div>` +
      (s.loadable ? "" : `<div class="wb-note" style="color:#f0a0a0;">rejection fixture: intake must refuse this shape</div>`);
    if (s.loadable) {
      el.addEventListener("click", () => {
        state.selectedShape = s.name;
        document.querySelectorAll(".wb-shape").forEach((x) => x.classList.remove("selected"));
        el.classList.add("selected");
        $("srcSel").value = "library";
        srcChanged();
        showLibPreview(s.name);
      });
    }
    g.appendChild(el);
  }
}

// ── STL preview (item 1: exact mesh, never a voxelization) ──────────────────
let libViewer = null;
const stlLoader = new STLLoader();

async function fetchStlGeometry(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`STL fetch failed (HTTP ${r.status})`);
  const geo = stlLoader.parse(await r.arrayBuffer());
  geo.computeVertexNormals();
  geo.center();
  return geo;
}

function makeMiniViewer(el) {
  const w = el.clientWidth || 300, h = el.clientHeight || 240;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(w, h);
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x11151d);
  const camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 2000);
  camera.position.set(28, 20, 34);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const d1 = new THREE.DirectionalLight(0xffffff, 1.0); d1.position.set(40, 60, 40); scene.add(d1);
  const d2 = new THREE.DirectionalLight(0x88aaff, 0.35); d2.position.set(-30, -20, -30); scene.add(d2);
  const st = { renderer, scene, camera, controls, mesh: null };
  (function anim() { requestAnimationFrame(anim); controls.update(); renderer.render(scene, camera); })();
  st.show = (geo) => {
    if (st.mesh) { scene.remove(st.mesh); st.mesh.geometry.dispose(); st.mesh.material.dispose(); }
    const mat = new THREE.MeshPhongMaterial({ color: 0xe8a25c, shininess: 28,
      specular: 0x333333, side: THREE.DoubleSide });
    st.mesh = new THREE.Mesh(geo, mat);
    scene.add(st.mesh);
    geo.computeBoundingSphere();
    const r = geo.boundingSphere.radius;
    camera.position.setLength(r * 2.7);
    controls.update();
  };
  return st;
}

async function showLibPreview(name) {
  const box = $("libPreviewBox"), lab = $("libPreviewLabel");
  if (!box) return;
  box.style.display = "";
  lab.textContent = `loading ${name}.stl...`;
  try {
    const geo = await fetchStlGeometry(`/api/heatr3d/wb/stl?shape=${encodeURIComponent(name)}`);
    if (!libViewer) libViewer = makeMiniViewer($("libPreview"));
    libViewer.show(geo);
    lab.textContent = `${name} - exact STL geometry (drag to orbit, scroll to zoom)`;
  } catch (e) { lab.textContent = `preview failed: ${e.message}`; }
}

function srcChanged() {
  const src = $("srcSel").value;
  $("paramBlock").style.display = src === "parametric" ? "" : "none";
  $("stlBlock").style.display = src === "stl" ? "" : "none";
  $("libPickNote").style.display = src === "library" ? "" : "none";
  $("libPickNote").textContent = state.selectedShape
    ? `Selected library shape: ${state.selectedShape}` : "Pick a shape from the gallery.";
  updateCost();
}
$("srcSel").addEventListener("change", srcChanged);

// Cost estimate from MEASURED anchors (spec 4.1): circle melt-onset 147 s at
// n=64, 558 s at n=96; n=48 densify smoke runs 544-773 s. Labeled an estimate.
function updateCost() {
  const n = parseInt($("gridN").value, 10);
  const densify = $("densify").checked;
  const anchors = { 32: 25, 48: 60, 64: 147, 96: 558 };   // melt-onset seconds
  let sec = anchors[n] || 147;
  let note = "melt-onset march (stops at phi = 0.90)";
  if (densify) { sec = Math.max(sec * 4, n <= 48 ? 650 : sec * 4); note = "densify marches the FULL exposure"; }
  if ($("fgmMode").value !== "none") { sec *= 2; note += "; FGM adds a probe run"; }
  $("costEstimate").textContent =
    `Estimated wall time ~${sec >= 120 ? (sec / 60).toFixed(0) + " min" : sec.toFixed(0) + " s"} (${note}). ` +
    `Measured anchors, not a promise.`;
}
["gridN", "densify", "fgmMode"].forEach((id) => $(id).addEventListener("change", updateCost));

$("densify").addEventListener("change", () => {
  const d = $("densify").checked;
  $("snapshots").disabled = d;
  if (d) $("snapshots").checked = false;
  $("snapshots").parentElement.title = d
    ? "Time snapshots need density-state chaining the solver cannot re-inject yet (spec 4.2 Tier 3)" : "";
});

async function runIntake() {
  const f = $("stlFile").files[0];
  const box = $("intakeVerdict");
  if (!f) { box.innerHTML = `<div class="wb-error">choose an STL file first</div>`; return; }
  box.innerHTML = `<div class="wb-note">validating (watertight, self-intersection, chamber fit)...</div>`;
  const buf = new Uint8Array(await f.arrayBuffer());
  let bin = ""; for (let i = 0; i < buf.length; i++) bin += String.fromCharCode(buf[i]);
  try {
    const r = await fetch("/api/heatr3d/wb/intake", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stl_b64: btoa(bin), stl_name: f.name }) });
    const v = await r.json();
    if (v.error) { state.intake = null; box.innerHTML = `<div class="wb-error">${esc(v.error)}</div>`; return; }
    if (!v.accepted) {
      state.intake = null;
      box.innerHTML = `<div class="wb-error">REFUSED (${esc(v.error_type)}):\n${esc(v.error)}</div>`;
      return;
    }
    state.intake = v;
    box.innerHTML = `<div class="wb-ok">accepted: ${esc(f.name)} &middot; ${v.volume_mm3.toFixed(0)} mm^3 &middot; ` +
      `extents ${v.extents_mm.map((x) => x.toFixed(1)).join(" x ")} mm &middot; ${v.n_faces} faces</div>`;
  } catch (e) { box.innerHTML = `<div class="wb-error">intake failed: ${esc(e.message)}</div>`; }
}
$("intakeBtn").addEventListener("click", runIntake);

function buildCfg() {
  const src = $("srcSel").value;
  const cfg = {
    source: src,
    n: parseInt($("gridN").value, 10),
    fgm: $("fgmMode").value,
    magnitude: parseFloat($("magnitude").value),
    densify: $("densify").checked,
    snapshots: $("snapshots").checked,
    exposure_s: parseFloat($("exposure").value),
    stop_mean_rho: parseFloat($("stopRho").value),
    phase_update: $("phaseUpdate").value,
  };
  const eqs = parseFloat($("eqsInterval").value), st = parseFloat($("sigmaTemp").value);
  if (eqs > 0) cfg.eqs_update_interval_s = eqs;
  if (st !== 0) cfg.sigma_temp_coeff_per_K = st;
  if (src === "library") cfg.library_shape = state.selectedShape;
  if (src === "parametric") {
    cfg.shape = $("shapeSel").value;
    cfg.diam = parseFloat($("diam").value) / 1000.0;
    cfg.zspan = parseFloat($("zspan").value) / 1000.0;
  }
  if (src === "stl" && state.intake) { cfg.stl = state.intake.stl; cfg.stl_name = state.intake.stl_name; }
  return cfg;
}

async function enqueue(cfg, campaign) {
  const r = await fetch("/api/heatr3d/wb/enqueue", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(campaign ? { ...cfg, campaign } : cfg) });
  return r.json();
}

$("launchBtn").addEventListener("click", async () => {
  const msg = $("launchMsg");
  const cfg = buildCfg();
  if (cfg.source === "library" && !cfg.library_shape) {
    msg.innerHTML = `<div class="wb-error">pick a library shape first</div>`; return;
  }
  if (cfg.source === "stl" && !cfg.stl) {
    msg.innerHTML = `<div class="wb-error">validate an STL first (intake gate)</div>`; return;
  }
  const r = await enqueue(cfg);
  msg.innerHTML = r.error ? `<div class="wb-error">${esc(r.error)}</div>`
    : `<div class="wb-ok">queued run ${esc(r.id)}</div>`;
  pollQueue();
});

$("campaignBtn").addEventListener("click", async () => {
  const msg = $("launchMsg");
  const base = buildCfg();
  const loadable = state.library.filter((s) => s.loadable).map((s) => s.name);
  if (!loadable.length) { msg.innerHTML = `<div class="wb-error">library unavailable</div>`; return; }
  const stamp = new Date().toISOString().slice(0, 16).replace("T", " ");
  const campaign = `library ${stamp} n=${base.n}${base.densify ? " densify" : ""}`;
  let ok = 0, errs = [];
  for (const name of loadable) {
    const r = await enqueue({ ...base, source: "library", library_shape: name }, campaign);
    if (r.error) errs.push(`${name}: ${r.error}`); else ok++;
  }
  msg.innerHTML = `<div class="${errs.length ? "wb-error" : "wb-ok"}">queued ${ok}/${loadable.length} shapes as "${esc(campaign)}"` +
    (errs.length ? `\n${esc(errs.join("\n"))}` : "") + `</div>`;
  pollQueue();
});

// ── RUN RAIL / queue ────────────────────────────────────────────────────────
async function pollQueue() {
  const box = $("wbQueue");
  try {
    const r = await wbFetch("/api/heatr3d/wb/queue");
    state.queue = r.jobs || [];
  } catch (e) {
    box.innerHTML = `<span class="wb-error" style="display:block;">queue unavailable: ${esc(e.message)}</span>`;
    return;
  }
  if (!state.queue.length) { box.innerHTML = `<span class="wb-note">no workbench runs yet</span>`; return; }
  box.innerHTML = "";
  for (const j of state.queue.slice(0, 25)) {
    const el = document.createElement("div");
    el.className = "wb-qitem";
    const cs = j.cfg_summary || {};
    const prog = j.state === "running" ? (j.progress ?? 0) : (j.state === "done" ? 100 : 0);
    el.innerHTML =
      `<div class="top"><span class="name">${esc(j.shape)}</span>` +
      `<span class="wb-state st-${esc(j.state)}">${esc(j.state)}${j.phase && j.state === "running" ? " &middot; " + esc(j.phase) : ""}</span></div>` +
      `<div class="wb-note">${esc(cs.fgm || "none")}${cs.densify ? "+dens" : ""} &middot; n${esc(cs.n ?? "?")}${j.campaign ? " &middot; " + esc(j.campaign) : ""}</div>` +
      (j.state === "running" || j.state === "queued" ? `<div class="wb-qprog"><div style="width:${prog}%"></div></div>` : "");
    el.addEventListener("click", () => openStudy(j.id));
    if (j.state === "running" || j.state === "queued") {
      const c = document.createElement("button");
      c.className = "wb-navbtn"; c.style.cssText = "width:auto;height:auto;padding:1px 8px;font-size:.62rem;align-self:flex-end;";
      c.textContent = "cancel";
      c.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        await fetch("/api/heatr3d/wb/cancel", { method: "POST",
          headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: j.id }) });
        pollQueue();
      });
      el.appendChild(c);
    }
    box.appendChild(el);
  }
}
setInterval(pollQueue, 2000);

// ── runs list (legacy endpoint; shape-grouped pickers preserved) ────────────
function runLabel(r) {
  return `${r.shape || "?"} · ${r.fgm || "none"}${r.densify ? "+dens" : ""} · n${r.grid_n || "?"}`;
}
async function loadRuns() {
  try { state.runs = await (await fetch("/api/heatr3d/runs")).json(); } catch (e) { state.runs = []; }
  fillRunPicker($("studyRunSel"), "pick a run...");
  fillComparePickers();
}
function fillRunPicker(sel, placeholder) {
  const keep = sel.value;
  sel.innerHTML = `<option value="">${placeholder}</option>`;
  const byShape = {};
  for (const r of state.runs) (byShape[r.shape || "?"] = byShape[r.shape || "?"] || []).push(r);
  for (const shape of Object.keys(byShape).sort()) {
    const og = document.createElement("optgroup"); og.label = shape;
    const rs = byShape[shape].slice().sort((a, b) =>
      (a.fgm || "none").localeCompare(b.fgm || "none") || (b.mtime || 0) - (a.mtime || 0));
    for (const r of rs) {
      const o = document.createElement("option");
      o.value = r.id;
      const sig = (r.sigma_T != null) ? ` · sigT3D ${Number(r.sigma_T).toFixed(1)}` : "";
      o.textContent = `${r.fgm || "none"}${r.densify ? "+dens" : ""} · n${r.grid_n || "?"}${sig}`;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  if (keep) sel.value = keep;
}

// ── STUDY screen ────────────────────────────────────────────────────────────
let viewer = null;   // three.js state, created on demand

function openStudy(id) {
  showScreen("study");
  $("studyRunSel").value = id;
  loadStudy(id);
}
$("studyRunSel").addEventListener("change", () => loadStudy($("studyRunSel").value));

async function loadStudy(id) {
  if (!id) return;
  state.currentRun = id;
  history.replaceState(null, "", `?run=${encodeURIComponent(id)}`);
  const body = $("studyBody");
  body.innerHTML = `<p class="wb-note">loading run ${esc(id)}...</p>`;
  let d;
  try {
    const r = await fetch(`/api/heatr3d/wb/run?id=${encodeURIComponent(id)}`);
    if (r.status === 404 && r.headers.get("Content-Type")?.includes("json") === false)
      throw new Error(STALE_SERVER_MSG);
    d = await r.json();
  }
  catch (e) { body.innerHTML = `<div class="wb-error">failed to load run: ${esc(e.message)}</div>`; return; }
  state.detail = d;
  if (d.error && !d.results) {
    const st = d.state ? ` (state: ${esc(d.state)})` : "";
    body.innerHTML = `<div class="wb-error">${esc(d.error)}${st}</div>` + liveMarchHtml(d);
    if (d.state === "running" || d.state === "queued") scheduleStudyRepoll(id);
    return;
  }
  renderStudy(d);
  // legacy geometry + warp views come from the preserved endpoints
  hydrateViewer(id, d);
}

function scheduleStudyRepoll(id) {
  if (state.studyPollTimer) clearTimeout(state.studyPollTimer);
  state.studyPollTimer = setTimeout(() => { if (state.currentRun === id) loadStudy(id); }, 2500);
}

function liveMarchHtml(d) {
  const m = d.march;
  if (!m || !m.series || !m.series.t_s.length) return "";
  const s = m.series, i = s.t_s.length - 1;
  return `<div class="wb-note">live march: t=${s.t_s[i].toFixed(1)} s, T_max=${fmt(s.T_max_c[i], 1)} C, ` +
    `phi_bar=${fmt(s.phi_bar[i], 3)} (${m.progress}%, phase ${esc(m.phase)})</div>`;
}

function stat(v, label, delta = "") {
  return `<div class="wb-stat"><div class="v">${v}</div><div class="l">${label}</div>${delta}</div>`;
}

function shapeStripHtml(res, sm, badges) {
  const pick = (k) => (sm && sm[k] !== undefined && sm[k] !== null) ? sm[k]
    : (res[`shape_${k}`] !== undefined ? res[`shape_${k}`] : null);
  const legacy = !sm && res.shape_iou_phi90 === undefined;
  if (legacy) {
    return `<div class="wb-metric-title">Shape fidelity ${BADGE(badges && badges.shape)}</div>` +
      `<div class="wb-note">not computed (legacy run - shape metrics ship with workbench runs)</div>`;
  }
  return `<div class="wb-metric-title">Shape fidelity (headline) ${BADGE(badges && badges.shape)}</div>` +
    `<div class="wb-strip">` +
    stat(fmt(pick("iou_phi90")), "IoU vs nominal, phi &ge; 0.9") +
    stat(fmt(pick("iou_phi80")), "IoU vs nominal, phi &ge; 0.8") +
    stat(fmt(pick("oob_melt_frac_phi90"), 4), "out-of-bounds melt / part vol (dense iff in-bounds: hard side)") +
    stat(fmt(pick("in_part_melt_frac_phi90")), "in-part melt fraction") +
    stat(pick("front_dist_phi90_mm") == null ? "not computed" : fmt(pick("front_dist_phi90_mm"), 2) + " mm",
      "front distance phi = 0.9") +
    `</div>`;
}

function renderStudy(d) {
  const res = d.results, b = d.badges || {}, cfg = d.config || {};
  const flagsHtml = (d.flags || []).map((f) =>
    `<span class="wb-flag fl-${f.state}" title="${esc(f.text)}">${esc(f.id)}: ${esc(f.state)}</span>`).join("");
  const bannerText = { fail: "A standing gate FAILED on this run. Numbers below are suspect.",
    warn: "Standing-gate caveats on this run (hover the flags).",
    ok: "All standing gates pass." }[d.banner || "warn"];
  const dens = res.rho_final_mean !== undefined && res.rho_final_mean !== null;

  $("studyBody").innerHTML = `
    <div class="wb-banner ${esc(d.banner)}">${bannerText}<div class="wb-flags">${flagsHtml}</div></div>
    ${shapeStripHtml(res, d.shape_metrics, b)}
    <div class="wb-metric-title">Diagnostics ${BADGE(b.eqs)}</div>
    <div class="wb-strip">
      ${stat(fmt(res.sigma_T, 2) + " C", "sigma_T^3D (diagnostic only; NOT comparable to 2-D sigma_T)")}
      ${stat(fmt(res.T_max_C, 1) + " C", "T_max (ceiling 250 C)")}
      ${stat(res.reached_phi90 ? fmt(res.t_phi90_s, 1) + " s" : "not reached", "t at phi = 0.90")}
      ${stat(fmt(res.dice), "Dice (demoted: cannot see out-of-bounds spill)")}
      ${stat(fmt(res.solve_s, 1) + " s", "solve wall time")}
    </div>
    ${dens ? `<div class="wb-metric-title">Densification and shrinkage ${BADGE(b.shrinkage)}</div>
    <div class="wb-strip">
      ${stat(fmt(res.rho_final_mean), "mean relative density")}
      ${stat(fmt(res.rho_final_std), "density spread")}
      ${stat(fmt(res.z_shrink_pct, 1) + "% / " + fmt(res.xy_shrink_pct, 1) + "%", "Z / XY shrink")}
      ${stat(fmt(res.warp_std_pct, 2) + "%", "warp (column scatter)")}
      ${stat(fmt(res.green_layers, 0) + " x " + fmt(res.layer_multiplier, 2), "green layers x multiplier")}
    </div>` : `<div class="wb-note" style="margin:6px 0;">density not computed (densify was off)</div>`}
    <div class="wb-metric-title">Run provenance</div>
    <div class="wb-strip">
      ${stat(esc(res.phase_update || "apparent_cp (legacy)"), "phase update scheme")}
      ${stat(esc(res.heatr3d_engine_version || "pre-workbench"), "engine version")}
      ${stat(esc(String(res.grid_n)), "grid n")}
      ${stat(esc(res.fgm || "none") + (res.densify ? " + densify" : ""), "mode")}
    </div>

    <div class="wb-2col" style="margin-top:12px;">
      <div>
        <div class="wb-metric-title">3-D view</div>
        <div id="wbViewport"><div class="wb-vp-hint">drag to orbit &middot; scroll to zoom</div></div>
        <div class="wb-slice-ctrls">
          <label class="wb-check"><input type="checkbox" id="wbCut" checked> cutting plane</label>
          <label class="wb-check" id="wbSmoothRow" style="display:none;">
            <input type="checkbox" id="wbSmoothPart" checked> smooth part (exact STL)</label>
          <label class="wb-check" id="wbIsoRow" style="display:none;">
            <input type="checkbox" id="wbIso"> melt front surface (phi = 0.9)</label>
          <label class="wb-check" id="wbWarpRow" style="display:none;">
            <input type="checkbox" id="wbWarp"> post-sinter warp <span id="wbWarpMax" class="wb-note"></span></label>
          <label class="wb-check"><input type="checkbox" id="wbColorSat"> color shell by dopant</label>
        </div>
        <div class="wb-note" id="wbResNote"></div>
        <div class="wb-time">
          <div class="wb-metric-title">Time ${d.snapshots ? "(volume snapshots)" : "(scalar series; volumes are final-state)"}</div>
          <canvas id="wbTimeChart"></canvas>
          <div class="wb-slice-ctrls">
            <input type="range" id="wbTimeScrub" min="0" max="100" value="100">
            <span id="wbTimeLabel" class="wb-note" style="flex-basis:100%;"></span>
          </div>
          <div id="wbSnapView"></div>
        </div>
      </div>
      <div>
        <div class="wb-metric-title">Cross sections</div>
        <div class="wb-slice-ctrls">
          <select id="wbField"></select>
          <select id="wbAxis"></select>
          <button class="wb-navbtn" id="wbPrev" title="previous layer">&#9664;</button>
          <input type="range" id="wbLayer" min="0" max="0" value="0">
          <button class="wb-navbtn" id="wbNext" title="next layer">&#9654;</button>
          <button class="wb-navbtn" id="wbPlay" title="play / pause">&#9654;</button>
          <span id="wbSliceLabel" class="wb-note" style="flex-basis:100%;"></span>
        </div>
        <div class="wb-slice-stage">
          <div class="wb-slice-view"><img id="wbSliceImg" alt="cross section"></div>
          <div class="wb-colorbar">
            <span id="wbCbMax">-</span>
            <div class="cbar cb-viridis" id="wbCbBar"></div>
            <span id="wbCbMin">-</span>
            <span id="wbCbUnits" class="wb-note" style="text-align:center;"></span>
          </div>
        </div>
        <div class="wb-note" id="wbSliceNote" style="margin-top:5px;"></div>
        <div class="wb-metric-title">Summary plots</div>
        <div class="wb-plots" id="wbPlots"></div>
      </div>
    </div>`;

  wireSlices(d);
  wireTime(d);
  wirePlots(d);
}

// slices --------------------------------------------------------------------
function wireSlices(d) {
  const id = d.id, meta = d.fieldmeta;
  const fieldSel = $("wbField"), axisSel = $("wbAxis"), layer = $("wbLayer");
  if (!meta || !meta.fields || !Object.keys(meta.fields).length) {
    $("wbSliceNote").textContent = "no field slices for this run";
    return;
  }
  fieldSel.innerHTML = "";
  for (const f of Object.keys(meta.fields)) {
    const o = document.createElement("option");
    o.value = f; o.textContent = meta.fields[f].label || f;
    fieldSel.appendChild(o);
  }
  const axes = d.slice_axes || ["z"];
  axisSel.innerHTML = "";
  for (const a of ["x", "y", "z"]) {
    if (!axes.includes(a)) continue;
    const o = document.createElement("option");
    o.value = a; o.textContent = `${a.toUpperCase()} axis`;
    axisSel.appendChild(o);
  }
  axisSel.value = axes.includes("z") ? "z" : axes[0];
  $("wbSliceNote").textContent = axes.length === 1
    ? "legacy run: z-axis slices only (x/y ship with workbench runs)" : "";

  const dims = meta.dims || [0, 0, 0];
  const axisN = { x: dims[0], y: dims[1], z: dims[2] };
  const fmtNum = (v) => (Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-2)) ? v.toExponential(2) : v.toPrecision(4);

  const show = () => {
    const f = fieldSel.value, ax = axisSel.value, k = parseInt(layer.value, 10);
    const info = meta.fields[f];
    $("wbSliceImg").src = FILES(id, `slices/${f}_${ax}_${String(k).padStart(3, "0")}.png`);
    $("wbSliceLabel").textContent = `${f} · ${ax} layer ${k} / ${axisN[ax] - 1} · phi front contours: solid 0.9, dashed 0.5`;
    $("wbCbMax").textContent = fmtNum(info.max);
    $("wbCbMin").textContent = fmtNum(info.min);
    $("wbCbUnits").textContent = FIELD_UNITS[f] || "";
    $("wbCbBar").className = "cbar " + (CMAP_CLASS[f] || "cb-viridis");
    if (viewer) viewer.setCut(ax, k, axisN[ax], $("wbSliceImg").src);
  };
  const reset = () => {
    const ax = axisSel.value;
    layer.max = Math.max(0, (axisN[ax] || 1) - 1);
    layer.value = Math.floor((axisN[ax] || 1) / 2);
    show();
  };
  fieldSel.onchange = show; axisSel.onchange = reset;
  layer.oninput = show;
  $("wbPrev").onclick = () => { layer.value = Math.max(0, +layer.value - 1); show(); };
  $("wbNext").onclick = () => { layer.value = Math.min(+layer.max, +layer.value + 1); show(); };
  let playT = null;
  $("wbPlay").onclick = () => {
    if (playT) { clearInterval(playT); playT = null; $("wbPlay").innerHTML = "&#9654;"; }
    else {
      $("wbPlay").innerHTML = "&#10073;&#10073;";
      playT = setInterval(() => { layer.value = (+layer.value + 1) % (+layer.max + 1); show(); }, 180);
    }
  };
  reset();
}

// time panel ----------------------------------------------------------------
function wireTime(d) {
  const canvas = $("wbTimeChart");
  const series = (d.march && d.march.series) || { t_s: [], phi_bar: [], T_max_c: [] };
  const snaps = d.snapshots;
  const scrub = $("wbTimeScrub"), label = $("wbTimeLabel"), snapView = $("wbSnapView");
  const W = canvas.clientWidth || 500, H = 120;
  canvas.width = W * devicePixelRatio; canvas.height = H * devicePixelRatio;
  const ctx = canvas.getContext("2d");
  ctx.scale(devicePixelRatio, devicePixelRatio);

  const draw = (cursorFrac) => {
    ctx.clearRect(0, 0, W, H);
    const t = series.t_s;
    if (t.length > 1) {
      const tmax = t[t.length - 1] || 1;
      const line = (vals, color, vmax) => {
        ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.beginPath();
        vals.forEach((v, i) => {
          const x = (t[i] / tmax) * (W - 10) + 5;
          const y = H - 8 - ((v ?? 0) / vmax) * (H - 20);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        });
        ctx.stroke();
      };
      line(series.phi_bar, "#4fd08a", 1.0);
      const tm = Math.max(...series.T_max_c.filter((v) => v != null), 1);
      line(series.T_max_c, "#f0a05a", tm);
      ctx.fillStyle = "#8a94a6"; ctx.font = "10px sans-serif";
      ctx.fillText("phi_bar (green) / T_max (orange), t 0.." + tmax.toFixed(0) + " s", 8, 12);
      if (cursorFrac != null) {
        const x = cursorFrac * (W - 10) + 5;
        ctx.strokeStyle = "#ffffff88"; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      }
    } else {
      ctx.fillStyle = "#8a94a6"; ctx.font = "11px sans-serif";
      ctx.fillText("no march series recorded for this run (legacy run)", 10, 20);
    }
  };

  const update = () => {
    const frac = (+scrub.value) / 100;
    draw(frac);
    const t = series.t_s;
    if (t.length > 1) {
      const i = Math.min(t.length - 1, Math.round(frac * (t.length - 1)));
      label.textContent = `t = ${t[i].toFixed(1)} s · phi_bar ${fmt(series.phi_bar[i], 3)} · T_max ${fmt(series.T_max_c[i], 1)} C` +
        (snaps ? "" : " · volumes shown elsewhere are FINAL-STATE, not this instant");
    } else label.textContent = "";
    if (snaps && snaps.snaps && snaps.snaps.length) {
      const j = Math.min(snaps.snaps.length - 1, Math.round(frac * (snaps.snaps.length - 1)));
      const s = snaps.snaps[j];
      snapView.innerHTML = `<img class="wb-snapimg" src="${FILES(d.id, s.T_png)}" alt="snapshot">` +
        `<div class="wb-note">snapshot ${j + 1}/${snaps.snaps.length} at t = ${s.t_s.toFixed(1)} s ` +
        `(T mid-z slice, inferno, ${snaps.vmin_c.toFixed(0)}..${snaps.vmax_c.toFixed(0)} C; white contour = phi 0.9 front)</div>`;
    }
  };
  scrub.oninput = update;
  update();
}

// plots ---------------------------------------------------------------------
function wirePlots(d) {
  const gal = $("wbPlots");
  gal.innerHTML = "";
  const plots = ["melt_vs_cad", "ortho_slices", "melt_progression", "fgm_z_profile",
    "temperature_hist", "density_hist", "radial_density"];
  for (const name of plots) {
    const img = document.createElement("img");
    img.src = FILES(d.id, `plots/${name}.png`);
    img.alt = name; img.loading = "lazy";
    img.onerror = () => img.remove();
    gal.appendChild(img);
  }
}

// three.js viewer -----------------------------------------------------------
function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[0.27, 0, 0.33], [0.13, 0.57, 0.55], [0.99, 0.91, 0.14]];
  const i = t < 0.5 ? 0 : 1, f = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  const a = stops[i], b = stops[i + 1];
  return new THREE.Color(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f);
}

function makeViewer(vp) {
  const w = vp.clientWidth, h = vp.clientHeight;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(w, h);
  vp.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e14);
  const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 5000);
  camera.position.set(70, 50, 90);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const d1 = new THREE.DirectionalLight(0xffffff, 0.9); d1.position.set(60, 80, 50); scene.add(d1);
  const d2 = new THREE.DirectionalLight(0x88aaff, 0.4); d2.position.set(-50, -30, -40); scene.add(d2);
  scene.add(new THREE.AxesHelper(35));
  const st = { renderer, scene, camera, controls, shell: null, cutPlane: null, electrodes: null,
    geom: null, warp: null, cutOn: true, halfL: 30 };
  (function anim() { requestAnimationFrame(anim); controls.update(); renderer.render(scene, camera); })();

  st.renderShell = (geom, colorBy) => {
    if (st.shell) { scene.remove(st.shell); st.shell.geometry.dispose(); st.shell.material.dispose(); st.shell = null; }
    if (st.electrodes) { scene.remove(st.electrodes); st.electrodes = null; }
    const pts = geom.surface_xyz_mm, hmm = geom.h_mm;
    const vals = colorBy === "sat" ? (geom.surface_sat || null)
      : colorBy === "disp" ? (geom.surface_disp || null) : null;
    if (colorBy === "sat" && !geom.surface_sat) {
      // explicit state, never a silent blue fallback (F7 intent)
      $("wbSliceNote").textContent = "no dopant field on this run; shell shown uncolored";
    }
    const box = new THREE.BoxGeometry(hmm, hmm, hmm);
    const mat = new THREE.MeshLambertMaterial({ vertexColors: !!vals, transparent: true,
      opacity: st.cutOn ? 0.55 : 1.0 });
    if (!vals) mat.color = new THREE.Color(0x4f9dff);
    const mesh = new THREE.InstancedMesh(box, mat, pts.length);
    const m = new THREE.Matrix4();
    for (let i = 0; i < pts.length; i++) {
      m.setPosition(pts[i][0], pts[i][1], pts[i][2]);
      mesh.setMatrixAt(i, m);
      if (vals) mesh.setColorAt(i, viridis(vals[i]));
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    scene.add(mesh); st.shell = mesh;
    const L = geom.L_mm || 60; st.halfL = L / 2;
    const grp = new THREE.Group();
    const pg = new THREE.PlaneGeometry(L, L);
    for (const sgn of [-1, 1]) {
      const pm = new THREE.Mesh(pg, new THREE.MeshBasicMaterial({ color: 0x666e7a, transparent: true,
        opacity: 0.10, side: THREE.DoubleSide }));
      pm.rotation.x = Math.PI / 2; pm.position.y = sgn * L / 2; grp.add(pm);
    }
    scene.add(grp); st.electrodes = grp;
  };

  st.setCut = (axis, k, n, imgUrl) => {
    if (st.cutPlane) { scene.remove(st.cutPlane); st.cutPlane.geometry.dispose(); st.cutPlane.material.map?.dispose(); st.cutPlane.material.dispose(); st.cutPlane = null; }
    if (!st.cutOn || !n) return;
    const L = st.halfL * 2;
    const pos = -st.halfL + (k + 0.5) * (L / n);
    const tex = new THREE.TextureLoader().load(imgUrl);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.NearestFilter;
    // slice PNGs carry an opaque slate bed tone; keep the plane translucent so
    // the part remains readable behind it
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true,
      opacity: 0.82, depthWrite: false, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(L, L), mat);
    // slice images are rendered as (first-axis horizontal, second-axis vertical)
    if (axis === "z") { plane.rotation.x = -Math.PI / 2; plane.rotation.z = 0; plane.position.z = pos; plane.rotation.set(0, 0, 0); plane.position.set(0, 0, pos); }
    if (axis === "x") { plane.rotation.y = Math.PI / 2; plane.position.set(pos, 0, 0); }
    if (axis === "y") { plane.rotation.x = Math.PI / 2; plane.position.set(0, pos, 0); }
    scene.add(plane); st.cutPlane = plane;
    if (st.shell) st.shell.material.opacity = 0.55;
  };
  st.clearCut = () => {
    if (st.cutPlane) { scene.remove(st.cutPlane); st.cutPlane = null; }
    if (st.shell) st.shell.material.opacity = 1.0;
  };
  return st;
}

async function hydrateViewer(id, d) {
  const vp = $("wbViewport");
  if (!vp) return;
  viewer = makeViewer(vp);
  let geom = null;
  try {
    const s = await (await fetch(`/api/heatr3d/status?id=${encodeURIComponent(id)}`)).json();
    geom = s.geometry || null;
  } catch (e) { /* no geometry */ }
  if (!geom) { $("wbSliceNote").textContent = "no geometry.json on disk for this run"; return; }
  viewer.geom = geom;
  viewer.renderShell(geom, "geom");

  const n = (d.fieldmeta && d.fieldmeta.dims) ? d.fieldmeta.dims[0] : "?";
  $("wbResNote").textContent =
    `Part surface: exact STL where available. Field surfaces, slice smoothing ` +
    `and interpolation are DISPLAY-side only - the physics is computed at n = ${n}. ` +
    `Bed/outside-part renders as slate, never the colormap floor.`;

  // Smooth exact-STL part surface (item 1/2a): replaces the voxel shell for
  // library/STL runs; the voxel shell remains for dopant coloring and warp.
  const applyShellMode = () => {
    const smooth = $("wbSmoothPart").checked && viewer.smoothMesh &&
      !viewer.warpOn && !$("wbColorSat").checked;
    if (viewer.smoothMesh) viewer.smoothMesh.visible = !!smooth;
    if (viewer.shell) viewer.shell.visible = !smooth;
  };
  try {
    const stlGeo = await fetchStlGeometry(`/api/heatr3d/wb/stl?run=${encodeURIComponent(id)}`);
    const mat = new THREE.MeshPhongMaterial({ color: 0xe8a25c, shininess: 24,
      specular: 0x222222, transparent: true, opacity: 0.92, side: THREE.DoubleSide });
    viewer.smoothMesh = new THREE.Mesh(stlGeo, mat);
    viewer.scene.add(viewer.smoothMesh);
    $("wbSmoothRow").style.display = "";
    $("wbSmoothPart").addEventListener("change", applyShellMode);
    applyShellMode();
  } catch (e) { /* parametric run: no STL; voxel shell stays */ }

  // Melt-front isosurface (item 2b): marching-cubes display interpolation.
  try {
    const iso = await (await fetch(FILES(id, "isosurfaces.json"))).json();
    const s = (iso.surfaces || []).find((x) => x.field === "phi_final" && x.level === 0.9);
    if (s && iso.offset_mm) {
      const pos = new Float32Array(s.vertices_mm.length * 3);
      s.vertices_mm.forEach((v, i) => {
        pos[3 * i] = v[0] + iso.offset_mm[0];
        pos[3 * i + 1] = v[1] + iso.offset_mm[1];
        pos[3 * i + 2] = v[2] + iso.offset_mm[2];
      });
      const g2 = new THREE.BufferGeometry();
      g2.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g2.setIndex(s.faces.flat());
      g2.computeVertexNormals();
      const m2 = new THREE.Mesh(g2, new THREE.MeshPhongMaterial({
        color: 0xff5533, transparent: true, opacity: 0.55, side: THREE.DoubleSide }));
      m2.visible = false;
      viewer.scene.add(m2);
      viewer.isoMesh = m2;
      $("wbIsoRow").style.display = "";
      $("wbIso").addEventListener("change", () => { m2.visible = $("wbIso").checked; });
    }
  } catch (e) { /* no isosurfaces for this run (legacy) */ }

  $("wbColorSat").addEventListener("change", () => {
    viewer.renderShell(viewer.warpOn ? viewer.warpGeom() : geom, $("wbColorSat").checked ? "sat" : "geom");
    applyShellMode();
  });
  $("wbCut").addEventListener("change", () => {
    viewer.cutOn = $("wbCut").checked;
    if (!viewer.cutOn) viewer.clearCut();
    else $("wbLayer").dispatchEvent(new Event("input"));
  });
  // warp (densify runs)
  if (d.has_warp) {
    try {
      const w = await (await fetch(`/api/heatr3d/warp?id=${encodeURIComponent(id)}`)).json();
      if (w && Array.isArray(w.warped_xyz_mm) && w.warped_xyz_mm.length) {
        $("wbWarpRow").style.display = "";
        $("wbWarpMax").textContent = `max ${w.disp_max_mm.toFixed(2)} mm`;
        viewer.warpGeom = () => ({
          surface_xyz_mm: w.warped_xyz_mm,
          surface_disp: w.disp_mm.map((x) => x / (w.disp_max_mm || 1)),
          h_mm: w.h_mm, L_mm: geom.L_mm,
        });
        $("wbWarp").addEventListener("change", () => {
          viewer.warpOn = $("wbWarp").checked;
          if (viewer.warpOn) viewer.renderShell(viewer.warpGeom(), "disp");
          else viewer.renderShell(geom, $("wbColorSat").checked ? "sat" : "geom");
          applyShellMode();
        });
      }
    } catch (e) { /* warp unavailable */ }
  }
  // initial cut
  $("wbLayer").dispatchEvent(new Event("input"));
}

// ── SOLVED screen ───────────────────────────────────────────────────────────
async function loadSolved() {
  const body = $("solvedBody");
  let cards;
  try { cards = (await wbFetch("/api/heatr3d/wb/solved")).cards || []; }
  catch (e) { body.innerHTML = `<div class="wb-error">failed to load solve3d results: ${esc(e.message)}</div>`; return; }
  if (!cards.length) { body.innerHTML = `<div class="wb-note">no solved-map artifacts found in solve3d/results</div>`; return; }
  const rows = cards.filter((c) => c.kind !== "report");
  const report = cards.find((c) => c.kind === "report");
  const num = (v, d = 4) => (v === undefined || v === null) ? "-" : Number(v).toExponential(d);
  let html = "";
  const uniform = rows.find((c) => c.arm === "uniform_baseline");
  for (const c of rows) {
    const chips = [];
    if (c.solved_label) chips.push(`<span class="chip chip-solved">SOLVED</span>`);
    if (c.status === "DROPPED") chips.push(`<span class="chip chip-dropped">DROPPED</span>`);
    if (c.status === "NOT_RUN") chips.push(`<span class="chip chip-notrun">NOT RUN</span>`);
    if (c.deviation) chips.push(`<span class="chip chip-dev" title="recorded deviation: 1/|g0| first-step rescale (pure reparameterization)">recorded deviation</span>`);
    if (c.gates) {
      chips.push(`<span class="chip ${c.gates.mesh_holdout_pass ? "chip-pass" : "chip-fail"}">mesh hold-out ${c.gates.mesh_holdout_pass ? "PASS" : "FAIL"}</span>`);
      chips.push(`<span class="chip ${c.gates.smoothing_pass ? "chip-pass" : "chip-fail"}">smoothing ${c.gates.smoothing_pass ? "PASS" : "FAIL"}</span>`);
    }
    let margin = "";
    if (uniform && c !== uniform && c.J_asymmetric != null && uniform.J_asymmetric != null) {
      const m = (1 - c.J_asymmetric / uniform.J_asymmetric) * 100;
      margin = `<span class="wb-note">J improvement vs uniform: <b class="${m > 0 ? "d-good" : "d-bad"}">${m.toFixed(2)}%</b></span>`;
    }
    html += `<div class="wb-solved-card">
      <h4>${esc(c.arm)} ${chips.join(" ")} ${BADGE(c.badge)}</h4>
      <div class="wb-note">${esc(c.shape)} &middot; campaign ${esc(c.campaign)} &middot; status ${esc(c.status)}</div>
      ${c.status_detail ? `<div class="wb-note">${esc(c.status_detail)}</div>` : ""}
      ${margin}
      ${c.J_asymmetric != null ? `<table class="wb-table" style="margin-top:6px;">
        <tr><th>quantity</th><th>value</th></tr>
        <tr><td>J asymmetric (PRIMARY)</td><td>${num(c.J_asymmetric)}</td></tr>
        <tr><td>J symmetric (2-D comparable control)</td><td>${num(c.J_symmetric)}</td></tr>
        <tr><td>J out-of-bounds (bed growth)</td><td>${num(c.J_out_of_bounds)}</td></tr>
        <tr><td>J in-bounds deficit</td><td>${num(c.J_in_bounds_deficit)}</td></tr>
        ${c.bed_melt_frac_phi09 != null ? `<tr><td>bed melt fraction, phi &ge; 0.9</td><td>${Number(c.bed_melt_frac_phi09).toFixed(6)}</td></tr>` : ""}
        ${c.in_part_melt_frac_phi09 != null ? `<tr><td>in-part melt fraction, phi &ge; 0.9</td><td>${Number(c.in_part_melt_frac_phi09).toFixed(6)}</td></tr>` : ""}
        ${c.map_mean != null ? `<tr><td>map mean / min / max</td><td>${Number(c.map_mean).toFixed(4)} / ${Number(c.map_min).toFixed(4)} / ${Number(c.map_max).toFixed(4)}</td></tr>` : ""}
      </table>` : ""}
    </div>`;
  }
  if (report) {
    html += `<p class="wb-note">Full campaign record: ${esc(report.report)} (pre-registration commit ${esc(report.prereg_commit)}).
      Solved maps live on the dolfinx tetrahedral mesh; they are NOT resampled onto the heatr3d voxel grid for display
      (measured staircase-support transfer hazard: 7.15% dopant shift).</p>`;
  }
  body.innerHTML = html;
}

// ── COMPARE screen ──────────────────────────────────────────────────────────
function fillComparePickers() {
  fillRunPicker($("cmpA"), "run A...");
  fillRunPicker($("cmpB"), "run B...");
}
$("cmpA").addEventListener("change", renderCompare);
$("cmpB").addEventListener("change", renderCompare);

async function renderCompare() {
  const a = $("cmpA").value, b = $("cmpB").value;
  const body = $("cmpBody");
  if (!a || !b) return;
  body.innerHTML = `<p class="wb-note">loading...</p>`;
  let da, db;
  try {
    [da, db] = await Promise.all([
      (await fetch(`/api/heatr3d/wb/run?id=${encodeURIComponent(a)}`)).json(),
      (await fetch(`/api/heatr3d/wb/run?id=${encodeURIComponent(b)}`)).json(),
    ]);
  } catch (e) { body.innerHTML = `<div class="wb-error">${esc(e.message)}</div>`; return; }
  if (!da.results || !db.results) {
    body.innerHTML = `<div class="wb-error">one of the runs has no results on disk</div>`; return;
  }
  const ra = da.results, rb = db.results;
  const shapeA = (da.config || {}).shape || (da.config || {}).library_shape || "?";
  const shapeB = (db.config || {}).shape || (db.config || {}).library_shape || "?";
  const mism = shapeA !== shapeB ? `<div class="wb-banner warn">shape mismatch: ${esc(shapeA)} vs ${esc(shapeB)} - metric deltas are not like-for-like</div>` : "";

  const sm = (d, k) => {
    if (d.shape_metrics && d.shape_metrics[k] != null) return d.shape_metrics[k];
    if (d.results[`shape_${k}`] != null) return d.results[`shape_${k}`];
    return null;
  };
  // headline: higher-better for IoU/in-part; lower-better for oob/front
  const rows = [
    ["iou_phi90", "IoU vs nominal (phi >= 0.9)", +1],
    ["iou_phi80", "IoU vs nominal (phi >= 0.8)", +1],
    ["oob_melt_frac_phi90", "out-of-bounds melt fraction", -1],
    ["in_part_melt_frac_phi90", "in-part melt fraction", +1],
    ["front_dist_phi90_mm", "front distance (mm)", -1],
  ];
  const strip = rows.map(([k, label, sign]) => {
    const va = sm(da, k), vb = sm(db, k);
    let delta = `<div class="d d-neutral">not computed on both</div>`;
    if (va != null && vb != null) {
      const diff = vb - va;
      const good = sign * diff > 0;
      delta = `<div class="d ${Math.abs(diff) < 1e-12 ? "d-neutral" : good ? "d-good" : "d-bad"}">B - A: ${diff >= 0 ? "+" : ""}${diff.toFixed(4)}</div>`;
    }
    return stat(`${fmt(va)} vs ${fmt(vb)}`, label, delta);
  }).join("");

  const badges = da.badges || {};
  const cfgKeys = ["n", "phase_update", "fgm", "magnitude", "densify", "exposure_s",
    "eqs_update_interval_s", "source", "library_shape", "shape"];
  const cfgRows = cfgKeys.map((k) => {
    const va = (da.config || {})[k], vb = (db.config || {})[k];
    if (va === undefined && vb === undefined) return "";
    const differ = JSON.stringify(va) !== JSON.stringify(vb);
    return `<tr${differ ? ' style="color:#eeda9e;"' : ""}><td>${esc(k)}</td><td>${esc(fmt(va))}</td><td>${esc(fmt(vb))}</td></tr>`;
  }).join("");

  const sigA = ra.sigma_T, sigB = rb.sigma_T;
  const maxSig = Math.max(sigA || 0, sigB || 0, 1e-9);
  const bar = (labelTxt, v, best) =>
    `<div class="wb-cmp-bar"><span class="lbl">${esc(labelTxt)}</span>` +
    `<div class="track"><div class="fill" style="width:${(v / maxSig * 100).toFixed(1)}%;background:${best ? "#40c080" : "#e0922a"};"></div></div>` +
    `<span class="val">${v == null ? "-" : v.toFixed(1)} C</span></div>`;

  body.innerHTML = `${mism}
    <div class="wb-metric-title">Shape metrics (headline) ${BADGE(badges.shape)}</div>
    <div class="wb-strip">${strip}</div>
    <div class="wb-cmp-2" style="margin-top:12px;">
      <div>
        <div class="wb-metric-title">A: ${esc(runLabel(state.runs.find((r) => r.id === a) || {}))}</div>
        ${cmpSliceHtml(a, da)}
      </div>
      <div>
        <div class="wb-metric-title">B: ${esc(runLabel(state.runs.find((r) => r.id === b) || {}))}</div>
        ${cmpSliceHtml(b, db)}
      </div>
    </div>
    <div class="wb-slice-ctrls" style="margin-top:6px;">
      <select id="cmpField"></select>
      <input type="range" id="cmpLayer" min="0" max="0" value="0">
      <span id="cmpSliceLabel" class="wb-note"></span>
    </div>
    <div class="wb-metric-title">Diagnostics (sigma_T^3D is a diagnostic, never the headline) ${BADGE(badges.eqs)}</div>
    ${bar("A sigma_T^3D", sigA, sigA <= sigB)}
    ${bar("B sigma_T^3D", sigB, sigB <= sigA)}
    <div class="wb-kv"><span class="k">t90 (s)</span><span class="v">${fmt(ra.t_phi90_s, 1)} vs ${fmt(rb.t_phi90_s, 1)}</span></div>
    <div class="wb-kv"><span class="k">T_max (C)</span><span class="v">${fmt(ra.T_max_C, 1)} vs ${fmt(rb.T_max_C, 1)}</span></div>
    <div class="wb-kv"><span class="k">gate banner</span><span class="v">${esc(da.banner)} vs ${esc(db.banner)}</span></div>
    <div class="wb-metric-title">Config diff (differences highlighted)</div>
    <table class="wb-table"><tr><th>key</th><th>A</th><th>B</th></tr>${cfgRows}</table>`;

  wireCompareSlices(a, da, b, db);
}

function cmpSliceHtml(id, d) {
  return `<div class="wb-slice-view" style="min-height:230px;"><img id="cmpImg_${esc(id)}" style="width:230px;height:230px;" alt="slice"></div>`;
}

function wireCompareSlices(a, da, b, db) {
  const fa = (da.fieldmeta && da.fieldmeta.fields) || {};
  const fb = (db.fieldmeta && db.fieldmeta.fields) || {};
  const common = Object.keys(fa).filter((k) => k in fb);
  const sel = $("cmpField"), layer = $("cmpLayer"), lab = $("cmpSliceLabel");
  if (!common.length) { lab.textContent = "no common field slices between these runs"; return; }
  sel.innerHTML = common.map((f) => `<option value="${f}">${esc(fa[f].label || f)}</option>`).join("");
  const nz = Math.min((da.fieldmeta.dims || [0, 0, 1])[2], (db.fieldmeta.dims || [0, 0, 1])[2]);
  layer.max = nz - 1; layer.value = Math.floor(nz / 2);
  const show = () => {
    const f = sel.value, k = String(parseInt(layer.value, 10)).padStart(3, "0");
    const imgA = document.getElementById(`cmpImg_${a}`), imgB = document.getElementById(`cmpImg_${b}`);
    if (imgA) imgA.src = FILES(a, `slices/${f}_z_${k}.png`);
    if (imgB) imgB.src = FILES(b, `slices/${f}_z_${k}.png`);
    lab.textContent = `${f} · z layer ${+layer.value} / ${nz - 1} · NOTE: color ranges are per-run (each run's own min/max)`;
  };
  sel.onchange = show; layer.oninput = show;
  show();
}

// ── init ────────────────────────────────────────────────────────────────────
srcChanged();
updateCost();
loadLibrary();
pollQueue();
loadRuns().then(() => {
  const qp = new URLSearchParams(location.search);
  const deep = qp.get("run") || qp.get("id");  // legacy ?id= links still work
  if (deep) openStudy(deep);                    // picker options exist by now
});
