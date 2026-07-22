// HEATR-3D page — geometry preview + run via heatr3d_job subprocess + results.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const statusLine = $("statusLine");
const progBar = $("progBar");

// ── three.js scene ──────────────────────────────────────────────────────────
const vp = $("h3dViewport");
let renderer, scene, camera, controls, partMesh = null, electrodes = null;
let lastGeom = null, nominalGeom = null, lastRunId = null, lastWarp = null;

function initScene() {
  const w = vp.clientWidth, h = vp.clientHeight;
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(w, h);
  vp.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e14);
  camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 5000);
  camera.position.set(70, 50, 90);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const d1 = new THREE.DirectionalLight(0xffffff, 0.9); d1.position.set(60, 80, 50); scene.add(d1);
  const d2 = new THREE.DirectionalLight(0x88aaff, 0.4); d2.position.set(-50, -30, -40); scene.add(d2);
  const ax = new THREE.AxesHelper(35); scene.add(ax);        // x=red y=green(field) z=blue(build)
  animate();
  window.addEventListener("resize", onResize);
}
function onResize() {
  const w = vp.clientWidth, h = vp.clientHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }

// viridis-ish colormap
function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[0.27,0,0.33],[0.13,0.57,0.55],[0.99,0.91,0.14]];
  const i = t < 0.5 ? 0 : 1, f = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  const a = stops[i], b = stops[i + 1];
  return new THREE.Color(a[0] + (b[0]-a[0])*f, a[1] + (b[1]-a[1])*f, a[2] + (b[2]-a[2])*f);
}

function renderSurface(geom, colorBy) {
  if (partMesh) { scene.remove(partMesh); partMesh.geometry.dispose(); partMesh.material.dispose(); partMesh = null; }
  if (electrodes) { scene.remove(electrodes); electrodes = null; }
  lastGeom = geom;
  const pts = geom.surface_xyz_mm, hmm = geom.h_mm;
  const vals = colorBy === "sat" ? (geom.surface_sat || null)
            : colorBy === "disp" ? (geom.surface_disp || null) : null;
  const box = new THREE.BoxGeometry(hmm, hmm, hmm);
  const mat = new THREE.MeshLambertMaterial({ vertexColors: !!vals });
  if (!vals) mat.color = new THREE.Color(0x4f9dff);
  const mesh = new THREE.InstancedMesh(box, mat, pts.length);
  const m = new THREE.Matrix4();
  for (let i = 0; i < pts.length; i++) {
    m.setPosition(pts[i][0], pts[i][1], pts[i][2]);   // x, y(field), z(build)
    mesh.setMatrixAt(i, m);
    if (vals) mesh.setColorAt(i, viridis(vals[i]));
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  scene.add(mesh); partMesh = mesh;
  // electrode planes at y = ±L/2 (field axis = y/green)
  const L = geom.L_mm, grp = new THREE.Group();
  const pg = new THREE.PlaneGeometry(L, L);
  for (const sgn of [-1, 1]) {
    const pm = new THREE.Mesh(pg, new THREE.MeshBasicMaterial({ color: 0x666e7a, transparent: true, opacity: 0.12, side: THREE.DoubleSide }));
    pm.rotation.x = Math.PI / 2; pm.position.y = sgn * L / 2; grp.add(pm);
  }
  scene.add(grp); electrodes = grp;
}

// F5: render the post-sinter WARPED surface, colored by displacement magnitude.
function renderWarped() {
  if (!lastWarp) return;
  const dmax = lastWarp.disp_max_mm || 1;
  renderSurface({
    surface_xyz_mm: lastWarp.warped_xyz_mm,
    surface_disp: lastWarp.disp_mm.map((d) => d / (dmax || 1)),
    h_mm: lastWarp.h_mm,
    L_mm: (nominalGeom && nominalGeom.L_mm) || 60,
  }, "disp");
}
// Show nominal geometry (respecting the Color-by dropdown) or the warped view.
function applyGeomView() {
  const wt = $("warpToggle");
  if (wt && wt.checked && lastWarp) renderWarped();
  else if (nominalGeom) renderSurface(nominalGeom, $("colorBy").value);
}

// ── form → config ───────────────────────────────────────────────────────────
function cfg() {
  const c = {
    src: $("srcSel").value, shape: $("shapeSel").value,
    diam: parseFloat($("diam").value) / 1000.0, zspan: parseFloat($("zspan").value) / 1000.0,
    n: parseInt($("gridN").value, 10), fgm: $("fgmMode").value,
    densify: $("densify").checked, exposure_s: parseFloat($("exposure").value),
    stop_mean_rho: parseFloat($("stopRho").value),
  };
  return c;
}
async function withStl(c) {
  if (c.src !== "stl") return c;
  const f = $("stlFile").files[0];
  if (!f) { throw new Error("choose an STL file"); }
  const buf = await f.arrayBuffer();
  let bin = ""; const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  c.stl_b64 = btoa(bin); c.stl_name = f.name;
  return c;
}

// ── actions ─────────────────────────────────────────────────────────────────
function setProg(p) { progBar.style.width = `${Math.max(0, Math.min(100, p))}%`; }

async function preview() {
  try {
    setProg(0); statusLine.textContent = "building geometry…";
    const c = await withStl(cfg());
    const r = await fetch("/api/heatr3d/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(c) });
    if (!r.ok) throw new Error(await r.text());
    const geom = await r.json();
    nominalGeom = geom;
    if ($("warpToggle")) $("warpToggle").checked = false;   // preview is nominal geometry
    applyGeomView();
    setProg(100); statusLine.textContent = `geometry: ${geom.n_voxels} voxels (${geom.dims.join("×")}), h=${geom.h_mm} mm`;
  } catch (e) { statusLine.textContent = "preview failed: " + e.message; setProg(0); }
}

let pollTimer = null;
async function run() {
  try {
    setProg(2); statusLine.textContent = "submitting run…";
    const c = await withStl(cfg());
    const r = await fetch("/api/heatr3d/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(c) });
    if (!r.ok) throw new Error(await r.text());
    const { id } = await r.json();
    statusLine.textContent = "running… (solve can take 30 s–several min)";
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => poll(id), 1500);
  } catch (e) { statusLine.textContent = "run failed: " + e.message; setProg(0); }
}

async function poll(id) {
  try {
    const r = await fetch(`/api/heatr3d/status?id=${encodeURIComponent(id)}`);
    const s = await r.json();
    if (typeof s.progress === "number") setProg(s.progress);
    if (s.geometry) { nominalGeom = s.geometry; applyGeomView(); }
    if (s.done) {
      clearInterval(pollTimer); pollTimer = null;
      if (s.error) { statusLine.textContent = "run error: " + s.error; return; }
      setProg(100); statusLine.textContent = "done.";
      if (s.results) showResults(s.results);
      if (s.geometry) { nominalGeom = s.geometry; $("colorBy").value = (cfg().fgm !== "none") ? "sat" : "geom"; applyGeomView(); }
      loadRunViews(id);
    }
  } catch (e) { /* keep polling */ }
}

// Grouped, threshold-colored metrics panel (mirrors the 2D run gauges).
function _clampPct(v, lo, hi) { return Math.min(100, Math.max(0, (v - lo) / (hi - lo) * 100)); }
function _gauge(label, valStr, pct, color, tip, fillClass) {
  const cls = fillClass ? `gauge-fill ${fillClass}` : "gauge-fill";
  const bg = color ? `background:${color};` : "";
  const cs = color ? `style="color:${color};"` : "";
  return `<div class="gauge-row" title="${tip}"><span class="gauge-label" ${cs}>${label}</span>` +
    `<div class="gauge-track"><div class="${cls}" style="width:${pct.toFixed(1)}%;${bg}"></div></div>` +
    `<span class="gauge-val" ${cs}>${valStr}</span></div>`;
}
function _txt(label, valStr, tip) {
  return `<div class="metric-txt" title="${tip || ""}"><span class="lbl">${label}</span><span class="val">${valStr}</span></div>`;
}
function _group(title, rows) {
  const r = rows.filter(Boolean).join("");
  return r ? `<div class="h3d-metric-group"><div class="h3d-metric-title">${title}</div>${r}</div>` : "";
}
function showResults(res) {
  const has = (k) => res[k] !== undefined && res[k] !== null;
  const num = (k, d) => has(k) ? (Number.isInteger(res[k]) ? res[k] : Number(res[k]).toFixed(d)) : "—";
  const sigT = res.sigma_T, dice = res.dice, warp = res.warp_std_pct;
  const sigColor = sigT == null ? null : (sigT < 3 ? "#40c080" : sigT < 5 ? "#e0c060" : "#e05050");
  const diceColor = dice == null ? null : (dice >= 0.97 ? "#40c080" : dice >= 0.9 ? "#e0c060" : "#e05050");
  const warpColor = warp == null ? null : (warp < 1 ? "#40c080" : warp < 3 ? "#e0c060" : "#e05050");
  const A = _group("Thermal uniformity", [
    has("sigma_T") && _gauge("σ_T", `${num("sigma_T", 2)} °C`, _clampPct(sigT, 0, 20), sigColor, "Spatial temperature std across the part. Target under 3 °C.", null),
    has("T_max_C") && _gauge("T_max", `${num("T_max_C", 1)} °C`, _clampPct(res.T_max_C, 25, 260), null, "Peak part temperature.", "gauge-tmax-fill"),
    _txt("φ=0.90", (has("reached_phi90") && !res.reached_phi90) ? "not reached" : (has("t_phi90_s") ? `${num("t_phi90_s", 1)} s` : "—"), "Time to reach 90% melt fraction (melt onset)."),
  ]);
  const B = _group("Sintering fidelity vs CAD", [
    has("dice") && _gauge("Dice", num("dice", 3), _clampPct(dice, 0, 1), diceColor, "Overlap of the sintered body with the CAD part (1 = perfect).", null),
    has("sintered_frac") && _gauge("sintered", `${(res.sintered_frac * 100).toFixed(0)}%`, res.sintered_frac * 100, null, "Fraction of CAD voxels that sintered.", "gauge-melt-fill"),
    (has("unsintered_vox") && has("nominal_vox")) && _txt("unsintered", `${res.unsintered_vox} / ${res.nominal_vox} vox`, "CAD voxels that stayed unsintered."),
  ]);
  const C = has("rho_final_mean") ? _group("Densification", [
    _gauge("ρ̄", num("rho_final_mean", 3), _clampPct(res.rho_final_mean, 0.45, 1.0), null, "Mean relative density. Target ≥ 0.95.", "gauge-dens-fill"),
    has("rho_final_std") && _gauge("ρ std", num("rho_final_std", 3), _clampPct(res.rho_final_std, 0, 0.15), null, "Density spread (lower = more uniform).", "gauge-dens-fill"),
  ]) : "";
  const D = has("z_shrink_pct") ? _group("Shrinkage & warp", [
    _txt("Z / XY shrink", `${num("z_shrink_pct", 1)}% / ${num("xy_shrink_pct", 1)}%`, "Linear sinter shrinkage (Z-dominant)."),
    has("warp_std_pct") && _gauge("warp", `${num("warp_std_pct", 2)}%`, _clampPct(warp, 0, 10), warpColor, "Column-to-column shrink scatter (0 = flat).", null),
    has("layer_multiplier") && _txt("green layers", `${num("green_layers", 0)} · ×${num("layer_multiplier", 2)}`, "Green layers needed to compact to the final height."),
  ]) : "";
  const E = _group("Run", [
    _txt("mode", `${res.fgm || "?"}${res.densify ? " · densify" : ""} · n=${res.grid_n || "?"}`, "FGM mode / densification / grid size."),
    has("solve_s") && _txt("solve time", `${num("solve_s", 1)} s`, "Wall-clock solve time."),
  ]);
  const g = $("resultsGrid"); g.className = "h3d-metrics";
  g.innerHTML = (A + B + C + D + E) || `<span class="k">—</span><span class="v">run a simulation</span>`;
}

// ── per-run views: layer slices + summary plots ─────────────────────────────
const FIELD_UNITS = {
  sat: "dopant\nfraction", T_phi90: "°C", phi_final: "melt\nfraction",
  rho_final: "rel.\ndensity", Qrf: "W/m³",
};
let slicePlayTimer = null;
async function loadRunViews(id) {
  const views = $("h3dViews");
  try {
    const meta = await (await fetch(`/api/heatr3d/fields?id=${encodeURIComponent(id)}`)).json();
    const fields = Object.keys(meta.fields || {});
    if (fields.length) {
      const sel = $("sliceField"); sel.innerHTML = "";
      for (const f of fields) {
        const o = document.createElement("option");
        o.value = f; o.textContent = meta.fields[f].label || f; sel.appendChild(o);
      }
      const nz = meta.dims[2], zr = $("sliceZ");
      zr.min = 0; zr.max = nz - 1; zr.value = Math.floor(nz / 2);
      const fmtNum = (v) => (Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-2)) ? v.toExponential(2) : v.toPrecision(4);
      const showSlice = () => {
        const f = sel.value, k = parseInt(zr.value, 10), fi = meta.fields[f];
        $("sliceImg").src = `/api/heatr3d/slice?id=${encodeURIComponent(id)}&field=${f}&k=${k}`;
        $("sliceLabel").textContent = `${f} · layer ${k} / ${nz - 1}`;
        $("cbMax").textContent = fmtNum(fi.max);
        $("cbMin").textContent = fmtNum(fi.min);
        $("cbUnits").textContent = FIELD_UNITS[f] || "";
      };
      const step = (d) => { zr.value = Math.max(0, Math.min(nz - 1, parseInt(zr.value, 10) + d)); showSlice(); };
      sel.onchange = showSlice; zr.oninput = showSlice;
      $("slicePrev").onclick = () => step(-1);
      $("sliceNext").onclick = () => step(1);
      $("slicePlay").onclick = () => {
        if (slicePlayTimer) { clearInterval(slicePlayTimer); slicePlayTimer = null; $("slicePlay").textContent = "▶"; }
        else { $("slicePlay").textContent = "‖"; slicePlayTimer = setInterval(() => {
          zr.value = (parseInt(zr.value, 10) + 1) % nz; showSlice(); }, 180); }
      };
      zr.onkeydown = (e) => { if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
                              else if (e.key === "ArrowRight") { step(1); e.preventDefault(); } };
      showSlice();
      views.style.display = "";
    }
  } catch (e) { /* no slices for this run */ }
  // F5: post-sinter warped geometry (densify runs only). Enable the viewport toggle if present.
  lastRunId = id; lastWarp = null;
  const wt = $("warpToggle"), wr = $("warpRow");
  if (wt) { wt.checked = false; wt.disabled = true; }
  try {
    const w = await (await fetch(`/api/heatr3d/warp?id=${encodeURIComponent(id)}`)).json();
    if (w && Array.isArray(w.warped_xyz_mm) && w.warped_xyz_mm.length) {
      lastWarp = w;
      if (wt) wt.disabled = false;
      if (wr) { wr.style.display = ""; const dm = $("warpMax"); if (dm) dm.textContent = `max ${(w.disp_max_mm).toFixed(2)} mm`; }
    } else if (wr) { wr.style.display = "none"; }
  } catch (e) { if (wr) wr.style.display = "none"; }
  const plots = ["melt_vs_cad", "ortho_slices", "melt_progression", "fgm_z_profile", "temperature_hist", "density_hist"];
  const gal = $("plotsGallery"); gal.innerHTML = "";
  for (const name of plots) {
    const img = document.createElement("img");
    img.src = `/files/outputs_eqs/_heatr3d/${encodeURIComponent(id)}/plots/${name}.png`;
    img.alt = name; img.loading = "lazy";
    img.onerror = () => img.remove();   // plot not generated for this run
    gal.appendChild(img);
  }
}

// Reopen a PAST run in the full 3D viewer (geometry + metrics + slices + plots + warp).
// Uses the status endpoint's on-disk fallback, which returns geometry+results for finished runs.
async function loadPastRun(id) {
  if (!id) return;
  try {
    statusLine.textContent = `loading run ${id}…`;
    const s = await (await fetch(`/api/heatr3d/status?id=${encodeURIComponent(id)}`)).json();
    if (!s.geometry) { statusLine.textContent = `run ${id} has no geometry on disk`; return; }
    nominalGeom = s.geometry;
    $("colorBy").value = s.geometry.surface_sat ? "sat" : "geom";
    if ($("warpToggle")) $("warpToggle").checked = false;
    applyGeomView();
    if (s.results) showResults(s.results);
    setProg(100);
    statusLine.textContent = `loaded run ${id}`;
    const sel = $("pastRun"); if (sel && sel.value !== id) sel.value = id;
    loadRunViews(id);
  } catch (e) { statusLine.textContent = "failed to load run: " + e.message; }
}

async function loadRunList() {
  try {
    const runs = await (await fetch("/api/heatr3d/runs")).json();
    const sel = $("pastRun");
    while (sel.options.length > 1) sel.remove(1);
    for (const r of runs) {
      const o = document.createElement("option");
      o.value = r.id;
      const sig = (r.sigma_T != null) ? ` · σT ${Number(r.sigma_T).toFixed(1)}` : "";
      o.textContent = `${r.shape || "?"} · ${r.fgm || "none"}${r.densify ? "+dens" : ""} · n${r.grid_n || "?"}${sig}`;
      sel.appendChild(o);
    }
  } catch (e) { /* no run list */ }
}

// ── wire up ───────────────────────────────────────────────────────────────
$("srcSel").addEventListener("change", () => {
  const stl = $("srcSel").value === "stl";
  $("stlBlock").style.display = stl ? "" : "none";
  $("paramBlock").style.display = stl ? "none" : "";
});
$("colorBy").addEventListener("change", applyGeomView);
if ($("warpToggle")) $("warpToggle").addEventListener("change", applyGeomView);
$("previewBtn").addEventListener("click", preview);
$("runBtn").addEventListener("click", run);
$("pastRun").addEventListener("change", () => loadPastRun($("pastRun").value));
initScene();
loadRunList();
const _qid = new URLSearchParams(location.search).get("id");
if (_qid) loadPastRun(_qid); else preview();
