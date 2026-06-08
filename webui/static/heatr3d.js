// HEATR-3D page — geometry preview + run via heatr3d_job subprocess + results.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const statusLine = $("statusLine");
const progBar = $("progBar");

// ── three.js scene ──────────────────────────────────────────────────────────
const vp = $("h3dViewport");
let renderer, scene, camera, controls, partMesh = null, electrodes = null;
let lastGeom = null;

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
  const pts = geom.surface_xyz_mm, hmm = geom.h_mm, sat = geom.surface_sat || null;
  const box = new THREE.BoxGeometry(hmm, hmm, hmm);
  const mat = new THREE.MeshLambertMaterial({ vertexColors: (colorBy === "sat" && sat) });
  if (!(colorBy === "sat" && sat)) mat.color = new THREE.Color(0x4f9dff);
  const mesh = new THREE.InstancedMesh(box, mat, pts.length);
  const m = new THREE.Matrix4();
  for (let i = 0; i < pts.length; i++) {
    m.setPosition(pts[i][0], pts[i][1], pts[i][2]);   // x, y(field), z(build)
    mesh.setMatrixAt(i, m);
    if (colorBy === "sat" && sat) mesh.setColorAt(i, viridis(sat[i]));
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
    renderSurface(geom, $("colorBy").value);
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
    if (s.geometry) renderSurface(s.geometry, $("colorBy").value);
    if (s.done) {
      clearInterval(pollTimer); pollTimer = null;
      if (s.error) { statusLine.textContent = "run error: " + s.error; return; }
      setProg(100); statusLine.textContent = "done.";
      if (s.results) showResults(s.results);
      if (s.geometry) { $("colorBy").value = (cfg().fgm !== "none") ? "sat" : "geom"; renderSurface(s.geometry, $("colorBy").value); }
    }
  } catch (e) { /* keep polling */ }
}

const RESULT_KEYS = [
  ["sigma_T", "σ_T (°C)"], ["t_phi90_s", "t_φ90 (s)"], ["T_max_C", "T_max (°C)"],
  ["sintered_frac", "sintered frac"], ["dice", "Dice vs CAD"],
  ["rho_final_mean", "ρ̄ final"], ["rho_final_std", "ρ std"],
  ["z_shrink_pct", "Z shrink (%)"], ["xy_shrink_pct", "XY shrink (%)"],
  ["warp_std_pct", "warpage (%)"], ["green_layers", "green layers"],
  ["layer_multiplier", "layer ×"], ["extra_layers", "extra layers"], ["solve_s", "solve (s)"],
];
function showResults(res) {
  const g = $("resultsGrid"); g.innerHTML = "";
  for (const [k, label] of RESULT_KEYS) {
    if (res[k] === undefined) continue;
    const kk = document.createElement("span"); kk.className = "k"; kk.textContent = label;
    const vv = document.createElement("span"); vv.className = "v";
    vv.textContent = (typeof res[k] === "number") ? (Number.isInteger(res[k]) ? res[k] : res[k].toFixed(3)) : res[k];
    g.appendChild(kk); g.appendChild(vv);
  }
}

// ── wire up ───────────────────────────────────────────────────────────────
$("srcSel").addEventListener("change", () => {
  const stl = $("srcSel").value === "stl";
  $("stlBlock").style.display = stl ? "" : "none";
  $("paramBlock").style.display = stl ? "none" : "";
});
$("colorBy").addEventListener("change", () => { if (lastGeom) renderSurface(lastGeom, $("colorBy").value); });
$("previewBtn").addEventListener("click", preview);
$("runBtn").addEventListener("click", run);
initScene();
preview();
