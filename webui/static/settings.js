const HEATR_DEFAULTS = {
  mode: "dark",
  accent: "blue",
  bgstyle: "grid",
  layout: "layout-1",
};

const HEATR_SECRET_KEY = "heatr_secret_spacex";
const HEATR_THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js";
const HEATR_MARS_COLOR_URL = "https://threejs.org/examples/textures/planets/mars_1k_color.jpg";
const HEATR_MARS_NORMAL_URL = "https://threejs.org/examples/textures/planets/mars_1k_normal.jpg";

let _heatrShiftSAt = 0;
let _heatrMarsLayer = null;
let _heatrMarsMounting = false;
let _heatrThreePromise = null;

function heatrClamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function heatrLoadPrefs() {
  try {
    const raw = localStorage.getItem("heatr_ui_prefs");
    if (!raw) return { ...HEATR_DEFAULTS };
    const parsed = JSON.parse(raw);
    return { ...HEATR_DEFAULTS, ...parsed };
  } catch {
    return { ...HEATR_DEFAULTS };
  }
}

function heatrSavePrefs(prefs) {
  localStorage.setItem("heatr_ui_prefs", JSON.stringify(prefs));
}

function heatrSecretEnabled() {
  try {
    return localStorage.getItem(HEATR_SECRET_KEY) === "1";
  } catch {
    return false;
  }
}

function heatrSetSecretEnabled(on) {
  try {
    localStorage.setItem(HEATR_SECRET_KEY, on ? "1" : "0");
  } catch {
    // best-effort only
  }
}

function heatrFract(v) {
  return v - Math.floor(v);
}

function heatrHash2(x, y) {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453123;
  return heatrFract(s);
}

function heatrValueNoise2(x, y) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  const sx = tx * tx * (3 - 2 * tx);
  const sy = ty * ty * (3 - 2 * ty);
  const n00 = heatrHash2(x0, y0);
  const n10 = heatrHash2(x0 + 1, y0);
  const n01 = heatrHash2(x0, y0 + 1);
  const n11 = heatrHash2(x0 + 1, y0 + 1);
  const nx0 = n00 + (n10 - n00) * sx;
  const nx1 = n01 + (n11 - n01) * sx;
  return nx0 + (nx1 - nx0) * sy;
}

function heatrFbm2(x, y) {
  let amp = 0.5;
  let freq = 1;
  let sum = 0;
  for (let i = 0; i < 5; i += 1) {
    sum += amp * heatrValueNoise2(x * freq, y * freq);
    freq *= 2;
    amp *= 0.5;
  }
  return sum;
}

function heatrBuildMarsTexture(width, height) {
  const out = new Uint8ClampedArray(width * height * 3);
  const wrapDist = (a, b) => {
    let d = Math.abs(a - b);
    if (d > 0.5) d = 1 - d;
    return d;
  };
  const gauss = (d, s) => Math.exp(-(d * d) / (2 * s * s));

  for (let y = 0; y < height; y += 1) {
    const v = y / (height - 1);
    const lat = (v - 0.5) * Math.PI;
    const latFade = Math.cos(lat);
    for (let x = 0; x < width; x += 1) {
      const u = x / (width - 1);
      const lon = u * Math.PI * 2;
      const nx = Math.cos(lon) * latFade;
      const ny = Math.sin(lat);
      const nz = Math.sin(lon) * latFade;
      const base = heatrFbm2(u * 4.2 + 3.0, v * 3.8 + 7.0);
      const ridges = Math.abs(heatrValueNoise2(u * 13.5 + 19.0, v * 10.5 + 4.0) - 0.5) * 2.0;
      const dust = heatrValueNoise2(u * 24.0 + 13.0, v * 20.0 + 11.0);
      const dunes = heatrValueNoise2(u * 42.0 + 3.0, v * 14.0 + 27.0);
      let h = base * 0.72 + ridges * 0.22 + dust * 0.06;
      h = heatrClamp(h, 0, 1);

      let r = 96 + h * 118;
      let g = 45 + h * 55;
      let b = 29 + h * 32;

      const canyon = gauss(wrapDist(u, 0.58), 0.045) * gauss(Math.abs(v - 0.52), 0.03);
      const syrtis = gauss(wrapDist(u, 0.28), 0.06) * gauss(Math.abs(v - 0.43), 0.06);
      const brightBasin = gauss(wrapDist(u, 0.75), 0.08) * gauss(Math.abs(v - 0.38), 0.07);

      r -= 30 * canyon + 24 * syrtis;
      g -= 18 * canyon + 16 * syrtis;
      b -= 10 * canyon + 9 * syrtis;
      r += 20 * brightBasin;
      g += 13 * brightBasin;
      b += 8 * brightBasin;

      const dustBands = Math.sin((u * Math.PI * 2 * 3.0) + (v * 10.0)) * 0.5 + 0.5;
      r += dunes * 12 + dustBands * 8;
      g += dunes * 4 + dustBands * 2;
      b += dunes * 2;

      const polar = Math.max(0, Math.abs(ny) - 0.82) / 0.18;
      if (polar > 0) {
        r = r * (1 - polar * 0.7) + 198 * polar * 0.7;
        g = g * (1 - polar * 0.7) + 170 * polar * 0.7;
        b = b * (1 - polar * 0.7) + 150 * polar * 0.7;
      }

      const orient = 0.5 + 0.5 * (0.7 * nx - 0.15 * nz);
      r *= 0.82 + orient * 0.25;
      g *= 0.82 + orient * 0.2;
      b *= 0.84 + orient * 0.15;

      const idx = (y * width + x) * 3;
      out[idx] = heatrClamp(Math.round(r), 0, 255);
      out[idx + 1] = heatrClamp(Math.round(g), 0, 255);
      out[idx + 2] = heatrClamp(Math.round(b), 0, 255);
    }
  }

  return out;
}

function heatrBuildMarsPbrTextures(THREE, width, height) {
  const color = heatrBuildMarsTexture(width, height);
  const h = new Float32Array(width * height);
  for (let i = 0, j = 0; i < h.length; i += 1, j += 3) {
    const lum = (0.2126 * color[j] + 0.7152 * color[j + 1] + 0.0722 * color[j + 2]) / 255;
    const extra = heatrValueNoise2((i % width) * 0.05 + 11.0, Math.floor(i / width) * 0.05 + 17.0) * 0.12;
    h[i] = heatrClamp(lum * 0.88 + extra, 0, 1);
  }

  const colorRgba = new Uint8Array(width * height * 4);
  const normalRgba = new Uint8Array(width * height * 4);
  const roughRgba = new Uint8Array(width * height * 4);
  const dispRgba = new Uint8Array(width * height * 4);
  const idxWrap = (x, y) => {
    const xx = (x + width) % width;
    const yy = Math.min(height - 1, Math.max(0, y));
    return yy * width + xx;
  };

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x;
      const j = i * 3;
      const o = i * 4;

      colorRgba[o] = color[j];
      colorRgba[o + 1] = color[j + 1];
      colorRgba[o + 2] = color[j + 2];
      colorRgba[o + 3] = 255;

      const hl = h[idxWrap(x - 1, y)];
      const hr = h[idxWrap(x + 1, y)];
      const hd = h[idxWrap(x, y - 1)];
      const hu = h[idxWrap(x, y + 1)];
      const sx = (hr - hl) * 2.8;
      const sy = (hu - hd) * 2.8;
      let nx = -sx;
      let ny = -sy;
      let nz = 1.0;
      const inv = 1 / Math.hypot(nx, ny, nz);
      nx *= inv;
      ny *= inv;
      nz *= inv;
      normalRgba[o] = Math.round((nx * 0.5 + 0.5) * 255);
      normalRgba[o + 1] = Math.round((ny * 0.5 + 0.5) * 255);
      normalRgba[o + 2] = Math.round((nz * 0.5 + 0.5) * 255);
      normalRgba[o + 3] = 255;

      const rough = heatrClamp(0.82 + h[i] * 0.15, 0.68, 0.98);
      const disp = heatrClamp((h[i] - 0.45) * 0.9 + 0.5, 0, 1);
      const rv = Math.round(rough * 255);
      const dv = Math.round(disp * 255);
      roughRgba[o] = rv;
      roughRgba[o + 1] = rv;
      roughRgba[o + 2] = rv;
      roughRgba[o + 3] = 255;
      dispRgba[o] = dv;
      dispRgba[o + 1] = dv;
      dispRgba[o + 2] = dv;
      dispRgba[o + 3] = 255;
    }
  }

  const mk = (arr) => {
    const t = new THREE.DataTexture(arr, width, height, THREE.RGBAFormat);
    t.needsUpdate = true;
    t.wrapS = THREE.RepeatWrapping;
    t.wrapT = THREE.ClampToEdgeWrapping;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.magFilter = THREE.LinearFilter;
    return t;
  };

  return {
    map: mk(colorRgba),
    normalMap: mk(normalRgba),
    roughnessMap: mk(roughRgba),
    displacementMap: mk(dispRgba),
  };
}

function heatrBuildMarsSphereBuffers(sizePx, dpr) {
  const px = Math.max(96, Math.round(sizePx * dpr));
  const radius = px * 0.5 - 1;
  const total = px * px;
  const nx = new Float32Array(total);
  const ny = new Float32Array(total);
  const nz = new Float32Array(total);
  const lon0 = new Float32Array(total);
  const latV = new Float32Array(total);
  const alpha = new Uint8ClampedArray(total);

  for (let y = 0; y < px; y += 1) {
    const fy = (y + 0.5 - px * 0.5) / radius;
    for (let x = 0; x < px; x += 1) {
      const fx = (x + 0.5 - px * 0.5) / radius;
      const i = y * px + x;
      const r2 = fx * fx + fy * fy;
      if (r2 > 1) {
        alpha[i] = 0;
        continue;
      }
      const fz = Math.sqrt(Math.max(0, 1 - r2));
      nx[i] = fx;
      ny[i] = fy;
      nz[i] = fz;
      lon0[i] = Math.atan2(fx, fz);
      latV[i] = 0.5 - Math.asin(fy) / Math.PI;
      const edge = heatrClamp((1 - Math.sqrt(r2)) * 18.0, 0, 1);
      alpha[i] = Math.round(248 * edge);
    }
  }

  return { px, nx, ny, nz, lon0, latV, alpha };
}

function heatrCreateMarsLayerProcedural() {
  const canvas = document.createElement("canvas");
  canvas.id = "spacexMarsOrb";
  canvas.setAttribute("aria-hidden", "true");
  canvas.style.position = "fixed";
  canvas.style.right = "2rem";
  canvas.style.bottom = "1.5rem";
  canvas.style.width = "clamp(165px, 20vw, 300px)";
  canvas.style.height = "clamp(165px, 20vw, 300px)";
  canvas.style.pointerEvents = "none";
  canvas.style.zIndex = "0";
  canvas.style.opacity = "0.7";
  canvas.style.filter = "saturate(0.92) contrast(1.06)";
  canvas.style.clipPath = "circle(49.5% at 50% 50%)";
  canvas.style.borderRadius = "50%";
  canvas.style.background = "transparent";
  document.body.appendChild(canvas);

  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) {
    canvas.remove();
    return null;
  }

  const texW = 1024;
  const texH = 512;
  const texture = heatrBuildMarsTexture(texW, texH);

  const layer = {
    type: "procedural",
    canvas,
    ctx,
    texW,
    texH,
    texture,
    buffers: null,
    frame: 0,
    lastMs: 0,
    resize: null,
    dispose: null,
  };

  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
    const side = Math.max(96, Math.round(Math.min(rect.width, rect.height)));
    layer.buffers = heatrBuildMarsSphereBuffers(side, dpr);
    canvas.width = layer.buffers.px;
    canvas.height = layer.buffers.px;
  };

  const render = (ms) => {
    if (!_heatrMarsLayer || _heatrMarsLayer !== layer) return;
    if (!layer.lastMs) layer.lastMs = ms;
    if (ms - layer.lastMs < 42) {
      layer.frame = requestAnimationFrame(render);
      return;
    }
    layer.lastMs = ms;

    const { ctx: c, buffers: b } = layer;
    if (!b) {
      layer.frame = requestAnimationFrame(render);
      return;
    }

    const img = c.createImageData(b.px, b.px);
    const out = img.data;
    const rot = (ms * 0.000095) % (Math.PI * 2);
    const lx = -0.36;
    const ly = -0.2;
    const lz = 0.91;

    for (let i = 0; i < b.alpha.length; i += 1) {
      const a = b.alpha[i];
      const o = i * 4;
      if (!a) {
        out[o + 3] = 0;
        continue;
      }

      let u = b.lon0[i] + rot;
      u = u / (Math.PI * 2) + 0.5;
      u -= Math.floor(u);
      const tx = Math.min(layer.texW - 1, Math.max(0, Math.floor(u * (layer.texW - 1))));
      const ty = Math.min(layer.texH - 1, Math.max(0, Math.floor(b.latV[i] * (layer.texH - 1))));
      const ti = (ty * layer.texW + tx) * 3;
      const tr = layer.texture[ti];
      const tg = layer.texture[ti + 1];
      const tb = layer.texture[ti + 2];

      const ndl = Math.max(0, b.nx[i] * lx + b.ny[i] * ly + b.nz[i] * lz);
      const ambient = 0.22;
      const diff = 0.88 * ndl;
      const rim = Math.pow(1 - b.nz[i], 2.1) * 0.12;
      const shade = heatrClamp(ambient + diff + rim, 0.08, 1.2);

      out[o] = heatrClamp(Math.round(tr * shade), 0, 255);
      out[o + 1] = heatrClamp(Math.round(tg * shade), 0, 255);
      out[o + 2] = heatrClamp(Math.round(tb * shade), 0, 255);
      out[o + 3] = a;
    }

    c.clearRect(0, 0, b.px, b.px);
    c.putImageData(img, 0, 0);

    const glow = c.createRadialGradient(b.px * 0.52, b.px * 0.53, b.px * 0.3, b.px * 0.52, b.px * 0.53, b.px * 0.54);
    glow.addColorStop(0, "rgba(255, 180, 130, 0)");
    glow.addColorStop(1, "rgba(255, 115, 70, 0.03)");
    c.globalCompositeOperation = "lighter";
    c.fillStyle = glow;
    c.beginPath();
    c.arc(b.px * 0.5, b.px * 0.5, b.px * 0.62, 0, Math.PI * 2);
    c.fill();
    c.globalCompositeOperation = "source-over";

    layer.frame = requestAnimationFrame(render);
  };

  resize();
  window.addEventListener("resize", resize);
  layer.resize = resize;
  layer.frame = requestAnimationFrame(render);
  layer.dispose = () => {
    cancelAnimationFrame(layer.frame);
    window.removeEventListener("resize", resize);
    canvas.remove();
  };

  return layer;
}

function heatrEnsureThree() {
  if (window.THREE) return Promise.resolve(window.THREE);
  if (_heatrThreePromise) return _heatrThreePromise;
  _heatrThreePromise = new Promise((resolve) => {
    const existing = document.querySelector("script[data-heatr-three='1']");
    if (existing) {
      existing.addEventListener("load", () => resolve(window.THREE || null), { once: true });
      existing.addEventListener("error", () => resolve(null), { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = HEATR_THREE_CDN;
    script.async = true;
    script.dataset.heatrThree = "1";
    script.onload = () => resolve(window.THREE || null);
    script.onerror = () => resolve(null);
    document.head.appendChild(script);
  });
  return _heatrThreePromise;
}

async function heatrCreateMarsLayerThree() {
  const THREE = await heatrEnsureThree();
  if (!THREE || !document.body) return null;

  const host = document.createElement("div");
  host.id = "spacexMarsOrb";
  host.setAttribute("aria-hidden", "true");
  host.style.position = "fixed";
  host.style.right = "2rem";
  host.style.bottom = "1.5rem";
  host.style.width = "clamp(165px, 20vw, 300px)";
  host.style.height = "clamp(165px, 20vw, 300px)";
  host.style.pointerEvents = "none";
  host.style.zIndex = "0";
  host.style.opacity = "0.74";
  host.style.filter = "saturate(0.95) contrast(1.08)";
  host.style.borderRadius = "50%";
  host.style.overflow = "hidden";
  document.body.appendChild(host);

  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: "low-power" });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace || renderer.outputColorSpace;
  host.appendChild(renderer.domElement);
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  renderer.domElement.style.display = "block";
  renderer.domElement.style.background = "transparent";

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
  camera.position.set(0, 0, 3.2);

  const ambient = new THREE.AmbientLight(0xffffff, 0.36);
  const key = new THREE.DirectionalLight(0xffe6d6, 1.24);
  key.position.set(2.2, 0.7, 2.8);
  const fill = new THREE.DirectionalLight(0xa9c6ff, 0.18);
  fill.position.set(-2.4, -0.5, -1.8);
  scene.add(ambient, key, fill);

  const geometry = new THREE.SphereGeometry(1, 96, 96);
  const loader = new THREE.TextureLoader();
  loader.setCrossOrigin("anonymous");

  const loadTexture = (url) => new Promise((resolve) => {
    loader.load(url, resolve, undefined, () => resolve(null));
  });

  const [map, normalMap] = await Promise.all([
    loadTexture(HEATR_MARS_COLOR_URL),
    loadTexture(HEATR_MARS_NORMAL_URL),
  ]);

  let resolvedMap = map;
  let resolvedNormalMap = normalMap;
  const generated = heatrBuildMarsPbrTextures(THREE, 1024, 512);
  const maxAniso = Math.min(8, renderer.capabilities.getMaxAnisotropy());

  if (!resolvedMap) {
    resolvedMap = generated.map;
  }
  if (!resolvedNormalMap) {
    resolvedNormalMap = generated.normalMap;
  }

  if (resolvedMap) {
    resolvedMap.colorSpace = THREE.SRGBColorSpace || resolvedMap.colorSpace;
    resolvedMap.anisotropy = maxAniso;
  }
  if (resolvedNormalMap) {
    resolvedNormalMap.anisotropy = maxAniso;
  }
  const resolvedRoughnessMap = generated.roughnessMap;
  const resolvedDisplacementMap = generated.displacementMap;
  if (resolvedRoughnessMap) resolvedRoughnessMap.anisotropy = maxAniso;
  if (resolvedDisplacementMap) resolvedDisplacementMap.anisotropy = maxAniso;

  const material = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    map: resolvedMap,
    normalMap: resolvedNormalMap,
    normalScale: new THREE.Vector2(1.15, 1.15),
    roughnessMap: resolvedRoughnessMap,
    roughness: 0.9,
    displacementMap: resolvedDisplacementMap,
    displacementScale: resolvedDisplacementMap ? 0.05 : 0.0,
    metalness: 0.02,
    transparent: true,
    opacity: 0.92,
  });

  const sphere = new THREE.Mesh(geometry, material);
  sphere.rotation.x = 0.22;
  scene.add(sphere);

  const atmo = new THREE.Mesh(
    new THREE.SphereGeometry(1.03, 64, 64),
    new THREE.MeshBasicMaterial({
      color: 0xff8d5f,
      transparent: true,
      opacity: 0.06,
      side: THREE.BackSide,
    }),
  );
  scene.add(atmo);

  const resize = () => {
    const rect = host.getBoundingClientRect();
    const w = Math.max(64, Math.round(rect.width));
    const h = Math.max(64, Math.round(rect.height));
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  };

  let frame = 0;
  const animate = () => {
    if (!_heatrMarsLayer) return;
    sphere.rotation.y += 0.0016;
    atmo.rotation.y += 0.0013;
    renderer.render(scene, camera);
    frame = requestAnimationFrame(animate);
  };

  resize();
  window.addEventListener("resize", resize);
  frame = requestAnimationFrame(animate);

  return {
    type: "three",
    dispose: () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      geometry.dispose();
      material.dispose();
      atmo.geometry.dispose();
      atmo.material.dispose();
      if (resolvedMap) resolvedMap.dispose();
      if (resolvedNormalMap) resolvedNormalMap.dispose();
      if (resolvedRoughnessMap) resolvedRoughnessMap.dispose();
      if (resolvedDisplacementMap) resolvedDisplacementMap.dispose();
      renderer.dispose();
      host.remove();
    },
  };
}

function heatrMountMarsLayer() {
  if (_heatrMarsLayer || _heatrMarsMounting || !document.body) return;
  _heatrMarsMounting = true;

  heatrCreateMarsLayerThree()
    .then((layer) => {
      if (!heatrSecretEnabled()) {
        if (layer && layer.dispose) layer.dispose();
        return;
      }
      if (layer) {
        _heatrMarsLayer = layer;
        return;
      }
      _heatrMarsLayer = heatrCreateMarsLayerProcedural();
    })
    .catch(() => {
      if (!heatrSecretEnabled()) return;
      _heatrMarsLayer = heatrCreateMarsLayerProcedural();
    })
    .finally(() => {
      _heatrMarsMounting = false;
    });
}

function heatrUnmountMarsLayer() {
  if (!_heatrMarsLayer) return;
  if (_heatrMarsLayer.dispose) _heatrMarsLayer.dispose();
  _heatrMarsLayer = null;
}

function heatrApplyPrefs(prefs) {
  document.body.setAttribute("data-mode", prefs.mode);
  document.body.setAttribute("data-accent", prefs.accent);
  document.body.setAttribute("data-bgstyle", prefs.bgstyle);
  document.body.setAttribute("data-layout", prefs.layout);
  heatrApplySecretTheme();
}

function heatrApplySecretTheme() {
  if (!document.body) return;
  const active = heatrSecretEnabled();
  document.body.classList.toggle("spacex-mode", active);

  const modeSel = document.getElementById("prefMode");
  if (active) {
    if (!document.body.dataset.preSpacexMode) {
      document.body.dataset.preSpacexMode = document.body.getAttribute("data-mode") || "dark";
    }
    document.body.setAttribute("data-mode", "dark");
    if (modeSel) modeSel.value = "dark";
    heatrMountMarsLayer();
  } else if (document.body.dataset.preSpacexMode) {
    const restoreMode = document.body.dataset.preSpacexMode;
    document.body.setAttribute("data-mode", restoreMode);
    if (modeSel) modeSel.value = restoreMode;
    delete document.body.dataset.preSpacexMode;
    heatrUnmountMarsLayer();
  } else {
    heatrUnmountMarsLayer();
  }

  const s = document.getElementById("serverStatus");
  if (s) {
    if (active) s.textContent = "Mission control linked";
    else if (s.textContent === "Mission control linked") s.textContent = "HEATR service connected";
  }
}

function heatrBindSecretToggle() {
  window.addEventListener("keydown", (ev) => {
    const key = String(ev.key || "").toLowerCase();
    const now = Date.now();
    if (ev.shiftKey && key === "s") {
      _heatrShiftSAt = now;
      return;
    }
    if (ev.shiftKey && key === "x" && now - _heatrShiftSAt <= 1200) {
      const next = !heatrSecretEnabled();
      heatrSetSecretEnabled(next);
      heatrApplySecretTheme();
      _heatrShiftSAt = 0;
    }
  });
}

function heatrInitSettings() {
  const prefs = heatrLoadPrefs();
  heatrApplyPrefs(prefs);
  heatrBindSecretToggle();

  const modeSel = document.getElementById("prefMode");
  const accentSel = document.getElementById("prefAccent");
  const bgSel = document.getElementById("prefBackground");
  const layoutSel = document.getElementById("prefLayout");

  if (modeSel) modeSel.value = prefs.mode;
  if (accentSel) accentSel.value = prefs.accent;
  if (bgSel) bgSel.value = prefs.bgstyle;
  if (layoutSel) layoutSel.value = prefs.layout;
  heatrApplySecretTheme();

  const onChange = () => {
    const next = {
      mode: modeSel?.value || prefs.mode,
      accent: accentSel?.value || prefs.accent,
      bgstyle: bgSel?.value || prefs.bgstyle,
      layout: layoutSel?.value || prefs.layout,
    };
    heatrApplyPrefs(next);
    heatrSavePrefs(next);
  };

  [modeSel, accentSel, bgSel, layoutSel].forEach((el) => {
    if (el) el.addEventListener("change", onChange);
  });
}

heatrInitSettings();
