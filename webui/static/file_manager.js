const byId = (id) => document.getElementById(id);

const state = {
  cwd: ".",
  selectedPath: "",
  selectedIsDir: false,
  cwdBackfillCapabilities: [],
  itemByPath: new Map(),
};

const TEXT_PREVIEW_EXT = new Set([
  "txt", "md", "json", "yaml", "yml", "log", "csv", "tsv", "py", "js", "ts",
  "html", "css", "sh", "toml", "ini", "cfg", "xml",
]);
const IMAGE_PREVIEW_EXT = new Set(["png", "jpg", "jpeg", "gif", "bmp", "webp", "svg"]);

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

function fmtBytes(n) {
  const v = Number(n || 0);
  if (!Number.isFinite(v)) return "-";
  if (v < 1024) return `${v} B`;
  if (v < 1024 * 1024) return `${(v / 1024).toFixed(1)} KB`;
  if (v < 1024 * 1024 * 1024) return `${(v / (1024 * 1024)).toFixed(1)} MB`;
  return `${(v / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function setStatus(msg, isError = false) {
  const el = byId("fmStatus");
  if (!el) return;
  el.textContent = msg || "";
  el.style.color = isError ? "#d44949" : "";
}

function setSelected(path, isDir) {
  state.selectedPath = String(path || "");
  state.selectedIsDir = !!isDir;
  const sel = byId("fmSelected");
  if (sel) sel.textContent = `Selected: ${state.selectedPath || "(none)"}`;
  renderPreview();
  updateBackfillButtonVisibility();
}

function rowNameLabel(item) {
  return item.is_dir ? `${item.name}/` : item.name;
}

function extOf(path) {
  const s = String(path || "");
  const idx = s.lastIndexOf(".");
  if (idx < 0) return "";
  return s.slice(idx + 1).toLowerCase();
}

function clearPreview(msg = "No file selected.") {
  const meta = byId("fmPreviewMeta");
  const root = byId("fmPreviewRoot");
  if (meta) meta.textContent = msg;
  if (root) root.innerHTML = "";
}

function isBackfillCandidate() {
  const hasCaps = (arr) => Array.isArray(arr) && arr.length > 0;
  if (state.selectedPath) {
    if (state.selectedIsDir) {
      const it = state.itemByPath.get(state.selectedPath);
      return !!(it && hasCaps(it.backfill_capabilities));
    }
    return hasCaps(state.cwdBackfillCapabilities);
  }
  return hasCaps(state.cwdBackfillCapabilities);
}

function currentBackfillCapabilities() {
  if (state.selectedPath && state.selectedIsDir) {
    const it = state.itemByPath.get(state.selectedPath);
    return Array.isArray(it?.backfill_capabilities) ? it.backfill_capabilities : [];
  }
  return Array.isArray(state.cwdBackfillCapabilities) ? state.cwdBackfillCapabilities : [];
}

function updateBackfillButtonVisibility() {
  const btn = byId("fmBackfillBtn");
  const advBtn = byId("fmBackfillAdvBtn");
  if (!btn) return;
  const ok = isBackfillCandidate();
  btn.disabled = !ok;
  btn.style.opacity = ok ? "1" : "0.45";
  btn.title = ok ? "Run report backfill for the selected/current run folder." : "Backfill is available only for run folders with compatible modules.";
  if (advBtn) {
    advBtn.disabled = !ok;
    advBtn.style.opacity = ok ? "1" : "0.45";
    advBtn.title = btn.title;
  }
}

function buildFileUrl(path) {
  return `/files/${encodeURIComponent(path).replaceAll("%2F", "/")}`;
}

async function renderPreview() {
  if (!state.selectedPath) {
    clearPreview("No file selected.");
    return;
  }
  if (state.selectedIsDir) {
    clearPreview("Folder selected. Double-click to enter.");
    return;
  }
  const path = state.selectedPath;
  const ext = extOf(path);
  const meta = byId("fmPreviewMeta");
  const root = byId("fmPreviewRoot");
  if (!root || !meta) return;
  meta.textContent = `Previewing: ${path}`;
  root.innerHTML = "";
  const url = buildFileUrl(path);

  if (IMAGE_PREVIEW_EXT.has(ext)) {
    const img = document.createElement("img");
    img.className = "file-preview-image";
    img.src = `${url}?t=${Date.now()}`;
    img.alt = path;
    root.appendChild(img);
    return;
  }

  if (ext === "pdf") {
    const frame = document.createElement("iframe");
    frame.className = "file-preview-frame";
    frame.src = url;
    frame.title = path;
    root.appendChild(frame);
    return;
  }

  if (TEXT_PREVIEW_EXT.has(ext)) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const pre = document.createElement("pre");
      pre.className = "file-preview-text";
      const maxChars = 250000;
      pre.textContent = text.length > maxChars ? `${text.slice(0, maxChars)}\n\n... [truncated]` : text;
      root.appendChild(pre);
      return;
    } catch (err) {
      clearPreview(`Preview error: ${err.message}`);
      return;
    }
  }

  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = "Open file in new tab";
  root.appendChild(link);
  const hint = document.createElement("div");
  hint.className = "muted";
  hint.textContent = "Binary/unsupported preview type.";
  root.appendChild(hint);
}

function renderRows(items) {
  const root = byId("fmRows");
  if (!root) return;
  root.innerHTML = "";
  if (!items.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="4" class="muted">No files in this folder.</td>`;
    root.appendChild(tr);
    return;
  }

  items.forEach((item) => {
    const tr = document.createElement("tr");
    tr.className = "file-row";
    tr.dataset.path = item.path;
    tr.dataset.isdir = item.is_dir ? "1" : "0";
    tr.innerHTML = `
      <td class="file-name">${rowNameLabel(item)}</td>
      <td>${item.is_dir ? "dir" : "file"}</td>
      <td>${item.is_dir ? "-" : fmtBytes(item.size_bytes)}</td>
      <td>${item.mtime || ""}</td>
    `;
    tr.addEventListener("click", () => {
      document.querySelectorAll(".file-row.active").forEach((r) => r.classList.remove("active"));
      tr.classList.add("active");
      setSelected(item.path, !!item.is_dir);
    });
    tr.addEventListener("dblclick", () => {
      if (item.is_dir) {
        loadDir(item.path);
      }
    });
    root.appendChild(tr);
  });
}

async function loadDir(path = state.cwd) {
  try {
    const data = await fetchJson(`/api/fs/list?path=${encodeURIComponent(path || ".")}`);
    state.cwd = data.cwd || ".";
    state.cwdBackfillCapabilities = Array.isArray(data.cwd_backfill_capabilities) ? data.cwd_backfill_capabilities : [];
    byId("fmCwd").textContent = state.cwd;
    byId("fmPath").value = state.cwd;
    const items = Array.isArray(data.items) ? data.items : [];
    state.itemByPath = new Map(items.map((it) => [it.path, it]));
    renderRows(items);
    setSelected("", false);
    setStatus("");
    updateBackfillButtonVisibility();
  } catch (err) {
    setStatus(`List error: ${err.message}`, true);
  }
}

function parentPath(path) {
  const p = String(path || ".").replace(/^\/+|\/+$/g, "");
  if (!p || p === ".") return ".";
  const parts = p.split("/");
  parts.pop();
  return parts.length ? parts.join("/") : ".";
}

async function moveSelected() {
  if (!state.selectedPath) {
    setStatus("Pick a file or folder first.", true);
    return;
  }
  const dst = String(byId("fmMoveDst")?.value || "").trim();
  if (!dst) {
    setStatus("Destination path is required.", true);
    return;
  }
  try {
    await fetchJson("/api/fs/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ src: state.selectedPath, dst }),
    });
    setStatus("Move completed.");
    byId("fmMoveDst").value = "";
    await loadDir(state.cwd);
  } catch (err) {
    setStatus(`Move error: ${err.message}`, true);
  }
}

async function deleteSelected() {
  if (!state.selectedPath) {
    setStatus("Pick a file or folder first.", true);
    return;
  }
  const recursive = !!byId("fmRecursive")?.checked;
  const ok = window.confirm(`Delete ${state.selectedPath}?`);
  if (!ok) return;
  try {
    await fetchJson("/api/fs/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: state.selectedPath, recursive }),
    });
    setStatus("Delete completed.");
    await loadDir(state.cwd);
  } catch (err) {
    setStatus(`Delete error: ${err.message}`, true);
  }
}

async function backfillHere() {
  if (!isBackfillCandidate()) {
    setStatus("Backfill is only available for run folders with compatible modules.", true);
    return;
  }
  let target = state.cwd || ".";
  if (state.selectedPath) {
    target = state.selectedIsDir ? state.selectedPath : parentPath(state.selectedPath);
  }
  try {
    setStatus(`Backfill running in ${target} ...`);
    const data = await fetchJson("/api/tools/backfill-reports", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ output_dir: target, modules: [], mode: "quick" }),
    });
    setStatus(`Backfill queued (job ${data.job_id}).`);
    await loadDir(state.cwd);
  } catch (err) {
    setStatus(`Backfill error: ${err.message}`, true);
  }
}

function pickModulesInteractive(capabilities) {
  const caps = Array.isArray(capabilities) ? capabilities : [];
  if (!caps.length) return [];
  const defaultStr = caps.join(",");
  const ans = window.prompt(
    `Select backfill modules (comma-separated).\nAvailable:\n${caps.join("\n")}`,
    defaultStr,
  );
  if (ans === null) return [];
  const chosen = String(ans)
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  return chosen.filter((m) => caps.includes(m));
}

async function backfillHereAdvanced() {
  if (!isBackfillCandidate()) {
    setStatus("Backfill is not available for this folder.", true);
    return;
  }
  let target = state.cwd || ".";
  if (state.selectedPath) {
    target = state.selectedIsDir ? state.selectedPath : parentPath(state.selectedPath);
  }
  const caps = currentBackfillCapabilities();
  const chosen = pickModulesInteractive(caps);
  if (!chosen.length) return;
  try {
    setStatus(`Backfill running in ${target} ...`);
    const data = await fetchJson("/api/tools/backfill-reports", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ output_dir: target, modules: chosen, mode: "advanced" }),
    });
    setStatus(`Backfill queued (job ${data.job_id}).`);
    await loadDir(state.cwd);
  } catch (err) {
    setStatus(`Backfill error: ${err.message}`, true);
  }
}

function bindEvents() {
  byId("fmRefreshBtn")?.addEventListener("click", () => loadDir(state.cwd));
  byId("fmUpBtn")?.addEventListener("click", () => loadDir(parentPath(state.cwd)));
  byId("fmPath")?.addEventListener("change", () => loadDir(byId("fmPath").value));
  byId("fmMoveBtn")?.addEventListener("click", moveSelected);
  byId("fmDeleteBtn")?.addEventListener("click", deleteSelected);
  byId("fmBackfillBtn")?.addEventListener("click", backfillHere);
  byId("fmBackfillAdvBtn")?.addEventListener("click", backfillHereAdvanced);
}

function init() {
  bindEvents();
  loadDir(".");
}

init();
