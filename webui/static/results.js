const byId = (id) => document.getElementById(id);
const viewer = { items: [], index: 0 };
let RUNS = [];
let SELECTED_RUN = "";
const DETAIL_CACHE = new Map();

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function runBackfillForRun(runName, modules = [], mode = "quick") {
  const status = byId("serverStatus");
  if (status) status.textContent = "Backfill running...";
  const data = await fetchJson("/api/tools/backfill-reports", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ output_dir: `outputs_eqs/${runName}`, modules, mode }),
  });
  if (status) status.textContent = `Backfill queued: ${data.job_id}`;
  return data;
}

async function deleteRun(runName) {
  return fetchJson("/api/results/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_name: runName }),
  });
}

async function setRunStar(runName, starred) {
  return fetchJson("/api/results/star", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_name: runName, starred: Boolean(starred) }),
  });
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

function openViewer(items, index) {
  viewer.items = items;
  viewer.index = index;
  const dlg = byId("imageModal");
  const render = () => {
    const cur = viewer.items[viewer.index];
    byId("imageModalImg").src = cur.url;
    byId("imageModalTitle").textContent = cur.title;
  };
  byId("imagePrev").onclick = () => { viewer.index = (viewer.index - 1 + viewer.items.length) % viewer.items.length; render(); };
  byId("imageNext").onclick = () => { viewer.index = (viewer.index + 1) % viewer.items.length; render(); };
  byId("imageModalClose").onclick = () => dlg.close();
  render();
  if (!dlg.open) dlg.showModal();
}

function metricLine(ex) {
  const parts = [];
  if (ex.model_family) parts.push(`model ${String(ex.model_family)}`);
  if (ex.calibration_version) parts.push(`cal ${String(ex.calibration_version)}`);
  if (ex.ab_bucket_id) parts.push(`ab ${String(ex.ab_bucket_id)}`);
  if (ex.max_T_part_c !== undefined) parts.push(`Tmax ${Number(ex.max_T_part_c).toFixed(1)} C`);
  if (ex.mean_phi_part !== undefined) parts.push(`phi ${Number(ex.mean_phi_part).toFixed(3)}`);
  if (ex.mean_rho_rel_part !== undefined) parts.push(`rho ${Number(ex.mean_rho_rel_part).toFixed(3)}`);
  if (ex.t_final_s !== undefined) parts.push(`t ${Number(ex.t_final_s).toFixed(1)} s`);
  return parts.join(" | ") || "No summary metrics";
}

function renderGroupFilter(runs) {
  const sel = byId("runGroupFilter");
  const prev = sel.value;
  const groups = [...new Set(runs.map((r) => r.group || "runs"))].sort();
  sel.innerHTML = '<option value="">All groups</option>';
  groups.forEach((g) => {
    const o = document.createElement("option");
    o.value = g;
    o.textContent = g;
    sel.appendChild(o);
  });
  sel.value = groups.includes(prev) ? prev : "";
}

function renderModeFilter(runs) {
  const sel = byId("runModeFilter");
  if (!sel) return;
  const prev = sel.value;
  const modes = [...new Set(runs.map((r) => String(r.run_type || "unknown")))].sort();
  const modeLabel = (m) => {
    const k = String(m || "unknown");
    if (k === "single") return "Single Exposure";
    if (k === "sweep") return "Exposure Sweep";
    if (k === "optimizer") return "Exposure Optimizer";
    if (k === "turntable") return "Turntable Run";
    if (k === "orientation_optimizer") return "Orientation Optimizer";
    if (k === "placement_optimizer") return "Placement Optimizer";
    if (k === "shell_sweep") return "Shell Thickness Sweep";
    return k;
  };
  sel.innerHTML = '<option value="">All modes</option>';
  modes.forEach((m) => {
    const o = document.createElement("option");
    o.value = m;
    o.textContent = modeLabel(m);
    sel.appendChild(o);
  });
  sel.value = modes.includes(prev) ? prev : "";
}

function filterRuns() {
  const q = (byId("runSearch").value || "").trim().toLowerCase();
  const g = byId("runGroupFilter").value;
  const m = byId("runModeFilter")?.value || "";
  const star = byId("runStarFilter")?.value || "";
  const out = RUNS.filter((r) => {
    if (g && r.group !== g) return false;
    if (m && String(r.run_type || "unknown") !== m) return false;
    if (star === "starred" && !Boolean(r.starred)) return false;
    if (star === "unstarred" && Boolean(r.starred)) return false;
    if (q && !r.name.toLowerCase().includes(q)) return false;
    return true;
  });
  return sortRuns(out);
}

function numOrNegInf(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : Number.NEGATIVE_INFINITY;
}

function epochOrZero(v) {
  const t = Date.parse(String(v || ""));
  return Number.isFinite(t) ? t : 0;
}

function sortRuns(rows) {
  const mode = String(byId("runSort")?.value || "created_newest");
  const out = [...rows];
  const cmpName = (a, b) => String(a.name || "").localeCompare(String(b.name || ""));
  if (mode === "created_oldest") {
    out.sort((a, b) => epochOrZero(a.run_created_at) - epochOrZero(b.run_created_at) || cmpName(a, b));
  } else if (mode === "created_newest") {
    out.sort((a, b) => epochOrZero(b.run_created_at) - epochOrZero(a.run_created_at) || cmpName(a, b));
  } else if (mode === "oldest") {
    out.sort((a, b) => epochOrZero(a.updated_at) - epochOrZero(b.updated_at) || cmpName(a, b));
  } else if (mode === "rho_desc") {
    out.sort((a, b) =>
      numOrNegInf(b?.summary_excerpt?.mean_rho_rel_part) - numOrNegInf(a?.summary_excerpt?.mean_rho_rel_part)
      || epochOrZero(b.updated_at) - epochOrZero(a.updated_at)
      || cmpName(a, b));
  } else if (mode === "tmax_desc") {
    out.sort((a, b) =>
      numOrNegInf(b?.summary_excerpt?.max_T_part_c) - numOrNegInf(a?.summary_excerpt?.max_T_part_c)
      || epochOrZero(b.updated_at) - epochOrZero(a.updated_at)
      || cmpName(a, b));
  } else if (mode === "images_desc") {
    out.sort((a, b) =>
      numOrNegInf(b?.image_count) - numOrNegInf(a?.image_count)
      || epochOrZero(b.updated_at) - epochOrZero(a.updated_at)
      || cmpName(a, b));
  } else if (mode === "name_az") {
    out.sort((a, b) => cmpName(a, b));
  } else if (mode === "name_za") {
    out.sort((a, b) => cmpName(b, a));
  } else {
    // Default: newest to oldest.
    out.sort((a, b) => epochOrZero(b.updated_at) - epochOrZero(a.updated_at) || cmpName(a, b));
  }
  return out;
}

function fmtNum(v, digits = 4) {
  if (v === null || v === undefined || String(v).trim() === "") return "n/a";
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(digits) : "n/a";
}

function fmtTemp(v) {
  if (v === null || v === undefined || String(v).trim() === "") return "n/a";
  const n = Number(v);
  return Number.isFinite(n) ? `${n.toFixed(1)} C` : "n/a";
}

function fmtSecs(v) {
  if (v === null || v === undefined || String(v).trim() === "") return "n/a";
  const n = Number(v);
  return Number.isFinite(n) ? `${n.toFixed(1)} s` : "n/a";
}

function toFiniteOrNull(v) {
  if (v === null || v === undefined || String(v).trim() === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

const METRIC_DESCRIPTIONS = {
  "Run mode": "Simulation workflow used to generate this result set.",
  "Model family": "Physics model package controlling EM + thermal coupling behavior.",
  "Calibration": "DSC/fit profile used to map field heating into melt and densification response.",
  "Exposure": "Total applied dwell time; longer exposure generally increases energy deposition.",
  "Mean rho (final)": "Average final relative density in the part; higher means better overall densification.",
  "Min rho (final)": "Lowest final local density in the part; highlights worst under-densified region.",
  "Rho std": "Spatial spread of final density; lower indicates more uniform densification.",
  "Mean phi (final)": "Average melt fraction at end of run; near 1.0 indicates broad melt completion.",
  "Max T (part)": "Peak part temperature; used to detect overheating and thermal damage risk.",
  "Mean T (part)": "Average part temperature at final time; indicates global thermal load.",
  "Temp ceiling": "Configured maximum acceptable part temperature threshold.",
  "Antennae count": "Number of antennae assist features injected into underheated boundary regions.",
  "Shell enabled": "Whether hollow/shelled geometry mode was active for this run.",
};

function metricsRowsFromSummary(summary = {}, fallback = {}, run = null) {
  const getNum = (...keys) => {
    for (const k of keys) {
      const n = toFiniteOrNull(summary?.[k]);
      if (n !== null) return n;
    }
    for (const k of keys) {
      const n = toFiniteOrNull(fallback?.[k]);
      if (n !== null) return n;
    }
    return null;
  };
  const getAny = (...keys) => {
    for (const k of keys) {
      const v = summary?.[k];
      if (v !== undefined && v !== null && String(v) !== "") return v;
    }
    for (const k of keys) {
      const v = fallback?.[k];
      if (v !== undefined && v !== null && String(v) !== "") return v;
    }
    return null;
  };
  const p0 = Array.isArray(summary?.part_stats) && summary.part_stats.length ? summary.part_stats[0] : null;
  const minRhoPart = toFiniteOrNull(p0?.min_rho_rel);
  const stdRhoPart = toFiniteOrNull(p0?.std_rho_rel);
  const tempCeiling =
    getNum("temp_ceiling_c")
    ?? toFiniteOrNull(summary?.orientation_optimizer?.best?.temp_ceiling_c)
    ?? toFiniteOrNull(summary?.placement_optimizer?.constraints?.temp_ceiling_c);
  const rows = [
    ["Run mode", String(run?.run_type || getAny("mode", "run_type") || "n/a")],
    ["Model family", getAny("model_family") ?? "n/a"],
    ["Calibration", getAny("calibration_version") ?? "n/a"],
    ["Exposure", fmtSecs(getNum("exposure_time_s", "t_final_s", "time_final_s"))],
    ["Mean rho (final)", fmtNum(getNum("mean_rho_rel_part_final", "mean_rho_rel_part"), 4)],
    ["Min rho (final)", fmtNum(getNum("min_rho_rel_part_final", "min_rho_rel_part") ?? minRhoPart, 4)],
    ["Mean phi (final)", fmtNum(getNum("mean_phi_part_final", "mean_phi_part"), 4)],
    ["Max T (part)", fmtTemp(getNum("max_T_part_final_c", "max_T_part_c"))],
    ["Mean T (part)", fmtTemp(getNum("mean_T_part_final_c", "mean_T_part_c"))],
    ["Temp ceiling", fmtTemp(tempCeiling)],
    ["Antennae count", getAny("antennae_count") ?? "n/a"],
    ["Shell enabled", getAny("shell_enabled") ?? "n/a"],
  ];
  if ((getNum("rho_uniformity_std", "std_rho_rel_part_final") ?? stdRhoPart) !== null) {
    rows.splice(6, 0, ["Rho std", fmtNum(getNum("rho_uniformity_std", "std_rho_rel_part_final") ?? stdRhoPart, 4)]);
  }
  return rows.map(([metric, value]) => [metric, value, METRIC_DESCRIPTIONS[metric] || ""]);
}

function renderMetricsTable(run, detail = null) {
  const runName = byId("metricsRunName");
  const runMeta = byId("metricsMeta");
  const wrap = byId("metricsTableWrap");
  if (!run || !wrap || !runName || !runMeta) return;
  const summary = detail?.summary && typeof detail.summary === "object" ? detail.summary : {};
  const fallback = run?.summary_excerpt && typeof run.summary_excerpt === "object" ? run.summary_excerpt : {};
  const rows = metricsRowsFromSummary(summary, fallback, run);
  runName.textContent = run.name;
  runMeta.textContent = `${run.group || "runs"} • created ${run.run_created_at || "unknown"} • updated ${run.updated_at || "unknown"} • ${run.run_type || "unknown"}`;
  const table = document.createElement("table");
  table.className = "metrics-table";
  table.innerHTML = `
    <thead><tr><th>Metric</th><th>Value</th><th>Why it matters</th></tr></thead>
    <tbody>
      ${rows.map(([k, v, d]) => `<tr><td>${k}</td><td>${v}</td><td>${d}</td></tr>`).join("")}
    </tbody>
  `;
  wrap.innerHTML = "";
  wrap.appendChild(table);
}

async function updateMetricsPanel() {
  const run = RUNS.find((r) => r.name === SELECTED_RUN);
  if (!run) return;
  const cached = DETAIL_CACHE.get(run.name);
  if (cached) {
    renderMetricsTable(run, cached);
    return;
  }
  try {
    const detail = await fetchJson(`/api/results/${encodeURIComponent(run.name)}`);
    DETAIL_CACHE.set(run.name, detail);
    renderMetricsTable(run, detail);
  } catch (err) {
    const wrap = byId("metricsTableWrap");
    if (wrap) wrap.textContent = `Failed to load run summary: ${err.message}`;
  }
}

function renderRunCards() {
  const root = byId("runCards");
  const rows = filterRuns();
  if (!SELECTED_RUN || !rows.some((r) => r.name === SELECTED_RUN)) {
    SELECTED_RUN = rows.length ? rows[0].name : "";
  }
  root.innerHTML = "";
  if (!rows.length) {
    root.textContent = "No runs match current filters.";
    const runName = byId("metricsRunName");
    const runMeta = byId("metricsMeta");
    const wrap = byId("metricsTableWrap");
    if (runName) runName.textContent = "No run selected.";
    if (runMeta) runMeta.textContent = "";
    if (wrap) wrap.textContent = "";
    return;
  }

  rows.forEach((run) => {
    const card = document.createElement("article");
    card.className = "run-card";
    if (run.name === SELECTED_RUN) card.classList.add("selected");
    if (run.starred) card.classList.add("starred");

    const allItems = (run.images || []).map((img) => ({ url: img.url, title: `${run.name}/${img.path}` }));
    const hero = run.hero_images || [];

    const caps = Array.isArray(run.backfill_capabilities) ? run.backfill_capabilities : [];
    const backfillAction = caps.length
      ? `<div class="run-actions">
          <button type="button" class="run-star-btn" aria-label="Toggle star" title="Star run">${run.starred ? "★ Starred" : "☆ Star"}</button>
          <button type="button" class="run-backfill-btn">Backfill Reports</button>
          <button type="button" class="run-backfill-adv-btn">Backfill (Advanced)</button>
          <button type="button" class="run-delete-btn">Delete Run</button>
        </div>`
      : `<div class="run-actions">
          <button type="button" class="run-star-btn" aria-label="Toggle star" title="Star run">${run.starred ? "★ Starred" : "☆ Star"}</button>
          <button type="button" class="run-delete-btn">Delete Run</button>
        </div>`
      ;

    card.innerHTML = `
      <div class="run-head">
        <div>
          <strong>${run.name}</strong>
          <div class="muted">${run.group} • created ${run.run_created_at || "unknown"} • updated ${run.updated_at} • ${run.image_count} image(s) • ${run.run_type || "unknown"}</div>
        </div>
        <div class="muted">${metricLine(run.summary_excerpt || {})}</div>
      </div>
      ${backfillAction}
      <div class="run-hero"></div>
      <details class="run-details">
        <summary>Show all images</summary>
        <div class="run-all-images"></div>
      </details>
    `;
    card.addEventListener("click", (ev) => {
      if (ev.target instanceof HTMLElement && ev.target.closest("button,summary,a,input,select,textarea")) return;
      SELECTED_RUN = run.name;
      renderRunCards();
      void updateMetricsPanel();
    });

    const heroWrap = card.querySelector(".run-hero");
    hero.forEach((img) => {
      const idx = allItems.findIndex((it) => it.url === img.url);
      const b = document.createElement("button");
      b.type = "button";
      b.className = "thumb-btn";
      b.innerHTML = `<img src="${img.url}" alt="${img.path}" loading="lazy" />`;
      b.onclick = () => openViewer(allItems, Math.max(0, idx));
      heroWrap.appendChild(b);
    });

    const allWrap = card.querySelector(".run-all-images");
    (run.images || []).forEach((img, idx) => {
      const fig = document.createElement("figure");
      fig.innerHTML = `
        <button type="button" class="thumb-btn"><img src="${img.url}" alt="${img.path}" loading="lazy" /></button>
        <figcaption>${img.path}</figcaption>
      `;
      fig.querySelector("button").onclick = () => openViewer(allItems, idx);
      allWrap.appendChild(fig);
    });

    const backfillBtn = card.querySelector(".run-backfill-btn");
    if (backfillBtn && caps.length) {
      backfillBtn.addEventListener("click", async () => {
        try {
          await runBackfillForRun(run.name, [], "quick");
          await refresh();
        } catch (err) {
          const status = byId("serverStatus");
          if (status) status.textContent = `Backfill error: ${err.message}`;
        }
      });
    }
    const backfillAdvBtn = card.querySelector(".run-backfill-adv-btn");
    if (backfillAdvBtn && caps.length) {
      backfillAdvBtn.addEventListener("click", async () => {
        try {
          const chosen = pickModulesInteractive(caps);
          if (!chosen.length) return;
          await runBackfillForRun(run.name, chosen, "advanced");
          await refresh();
        } catch (err) {
          const status = byId("serverStatus");
          if (status) status.textContent = `Backfill error: ${err.message}`;
        }
      });
    }
    const deleteBtn = card.querySelector(".run-delete-btn");
    const starBtn = card.querySelector(".run-star-btn");
    if (starBtn) {
      starBtn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const next = !Boolean(run.starred);
        try {
          await setRunStar(run.name, next);
          run.starred = next;
          starBtn.textContent = next ? "★ Starred" : "☆ Star";
          card.classList.toggle("starred", next);
          const status = byId("serverStatus");
          if (status) status.textContent = next ? `Starred: ${run.name}` : `Unstarred: ${run.name}`;
        } catch (err) {
          const status = byId("serverStatus");
          if (status) status.textContent = `Star error: ${err.message}`;
        }
      });
    }
    if (deleteBtn) {
      deleteBtn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const ok = window.confirm(`Delete run '${run.name}' and all files under outputs_eqs? This cannot be undone.`);
        if (!ok) return;
        try {
          await deleteRun(run.name);
          DETAIL_CACHE.delete(run.name);
          if (SELECTED_RUN === run.name) SELECTED_RUN = "";
          const status = byId("serverStatus");
          if (status) status.textContent = `Deleted run: ${run.name}`;
          await refresh();
        } catch (err) {
          const status = byId("serverStatus");
          if (status) status.textContent = `Delete error: ${err.message}`;
        }
      });
    }

    root.appendChild(card);
  });
  void updateMetricsPanel();
}

async function refresh() {
  RUNS = await fetchJson("/api/results-runview");
  renderGroupFilter(RUNS);
  renderModeFilter(RUNS);
  renderRunCards();
}

async function init() {
  byId("refreshResults").onclick = refresh;
  byId("runSearch").oninput = renderRunCards;
  byId("runGroupFilter").onchange = renderRunCards;
  byId("runModeFilter").onchange = renderRunCards;
  byId("runStarFilter").onchange = renderRunCards;
  byId("runSort").onchange = renderRunCards;
  await refresh();
  setInterval(refresh, 6000);
}

init().catch((err) => {
  const s = byId("serverStatus");
  if (s) s.textContent = "HEATR service unreachable";
});
