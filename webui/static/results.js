const byId = (id) => document.getElementById(id);
const viewer = { items: [], index: 0 };
let RUNS = [];
let SELECTED_RUN = "";
const DETAIL_CACHE = new Map();

// ── View-mode state ───────────────────────────────────────────────────────────
let VIEW_MODE = localStorage.getItem("heatr_view_mode") || "flat"; // "flat" | "grouped"

// ── Compare mode state ────────────────────────────────────────────────────────
let COMPARE_MODE = false;
const COMPARE_SELECTED = new Set(); // Set of run.name strings

// ── URL deep-linking ──────────────────────────────────────────────────────────
function _applyUrlState() {
  try {
    const params = new URLSearchParams(window.location.search);
    const run = params.get("run");
    if (run) SELECTED_RUN = run;
  } catch (_) {}
}

function _pushUrlState(runName) {
  try {
    const url = new URL(window.location.href);
    if (runName) url.searchParams.set("run", runName);
    else url.searchParams.delete("run");
    window.history.replaceState(null, "", url.toString());
  } catch (_) {}
}

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

/**
 * Show a modal preview of a generated FGM PNG with stats and file paths.
 * Creates a lightweight overlay that is removed on click-outside or close button.
 */
function _copyToClipboard(text) {
  navigator.clipboard.writeText(text).catch(() => {});
}

function _showFgmPreview(runName, resp) {
  // Remove any existing FGM overlay
  const existing = document.getElementById("fgmPreviewOverlay");
  if (existing) existing.remove();

  const stats    = resp.level_stats || {};
  const unique   = (stats.unique || []).join(", ") || "—";
  const npzPath  = resp.npz_path  || "";
  const pngPath  = resp.png_path  || "";
  // Auto-derive the RIP output dir: same folder as the NPZ, subfolder "rip_output"
  const npzDir   = npzPath.substring(0, npzPath.lastIndexOf("/") + 1);
  const ripOutDir = npzDir ? npzDir + "rip_output" : "rip_output";
  // Auto output name for re-simulation: leaf run name + _fgm + proxy + mag suffix
  const leafName  = runName.replace(/.*\//, "");
  const autoName  = (leafName + "_fgm").replace(/[^a-zA-Z0-9_-]/g, "_").replace(/_+/g, "_");

  function pathRow(label, pathVal) {
    if (!pathVal) return "";
    return `<tr>
      <td style="color:#aaa;padding:3px 8px 3px 0;white-space:nowrap;">${label}</td>
      <td style="word-break:break-all;font-size:11px;color:#c8d8f0;">${pathVal}</td>
      <td style="padding-left:6px;">
        <button onclick="_copyToClipboard('${pathVal.replace(/'/g,"\\'")}');"
                style="background:#2a3a5a;border:1px solid #3a5a8a;color:#90b8e8;
                       border-radius:3px;cursor:pointer;padding:1px 6px;font-size:10px;
                       white-space:nowrap;">copy</button>
      </td></tr>`;
  }

  const overlay = document.createElement("div");
  overlay.id    = "fgmPreviewOverlay";
  overlay.style.cssText = [
    "position:fixed", "inset:0", "background:rgba(0,0,0,0.75)",
    "display:flex", "align-items:center", "justify-content:center",
    "z-index:9999", "padding:16px",
  ].join(";");

  overlay.innerHTML = `
    <div style="background:#1a1a2e;border-radius:8px;padding:20px;max-width:700px;
                width:100%;box-shadow:0 8px 32px rgba(0,0,0,0.6);color:#e0e0e0;
                font-family:monospace;font-size:13px;overflow-y:auto;max-height:92vh;">

      <!-- Header -->
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <strong style="font-size:15px;color:#f0c060;">FGM — ${leafName}</strong>
        <button id="fgmPreviewClose"
                style="background:none;border:1px solid #555;color:#ccc;cursor:pointer;
                       border-radius:4px;padding:2px 10px;font-size:14px;">✕</button>
      </div>

      <!-- Preview image -->
      <img src="data:image/png;base64,${resp.png_b64}"
           style="width:100%;border-radius:4px;border:1px solid #333;
                  image-rendering:pixelated;margin-bottom:10px;"
           alt="FGM saturation map" />
      <p style="margin:0 0 10px;color:#888;font-size:11px;">
        <strong style="color:#b0d080;">White = more ink (high saturation / more CB dopant).</strong>
        Black = no ink. This matches Meteor RIP convention — feed this PNG directly to the RIPer.
        <br>${resp.bpp}bpp · proxy=${resp.proxy_field} · mag=${resp.magnitude.toFixed(2)} · levels present: ${unique} · mean=${(stats.mean||0).toFixed(2)}
      </p>

      <!-- Output files (prominent) -->
      <div style="background:#111828;border-radius:5px;padding:10px;margin-bottom:14px;">
        <div style="color:#f0c060;font-size:11px;margin-bottom:6px;letter-spacing:0.05em;">OUTPUT FILES</div>
        <table style="width:100%;border-collapse:collapse;font-size:12px;">
          ${pathRow("NPZ (machine)", npzPath)}
          ${pathRow("PNG (preview)", pngPath)}
          ${pathRow("RIP output dir", ripOutDir)}
        </table>
      </div>

      <!-- Action buttons -->
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;">
        <button id="fgmBtnResim"
                style="flex:1;min-width:200px;padding:10px 14px;border-radius:5px;cursor:pointer;
                       background:#1e3a6e;border:1px solid #3a6abf;color:#c8dcff;font-size:13px;
                       font-family:monospace;text-align:left;">
          ▶&nbsp; Re-simulate with FGM
          <div style="font-size:10px;color:#7090c0;margin-top:2px;">Queues a new run → Results tab</div>
        </button>
        <button id="fgmBtnRip"
                style="flex:1;min-width:200px;padding:10px 14px;border-radius:5px;cursor:pointer;
                       background:#1a3d28;border:1px solid #3a7a50;color:#b0f0c8;font-size:13px;
                       font-family:monospace;text-align:left;">
          🖨&nbsp; Send to RIP
          <div style="font-size:10px;color:#60a878;margin-top:2px;">Writes TIFFs → ${ripOutDir}</div>
        </button>
      </div>

      <!-- Status line -->
      <div id="fgmActionStatus" style="font-size:12px;color:#aaa;min-height:18px;padding:4px 0;"></div>
    </div>
  `;

  document.body.appendChild(overlay);
  document.getElementById("fgmPreviewClose").onclick = () => overlay.remove();
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

  const statusEl  = () => document.getElementById("fgmActionStatus");
  const resimBtn  = () => document.getElementById("fgmBtnResim");
  const ripBtn    = () => document.getElementById("fgmBtnRip");

  // ── Re-simulate with FGM ──────────────────────────────────────────────────
  resimBtn().onclick = async () => {
    // One prompt: just confirm/edit the output name; magnitude comes from the FGM
    const rawName = window.prompt(
      "Output name for re-simulation run:",
      autoName
    );
    if (rawName === null) return;
    const outputName = rawName.trim().replace(/[^a-zA-Z0-9_-]/g, "_") || autoName;

    statusEl().style.color = "#aaa";
    statusEl().textContent = "Queuing re-simulation…";
    resimBtn().disabled = true;

    try {
      const r = await fetchJson("/api/tools/fgm-resimulate", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          source_run_dir: runName,
          fgm_npz_path:   npzPath,
          output_name:    outputName,
          magnitude:      resp.magnitude,
        }),
      });
      if (!r.ok) throw new Error(r.error || "fgm-resimulate failed");
      statusEl().style.color = "#80d080";
      statusEl().innerHTML =
        `✓ Job <strong>${r.job_id}</strong> queued — output: <code style="color:#c8dcff">${outputName}</code>. ` +
        `<a href="#" onclick="document.getElementById('fgmPreviewOverlay').remove();return false;"
              style="color:#80a0ff;">Close</a> and check the Queue tab.`;
    } catch (err) {
      statusEl().style.color = "#e08080";
      statusEl().textContent = `✗ ${err.message}`;
      resimBtn().disabled = false;
    }
  };

  // ── Send to RIP ───────────────────────────────────────────────────────────
  ripBtn().onclick = async () => {
    const layersStr = window.prompt("Number of Z-layers to emit (same FGM repeated per layer):", "1");
    if (layersStr === null) return;
    const nLayers = Math.max(1, parseInt(layersStr) || 1);

    statusEl().style.color = "#aaa";
    statusEl().textContent = `Writing ${nLayers} TIFF layer(s)…`;
    ripBtn().disabled = true;

    try {
      const r = await fetchJson("/api/tools/fgm-to-rip", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          fgm_npz_path: npzPath,
          n_layers:     nLayers,
          job_name:     leafName.replace(/[^a-zA-Z0-9_-]/g, "_"),
          // output_dir omitted → server defaults to rip_output/ next to the NPZ
        }),
      });
      if (!r.ok) throw new Error(r.error || "fgm-to-rip failed");
      statusEl().style.color = "#80d080";
      statusEl().innerHTML =
        `✓ ${r.n_layers} TIFF(s) written → ` +
        `<code style="color:#b0f0c8">${r.output_dir}</code> ` +
        `<button onclick="_copyToClipboard('${(r.output_dir||"").replace(/'/g,"\\'")}');"
                 style="background:#1a3d28;border:1px solid #3a7a50;color:#80d0a0;
                        border-radius:3px;cursor:pointer;padding:1px 6px;font-size:10px;">copy path</button>`;
    } catch (err) {
      statusEl().style.color = "#e08080";
      statusEl().textContent = `✗ ${err.message}`;
      ripBtn().disabled = false;
    }
  };
}

/**
 * Show the FGM convergence dashboard for an fgm_iterate run.
 * Fetches /api/convergence/<runName> then renders a modal with per-iteration
 * metrics, FGM thumbnails, composite scores, and "Use this FGM" buttons.
 */
async function _showConvergenceDashboard(runName) {
  // Remove any stale overlay
  const existing = document.getElementById("convDashOverlay");
  if (existing) existing.remove();

  // Build a loading overlay immediately
  const overlay = document.createElement("div");
  overlay.id = "convDashOverlay";
  overlay.style.cssText = [
    "position:fixed", "inset:0", "background:rgba(0,0,0,0.80)",
    "display:flex", "align-items:center", "justify-content:center",
    "z-index:9999", "padding:16px",
  ].join(";");
  overlay.innerHTML = `
    <style>
      .conv-criterion { color: #6090b0; font-size: 11px; margin-left: 4px; }
      .conv-reason { cursor: help; color: #4080a0; margin-left: 4px; }
    </style>
    <div id="convDashInner" style="background:#131826;border-radius:10px;padding:24px;
         max-width:900px;width:100%;max-height:92vh;overflow-y:auto;
         box-shadow:0 12px 48px rgba(0,0,0,0.7);color:#e0e0e0;font-family:monospace;font-size:13px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
        <strong style="font-size:16px;color:#f0c060;">📊 FGM Convergence — ${runName}</strong>
        <button id="convDashClose" style="background:none;border:1px solid #555;color:#ccc;
                cursor:pointer;border-radius:4px;padding:3px 12px;font-size:14px;">✕</button>
      </div>
      <div id="convDashBody" style="color:#aaa;">Loading convergence data…</div>
    </div>`;
  document.body.appendChild(overlay);

  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
  document.getElementById("convDashClose").onclick = () => overlay.remove();

  const body = document.getElementById("convDashBody");

  // Fetch convergence data
  let data;
  try {
    data = await fetchJson(`/api/convergence/${encodeURIComponent(runName)}`);
  } catch (err) {
    body.innerHTML = `<span style="color:#e08080;">Failed to load convergence data: ${err.message}</span>`;
    return;
  }

  const iters = Array.isArray(data.iterations) ? data.iterations : [];
  const bestIter = Number(data.best_iter || 0);
  const converged = Boolean(data.converged);
  const convergenceType = String(data.convergence_type || "");   // "sigma_T" | "rho_plateau" | ""
  const convergenceNote = String(data.convergence_note || "");
  const proxyField = String(data.proxy_field || "T");
  const magnitude0 = Number(data.magnitude_base || data.magnitude || 1.0);
  const magnitudeDecay = Number(data.magnitude_decay || 0.7);
  const fgmMomentum = Number(data.fgm_momentum ?? 0.3);
  const bpp = Number(data.bpp || 2);
  const useDeltaCorrection = Boolean(data.use_delta_correction);
  const useHybrid = Boolean(data.use_hybrid);
  const moveLimit = Number(data.move_limit ?? 0);
  const sensitivityFilterSigma = Number(data.sensitivity_filter_sigma ?? 0);
  const hybridPhase2Mag = Number(data.hybrid_phase2_magnitude ?? 0.4);
  const hybridPhase2ML  = Number(data.hybrid_phase2_move_limit ?? 0.15);
  const hybridBaseline  = Number(data.hybrid_phase2_baseline_sigma_T ?? 0);
  const bestScore = Number(data.best_score ?? NaN);
  // Did the loop run to completion without stopping early?
  const nIterationsRun = Number(data.n_iterations_run || iters.length);
  const loopRanToCompletion = !data.aborted && (nIterationsRun >= iters.length);

  if (!iters.length) {
    body.innerHTML = `<span style="color:#e08080;">No iteration data found in convergence.json.</span>`;
    return;
  }

  // Score bar helper — lower score = better (greener)
  function scoreBar(score, maxScore) {
    if (score == null || !Number.isFinite(Number(score))) return "—";
    const s = Number(score);
    const pct = Math.min(100, Math.max(0, (s / Math.max(maxScore, 1)) * 100));
    // green for low, red for high
    const hue = Math.round(120 - pct * 1.2);
    return `<div style="display:inline-flex;align-items:center;gap:6px;">
      <div style="width:60px;height:8px;background:#222;border-radius:4px;overflow:hidden;">
        <div style="width:${pct.toFixed(1)}%;height:100%;background:hsl(${hue},70%,50%);"></div>
      </div>
      <span>${s.toFixed(2)}</span>
    </div>`;
  }

  // Sigma_T trend sparkline (inline SVG)
  function sparkline(values, highlightIdx) {
    if (!values.length) return "";
    const W = 120, H = 28, pad = 3;
    const lo = Math.min(...values), hi = Math.max(...values);
    const span = hi - lo || 1;
    const pts = values.map((v, i) => {
      const x = pad + (i / Math.max(values.length - 1, 1)) * (W - pad * 2);
      const y = pad + (1 - (v - lo) / span) * (H - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    const dots = values.map((v, i) => {
      const x = pad + (i / Math.max(values.length - 1, 1)) * (W - pad * 2);
      const y = pad + (1 - (v - lo) / span) * (H - pad * 2);
      const r = i === highlightIdx ? 4 : 2;
      const fill = i === highlightIdx ? "#f0c060" : "#6090c8";
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" fill="${fill}"/>`;
    }).join("");
    return `<svg width="${W}" height="${H}" style="vertical-align:middle;">
      <polyline points="${pts}" fill="none" stroke="#4070a0" stroke-width="1.5"/>
      ${dots}
    </svg>`;
  }

  const sigTValues = iters.map((it) => Number(it.sigma_T ?? it.delta_T_c ?? 0));
  const scoreValues = iters.map((it) => Number(it.score ?? 0));
  const maxScore = Math.max(...scoreValues, 0.01);

  // Summary header
  const summaryHTML = `
    <div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px;padding:12px;
         background:#0e1520;border-radius:6px;border:1px solid #2a3a5a;">
      <div>
        <div style="color:#888;font-size:11px;">Proxy field</div>
        <div style="color:#f0c060;">${proxyField}</div>
      </div>
      <div>
        <div style="color:#888;font-size:11px;">Initial magnitude</div>
        <div style="color:#c8dcff;">${magnitude0.toFixed(2)}</div>
      </div>
      <div>
        <div style="color:#888;font-size:11px;">Bits per pixel</div>
        <div style="color:#c8dcff;">${bpp} bpp (${1 << bpp} levels)</div>
      </div>
      <div>
        <div style="color:#888;font-size:11px;">Iterations run</div>
        <div style="color:#c8dcff;">${iters.length}</div>
      </div>
      <div>
        <div style="color:#888;font-size:11px;">Best iteration</div>
        <div style="color:#${bestIter > 0 ? "80d080" : "e08080"};">iter ${bestIter}</div>
      </div>
      <div>
        <div style="color:#888;font-size:11px;"
             title="${convergenceNote || (converged ? "Convergence criterion met" : "Loop ran to completion without meeting convergence criterion")}">
          Converged
        </div>
        <div style="color:#${converged ? "80d080" : "e0922a"};">
          ${converged
            ? (convergenceType === "rho_plateau"
                ? `yes ✓ <span style="font-size:10px;opacity:0.7">(ρ̄ plateau)</span>`
                : `yes ✓ <span style="font-size:10px;opacity:0.7">(σ_T)</span>`)
            : "no"}
        </div>
      </div>
      <div>
        <div style="color:#888;font-size:11px;">Best score</div>
        <div style="color:#80d080;">${Number.isFinite(bestScore) ? bestScore.toFixed(2) : "—"}</div>
      </div>
      <div>
        <div style="color:#888;font-size:11px;" title="Scoring regime of the best iteration: 'active' means ρ̄ < 0.82 (still sintering — σ_T not penalized); 'near-final' means ρ̄ ≥ 0.82 (uniformity is the primary metric).">Best regime</div>
        <div style="color:${data.best_score_regime === 'active' ? '#70c870' : '#80a0d0'};">
          ${data.best_score_regime || "—"}
        </div>
      </div>
      <div>
        <div style="color:#888;font-size:11px;">Mag decay / momentum</div>
        <div style="color:#c8dcff;">${magnitudeDecay.toFixed(2)} / ${fgmMomentum.toFixed(2)}</div>
      </div>
      <div title="${useHybrid
          ? 'Hybrid: Proportional phase-1 (iter-1) achieves the maximum single-step reduction, then OC-TO phase-2 (iter-2+) refines with bounded corrections for stable convergence.'
          : useDeltaCorrection
          ? 'OC-TO Integral: Optimality Criteria topology-optimization update. Move limits + sensitivity filter + OC bisection for exact volume conservation.'
          : 'Proportional mode: full saturation map recomputed from current proxy field each iteration. Strong first correction; may oscillate on iter-2+.'}">
        <div style="color:#888;font-size:11px;">Correction mode</div>
        <div style="color:${useHybrid ? '#4aaa88' : useDeltaCorrection ? '#4ac888' : '#aaaaaa'};">
          ${useHybrid ? '⚡ Hybrid (Prop→OC-TO)' : useDeltaCorrection ? '∫ OC-TO' : '÷ Proportional'}
        </div>
      </div>
      ${(useDeltaCorrection || useHybrid) ? `<div title="OC-TO parameters. Move limit: max |Δs| per pixel. Sensitivity filter σ: Gaussian on sensitivity field.">
        <div style="color:#888;font-size:11px;">${useHybrid ? 'Phase-2 params' : 'TO params'}</div>
        <div style="color:#a0c8e0;font-size:11px;">
          m=${(useHybrid ? hybridPhase2ML : moveLimit).toFixed(2)} / σ_f=${sensitivityFilterSigma.toFixed(1)}
          ${useHybrid && hybridBaseline > 0 ? ` / phase-2 base=${hybridBaseline.toFixed(1)}°C` : ''}
        </div>
      </div>` : ''}
      <div>
        <div style="color:#888;font-size:11px;">σ_T trend</div>
        <div>${sparkline(sigTValues, bestIter > 0 ? bestIter - 1 : -1)}</div>
      </div>
      ${data.optimizer_chosen_time_min != null ? `
      <div>
        <div style="color:#888;font-size:11px;">Optimal time</div>
        <div style="color:#f0c060;">
          <strong>${Number(data.optimizer_chosen_time_min).toFixed(2)} min</strong>
          ${data.optimizer_criterion ? `<span class="conv-criterion">(${data.optimizer_criterion.replace('_',' ')})</span>` : ''}
          ${data.optimizer_reason ? `<span class="conv-reason" title="${data.optimizer_reason}">ℹ</span>` : ''}
        </div>
      </div>` : ''}
    </div>`;

  // ── Convergence chart (SVG) ───────────────────────────────────────────────
  // Multi-series line chart: σ_T (blue, left axis), ρ̄ (green, right axis),
  // frac_bleed (red, right axis).  Reference lines: ρ̄=0.82 regime boundary,
  // σ_T convergence threshold.  Best iteration marked with ★ gold dot.
  const chartHTML = (() => {
    if (iters.length < 2) return "";
    const CW = 520, CH = 110, pL = 38, pR = 44, pT = 10, pB = 28;
    const gW = CW - pL - pR, gH = CH - pT - pB;
    const iterNums   = iters.map((it) => Number(it.iter ?? 0));
    const iterMax    = Math.max(...iterNums, 1);
    const sigTs      = iters.map((it) => Number(it.sigma_T ?? 0));
    const rhos       = iters.map((it) => Number(it.mean_rho ?? 0));
    const bleeds     = iters.map((it) => Number(it.frac_bleed ?? 0) * 100);  // as %

    const sigTMax  = Math.max(...sigTs, 5);
    const rhoMinV  = Math.min(...rhos, 0.55);
    const rhoMaxV  = Math.max(...rhos, 0.82);
    const rhoSpan  = Math.max(rhoMaxV - rhoMinV, 0.05);

    function ix(iter)  { return pL + (iter / iterMax) * gW; }
    function iyL(v)    { return pT + (1 - v / (sigTMax || 1)) * gH; }  // left axis: σ_T
    function iyR(v)    { return pT + (1 - (v - rhoMinV) / rhoSpan) * gH; } // right: ρ̄

    const ptSigT  = iters.map((it) => `${ix(it.iter).toFixed(1)},${iyL(Number(it.sigma_T??0)).toFixed(1)}`).join(" ");
    const ptRho   = iters.map((it) => `${ix(it.iter).toFixed(1)},${iyR(Number(it.mean_rho??0)).toFixed(1)}`).join(" ");
    const ptBleed = bleeds.some((b) => b > 0.01)
      ? iters.map((it) => `${ix(it.iter).toFixed(1)},${iyL(Number(it.frac_bleed??0)*100*2).toFixed(1)}`).join(" ")
      : null;

    // Reference lines
    const rho82Y  = iyR(0.82).toFixed(1);
    const sigTConvY = iyL(3.0).toFixed(1);
    const refLines = [
      rhoMinV <= 0.82 && 0.82 <= rhoMaxV
        ? `<line x1="${pL}" y1="${rho82Y}" x2="${CW-pR}" y2="${rho82Y}"
                stroke="#6050a0" stroke-width="1" stroke-dasharray="4,3" opacity="0.7"/>
           <text x="${CW-pR+2}" y="${(Number(rho82Y)+3).toFixed(1)}" font-size="8" fill="#6050a0">0.82</text>`
        : "",
      Number(sigTMax) >= 3.0
        ? `<line x1="${pL}" y1="${sigTConvY}" x2="${CW-pR}" y2="${sigTConvY}"
                stroke="#40c080" stroke-width="1" stroke-dasharray="4,3" opacity="0.6"/>
           <text x="${pL-2}" y="${(Number(sigTConvY)+3).toFixed(1)}" text-anchor="end" font-size="8" fill="#40c080">3°</text>`
        : "",
    ].join("");

    // Regime transition shading (vertical band where regime changes)
    let regimeBands = "";
    for (let k = 1; k < iters.length; k++) {
      if (iters[k-1].score_regime === "active" && iters[k].score_regime === "near-final") {
        const xBand = ix(iters[k].iter);
        regimeBands += `<rect x="${(xBand-1).toFixed(1)}" y="${pT}" width="2" height="${gH}"
                              fill="#8060ff" opacity="0.4"/>
                        <text x="${xBand.toFixed(1)}" y="${(pT+CH-pB-2).toFixed(1)}" text-anchor="middle"
                              font-size="7" fill="#a080ff">active→NF</text>`;
      }
    }

    // Dots — best iter in gold
    const sigTDots = iters.map((it) => {
      const isBest = Number(it.iter) === bestIter;
      return `<circle cx="${ix(it.iter).toFixed(1)}" cy="${iyL(Number(it.sigma_T??0)).toFixed(1)}"
                      r="${isBest?4:2.5}" fill="${isBest?'#f0c060':'#5090e0'}"
                      opacity="0.95">
                <title>iter-${it.iter}: σ_T=${Number(it.sigma_T??0).toFixed(2)}°C${isBest?' ★ best':''}</title>
              </circle>`;
    }).join("");

    // Left Y-axis ticks (σ_T)
    const yTicksL = [0, sigTMax/2, sigTMax].map((v) =>
      `<text x="${pL-3}" y="${(iyL(v)+4).toFixed(1)}" text-anchor="end" font-size="9" fill="#5090e0">
        ${v.toFixed(0)}
       </text>`
    ).join("");

    // Right Y-axis ticks (ρ̄)
    const rhoTicks = [rhoMinV, (rhoMinV+rhoMaxV)/2, rhoMaxV].map((v) =>
      `<text x="${CW-pR+3}" y="${(iyR(v)+4).toFixed(1)}" text-anchor="start" font-size="9" fill="#40a060">
        ${v.toFixed(2)}
       </text>`
    ).join("");

    // X-axis iter labels
    const xLabels = iters.map((it) =>
      `<text x="${ix(it.iter).toFixed(1)}" y="${(CH-6).toFixed(1)}" text-anchor="middle"
              font-size="9" fill="#607080">${it.iter}</text>`
    ).join("");

    return `
      <div style="margin-bottom:16px;padding:10px 12px;background:#0a0f1a;border-radius:8px;
                  border:1px solid #1e2d44;">
        <div style="font-size:11px;color:#7090b0;margin-bottom:6px;display:flex;
                    justify-content:space-between;align-items:center;">
          <span style="font-weight:bold;color:#90b0d0;">📈 Convergence Chart</span>
          <span>
            <span style="color:#5090e0;">━ σ_T (°C)</span>
            &nbsp;
            <span style="color:#40a060;">━ ρ̄</span>
            &nbsp;
            <span style="color:#6050a0;border-bottom:1px dashed #6050a0;">— ρ̄=0.82 threshold</span>
            &nbsp;
            <span style="color:#40c080;border-bottom:1px dashed #40c080;">— σ_T=3°C target</span>
          </span>
        </div>
        <svg width="${CW}" height="${CH}" style="display:block;max-width:100%;">
          <rect x="${pL}" y="${pT}" width="${gW}" height="${gH}" fill="#0d1520" rx="3"/>
          <!-- grid lines -->
          ${[1,2,3].map((k) => {
            const gy = (pT + (k/4)*gH).toFixed(1);
            return `<line x1="${pL}" y1="${gy}" x2="${CW-pR}" y2="${gy}"
                          stroke="#1a2535" stroke-width="1"/>`;
          }).join("")}
          ${regimeBands}
          ${refLines}
          <!-- ρ̄ line (right axis, green) -->
          <polyline points="${ptRho}" fill="none" stroke="#40a060" stroke-width="2" opacity="0.8"/>
          ${ptBleed ? `<polyline points="${ptBleed}" fill="none" stroke="#e05050" stroke-width="1.5" opacity="0.7" stroke-dasharray="3,2"/>` : ""}
          <!-- σ_T line (blue, primary) -->
          <polyline points="${ptSigT}" fill="none" stroke="#5090e0" stroke-width="2.5"/>
          ${sigTDots}
          <!-- axes -->
          <line x1="${pL}" y1="${pT}" x2="${pL}" y2="${pT+gH}" stroke="#2a3a5a" stroke-width="1"/>
          <line x1="${pL}" y1="${pT+gH}" x2="${CW-pR}" y2="${pT+gH}" stroke="#2a3a5a" stroke-width="1"/>
          <line x1="${CW-pR}" y1="${pT}" x2="${CW-pR}" y2="${pT+gH}" stroke="#2a3a5a" stroke-width="1"/>
          ${yTicksL}
          ${rhoTicks}
          ${xLabels}
          <!-- axis labels -->
          <text x="${pL-24}" y="${(pT + gH/2).toFixed(1)}" text-anchor="middle" font-size="9"
                fill="#5090e0" transform="rotate(-90 ${pL-24} ${(pT + gH/2).toFixed(1)})">σ_T (°C)</text>
          <text x="${CW-pR+28}" y="${(pT + gH/2).toFixed(1)}" text-anchor="middle" font-size="9"
                fill="#40a060" transform="rotate(90 ${CW-pR+28} ${(pT + gH/2).toFixed(1)})">ρ̄</text>
          <text x="${pL + gW/2}" y="${CH-1}" text-anchor="middle" font-size="9" fill="#607080">
            iteration
          </text>
        </svg>
      </div>`;
  })();

  // Selectivity colour helper: green ≥ 2.5, yellow 1.5–2.5, red < 1.5
  function selColor(sel) {
    if (sel == null || !Number.isFinite(Number(sel))) return "#888";
    const s = Number(sel);
    if (s >= 2.5) return "#80d080";
    if (s >= 1.5) return "#e0d060";
    return "#e08080";
  }

  // Per-iteration table rows
  const rowsHTML = iters.map((it, idx) => {
    const isBest = Number(it.iter) === bestIter;
    const sigT  = it.sigma_T   != null ? Number(it.sigma_T).toFixed(2)   : "—";
    const dT    = it.dT_c      != null ? Number(it.dT_c).toFixed(1)
                : it.delta_T_c != null ? Number(it.delta_T_c).toFixed(1) : "—";
    const sigR  = it.sigma_rho != null ? Number(it.sigma_rho).toFixed(4) : "—";
    const rho   = it.mean_rho  != null ? Number(it.mean_rho).toFixed(3)  : "—";
    const mag   = it.magnitude_used != null ? Number(it.magnitude_used).toFixed(3) : "—";

    // Selectivity columns (new)
    const sel   = it.thermal_selectivity != null ? Number(it.thermal_selectivity) : null;
    const selStr= sel != null ? sel.toFixed(2) : "—";
    const tOut  = it.T_out_mean_c != null ? Number(it.T_out_mean_c).toFixed(1)+"°C" : "—";
    const bleed = it.frac_bleed  != null ? (Number(it.frac_bleed)*100).toFixed(2)+"%" : "—";
    const bleedColor = it.frac_bleed > 0.001 ? "#e08080"
                     : it.frac_bleed > 0     ? "#e0d060" : "#80d080";

    // Score regime badge: "active" = still sintering, σ_T rise is expected
    //                     "near-final" = uniformity is the primary metric
    const regime = String(it.score_regime || "");
    const regimeBadge = regime === "active"
      ? `<span title="Active-sintering regime: ρ̄ < 0.82. σ_T increase is expected while the part is still sintering — score measures density progress and bleed only."
               style="background:#1a3a20;color:#70c870;border-radius:3px;
                      padding:1px 5px;font-size:9px;margin-left:5px;cursor:help;">active</span>`
      : regime === "near-final"
        ? `<span title="Near-final regime: ρ̄ ≥ 0.82. Uniformity (σ_T) is the primary metric — lower σ_T should be the goal from here."
                 style="background:#1a2540;color:#80a0d0;border-radius:3px;
                        padding:1px 5px;font-size:9px;margin-left:5px;cursor:help;">near-final</span>`
        : "";

    const npzPath      = String(it.fgm_npz || "");
    const hasPng       = Boolean(it.fgm_png_b64);
    const hasMeteorPng = Boolean(it.fgm_meteor_png_b64);
    const hasThermal   = Boolean(it.thermal_png_b64);

    const bestBadge = isBest
      ? `<span style="background:#2a5a20;color:#80d080;border-radius:3px;
                      padding:1px 6px;font-size:10px;margin-left:6px;">★ best</span>`
      : "";

    // Phase badge (hybrid mode only): show which phase generated this iter's FGM
    const fgmAlgo = String(it.fgm_algo || "");
    const phaseBadge = useHybrid && fgmAlgo
      ? (fgmAlgo === "OC-TO"
          ? `<span title="FGM generated by OC-TO (phase-2) — bounded gradient correction"
                 style="background:#0a2a1a;color:#4aaa88;border-radius:3px;
                        padding:1px 5px;font-size:9px;margin-left:4px;">∫ OC-TO</span>`
          : `<span title="FGM generated by Proportional (phase-1) — full magnitude correction"
                 style="background:#2a2a0a;color:#c8aa40;border-radius:3px;
                        padding:1px 5px;font-size:9px;margin-left:4px;">÷ Prop</span>`)
      : "";

    // FGM thumbnail + thermal thumbnail stacked vertically
    const fgmThumb = hasPng
      ? `<span class="conv-thumb-slot" data-iter="${it.iter}"
              data-type="fgm"
              style="display:block;cursor:pointer;margin-bottom:3px;"
              title="FGM saturation map — click to zoom"></span>`
      : "";
    const thermalThumb = hasThermal
      ? `<span class="conv-thumb-slot" data-iter="${it.iter}"
              data-type="thermal"
              style="display:block;cursor:pointer;"
              title="Thermal/density fields — click to zoom"></span>`
      : "";
    const thumbHTML = (hasPng || hasThermal)
      ? `<div style="display:inline-flex;flex-direction:column;align-items:center;gap:2px;">
           ${fgmThumb}${thermalThumb}
         </div>`
      : `<span style="color:#555;font-size:11px;">—</span>`;

    const meteorDlBtn = hasMeteorPng
      ? `<button class="conv-meteor-dl-btn"
               data-iter="${it.iter}"
               style="padding:5px 10px;border-radius:4px;cursor:pointer;font-size:11px;
                      background:#2a1a3a;border:1px solid #7060a0;color:#c0a0e8;
                      font-family:monospace;white-space:nowrap;display:block;margin-bottom:3px;"
               title="Download Meteor-import PNG (black=max ink) for direct import into Meteor RIP">
           📥 Meteor PNG
         </button>`
      : "";

    const useBtn = npzPath
      ? `${meteorDlBtn}<button class="conv-use-fgm-btn"
               data-npz="${npzPath.replace(/"/g, "&quot;")}"
               data-iter="${it.iter}"
               style="padding:5px 10px;border-radius:4px;cursor:pointer;font-size:11px;
                      background:#1e3a6e;border:1px solid #3a6abf;color:#c8dcff;
                      font-family:monospace;white-space:nowrap;margin-bottom:4px;display:block;">
           ▶ Resimulate
         </button>
         <button class="conv-rip-btn"
               data-npz="${npzPath.replace(/"/g, "&quot;")}"
               data-iter="${it.iter}"
               style="padding:5px 10px;border-radius:4px;cursor:pointer;font-size:11px;
                      background:#1a3a28;border:1px solid #3a7a50;color:#80d0a0;
                      font-family:monospace;white-space:nowrap;display:block;">
           → TIFF Stack
         </button>`
      : "";

    const rowBg = isBest ? "background:#1a2a1a;" : (idx % 2 === 0 ? "background:#0e1520;" : "background:#111928;");

    return `<tr style="${rowBg}">
      <td style="padding:8px 10px;text-align:center;font-weight:bold;color:${isBest?"#f0c060":"#c0c0c0"};">
        ${it.iter}${bestBadge}${phaseBadge}
      </td>
      <td style="padding:8px 10px;text-align:center;">${sigT}°C</td>
      <td style="padding:8px 10px;text-align:center;">${dT}°C</td>
      <td style="padding:8px 10px;text-align:center;">${sigR}</td>
      <td style="padding:8px 10px;text-align:center;">${rho}</td>
      <td style="padding:8px 10px;text-align:center;">${mag}</td>
      <td style="padding:8px 10px;text-align:center;color:${selColor(sel)};font-weight:bold;">${selStr}</td>
      <td style="padding:8px 10px;text-align:center;">${tOut}</td>
      <td style="padding:8px 10px;text-align:center;color:${bleedColor};">${bleed}</td>
      <td style="padding:8px 10px;text-align:center;">${scoreBar(it.score, maxScore)}${regimeBadge}</td>
      <td style="padding:8px 10px;text-align:center;font-size:10px;color:#a0b8d0;">${
        (() => {
          const p = it.proxy_field || "";
          const sw = it.proxy_switch || "";
          return (p ? `<span title="${p}">${p}</span>` : "") +
                 (sw ? `<br><span style="color:#c06000;font-size:9px;" title="${sw}">⇄</span>` : "");
        })()
      }</td>
      <td style="padding:8px 10px;text-align:center;">${thumbHTML}</td>
      <td style="padding:8px 10px;text-align:center;">${useBtn}</td>
    </tr>`;
  }).join("");

  // Column header tooltips — shown on hover (cursor:help)
  const TH = (label, tip, dir = "") => `<th style="padding:8px 10px;cursor:help;white-space:nowrap;"
      title="${tip}">${label}${dir ? ` <span style="color:#607080;font-size:9px;">${dir}</span>` : ""}</th>`;

  const tableHTML = `
    <div style="overflow-x:auto;margin-bottom:16px;">
      <table style="width:100%;border-collapse:collapse;font-size:12px;">
        <thead>
          <tr style="background:#1a2540;color:#90b0d8;font-size:11px;letter-spacing:0.05em;">
            ${TH("Iter", "Iteration number. Iter 0 = baseline (no FGM applied). Each subsequent iter applies a corrected FGM derived from the previous run's T_phi90 field.")}
            ${TH("σ_T", "Spatial standard deviation of temperature inside the part at the moment mean melt fraction φ first crosses 0.90 (T_phi90 snapshot). This is the primary convergence metric. Target: σ_T < 3°C = excellent uniformity.", "↓")}
            ${TH("ΔT", "T_max − T_mean inside the part. Measures peak-to-mean temperature spread. Related to σ_T but more sensitive to single hot-spot outliers. Target: ΔT < 10°C.", "↓")}
            ${TH("σ_ρ", "Spatial standard deviation of relative density inside the part at end of run. Lower = more uniform densification. Target: σ_ρ < 0.03.", "↓")}
            ${TH("ρ̄", "Mean relative density inside the part at end of run. Range [0,1]; 1.0 = fully dense. Target ≥ 0.95. Below 0.82 = active sintering regime (score formula changes).")}
            ${TH("φ%", "Mean melt fraction inside part at end of run — fraction of part volume that reached or exceeded liquidus temperature. 100% = entire part melted.")}
            ${TH("mag", "FGM magnitude used in this iteration. Starts at the configured value and decays geometrically (×magnitude_decay each iter) to prevent overshoot oscillation.")}
            ${TH("sel", "Thermal selectivity = (T̄_part − T_amb) / (T̄_outside − T_amb). Measures how well energy stays inside the part vs. leaking to surrounding powder. ≥2.5 = green (good), 1.5–2.5 = yellow (marginal), <1.5 = red (poor).", "↑")}
            ${TH("T_out", "Mean temperature of loose powder OUTSIDE the part boundary. Lower is better. High T_out means energy is bleeding into surrounding powder, which can sinter unintended material and blur part edges.")}
            ${TH("bleed%", "Fraction of outside-mask powder cells above the melt reference temperature. Non-zero = surrounding loose powder is sintering. This directly causes dimensional inaccuracy (part grows larger than designed).", "↓")}
            ${TH("score", "Composite optimization score. Lower is better. Two regimes:\\nACTIVE (ρ̄ < 0.82): score = 2σ_ρ + (0.82−ρ̄)×15 + bleed_pen + sel_pen + dmg\\nNEAR-FINAL (ρ̄ ≥ 0.82): score = 2σ_T + 25σ_ρ + |ρ̄−0.95| + bleed_pen + sel_pen + dmg\\nScores from different regimes are NOT directly comparable.", "↓")}
            ${TH("proxy", "Proxy field used to generate the FGM for this iteration. In Thorough / Regime-adaptive mode, this can change mid-run. ⇄ = proxy switched this iteration.")}
            ${TH("FGM", "Preview thumbnail of the functionally graded map generated from this iteration. White = max binder saturation (more CB dopant = more RF absorption). Black = no ink. Click to zoom. The FGM from each iteration becomes the INPUT for the next.")}
            ${TH("Action", "Use this iteration's FGM to launch a new production re-simulation, or use the best iteration as the definitive result.")}
          </tr>
        </thead>
        <tbody>${rowsHTML}</tbody>
      </table>
    </div>`;

  // Melt pre-check warning + advisory notices
  let abortWarn = "";
  if (data.aborted) {
    abortWarn += `<div style="background:#3a1a1a;border:1px solid #a04040;border-radius:5px;
                      padding:10px 14px;margin-bottom:14px;color:#f08080;font-size:12px;">
      ⚠ Run aborted: ${String(data.abort_reason || "see convergence log")}.
      Adjust exposure time or power before re-running FGM optimization.
    </div>`;
  }
  // 4bpp recommendation: when 2bpp and not converged, coarse quantization limits corrections
  if (!converged && bpp === 2) {
    abortWarn += `<div style="background:#2a2a14;border:1px solid #80700a;border-radius:5px;
                      padding:10px 14px;margin-bottom:14px;color:#e0d060;font-size:12px;">
      💡 <strong>Try 4 bpp (16 levels)</strong> — at 2 bpp (4 levels) the FGM can only make
      coarse corrections. For complex geometries with uneven coupling (hexagon, star, etc.),
      finer quantization gives the optimizer more resolution to correct hot spots without
      oscillation. Re-run with bpp set to 4.
    </div>`;
  }
  // Advisory when best iter is not the last iter
  if (!converged && !data.aborted && bestIter > 0 && bestIter < iters.length - 1) {
    const bestRegime = String(data.best_score_regime || "");
    if (loopRanToCompletion) {
      // Loop finished all N iterations — best just happened to be in the middle
      const note = bestRegime === "active"
        ? `In the <em>active</em> regime the score rewards density gains over variance.
           Later iters may have had marginally higher σ_ρ that outweighed the small density improvement.
           Check the ρ̄ and bleed% columns — if iter-${iters.length-1} has higher ρ̄ and zero bleed,
           consider using it instead.`
        : `Later iterations scored slightly worse. Use <strong>▶ Use iter ${bestIter}</strong> to
           re-simulate with the best FGM as the permanent map.`;
      abortWarn += `<div style="background:#1a2a3a;border:1px solid #3a6090;border-radius:5px;
                        padding:10px 14px;margin-bottom:14px;color:#90b8e0;font-size:12px;">
        ℹ Loop completed all ${iters.length} iterations. Best score was at iter-${bestIter}. ${note}
      </div>`;
    } else {
      // Loop stopped early (regression abort)
      const regimeNote = bestRegime === "active"
        ? " The best iteration was still in the <em>active-sintering</em> regime (ρ̄&nbsp;&lt;&nbsp;0.82); later iterations showed declining density or increasing bleed."
        : " Later iterations showed score regression.";
      abortWarn += `<div style="background:#1a2a3a;border:1px solid #3a6090;border-radius:5px;
                        padding:10px 14px;margin-bottom:14px;color:#90b8e0;font-size:12px;">
        ℹ Best result was iter-${bestIter} (of ${iters.length - 1} iterations).${regimeNote}
        Use <strong>▶ Use iter ${bestIter}</strong> below to re-simulate with the optimal FGM
        from that iteration as the permanent map.
      </div>`;
    }
  }

  // ρ̄ plateau convergence notice (active regime)
  if (converged && convergenceType === "rho_plateau") {
    abortWarn += `<div style="background:#1a2a14;border:1px solid #507040;border-radius:5px;
                      padding:10px 14px;margin-bottom:14px;color:#a0d080;font-size:12px;">
      ✓ <strong>Converged (ρ̄ plateau):</strong> The FGM has fully redistributed energy within
      this operating point — density is no longer improving between iterations.
      Current ρ̄ = ${(iters[iters.length-1]?.mean_rho ?? 0).toFixed(3)}.
      To reach the near-final regime (ρ̄&nbsp;≥&nbsp;0.82), <strong>increase the exposure time</strong>
      and re-run FGM iterate. The optimized FGM from iter-${bestIter} is the correct map to use at
      this power level.
    </div>`;
  }

  // Toolbar: export + gallery + continue
  const hasAnyPng = iters.some((it) => it.fgm_png_b64);
  const showContinue = !converged && !data.aborted;
  // Best-iter NPZ path (for top-level "Export Best to RIP" button)
  const bestIterData = iters.find((it) => Number(it.iter) === bestIter);
  const bestNpzPath  = bestIterData ? String(bestIterData.fgm_npz || "") : "";
  const exportBarHTML = `
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;align-items:center;">
      <span style="color:#888;font-size:11px;margin-right:4px;">Export:</span>
      <button id="convExportCSV"
              style="padding:5px 12px;border-radius:4px;cursor:pointer;font-size:11px;
                     background:#1a2a3a;border:1px solid #3a6090;color:#90c8e8;font-family:monospace;">
        📥 CSV
      </button>
      <button id="convExportJSON"
              style="padding:5px 12px;border-radius:4px;cursor:pointer;font-size:11px;
                     background:#1a2a3a;border:1px solid #3a6090;color:#90c8e8;font-family:monospace;">
        📥 JSON
      </button>
      ${bestNpzPath ? `<button id="convExportBestRip"
              data-npz="${bestNpzPath.replace(/"/g, "&quot;")}"
              data-iter="${bestIter}"
              style="padding:5px 12px;border-radius:4px;cursor:pointer;font-size:11px;
                     background:#1a3a28;border:1px solid #3a8060;color:#80e0a8;font-family:monospace;
                     font-weight:bold;"
              title="Export best iteration FGM (iter ${bestIter}) to Meteor RIP as a TIFF stack">
        🖨 Best (iter ${bestIter}) → TIFF Stack
      </button>` : ""}
      ${bestIterData?.fgm_meteor_png_b64 ? `<button id="convExportBestMeteorPng"
              style="padding:5px 12px;border-radius:4px;cursor:pointer;font-size:11px;
                     background:#2a1a3a;border:1px solid #7060a0;color:#c0a0e8;font-family:monospace;
                     font-weight:bold;"
              title="Download Meteor-import PNG for best iteration (iter ${bestIter}). Black=max ink — import directly into Meteor RIP.">
        📥 Best (iter ${bestIter}) Meteor PNG
      </button>` : ""}
      <button id="convPrintBtn"
              style="padding:5px 12px;border-radius:4px;cursor:pointer;font-size:11px;
                     background:#1a2030;border:1px solid #3a5070;color:#809090;font-family:monospace;"
              title="Print or save as PDF (use browser Print → Save as PDF)">
        🖨 Print / PDF
      </button>
      ${hasAnyPng ? `<button id="convFgmGallery"
              style="padding:5px 12px;border-radius:4px;cursor:pointer;font-size:11px;
                     background:#1a3a2a;border:1px solid #3a8060;color:#80e0a8;font-family:monospace;">
        📷 Gallery
      </button>` : ""}
      ${hasAnyPng ? `<button id="convFgmAnimate"
              style="padding:5px 12px;border-radius:4px;cursor:pointer;font-size:11px;
                     background:#2a1a3a;border:1px solid #604090;color:#b080e0;font-family:monospace;"
              title="Cycle through FGM thumbnails to see how the saturation map evolves">
        ▶ Animate FGMs
      </button>` : ""}
      ${showContinue ? `<button id="convContinueBtn"
              style="padding:5px 14px;border-radius:4px;cursor:pointer;font-size:11px;
                     background:#2a1a3a;border:1px solid #7060a0;color:#c0a0e8;font-family:monospace;
                     font-weight:bold;"
              title="Resume from best iteration and run 4 more FGM optimization iterations">
        ⟳ Continue 4 Iterations
      </button>` : ""}
      <button id="convScoreInfo"
              style="padding:5px 12px;border-radius:4px;cursor:pointer;font-size:11px;
                     background:#1a1a2a;border:1px solid #403060;color:#a090c0;font-family:monospace;"
              title="Explain scoring system and all metrics">
        ℹ Score Formula
      </button>
    </div>`;

  // Status line for "Use FGM" action feedback
  const actionLineHTML = `<div id="convDashActionStatus" style="min-height:18px;font-size:12px;
    color:#aaa;padding:4px 0;"></div>`;

  body.innerHTML = summaryHTML + chartHTML + abortWarn + tableHTML + exportBarHTML + actionLineHTML;

  // ── Lightbox helper ─────────────────────────────────────────────────────────
  function _showFgmLightbox(iterNum, allIters) {
    const idx = allIters.findIndex((it) => Number(it.iter) === iterNum);
    const entry = allIters[idx];
    if (!entry || !entry.fgm_png_b64) return;

    const lb = document.createElement("div");
    lb.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.92);z-index:9999;" +
      "display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;";
    lb.addEventListener("click", (e) => { if (e.target === lb) lb.remove(); });

    // Navigation buttons
    const prevBtn = document.createElement("button");
    prevBtn.textContent = "◀";
    prevBtn.style.cssText = "position:absolute;left:16px;top:50%;transform:translateY(-50%);" +
      "font-size:24px;background:none;border:none;color:#aaa;cursor:pointer;padding:12px;";
    const nextBtn = document.createElement("button");
    nextBtn.textContent = "▶";
    nextBtn.style.cssText = "position:absolute;right:16px;top:50%;transform:translateY(-50%);" +
      "font-size:24px;background:none;border:none;color:#aaa;cursor:pointer;padding:12px;";

    let currentIdx = idx;

    function renderLightboxContent() {
      const cur = allIters[currentIdx];
      const isBestLb = Number(cur.iter) === bestIter;
      img.src = `data:image/png;base64,${cur.fgm_png_b64}`;
      label.textContent = `Iter ${cur.iter}${isBestLb ? " ★ best" : ""}` +
        ` — ρ̄=${Number(cur.mean_rho||0).toFixed(3)}` +
        ` σ_T=${Number(cur.sigma_T||0).toFixed(1)}°C` +
        ` bleed=${(Number(cur.frac_bleed||0)*100).toFixed(2)}%` +
        ` score=${Number(cur.score||0).toFixed(2)}`;
      prevBtn.disabled = currentIdx === 0;
      nextBtn.disabled = currentIdx === allIters.length - 1;
      prevBtn.style.opacity = prevBtn.disabled ? "0.2" : "1";
      nextBtn.style.opacity = nextBtn.disabled ? "0.2" : "1";
    }

    prevBtn.addEventListener("click", () => {
      if (currentIdx > 0) { currentIdx--; renderLightboxContent(); }
    });
    nextBtn.addEventListener("click", () => {
      if (currentIdx < allIters.length - 1) { currentIdx++; renderLightboxContent(); }
    });

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕";
    closeBtn.style.cssText = "position:absolute;top:12px;right:16px;font-size:18px;" +
      "background:none;border:none;color:#aaa;cursor:pointer;";
    closeBtn.addEventListener("click", () => lb.remove());

    const img = document.createElement("img");
    img.style.cssText = "max-height:75vh;max-width:80vw;image-rendering:pixelated;" +
      "border:2px solid #4a6090;border-radius:4px;background:#111;";

    const label = document.createElement("div");
    label.style.cssText = "color:#c8dcff;font-size:13px;font-family:monospace;text-align:center;";

    const hint = document.createElement("div");
    hint.style.cssText = "color:#555;font-size:11px;";
    hint.textContent = "← → to navigate • click outside to close";

    lb.appendChild(prevBtn);
    lb.appendChild(nextBtn);
    lb.appendChild(closeBtn);
    lb.appendChild(img);
    lb.appendChild(label);
    lb.appendChild(hint);
    document.body.appendChild(lb);
    renderLightboxContent();

    // Keyboard navigation
    function onKey(e) {
      if (e.key === "ArrowLeft"  && currentIdx > 0)                  { currentIdx--; renderLightboxContent(); }
      if (e.key === "ArrowRight" && currentIdx < allIters.length - 1){ currentIdx++; renderLightboxContent(); }
      if (e.key === "Escape") { lb.remove(); document.removeEventListener("keydown", onKey); }
    }
    document.addEventListener("keydown", onKey);
    lb.addEventListener("click", () => document.removeEventListener("keydown", onKey), { once: true });
  }

  // ── Thermal lightbox (full-width, shows T/φ/ρ panels) ──────────────────────
  function _showThermalLightbox(iterNum, allIters) {
    const idx = allIters.findIndex((it) => Number(it.iter) === iterNum);
    const entry = allIters[idx];
    if (!entry || !entry.thermal_png_b64) return;

    const lb = document.createElement("div");
    lb.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.92);z-index:9999;" +
      "display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;";
    lb.addEventListener("click", (e) => { if (e.target === lb) lb.remove(); });

    const prevBtn = document.createElement("button");
    prevBtn.textContent = "◀";
    prevBtn.style.cssText = "position:absolute;left:16px;top:50%;transform:translateY(-50%);" +
      "font-size:24px;background:none;border:none;color:#aaa;cursor:pointer;padding:12px;";
    const nextBtn = document.createElement("button");
    nextBtn.textContent = "▶";
    nextBtn.style.cssText = "position:absolute;right:16px;top:50%;transform:translateY(-50%);" +
      "font-size:24px;background:none;border:none;color:#aaa;cursor:pointer;padding:12px;";

    let currentIdx = idx;
    const img = document.createElement("img");
    img.style.cssText = "max-height:80vh;max-width:90vw;border:2px solid #4a6090;" +
      "border-radius:4px;background:#111;";
    const label = document.createElement("div");
    label.style.cssText = "color:#c8dcff;font-size:13px;font-family:monospace;text-align:center;";
    const hint = document.createElement("div");
    hint.style.cssText = "color:#555;font-size:11px;";
    hint.textContent = "← → to navigate between iterations • click outside to close";

    function renderThLb() {
      const cur = allIters[currentIdx];
      if (!cur.thermal_png_b64) {
        img.src = "";
        label.textContent = `Iter ${cur.iter} — no thermal plot`;
      } else {
        img.src = `data:image/png;base64,${cur.thermal_png_b64}`;
        const isBestLb = Number(cur.iter) === bestIter;
        label.textContent = `Iter ${cur.iter}${isBestLb?" ★ best":""} — T/φ/ρ fields` +
          `  ρ̄=${Number(cur.mean_rho||0).toFixed(3)}` +
          `  σ_T=${Number(cur.sigma_T||0).toFixed(1)}°C` +
          `  bleed=${(Number(cur.frac_bleed||0)*100).toFixed(2)}%`;
      }
      prevBtn.disabled = currentIdx === 0;
      nextBtn.disabled = currentIdx === allIters.length - 1;
      prevBtn.style.opacity = prevBtn.disabled ? "0.2" : "1";
      nextBtn.style.opacity = nextBtn.disabled ? "0.2" : "1";
    }

    prevBtn.addEventListener("click", () => { if (currentIdx > 0) { currentIdx--; renderThLb(); } });
    nextBtn.addEventListener("click", () => { if (currentIdx < allIters.length-1) { currentIdx++; renderThLb(); } });

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕";
    closeBtn.style.cssText = "position:absolute;top:12px;right:16px;font-size:18px;" +
      "background:none;border:none;color:#aaa;cursor:pointer;";
    closeBtn.addEventListener("click", () => lb.remove());

    function onThKey(e) {
      if (e.key === "ArrowLeft"  && currentIdx > 0)                   { currentIdx--; renderThLb(); }
      if (e.key === "ArrowRight" && currentIdx < allIters.length - 1) { currentIdx++; renderThLb(); }
      if (e.key === "Escape") { lb.remove(); document.removeEventListener("keydown", onThKey); }
    }
    document.addEventListener("keydown", onThKey);
    lb.addEventListener("click", () => document.removeEventListener("keydown", onThKey), {once:true});

    lb.appendChild(prevBtn); lb.appendChild(nextBtn); lb.appendChild(closeBtn);
    lb.appendChild(img); lb.appendChild(label); lb.appendChild(hint);
    document.body.appendChild(lb);
    renderThLb();
  }

  // ── Gallery helper ───────────────────────────────────────────────────────────
  function _showFgmGallery(allIters) {
    const withPng = allIters.filter((it) => it.fgm_png_b64);
    if (!withPng.length) return;

    const overlay = document.createElement("div");
    overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.95);z-index:9999;" +
      "overflow-y:auto;padding:24px;box-sizing:border-box;";
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕ Close Gallery";
    closeBtn.style.cssText = "position:sticky;top:0;float:right;z-index:1;padding:6px 14px;" +
      "background:#2a2a3a;border:1px solid #5a5a7a;color:#ccc;border-radius:4px;cursor:pointer;" +
      "font-size:12px;margin-bottom:12px;";
    closeBtn.addEventListener("click", () => overlay.remove());

    const title = document.createElement("div");
    title.style.cssText = "color:#90b0d8;font-size:13px;font-family:monospace;margin-bottom:16px;";
    title.textContent = `FGM Gallery — ${withPng.length} iteration(s) • click image to zoom`;

    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;gap:16px;" +
      `grid-template-columns:repeat(auto-fill,minmax(200px,1fr));`;

    const withAny = allIters.filter((it) => it.fgm_png_b64 || it.thermal_png_b64);
    withAny.forEach((entry) => {
      const isBestG = Number(entry.iter) === bestIter;
      const card = document.createElement("div");
      card.style.cssText = `background:#0e1520;border-radius:6px;padding:10px;` +
        `border:2px solid ${isBestG ? "#4a8a40" : "#2a3a5a"};`;

      // FGM image
      if (entry.fgm_png_b64) {
        const fgmLabel = document.createElement("div");
        fgmLabel.style.cssText = "font-size:9px;color:#80a860;margin-bottom:2px;font-family:monospace;";
        fgmLabel.textContent = "FGM saturation";
        const img = document.createElement("img");
        img.src = `data:image/png;base64,${entry.fgm_png_b64}`;
        img.style.cssText = "width:100%;image-rendering:pixelated;border-radius:3px;cursor:pointer;" +
          "background:#111;display:block;margin-bottom:6px;border:1px solid #3a5a3a;";
        img.title = `FGM — iter ${entry.iter} (click to zoom)`;
        img.addEventListener("click", () => _showFgmLightbox(Number(entry.iter), allIters));
        card.appendChild(fgmLabel);
        card.appendChild(img);
      }

      // Thermal image
      if (entry.thermal_png_b64) {
        const thLabel = document.createElement("div");
        thLabel.style.cssText = "font-size:9px;color:#6080a8;margin-bottom:2px;font-family:monospace;";
        thLabel.textContent = "T / φ / ρ fields";
        const thImg = document.createElement("img");
        thImg.src = `data:image/png;base64,${entry.thermal_png_b64}`;
        thImg.style.cssText = "width:100%;border-radius:3px;cursor:pointer;" +
          "background:#111;display:block;margin-bottom:8px;border:1px solid #3a4a6a;";
        thImg.title = `Thermal fields — iter ${entry.iter} (click to zoom)`;
        thImg.addEventListener("click", () => _showThermalLightbox(Number(entry.iter), allIters));
        card.appendChild(thLabel);
        card.appendChild(thImg);
      }

      const info = document.createElement("div");
      info.style.cssText = "font-size:11px;font-family:monospace;color:#90a8c8;line-height:1.6;";
      info.innerHTML =
        `<strong style="color:${isBestG?"#f0c060":"#c8dcff"};">` +
        `Iter ${entry.iter}${isBestG?" ★":""}` +
        `</strong><br>` +
        `ρ̄ = ${Number(entry.mean_rho||0).toFixed(3)}<br>` +
        `σ_T = ${Number(entry.sigma_T||0).toFixed(1)}°C<br>` +
        `bleed = ${(Number(entry.frac_bleed||0)*100).toFixed(2)}%<br>` +
        `score = ${Number(entry.score||0).toFixed(3)} ` +
        `<span style="font-size:9px;color:${entry.score_regime==="active"?"#70c870":"#80a0d0"};">` +
        `${entry.score_regime||""}</span>`;

      card.appendChild(info);
      grid.appendChild(card);
    });

    overlay.appendChild(closeBtn);
    overlay.appendChild(title);
    overlay.appendChild(grid);
    document.body.appendChild(overlay);
  }

  // ── Inject thumbnail images via DOM ─────────────────────────────────────────
  body.querySelectorAll(".conv-thumb-slot").forEach((slot) => {
    const iterNum  = parseInt(slot.dataset.iter, 10);
    const imgType  = slot.dataset.type || "fgm";   // "fgm" | "thermal"
    const entry    = iters.find((it) => Number(it.iter) === iterNum);
    if (!entry) return;

    const b64 = imgType === "thermal" ? entry.thermal_png_b64 : entry.fgm_png_b64;
    if (!b64) return;

    const isFgm    = imgType === "fgm";
    const borderColor = isFgm ? "#3a5a3a" : "#3a4a6a";
    const label    = isFgm ? "FGM map" : "T/ρ fields";

    const img = document.createElement("img");
    img.src = `data:image/png;base64,${b64}`;
    // FGM maps are square-ish; thermal plots are wide (3 panels) → different aspect
    img.style.cssText = `width:${isFgm?120:180}px;height:${isFgm?90:60}px;object-fit:contain;` +
      `border:1px solid ${borderColor};border-radius:3px;` +
      `image-rendering:${isFgm?"pixelated":"auto"};background:#111;cursor:zoom-in;` +
      "transition:transform 0.1s;display:block;";
    img.title = `${label} iter ${iterNum} — click to zoom`;
    img.addEventListener("mouseenter", () => { img.style.transform = "scale(1.04)"; });
    img.addEventListener("mouseleave", () => { img.style.transform = ""; });
    img.addEventListener("click", () => {
      if (isFgm) {
        _showFgmLightbox(iterNum, iters);
      } else {
        _showThermalLightbox(iterNum, iters);
      }
    });
    slot.appendChild(img);
  });

  // Wire up "Use this FGM" buttons
  body.querySelectorAll(".conv-use-fgm-btn").forEach((btn) => {
    btn.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      const npzPath = btn.dataset.npz;
      const iterNum = btn.dataset.iter;
      const statusEl = document.getElementById("convDashActionStatus");

      const rawName = window.prompt(
        `Re-simulate using FGM from iteration ${iterNum}.\nOutput run name:`,
        `${runName}_fgmopt_iter${iterNum}`
      );
      if (rawName === null) return;
      const outputName = rawName.trim().replace(/[^a-zA-Z0-9_-]/g, "_") || `${runName}_fgmopt_iter${iterNum}`;

      if (statusEl) { statusEl.style.color = "#aaa"; statusEl.textContent = "Queuing re-simulation…"; }
      btn.disabled = true;

      try {
        const r = await fetchJson("/api/tools/fgm-resimulate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            source_run_dir: runName,
            fgm_npz_path:   npzPath,
            output_name:    outputName,
          }),
        });
        if (!r.ok) throw new Error(r.error || "fgm-resimulate failed");
        if (statusEl) {
          statusEl.style.color = "#80d080";
          statusEl.innerHTML =
            `✓ Job <strong>${r.job_id}</strong> queued — output: ` +
            `<code style="color:#c8dcff">${outputName}</code>. ` +
            `<a href="#" onclick="document.getElementById('convDashOverlay').remove();return false;"
                  style="color:#80a0ff;">Close</a> and check the Queue tab.`;
        }
      } catch (err) {
        if (statusEl) { statusEl.style.color = "#e08080"; statusEl.textContent = `✗ ${err.message}`; }
        btn.disabled = false;
      }
    });
  });

  // ── Helper: send FGM NPZ to Meteor RIP ────────────────────────────────────
  async function _exportNpzToRip(npzPath, iterNum) {
    const statusEl = document.getElementById("convDashActionStatus");

    const layersStr = window.prompt(
      `Export iter-${iterNum} FGM to Meteor RIP.\n\n` +
      `Number of Z-layers to emit (same 2-D FGM repeated per layer):\n` +
      `(Enter 1 for a single test layer, or the total number of print layers in your job)`,
      "1"
    );
    if (layersStr === null) return;  // user cancelled
    const nLayers = Math.max(1, parseInt(layersStr) || 1);

    if (statusEl) { statusEl.style.color = "#aaa"; statusEl.textContent = `Writing ${nLayers} TIFF layer(s) to RIP…`; }

    try {
      const jobBase = (npzPath.split("/").pop() || "fgm").replace(/\.npz$/i, "").replace(/[^a-zA-Z0-9_-]/g, "_");
      const r = await fetchJson("/api/tools/fgm-to-rip", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fgm_npz_path: npzPath,
          n_layers:     nLayers,
          job_name:     jobBase,
          // output_dir omitted → server defaults to rip_output/ sibling of NPZ
        }),
      });
      if (!r.ok) throw new Error(r.error || "fgm-to-rip failed");
      if (statusEl) {
        const dimStr = (r.width_mm != null && r.height_mm != null)
          ? ` &nbsp;·&nbsp; <span style="color:#90d0b0;">${r.width_mm.toFixed(2)} × ${r.height_mm.toFixed(2)} mm` +
            (r.dpi_npz ? ` @ ${r.dpi_npz} DPI` : "") + `</span>`
          : "";
        statusEl.style.color = "#80d080";
        statusEl.innerHTML =
          `✓ ${r.n_layers} TIFF(s) written → ` +
          `<code style="color:#b0f0c8">${r.output_dir}</code>${dimStr} ` +
          `<button onclick="_copyToClipboard('${(r.output_dir||"").replace(/'/g,"\\'")}');"
                   style="background:#1a3d28;border:1px solid #3a7a50;color:#80d0a0;
                          border-radius:3px;cursor:pointer;padding:1px 6px;font-size:10px;">
            copy path
          </button>`;
      }
    } catch (err) {
      if (statusEl) { statusEl.style.color = "#e08080"; statusEl.textContent = `✗ ${err.message}`; }
    }
  }

  // Wire up per-row "📥 Meteor PNG" download buttons
  body.querySelectorAll(".conv-meteor-dl-btn").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const iterNum = btn.dataset.iter;
      const entry   = iters.find((it) => String(it.iter) === String(iterNum));
      if (!entry || !entry.fgm_meteor_png_b64) return;
      // Build a predictable filename
      const fname = `${runName.replace(/\//g,"_")}_fgm_iter${iterNum}_meteor_import.png`;
      const a = document.createElement("a");
      a.href     = `data:image/png;base64,${entry.fgm_meteor_png_b64}`;
      a.download = fname;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      const statusEl = document.getElementById("convDashActionStatus");
      if (statusEl) {
        statusEl.style.color = "#c0a0e8";
        statusEl.textContent =
          `📥 Downloaded: ${fname}  (black=max ink — import directly into Meteor RIP)`;
      }
    });
  });

  // Wire up per-row "→ RIP" buttons
  body.querySelectorAll(".conv-rip-btn").forEach((btn) => {
    btn.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      await _exportNpzToRip(btn.dataset.npz, btn.dataset.iter);
    });
  });

  // Wire up top-level "Export Best → TIFF Stack" button
  const exportBestRipBtn = document.getElementById("convExportBestRip");
  if (exportBestRipBtn) {
    exportBestRipBtn.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      await _exportNpzToRip(exportBestRipBtn.dataset.npz, exportBestRipBtn.dataset.iter);
    });
  }

  // Wire up top-level "Best Meteor PNG" download button
  const exportBestMeteorBtn = document.getElementById("convExportBestMeteorPng");
  if (exportBestMeteorBtn && bestIterData?.fgm_meteor_png_b64) {
    exportBestMeteorBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const fname = `${runName.replace(/\//g,"_")}_fgm_iter${bestIter}_meteor_import.png`;
      const a = document.createElement("a");
      a.href     = `data:image/png;base64,${bestIterData.fgm_meteor_png_b64}`;
      a.download = fname;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      const statusEl = document.getElementById("convDashActionStatus");
      if (statusEl) {
        statusEl.style.color = "#c0a0e8";
        statusEl.textContent = `📥 Downloaded: ${fname}  (black=max ink — import directly into Meteor RIP)`;
      }
    });
  }

  // ── CSV export ─────────────────────────────────────────────────────────────
  const csvBtn = document.getElementById("convExportCSV");
  if (csvBtn) {
    csvBtn.addEventListener("click", () => {
      const cols = [
        "iter","sigma_T","dT_c","sigma_rho","mean_rho","frac_melt","mean_phi",
        "T_mean_c","T_max_c","T_out_mean_c","thermal_selectivity","frac_bleed",
        "magnitude_used","score","output_dir","fgm_npz",
      ];
      const header = cols.join(",");
      const rows = iters.map((it) =>
        cols.map((c) => {
          const v = it[c];
          if (v === null || v === undefined) return "";
          if (typeof v === "string" && v.includes(",")) return `"${v.replace(/"/g,'""')}"`;
          return String(v);
        }).join(",")
      );
      const csv = [header, ...rows].join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href     = url;
      a.download = `convergence_${runName.replace(/\//g,"_")}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  }

  // ── JSON export ────────────────────────────────────────────────────────────
  const jsonBtn = document.getElementById("convExportJSON");
  if (jsonBtn) {
    jsonBtn.addEventListener("click", () => {
      // Strip the large base64 PNG blobs from the export (keep paths only)
      const exportData = JSON.parse(JSON.stringify(data));
      (exportData.iterations || []).forEach((it) => { delete it.fgm_png_b64; });
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href     = url;
      a.download = `convergence_${runName.replace(/\//g,"_")}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  }

  // ── Print / PDF ───────────────────────────────────────────────────────────
  const printBtn = document.getElementById("convPrintBtn");
  if (printBtn) {
    printBtn.addEventListener("click", () => _printConvergenceReport(data, runName));
  }

  // ── FGM + Thermal Gallery ─────────────────────────────────────────────────
  const galleryBtn = document.getElementById("convFgmGallery");
  if (galleryBtn) {
    galleryBtn.addEventListener("click", () => _showFgmGallery(iters));
  }

  // ── FGM Animation ─────────────────────────────────────────────────────────
  const animBtn = document.getElementById("convFgmAnimate");
  if (animBtn) {
    animBtn.addEventListener("click", () => {
      const withPng = iters.filter((it) => it.fgm_png_b64);
      if (withPng.length < 2) { alert("Need at least 2 iterations with FGM images to animate."); return; }

      const existing = document.getElementById("fgmAnimOverlay");
      if (existing) { existing.remove(); return; }

      const overlay = document.createElement("div");
      overlay.id = "fgmAnimOverlay";
      overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,5,0.9);z-index:4000;" +
        "display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;";

      const imgEl = document.createElement("img");
      imgEl.style.cssText = "max-width:80vw;max-height:60vh;border-radius:6px;border:1px solid #2a4060;";

      const label = document.createElement("div");
      label.style.cssText = "color:#90b0d0;font-size:13px;font-family:monospace;text-align:center;";

      const controls = document.createElement("div");
      controls.style.cssText = "display:flex;gap:12px;align-items:center;";

      const playPauseBtn = document.createElement("button");
      playPauseBtn.textContent = "⏸ Pause";
      playPauseBtn.style.cssText = "padding:5px 14px;border-radius:4px;cursor:pointer;font-size:12px;" +
        "background:#1a2a3a;border:1px solid #3a6090;color:#90c8e8;";

      const fpsLabel = document.createElement("span");
      fpsLabel.style.cssText = "color:#607080;font-size:11px;";
      fpsLabel.textContent = "1 fps";

      const closeBtn = document.createElement("button");
      closeBtn.textContent = "✕ Close";
      closeBtn.style.cssText = "padding:5px 14px;border-radius:4px;cursor:pointer;font-size:12px;" +
        "background:#200d0d;border:1px solid #502020;color:#e08080;";

      controls.append(playPauseBtn, fpsLabel, closeBtn);
      overlay.append(imgEl, label, controls);
      document.body.appendChild(overlay);

      let frameIdx = 0;
      let playing = true;
      let intervalId = null;

      function showFrame(i) {
        const it = withPng[i];
        imgEl.src = `data:image/png;base64,${it.fgm_png_b64}`;
        const sigT = it.sigma_T != null ? ` | σ_T=${Number(it.sigma_T).toFixed(1)}°C` : "";
        const rho  = it.mean_rho != null ? ` | ρ̄=${Number(it.mean_rho).toFixed(3)}` : "";
        label.textContent = `iter-${it.iter} (${i+1}/${withPng.length})${sigT}${rho}`;
      }

      function startPlay() {
        intervalId = setInterval(() => {
          frameIdx = (frameIdx + 1) % withPng.length;
          showFrame(frameIdx);
        }, 1000);
      }

      showFrame(0);
      startPlay();

      playPauseBtn.onclick = () => {
        playing = !playing;
        playPauseBtn.textContent = playing ? "⏸ Pause" : "▶ Play";
        if (playing) { startPlay(); } else { clearInterval(intervalId); }
      };

      closeBtn.onclick = () => { clearInterval(intervalId); overlay.remove(); };
      overlay.addEventListener("click", (ev) => {
        if (ev.target === overlay) { clearInterval(intervalId); overlay.remove(); }
      });
    });
  }

  // ── Score Formula Info ─────────────────────────────────────────────────────
  const scoreInfoBtn = document.getElementById("convScoreInfo");
  if (scoreInfoBtn) {
    scoreInfoBtn.addEventListener("click", () => _showScoreExplanation());
  }

  // ── Continue Iterations ────────────────────────────────────────────────────
  const continueBtn = document.getElementById("convContinueBtn");
  if (continueBtn) {
    continueBtn.addEventListener("click", async () => {
      const statusEl = document.getElementById("convDashActionStatus");
      continueBtn.disabled = true;
      continueBtn.textContent = "⏳ Queuing…";
      if (statusEl) statusEl.textContent = "";
      try {
        const resp = await fetchJson("/api/tools/fgm-continue", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_name: runName, n_iterations: 4 }),
        });
        if (resp.ok) {
          if (statusEl) {
            statusEl.style.color = "#80d080";
            statusEl.textContent =
              `✓ Queued 4 more iterations — job ${resp.job_id} (${resp.status}). ` +
              `Close this panel and watch progress in the Jobs tab.`;
          }
          continueBtn.textContent = "✓ Queued";
        } else {
          throw new Error(resp.error || "Unknown error");
        }
      } catch (err) {
        if (statusEl) {
          statusEl.style.color = "#e08080";
          statusEl.textContent = `✗ Continue failed: ${err.message}`;
        }
        continueBtn.disabled = false;
        continueBtn.textContent = "⟳ Continue 4 Iterations";
      }
    });
  }
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

function metricLine(ex, fgmBest) {
  // For FGM iterate runs, show the best-iteration convergence metrics instead.
  if (fgmBest && Object.keys(fgmBest).length > 0) {
    const parts = [];
    const nI = fgmBest.n_iters != null ? fgmBest.n_iters : "?";
    const bi = fgmBest.best_iter != null ? fgmBest.best_iter : "?";
    parts.push(`iter ${bi}/${nI}`);
    if (fgmBest.sigma_T != null) parts.push(`\u03c3_T ${Number(fgmBest.sigma_T).toFixed(1)}\u00b0C`);
    if (fgmBest.mean_rho != null) parts.push(`\u03c1\u0304 ${Number(fgmBest.mean_rho).toFixed(3)}`);
    if (fgmBest.frac_melt != null) parts.push(`\u03c6 ${(Number(fgmBest.frac_melt)*100).toFixed(0)}%`);
    if (fgmBest.frac_bleed != null && Number(fgmBest.frac_bleed) > 0)
      parts.push(`bleed ${(Number(fgmBest.frac_bleed)*100).toFixed(1)}%`);
    if (fgmBest.exposure_minutes) parts.push(`${Number(fgmBest.exposure_minutes).toFixed(1)} min`);
    const status = fgmBest.converged ? " \u2713conv" : fgmBest.aborted ? " \u26a0abort" : "";
    return parts.join(" \u2022 ") + status || "FGM iterate run";
  }
  const parts = [];
  if (ex.model_family) parts.push(`model ${String(ex.model_family)}`);
  if (ex.calibration_version) parts.push(`cal ${String(ex.calibration_version)}`);
  if (ex.ab_bucket_id) parts.push(`ab ${String(ex.ab_bucket_id)}`);
  if (ex.max_T_part_c !== undefined) parts.push(`T\u2191 ${Number(ex.max_T_part_c).toFixed(1)}\u00b0C`);
  if (ex.mean_phi_part !== undefined) parts.push(`\u03c6\u0304 ${Number(ex.mean_phi_part).toFixed(3)}`);
  if (ex.mean_rho_rel_part !== undefined) parts.push(`\u03c1\u0304 ${Number(ex.mean_rho_rel_part).toFixed(3)}`);
  if (ex.t_final_s !== undefined) parts.push(`t ${Number(ex.t_final_s).toFixed(1)} s`);
  return parts.join(" \u2022 ") || "No summary metrics";
}

// Build a static gauge panel for a run card.
// For fgm_iterate runs (fgmBest provided), shows convergence-optimized metrics.
// For all other runs, shows standard thermal metrics from summary_excerpt.
function _buildRunGaugesHTML(ex, fgmBest) {
  if (!ex && !fgmBest) return "";
  const clamp = (v, lo, hi) => Math.min(100, Math.max(0, (v - lo) / (hi - lo) * 100));

  // FGM iterate — show best-iter σ_T / ρ̅ / φ / score
  if (fgmBest && Object.keys(fgmBest).length > 0) {
    const sigT  = fgmBest.sigma_T  != null ? Number(fgmBest.sigma_T)  : null;
    const rho   = fgmBest.mean_rho != null ? Number(fgmBest.mean_rho) : null;
    const phi   = fgmBest.frac_melt!= null ? Number(fgmBest.frac_melt): null;
    const score = fgmBest.score    != null ? Number(fgmBest.score)    : null;
    if (sigT == null && rho == null) return "";
    const sigTPct = sigT != null ? clamp(sigT, 0, 20) : 0;
    const rhoPct  = rho  != null ? clamp(rho, 0.45, 1.0) : 0;
    const phiPct  = phi  != null ? Math.min(100, phi * 100) : 0;
    const scorePct= score!= null ? Math.min(100, score / 5 * 100) : 0;
    const sigTColor = sigT != null ? (sigT < 3 ? "#40c080" : sigT < 5 ? "#e0c060" : "#e05050") : "#666";
    const regime = String(fgmBest.score_regime || "");
    const regimeDot = regime === "near-final"
      ? `<span style="font-size:9px;color:#80c080;" title="Near-final sintering regime"> ●NF</span>`
      : regime === "active"
      ? `<span style="font-size:9px;color:#80a0e0;" title="Active sintering regime"> ●act</span>` : "";
    return `<div class="run-gauges">
      <div class="gauge-row" title="σ_T at best iteration — spatial temperature uniformity. Target: < 3°C">
        <span class="gauge-label" style="color:${sigTColor};">σ_T</span>
        <div class="gauge-track"><div class="gauge-fill" style="width:${sigTPct.toFixed(1)}%;background:${sigTColor};"></div></div>
        <span class="gauge-val" style="color:${sigTColor};">${sigT != null ? sigT.toFixed(1)+"°C" : "—"}${regimeDot}</span>
      </div>
      <div class="gauge-row" title="Mean relative density at best iteration. Target: ≥ 0.95">
        <span class="gauge-label">ρ̅</span>
        <div class="gauge-track"><div class="gauge-fill gauge-dens-fill" style="width:${rhoPct.toFixed(1)}%"></div></div>
        <span class="gauge-val">${rho != null ? rho.toFixed(3) : "—"}</span>
      </div>
      <div class="gauge-row" title="Melt fraction at best iteration — fraction of part above liquidus">
        <span class="gauge-label">φ̅</span>
        <div class="gauge-track"><div class="gauge-fill gauge-melt-fill" style="width:${phiPct.toFixed(1)}%"></div></div>
        <span class="gauge-val">${phi != null ? (phi*100).toFixed(0)+"%" : "—"}</span>
      </div>
      <div class="gauge-row" title="Composite score (lower is better). Combines σ_T, density uniformity, bleed, and thermal selectivity.">
        <span class="gauge-label">score</span>
        <div class="gauge-track"><div class="gauge-fill gauge-err-fill" style="width:${scorePct.toFixed(1)}%"></div></div>
        <span class="gauge-val">${score != null ? score.toFixed(2) : "—"}</span>
      </div>
    </div>`;
  }

  if (!ex) return "";
  const T_max  = Number(ex.max_T_part_c      ?? 0);
  const T_mean = Number(ex.mean_T_part_c     ?? 0);
  const phi    = Number(ex.mean_phi_part     ?? 0);
  const rho    = Number(ex.mean_rho_rel_part ?? 0);
  const err    = ex.energy_err_pct != null ? Number(ex.energy_err_pct) : -1;
  if (T_mean === 0 && T_max === 0 && phi === 0 && rho === 0) return "";
  const T_max_pct  = clamp(T_max,  25, 240);
  const T_mean_pct = clamp(T_mean, 25, 240);
  const phi_pct    = Math.min(100, Math.max(0, phi * 100));
  const rho_pct    = clamp(rho, 0.45, 1.0);
  const errRow = err >= 0 ? `
    <div class="gauge-row" title="Energy balance residual \u2014 lower is better (< 0.1% is excellent)">
      <span class="gauge-label">err</span>
      <div class="gauge-track"><div class="gauge-fill gauge-err-fill" style="width:${Math.min(100, err / 2 * 100).toFixed(1)}%"></div></div>
      <span class="gauge-val">${err.toFixed(3)}%</span>
    </div>` : "";
  return `<div class="run-gauges">
    <div class="gauge-row" title="Peak temperature in part \u2014 values > 245\u00b0C risk thermal damage">
      <span class="gauge-label">T\u2191</span>
      <div class="gauge-track"><div class="gauge-fill gauge-tmax-fill" style="width:${T_max_pct.toFixed(1)}%"></div></div>
      <span class="gauge-val">${T_max.toFixed(1)}\u00b0C</span>
    </div>
    <div class="gauge-row" title="Mean part temperature at end of run">
      <span class="gauge-label">T\u0305</span>
      <div class="gauge-track"><div class="gauge-fill gauge-temp-fill" style="width:${T_mean_pct.toFixed(1)}%"></div></div>
      <span class="gauge-val">${T_mean.toFixed(1)}\u00b0C</span>
    </div>
    <div class="gauge-row" title="Mean melt fraction \u2014 fraction of part above liquidus temperature">
      <span class="gauge-label">\u03c6\u0305</span>
      <div class="gauge-track"><div class="gauge-fill gauge-melt-fill" style="width:${phi_pct.toFixed(1)}%"></div></div>
      <span class="gauge-val">${phi.toFixed(3)}</span>
    </div>
    <div class="gauge-row" title="Mean relative density \u2014 1.0 = fully dense, 0 = powder. Target \u2265 0.95">
      <span class="gauge-label">\u03c1\u0305</span>
      <div class="gauge-track"><div class="gauge-fill gauge-dens-fill" style="width:${rho_pct.toFixed(1)}%"></div></div>
      <span class="gauge-val">${rho.toFixed(3)}</span>
    </div>${errRow}
  </div>`;
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
    if (k === "fgm_iterate") return "FGM Iterative Optimize";
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
  // Multi-word AND: all space-separated tokens must match run name
  const tokens = q ? q.split(/\s+/).filter(Boolean) : [];
  const g = byId("runGroupFilter").value;
  const m = byId("runModeFilter")?.value || "";
  const star = byId("runStarFilter")?.value || "";
  const out = RUNS.filter((r) => {
    if (g && r.group !== g) return false;
    if (m && String(r.run_type || "unknown") !== m) return false;
    if (star === "starred" && !Boolean(r.starred)) return false;
    if (star === "unstarred" && Boolean(r.starred)) return false;
    if (tokens.length > 0) {
      const nameLow = r.name.toLowerCase();
      if (!tokens.every((t) => nameLow.includes(t))) return false;
    }
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
  } else if (mode === "sigma_t_asc") {
    // σ_T: lower is better. Use fgm_best.sigma_T if available, else treat as ∞.
    const getSigT = (r) => {
      const v = r?.fgm_best?.sigma_T;
      return (v != null && Number.isFinite(Number(v))) ? Number(v) : Infinity;
    };
    out.sort((a, b) => getSigT(a) - getSigT(b) || epochOrZero(b.updated_at) - epochOrZero(a.updated_at) || cmpName(a, b));
  } else if (mode === "score_asc") {
    // composite score: lower is better
    const getScore = (r) => {
      const v = r?.fgm_best?.score;
      return (v != null && Number.isFinite(Number(v))) ? Number(v) : Infinity;
    };
    out.sort((a, b) => getScore(a) - getScore(b) || epochOrZero(b.updated_at) - epochOrZero(a.updated_at) || cmpName(a, b));
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

function renderAntennaePanel(detail) {
  const container = byId("antennaeDetailPanel");
  if (!container) return;
  const placement = detail?.antennae_placement;
  if (!placement || !Array.isArray(placement.instances) || placement.instances.length === 0) {
    container.innerHTML = "";
    container.classList.add("hidden");
    return;
  }
  const rows = placement.instances.map((inst, i) => {
    const xmm = ((inst.center_x ?? 0) * 1000).toFixed(1);
    const ymm = ((inst.center_y ?? 0) * 1000).toFixed(1);
    const wx = (inst.size_x_mm ?? inst.size_mm ?? "?");
    const wy = (inst.size_y_mm ?? inst.size_mm ?? "?");
    const wxFmt = typeof wx === "number" ? wx.toFixed(1) : wx;
    const wyFmt = typeof wy === "number" ? wy.toFixed(1) : wy;
    return `<tr><td>${i + 1}</td><td>(${xmm}, ${ymm})</td><td>${wxFmt}&times;${wyFmt}</td></tr>`;
  }).join("");
  container.innerHTML = `
    <div class="ant-detail-heading">Antenna Placement (${placement.instances.length})</div>
    <table class="ant-detail-table">
      <thead><tr><th>#</th><th>Position (mm)</th><th>Size W&times;H (mm)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  container.classList.remove("hidden");
}

const PROFILE_CACHE = new Map();

function _sidebarProfileLine(ctx, vals, mask, vmin, vmax, W, H, color, axis) {
  // axis: "h" (x=position, y=value) or "v" (x=value, y=position)
  const n = vals.length;
  if (!n) return;
  const span = Math.max(vmax - vmin, 1e-12);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < n; i++) {
    const t = (vals[i] - vmin) / span;
    let px, py;
    if (axis === "h") {
      px = (i / (n - 1)) * W;
      py = H - 4 - t * (H - 10);
    } else {
      px = 4 + t * (W - 10);
      py = (i / (n - 1)) * H;
    }
    const inside = mask ? mask[i] : true;
    if (!inside) { started = false; continue; }
    if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

async function _renderProfilePanel(runName) {
  const panel  = byId("profilePanel");
  const hCv    = byId("profileHCanvas");
  const vCv    = byId("profileVCanvas");
  const stats  = byId("profileStats");
  const fieldSel = byId("profileFieldSel");
  if (!panel || !hCv || !vCv) return;

  // Load profile data (cached)
  let pdata = PROFILE_CACHE.get(runName);
  if (!pdata) {
    try {
      pdata = await fetchJson(`/api/profile/${encodeURIComponent(runName)}`);
      if (pdata.error) { panel.classList.add("hidden"); return; }
      PROFILE_CACHE.set(runName, pdata);
    } catch (_) { panel.classList.add("hidden"); return; }
  }

  const avail = pdata.available || [];
  if (!avail.length) { panel.classList.add("hidden"); return; }

  // Populate field selector
  if (fieldSel.dataset.loadedFor !== runName) {
    fieldSel.innerHTML = "";
    avail.forEach((f) => {
      const o = document.createElement("option");
      o.value = o.textContent = f;
      fieldSel.appendChild(o);
    });
    // prefer T_phi90 > T
    const pref = avail.includes("T_phi90") ? "T_phi90" : (avail.includes("T") ? "T" : avail[0]);
    fieldSel.value = pref;
    fieldSel.dataset.loadedFor = runName;
    fieldSel.onchange = () => _drawProfiles(pdata, fieldSel.value, hCv, vCv, stats);
  }

  _drawProfiles(pdata, fieldSel.value, hCv, vCv, stats);
  panel.classList.remove("hidden");
}

function _drawProfiles(pdata, fieldName, hCv, vCv, stats) {
  const prof = (pdata.profiles || {})[fieldName];
  if (!prof) return;

  const x_m = pdata.x_m;
  const y_m = pdata.y_m;

  function _xLabel(i) {
    return x_m ? `${(x_m[i]*1000).toFixed(1)}mm` : String(i);
  }
  function _yLabel(i) {
    return y_m ? `${(y_m[i]*1000).toFixed(1)}mm` : String(i);
  }

  function drawOnCanvas(cv, vals, mask, isHoriz) {
    const W = cv.clientWidth || 200, H = cv.height || 70;
    cv.width = W;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, W, H);

    // grid lines
    ctx.strokeStyle = "#0e2030";
    ctx.lineWidth = 1;
    for (let g = 1; g < 4; g++) {
      const y = H - 4 - (g / 4) * (H - 10);
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    const inside = vals.filter((v, i) => !mask || mask[i]);
    const vmin = Math.min(...inside);
    const vmax = Math.max(...inside);

    // Area fill
    const span = Math.max(vmax - vmin, 1e-12);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < vals.length; i++) {
      if (!mask || mask[i]) {
        const t = (vals[i] - vmin) / span;
        const px = (i / (vals.length - 1)) * W;
        const py = H - 4 - t * (H - 10);
        if (!started) { ctx.moveTo(px, H); ctx.lineTo(px, py); started = true; }
        else ctx.lineTo(px, py);
      }
    }
    if (started) { ctx.lineTo((vals.length-1)/(vals.length-1)*W, H); ctx.closePath(); }
    ctx.fillStyle = "rgba(80,180,220,0.12)";
    ctx.fill();

    _sidebarProfileLine(ctx, vals, mask, vmin, vmax, W, H, "#50b8e0", "h");

    // Axis labels
    ctx.fillStyle = "#3a6070";
    ctx.font = "9px monospace";
    ctx.fillText(`${vmin.toFixed(1)}`, 2, H - 2);
    ctx.fillText(`${vmax.toFixed(1)}`, 2, 10);
    const dir = isHoriz ? "H-centerline" : "V-centerline";
    ctx.fillStyle = "#2a5060";
    ctx.fillText(`${fieldName} ${dir}`, W / 2 - 40, H - 2);
  }

  drawOnCanvas(hCv, prof.h, prof.h_mask, true);
  drawOnCanvas(vCv, prof.v, prof.v_mask, false);

  const hInside = prof.h.filter((_, i) => !prof.h_mask || prof.h_mask[i]);
  const vInside = prof.v.filter((_, i) => !prof.v_mask || prof.v_mask[i]);
  const hMean = hInside.length ? (hInside.reduce((a,b)=>a+b,0)/hInside.length).toFixed(2) : "n/a";
  const vMean = vInside.length ? (vInside.reduce((a,b)=>a+b,0)/vInside.length).toFixed(2) : "n/a";
  if (stats) stats.textContent = `H-mean=${hMean}  V-mean=${vMean}  row=${prof.h_row}  col=${prof.v_col}`;
}

async function updateMetricsPanel() {
  const run = RUNS.find((r) => r.name === SELECTED_RUN);
  if (!run) return;
  const cached = DETAIL_CACHE.get(run.name);
  if (cached) {
    renderMetricsTable(run, cached);
    renderAntennaePanel(cached);
  } else {
    try {
      const detail = await fetchJson(`/api/results/${encodeURIComponent(run.name)}`);
      DETAIL_CACHE.set(run.name, detail);
      renderMetricsTable(run, detail);
      renderAntennaePanel(detail);
    } catch (err) {
      const wrap = byId("metricsTableWrap");
      if (wrap) wrap.textContent = `Failed to load run summary: ${err.message}`;
    }
  }
  // Profile panel — load in background, non-blocking
  _renderProfilePanel(run.name).catch(() => {});
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

  // ── Helper: build and wire one run card element ──────────────────────────
  function _buildCard(run) {
    const card = document.createElement('article');
    card.className = 'run-card';
    if (run.name === SELECTED_RUN) card.classList.add('selected');
    if (run.starred) card.classList.add('starred');
    if (COMPARE_MODE && COMPARE_SELECTED.has(run.name)) card.classList.add('compare-selected');

    const fgmBest = (run.fgm_best && Object.keys(run.fgm_best).length > 0) ? run.fgm_best : null;
    const abortBadge = (fgmBest?.aborted)
      ? `<span style="display:inline-block;margin-top:4px;padding:2px 7px;border-radius:3px;
                      font-size:10px;background:#3a0a0a;border:1px solid #8a2020;color:#f07070;
                      font-family:monospace;white-space:normal;word-break:break-word;">
           ⚠ ABORTED — ${(fgmBest.abort_reason || 'see convergence dashboard').slice(0, 80)}
         </span>`
      : '';
    const allItems = (run.images || []).map((img) => ({ url: img.url, title: `${run.name}/${img.path}` }));
    const hero = run.hero_images || [];
    const caps = Array.isArray(run.backfill_capabilities) ? run.backfill_capabilities : [];

    const fgmBtn = `<button type="button" class="run-fgm-btn"
        title="Generate a functionally graded binder-saturation map from this run's Qrf/T field">
        🎨 FGM</button>`;
    const importFgmBtn = `<button type="button" class="run-import-fgm-btn"
        title="Import an external FGM PNG and run a single simulation with it">
        📥 Import FGM</button>`;
    const convBtn = run.run_type === 'fgm_iterate'
      ? `<button type="button" class="run-conv-btn" title="View FGM convergence dashboard">📊 Convergence</button>`
      : '';
    const fieldsBtn = `<button type="button" class="run-fields-btn"
        title="Interactive field viewer: T, Qrf, rho_rel, T_phi90 with colormap and crosshair">
        🔬 Fields</button>`;
    const isCompSel = COMPARE_SELECTED.has(run.name);
    const compareChk = COMPARE_MODE
      ? `<button type="button" class="run-compare-chk"
             style="padding:2px 8px;font-size:11px;border-radius:3px;cursor:pointer;"
             title="${isCompSel ? 'Deselect' : 'Select for comparison'}">
             ${isCompSel ? '✓ Selected' : '+ Select'}</button>`
      : '';

    const starLabel = run.starred ? '★' : '☆';
    const backfillAction = `<div class="run-actions">
        <button type="button" class="run-star-btn" aria-label="Toggle star"
                title="${run.starred ? 'Unstar' : 'Star this run'}"
                style="min-width:28px;">${starLabel}</button>
        ${compareChk}
        ${caps.length ? `<button type="button" class="run-backfill-btn" title="Quick backfill">↻ Reports</button>` : ''}
        ${caps.length ? `<button type="button" class="run-backfill-adv-btn" title="Advanced backfill">↻ Advanced</button>` : ''}
        ${fgmBtn}${importFgmBtn}${convBtn}${fieldsBtn}
        <button type="button" class="run-delete-btn" title="Delete run" style="color:#e07070;">🗑 Delete</button>
      </div>`;

    const modeBadgeMap = {
      'single': 'single', 'sweep': 'sweep', 'optimizer': 'optimizer',
      'turntable': 'turntable', 'fgm_iterate': 'FGM-iter',
      'orientation_optimizer': 'orient-opt', 'placement_optimizer': 'place-opt',
      'shell_sweep': 'shell-sweep',
    };
    const modeBadge = modeBadgeMap[String(run.run_type || '')] || String(run.run_type || '');
    const createdDate = (run.run_created_at || '').split('T')[0] || 'unknown';
    const diskStr = run.disk_bytes > 0
      ? (run.disk_bytes >= 1e9 ? `${(run.disk_bytes/1e9).toFixed(1)} GB`
       : run.disk_bytes >= 1e6 ? `${(run.disk_bytes/1e6).toFixed(0)} MB`
       : `${(run.disk_bytes/1e3).toFixed(0)} KB`)
      : '';

    card.innerHTML = `
      <div class="run-head">
        <div style="flex:1;min-width:0;">
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
            <strong style="font-size:13px;word-break:break-all;">${run.name}</strong>
            <span style="font-size:10px;padding:1px 6px;border-radius:3px;background:#0d1f30;
                         border:1px solid #1e3a50;color:#70a0c0;white-space:nowrap;"
                  title="Run mode">${modeBadge}</span>
            ${fgmBest && fgmBest.converged ? `<span style="font-size:10px;padding:1px 5px;border-radius:3px;background:#0d2010;border:1px solid #2a5020;color:#70c070;" title="FGM converged">✓ conv</span>` : ''}
            ${fgmBest && fgmBest.aborted  ? `<span style="font-size:10px;padding:1px 5px;border-radius:3px;background:#201008;border:1px solid #503020;color:#d09060;" title="Aborted early">⚠ abort</span>` : ''}
          </div>
          <div class="muted" style="font-size:11px;margin-top:2px;">
            ${run.group} &bull; ${createdDate} &bull; ${run.image_count} img${diskStr ? `<span style="font-size:10px;color:#3a5060;margin-left:6px;" title="Disk usage for this run">${diskStr}</span>` : ''}
          </div>
        </div>
        <div style="flex:0 0 auto;min-width:140px;text-align:right;">
          <div class="muted run-metric-line" style="font-size:11px;">${metricLine(run.summary_excerpt || {}, fgmBest)}</div>
          ${abortBadge}
          ${_buildRunGaugesHTML(run.summary_excerpt || {}, fgmBest)}
        </div>
      </div>
      ${backfillAction}
      <div class="run-hero"></div>
      <details class="run-details">
        <summary>All images (${run.image_count})</summary>
        <div class="run-all-images"></div>
      </details>
    `;

    card.addEventListener('click', (ev) => {
      if (ev.target instanceof HTMLElement && ev.target.closest('button,summary,a,input,select,textarea')) return;
      if (COMPARE_MODE) {
        if (COMPARE_SELECTED.has(run.name)) {
          COMPARE_SELECTED.delete(run.name);
        } else if (COMPARE_SELECTED.size < 4) {
          COMPARE_SELECTED.add(run.name);
        }
        _updateCompareBar();
        renderRunCards();
        return;
      }
      SELECTED_RUN = run.name;
      _pushUrlState(run.name);
      renderRunCards();
      void updateMetricsPanel();
    });

    return { card, allItems, hero, caps };
  }

  // ── Wire buttons after innerHTML is set ──────────────────────────────────────
  function _wireCardHandlers(card, run, allItems, hero, caps) {
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
    const fgmActionBtn = card.querySelector(".run-fgm-btn");
    if (fgmActionBtn) {
      fgmActionBtn.addEventListener("click", async (ev) => {
        ev.stopPropagation();

        // Collect parameters via simple prompts
        const bppRaw = window.prompt(
          "FGM bits-per-pixel (2 = 4 levels, 4 = 16 levels):", "2"
        );
        if (bppRaw === null) return;  // cancelled
        const bpp = parseInt(bppRaw, 10);
        if (![2, 4].includes(bpp)) {
          window.alert("bpp must be 2 or 4.");
          return;
        }

        const proxyRaw = window.prompt(
          "Proxy field (Qrf | T | rho_rel):\n" +
          "  Qrf     = RF power deposition  [overheated → less ink]  ← recommended\n" +
          "  T       = final temperature    [overheated → less ink]\n" +
          "  rho_rel = relative density     [under-dense → more ink]",
          "Qrf"
        );
        if (proxyRaw === null) return;
        const proxy = proxyRaw.trim() || "Qrf";

        const magRaw = window.prompt(
          "Gradient magnitude:\n" +
          "  0.0 = flat (uniform saturation, no FGM effect)\n" +
          "  0.5 = gentle gradient  ← good starting point\n" +
          "  1.0 = full contrast\n" +
          "  1.5 = exaggerated",
          "0.5"
        );
        if (magRaw === null) return;
        const magnitude = parseFloat(magRaw);
        if (isNaN(magnitude) || magnitude < 0) {
          window.alert("Magnitude must be a non-negative number.");
          return;
        }

        const statusEl = byId("serverStatus");
        if (statusEl) statusEl.textContent = `Generating FGM for ${run.name}…`;
        fgmActionBtn.disabled = true;
        fgmActionBtn.textContent = "Generating…";

        try {
          const resp = await fetchJson("/api/tools/generate-fgm", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              output_dir:  `outputs_eqs/${run.name}`,
              bpp,
              proxy_field: proxy,
              invert:      true,
              magnitude,
            }),
          });

          if (!resp.ok) throw new Error(resp.error || "generate-fgm failed");

          // Show inline result summary + preview image
          const stats = resp.level_stats || {};
          const summary =
            `FGM generated (${resp.bpp}bpp, proxy=${resp.proxy_field}, mag=${resp.magnitude.toFixed(2)})\n` +
            `  Levels: ${(stats.unique || []).join(", ")}   mean=${(stats.mean || 0).toFixed(2)}\n` +
            `  NPZ : ${resp.npz_path || "—"}\n` +
            `  PNG : ${resp.png_path || "—"}`;
          if (statusEl) statusEl.textContent = summary;

          // If we have a preview PNG, pop open a modal/overlay
          if (resp.png_b64) {
            _showFgmPreview(run.name, resp);
          } else {
            window.alert(summary);
          }
        } catch (err) {
          if (statusEl) statusEl.textContent = `FGM error: ${err.message}`;
          window.alert(`FGM generation failed: ${err.message}`);
        } finally {
          fgmActionBtn.disabled = false;
          fgmActionBtn.textContent = "Generate FGM";
        }
      });
    }

    // ── Import FGM PNG ─────────────────────────────────────────────────────────
    const importFgmActionBtn = card.querySelector(".run-import-fgm-btn");
    if (importFgmActionBtn) {
      importFgmActionBtn.addEventListener("click", async (ev) => {
        ev.stopPropagation();

        // Create an invisible file input, trigger it, then read the result
        const fileInput = document.createElement("input");
        fileInput.type   = "file";
        fileInput.accept = ".png,image/png";
        fileInput.style.display = "none";
        document.body.appendChild(fileInput);

        fileInput.addEventListener("change", async () => {
          const file = fileInput.files[0];
          document.body.removeChild(fileInput);
          if (!file) return;

          // Ask for bpp and output_name
          const bppRaw = window.prompt(
            `Import FGM "${file.name}" — bits-per-pixel used when generating this PNG (2 or 4):`, "2"
          );
          if (bppRaw === null) return;
          const bppImp = parseInt(bppRaw, 10);
          if (![2, 4].includes(bppImp)) { window.alert("bpp must be 2 or 4."); return; }

          const outName = window.prompt(
            "Output run name (leave blank for auto-generated):", ""
          );
          if (outName === null) return;

          importFgmActionBtn.disabled = true;
          importFgmActionBtn.textContent = "Importing…";
          const statusEl = byId("serverStatus");
          if (statusEl) statusEl.textContent = `Importing FGM PNG for ${run.name}…`;

          try {
            // Read file as base64
            const png_b64 = await new Promise((resolve, reject) => {
              const reader = new FileReader();
              reader.onload = (e) => {
                // result is "data:image/png;base64,XXXXX" – strip prefix
                const b64 = e.target.result.split(",")[1];
                resolve(b64);
              };
              reader.onerror = reject;
              reader.readAsDataURL(file);
            });

            const resp = await fetchJson("/api/tools/import-fgm-png", {
              method:  "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                source_run_dir: run.name,
                png_b64,
                bpp:         bppImp,
                output_name: outName.trim(),
                invert_vis:  true,
              }),
            });

            if (!resp.ok) throw new Error(resp.error || "import-fgm-png failed");
            const msg =
              `✓ FGM imported → job ${resp.job_id} (${resp.status})\n` +
              `  Output run: ${resp.output_name}\n` +
              `  NPZ: ${resp.npz_path}\n` +
              `  Level map: ${(resp.level_map_shape || []).join("×")} px  ${resp.bpp}bpp`;
            if (statusEl) statusEl.textContent = msg;
            window.alert(msg);
            setTimeout(() => refresh(), 2000);
          } catch (err) {
            if (statusEl) statusEl.textContent = `Import FGM error: ${err.message}`;
            window.alert(`FGM import failed: ${err.message}`);
          } finally {
            importFgmActionBtn.disabled = false;
            importFgmActionBtn.textContent = "📥 Import FGM";
          }
        });

        fileInput.click();
      });
    }

    const convActionBtn = card.querySelector(".run-conv-btn");
    if (convActionBtn) {
      convActionBtn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        convActionBtn.disabled = true;
        convActionBtn.textContent = "Loading…";
        try {
          await _showConvergenceDashboard(run.name);
        } catch (err) {
          const status = byId("serverStatus");
          if (status) status.textContent = `Convergence dashboard error: ${err.message}`;
        } finally {
          convActionBtn.disabled = false;
          convActionBtn.textContent = "📊 Convergence";
        }
      });
    }

    const fieldsActionBtn = card.querySelector(".run-fields-btn");
    if (fieldsActionBtn) {
      fieldsActionBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        _showFieldViewer(run.name);
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
  }

  // ── Render: flat or grouped tree ───────────────────────────────────────────────
  if (VIEW_MODE === 'grouped') {
    const groups = {};
    rows.forEach((run) => {
      const g = run.group || 'other';
      const m = String(run.run_type || 'unknown');
      if (!groups[g]) groups[g] = {};
      if (!groups[g][m]) groups[g][m] = [];
      groups[g][m].push(run);
    });
    Object.keys(groups).sort().forEach((shape) => {
      const groupBody = document.createElement('div');
      const shapeCount = Object.values(groups[shape]).reduce((s, a) => s + a.length, 0);
      const shapeHeader = document.createElement('div');
      shapeHeader.style.cssText = 'padding:8px 4px 4px;font-size:13px;font-weight:bold;' +
        'color:#90b0d0;border-bottom:1px solid #1e2d44;margin:8px 0 4px;' +
        'display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none;';
      shapeHeader.innerHTML = `<span>▼ ${shape}</span>` +
        `<span style="font-size:11px;color:#506070;font-weight:normal;">${shapeCount} run${shapeCount !== 1 ? 's' : ''}</span>`;
      shapeHeader.onclick = () => {
        const collapsed = groupBody.style.display === 'none';
        groupBody.style.display = collapsed ? '' : 'none';
        shapeHeader.querySelector('span').textContent = (collapsed ? '▼ ' : '► ') + shape;
      };
      root.appendChild(shapeHeader);
      root.appendChild(groupBody);
      Object.keys(groups[shape]).sort().forEach((mode) => {
        const modeHeader = document.createElement('div');
        modeHeader.style.cssText = 'padding:2px 12px;font-size:11px;color:#607080;';
        modeHeader.textContent = mode + ` (${groups[shape][mode].length})`;
        groupBody.appendChild(modeHeader);
        groups[shape][mode].forEach((run) => {
          const { card, allItems, hero, caps } = _buildCard(run);
          _wireCardHandlers(card, run, allItems, hero, caps);
          groupBody.appendChild(card);
        });
      });
    });
  } else {
    rows.forEach((run) => {
      const { card, allItems, hero, caps } = _buildCard(run);
      _wireCardHandlers(card, run, allItems, hero, caps);
      root.appendChild(card);
    });
  }
  void updateMetricsPanel();
}

async function refresh() {
  RUNS = await fetchJson("/api/results-runview");
  renderGroupFilter(RUNS);
  renderModeFilter(RUNS);
  renderRunCards();
}

// ── Compare mode helpers ──────────────────────────────────────────────────────
function _updateCompareBar() {
  const bar = byId("compareBar");
  const countEl = byId("compareCount");
  const btn = byId("compareRunsBtn");
  if (!bar) return;
  bar.style.display = COMPARE_MODE ? "flex" : "none";
  if (countEl) countEl.textContent = `${COMPARE_SELECTED.size} selected`;
  if (btn) btn.disabled = COMPARE_SELECTED.size < 2;
}

function _showCompareModal(runNames) {
  const runs = runNames.map((n) => RUNS.find((r) => r.name === n)).filter(Boolean);
  if (!runs.length) return;

  const existing = document.getElementById("compareModalOverlay");
  if (existing) existing.remove();

  // Build columns: one per run
  const metricKeys = [
    { key: "sigma_T",    label: "σ_T (°C)",    src: "fgm", fmt: (v) => v != null ? Number(v).toFixed(2) : "—", good: "low",  target: "< 3°C" },
    { key: "mean_rho",   label: "ρ̄",           src: "fgm", fmt: (v) => v != null ? Number(v).toFixed(4) : "—", good: "high", target: "≥ 0.95" },
    { key: "frac_melt",  label: "φ (melt%)",   src: "fgm", fmt: (v) => v != null ? (Number(v)*100).toFixed(1)+"%" : "—", good: "high", target: "" },
    { key: "frac_bleed", label: "bleed%",       src: "fgm", fmt: (v) => v != null ? (Number(v)*100).toFixed(2)+"%" : "—", good: "low",  target: "0%" },
    { key: "score",      label: "Score",        src: "fgm", fmt: (v) => v != null ? Number(v).toFixed(3) : "—", good: "low",  target: "" },
    { key: "best_iter",  label: "Best iter",    src: "fgm", fmt: (v) => v != null ? String(v) : "—", good: null, target: "" },
    { key: "max_T_part_c",      label: "T_max (°C)",   src: "ex", fmt: (v) => v != null ? Number(v).toFixed(1) : "—", good: "low",  target: "" },
    { key: "mean_T_part_c",     label: "T_mean (°C)",  src: "ex", fmt: (v) => v != null ? Number(v).toFixed(1) : "—", good: null, target: "" },
    { key: "mean_phi_part",     label: "φ_part",       src: "ex", fmt: (v) => v != null ? Number(v).toFixed(4) : "—", good: "high", target: "" },
    { key: "mean_rho_rel_part", label: "ρ_part",       src: "ex", fmt: (v) => v != null ? Number(v).toFixed(4) : "—", good: "high", target: "" },
    { key: "t_final_s",         label: "Exposure (s)", src: "ex", fmt: (v) => v != null ? Number(v).toFixed(1) : "—", good: null, target: "" },
  ];

  const headerRow = `<tr><th style="text-align:left;padding:4px 8px;color:#70a0c0;">Metric</th>` +
    runs.map((r) => `<th style="padding:4px 8px;color:#90b0d0;max-width:160px;word-break:break-all;">${r.name.split("/").pop()}</th>`).join("") +
    `</tr>`;

  const dataRows = metricKeys.map(({ key, label, src, fmt, good, target }) => {
    const vals = runs.map((r) => {
      if (src === "fgm") return r.fgm_best ? r.fgm_best[key] : null;
      if (src === "ex")  return r.summary_excerpt ? r.summary_excerpt[key] : null;
      return null;
    });
    const nums = vals.map(Number).filter((n) => Number.isFinite(n));
    const bestNum = nums.length ? (good === "high" ? Math.max(...nums) : good === "low" ? Math.min(...nums) : null) : null;
    const cells = vals.map((v) => {
      const num = Number(v);
      const isBest = good && Number.isFinite(num) && num === bestNum;
      const style = isBest ? "background:#0d2a10;color:#70e070;font-weight:bold;" : "";
      return `<td style="padding:4px 8px;text-align:center;${style}">${fmt(v)}</td>`;
    }).join("");
    return `<tr style="border-bottom:1px solid #111a28;">
      <td style="padding:4px 8px;color:#8090a0;font-size:11px;" title="${target}">${label}</td>${cells}</tr>`;
  }).join("");

  const overlay = document.createElement("div");
  overlay.id = "compareModalOverlay";
  overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,5,0.85);z-index:3000;display:flex;align-items:flex-start;justify-content:center;padding-top:40px;overflow-y:auto;";
  overlay.innerHTML = `
    <div style="background:#0a0f1a;border:1px solid #1e3050;border-radius:10px;padding:20px;
                max-width:90vw;min-width:400px;position:relative;">
      <button id="compareModalClose" style="position:absolute;top:10px;right:14px;background:none;
              border:none;color:#8090a0;font-size:18px;cursor:pointer;">✕</button>
      <div style="font-size:14px;font-weight:bold;color:#90b0d0;margin-bottom:12px;">
        ⚖ Run Comparison (${runs.length} runs)
        <span style="font-size:10px;color:#506070;margin-left:8px;">Green highlight = best value in row</span>
      </div>
      <div style="overflow-x:auto;">
        <table style="border-collapse:collapse;width:100%;font-size:12px;color:#c0c8d8;">
          <thead style="background:#0d1a2a;">${headerRow}</thead>
          <tbody>${dataRows}</tbody>
        </table>
      </div>
      <div style="margin-top:12px;font-size:11px;color:#506070;">
        Runs: ${runs.map((r) => r.name).join(" | ")}
      </div>
    </div>`;

  overlay.addEventListener("click", (ev) => {
    if (ev.target === overlay || byId("compareModalClose")?.contains(ev.target)) overlay.remove();
  });
  document.body.appendChild(overlay);
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
function _initKeyboardShortcuts() {
  document.addEventListener("keydown", (ev) => {
    // Skip if focus is in an input/textarea/select
    const tag = document.activeElement?.tagName?.toLowerCase() || "";
    if (["input", "textarea", "select"].includes(tag)) return;

    if (ev.key === "/") {
      ev.preventDefault();
      byId("runSearch")?.focus();
    } else if (ev.key === "r" || ev.key === "R") {
      if (!ev.ctrlKey && !ev.metaKey) { ev.preventDefault(); void refresh(); }
    } else if (ev.key === "g" || ev.key === "G") {
      ev.preventDefault();
      VIEW_MODE = VIEW_MODE === "grouped" ? "flat" : "grouped";
      localStorage.setItem("heatr_view_mode", VIEW_MODE);
      _updateViewToggleBtn();
      renderRunCards();
    } else if (ev.key === "ArrowDown") {
      ev.preventDefault();
      const filtered = filterRuns();
      const idx = filtered.findIndex((r) => r.name === SELECTED_RUN);
      if (idx < filtered.length - 1) {
        SELECTED_RUN = filtered[idx + 1].name;
        _pushUrlState(SELECTED_RUN);
        renderRunCards();
        void updateMetricsPanel();
        document.querySelector(".run-card.selected")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      const filtered = filterRuns();
      const idx = filtered.findIndex((r) => r.name === SELECTED_RUN);
      if (idx > 0) {
        SELECTED_RUN = filtered[idx - 1].name;
        _pushUrlState(SELECTED_RUN);
        renderRunCards();
        void updateMetricsPanel();
        document.querySelector(".run-card.selected")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    } else if (ev.key === "Escape") {
      // Close any open overlay
      document.getElementById("compareModalOverlay")?.remove();
      document.getElementById("fgmPreviewOverlay")?.remove();
      document.getElementById("convergenceDashOverlay")?.remove();
    }
  });
}

function _updateViewToggleBtn() {
  const btn = byId("runViewToggle");
  if (!btn) return;
  btn.textContent = VIEW_MODE === "grouped" ? "📋 Flat" : "📁 Group";
  btn.title = VIEW_MODE === "grouped" ? "Switch to flat list view" : "Switch to grouped tree view (by shape → mode)";
}

async function init() {
  _applyUrlState();
  byId("refreshResults").onclick = refresh;
  byId("runSearch").oninput = renderRunCards;
  byId("runGroupFilter").onchange = renderRunCards;
  byId("runModeFilter").onchange = renderRunCards;
  byId("runStarFilter").onchange = renderRunCards;
  byId("runSort").onchange = renderRunCards;

  // View mode toggle
  const viewToggle = byId("runViewToggle");
  if (viewToggle) {
    _updateViewToggleBtn();
    viewToggle.onclick = () => {
      VIEW_MODE = VIEW_MODE === "grouped" ? "flat" : "grouped";
      localStorage.setItem("heatr_view_mode", VIEW_MODE);
      _updateViewToggleBtn();
      renderRunCards();
    };
  }

  // Compare mode toggle
  const compareToggle = byId("runCompareToggle");
  if (compareToggle) {
    compareToggle.onclick = () => {
      COMPARE_MODE = !COMPARE_MODE;
      if (!COMPARE_MODE) COMPARE_SELECTED.clear();
      compareToggle.textContent = COMPARE_MODE ? "✕ Exit Compare" : "⚖ Compare";
      compareToggle.style.background = COMPARE_MODE ? "#1a0d2a" : "";
      compareToggle.style.borderColor = COMPARE_MODE ? "#8060c0" : "";
      compareToggle.style.color = COMPARE_MODE ? "#c090e0" : "";
      _updateCompareBar();
      renderRunCards();
    };
  }

  // Compare bar buttons
  const compareRunsBtn = byId("compareRunsBtn");
  if (compareRunsBtn) {
    compareRunsBtn.onclick = () => {
      _showCompareModal([...COMPARE_SELECTED]);
    };
  }
  const compareCancelBtn = byId("compareCancelBtn");
  if (compareCancelBtn) {
    compareCancelBtn.onclick = () => {
      COMPARE_MODE = false;
      COMPARE_SELECTED.clear();
      const ct = byId("runCompareToggle");
      if (ct) { ct.textContent = "⚖ Compare"; ct.style.background = ct.style.borderColor = ct.style.color = ""; }
      _updateCompareBar();
      renderRunCards();
    };
  }

  _initKeyboardShortcuts();
  await refresh();
  setInterval(refresh, 6000);
}

// ─────────────────────────────────────────────────────────────────────────────
// FGM + Thermal Overlay
// ─────────────────────────────────────────────────────────────────────────────

// ── Print / PDF — dedicated print window ─────────────────────────────────────
function _printConvergenceReport(data, runName) {
  const iters = data.iterations || [];

  // Build FGM image rows (base64 inline — works without a server when printed)
  const fgmRows = iters.map((it) => {
    const img = it.fgm_png_b64
      ? `<img src="data:image/png;base64,${it.fgm_png_b64}"
              style="max-width:160px;max-height:120px;image-rendering:pixelated;
                     border:1px solid #bbb;border-radius:2px;" />`
      : `<span style="color:#aaa;font-size:11px;">no image</span>`;
    const best = (it.iter === data.best_iter)
      ? ' style="background:#fffbe6;font-weight:bold;"' : "";
    const score  = it.score    != null ? Number(it.score).toFixed(3)    : "—";
    const sigT   = it.sigma_T  != null ? Number(it.sigma_T).toFixed(2)  : "—";
    const dT     = it.dT_c     != null ? Number(it.dT_c).toFixed(1)     : "—";
    const rho    = it.mean_rho != null ? Number(it.mean_rho).toFixed(3)  : "—";
    const sigRho = it.sigma_rho!= null ? Number(it.sigma_rho).toFixed(4) : "—";
    const Tmax   = it.T_max_c  != null ? Number(it.T_max_c).toFixed(1)   : "—";
    const bleed  = it.frac_bleed!=null ? (Number(it.frac_bleed)*100).toFixed(2)+"%" : "—";
    const mag    = it.magnitude_used != null ? Number(it.magnitude_used).toFixed(3) : "—";
    const regime = it.score_regime || "";
    const proxy  = it.proxy_field  || data.proxy_field || "—";
    const note   = it.note ? `<br><em style="color:#888;font-size:10px;">${it.note}</em>` : "";
    const switchNote = it.proxy_switch
      ? `<br><em style="color:#c06000;font-size:10px;">⇄ ${it.proxy_switch}</em>` : "";
    return `<tr${best}>
      <td>${it.iter}${it.iter === data.best_iter ? " ★" : ""}</td>
      <td>${sigT}</td><td>${dT}</td><td>${sigRho}</td><td>${rho}</td>
      <td>${Tmax}</td><td>${bleed}</td><td>${score}</td><td>${regime}</td>
      <td>${mag}</td><td style="font-size:10px;">${proxy}</td>
      <td>${img}${note}${switchNote}</td>
    </tr>`;
  }).join("\n");

  const optLine = data.optimizer_chosen_time_min != null
    ? `<p><strong>Optimizer-chosen time:</strong> ${Number(data.optimizer_chosen_time_min).toFixed(2)} min
       (${(data.optimizer_criterion||"").replace("_"," ")})
       ${data.optimizer_reason ? `— ${data.optimizer_reason}` : ""}</p>`
    : "";
  const abortLine = data.aborted
    ? `<p style="color:#c00;font-weight:bold;">⚠ ABORTED: ${data.abort_reason||""}</p>`
    : "";
  const convLine = data.converged
    ? `<p style="color:green;">✔ Converged (σ_T &lt; threshold)</p>`
    : "";

  const html = `<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>HEATR FGM Convergence — ${runName}</title>
<style>
  body { font-family: Arial, sans-serif; font-size: 12px; color: #111; margin: 20px; }
  h1   { font-size: 16px; margin-bottom: 4px; }
  h2   { font-size: 13px; color: #333; margin: 14px 0 4px; border-bottom: 1px solid #bbb; }
  p    { margin: 4px 0; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  th, td { border: 1px solid #bbb; padding: 4px 6px; text-align: center; font-size: 11px; }
  th   { background: #eef; font-weight: bold; }
  tr:nth-child(even) { background: #f9f9f9; }
  .meta { color: #444; font-size: 11px; margin-bottom: 10px; }
  @media print { body { margin: 10px; } }
</style>
</head><body>
<h1>HEATR FGM Convergence Report — ${runName}</h1>
<div class="meta">
  <p><strong>Printed:</strong> ${new Date().toLocaleString()}</p>
  <p><strong>Proxy field:</strong> ${data.proxy_field||"—"} &nbsp;|&nbsp;
     <strong>bpp:</strong> ${data.bpp||"—"} &nbsp;|&nbsp;
     <strong>Initial magnitude:</strong> ${data.magnitude||"—"} &nbsp;|&nbsp;
     <strong>Iterations run:</strong> ${data.n_iterations_run||iters.length} &nbsp;|&nbsp;
     <strong>Best iter:</strong> ${data.best_iter != null ? data.best_iter : "—"} &nbsp;|&nbsp;
     <strong>Best score:</strong> ${data.best_score != null ? Number(data.best_score).toFixed(3) : "—"}
  </p>
  ${optLine}${abortLine}${convLine}
</div>

<h2>Per-iteration metrics</h2>
<p style="font-size:10px;color:#555;">
  σ_T = std dev of T inside part [°C] (primary uniformity metric, lower=better) &nbsp;|&nbsp;
  ΔT = T_max − T̄ [°C] &nbsp;|&nbsp;
  σ_ρ = std dev of relative density &nbsp;|&nbsp;
  ρ̄ = mean relative density (target ≈ 0.92) &nbsp;|&nbsp;
  T_max = peak temperature [°C] &nbsp;|&nbsp;
  bleed% = % of outside-part pixels above melt point &nbsp;|&nbsp;
  score = composite (lower=better) &nbsp;|&nbsp;
  mag = magnitude used this iteration
</p>
<table>
  <thead><tr>
    <th>Iter</th><th>σ_T °C</th><th>ΔT °C</th><th>σ_ρ</th><th>ρ̄</th>
    <th>T_max °C</th><th>bleed%</th><th>score</th><th>regime</th><th>mag</th>
    <th>proxy</th><th>FGM (white=max ink)</th>
  </tr></thead>
  <tbody>${fgmRows}</tbody>
</table>

<p style="margin-top:16px;font-size:10px;color:#888;">
  FGM convention: white = maximum binder saturation (most CB dopant / most ink).
  Black = no ink. Feed PNG directly to Meteor RIP.
  Generated by HEATR — High-frequency Electrothermal Additive Thermal Resolver.
</p>
</body></html>`;

  const win = window.open("", "_blank", "width=1000,height=750,scrollbars=yes");
  if (!win) { alert("Pop-up blocked — please allow pop-ups for this page and try again."); return; }
  win.document.write(html);
  win.document.close();
  win.focus();
  // Small delay lets images load before the print dialog fires
  setTimeout(() => win.print(), 600);
}

function _showFgmOverlay(iters, runName) {
  const existing = document.getElementById("fgmOverlayModal");
  if (existing) { existing.remove(); return; }

  // Collect iters that have both FGM and thermal images
  const bothIters = iters.filter((it) => it.fgm_png_b64 && it.thermal_png_b64);
  const fgmOnly   = iters.filter((it) => it.fgm_png_b64);
  const pool      = bothIters.length > 0 ? bothIters : fgmOnly;
  if (!pool.length) {
    alert("No iterations have FGM images to overlay.");
    return;
  }

  const modal = document.createElement("div");
  modal.id = "fgmOverlayModal";
  modal.style.cssText =
    "position:fixed;inset:0;background:rgba(0,0,0,0.88);z-index:5000;" +
    "display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px;";

  const panel = document.createElement("div");
  panel.style.cssText =
    "max-width:680px;width:100%;background:#08121e;border:1px solid #1e3050;" +
    "border-radius:8px;padding:14px;position:relative;";

  panel.innerHTML = `
    <button id="fovClose" style="position:absolute;top:10px;right:12px;background:none;border:none;
            color:#90b0d0;font-size:18px;cursor:pointer;">✕</button>
    <h3 style="margin:0 0 10px;color:#c8dff0;font-size:14px;">🌡 FGM / Thermal Overlay — <em>${runName}</em></h3>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px;">
      <label style="font-size:12px;color:#7090b0;">Iteration
        <select id="fovIterSel" style="margin-left:4px;font-size:12px;background:#0d1a2e;
                border:1px solid #2a4060;border-radius:3px;color:#c0d0e0;padding:2px 5px;">
          ${pool.map((it) => `<option value="${it.iter}">Iter ${it.iter}</option>`).join("")}
        </select>
      </label>
      <label style="font-size:12px;color:#7090b0;">FGM opacity
        <input id="fovSlider" type="range" min="0" max="100" value="50"
               style="width:120px;margin-left:6px;vertical-align:middle;" />
        <span id="fovSliderVal" style="font-size:11px;color:#90b0d0;margin-left:4px;font-family:monospace;">50%</span>
      </label>
      ${bothIters.length > 0
        ? `<label style="font-size:12px;color:#7090b0;">Layer
            <select id="fovLayerSel" style="margin-left:4px;font-size:12px;background:#0d1a2e;
                    border:1px solid #2a4060;border-radius:3px;color:#c0d0e0;padding:2px 5px;">
              <option value="thermal">Thermal (background)</option>
              <option value="fgm" selected>FGM (background)</option>
            </select>
           </label>`
        : ""}
    </div>
    <div style="position:relative;background:#000;border-radius:4px;overflow:hidden;line-height:0;">
      <img id="fovBase"   style="display:block;width:100%;opacity:1;"    alt="base" />
      <img id="fovTop"    style="display:block;width:100%;position:absolute;inset:0;opacity:0.5;" alt="overlay" />
    </div>
    <div id="fovInfo" style="margin-top:6px;font-size:11px;font-family:monospace;color:#406060;"></div>
  `;

  modal.appendChild(panel);
  document.body.appendChild(modal);

  const iterSel    = panel.querySelector("#fovIterSel");
  const slider     = panel.querySelector("#fovSlider");
  const sliderVal  = panel.querySelector("#fovSliderVal");
  const layerSel   = panel.querySelector("#fovLayerSel");
  const baseImg    = panel.querySelector("#fovBase");
  const topImg     = panel.querySelector("#fovTop");
  const fovInfo    = panel.querySelector("#fovInfo");

  function update() {
    const iterIdx = parseInt(iterSel.value);
    const it = pool.find((x) => x.iter === iterIdx) || pool[0];
    const opacity = parseInt(slider.value) / 100;
    sliderVal.textContent = `${slider.value}%`;

    const hasBoth = !!(it.fgm_png_b64 && it.thermal_png_b64);
    const isTopFgm = !layerSel || layerSel.value === "fgm";

    if (hasBoth) {
      baseImg.src = isTopFgm
        ? `data:image/png;base64,${it.thermal_png_b64}`
        : `data:image/png;base64,${it.fgm_png_b64}`;
      topImg.src = isTopFgm
        ? `data:image/png;base64,${it.fgm_png_b64}`
        : `data:image/png;base64,${it.thermal_png_b64}`;
      topImg.style.opacity = opacity;
      topImg.style.mixBlendMode = isTopFgm ? "multiply" : "screen";
    } else if (it.fgm_png_b64) {
      baseImg.src = `data:image/png;base64,${it.fgm_png_b64}`;
      topImg.src = "";
      topImg.style.opacity = 0;
    }

    const sigT = it.sigma_T != null ? `σ_T=${Number(it.sigma_T).toFixed(1)}°C` : "";
    const rho  = it.mean_rho != null ? `ρ̄=${Number(it.mean_rho).toFixed(3)}` : "";
    const mag  = it.magnitude_used != null ? `mag=${Number(it.magnitude_used).toFixed(3)}` : "";
    fovInfo.textContent = [sigT, rho, mag].filter(Boolean).join("  |  ");
  }

  iterSel.onchange = update;
  slider.oninput   = update;
  if (layerSel) layerSel.onchange = update;
  panel.querySelector("#fovClose").onclick = () => modal.remove();
  modal.addEventListener("click", (e) => { if (e.target === modal) modal.remove(); });
  document.addEventListener("keydown", function esc(e) {
    if (e.key === "Escape") { modal.remove(); document.removeEventListener("keydown", esc); }
  });

  update();
}

// ─────────────────────────────────────────────────────────────────────────────
// Interactive Field Viewer
// ─────────────────────────────────────────────────────────────────────────────

// Inferno colormap (256 entries, each [r,g,b] 0-255) — approximated
const _INFERNO = (() => {
  const stops = [
    [0,0,4],[1,0,5],[13,8,135],[57,15,112],[87,16,110],[114,31,93],
    [144,12,68],[180,54,60],[200,87,32],[229,134,1],[247,176,1],[252,224,77],[252,255,164]
  ];
  const ramp = new Uint8ClampedArray(256*3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (stops.length - 1);
    const lo = Math.floor(t), hi = Math.min(lo + 1, stops.length - 1);
    const f = t - lo;
    ramp[i*3]   = Math.round(stops[lo][0] + (stops[hi][0]-stops[lo][0])*f);
    ramp[i*3+1] = Math.round(stops[lo][1] + (stops[hi][1]-stops[lo][1])*f);
    ramp[i*3+2] = Math.round(stops[lo][2] + (stops[hi][2]-stops[lo][2])*f);
  }
  return ramp;
})();

// Viridis colormap
const _VIRIDIS = (() => {
  const stops = [
    [68,1,84],[72,40,120],[62,84,139],[49,104,142],[38,130,142],
    [31,158,137],[53,183,121],[110,206,88],[181,222,43],[253,231,37]
  ];
  const ramp = new Uint8ClampedArray(256*3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (stops.length - 1);
    const lo = Math.floor(t), hi = Math.min(lo + 1, stops.length - 1);
    const f = t - lo;
    ramp[i*3]   = Math.round(stops[lo][0] + (stops[hi][0]-stops[lo][0])*f);
    ramp[i*3+1] = Math.round(stops[lo][1] + (stops[hi][1]-stops[lo][1])*f);
    ramp[i*3+2] = Math.round(stops[lo][2] + (stops[hi][2]-stops[lo][2])*f);
  }
  return ramp;
})();

// Coolwarm (blue→white→red)
const _COOLWARM = (() => {
  const stops = [
    [59,76,192],[98,130,234],[141,176,254],[184,208,249],[220,227,243],
    [243,243,243],[250,219,194],[245,177,148],[211,97,72],[180,4,38]
  ];
  const ramp = new Uint8ClampedArray(256*3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (stops.length - 1);
    const lo = Math.floor(t), hi = Math.min(lo + 1, stops.length - 1);
    const f = t - lo;
    ramp[i*3]   = Math.round(stops[lo][0] + (stops[hi][0]-stops[lo][0])*f);
    ramp[i*3+1] = Math.round(stops[lo][1] + (stops[hi][1]-stops[lo][1])*f);
    ramp[i*3+2] = Math.round(stops[lo][2] + (stops[hi][2]-stops[lo][2])*f);
  }
  return ramp;
})();

const _CMAPS = { inferno: _INFERNO, viridis: _VIRIDIS, coolwarm: _COOLWARM };

function _renderFieldToCanvas(canvas, data, mask, vmin, vmax, cmapName, showOutside) {
  const ramp = _CMAPS[cmapName] || _INFERNO;
  const ny = data.length, nx = data[0].length;
  canvas.width  = nx;
  canvas.height = ny;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(nx, ny);
  const span = Math.max(vmax - vmin, 1e-12);
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const idx = (y * nx + x) * 4;
      const inside = mask[y][x];
      if (!inside && !showOutside) {
        img.data[idx]=0; img.data[idx+1]=0; img.data[idx+2]=0; img.data[idx+3]=0;
      } else {
        const t = Math.max(0, Math.min(1, (data[y][x] - vmin) / span));
        const ci = Math.round(t * 255);
        img.data[idx]   = ramp[ci*3];
        img.data[idx+1] = ramp[ci*3+1];
        img.data[idx+2] = ramp[ci*3+2];
        img.data[idx+3] = inside ? 255 : 60;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

function _drawColorbar(cbCanvas, vmin, vmax, cmapName, label) {
  const ramp = _CMAPS[cmapName] || _INFERNO;
  const W = cbCanvas.width, H = cbCanvas.height;
  const ctx = cbCanvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const barH = H - 30, barY = 5;
  for (let i = 0; i < 256; i++) {
    const y = barY + barH - (i / 255) * barH;
    ctx.fillStyle = `rgb(${ramp[i*3]},${ramp[i*3+1]},${ramp[i*3+2]})`;
    ctx.fillRect(0, y, W - 30, barH / 255 + 1);
  }
  ctx.strokeStyle = "#4a6080";
  ctx.strokeRect(0, barY, W - 30, barH);
  ctx.fillStyle = "#c0d0e8";
  ctx.font = "11px monospace";
  ctx.textAlign = "right";
  ctx.fillText(vmax.toFixed(1), W - 2, barY + 10);
  ctx.fillText(((vmin + vmax) / 2).toFixed(1), W - 2, barY + barH / 2 + 4);
  ctx.fillText(vmin.toFixed(1), W - 2, barY + barH + 2);
  ctx.save();
  ctx.translate(W - 8, barY + barH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.font = "10px sans-serif";
  ctx.fillStyle = "#7090b0";
  ctx.fillText(label, 0, 0);
  ctx.restore();
}

async function _showFieldViewer(runName) {
  // Remove any existing viewer
  const existing = document.getElementById("fieldViewerOverlay");
  if (existing) existing.remove();

  // Build overlay
  const overlay = document.createElement("div");
  overlay.id = "fieldViewerOverlay";
  overlay.style.cssText = `
    position:fixed;inset:0;background:rgba(0,0,0,0.82);z-index:9000;
    display:flex;flex-direction:column;align-items:center;justify-content:flex-start;
    overflow-y:auto;padding:16px;
  `;

  overlay.innerHTML = `
    <div id="fvPanel" style="width:100%;max-width:900px;background:#08121e;border:1px solid #1e3050;
         border-radius:8px;padding:14px;position:relative;">
      <button id="fvClose" type="button"
        style="position:absolute;top:10px;right:12px;background:none;border:none;color:#90b0d0;
               font-size:18px;cursor:pointer;line-height:1;" title="Close">✕</button>
      <h3 style="margin:0 0 10px;color:#c8dff0;font-size:15px;">🔬 Field Viewer — <em>${runName}</em></h3>

      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px;">
        <label style="font-size:12px;color:#7090b0;">Field
          <select id="fvFieldSel" style="margin-left:4px;font-size:12px;background:#0d1a2e;border:1px solid #2a4060;
                  border-radius:3px;color:#c0d0e0;padding:2px 6px;"></select>
        </label>
        <label style="font-size:12px;color:#7090b0;">Colormap
          <select id="fvCmapSel" style="margin-left:4px;font-size:12px;background:#0d1a2e;border:1px solid #2a4060;
                  border-radius:3px;color:#c0d0e0;padding:2px 6px;">
            <option value="inferno">Inferno</option>
            <option value="viridis">Viridis</option>
            <option value="coolwarm">Coolwarm</option>
          </select>
        </label>
        <label style="font-size:12px;color:#7090b0;">
          <input type="checkbox" id="fvShowOutside" style="margin-right:4px;" />Show outside part
        </label>
        <span id="fvStatus" style="font-size:11px;color:#4a7090;margin-left:auto;"></span>
      </div>

      <div style="display:flex;gap:8px;align-items:flex-start;">
        <div style="position:relative;flex:1;min-width:0;background:#000;border-radius:4px;overflow:hidden;">
          <canvas id="fvCanvas" style="display:block;width:100%;image-rendering:pixelated;cursor:crosshair;"></canvas>
          <div id="fvCrosshair" style="position:absolute;pointer-events:none;display:none;">
            <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:rgba(255,255,255,0.5);transform:translateX(-50%);"></div>
            <div style="position:absolute;top:50%;left:0;right:0;height:1px;background:rgba(255,255,255,0.5);transform:translateY(-50%);"></div>
          </div>
          <div id="fvTooltip" style="position:absolute;top:8px;left:8px;font-size:11px;font-family:monospace;
               color:#e0f0ff;background:rgba(0,0,0,0.7);padding:3px 7px;border-radius:3px;pointer-events:none;
               display:none;white-space:pre;"></div>
        </div>
        <canvas id="fvColorbar" width="54" style="flex-shrink:0;image-rendering:pixelated;height:260px;"></canvas>
      </div>

      <div id="fvProfileRow" style="margin-top:10px;display:none;">
        <canvas id="fvProfileCanvas" style="width:100%;height:80px;display:block;background:#060e1a;
                border:1px solid #1e3050;border-radius:3px;"></canvas>
      </div>
      <div id="fvStats" style="margin-top:8px;font-size:11px;font-family:monospace;color:#5a8090;"></div>
    </div>
  `;

  document.body.appendChild(overlay);

  const fvClose  = overlay.querySelector("#fvClose");
  const fvCanvas = overlay.querySelector("#fvCanvas");
  const fvCbar   = overlay.querySelector("#fvColorbar");
  const fvField  = overlay.querySelector("#fvFieldSel");
  const fvCmap   = overlay.querySelector("#fvCmapSel");
  const fvStatus = overlay.querySelector("#fvStatus");
  const fvStats  = overlay.querySelector("#fvStats");
  const fvTip    = overlay.querySelector("#fvTooltip");
  const fvCross  = overlay.querySelector("#fvCrosshair");
  const fvOut    = overlay.querySelector("#fvShowOutside");
  const fvProfileRow = overlay.querySelector("#fvProfileRow");
  const fvProfileCv  = overlay.querySelector("#fvProfileCanvas");

  fvClose.onclick = () => overlay.remove();
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
  document.addEventListener("keydown", function onKey(e) {
    if (e.key === "Escape") { overlay.remove(); document.removeEventListener("keydown", onKey); }
  });

  let _fieldData = null;  // current loaded data payload

  async function loadField(fieldName) {
    fvStatus.textContent = "Loading…";
    try {
      const url = `/api/fields/${encodeURIComponent(runName)}?field=${encodeURIComponent(fieldName)}&maxpx=300`;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      _fieldData = await resp.json();
      if (_fieldData.error) throw new Error(_fieldData.error);

      // Populate field selector if not yet done
      if (fvField.options.length === 0) {
        (_fieldData.available_fields || [fieldName]).forEach((f) => {
          const o = document.createElement("option");
          o.value = o.textContent = f;
          if (f === fieldName) o.selected = true;
          fvField.appendChild(o);
        });
      } else {
        fvField.value = fieldName;
      }

      fvStatus.textContent = `${_fieldData.shape[1]}×${_fieldData.shape[0]} px`;
      renderField();
    } catch (e) {
      fvStatus.textContent = `Error: ${e.message}`;
    }
  }

  function renderField() {
    if (!_fieldData) return;
    const cmap = fvCmap.value;
    const showOut = fvOut.checked;
    _renderFieldToCanvas(fvCanvas, _fieldData.data, _fieldData.mask,
      _fieldData.vmin, _fieldData.vmax, cmap, showOut);
    fvCbar.height = 260;
    _drawColorbar(fvCbar, _fieldData.vmin, _fieldData.vmax, cmap, fvField.value);
    fvStats.textContent =
      `min=${_fieldData.vmin.toFixed(2)}  mean=${_fieldData.vmean.toFixed(2)}  max=${_fieldData.vmax.toFixed(2)}` +
      `  shape=${_fieldData.shape[1]}×${_fieldData.shape[0]}`;
  }

  // Crosshair + tooltip on mouse move
  fvCanvas.addEventListener("mousemove", (e) => {
    if (!_fieldData) return;
    const rect = fvCanvas.getBoundingClientRect();
    const scaleX = _fieldData.shape[1] / rect.width;
    const scaleY = _fieldData.shape[0] / rect.height;
    const px = Math.floor((e.clientX - rect.left) * scaleX);
    const py = Math.floor((e.clientY - rect.top)  * scaleY);
    if (px < 0 || py < 0 || px >= _fieldData.shape[1] || py >= _fieldData.shape[0]) return;
    const val = _fieldData.data[py][px];
    const inside = _fieldData.mask[py][px];
    const xm = Array.isArray(_fieldData.x_m) ? _fieldData.x_m[px] : px;
    const ym = Array.isArray(_fieldData.y_m) ? _fieldData.y_m[py] : py;
    const xLabel = typeof xm === 'number' ? `x=${(xm*1000).toFixed(2)}mm` : `x=${px}`;
    const yLabel = typeof ym === 'number' ? `y=${(ym*1000).toFixed(2)}mm` : `y=${py}`;
    fvTip.style.display = "block";
    fvTip.textContent = `${fvField.value}=${val.toFixed(4)}\n${xLabel}  ${yLabel}${inside ? '' : '  (outside)'}`;
    fvTip.style.left = (e.clientX - rect.left + 12) + "px";
    fvTip.style.top  = (e.clientY - rect.top  - 10) + "px";
    fvCross.style.display = "block";
    fvCross.style.left = (e.clientX - rect.left) + "px";
    fvCross.style.top  = (e.clientY - rect.top)  + "px";

    // Horizontal profile at cursor row
    _drawProfile(fvProfileCv, _fieldData.data[py], _fieldData.mask[py],
      _fieldData.vmin, _fieldData.vmax, px, fvField.value,
      Array.isArray(_fieldData.x_m) ? _fieldData.x_m : null);
    fvProfileRow.style.display = "block";
  });
  fvCanvas.addEventListener("mouseleave", () => {
    fvTip.style.display = "none";
    fvCross.style.display = "none";
  });

  fvField.addEventListener("change", () => loadField(fvField.value));
  fvCmap.addEventListener("change", renderField);
  fvOut.addEventListener("change", renderField);

  // Initial load — prefer T_phi90, fallback to T
  await loadField("T_phi90");
}

function _drawProfile(canvas, rowData, rowMask, vmin, vmax, cursorX, label, x_m) {
  const W = canvas.clientWidth || 600, H = canvas.clientHeight || 80;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const nx = rowData.length;
  if (!nx) return;
  const span = Math.max(vmax - vmin, 1e-12);
  ctx.strokeStyle = "#2a4060";
  ctx.lineWidth = 1;
  for (let ticks = 0; ticks <= 4; ticks++) {
    const y = H - 4 - ((ticks / 4) * (H - 12));
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  // Area fill
  ctx.beginPath();
  for (let i = 0; i < nx; i++) {
    const x = (i / (nx - 1)) * W;
    const y = H - 4 - ((rowData[i] - vmin) / span) * (H - 12);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
  ctx.fillStyle = "rgba(64,160,220,0.18)";
  ctx.fill();
  // Line
  ctx.beginPath();
  ctx.strokeStyle = "#50b0e0";
  ctx.lineWidth = 1.5;
  for (let i = 0; i < nx; i++) {
    const x = (i / (nx - 1)) * W;
    const y = H - 4 - ((rowData[i] - vmin) / span) * (H - 12);
    if (!rowMask[i]) ctx.strokeStyle = "#2a5060";
    else ctx.strokeStyle = "#50b0e0";
    if (i === 0) ctx.moveTo(x, y); else { ctx.lineTo(x, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y); }
  }
  // Cursor line
  if (cursorX >= 0 && cursorX < nx) {
    const cx = (cursorX / (nx - 1)) * W;
    ctx.strokeStyle = "rgba(255,220,100,0.7)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
  }
  // Label
  ctx.fillStyle = "#4a7090";
  ctx.font = "9px monospace";
  ctx.fillText(`${label} profile (row)  min=${vmin.toFixed(1)} max=${vmax.toFixed(1)}`, 4, H - 3);
}

function _showScoreExplanation() {
  const existing = document.getElementById("scoreExplainOverlay");
  if (existing) { existing.remove(); return; }
  const overlay = document.createElement("div");
  overlay.id = "scoreExplainOverlay";
  overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.82);z-index:9500;display:flex;align-items:center;justify-content:center;padding:16px;";
  overlay.innerHTML = `
    <div style="max-width:640px;width:100%;background:#08121e;border:1px solid #2a4060;border-radius:8px;padding:20px;position:relative;overflow-y:auto;max-height:90vh;">
      <button onclick="document.getElementById('scoreExplainOverlay').remove()"
              style="position:absolute;top:10px;right:12px;background:none;border:none;color:#90b0d0;font-size:18px;cursor:pointer;">✕</button>
      <h3 style="margin:0 0 14px;color:#c8e0f8;font-size:15px;">📐 HEATR FGM Scoring & Metrics</h3>
      <div style="font-size:12px;color:#8090a8;line-height:1.7;font-family:sans-serif;">

        <p style="color:#c0d0e8;margin:0 0 6px;font-weight:600;">What are we converging toward?</p>
        <p style="margin:0 0 12px;">Each outer iteration re-runs the full HEATR simulation with a new FGM saturation map derived from the previous run's temperature field. The goal is to make the temperature distribution inside the part as <em>uniform</em> as possible at the sintering moment — so that all regions densify equally. A perfectly uniform field would mean every powder particle receives the same energy dose and sinters to the same density.</p>

        <p style="color:#c0d0e8;margin:0 0 6px;font-weight:600;">Primary metric: σ_T (temperature standard deviation)</p>
        <p style="margin:0 0 12px;">σ_T is the spatial standard deviation of temperature inside the part, measured at the T_phi90 snapshot (the moment mean melt fraction φ first crosses 0.90). Lower σ_T = more uniform heating. Target: σ_T &lt; 3°C is excellent; σ_T &lt; 6°C is good; σ_T &gt; 15°C indicates significant non-uniformity.</p>

        <p style="color:#c0d0e8;margin:0 0 6px;font-weight:600;">Composite score formula</p>
        <pre style="background:#060e1a;padding:10px;border-radius:4px;color:#90e080;font-size:11px;overflow-x:auto;">score = 1.0 × σ_T  +  50 × frac_melt  +  20 × σ_ρ  +  0.3 × |ρ̄ − 0.90|</pre>
        <p style="margin:0 0 12px;">Lower score = better. Each term penalises a different failure mode: σ_T penalises non-uniform heating, frac_melt penalises over-melting (too much energy), σ_ρ penalises non-uniform density, and the |ρ̄−0.90| term penalises deviation from the target mean density of 0.90.</p>

        <p style="color:#c0d0e8;margin:0 0 6px;font-weight:600;">Why ρ̄ = 0.82 threshold on the chart?</p>
        <p style="margin:0 0 12px;">ρ̄ = 0.82 is a <em>scoring regime boundary</em>, not the density target. Below 0.82, the part is still in an "active sintering" phase where large density gains are still possible — the score formula above applies. Above 0.82, the part is in a "near-final" phase. The scores from these two regimes use different normalisation and should not be directly compared. The target is always ρ̄ → 1.0; 0.82 just marks the crossover between regime definitions in the scoring system.</p>

        <p style="color:#c0d0e8;margin:0 0 6px;font-weight:600;">Why one point per iteration on the convergence chart?</p>
        <p style="margin:0 0 12px;">Each point on the σ_T chart represents one <em>complete HEATR simulation</em> — typically 200–1000 timesteps of coupled EM + thermal + sintering physics. The FGM is updated <em>between</em> runs (outer loop), not during. So unlike gradient-descent optimizers that evaluate thousands of cheap function calls, each evaluation here is a 2–10 minute simulation. Convergence in 3–5 iterations is the expected and typical behavior.</p>

        <p style="color:#c0d0e8;margin:0 0 6px;font-weight:600;">frac_bleed — what is it?</p>
        <p style="margin:0 0 12px;">Fraction of the part boundary pixels that exceed the melt temperature. Non-zero bleed indicates the melting zone has reached the part edge, which can cause edge delamination or shape distortion. Target: 0.0.</p>

        <p style="color:#c0d0e8;margin:0 0 6px;font-weight:600;">thermal_selectivity</p>
        <p style="margin:0 0 4px;">Ratio of mean temperature inside the part to mean temperature outside it. Higher = better (part heats more than surroundings). Values above 3× are typical for well-coupled CB-doped geometries.</p>
      </div>
    </div>`;
  document.body.appendChild(overlay);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
  document.addEventListener("keydown", function esc(e) {
    if (e.key === "Escape") { overlay.remove(); document.removeEventListener("keydown", esc); }
  });
}

init().catch((err) => {
  const s = byId("serverStatus");
  if (s) s.textContent = "HEATR service unreachable";
});
