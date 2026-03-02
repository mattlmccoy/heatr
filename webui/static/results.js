const byId = (id) => document.getElementById(id);
const viewer = { items: [], index: 0 };
let RUNS = [];

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
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

function filterRuns() {
  const q = (byId("runSearch").value || "").trim().toLowerCase();
  const g = byId("runGroupFilter").value;
  return RUNS.filter((r) => {
    if (g && r.group !== g) return false;
    if (q && !r.name.toLowerCase().includes(q)) return false;
    return true;
  });
}

function renderRunCards() {
  const root = byId("runCards");
  const rows = filterRuns();
  root.innerHTML = "";
  if (!rows.length) {
    root.textContent = "No runs match current filters.";
    return;
  }

  rows.forEach((run) => {
    const card = document.createElement("article");
    card.className = "run-card";

    const allItems = (run.images || []).map((img) => ({ url: img.url, title: `${run.name}/${img.path}` }));
    const hero = run.hero_images || [];

    card.innerHTML = `
      <div class="run-head">
        <div>
          <strong>${run.name}</strong>
          <div class="muted">${run.group} • ${run.updated_at} • ${run.image_count} image(s)</div>
        </div>
        <div class="muted">${metricLine(run.summary_excerpt || {})}</div>
      </div>
      <div class="run-hero"></div>
      <details class="run-details">
        <summary>Show all images</summary>
        <div class="run-all-images"></div>
      </details>
    `;

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

    root.appendChild(card);
  });
}

async function refresh() {
  RUNS = await fetchJson("/api/results-runview");
  renderGroupFilter(RUNS);
  renderRunCards();
}

async function init() {
  byId("refreshResults").onclick = refresh;
  byId("runSearch").oninput = renderRunCards;
  byId("runGroupFilter").onchange = renderRunCards;
  await refresh();
  setInterval(refresh, 6000);
}

init().catch((err) => {
  byId("serverStatus").textContent = `Error: ${err.message}`;
});
