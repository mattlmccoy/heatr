const byId = (id) => document.getElementById(id);
const viewer = { items: [], index: 0 };

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

function renderMaster(url) {
  const root = byId("masterSweep");
  if (!url) {
    root.textContent = "No master sweep image found.";
    return;
  }
  const items = [{ url, title: "Master Sweep" }];
  root.innerHTML = `<button type=\"button\" class=\"thumb-btn\"><img src=\"${url}\" alt=\"master sweep\" class=\"master-image\" /></button>`;
  root.querySelector("button").onclick = () => openViewer(items, 0);
}

function pathUrl(prefix, name, img) {
  return `/files/outputs_eqs/${prefix}/${encodeURIComponent(name)}/${encodeURIComponent(img)}`;
}

function renderSection(rootId, prefix, rows) {
  const root = byId(rootId);
  root.innerHTML = "";
  if (!rows?.length) {
    root.textContent = "No examples found.";
    return;
  }

  rows.forEach((row) => {
    const images = row.images || [];
    const items = images.map((img) => ({ url: pathUrl(prefix, row.name, img), title: `${row.name}/${img}` }));
    const preview = row.preview ? pathUrl(prefix, row.name, row.preview) : (items[0]?.url || "");

    const el = document.createElement("article");
    el.className = "example-card";
    el.innerHTML = `
      <h3>${row.name}</h3>
      ${preview ? `<button type=\"button\" class=\"thumb-btn\"><img src=\"${preview}\" alt=\"${row.name}\" /></button>` : ""}
      <div class="example-links"></div>
    `;

    const imgBtn = el.querySelector("button");
    if (imgBtn) imgBtn.onclick = () => openViewer(items, 0);

    const links = el.querySelector(".example-links");
    images.slice(0, 6).forEach((img, idx) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.textContent = img;
      b.onclick = () => openViewer(items, idx);
      links.appendChild(b);
    });

    root.appendChild(el);
  });
}

async function init() {
  const data = await fetchJson("/api/examples");
  renderMaster(data.master_sweep);
  renderSection("sweepGrid", "sweeps", data.sweeps || []);
  renderSection("shapeGrid", "shapes", data.shapes || []);
}

init().catch((err) => {
  byId("serverStatus").textContent = `Error: ${err.message}`;
});
