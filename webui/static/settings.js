const HEATR_DEFAULTS = {
  mode: "dark",
  accent: "blue",
  bgstyle: "grid",
  layout: "layout-1",
};

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

function heatrApplyPrefs(prefs) {
  document.body.setAttribute("data-mode", prefs.mode);
  document.body.setAttribute("data-accent", prefs.accent);
  document.body.setAttribute("data-bgstyle", prefs.bgstyle);
  document.body.setAttribute("data-layout", prefs.layout);
}

function heatrInitSettings() {
  const prefs = heatrLoadPrefs();
  heatrApplyPrefs(prefs);

  const modeSel = document.getElementById("prefMode");
  const accentSel = document.getElementById("prefAccent");
  const bgSel = document.getElementById("prefBackground");
  const layoutSel = document.getElementById("prefLayout");

  if (modeSel) modeSel.value = prefs.mode;
  if (accentSel) accentSel.value = prefs.accent;
  if (bgSel) bgSel.value = prefs.bgstyle;
  if (layoutSel) layoutSel.value = prefs.layout;

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
