// nav-badge.js — persistent top-nav live job/queue counter
// Include in any HEATR page after the <nav> markup.
// Polls /api/jobs every 5 s and updates #navJobBadge.

(function () {
  "use strict";
  const POLL_MS = 5000;

  function updateBadge() {
    const badge = document.getElementById("navJobBadge");
    if (!badge) return;

    fetch("/api/jobs")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data) => {
        const jobs = Array.isArray(data) ? data : (data.jobs || []);
        const running = jobs.filter((j) => j.status === "running").length;
        const queued  = jobs.filter((j) => j.status === "queued").length;
        const total   = running + queued;

        if (total === 0) {
          badge.classList.add("hidden");
          return;
        }
        badge.classList.remove("hidden");

        const parts = [];
        if (running) parts.push(`${running} running`);
        if (queued)  parts.push(`${queued} queued`);
        badge.textContent = parts.join(" · ");

        // Colour: amber while running, grey while only queued
        badge.style.background = running ? "#7a4a00" : "#1e3050";
        badge.style.color       = running ? "#ffd060" : "#6090b0";
        badge.style.borderColor = running ? "#c07000" : "#2a5080";

        // Pulse animation while running
        badge.classList.toggle("badge-pulse", running > 0);
      })
      .catch(() => {
        const badge = document.getElementById("navJobBadge");
        if (badge) badge.classList.add("hidden");
      });
  }

  updateBadge();
  setInterval(updateBadge, POLL_MS);
})();
