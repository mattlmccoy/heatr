// HEATR animated logo mark — a faithful, lightweight distillation of the splash
// animation (heatr-splash.html): a small square part that heats corner-first
// through HEATR's thermal colormap, then cools, on a seamless loop.
//
// Self-contained: no CDN, no iframe, ~2 KB. Drop <script src="/static/heatr-logo.js">
// into a page whose topbar uses `.topbar-ids h1`; the mark replaces the static
// accent dot (`.topbar-ids h1::before`). Honors prefers-reduced-motion.
(function () {
  "use strict";

  // HEATR's signature thermal colormap (turbo-like): cold purple -> hot red.
  // Identical stops to heatColor() in heatr-splash.html.
  var STOPS = [
    [0.00, [18, 10, 34]], [0.14, [46, 28, 110]], [0.32, [43, 86, 196]],
    [0.48, [30, 172, 190]], [0.63, [118, 200, 96]], [0.77, [240, 214, 66]],
    [0.90, [241, 136, 44]], [1.00, [212, 52, 40]]
  ];
  function heatColor(v) {
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    for (var i = 0; i < STOPS.length - 1; i++) {
      var a = STOPS[i], b = STOPS[i + 1];
      if (v >= a[0] && v <= b[0]) {
        var t = (v - a[0]) / ((b[0] - a[0]) || 1);
        var c = a[1], d = b[1];
        return "rgb(" + Math.round(c[0] + (d[0] - c[0]) * t) + "," +
                        Math.round(c[1] + (d[1] - c[1]) * t) + "," +
                        Math.round(c[2] + (d[2] - c[2]) * t) + ")";
      }
    }
    return "rgb(212,52,40)";
  }

  // Field concentrates at edges/corners, so they heat first and hottest.
  // Same formula as the splash's per-cell concentration.
  function conc(cx, cy) {
    var edge = Math.max(Math.abs(cx), Math.abs(cy));
    var corner = Math.min(Math.abs(cx), Math.abs(cy));
    return 0.35 + 0.75 * Math.pow(edge, 1.6) + 0.25 * Math.pow(corner * edge, 1.2);
  }

  // Seamless heat envelope: cold -> hot (ignite) -> hold -> cool back to cold.
  function ramp(p) {
    if (p < 0.55) return p / 0.55;         // ignite
    if (p < 0.82) return 1;                // hold hot
    return 1 - (p - 0.82) / 0.18;          // cool for a clean loop seam
  }

  var N = 6;                 // 6x6 grid (variant A)
  var PERIOD = 3400;         // ms per loop, matches the splash cadence

  function drawGrid(ctx, S, p) {
    var r = ramp(p), cell = S / N, gap = Math.max(0.5, cell * 0.11);
    ctx.clearRect(0, 0, S, S);
    for (var i = 0; i < N; i++) {
      for (var j = 0; j < N; j++) {
        var cx = (i + 0.5) / N * 2 - 1;
        var cy = (j + 0.5) / N * 2 - 1;
        var h = r * conc(cx, cy) * 1.15;
        ctx.fillStyle = heatColor(h > 1 ? 1 : h);
        ctx.fillRect(i * cell, j * cell, cell - gap, cell - gap);
      }
    }
  }

  function mountInto(h1) {
    if (!h1 || h1.querySelector(".heatr-logo")) return;
    var S = 22; // topbar mark size in CSS px
    var cv = document.createElement("canvas");
    cv.className = "heatr-logo";
    cv.setAttribute("aria-hidden", "true");
    var dpr = window.devicePixelRatio || 1;
    cv.width = S * dpr;
    cv.height = S * dpr;
    cv.style.width = S + "px";
    cv.style.height = S + "px";
    cv.style.flexShrink = "0";
    cv.style.borderRadius = "3px";
    var ctx = cv.getContext("2d");
    ctx.scale(dpr, dpr);
    h1.insertBefore(cv, h1.firstChild);
    h1.classList.add("has-heatr-logo"); // hides the static ::before dot

    // Always paint a representative (mostly-heated) frame synchronously, so the
    // mark is never blank before the first animation frame, on a reduced-motion
    // system, or on a hidden/throttled surface where rAF is paused.
    drawGrid(ctx, S, 0.72);

    var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return; // keep the static frame, no motion

    var t0 = null;
    function frame(now) {
      if (t0 === null) t0 = now;
      drawGrid(ctx, S, ((now - t0) / PERIOD) % 1);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function init() {
    if (!document.getElementById("heatr-logo-style")) {
      var st = document.createElement("style");
      st.id = "heatr-logo-style";
      st.textContent = ".topbar-ids h1.has-heatr-logo::before{display:none!important;}" +
                       ".topbar-ids h1 .heatr-logo{display:block;}";
      document.head.appendChild(st);
    }
    var h1 = document.querySelector(".topbar-ids h1");
    if (h1) mountInto(h1);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
