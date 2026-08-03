#!/usr/bin/env python3
"""Merge the s4_rescore_coupled.py shards into one results file + one figure.

The coupled sweep runs for hours and was executed in shards (case A, case B, and
the two in-validity negative-branch points added afterwards), so this collects
`results_coupled*.json` / `fields_coupled*.npz` into `results_coupled.json` and
`fields_coupled.npz` and draws `figs/coupled_topology.png`.

Nothing here recomputes a score; it only concatenates and plots.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s4_flir_lib as L  # noqa: E402

SHARDS = ["results_coupled.json", "results_coupled_B.json",
          "results_coupled_neg.json", "results_coupled_neg4.json"]
ORDER = ["baseline_frozen", "a+0.0000", "a+0.0020", "a+0.0100",
         "a-0.0020", "a-0.0040", "a-0.0100"]


def merge() -> tuple[dict, dict]:
    out: dict | None = None
    fields: dict[str, np.ndarray] = {}
    for name in SHARDS:
        p = HERE / name
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if out is None:
            out = d
        else:
            for ck, ce in d["cases"].items():
                if ck not in out["cases"]:
                    out["cases"][ck] = ce
                else:
                    out["cases"][ck]["variants"].update(ce["variants"])
        f = HERE / name.replace("results", "fields").replace(".json", ".npz")
        if f.exists():
            with np.load(f) as z:
                fields.update({k: z[k] for k in z.files})
    assert out is not None, "no shards found"
    for ck in out["cases"]:
        v = out["cases"][ck]["variants"]
        out["cases"][ck]["variants"] = {k: v[k] for k in ORDER if k in v}
    out["meta"]["sigma_temp_coeffs"] = sorted(
        {float(k[1:]) for ce in out["cases"].values() for k in ce["variants"]
         if k.startswith("a")})
    return out, fields


def figure(out: dict, fields: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meas = np.load(HERE / "fields.npz")
    cases = [c for c in ("A", "B") if c in out["cases"]]
    variants = [k for k in ORDER if any(k in out["cases"][c]["variants"] for c in cases)]
    ncol = 1 + len(variants)
    fig, axes = plt.subplots(len(cases), ncol, figsize=(2.05 * ncol, 2.35 * len(cases)),
                             squeeze=False)
    for i, ck in enumerate(cases):
        amb = out["cases"][ck]["ambient_c"]
        m = L.normalize_rise(meas[f"meas_{ck}_0.95"], amb)
        ax = axes[i][0]
        ax.imshow(m.T, origin="lower", cmap="inferno")
        ax.set_title(f"case {ck} MEASURED\n95 % state", fontsize=7)
        ax.set_ylabel(f"case {ck}", fontsize=8)
        for j, vk in enumerate(variants):
            ax = axes[i][j + 1]
            key = f"{ck}_{vk}_0.95"
            if key not in fields:
                # the first shard was killed before its npz write, so a few
                # registered fields were not retained. The SCORES for these
                # variants are intact in results_coupled.json; only the picture
                # is missing. Say so rather than leaving a silent blank.
                ax.set_xticks([]); ax.set_yticks([])
                sc = out["cases"][ck]["variants"].get(vk)
                ax.text(0.5, 0.5, "field not\nretained\n(scores in\nresults JSON)",
                        ha="center", va="center", fontsize=6.5, color="gray",
                        transform=ax.transAxes)
                if sc is not None:
                    ax.set_title(f"{vk}\nr={sc['scores']['0.95']['r']:+.3f}  "
                                 f"cmr={sc['scores']['0.95']['topology_pred']['centre_minus_ring']:+.3f}",
                                 fontsize=7, color="gray")
                continue
            b = L.normalize_rise(fields[key], amb)
            ax.imshow(b.T, origin="lower", cmap="inferno")
            sc = out["cases"][ck]["variants"][vk]["scores"]["0.95"]
            g = out["cases"][ck]["variants"][vk]["gates"]
            bad = g["clamp_bound"] or not g["energy_gate_pass"]
            ax.set_title(f"{vk}{'  VOID' if bad else ''}\nr={sc['r']:+.3f}  "
                         f"cmr={sc['topology_pred']['centre_minus_ring']:+.3f}",
                         fontsize=7, color=("crimson" if bad else "black"))
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("S4 re-score: 95 %-state free-face pattern under sigma(T) coupling "
                 "(measured cmr: A +0.399, B +0.295)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    (HERE / "figs").mkdir(exist_ok=True)
    fig.savefig(HERE / "figs" / "coupled_topology.png", dpi=170)
    print("wrote", HERE / "figs" / "coupled_topology.png")


def main() -> None:
    out, fields = merge()
    (HERE / "results_coupled.json").write_text(json.dumps(out, indent=1, default=float))
    np.savez_compressed(HERE / "fields_coupled.npz", **fields)
    print("merged variants:",
          {c: list(out["cases"][c]["variants"]) for c in out["cases"]})
    figure(out, fields)


if __name__ == "__main__":
    main()
