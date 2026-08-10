"""One-look diagnosis figure for the Tamper study:
 (a) the intrinsic radial Q_rf gradient (rim-hot / core-cold) that no scalar
     drive can fix; (b) the feasibility scissors: rim peak and below-floor
     fraction vs drive never overlap in the shippable box (dense AND <250)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SC = "/private/tmp/claude-501/-Users-mattmccoy-GaTech-Dropbox-Matthew-McCoy-mattmccoy-research-research-binderjet-code-geo-prewarp/632fc5f5-58d6-4ceb-a720-2930f7f099ff/scratchpad/"
q = np.load(SC + "tamper_qrf.npz")
TAMB, TPC_LO, FLOOR, PEAK1 = 50.0, 175.0, 0.85, 281.88

# panel a: radial Q profile
r, qu, vol = q["r"], q["q_uniform"], q["vol"]
edges = np.linspace(0, r.max(), 11)
gbar = np.average(qu, weights=vol)
rc, qq = [], []
for i in range(len(edges) - 1):
    m = (r >= edges[i]) & (r < edges[i + 1])
    if m.sum():
        rc.append(0.5 * (edges[i] + edges[i + 1]))
        qq.append(np.average(qu[m], weights=vol[m]) / gbar)

# panel b: feasibility scissors from end-state field
d = np.load("solve3d/phase_e/results/field_tamper_solve_filter_only.npz", allow_pickle=True)
T, chi, v2 = d["T_read"], d["chi"], d["vol"]
part = chi > 0.5
Tp, wp = T[part], (v2 * chi)[part]
A = np.linspace(0.4, 2.5, 80)
peak = TAMB + A * (PEAK1 - TAMB)
below = np.array([wp[np.clip((TAMB + a*(Tp-TAMB) - TPC_LO)/10, 0, 1) < FLOOR].sum() / wp.sum()
                  for a in A])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

ax1.axhline(1.0, color="gray", ls=":", lw=1)
ax1.plot(rc, qq, "o-", color="#c0392b", lw=2)
ax1.fill_between([0, 10], 0, 3, color="#3498db", alpha=0.12)
ax1.fill_between([12, r.max()], 0, 3, color="#e74c3c", alpha=0.12)
ax1.text(5, 2.5, "cold core\n(starves,\nunder-dense)", ha="center", color="#2471a3", fontsize=10)
ax1.text(17, 2.5, "hot rim\n(over-bakes,\n>ceiling)", ha="center", color="#a93226", fontsize=10)
ax1.set_xlabel("radius from axis (mm)"); ax1.set_ylabel("deposited power  Q / mean")
ax1.set_title("(a) Intrinsic radial power gradient (~3x)\nno scalar drive can flatten this")
ax1.set_ylim(0, 2.9); ax1.set_xlim(0, r.max())

ax2.axhspan(0, 250, color="#2ecc71", alpha=0.06)
ax2.plot(A, peak, color="#c0392b", lw=2.5, label="rim trajectory peak (est.)")
ax2.axhline(250, color="#c0392b", ls="--", lw=1.4)
ax2.text(0.42, 258, "250 C degradation ceiling", color="#c0392b", fontsize=9)
ax2b = ax2.twinx()
ax2b.plot(A, below * 100, color="#2471a3", lw=2.5, label="% part below density floor")
ax2b.axhline(15, color="#2471a3", ls=":", lw=1.2)
ax2b.text(2.0, 17, "15% floor target", color="#2471a3", fontsize=9)
# shade the two mutually exclusive windows
ax2.axvspan(0.4, 0.86, color="#2471a3", alpha=0.10)
ax2.axvspan(1.56, 2.5, color="#c0392b", alpha=0.10)
ax2.text(0.63, 470, "under\nceiling\nbut 100%\nunder-dense", ha="center", color="#2471a3", fontsize=8.5)
ax2.text(2.0, 470, "dense but\nrim 410-490 C", ha="center", color="#a93226", fontsize=8.5)
ax2.set_xlabel("drive multiplier  (x nominal power density)")
ax2.set_ylabel("rim peak temperature (C)", color="#c0392b")
ax2b.set_ylabel("% part below density floor", color="#2471a3")
ax2.set_title("(b) Feasibility scissors: no drive is both\ndense AND under-ceiling  (static uniform)")
ax2.set_ylim(100, 560); ax2b.set_ylim(0, 105); ax2.set_xlim(0.4, 2.5)

fig.suptitle("Tamper: why the dense-iff-in-bounds solve lands over-ceiling AND under-dense "
             "(first-order estimate from saved fields)", fontsize=11, y=1.02)
fig.tight_layout()
out = SC + "fig_tamper_study.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
