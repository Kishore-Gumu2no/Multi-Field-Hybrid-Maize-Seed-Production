"""fig13 -- decision value of simulated soil-temperature sensing against field heterogeneity."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st = json.load(open("records/E7_replicate_stats.json"))
sig = np.array(sorted(float(k) for k in st))
g = lambda k, f: np.array([st[("%g" % s) if ("%g" % s) in st else str(s)][k][f] if isinstance(st[("%g" % s) if ("%g" % s) in st else str(s)][k], dict) else st[("%g" % s) if ("%g" % s) in st else str(s)][k] for s in sig])
key = lambda s: ("%g" % s) if ("%g" % s) in st else str(s)
sen = np.array([st[key(s)]["sensor_mean"] for s in sig])
sl = np.array([st[key(s)]["sensor_ci"][0] for s in sig])
sh = np.array([st[key(s)]["sensor_ci"][1] for s in sig])
ora = np.array([st[key(s)]["oracle_mean"] for s in sig])
ol = np.array([st[key(s)]["oracle_ci"][0] for s in sig])
oh = np.array([st[key(s)]["oracle_ci"][1] for s in sig])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (lo, hi, title) in zip(axes, [(0, 1.15, "a  plausible range"),
                                      (0, 4.3, "b  full sweep")]):
    m = sig <= hi + 1e-9
    ax.axhspan(-0.3, 0, color="0.93", zorder=0)
    ax.axhline(0, color="0.3", lw=0.9, zorder=1)
    ax.fill_between(sig[m], ol[m], oh[m], color="tab:orange", alpha=0.18, lw=0)
    ax.plot(sig[m], ora[m], "o-", color="tab:orange", lw=1.8, ms=5,
            label="oracle (perfect knowledge)")
    ax.fill_between(sig[m], sl[m], sh[m], color="tab:blue", alpha=0.20, lw=0)
    ax.plot(sig[m], sen[m], "s-", color="tab:blue", lw=1.8, ms=5,
            label="simulated sensor")
    ax.axvline(0.5, color="tab:red", ls=":", lw=1.2)
    ax.set_xlabel("between-field soil thermal heterogeneity  $\\sigma$  (°C)")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlim(-0.05, hi)
    ax.grid(alpha=0.25, lw=0.5)

axes[0].set_ylabel("change in objective vs conventional CERES (%)")
axes[0].set_ylim(-0.3, 0.55)
axes[0].annotate("probe accuracy 0.5 °C", xy=(0.52, 0.50), fontsize=7.5,
                 color="tab:red", ha="left", rotation=90, va="top")
axes[0].annotate("sensor does harm when\nthere is nothing to measure",
                 xy=(0.03, -0.162), xytext=(0.30, -0.25), fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="0.4"))
axes[0].axvspan(0.11, 0.59, color="tab:green", alpha=0.10, lw=0)
axes[0].annotate("published spread between\nsimilarly managed fields",
                 xy=(0.35, 0.33), fontsize=7.5, ha="center", color="darkgreen")
axes[1].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Simulated soil-temperature sensing pays only above about 1 °C of between-field heterogeneity",
             fontsize=11, y=1.00)
fig.tight_layout()
os.makedirs("figs", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figs/fig13_iot_decision_value.{ext}", dpi=200, bbox_inches="tight")
print("wrote figs/fig13_iot_decision_value.png / .pdf")
