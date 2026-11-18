"""
fig13 -- decision value of simulated soil-temperature sensing.

(a) heterogeneity sweep, capacity-constrained against unconstrained
(b) sensor noise
(c) missing observations

Sources: records/E7_ci_from_rows.json (unconstrained, N=12),
         records/E8_sigma_stats.json, records/E8_ablation_stats.json,
         records/E8_visits_stats.json  (capacity-constrained, N=60).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

e7 = json.load(open("records/E7_ci_from_rows.json"))
e8 = json.load(open("records/E8_sigma_stats.json"))
abl = json.load(open("records/E8_ablation_stats.json"))

sig = [0.0, 0.5, 0.7, 1.0, 2.0]
k = lambda s: f"{s:g}"
g = lambda d, s, f: d[k(s)][f]

fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0),
                       gridspec_kw=dict(width_ratios=[1.55, 1, 1]))

# ---- (a) heterogeneity, constrained vs unconstrained -----------------------
a0 = ax[0]
a0.axhspan(-1.8, 0, color="0.93", zorder=0)
a0.axhline(0, color="0.3", lw=0.9, zorder=1)
for d, lab, c, ls, mk in ((e8, "capacity-constrained, N=60", "tab:blue", "-", "s"),
                          (e7, "unconstrained, N=12", "tab:cyan", "--", "o")):
    m = np.array([g(d, s, "sensor_mean") for s in sig])
    lo = np.array([g(d, s, "sensor_ci")[0] for s in sig])
    hi = np.array([g(d, s, "sensor_ci")[1] for s in sig])
    a0.fill_between(sig, lo, hi, color=c, alpha=0.18, lw=0)
    a0.plot(sig, m, mk + ls, color=c, lw=1.8, ms=5, label="sensor, " + lab)
for d, lab, c, ls, mk in ((e8, "capacity-constrained, N=60", "tab:orange", "-", "s"),
                          (e7, "unconstrained, N=12", "navajowhite", "--", "o")):
    m = np.array([g(d, s, "oracle_mean") for s in sig])
    a0.plot(sig, m, mk + ls, color=c, lw=1.6, ms=4.5, alpha=0.95,
            label="oracle, " + lab)
a0.errorbar([0.7], [0.191], yerr=[[0.199], [0.198]], fmt="D", color="tab:red",
            ms=5, capsize=3, lw=1.2, zorder=5,
            label="sensor, unconstrained at N=60")
a0.axvline(0.5, color="0.45", ls=":", lw=1.1)
a0.annotate("probe noise 0.5 °C", xy=(0.47, -1.55), fontsize=7.5,
            rotation=90, ha="right", va="bottom", color="0.35")
a0.axvspan(0.11, 0.59, color="tab:green", alpha=0.09, lw=0)
a0.set_xlabel("between-field soil thermal heterogeneity  $\\sigma$  (°C)")
a0.set_ylabel("change in objective vs conventional CERES (%)")
a0.set_title("a   scarcity amplifies both the gain and the harm",
             loc="left", fontsize=9.5)
a0.set_xlim(-0.05, 2.05)
a0.set_ylim(-1.8, 4.4)
a0.legend(frameon=False, fontsize=7.2, loc="upper left", ncol=1)
a0.grid(alpha=0.25, lw=0.5)

# ---- (b) noise, (c) missingness -------------------------------------------
for panel, key, xs, xlabel, title, vline in (
        (ax[1], "noise", [0.0, 0.25, 0.5, 1.0, 2.0],
         "probe noise  (°C)", "b   noise costs magnitude, not sign", 0.5),
        (ax[2], "missingness", [0.0, 0.25, 0.5, 0.75, 1.0],
         "fraction of observations missing",
         "c   estimation degrades gracefully", None)):
    d = abl[key]
    share = [d[k(x)]["captured"] for x in xs]
    m = np.array([d[k(x)]["sensor_mean"] for x in xs])
    lo = np.array([d[k(x)]["sensor_ci"][0] for x in xs])
    hi = np.array([d[k(x)]["sensor_ci"][1] for x in xs])
    panel.axhline(0, color="0.3", lw=0.9)
    panel.fill_between(xs, lo, hi, color="tab:blue", alpha=0.18, lw=0)
    panel.plot(xs, m, "s-", color="tab:blue", lw=1.8, ms=5, label="sensor gain")
    panel.axhline(2.279, color="tab:orange", lw=1.5, ls="-", label="oracle")
    for x, s_, y in zip(xs, share, m):
        panel.annotate(f"{s_:.0%}", xy=(x, y), xytext=(0, 7),
                       textcoords="offset points", fontsize=7.2,
                       ha="center", color="tab:blue")
    if vline is not None:
        panel.axvline(vline, color="0.45", ls=":", lw=1.1)
    panel.set_xlabel(xlabel)
    panel.set_title(title, loc="left", fontsize=9.5)
    panel.set_ylim(-0.25, 2.75)
    panel.grid(alpha=0.25, lw=0.5)
ax[1].set_ylabel("change in objective (%)")
ax[1].legend(frameon=False, fontsize=7.5, loc="center right")
ax[1].annotate("% = share of the oracle captured", xy=(0.04, 0.06),
               xycoords="axes fraction", fontsize=7, color="tab:blue")

fig.suptitle("Simulated soil-temperature sensing under a binding sowing-capacity constraint",
             fontsize=11, y=1.01)
fig.tight_layout()
os.makedirs("figs", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figs/fig13_iot_decision_value.{ext}", dpi=200, bbox_inches="tight")
print("wrote figs/fig13_iot_decision_value.png / .pdf")
