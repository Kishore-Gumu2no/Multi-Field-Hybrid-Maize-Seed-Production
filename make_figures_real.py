"""
Figures for the real-data experiments.

fig8  value of adaptivity against resource pressure (R2)
fig9  value of adaptivity against penalty sharpness, real weather (R2)
fig10 the mechanism: parent flowering gap variability from CERES-Maize, and
      what a day of gap costs under the measured yield response

Written to figs/ as both pdf (for the manuscript) and png.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sim.core import seed_yield

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
F = os.environ.get("FIGS_DIR", os.path.join(HERE, "figs"))
os.makedirs(F, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

COL = "#1f4e79"
ACC = "#b4451f"
GREY = "#8a8a8a"
W1 = 3.4          # one IEEE column, inches


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(F, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


def fig8():
    """
    Value of adaptivity against resource pressure.

    Uses the combined R3 + R4 pressure cells when the saturation-refinement
    run is present, so the trough is drawn with the density needed to show
    it has width rather than being a single point.
    """
    import os as _os
    if _os.path.exists(f"{R}/R4_pressure.csv"):
        d = pd.read_csv(f"{R}/R4_pressure.csv").sort_values("pressure")
        raw = pd.read_csv(f"{R}/R4_pressure_raw.csv")
        S = json.load(open(f"{R}/R4_summary.json"))
    else:
        S = json.load(open(f"{R}/R3_summary.json"))
        d = pd.DataFrame(S["pressure"]).sort_values("pressure")
        raw = pd.read_csv(f"{R}/R3_raw.csv")

    fig, axs = plt.subplots(1, 1, figsize=(W1, 2.4))
    ax = [axs]

    # gain vs pressure, with per-farm points
    g = raw[raw.kind == "p"].pivot_table(
        index=["n_fields", "farm_seed", "year"], columns="strategy",
        values="yield_t").reset_index()
    g["gain"] = 100 * (g["S4_dynamic"] - g["S3_static_robust"]) \
        / g["S3_static_robust"]
    pmap = dict(zip(d.n_fields, d.pressure))
    g["pressure"] = g.n_fields.map(pmap)

    per_farm = g.groupby(["pressure", "farm_seed"]).gain.mean().reset_index()
    ax[0].scatter(per_farm.pressure, per_farm.gain, s=12, color=GREY,
                  alpha=0.8, zorder=2, label="farm means")
    ax[0].plot(d.pressure, d.gain_pct, "o-", color=COL, lw=1.4, ms=4,
               zorder=3, label="pooled")
    ax[0].axhline(0, color="k", lw=0.6)
    y0, y1 = ax[0].get_ylim()
    ax[0].axvspan(0.90, 1.10, color="#f6e6e0", zorder=0)   # the band tested
    ax[0].axvline(1.0, color=ACC, lw=0.8, ls="--", zorder=1)
    ax[0].text(1.0, y1, "saturation", fontsize=6, color=ACC,
               ha="center", va="bottom")
    ax[0].text(0.3, y0 + 0.04 * (y1 - y0), "slack:\nrevisions free",
               fontsize=6, color=GREY)
    ax[0].text(1.45, y0 + 0.04 * (y1 - y0), "over-subscribed:\ntriage",
               fontsize=6, color=GREY)
    ax[0].set_ylim(y0, y1)
    ax[0].set_xlabel(r"resource pressure $\pi = NV/TK$")
    ax[0].set_ylabel(r"value of adaptivity, $S_4-S_3$ (%)")
    ax[0].legend(frameon=False, loc="best")

    # (b) the full ladder in absolute yield
    fig.tight_layout()
    save(fig, "fig8_pressure")


def fig9():
    S = json.load(open(f"{R}/R3_summary.json"))
    d = pd.DataFrame(S["sharpness"]).sort_values("sharpness")
    raw = pd.read_csv(f"{R}/R3_raw.csv")

    fig, axs = plt.subplots(1, 1, figsize=(W1, 2.4))
    ax = [axs]

    g = raw[raw.kind == "l"].pivot_table(
        index=["sharpness", "farm_seed", "year"], columns="strategy",
        values="yield_t").reset_index()
    g["gain"] = 100 * (g["S4_dynamic"] - g["S3_static_robust"]) \
        / g["S3_static_robust"]
    pf = g.groupby(["sharpness", "farm_seed"]).gain.mean().reset_index()

    for fs, sub in pf.groupby("farm_seed"):
        ax[0].plot(sub.sharpness, sub.gain, "-", color=GREY, lw=0.8,
                   alpha=0.8)
    ax[0].plot(d.sharpness, d.gain_pct, "o-", color=COL, lw=1.5, ms=4,
               label="pooled")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].axhline(5.0, color=ACC, lw=0.8, ls="--")
    ax[0].text(d.sharpness.min(), 5.2, "5% threshold", fontsize=6,
               color=ACC)
    ax[0].axvline(1.0, color="k", lw=0.6, ls=":")
    ax[0].text(1.03, ax[0].get_ylim()[0] * 0.9 if ax[0].get_ylim()[0] < 0
               else ax[0].get_ylim()[0], "measured", fontsize=6, rotation=90)
    ax[0].set_xlabel(r"penalty sharpness $\lambda$")
    ax[0].set_ylabel(r"value of adaptivity (%)")
    ax[0].legend(frameon=False)

    fig.tight_layout()
    save(fig, "fig9_sharpness_real")


def fig10():
    ph = np.load(f"{R}/phenology_table.npz", allow_pickle=True)
    adat = ph["adat"]           # cultivar x year x sow_day
    cults = list(ph["cultivars"])
    sow = ph["sow_days"]

    fig, ax = plt.subplots(1, 2, figsize=(W1 * 2, 2.5))

    # (a) year-to-year spread in days to anthesis for each cultivar
    mean_by_year = np.nanmean(adat, axis=2)         # cultivar x year
    order = np.argsort(np.nanmean(mean_by_year, axis=1))
    bp = ax[0].boxplot([mean_by_year[i] for i in order],
                       widths=0.6, patch_artist=True,
                       medianprops=dict(color="k", lw=1.0),
                       flierprops=dict(ms=2, mfc=GREY, mec=GREY))
    for b in bp["boxes"]:
        b.set(facecolor="#dce6f1", edgecolor=COL, lw=0.8)
    ax[0].set_xticklabels([cults[i] for i in order], rotation=60,
                          ha="right", fontsize=5.5)
    ax[0].set_ylabel("days to anthesis")
    ax[0].set_title("(a) CERES-Maize anthesis, 21 observed seasons",
                    loc="left")

    # (b) cost of asynchrony under the measured response
    x = np.arange(0, 13)
    y = np.array([seed_yield(float(v)) for v in x])
    ax[1].plot(x, y, "o-", color=COL, lw=1.4, ms=3)
    ax[1].set_xlabel("parent asynchrony, TSI (d)")
    ax[1].set_ylabel("seed yield (t ha$^{-1}$)")
    per_day = 100 * (y[0] - y[1]) / y[0]
    ax[1].annotate(f"first day of error: $-${per_day:.1f}%",
                   xy=(1, y[1]), xytext=(3.2, y[1] + 0.25), fontsize=6,
                   arrowprops=dict(arrowstyle="->", lw=0.6, color=ACC),
                   color=ACC)
    ax[1].set_title("(b) measured cost of asynchrony", loc="left")

    fig.tight_layout()
    save(fig, "fig10_mechanism")


if __name__ == "__main__":
    import sys
    want = sys.argv[1:] or ["8", "9", "10"]
    for w in want:
        {"8": fig8, "9": fig9, "10": fig10}[w]()
