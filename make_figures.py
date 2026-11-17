"""Figures for the manuscript. Run after run_experiments.py."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sim.core import seed_yield, YIELD_CURVES

R = os.path.join(os.path.dirname(__file__), "results")
F = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(F, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
COL = dict(S1="#8c8c8c", S2="#c4703d", S3="#2f6f9f", S4="#1a7f5a", S5="#4a4a4a")
LBL = {"S1_conventional": "S1 conventional", "S2_static_ev": "S2 static (EV)",
       "S3_static_robust": "S3 static (robust)", "S4_dynamic": "S4 dynamic",
       "S5_wait_and_see": "S5 wait-and-see"}
ORDER = list(LBL)


# ---------------------------------------------------------------- Fig 1
def fig_yield_curve():
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))

    x = np.linspace(-8, 16, 400)
    for key, sty in [("winter_y1", "-"), ("winter_y2", "--"), ("spring_y2", ":")]:
        ax[0].plot(x, seed_yield(x, curve=key), sty, lw=1.3,
                   label=f"{key.replace('_', ' ')}  $R^2$={YIELD_CURVES[key]['r2']:.2f}")
    ax[0].axvline(0, color="k", lw=0.5, ls=":")
    ax[0].set_xlabel("Tasseling–silking interval, TSI (days)")
    ax[0].set_ylabel("Seed yield (t ha$^{-1}$)")
    ax[0].set_title("(a) Published yield response")
    ax[0].legend(frameon=False, loc="upper right")

    for L, c in zip([0.0, 0.5, 1.0, 2.0, 3.0],
                    plt.cm.viridis(np.linspace(0.1, 0.85, 5))):
        ax[1].plot(x, seed_yield(x, sharpness=L), color=c, lw=1.3,
                   label=rf"$\lambda$={L:g}")
    ax[1].set_xlabel("Tasseling–silking interval, TSI (days)")
    ax[1].set_ylabel("Seed yield (t ha$^{-1}$)")
    ax[1].set_title(r"(b) Sharpness family $Y_\lambda$")
    ax[1].legend(frameon=False, ncol=2)

    fig.savefig(f"{F}/fig1_yield_curve.pdf")
    fig.savefig(f"{F}/fig1_yield_curve.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig_main():
    d = pd.read_csv(f"{R}/E1_main.csv")
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))

    data = [d[d.strategy == s]["yield_t"].values for s in ORDER]
    bp = ax[0].boxplot(data, patch_artist=True, widths=0.6,
                       medianprops=dict(color="k", lw=1.1),
                       flierprops=dict(marker=".", ms=2, alpha=0.4))
    for patch, s in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COL[s.split("_")[0]])
        patch.set_alpha(0.65)
        patch.set_linewidth(0.6)
    short = {"S1_conventional": "S1\nconv.", "S2_static_ev": "S2\nstatic\nEV",
             "S3_static_robust": "S3\nstatic\nrobust", "S4_dynamic": "S4\ndynamic",
             "S5_wait_and_see": "S5\noracle"}
    ax[0].set_xticklabels([short[s] for s in ORDER])
    ax[0].set_ylabel("Season seed yield (t)")
    ax[0].set_title("(a) Distribution over seasons")

    piv = d.pivot_table(index="season", columns="strategy", values="yield_t")
    gain = piv["S4_dynamic"] - piv["S3_static_robust"]
    order = np.argsort(piv["S3_static_robust"].values)
    ax[1].bar(range(len(gain)), gain.values[order], width=1.0,
              color=np.where(gain.values[order] >= 0, COL["S4"], COL["S2"]))
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("Season, ordered by static-robust yield (worst → best)")
    ax[1].set_ylabel("S4 − S3 (t)")
    ax[1].set_title("(b) Where adaptivity pays")

    fig.savefig(f"{F}/fig2_main.pdf")
    fig.savefig(f"{F}/fig2_main.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def fig_sharpness():
    s = pd.read_csv(f"{R}/E2_summary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))

    ax[0].plot(s.sharpness, s.adaptivity_gain_pct, "o-", color=COL["S4"], ms=4, lw=1.3)
    ax[0].axhline(5, color=COL["S2"], ls="--", lw=1.0)
    ax[0].text(0.06, 5 - 0.18, "Kelly et al. (2023) threshold: 5%",
               ha="left", va="top", fontsize=6.5, color=COL["S2"])
    ax[0].axvline(1.0, color="k", lw=0.5, ls=":")
    ax[0].text(1.06, 0.35, "published\nresponse", fontsize=6, va="bottom")
    ax[0].set_ylim(-0.3, 5.8)
    ax[0].set_xlabel(r"Penalty sharpness $\lambda$")
    ax[0].set_ylabel("Gain from adaptivity, S4 − S3 (%)")
    ax[0].set_title("(a) Value of adaptivity vs sharpness")

    # lambda = 0 has no penalty, so the share captured is undefined (0/0):
    # the policy correctly never fires. Exclude it from the trend.
    s2 = s[s.sharpness > 0]
    ax[1].plot(s2.sharpness, s2.closes_evpi_pct, "s-", color=COL["S3"], ms=4, lw=1.3)
    ax[1].axvline(1.0, color="k", lw=0.5, ls=":")
    ax[1].set_ylim(0, max(35, s2.closes_evpi_pct.max() * 1.15))
    ax[1].set_xlabel(r"Penalty sharpness $\lambda$")
    ax[1].set_ylabel("Share of EVPI captured (%)")
    ax[1].set_title("(b) Share of the achievable gap closed")

    fig.savefig(f"{F}/fig3_sharpness.pdf")
    fig.savefig(f"{F}/fig3_sharpness.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def fig_ablations():
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
    fig.subplots_adjust(wspace=0.42)

    f3 = pd.read_csv(f"{R}/E3_frequency.csv")
    g = (f3[f3.strategy == "S4_dynamic"].groupby("replan_every")["yield_t"].mean()
         - f3[f3.strategy == "S3_static_robust"].groupby("replan_every")["yield_t"].mean())
    ch = f3[f3.strategy == "S4_dynamic"].groupby("replan_every")["changes"].mean()
    a0 = ax[0]
    a0.plot(g.index, g.values, "o-", color=COL["S4"], ms=4.5, lw=1.3)
    for q in g.index:
        a0.annotate(f"{ch[q]:.0f}", (q, g[q]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6, color=COL["S1"])
    a0.set_xlabel("Replan interval (days)")
    a0.set_ylabel("Gain over S3 (t)")
    a0.set_title("(a) Replanning frequency")
    a0.margins(y=0.22)
    a0.text(0.97, 0.04, "labels: schedule changes", transform=a0.transAxes,
            ha="right", va="bottom", fontsize=6, color=COL["S1"])

    f4 = pd.read_csv(f"{R}/E4_sensor.csv")
    sub = f4[f4.strategy == "S4_dynamic"]
    m = sub.groupby("sensor")["yield_t"].mean()
    e = sub.groupby("sensor")["yield_t"].sem()
    ax[1].bar(["regional\ngauge only", "field-level\nsensing"],
              [m.get(False, np.nan), m.get(True, np.nan)],
              yerr=[e.get(False, 0), e.get(True, 0)],
              color=[COL["S2"], COL["S4"]], alpha=0.8, capsize=3, width=0.6)
    ax[1].set_ylabel("Season seed yield (t)")
    ax[1].set_title("(b) Observation quality")
    span = max(float(e.max()) * 4, 2.0)
    ax[1].set_ylim(m.min() - span, m.max() + span)
    ax[1].text(0.5, 0.06, "difference not significant", transform=ax[1].transAxes,
               ha="center", va="bottom", fontsize=6.5, style="italic")

    f5 = pd.read_csv(f"{R}/E5_capacity.csv")
    ax[2].plot(f5.capacity, f5.S3, "s-", color=COL["S3"], ms=4, lw=1.2, label="S3 static")
    ax[2].plot(f5.capacity, f5.S4, "o-", color=COL["S4"], ms=4, lw=1.2, label="S4 dynamic")
    ax[2].plot(f5.capacity, f5.S5, "^:", color=COL["S5"], ms=4, lw=1.0, label="S5 oracle")
    ax[2].set_xlabel("Crew capacity (visits day$^{-1}$)")
    ax[2].set_ylabel("Season seed yield (t)")
    ax[2].set_title("(c) Resource tightness")
    ax[2].set_xticks(f5.capacity)
    ax[2].legend(frameon=False, loc="lower right")

    fig.savefig(f"{F}/fig4_ablations.pdf")
    fig.savefig(f"{F}/fig4_ablations.png")
    plt.close(fig)


def main():
    fig_yield_curve()
    print("fig1 ok")
    for fn, name in [(fig_main, "fig2"), (fig_sharpness, "fig3"),
                     (fig_ablations, "fig4")]:
        try:
            fn()
            print(f"{name} ok")
        except Exception as exc:
            print(f"{name} skipped: {exc}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- Fig 5
def fig_multifarm():
    """Per-farm adaptivity gain: the single-instance result in context."""
    g = pd.read_csv(f"{R}/E6_gains.csv").sort_values("gain_pct")
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))

    colors = [COL["S4"] if v > 0 else COL["S2"] for v in g.gain_pct]
    edge = ["k" if fs == 7 else "none" for fs in g.farm_seed]
    ax[0].barh(range(len(g)), g.gain_pct, color=colors, alpha=0.85,
               edgecolor=edge, linewidth=1.1)
    ax[0].set_yticks(range(len(g)))
    ax[0].set_yticklabels([f"farm {int(f)}" for f in g.farm_seed])
    ax[0].axvline(0, color="k", lw=0.6)
    ax[0].axvline(g.gain_pct.mean(), color=COL["S3"], ls="--", lw=1.0)
    ax[0].text(g.gain_pct.mean(), -0.9, f" mean {g.gain_pct.mean():.1f}%",
               color=COL["S3"], fontsize=6.5, va="top")
    ax[0].set_xlabel("Gain from adaptivity, S4 − S3 (%)")
    ax[0].set_title("(a) Gain varies by farm instance")
    if 7 in set(g.farm_seed):
        row = g[g.farm_seed == 7].index[0]
        pos = list(g.farm_seed).index(7)
        ax[0].annotate("instance used in\nthe single-farm study",
                       xy=(g.gain_pct.iloc[pos], pos),
                       xytext=(0.45, 0.22), textcoords="axes fraction",
                       fontsize=6, ha="left",
                       arrowprops=dict(arrowstyle="->", lw=0.6))

    sig = g[g.p < 0.05]
    ns = g[g.p >= 0.05]
    ax[1].scatter(ns.S3, ns.gain_pct, s=26, facecolor="none",
                  edgecolor=COL["S1"], lw=1.0, label="not significant")
    ax[1].scatter(sig.S3, sig.gain_pct, s=26, color=COL["S4"],
                  label="$p<0.05$")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("Static-robust yield of that farm (t)")
    ax[1].set_ylabel("Gain from adaptivity (%)")
    ax[1].set_title("(b) Weaker farms gain more")
    ax[1].legend(frameon=False, loc="upper right")

    fig.savefig(f"{F}/fig5_multifarm.pdf")
    fig.savefig(f"{F}/fig5_multifarm.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6
def fig_sharpness_multifarm():
    """Sharpness sweep across farms -- replaces the single-farm version."""
    sg = pd.read_csv(f"{R}/E6_sharpness_gains.csv")
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))

    for fs, sub in sg.groupby("farm_seed"):
        sub = sub.sort_values("sharpness")
        ax[0].plot(sub.sharpness, sub.gain_pct, "-", color=COL["S1"],
                   lw=0.7, alpha=0.5)
    m = sg.groupby("sharpness")["gain_pct"].agg(["mean", "std"])
    ax[0].errorbar(m.index, m["mean"], yerr=m["std"], fmt="o-",
                   color=COL["S4"], ms=4.5, lw=1.6, capsize=3, zorder=5)
    ax[0].axhline(5, color=COL["S2"], ls="--", lw=1.0)
    ax[0].text(2.95, 4.75, "Kelly et al. (2023): 5%", ha="right",
               fontsize=6.5, color=COL["S2"], va="top")
    ax[0].axvline(1.0, color="k", lw=0.5, ls=":")
    ax[0].text(1.06, ax[0].get_ylim()[0], " published\n response",
               fontsize=6, va="bottom")
    ax[0].set_xlabel(r"Penalty sharpness $\lambda$")
    ax[0].set_ylabel("Gain from adaptivity, S4 − S3 (%)")
    ax[0].set_title("(a) Gain rises with sharpness on every farm")

    # where does each farm cross the 5% threshold?
    cross = []
    for fs, sub in sg.groupby("farm_seed"):
        sub = sub.sort_values("sharpness")
        x, y = sub.sharpness.values, sub.gain_pct.values
        above = np.where(y >= 5.0)[0]
        if len(above) and above[0] > 0:
            i = above[0]
            lam = x[i - 1] + (5.0 - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
            cross.append(lam)
    if cross:
        ax[1].hist(cross, bins=np.arange(0.5, 3.6, 0.35),
                   color=COL["S3"], alpha=0.85, edgecolor="white", lw=0.6)
        ax[1].axvline(np.median(cross), color=COL["S2"], ls="--", lw=1.1)
        ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.35)
        ax[1].text(np.median(cross), ax[1].get_ylim()[1] * 0.97,
                   f"median $\\lambda$={np.median(cross):.1f}", ha="center",
                   fontsize=6.5, color=COL["S2"], va="top")
        ax[1].axvline(1.0, color="k", lw=0.5, ls=":")
    ax[1].set_xlabel(r"Sharpness $\lambda$ at which the gain reaches 5%")
    ax[1].set_ylabel("Farms")
    ax[1].set_title("(b) Where adaptivity starts to pay")

    fig.savefig(f"{F}/fig6_sharpness_multifarm.pdf")
    fig.savefig(f"{F}/fig6_sharpness_multifarm.png")
    plt.close(fig)
    return cross


# ---------------------------------------------------------------- Fig 7
def fig_cimmyt():
    """External validation against CIMMYT's published recommendations."""
    import json
    d = pd.read_csv(f"{R}/V1_cimmyt_offsets.csv")
    s = json.load(open(f"{R}/V1_cimmyt_summary.json"))
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))

    sub = d.dropna(subset=["pred_days_from_spans"])
    lim = [-6, 12]
    ax[0].plot(lim, lim, "k-", lw=0.6, alpha=0.5)
    ax[0].scatter(d.rec_days, d.pred_days_from_gdd, s=30, color=COL["S3"],
                  label="from reported GDD offset")
    ax[0].scatter(sub.rec_days, sub.pred_days_from_spans, s=38, marker="s",
                  facecolor="none", edgecolor=COL["S4"], lw=1.2,
                  label="from reported parent GDD")
    for _, r in d.iterrows():
        ax[0].annotate(r.twh.replace("CLTH", ""),
                       (r.rec_days, r.pred_days_from_gdd),
                       textcoords="offset points", xytext=(4, -7), fontsize=5.5)
    ax[0].set_xlim(lim)
    ax[0].set_ylim(lim)
    ax[0].set_xlabel("CIMMYT recommended offset (days)")
    ax[0].set_ylabel("Model prediction (days)")
    ax[0].set_title("(a) Recommended sowing offsets")
    ax[0].legend(frameon=False, fontsize=6, loc="upper left")

    cim = s["cimmyt_order"]
    mod = s["model_order"]
    xs = [cim.index(t) for t in cim]
    ys = [mod.index(t) for t in cim]
    ax[1].scatter(xs, ys, s=34, color=COL["S4"])
    ax[1].plot([0, len(cim) - 1], [0, len(cim) - 1], "k-", lw=0.6, alpha=0.5)
    for t in cim:
        ax[1].annotate(t.replace("CLTH", ""),
                       (cim.index(t), mod.index(t)),
                       textcoords="offset points", xytext=(4, -3), fontsize=5.5)
    ax[1].set_xlabel("CIMMYT difficulty rank (easiest → hardest)")
    ax[1].set_ylabel("Model difficulty rank")
    ax[1].set_title(rf"(b) Difficulty ranking, $\rho$ = {s['rank_rho']:.2f}")
    ax[1].set_xticks(range(len(cim)))
    ax[1].set_yticks(range(len(cim)))

    fig.savefig(f"{F}/fig7_cimmyt.pdf")
    fig.savefig(f"{F}/fig7_cimmyt.png")
    plt.close(fig)
