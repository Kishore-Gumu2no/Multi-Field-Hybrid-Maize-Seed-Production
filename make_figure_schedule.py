"""
fig11 -- what the schedule actually does.

One farm, one real season. Panel (a) shows the robust static plan as executed;
panel (b) shows the dynamic plan on the same season. Each field is a row;
marks are the female sowing, the first male sowing and the split male sowing.
Days on which a field is untrafficable are shaded. The right-hand strip gives
the realised parent asynchrony per field, which is what the yield response is
applied to.

This is the figure that makes the mechanism concrete: the dynamic plan is not
"better scheduling" in the abstract, it is a visible reallocation of contested
days away from fields whose weather turned out well and towards fields whose
timing became urgent.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_real import build_real, load_seasons, make_analogue_replanner
from sim.scheduler import execute
from sim.strategies import access_for, plan_static_robust

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "figs")

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
})
COL, ACC, GREY = "#1f4e79", "#b4451f", "#8a8a8a"

N_FIELDS, CAP, SEED = 20, 2, 1

# Which season to draw. Picking one by hand invites cherry-picking, so we
# choose the season whose dynamic-vs-static difference is closest to the
# median across the scored seasons of this configuration: the figure then
# shows a typical season rather than a flattering one.
EVAL_CANDIDATES = [0, 3, 6, 9, 12, 15, 18]


def panel(ax, inst, res, acc, title):
    lo = inst.w0
    fids = sorted(res["detail"])
    for row, fid in enumerate(fids):
        ok = acc[fid]
        closed = [d for d in inst.days if not ok[d]]
        ax.scatter([d - lo for d in closed], [row] * len(closed),
                   s=1.2, color="#dddddd", marker="s", zorder=1)
    for row, fid in enumerate(fids):
        d_f, d_m, tsi, y = res["detail"][fid]
        if d_f is None:
            ax.text(-1.5, row, "x", color=ACC, fontsize=6, va="center")
            continue
        ax.scatter([d_f - lo], [row], s=14, color=COL, marker="o", zorder=3)
        ax.scatter([d_m - lo], [row], s=14, color=ACC, marker="^", zorder=3)
        ax.plot([min(d_f, d_m) - lo, max(d_f, d_m) - lo], [row, row],
                color=GREY, lw=0.6, zorder=2)
    ax.set_ylim(-1, len(fids))
    ax.set_xlim(-3, inst.T)
    ax.set_yticks([])
    ax.set_xlabel("day of sowing window")
    ax.set_title(title, loc="left")


def run_season(fields, inst, seasons, k):
    w = seasons[k]
    others = [s for i, s in enumerate(seasons) if i != k]
    rs = np.random.default_rng(1000 + k)
    ens = [others[i] for i in sorted(rs.choice(len(others), 8, replace=False))]
    ens_acc = [access_for(fields, s) for s in ens]
    acc = access_for(fields, w)
    p3, _ = plan_static_robust(inst, ens, ens_acc)
    r3 = execute(inst, p3, w, acc)
    rp = make_analogue_replanner(inst, w, acc, ens, every=5)
    r4 = execute(inst, p3, w, acc, replan_fn=rp)
    return w, acc, r3, r4


def main():
    seasons, yrs = load_seasons()
    _, fields, inst = build_real(n_fields=N_FIELDS, capacity=CAP,
                                 farm_seed=SEED)

    runs = {}
    for k in EVAL_CANDIDATES:
        if k >= len(seasons):
            continue
        runs[k] = run_season(fields, inst, seasons, k)
    gains = {k: (v[3]["total_yield"] - v[2]["total_yield"]) / v[2]["total_yield"]
             for k, v in runs.items()}
    med = float(np.median(list(gains.values())))
    k = min(gains, key=lambda kk: abs(gains[kk] - med))
    print("season gains:", {int(runs[j][0]['year']): round(100 * g, 1)
                            for j, g in gains.items()})
    print(f"median {100*med:+.1f}% -> drawing {runs[k][0]['year']}")
    w, acc, r3, r4 = runs[k]

    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6),
                           gridspec_kw=dict(width_ratios=[1, 1, 0.5]))
    panel(ax[0], inst, r3, acc,
          f"(a) static robust -- {r3['total_yield']:.1f} t")
    panel(ax[1], inst, r4, acc,
          f"(b) dynamic -- {r4['total_yield']:.1f} t, "
          f"{r4['n_replans']} replans")

    fids = sorted(r3["detail"])
    t3 = [abs(r3["detail"][f][2]) if r3["detail"][f][0] else np.nan
          for f in fids]
    t4 = [abs(r4["detail"][f][2]) if r4["detail"][f][0] else np.nan
          for f in fids]
    y = np.arange(len(fids))
    ax[2].barh(y - 0.2, t3, height=0.38, color=GREY, label="static")
    ax[2].barh(y + 0.2, t4, height=0.38, color=COL, label="dynamic")
    ax[2].set_yticks([])
    ax[2].set_xlabel("|TSI| (d)")
    ax[2].set_title("(c) asynchrony", loc="left")
    ax[2].legend(frameon=False, loc="lower right")

    handles = [plt.Line2D([], [], marker="o", ls="", color=COL,
                          label="female sowing", ms=4),
               plt.Line2D([], [], marker="^", ls="", color=ACC,
                          label="male sowing", ms=4),
               plt.Line2D([], [], marker="s", ls="", color="#dddddd",
                          label="field closed by rain", ms=4)]
    ax[0].legend(handles=handles, frameon=False, loc="upper left",
                 bbox_to_anchor=(0, 1.02), ncol=1)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(F, f"fig11_schedule.{ext}"))
    print(f"season {w.get('year')}: static {r3['total_yield']:.1f} t, "
          f"dynamic {r4['total_yield']:.1f} t "
          f"({100*(r4['total_yield']-r3['total_yield'])/r3['total_yield']:+.1f}%)")
    print("wrote fig11_schedule")


if __name__ == "__main__":
    main()
