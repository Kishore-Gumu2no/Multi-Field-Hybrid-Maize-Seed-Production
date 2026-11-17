"""
fig12 -- the hand-made calendar and the optimised one, side by side.

One farm (N = 55, K = 2, pi = 1.35), one real season chosen as the one whose
optimised-vs-practice gain is closest to the median over all 21 seasons, so
the figure shows a typical season rather than a flattering one.

(a) current practice: per-hybrid offsets, fields in order, first free day.
(b) optimised assignment (S3).  Fields the plan never manages to sow are
    marked with a cross at the left edge.
(c) realised |TSI| per field under each plan.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_real import build_real, load_seasons
from run_real_rep import ENS_CAP
from sim.scheduler import execute
from sim.strategies import access_for, plan_static_robust, plan_agronomic

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "figs")
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
COL, ACC, GREY = "#1f4e79", "#b4451f", "#8a8a8a"
N_FIELDS, CAP, SEED = 55, 2, 1


def panel(ax, inst, res, acc, title):
    lo = inst.w0
    fids = sorted(res["detail"])
    for row, fid in enumerate(fids):
        closed = [d for d in inst.days if not acc[fid][d]]
        ax.scatter([d - lo for d in closed], [row] * len(closed),
                   s=1.0, color="#dddddd", marker="s", zorder=1)
    unsown = 0
    for row, fid in enumerate(fids):
        d_f, d_m, tsi, y = res["detail"][fid]
        if d_f is None:
            ax.text(-2.2, row, "x", color=ACC, fontsize=6, va="center")
            unsown += 1
            continue
        ax.scatter([d_f - lo], [row], s=9, color=COL, marker="o", zorder=3)
        ax.scatter([d_m - lo], [row], s=9, color=ACC, marker="^", zorder=3)
        ax.plot([min(d_f, d_m) - lo, max(d_f, d_m) - lo], [row, row],
                color=GREY, lw=0.5, zorder=2)
    ax.set_ylim(-1, len(fids))
    ax.set_xlim(-4, inst.T)
    ax.set_yticks([])
    ax.set_xlabel("day of sowing window")
    ax.set_title(title, loc="left")
    return unsown


def run_season(fields, inst, seasons, k):
    w = seasons[k]
    others = [s for i, s in enumerate(seasons) if i != k]
    rs = np.random.default_rng(1000 + k)
    ens = [others[i] for i in sorted(rs.choice(len(others), ENS_CAP, replace=False))]
    ens_acc = [access_for(fields, s) for s in ens]
    acc = access_for(fields, w)
    r1 = execute(inst, plan_agronomic(inst, ens), w, acc)
    p3, _ = plan_static_robust(inst, ens, ens_acc)
    r3 = execute(inst, p3, w, acc)
    return w, acc, r1, r3


def main():
    seasons, yrs = load_seasons()
    _, fields, inst = build_real(n_fields=N_FIELDS, capacity=CAP, farm_seed=SEED)
    runs = {k: run_season(fields, inst, seasons, k) for k in range(len(seasons))}
    gains = {k: (v[3]["total_yield"] - v[2]["total_yield"]) / v[2]["total_yield"]
             for k, v in runs.items()}
    med = float(np.median(list(gains.values())))
    k = min(gains, key=lambda kk: abs(gains[kk] - med))
    print(f"median gain {100*med:+.1f}% -> drawing {runs[k][0]['year']}")
    w, acc, r1, r3 = runs[k]

    fig, ax = plt.subplots(1, 3, figsize=(7.0, 3.3),
                           gridspec_kw=dict(width_ratios=[1, 1, 0.5]))
    u1 = panel(ax[0], inst, r1, acc,
               f"(a) practice: {r1['total_yield']:.0f} t, {r1['n_sown']}/{N_FIELDS} sown")
    u3 = panel(ax[1], inst, r3, acc,
               f"(b) optimised: {r3['total_yield']:.0f} t, {r3['n_sown']}/{N_FIELDS} sown")
    fids = sorted(r1["detail"])
    t1 = [abs(r1["detail"][f][2]) if r1["detail"][f][0] else np.nan for f in fids]
    t3 = [abs(r3["detail"][f][2]) if r3["detail"][f][0] else np.nan for f in fids]
    y = np.arange(len(fids))
    ax[2].barh(y - 0.2, t1, height=0.38, color=GREY, label="practice")
    ax[2].barh(y + 0.2, t3, height=0.38, color=COL, label="optimised")
    ax[2].set_yticks([]); ax[2].set_xlabel("|TSI| (d)")
    ax[2].set_title("(c) asynchrony", loc="left")
    ax[2].legend(frameon=False, loc="lower right")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COL, label="female sowing", ms=4),
               plt.Line2D([], [], marker="^", ls="", color=ACC, label="first male sowing", ms=4),
               plt.Line2D([], [], marker="s", ls="", color="#dddddd", label="field closed by rain", ms=4),
               plt.Line2D([], [], marker="x", ls="", color=ACC, label="never sown", ms=4)]
    ax[0].legend(handles=handles, frameon=False, loc="lower right", ncol=1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(F, f"fig12_practice.{ext}"))
    print(f"season {w.get('year')}: practice {r1['total_yield']:.1f} t ({r1['n_sown']} sown), "
          f"optimised {r3['total_yield']:.1f} t ({r3['n_sown']} sown)")


if __name__ == "__main__":
    main()
