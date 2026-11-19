"""Family C figures: the sowing calendar comparison and the gain-versus-N curve."""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import build, capacity as cap, compare, response as R
from sim.phenology import DssatPhenology

W = list(range(274, 335))
HAND = ["B1_sequential","B2_earliest","B3_largest","B4_hardest","B1R_relaxed"]

def relaxed(fields, cands, capacity):
    cbf = compare.by_field(cands); load, chosen = {}, []
    for f in sorted(fields, key=lambda f: f.fid):
        for c in sorted(cbf.get(f.fid, []), key=lambda c: (-c.area_r, c.label)):
            if compare._fits(load, c.cost, capacity):
                chosen.append(c); compare._apply(load, c.cost); break
    return compare._solution("B1R_relaxed", chosen, len(cands))

ph = DssatPhenology("results/phenology_table.npz")
m = cap.make("visits"); m.capacity = 2.0
N, SEED = 70, 1
fields = build.build_farm(ph, N, SEED, ph.sow_days[len(ph.sow_days)//2])
cands, _ = build.build_candidates(ph, fields, W, m)
b1 = compare.plan_ordered(fields, cands, m.capacity, "B1_sequential")
rx = relaxed(fields, cands, m.capacity)
op = compare.plan_joint(cands, m.capacity, W, name="B6_joint", time_limit=300)

def rows(sol):
    out = {}
    for c in sol.chosen:
        out[c.fid] = sorted(int(d) for d in c.cost)
    return out

fig = plt.figure(figsize=(7.0, 2.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.28)
for k, (sol, ttl) in enumerate([(b1, "(a) practice"), (op, "(b) optimised")]):
    ax = fig.add_subplot(gs[0, k]); rr = rows(sol)
    for fid in range(N):
        if fid in rr:
            ds = rr[fid]
            ax.plot(ds, [fid]*len(ds), ".", ms=3.0, color="#1f4e79")
            ax.plot([min(ds), max(ds)], [fid, fid], "-", lw=0.5, color="#1f4e79", alpha=0.45)
        else:
            ax.plot([W[0]-1.2], [fid], "x", ms=3.0, color="#c0392b")
    ax.set_xlim(W[0]-3, W[-1]+1); ax.set_ylim(-1, N)
    ax.set_xlabel("day of year"); ax.set_title(ttl, fontsize=8.5)
    ax.text(0.97, 0.05, f"{sol.n_sown}/{N} sown", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8)
    if k == 0:
        ax.set_ylabel("field")
        ax.annotate("never sown", xy=(W[0]-1.2, N*0.80), xytext=(W[0]+9, N*0.93),
                    fontsize=7.4, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.7))

for sp in ("top","right"):
    for a in fig.axes: a.spines[sp].set_visible(False)
fig.savefig("figs/fig12_practice.pdf", bbox_inches="tight")
fig.savefig("figs/fig12_practice.png", dpi=150, bbox_inches="tight")
print(f"wrote figs/fig12_practice.pdf  (B1 {b1.n_sown}, relaxed {rx.n_sown}, exact {op.n_sown})")

# --- fig1: the implemented response -------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
xs = np.linspace(R.TSI_LO, R.TSI_HI, 400)
ax = axes[0]
for name, lab, col in (("winter_y1", "winter, year 1", "#1f4e79"),
                       ("spring_y2", "spring, year 2", "#c0392b")):
    ax.plot(xs[xs >= 0], [R.curve_absolute(float(u), name) for u in xs[xs >= 0]],
            lw=1.4, color=col, label=lab)
ax.set_xlabel("TSI (days)"); ax.set_ylabel(r"seed yield (t ha$^{-1}$)")
ax.set_title("(a) published regressions", fontsize=9)
ax.legend(fontsize=7.2, frameon=False)
ax = axes[1]
for kap, col in ((1.5, "#1f4e79"), (2.0, "#2e86c1"), (2.5, "#7fb3d5")):
    sc = R.Scenario(curve="winter_y1", plateau="PL2", kappa=kap).validate()
    ax.plot(xs, [R.relative_yield(float(u), sc) for u in xs], lw=1.4,
            color=col, label=rf"$\kappa={kap}$")
ax.axvspan(0, 2, color="0.88", zorder=0)
ax.text(1.0, 0.28, "plateau", fontsize=6.8, ha="center", color="0.35")
ax.set_xlabel("TSI (days)"); ax.set_ylabel("relative yield $r$")
ax.set_title("(b) implemented response", fontsize=9)
ax.legend(fontsize=7.2, frameon=False, loc="lower center")
for a in axes:
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("figs/fig1_yield_curve.pdf", bbox_inches="tight")
fig.savefig("figs/fig1_yield_curve.png", dpi=150, bbox_inches="tight")
print("wrote figs/fig1_yield_curve.pdf")
