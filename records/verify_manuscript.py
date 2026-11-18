"""Recompute every scheduling headline in the manuscript from raw result rows."""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_fc_main")
HAND = ["B1_sequential","B2_earliest","B3_largest","B4_hardest","B1R_relaxed"]

d = pd.read_csv(f"{R}/FC_all_planners.csv")
p = d.pivot_table(index=["n_fields","farm_seed"], columns="planner", values="execution")
p["BEST"] = p[HAND].max(axis=1)
rng = np.random.default_rng(0)

def pooled(num, den):
    return 100 * (p[num] - p[den]).sum() / p[den].sum()

def boot(num, den, B=10000):
    a, b = p[num].values, p[den].values
    idx = np.arange(a.size); out = np.empty(B)
    for i in range(B):
        k = rng.choice(idx, idx.size, replace=True)
        out[i] = 100 * (a[k] - b[k]).sum() / b[k].sum()
    return np.percentile(out, [2.5, 97.5])

checks = []
def chk(label, got, claimed, tol=0.05):
    ok = abs(got - claimed) <= tol
    checks.append(ok)
    print(f"{'OK ' if ok else 'FAIL'} {label:48s} claimed {claimed:+8.3f}  recomputed {got:+8.3f}")

chk("abstract/concl: exact vs best hand, pooled", pooled("B6_joint","BEST"), 20.8)
lo, hi = boot("B6_joint","BEST")
chk("  CI low", lo, 18.5, 0.15); chk("  CI high", hi, 23.1, 0.15)
chk("abstract: exact vs practice, pooled", pooled("B6_joint","B1_sequential"), 42.5)
chk("abstract: greedy vs best hand, pooled", pooled("B5_greedy","BEST"), -9.6)
q = p.loc[70]
chk("abstract: exact vs best hand, N=70", 100*(q["B6_joint"]/q["BEST"]-1).mean(), 35.2)
chk("abstract: exact vs practice, N=70", 100*(q["B6_joint"]/q["B1_sequential"]-1).mean(), 73.3)
for n, c in ((10, 0.09), (20, 0.21)):
    q = p.loc[n]
    chk(f"abstract: slack regime N={n}", 100*(q["B6_joint"]/q["BEST"]-1).mean(), c, 0.02)

rb = pd.read_csv(f"{R}/FC_robust_raw.csv")
sc = rb[rb.tag == "scenario"]
chk("tab:curves: scenario-grid mean vs best-of-4", sc.gain_vs_best.mean(), 38.7)
chk("tab:curves: scenario-grid min", sc.gain_vs_best.min(), 28.6)
chk("tab:curves: scenario-grid max", sc.gain_vs_best.max(), 44.9)
print(f"{'OK ' if (sc.gain_vs_best>0).all() else 'FAIL'} "
      f"{'tab:curves: positive in all cells':48s} {len(sc)} of {len(sc)}")
checks.append(bool((sc.gain_vs_best > 0).all()))
mg = rb[rb.tag.isin(["merge_allowed","merge_disallowed"])].groupby(["n_fields","tag"]).mean(numeric_only=True)
for n in (40, 70):
    for t, col in (("merge_allowed","allowed"), ("merge_disallowed","forbidden")):
        chk(f"tab:nomerge N={n} {col} vs best-of-4", mg.loc[(n,t),"gain_vs_best"],
            {(40,"merge_allowed"):26.9,(40,"merge_disallowed"):20.3,
             (70,"merge_allowed"):37.4,(70,"merge_disallowed"):23.3}[(n,t)])
chk("tab:nomerge N=70 forbidden fields sown", mg.loc[(70,"merge_disallowed"),"n_sown"], 29.0, 0.5)

m = pd.read_csv(f"{R}/FC_main_raw.csv")
chk("protocol: max |objective - execution| (1e-14)",
    (m.objective - m.execution).abs().max()*1e14, 4.3, 0.5)
chk("protocol: max LP gap (x1000)", pd.to_numeric(m.gap, errors="coerce").max()*1000, 3.2, 0.1)
chk("protocol: out-of-domain field-years", float(m.n_out_of_domain.sum()), 0.0, 0)
chk("protocol: field-years evaluated", float(m.n_field_years.sum()), 89418.0, 0)
chk("tab:solver: slowest solve (s)", m[m.planner=="B6_joint"].solve_s.max(), 4.7, 0.1)

print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
