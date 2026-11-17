"""
R7 -- sensitivity of the assignment result to the yield-response curve.

The primary analysis scores synchrony with Marahatta et al.'s winter year-1
regression (R^2 = 0.58).  The same paper reports two further usable fits
(winter y2, R^2 = 0.29; spring y2, R^2 = 0.36).  Here the optimiser is
re-solved with each alternative curve as its objective and scored with the
same curve, against the same current-practice baseline, on all 21 seasons,
for three farm sizes spanning the pressure range.

Writes results/R7_raw.csv
"""
import time
from multiprocessing import Pool
import numpy as np, pandas as pd

from run_real import build_real, load_seasons, pressure, RESULTS
from run_real_rep import ENS_CAP
from sim.scheduler import execute, solve_milp
from sim.strategies import access_for, plan_agronomic, yield_table

SCALES = [10, 40, 70]
FARMS = [1, 2, 3, 4, 5]
CURVES = ["winter_y1", "winter_y2", "spring_y2"]


def _job(args):
    n_fields, farm_seed, seasons = args
    t0 = time.time()
    _, fields, inst = build_real(n_fields=n_fields, capacity=2,
                                 farm_seed=farm_seed)
    rows = []
    for k in range(len(seasons)):
        w = seasons[k]
        others = [s for i, s in enumerate(seasons) if i != k]
        rs = np.random.default_rng(1000 + k)
        ens = [others[i] for i in
               sorted(rs.choice(len(others), size=min(ENS_CAP, len(others)),
                                replace=False))]
        ens_acc = [access_for(fields, s) for s in ens]
        acc = access_for(fields, w)
        p1 = plan_agronomic(inst, ens)
        for curve in CURVES:
            tbl = yield_table(inst, ens, accesses=ens_acc, curve=curve)
            p3 = solve_milp(inst, tbl, time_limit=60)
            for name, plan in [("S1_agronomic", p1), ("S3_static_robust", p3)]:
                r = execute(inst, plan, w, acc, curve=curve)
                rows.append(dict(year=int(w.get("year")), strategy=name,
                                 curve=curve, n_fields=n_fields,
                                 pressure=pressure(inst),
                                 yield_t=r["total_yield"], n_sown=r["n_sown"],
                                 farm_seed=farm_seed))
    print(f"  N{n_fields} f{farm_seed}  {time.time()-t0:5.0f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    seasons, _ = load_seasons()
    jobs = [(n, fs, seasons) for n in SCALES for fs in FARMS]
    parts = []
    with Pool(2) as pool:
        for df in pool.imap_unordered(_job, jobs):
            parts.append(df)
            pd.concat(parts, ignore_index=True).to_csv(f"{RESULTS}/R7_raw.csv", index=False)
    raw = pd.concat(parts, ignore_index=True)
    piv = raw.pivot_table(index=["curve", "n_fields", "farm_seed", "year"],
                          columns="strategy", values="yield_t")
    for c in CURVES:
        p = piv.loc[c]
        tot = p.groupby(level="n_fields").sum()
        print(f"{c}: pooled {100*(p['S3_static_robust']-p['S1_agronomic']).sum()/p['S1_agronomic'].sum():+.1f}%  per-scale "
              + " ".join(f"{v:+.0f}" for v in (100*(tot['S3_static_robust']-tot['S1_agronomic'])/tot['S1_agronomic']).values))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
