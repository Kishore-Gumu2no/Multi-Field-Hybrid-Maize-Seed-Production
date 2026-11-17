"""
R8 -- sensitivity: no visit merging.

In the main model a field needs three crew visits unless the male's first
or second sowing date coincides with the female's (o = 0 or o = -delta), in
which case the coincident operations are one visit.  Part of the
optimiser's throughput advantage comes from choosing such offsets.  Here
those two offsets are removed from the choice set for every strategy, so
every field needs exactly three distinct visit days and capacity is
effectively counted in sowing operations rather than field-visits.
S1 and S3 on all 21 seasons at three farm sizes.

Writes results/R8_raw.csv
"""
import time
from multiprocessing import Pool
import numpy as np, pandas as pd

from run_real import build_real, load_seasons, pressure, RESULTS
from run_real_rep import ENS_CAP
from sim.scheduler import execute, solve_milp
from sim.strategies import access_for, plan_agronomic, yield_table

SCALES = [70, 55]
FARMS = [1, 2, 3, 4, 5]


def _job(args):
    n_fields, farm_seed, seasons = args
    t0 = time.time()
    _, fields, inst = build_real(n_fields=n_fields, capacity=2, farm_seed=farm_seed)
    inst.offsets = [o for o in inst.offsets if o not in (0, -inst.male_split_gap)]
    rows = []
    for k in range(len(seasons)):
        w = seasons[k]
        others = [s for i, s in enumerate(seasons) if i != k]
        rs = np.random.default_rng(1000 + k)
        ens = [others[i] for i in sorted(rs.choice(len(others), size=min(ENS_CAP, len(others)), replace=False))]
        ens_acc = [access_for(fields, s) for s in ens]
        acc = access_for(fields, w)
        p1 = plan_agronomic(inst, ens)
        tbl = yield_table(inst, ens, accesses=ens_acc)
        p3 = solve_milp(inst, tbl, time_limit=60)
        for name, plan in [("S1_agronomic", p1), ("S3_static_robust", p3)]:
            r = execute(inst, plan, w, acc)
            rows.append(dict(year=int(w.get("year")), strategy=name, n_fields=n_fields,
                             pressure=pressure(inst), yield_t=r["total_yield"],
                             n_sown=r["n_sown"], farm_seed=farm_seed))
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
            pd.concat(parts + [pd.read_csv(f"{RESULTS}/R8_n40.csv")], ignore_index=True).to_csv(f"{RESULTS}/R8_raw.csv", index=False)
    raw = pd.concat(parts + [pd.read_csv(f"{RESULTS}/R8_n40.csv")], ignore_index=True)
    piv = raw.pivot_table(index=["n_fields", "farm_seed", "year"], columns="strategy", values="yield_t")
    tot = piv.groupby(level="n_fields").sum()
    print("no-merge gain S3 over S1 per scale:",
          " ".join(f"{v:+.0f}" for v in (100*(tot['S3_static_robust']-tot['S1_agronomic'])/tot['S1_agronomic']).values),
          f" pooled {100*(piv['S3_static_robust']-piv['S1_agronomic']).sum()/piv['S1_agronomic'].sum():+.1f}%")
    print(raw.pivot_table(index="n_fields", columns="strategy", values="n_sown").round(1))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
