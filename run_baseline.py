"""
R5 -- a fair baseline.

The conventional strategy used so far applied one rule-of-thumb male offset
to every field. With parent pairs differing by 1.5 to 14 days in flowering,
that is a straw man: the agronomic literature exists precisely so producers
can look up a per-hybrid offset, and they do.

This driver recomputes the baseline as current best practice -- correct
per-field agronomy from a climatological GDD table, naive logistics -- and
scores it on the same farms, the same held-out seasons and the same ensembles
as every other strategy. Only the baseline is recomputed; S2 to S5 are read
from the existing runs, so the comparison stays paired.

Writes results/R5_baseline.csv and merges into results/R5_raw.csv
"""

import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from run_real import build_real, load_seasons, pressure, RESULTS
from run_real_rep import ENS_CAP, N_EVAL_YEARS
from sim.scheduler import execute
from sim.strategies import access_for, plan_agronomic

N_PROC = max(1, min(2, os.cpu_count() or 1))


def _job(args):
    n_fields, capacity, farm_seed, seasons, eval_years = args
    t0 = time.time()
    _, fields, inst = build_real(n_fields=n_fields, capacity=capacity,
                                 farm_seed=farm_seed)
    rows = []
    for k in eval_years:
        w = seasons[k]
        others = [s for i, s in enumerate(seasons) if i != k]
        rs = np.random.default_rng(1000 + k)
        ens = [others[i] for i in
               sorted(rs.choice(len(others), size=min(ENS_CAP, len(others)),
                                replace=False))]
        acc = access_for(fields, w)
        plan = plan_agronomic(inst, ens)
        r = execute(inst, plan, w, acc)
        tsis = [v[2] for v in r["detail"].values() if v[2] == v[2]]
        rows.append(dict(
            year=int(w.get("year")), strategy="S1_agronomic", sharpness=1.0,
            n_fields=n_fields, capacity=capacity, pressure=pressure(inst),
            yield_t=r["total_yield"], n_sown=r["n_sown"], n_replans=0,
            changes=0,
            mean_abs_tsi=float(np.mean(np.abs(tsis))) if tsis else np.nan,
            farm_seed=farm_seed, kind="p", tag=f"B_N{n_fields}_f{farm_seed}"))
    print(f"  N{n_fields} f{farm_seed}  {time.time()-t0:5.0f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    seasons, yrs = load_seasons()
    step = max(1, len(seasons) // N_EVAL_YEARS)
    eval_years = list(range(0, len(seasons), step))[:N_EVAL_YEARS]

    # exactly the (n_fields, capacity, farm_seed) cells already evaluated
    src = pd.read_csv(f"{RESULTS}/R4_pressure_raw.csv")
    cells = (src[src.kind == "p"][["n_fields", "capacity", "farm_seed"]]
             .drop_duplicates().values.tolist())
    jobs = [(int(n), int(k), int(fs), seasons, eval_years)
            for n, k, fs in cells]
    print(f"{len(jobs)} baseline cells, {N_PROC} processes", flush=True)

    parts = []
    with Pool(N_PROC) as pool:
        for df in pool.imap_unordered(_job, jobs):
            parts.append(df)
    base = pd.concat(parts, ignore_index=True)
    base.to_csv(f"{RESULTS}/R5_baseline.csv", index=False)

    merged = pd.concat([src, base], ignore_index=True)
    merged.to_csv(f"{RESULTS}/R5_raw.csv", index=False)

    piv = merged.pivot_table(index=["n_fields", "farm_seed", "year"],
                             columns="strategy", values="yield_t")
    print("\nmean yield by strategy and scale")
    g = piv.groupby(level="n_fields").mean()
    cols = [c for c in ["S1_conventional", "S1_agronomic", "S2_static_ev",
                        "S3_static_robust", "S4_dynamic", "S5_wait_and_see"]
            if c in g]
    print(g[cols].round(1).to_string())

    print("\noptimiser gain over each baseline (%)")
    for b in ["S1_conventional", "S1_agronomic"]:
        if b not in piv:
            continue
        gain = 100 * (piv["S3_static_robust"] - piv[b]) / piv[b]
        print(f"  vs {b:18s} pooled "
              f"{100*(piv['S3_static_robust']-piv[b]).sum()/piv[b].sum():+6.1f}%"
              f"   per-scale " +
              " ".join(f"{v:+.0f}" for v in
                       gain.groupby(level='n_fields').mean().values))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
