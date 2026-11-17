"""
R6 -- the assignment result on every observed season, with a smart heuristic.

R3-R5 scored seven of the twenty-one seasons because the dynamic strategy
(S4) re-solves the MILP a dozen times per season.  The static comparison
that carries the paper's main claim -- optimised assignment against current
practice -- needs one solve per season and can afford all twenty-one.

Strategies scored here, on 5 farms x 11 scales x 21 held-out seasons:

  S1_agronomic   per-hybrid offsets, fixed field order (current practice)
  S1_greedy      per-hybrid agronomy + a sensible dispatch rule: fields sorted
                 hardest-cross-first, each given the best remaining
                 (day, offset) from the SAME ensemble yield table S3 uses.
                 This is the "could a script do it?" competitor.
  S3_static_robust  exact MILP on the ensemble yield table
  S5_wait_and_see   oracle bound

Also records CBC wall time and solver status for every S3 solve.

Writes results/R6_raw.csv, results/R6_summary.json
"""
import json, os, time
from multiprocessing import Pool
import numpy as np, pandas as pd, pulp

from run_real import build_real, load_seasons, pressure, RESULTS
from run_real_rep import ENS_CAP
from sim.scheduler import execute, Instance
from sim.strategies import (access_for, plan_agronomic, yield_table,
                            plan_wait_and_see)

SCALES = [10, 20, 30, 35, 37, 40, 43, 45, 48, 55, 70]
FARMS = [1, 2, 3, 4, 5]
N_PROC = 2


def solve_timed(inst, tbl, time_limit=60):
    """solve_milp with timing and status, no time-limit fallback."""
    from sim.scheduler import solve_milp
    t0 = time.time()
    prob_status = {}
    # replicate solve_milp but capture status
    fids = [f.fid for f in inst.fields]
    prob = pulp.LpProblem("a", pulp.LpMaximize)
    x = {}
    for (fid, d, o) in inst.feasible_choices():
        x[(fid, d, o)] = pulp.LpVariable(f"x_{fid}_{d}_{o}", cat="Binary")
    bonus = 10.0 * max(tbl.values())
    prob += pulp.lpSum((tbl[k] + bonus) * v for k, v in x.items())
    for fid in fids:
        prob += pulp.lpSum(v for k, v in x.items() if k[0] == fid) <= 1
    for day in inst.days:
        terms = [v for k, v in x.items() if day in inst.visits(k[1], k[2])]
        if terms:
            prob += pulp.lpSum(terms) <= inst.capacity
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    status = pulp.LpStatus[prob.status]
    plan = {k[0]: (k[1], k[2]) for k, v in x.items()
            if v.value() is not None and v.value() > 0.5}
    return plan, time.time() - t0, status, len(x)


def plan_greedy(inst, tbl, rec_offsets):
    """
    Hardest-cross-first greedy dispatch on the ensemble yield table.
    Fields are ordered by |recommended offset| descending (ties: larger area
    first).  Each field takes the feasible (day, offset) with the highest
    expected yield whose visit days still have crew capacity.
    """
    load = {d: 0 for d in inst.days}
    order = sorted(inst.fields,
                   key=lambda f: (-abs(rec_offsets[f.fid]), -f.area_ha))
    plan = {}
    for f in order:
        best, best_v = None, -1.0
        for (fid, d, o), v in tbl.items():
            if fid != f.fid or v <= best_v:
                continue
            vd = set(inst.visits(d, o))
            if all(load[x] < inst.capacity for x in vd):
                best, best_v = (d, o), v
        if best is None:
            continue
        plan[f.fid] = best
        for x in set(inst.visits(*best)):
            load[x] += 1
    return plan


def rec_offsets_from_plan(plan):
    return {fid: o for fid, (d, o) in plan.items()}


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
        tbl = yield_table(inst, ens, accesses=ens_acc)
        pg = plan_greedy(inst, tbl, rec_offsets_from_plan(p1))
        p3, secs, status, nvar = solve_timed(inst, tbl)
        p5, _ = plan_wait_and_see(inst, w, acc)

        for name, plan in [("S1_agronomic", p1), ("S1_greedy", pg),
                           ("S3_static_robust", p3), ("S5_wait_and_see", p5)]:
            r = execute(inst, plan, w, acc)
            tsis = [v[2] for v in r["detail"].values() if v[2] == v[2]]
            rows.append(dict(
                year=int(w.get("year")), strategy=name, n_fields=n_fields,
                capacity=2, pressure=pressure(inst), yield_t=r["total_yield"],
                n_sown=r["n_sown"],
                mean_abs_tsi=float(np.mean(np.abs(tsis))) if tsis else np.nan,
                farm_seed=farm_seed,
                solve_s=secs if name == "S3_static_robust" else np.nan,
                status=status if name == "S3_static_robust" else "",
                n_vars=nvar if name == "S3_static_robust" else np.nan))
    print(f"  N{n_fields} f{farm_seed}  {time.time()-t0:5.0f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    seasons, yrs = load_seasons()
    jobs = [(n, fs, seasons) for n in SCALES for fs in FARMS]
    print(f"{len(jobs)} cells x {len(seasons)} seasons, {N_PROC} procs", flush=True)
    parts = []
    with Pool(N_PROC) as pool:
        for df in pool.imap_unordered(_job, jobs):
            parts.append(df)
            pd.concat(parts, ignore_index=True).to_csv(
                f"{RESULTS}/R6_raw.csv", index=False)
    raw = pd.concat(parts, ignore_index=True)
    raw.to_csv(f"{RESULTS}/R6_raw.csv", index=False)

    piv = raw.pivot_table(index=["n_fields", "farm_seed", "year"],
                          columns="strategy", values="yield_t")
    print("\nmean yield by scale")
    print(piv.groupby(level="n_fields").mean().round(1).to_string())
    for b in ["S1_agronomic", "S1_greedy"]:
        tot = piv.groupby(level="n_fields").sum()
        print(f"\nS3 over {b}: pooled "
              f"{100*(piv['S3_static_robust']-piv[b]).sum()/piv[b].sum():+.1f}%  "
              "per-scale (ratio of totals): " +
              " ".join(f"{v:+.0f}" for v in
                       (100*(tot['S3_static_robust']-tot[b])/tot[b]).values))
    s3 = raw[raw.strategy == "S3_static_robust"]
    print("\nsolver: ", s3.groupby("n_fields").solve_s.agg(["mean", "max"]).round(1).to_string())
    print(s3.status.value_counts().to_string())
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
