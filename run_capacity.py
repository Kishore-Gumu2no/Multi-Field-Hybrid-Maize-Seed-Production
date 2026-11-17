"""
E1 + E2 -- does the headline gain survive the crew-capacity assumption,
and does it survive a competent hand rule?

Design.  Four capacity-accounting models (sim/capacity.py) x three levels of
resource pressure x five farms x twenty-one held-out seasons x six
strategies (sim/baselines.py).

Because the four models measure capacity in different units, farm size is
chosen per model so that RESOURCE PRESSURE is matched across models rather
than nominal capacity.  Pressure is computed from the conventional
three-separate-days cost of each field, so it does not depend on the schedule
the optimiser chooses.

Every strategy in a cell plans from the same ensemble expected-yield table,
built from eight seasons sampled from the twenty that exclude the scored one,
and is executed against the realised season under the same capacity model.

Writes results/E2_raw.csv
"""
import sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/proj")
from run_real import build_real, load_seasons, RESULTS
from run_real_rep import ENS_CAP
from sim import capacity as C
from sim import baselines as B
from sim.strategies import access_for, plan_agronomic, yield_table

PRESSURES = [1.5]
FARMS = [1, 2, 3, 4, 5]
T = 61
N_PROC = 2


def n_for(model_name, target_pi, probe_seed=1):
    """Farm size giving `target_pi` under this capacity model."""
    m = C.make(model_name)
    _, _, inst = build_real(n_fields=70, capacity=2, farm_seed=probe_seed)
    per_field = sum(m.field_demand(f) for f in inst.fields) / 70.0
    return max(5, int(round(target_pi * T * m.capacity / per_field)))


def _job(args):
    mname, target_pi, n_fields, farm_seed, seasons = args
    t0 = time.time()
    model = C.make(mname)
    _, fields, inst = build_real(n_fields=n_fields, capacity=2, farm_seed=farm_seed)
    pi = C.pressure(model, inst)
    rows = []

    for k in range(len(seasons)):
        w = seasons[k]
        others = [s for i, s in enumerate(seasons) if i != k]
        rs = np.random.default_rng(1000 + k)
        ens = [others[i] for i in sorted(rs.choice(len(others),
               size=min(ENS_CAP, len(others)), replace=False))]
        ens_acc = [access_for(fields, s) for s in ens]
        acc = access_for(fields, w)

        agro = plan_agronomic(inst, ens)
        rec_off = {fid: o for fid, (d, o) in agro.items()}
        for f in inst.fields:
            rec_off.setdefault(f.fid, 0)
        tbl = yield_table(inst, ens, accesses=ens_acc)

        plans = {which: B.plan_ordered(inst, model, rec_off, which)
                 for which in ("B1_sequential", "B2_earliest",
                               "B3_largest", "B4_hardest")}
        plans["B5_greedy"] = B.plan_greedy(inst, model, tbl, rec_off)
        p6, secs, status, nvar = B.plan_joint(inst, model, tbl)
        plans["B6_joint"] = p6

        for which, plan in plans.items():
            r = B.execute(inst, model, plan, w, acc)
            tsis = [v[2] for v in r["detail"].values() if v[2] == v[2]]
            merged = sum(1 for fid, (d, o) in plan.items()
                         if o == 0 or o == -inst.male_split_gap)
            rows.append(dict(
                year=int(w.get("year")), capacity_model=mname,
                target_pi=target_pi, pressure=pi, strategy=which,
                n_fields=n_fields, farm_seed=farm_seed,
                yield_t=r["total_yield"], n_sown=r["n_sown"],
                n_merged=merged,
                mean_abs_tsi=float(np.mean(np.abs(tsis))) if tsis else np.nan,
                solve_s=secs if which == "B6_joint" else np.nan,
                status=status if which == "B6_joint" else "",
                n_vars=nvar if which == "B6_joint" else np.nan,
                hit_limit=(secs >= 44.0) if which == "B6_joint" else False))
    print(f"  {mname:11s} pi={target_pi} N={n_fields} f{farm_seed}  "
          f"{time.time()-t0:5.0f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    seasons, _ = load_seasons()
    # Highest pressure first, and all four capacity models before moving to
    # the next pressure, so the decisive cross-model comparison is complete
    # earliest. Cells already finished in a previous run are skipped.
    import os
    done = set()
    prev = []
    for f in ():
        fp = f"{RESULTS}/{f}"
        if os.path.exists(fp):
            d = pd.read_csv(fp)
            full = d.groupby(["capacity_model", "target_pi", "farm_seed"]).year.nunique()
            for (m, pi, fs), nyr in full.items():
                if nyr >= len(seasons):
                    done.add((m, float(pi), int(fs)))
            prev.append(d)
    jobs = []
    for pi in sorted(PRESSURES, reverse=True):
        for mname in C.ALL:
            n = n_for(mname, pi)
            for fs in FARMS:
                if (mname, float(pi), int(fs)) in done:
                    continue
                jobs.append((mname, pi, n, fs, seasons))
    print(f"skipping {len(done)} cells already complete", flush=True)
    print(f"{len(jobs)} cells x {len(seasons)} seasons x 6 strategies",
          flush=True)
    for mname in C.ALL:
        print(f"  {mname:11s} N = "
              + ", ".join(f"{p}:{n_for(mname, p)}" for p in PRESSURES), flush=True)
    parts = list(prev)
    with Pool(N_PROC) as pool:
        for df in pool.imap_unordered(_job, jobs):
            parts.append(df)
            pd.concat(parts, ignore_index=True).to_csv(
                f"{RESULTS}/E2_raw.csv", index=False)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
