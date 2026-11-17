"""
R3 -- replicated real-data experiment.

Supersedes R1 (one farm per configuration, six seasons, nothing significant)
and R2 (three "farms" that turned out to be the same farm with its fields
relabelled, because the parent pairs were selected deterministically).

Three things are fixed here:

  * Farms are genuinely independent. `pick_parent_pairs` now draws a random
    representative from each difficulty stratum, so two seeds give two
    different sets of parent crosses with the same difficulty profile.
  * The sharpness sweep is run at a configuration where the shared crew
    actually binds. Sweeping it where adaptivity has no room to act measures
    nothing.
  * More farms and more scored seasons, which the vectorised yield table
    makes affordable.

Evaluation is leave-one-season-out: each scored season is held out and the
planning ensemble is drawn from the remaining twenty, so no strategy is ever
planned on the season it is tested against.

Writes results/R3_*.csv and results/R3_summary.json
"""

import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from run_real import (build_real, load_seasons, pressure, gains,
                      make_analogue_replanner, RESULTS)
from sim.scheduler import execute, solve_milp
from sim.strategies import (access_for, plan_conventional,
                            plan_static_robust, plan_wait_and_see,
                            yield_table)

FARM_SEEDS = [1, 2, 3, 4, 5]
PRESSURE_CONFIGS = [(10, 2), (20, 2), (30, 2), (40, 2), (55, 2), (70, 2)]
SHARPNESS = [0.25, 0.5, 1.0, 2.0, 3.0]
SHARP_CONFIG = (55, 2)           # where the crew binds (pi = 1.35)
N_EVAL_YEARS = 7
ENS_CAP = 8
N_PROC = max(1, min(2, os.cpu_count() or 1))


def evaluate(fields, inst, seasons, eval_years, sharpness=1.0,
             replan_every=5, tag=""):
    rows = []
    for k in eval_years:
        w = seasons[k]
        others = [s for i, s in enumerate(seasons) if i != k]
        rs = np.random.default_rng(1000 + k)
        ens = [others[i] for i in
               sorted(rs.choice(len(others), size=min(ENS_CAP, len(others)),
                                replace=False))]
        ens_acc = [access_for(fields, s) for s in ens]
        acc = access_for(fields, w)

        p1 = plan_conventional(inst)
        tot = [float(np.nansum(s["gdd"][inst.w0:inst.w0 + inst.T])) for s in ens]
        typical = ens[int(np.argsort(tot)[len(tot) // 2])]
        tbl_ev = yield_table(inst, [typical], sharpness=sharpness)
        p2 = solve_milp(inst, tbl_ev) if tbl_ev else None
        p3, _ = plan_static_robust(inst, ens, ens_acc, sharpness=sharpness)
        p5, _ = plan_wait_and_see(inst, w, acc, sharpness=sharpness)
        rp = make_analogue_replanner(inst, w, acc, ens,
                                     every=replan_every, sharpness=sharpness)

        for name, plan, fn in [("S1_conventional", p1, None),
                               ("S2_static_ev", p2, None),
                               ("S3_static_robust", p3, None),
                               ("S4_dynamic", p3, rp),
                               ("S5_wait_and_see", p5, None)]:
            if plan is None:
                continue
            r = execute(inst, plan, w, acc, replan_fn=fn, sharpness=sharpness)
            tsis = [v[2] for v in r["detail"].values() if v[2] == v[2]]
            rows.append(dict(
                tag=tag, year=int(w.get("year")), strategy=name,
                sharpness=sharpness, n_fields=len(fields),
                capacity=inst.capacity, pressure=pressure(inst),
                yield_t=r["total_yield"], n_sown=r["n_sown"],
                n_replans=r["n_replans"], changes=r["schedule_changes"],
                mean_abs_tsi=float(np.mean(np.abs(tsis))) if tsis else np.nan,
            ))
    return pd.DataFrame(rows)


def _job(args):
    kind, n_fields, capacity, farm_seed, lam, seasons, eval_years = args
    t0 = time.time()
    _, fields, inst = build_real(n_fields=n_fields, capacity=capacity,
                                 farm_seed=farm_seed)
    tag = (f"P_N{n_fields}K{capacity}_f{farm_seed}" if kind == "p"
           else f"L{lam}_f{farm_seed}")
    df = evaluate(fields, inst, seasons, eval_years, sharpness=lam, tag=tag)
    df["farm_seed"] = farm_seed
    df["kind"] = kind
    print(f"  {tag:24s} {time.time()-t0:6.0f}s", flush=True)
    return df


def per_cell(df, group):
    """Paired S4 - S3 within each cell."""
    out = []
    for key, sub in df.groupby(group):
        piv = sub.pivot_table(index=["farm_seed", "year"],
                              columns="strategy", values="yield_t")
        if "S4_dynamic" not in piv or "S3_static_robust" not in piv:
            continue
        a, b = piv["S3_static_robust"], piv["S4_dynamic"]
        d = (b - a).values
        t, p = stats.ttest_rel(b.values, a.values) if len(d) > 2 else (np.nan, np.nan)
        # farm-level: is the gain positive on most farms, not just on average?
        byfarm = (piv["S4_dynamic"] - piv["S3_static_robust"]).groupby(
            level="farm_seed").mean()
        rec = dict(zip(group if isinstance(group, list) else [group],
                       key if isinstance(key, tuple) else [key]))
        rec.update(
            S1=float(piv["S1_conventional"].mean()) if "S1_conventional" in piv else np.nan,
            S2=float(piv["S2_static_ev"].mean()) if "S2_static_ev" in piv else np.nan,
            S3=float(a.mean()), S4=float(b.mean()),
            S5=float(piv["S5_wait_and_see"].mean()) if "S5_wait_and_see" in piv else np.nan,
            gain_t=float(d.mean()), gain_sd=float(d.std(ddof=1)),
            gain_pct=float(100 * d.mean() / a.mean()) if a.mean() else np.nan,
            p=float(p), n=int(len(d)),
            n_farms=int(len(byfarm)), n_farms_pos=int((byfarm > 0).sum()),
        )
        out.append(rec)
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    seasons, yrs = load_seasons()
    step = max(1, len(seasons) // N_EVAL_YEARS)
    eval_years = list(range(0, len(seasons), step))[:N_EVAL_YEARS]
    print(f"{len(seasons)} real seasons ({yrs[0]}-{yrs[-1]}); "
          f"scoring {len(eval_years)} leave-one-out; "
          f"{len(FARM_SEEDS)} independent farms per cell", flush=True)

    jobs = []
    for (n, k) in PRESSURE_CONFIGS:
        for fs in FARM_SEEDS:
            jobs.append(("p", n, k, fs, 1.0, seasons, eval_years))
    sn, sk = SHARP_CONFIG
    for lam in SHARPNESS:
        if lam == 1.0:
            continue          # already run as the (sn, sk) pressure cell
        for fs in FARM_SEEDS:
            jobs.append(("l", sn, sk, fs, lam, seasons, eval_years))
    print(f"{len(jobs)} cells, {N_PROC} processes", flush=True)

    parts = []
    with Pool(N_PROC) as pool:
        for i, df in enumerate(pool.imap_unordered(_job, jobs), 1):
            parts.append(df)
            if i % 5 == 0:
                print(f"[{i}/{len(jobs)}] {time.time()-t0:.0f}s elapsed",
                      flush=True)
                pd.concat([p for p in parts if len(p)],
                          ignore_index=True).to_csv(f"{RESULTS}/R3_raw.csv",
                                                    index=False)
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df.to_csv(f"{RESULTS}/R3_raw.csv", index=False)

    pres = per_cell(df[df.kind == "p"],
                    ["n_fields", "capacity", "pressure"]).sort_values("pressure")
    pres.to_csv(f"{RESULTS}/R3_pressure.csv", index=False)
    print("\nresource pressure (lambda = 1)")
    print(pres[["n_fields", "capacity", "pressure", "S1", "S3", "S4", "S5",
                "gain_pct", "p", "n", "n_farms_pos", "n_farms"]]
          .round(3).to_string(index=False))

    if len(pres) > 2:
        r, pr = stats.pearsonr(pres.pressure, pres.gain_pct)
        print(f"\ngain vs pressure: r = {r:.3f} (p = {pr:.4f})")
    else:
        r = pr = np.nan

    # the sharpness sweep: the lambda = 1 point comes from the pressure run
    sh_src = pd.concat([
        df[df.kind == "l"],
        df[(df.kind == "p") & (df.n_fields == SHARP_CONFIG[0])
           & (df.capacity == SHARP_CONFIG[1])],
    ], ignore_index=True)
    sh = per_cell(sh_src, ["sharpness"]).sort_values("sharpness")
    sh.to_csv(f"{RESULTS}/R3_sharpness.csv", index=False)
    print(f"\nsharpness at N={SHARP_CONFIG[0]}, K={SHARP_CONFIG[1]}")
    print(sh[["sharpness", "S3", "S4", "gain_pct", "p", "n",
              "n_farms_pos", "n_farms"]].round(3).to_string(index=False))

    if len(sh) > 2:
        rs, ps = stats.pearsonr(sh.sharpness, sh.gain_pct)
        print(f"\ngain vs sharpness: r = {rs:.3f} (p = {ps:.4f})")
    else:
        rs = ps = np.nan

    # pooled test: is adaptivity worth anything at all, over everything run?
    piv = df[df.kind == "p"].pivot_table(
        index=["n_fields", "farm_seed", "year"], columns="strategy",
        values="yield_t")
    dd = (piv["S4_dynamic"] - piv["S3_static_robust"]).values
    tt, tp = stats.ttest_1samp(dd, 0.0)
    pooled_pct = 100 * dd.mean() / piv["S3_static_robust"].mean()
    print(f"\npooled over all pressure cells: {pooled_pct:+.2f}% "
          f"(t = {tt:.2f}, p = {tp:.4g}, n = {len(dd)})")

    with open(f"{RESULTS}/R3_summary.json", "w") as fh:
        json.dump(dict(
            site="Ranebennur, Karnataka (14.60 N, 75.60 E, 597 m)",
            n_seasons=len(seasons), years=[int(y) for y in yrs],
            n_eval_years=len(eval_years), farm_seeds=FARM_SEEDS,
            sharp_config=list(SHARP_CONFIG),
            pressure=pres.to_dict("records"),
            sharpness=sh.to_dict("records"),
            gain_vs_pressure_r=None if np.isnan(r) else float(r),
            gain_vs_pressure_p=None if np.isnan(pr) else float(pr),
            gain_vs_sharpness_r=None if np.isnan(rs) else float(rs),
            gain_vs_sharpness_p=None if np.isnan(ps) else float(ps),
            pooled_gain_pct=float(pooled_pct), pooled_p=float(tp),
            pooled_n=int(len(dd)),
        ), fh, indent=2)
    print(f"\ndone in {time.time()-t0:.0f}s -> {RESULTS}/R3_*")


if __name__ == "__main__":
    main()
