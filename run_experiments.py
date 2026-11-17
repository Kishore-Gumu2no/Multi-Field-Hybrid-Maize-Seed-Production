"""
Experiment driver.

E1  Main comparison of the five strategies at the published penalty sharpness.
E2  Sharpness sweep -- the central research question.
E3  Replanning frequency.
E4  Sensor ablation: field-level observation vs regional weather only.
E5  Farm size / crew capacity.

Results are written to results/*.csv and results/summary.json
"""

import json
import os
import time
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from sim.core import WeatherGenerator, make_farm, seed_yield
from sim.scheduler import Instance, execute
from sim.strategies import (access_for, plan_conventional, plan_static_ev,
                            plan_static_robust, plan_wait_and_see,
                            make_replanner)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------- configuration
N_FIELDS = 12
WINDOW = 26
CAPACITY = 2
ENS_SIZE = 12          # planning ensemble
N_SEASONS = 100        # evaluation seasons (Monte Carlo)
FARM_SEED = 7
ENS_SEED0 = 5000
EVAL_SEED0 = 20000


def build(n_fields=N_FIELDS, window=WINDOW, capacity=CAPACITY, farm_seed=FARM_SEED):
    gen = WeatherGenerator(seed=0)
    fields = make_farm(n_fields, seed=farm_seed)
    inst = Instance(fields, 0, window, capacity=capacity)
    ens = [WeatherGenerator(seed=ENS_SEED0 + s).sample() for s in range(ENS_SIZE)]
    ens_acc = [access_for(fields, w) for w in ens]
    return gen, fields, inst, ens, ens_acc


def run_block(gen, fields, inst, ens, ens_acc, sharpness=1.0, n_seasons=N_SEASONS,
              replan_every=3, forecast_days=0, sensor=True, tag=""):
    """
    Evaluate all five strategies over `n_seasons` independent weather draws.

    `sensor=False` models a farm with only a regional rain gauge: the planner
    assumes one common drainage rate for every field instead of resolving each
    field's own trafficability. Execution still happens against the true
    field-level conditions.
    """
    p1 = plan_conventional(inst)
    p2, _ = plan_static_ev(inst, gen, sharpness=sharpness)
    p3, _ = plan_static_robust(inst, ens, ens_acc, sharpness=sharpness)

    rows = []
    for k in range(n_seasons):
        w = WeatherGenerator(seed=EVAL_SEED0 + k).sample()
        acc_true = access_for(fields, w)

        if sensor:
            acc_plan = acc_true
            ens_acc_use = ens_acc
        else:
            # regional gauge only: one assumed drainage rate for all fields
            from sim.core import trafficability
            med = int(np.median([f.soil_drain_days for f in fields]))
            acc_plan = {f.fid: trafficability(w["rain"], med) for f in fields}
            ens_acc_use = [{f.fid: trafficability(e["rain"], med) for f in fields}
                           for e in ens]

        p5, _ = plan_wait_and_see(inst, w, acc_true, sharpness=sharpness)

        rp = make_replanner(inst, w, acc_plan, gen, ens, ens_acc_use,
                            every=replan_every, forecast_days=forecast_days,
                            sharpness=sharpness)

        for name, plan, fn in [("S1_conventional", p1, None),
                               ("S2_static_ev", p2, None),
                               ("S3_static_robust", p3, None),
                               ("S4_dynamic", p3, rp),
                               ("S5_wait_and_see", p5, None)]:
            r = execute(inst, plan, w, acc_true, replan_fn=fn, sharpness=sharpness)
            tsis = [v[2] for v in r["detail"].values() if v[2] == v[2]]
            rows.append(dict(
                tag=tag, season=k, strategy=name, sharpness=sharpness,
                replan_every=replan_every, forecast_days=forecast_days,
                sensor=sensor,
                yield_t=r["total_yield"], n_sown=r["n_sown"],
                n_replans=r["n_replans"], changes=r["schedule_changes"],
                mean_abs_tsi=float(np.mean(np.abs(tsis))) if tsis else np.nan,
                p_fail=float(np.mean([abs(t) > 4 for t in tsis])) if tsis else np.nan,
            ))
    return pd.DataFrame(rows)


def paired_test(df, a, b):
    """Paired comparison of two strategies across common seasons."""
    pa = df[df.strategy == a].set_index("season")["yield_t"]
    pb = df[df.strategy == b].set_index("season")["yield_t"]
    common = pa.index.intersection(pb.index)
    x, y = pb.loc[common].values, pa.loc[common].values
    d = x - y
    t, p = stats.ttest_rel(x, y)
    try:
        wst, wp = stats.wilcoxon(x, y)
    except ValueError:
        wp = np.nan
    sd = d.std(ddof=1)
    return dict(
        comparison=f"{b} - {a}",
        mean_diff=float(d.mean()),
        pct=float(100 * d.mean() / y.mean()) if y.mean() else np.nan,
        ci_lo=float(d.mean() - 1.96 * sd / np.sqrt(len(d))),
        ci_hi=float(d.mean() + 1.96 * sd / np.sqrt(len(d))),
        cohen_d=float(d.mean() / sd) if sd > 0 else np.nan,
        t_p=float(p), wilcoxon_p=float(wp), n=int(len(d)),
    )


def main():
    t_start = time.time()
    gen, fields, inst, ens, ens_acc = build()
    total_area = sum(f.area_ha for f in fields)
    print(f"farm: {len(fields)} fields, {total_area:.1f} ha, "
          f"window {WINDOW} d, capacity {CAPACITY} visits/d")

    summary = {"config": dict(n_fields=N_FIELDS, window=WINDOW,
                              capacity=CAPACITY, ens=ENS_SIZE,
                              seasons=N_SEASONS, total_area_ha=total_area)}

    # ---------------------------------------------------------------- E1
    print("\nE1 main comparison ...", flush=True)
    e1 = run_block(gen, fields, inst, ens, ens_acc, sharpness=1.0, tag="E1")
    e1.to_csv(f"{RESULTS}/E1_main.csv", index=False)
    agg = e1.groupby("strategy")["yield_t"].agg(["mean", "std", "min", "max"])
    print(agg.round(2))

    tests = [paired_test(e1, "S2_static_ev", "S3_static_robust"),      # VSS
             paired_test(e1, "S3_static_robust", "S4_dynamic"),        # adaptivity
             paired_test(e1, "S1_conventional", "S3_static_robust"),
             paired_test(e1, "S4_dynamic", "S5_wait_and_see")]         # EVPI
    summary["E1_tests"] = tests
    for t in tests:
        print(f"  {t['comparison']:<38} {t['mean_diff']:+7.2f} t "
              f"({t['pct']:+5.1f}%)  p={t['t_p']:.2g}  d={t['cohen_d']:.2f}")

    # ---------------------------------------------------------------- E2
    print("\nE2 sharpness sweep ...", flush=True)
    lambdas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    e2 = pd.concat([run_block(gen, fields, inst, ens, ens_acc, sharpness=L,
                              n_seasons=60, tag=f"E2_L{L}")
                    for L in lambdas], ignore_index=True)
    e2.to_csv(f"{RESULTS}/E2_sharpness.csv", index=False)

    sweep = []
    for L in lambdas:
        sub = e2[e2.sharpness == L]
        piv = sub.pivot_table(index="season", columns="strategy", values="yield_t")
        opt = piv["S5_wait_and_see"]
        row = dict(
            sharpness=L,
            S3=float(piv["S3_static_robust"].mean()),
            S4=float(piv["S4_dynamic"].mean()),
            S5=float(opt.mean()),
            adaptivity_gain_t=float((piv["S4_dynamic"] - piv["S3_static_robust"]).mean()),
            adaptivity_gain_pct=float(100 * (piv["S4_dynamic"] - piv["S3_static_robust"]).mean()
                                      / piv["S3_static_robust"].mean()),
            # share of the perfect-information gap that adaptivity closes
            closes_evpi_pct=float(
                100 * (piv["S4_dynamic"] - piv["S3_static_robust"]).mean()
                / max((opt - piv["S3_static_robust"]).mean(), 1e-9)),
        )
        row.update({f"p_{k}": v for k, v in
                    paired_test(sub, "S3_static_robust", "S4_dynamic").items()
                    if k in ("t_p", "cohen_d")})
        sweep.append(row)
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(f"{RESULTS}/E2_summary.csv", index=False)
    summary["E2_sweep"] = sweep_df.to_dict("records")
    print(sweep_df.round(3).to_string(index=False))

    # ---------------------------------------------------------------- E3
    print("\nE3 replanning frequency ...", flush=True)
    freqs = [1, 2, 3, 5, 7]
    e3 = pd.concat([run_block(gen, fields, inst, ens, ens_acc, sharpness=1.0,
                              n_seasons=50, replan_every=q, tag=f"E3_q{q}")
                    for q in freqs], ignore_index=True)
    e3.to_csv(f"{RESULTS}/E3_frequency.csv", index=False)
    f3 = (e3[e3.strategy == "S4_dynamic"]
          .groupby("replan_every")
          .agg(yield_t=("yield_t", "mean"), changes=("changes", "mean")))
    base3 = e3[e3.strategy == "S3_static_robust"].groupby("replan_every")["yield_t"].mean()
    f3["gain_vs_S3"] = f3["yield_t"] - base3
    print(f3.round(2))
    summary["E3"] = f3.reset_index().to_dict("records")

    # ---------------------------------------------------------------- E4
    print("\nE4 sensor ablation ...", flush=True)
    e4 = pd.concat([
        run_block(gen, fields, inst, ens, ens_acc, sharpness=1.0,
                  n_seasons=60, sensor=True, tag="E4_sensor"),
        run_block(gen, fields, inst, ens, ens_acc, sharpness=1.0,
                  n_seasons=60, sensor=False, tag="E4_nosensor"),
    ], ignore_index=True)
    e4.to_csv(f"{RESULTS}/E4_sensor.csv", index=False)
    f4 = e4[e4.strategy == "S4_dynamic"].groupby("sensor")["yield_t"].agg(["mean", "std"])
    print(f4.round(2))
    summary["E4"] = f4.reset_index().to_dict("records")

    # ---------------------------------------------------------------- E5
    print("\nE5 crew capacity ...", flush=True)
    rows5 = []
    for cap in [1, 2, 3, 4]:
        g2, f2, i2, en2, ea2 = build(capacity=cap)
        d = run_block(g2, f2, i2, en2, ea2, sharpness=1.0, n_seasons=40,
                      tag=f"E5_c{cap}")
        piv = d.pivot_table(index="season", columns="strategy", values="yield_t")
        rows5.append(dict(capacity=cap,
                          S3=float(piv["S3_static_robust"].mean()),
                          S4=float(piv["S4_dynamic"].mean()),
                          S5=float(piv["S5_wait_and_see"].mean()),
                          gain=float((piv["S4_dynamic"] - piv["S3_static_robust"]).mean())))
    e5 = pd.DataFrame(rows5)
    e5.to_csv(f"{RESULTS}/E5_capacity.csv", index=False)
    print(e5.round(2).to_string(index=False))
    summary["E5"] = e5.to_dict("records")

    with open(f"{RESULTS}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\ndone in {time.time() - t_start:.0f}s -> {RESULTS}/")


if __name__ == "__main__":
    main()
