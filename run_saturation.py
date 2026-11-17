"""
R4 -- resolving the saturation trough.

R3 found the value of adaptivity to be non-monotone in resource pressure,
with a single negative configuration at pi = 0.98. One negative point is not
a finding; a reviewer is entitled to call it noise. This driver fills in the
region around saturation so that the trough can be shown to have width, a
location and a shape, or shown not to exist.

Capacity is held at K = 2 and the window at T = 61 days throughout, so
pressure is varied only through the number of fields. That also makes the
experiment the direct test of the paper's operational claim: if the gain
recovers a little below pi = 1, then a producer carrying slightly fewer
fields than their crew can exactly serve is in a regime where replanning
pays, which is the "deliberate slack" recommendation.

Existing R3 points at N = 30, 40, 55 are reused rather than recomputed.

Writes results/R4_*.csv and results/R4_summary.json
"""

import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from run_real import build_real, load_seasons, pressure, RESULTS
from run_real_rep import evaluate, per_cell, N_EVAL_YEARS

# Three farms per new point rather than five: the band test below pools
# farm-seasons across cells, so resolution in pi buys more than replicates
# within a cell, and the existing R3 cells already carry five farms each.
FARM_SEEDS = [1, 2, 3]

# pi = 3N / (61 * 2) = N / 40.667
NEW_FIELDS = [35, 37, 43, 45, 48]        # pi = 0.86, 0.91, 1.06, 1.11, 1.18
CAPACITY = 2
N_PROC = max(1, min(2, os.cpu_count() or 1))


def _job(args):
    n_fields, farm_seed, seasons, eval_years = args
    t0 = time.time()
    _, fields, inst = build_real(n_fields=n_fields, capacity=CAPACITY,
                                 farm_seed=farm_seed)
    tag = f"S_N{n_fields}_f{farm_seed}"
    df = evaluate(fields, inst, seasons, eval_years, sharpness=1.0, tag=tag)
    df["farm_seed"] = farm_seed
    df["kind"] = "p"
    print(f"  {tag:16s} {time.time()-t0:6.0f}s", flush=True)
    return df


def main():
    t0 = time.time()
    seasons, yrs = load_seasons()
    step = max(1, len(seasons) // N_EVAL_YEARS)
    eval_years = list(range(0, len(seasons), step))[:N_EVAL_YEARS]

    jobs = [(n, fs, seasons, eval_years)
            for n in NEW_FIELDS for fs in FARM_SEEDS]
    print(f"{len(jobs)} new cells around saturation, {N_PROC} processes",
          flush=True)

    parts = []
    with Pool(N_PROC) as pool:
        for i, df in enumerate(pool.imap_unordered(_job, jobs), 1):
            parts.append(df)
            if i % 5 == 0:
                print(f"[{i}/{len(jobs)}] {time.time()-t0:.0f}s", flush=True)
                pd.concat(parts, ignore_index=True).to_csv(
                    f"{RESULTS}/R4_raw.csv", index=False)
    new = pd.concat(parts, ignore_index=True)
    new.to_csv(f"{RESULTS}/R4_raw.csv", index=False)

    # merge with the R3 pressure cells so the curve is continuous
    old = pd.read_csv(f"{RESULTS}/R3_raw.csv")
    old = old[old.kind == "p"]
    allp = pd.concat([old, new], ignore_index=True)
    allp.to_csv(f"{RESULTS}/R4_pressure_raw.csv", index=False)

    tab = per_cell(allp, ["n_fields", "capacity", "pressure"])
    tab = tab.sort_values("pressure")
    tab.to_csv(f"{RESULTS}/R4_pressure.csv", index=False)
    print("\nresource pressure, R3 + R4 combined")
    print(tab[["n_fields", "pressure", "S3", "S4", "gain_pct", "p", "n",
               "n_farms_pos", "n_farms"]].round(3).to_string(index=False))

    # Is the dip real? Compare the cells inside the saturation band against
    # every cell outside it, farm-season by farm-season.
    piv = allp.pivot_table(index=["n_fields", "farm_seed", "year"],
                           columns="strategy", values="yield_t")
    g = (100 * (piv["S4_dynamic"] - piv["S3_static_robust"])
         / piv["S3_static_robust"]).reset_index()
    g["pressure"] = g.n_fields.map(dict(zip(tab.n_fields, tab.pressure)))
    g = g.rename(columns={0: "gain"})
    g.columns = [*g.columns[:-2], "gain", "pressure"]

    band = (g.pressure >= 0.90) & (g.pressure <= 1.10)
    inside, outside = g.loc[band, "gain"], g.loc[~band, "gain"]
    t, p = stats.ttest_ind(inside, outside, equal_var=False)
    print(f"\nsaturation band 0.90 <= pi <= 1.10 : mean {inside.mean():+.2f}% "
          f"(n = {len(inside)})")
    print(f"outside the band                   : mean {outside.mean():+.2f}% "
          f"(n = {len(outside)})")
    print(f"Welch t = {t:.2f}, p = {p:.4g}")

    # one-sample: is the band itself below zero?
    tb, pb = stats.ttest_1samp(inside, 0.0)
    print(f"band vs zero: t = {tb:.2f}, p = {pb:.4g}")

    with open(f"{RESULTS}/R4_summary.json", "w") as fh:
        json.dump(dict(
            new_fields=NEW_FIELDS, capacity=CAPACITY,
            farm_seeds=list(FARM_SEEDS), n_eval_years=len(eval_years),
            pressure=tab.to_dict("records"),
            band_lo=0.90, band_hi=1.10,
            band_mean=float(inside.mean()), band_n=int(len(inside)),
            outside_mean=float(outside.mean()), outside_n=int(len(outside)),
            band_vs_outside_t=float(t), band_vs_outside_p=float(p),
            band_vs_zero_t=float(tb), band_vs_zero_p=float(pb),
        ), fh, indent=2)
    print(f"\ndone in {time.time()-t0:.0f}s -> {RESULTS}/R4_*")


if __name__ == "__main__":
    main()
