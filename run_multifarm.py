"""
E6 -- Multi-instance farm experiment.

The provisional study drew every result from a single farm (12 fields,
make_farm(seed=7)). That makes the headline numbers specific to one random
draw of field areas, soils, microclimates and parent genetics, which is the
most serious methodological weakness in the paper.

This driver repeats the main comparison and the sharpness sweep across many
independently generated farms, so the reported effects carry a between-farm
spread and we can state how many farms the conclusion holds for -- not just
its average size.

Writes results/E6_multifarm.csv, results/E6_sharpness.csv and
results/E6_summary.json
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from run_experiments import build, run_block, paired_test, RESULTS

# ---------------------------------------------------------------- configuration
FARM_SEEDS = list(range(1, 11))        # 10 independent farms
SEASONS_MAIN = 50                      # per farm, lambda = 1
SEASONS_SWEEP = 30                     # per farm per lambda
LAMBDAS = [0.25, 0.5, 1.0, 2.0, 3.0]
N_PROC = max(1, min(2, os.cpu_count() or 1))


def one_farm_main(farm_seed):
    """Main five-strategy comparison on a single farm instance."""
    gen, fields, inst, ens, ens_acc = build(farm_seed=farm_seed)
    df = run_block(gen, fields, inst, ens, ens_acc, sharpness=1.0,
                   n_seasons=SEASONS_MAIN, tag=f"E6_farm{farm_seed}")
    df["farm_seed"] = farm_seed
    df["total_area_ha"] = sum(f.area_ha for f in fields)
    return df


def one_farm_sweep(args):
    """Sharpness sweep on a single farm instance."""
    farm_seed, lam = args
    gen, fields, inst, ens, ens_acc = build(farm_seed=farm_seed)
    df = run_block(gen, fields, inst, ens, ens_acc, sharpness=lam,
                   n_seasons=SEASONS_SWEEP, tag=f"E6s_f{farm_seed}_L{lam}")
    df["farm_seed"] = farm_seed
    return df


def per_farm_gain(df, a="S3_static_robust", b="S4_dynamic"):
    """Paired adaptivity gain within one farm, with its own significance."""
    out = []
    for (fs, lam), sub in df.groupby(["farm_seed", "sharpness"]):
        piv = sub.pivot_table(index="season", columns="strategy", values="yield_t")
        if a not in piv or b not in piv:
            continue
        d = (piv[b] - piv[a]).values
        base = piv[a].mean()
        t, p = stats.ttest_rel(piv[b].values, piv[a].values)
        out.append(dict(
            farm_seed=int(fs), sharpness=float(lam),
            S3=float(base), S4=float(piv[b].mean()),
            S5=float(piv["S5_wait_and_see"].mean()) if "S5_wait_and_see" in piv else np.nan,
            gain_t=float(d.mean()),
            gain_pct=float(100 * d.mean() / base) if base else np.nan,
            p=float(p),
            n=int(len(d)),
        ))
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    print(f"E6 multi-farm: {len(FARM_SEEDS)} farms, {N_PROC} processes", flush=True)

    # ------------------------------------------------------------ main
    print("E6a main comparison across farms ...", flush=True)
    with Pool(N_PROC) as pool:
        parts = pool.map(one_farm_main, FARM_SEEDS)
    e6 = pd.concat(parts, ignore_index=True)
    e6.to_csv(f"{RESULTS}/E6_multifarm.csv", index=False)

    gains = per_farm_gain(e6)
    gains.to_csv(f"{RESULTS}/E6_gains.csv", index=False)

    n_pos = int((gains.gain_t > 0).sum())
    n_sig = int(((gains.gain_t > 0) & (gains.p < 0.05)).sum())
    print(f"\nper-farm adaptivity gain (lambda = 1), {len(gains)} farms")
    print(gains[["farm_seed", "S3", "S4", "gain_t", "gain_pct", "p"]]
          .round(3).to_string(index=False))
    print(f"\npositive on {n_pos}/{len(gains)} farms; "
          f"significant on {n_sig}/{len(gains)}")
    print(f"gain mean {gains.gain_pct.mean():.2f}%  "
          f"sd {gains.gain_pct.std():.2f}%  "
          f"range [{gains.gain_pct.min():.2f}, {gains.gain_pct.max():.2f}]")

    # farm-level paired test: does the farm-mean gain differ from zero?
    t_farm, p_farm = stats.ttest_1samp(gains.gain_pct.values, 0.0)
    try:
        _, wp_farm = stats.wilcoxon(gains.gain_pct.values)
    except ValueError:
        wp_farm = np.nan

    # ------------------------------------------------------------ sweep
    print("\nE6b sharpness sweep across farms ...", flush=True)
    jobs = [(fs, lam) for fs in FARM_SEEDS for lam in LAMBDAS]
    with Pool(N_PROC) as pool:
        parts = pool.map(one_farm_sweep, jobs)
    e6s = pd.concat(parts, ignore_index=True)
    e6s.to_csv(f"{RESULTS}/E6_sharpness.csv", index=False)

    sg = per_farm_gain(e6s)
    sg.to_csv(f"{RESULTS}/E6_sharpness_gains.csv", index=False)

    by_lam = sg.groupby("sharpness")["gain_pct"].agg(["mean", "std", "min", "max"])
    print("\nadaptivity gain (%) by sharpness, across farms")
    print(by_lam.round(3).to_string())

    # does the gain rise with sharpness WITHIN each farm?
    slopes = []
    for fs, sub in sg.groupby("farm_seed"):
        sub = sub.sort_values("sharpness")
        sl = np.polyfit(sub.sharpness.values, sub.gain_pct.values, 1)[0]
        r = np.corrcoef(sub.sharpness.values, sub.gain_pct.values)[0, 1]
        slopes.append(dict(farm_seed=int(fs), slope=float(sl), r=float(r)))
    sl_df = pd.DataFrame(slopes)
    n_rising = int((sl_df.slope > 0).sum())
    print(f"\ngain rises with sharpness on {n_rising}/{len(sl_df)} farms "
          f"(mean slope {sl_df.slope.mean():+.3f} %/unit, "
          f"mean r {sl_df.r.mean():+.2f})")

    # does it saturate below 5% on every farm?
    peak = sg.groupby("farm_seed")["gain_pct"].max()
    n_below5 = int((peak < 5.0).sum())
    print(f"peak gain stays below 5% on {n_below5}/{len(peak)} farms "
          f"(max observed {peak.max():.2f}%)")

    summary = dict(
        config=dict(farms=len(FARM_SEEDS), seasons_main=SEASONS_MAIN,
                    seasons_sweep=SEASONS_SWEEP, lambdas=LAMBDAS),
        main=dict(
            gain_pct_mean=float(gains.gain_pct.mean()),
            gain_pct_sd=float(gains.gain_pct.std()),
            gain_pct_min=float(gains.gain_pct.min()),
            gain_pct_max=float(gains.gain_pct.max()),
            gain_t_mean=float(gains.gain_t.mean()),
            n_farms=len(gains), n_positive=n_pos, n_significant=n_sig,
            farm_level_t_p=float(p_farm), farm_level_wilcoxon_p=float(wp_farm),
        ),
        sweep=dict(
            by_lambda={str(k): {kk: float(vv) for kk, vv in v.items()}
                       for k, v in by_lam.to_dict("index").items()},
            n_rising=n_rising, n_farms=len(sl_df),
            mean_slope=float(sl_df.slope.mean()),
            mean_r=float(sl_df.r.mean()),
            n_peak_below_5pct=n_below5,
            max_peak_pct=float(peak.max()),
        ),
    )
    with open(f"{RESULTS}/E6_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\ndone in {time.time() - t0:.0f}s -> {RESULTS}/E6_*")


if __name__ == "__main__":
    main()
