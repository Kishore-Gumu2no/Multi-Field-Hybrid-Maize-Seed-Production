"""
Precompute a DSSAT CERES-Maize anthesis lookup table.

The scheduling optimiser evaluates on the order of 10^4-10^5
(field, sowing day, offset, scenario) combinations per season, so DSSAT
cannot be called inside the optimisation loop. It does not need to be:
anthesis date depends only on (cultivar, sowing day, weather year) and never
on the schedule, so the whole decision grid can be enumerated once up front
and the optimiser reduced to table lookups.

This replaces the thermal-time approximation used in the provisional study
with the real crop model, at no cost to the optimiser's speed.

Output: results/phenology_table.npz with
    adat[c, y, d]  days from sowing to anthesis, NaN where not reached
    cultivars, years, sow_days  the index labels
"""

import argparse
import itertools
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from sim.dssat import run_dssat, read_cultivars

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _job(args):
    cult, year, doy, wsta = args
    try:
        return run_dssat(cult, year, doy, wsta)
    except Exception:
        return None


def build(cultivars, years, sow_days, wsta, n_proc=2):
    jobs = [(c, y, d, wsta)
            for c in cultivars for y in years for d in sow_days]
    print(f"running {len(jobs)} DSSAT simulations ...", flush=True)
    t0 = time.time()
    with Pool(n_proc) as pool:
        vals = pool.map(_job, jobs, chunksize=32)
    el = time.time() - t0

    adat = np.full((len(cultivars), len(years), len(sow_days)), np.nan)
    ci = {c: i for i, c in enumerate(cultivars)}
    yi = {y: i for i, y in enumerate(years)}
    di = {d: i for i, d in enumerate(sow_days)}
    for (c, y, d, _), v in zip(jobs, vals):
        if v is not None:
            adat[ci[c], yi[y], di[d]] = v

    ok = np.isfinite(adat).mean()
    print(f"done in {el:.0f}s ({el / len(jobs) * 1000:.0f} ms/run), "
          f"{100 * ok:.1f}% of cells reached anthesis")
    return adat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsta", default="KARN", help="4-char weather station code")
    ap.add_argument("--years", default="", help="comma-separated, default all .WTH present")
    ap.add_argument("--sow-start", type=int, default=140)
    ap.add_argument("--sow-end", type=int, default=200)
    ap.add_argument("--cultivars", default="",
                    help="comma-separated; default a spread of real inbreds")
    ap.add_argument("--procs", type=int, default=2)
    args = ap.parse_args()

    if args.cultivars:
        cults = args.cultivars.split(",")
    else:
        # A spread of real, published cultivars whose coefficients give
        # distinct flowering times -- the parent-line pool.
        c = read_cultivars()
        c = c[~c["name"].str.contains("MINIMA|MAXIMA", case=False)]
        c = c[(c.P1 > 150) & (c.P1 < 300)]
        cults = c["var"].tolist()[:8]

    if args.years:
        years = [int(y) for y in args.years.split(",")]
    else:
        wdir = "/DSSAT48/Weather"
        years = sorted({2000 + int(f[4:6]) if int(f[4:6]) < 50
                        else 1900 + int(f[4:6])
                        for f in os.listdir(wdir)
                        if f.startswith(args.wsta) and f.endswith(".WTH")})
    if not years:
        sys.exit(f"no .WTH files for station {args.wsta} in /DSSAT48/Weather")

    sow_days = list(range(args.sow_start, args.sow_end + 1))
    print(f"cultivars {cults}")
    print(f"years {years[0]}-{years[-1]} ({len(years)})")
    print(f"sowing days {sow_days[0]}-{sow_days[-1]} ({len(sow_days)})")

    adat = build(cults, years, sow_days, args.wsta, n_proc=args.procs)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "phenology_table.npz")
    np.savez_compressed(out, adat=adat, cultivars=np.array(cults),
                        years=np.array(years), sow_days=np.array(sow_days),
                        wsta=args.wsta)
    print(f"-> {out}")

    # quick look at the spread that drives the scheduling problem
    print("\nmean days to anthesis by cultivar (averaged over years and days)")
    for i, c in enumerate(cults):
        m = np.nanmean(adat[i])
        print(f"  {c}  {m:5.1f}")
    print("\nparent-gap variability: sd across sowing days of "
          "(cultivar j - cultivar i), averaged over years")
    best = None
    for i, j in itertools.combinations(range(len(cults)), 2):
        gap = adat[j] - adat[i]                     # years x sow_days
        sd = float(np.nanmean(np.nanstd(gap, axis=1)))
        mean = float(np.nanmean(gap))
        if best is None or sd > best[0]:
            best = (sd, cults[i], cults[j], mean)
        print(f"  {cults[i]}/{cults[j]}  mean {mean:+5.1f} d, "
              f"sd across sowing dates {sd:4.2f} d")
    if best:
        print(f"\nmost schedule-sensitive pair: {best[1]}/{best[2]} "
              f"(mean {best[3]:+.1f} d, sd {best[0]:.2f} d)")


if __name__ == "__main__":
    main()
