"""
Leave-one-out refit sensitivity check on the winter_y1 yield curve.

CONTEXT
-------
winter_y1's published coefficients (a=3.5918, b=-0.3379, c=0.012) are a least-
squares fit Marahatta et al. made on their own underlying plot-level data; we
only have the 4 published treatment means (MAR, in
tests/test_t03_t06_response.py), which that fit reproduces at RMSE 0.216,
R2 0.893. With only 4 aggregate points available to us and 3 free parameters,
a bootstrap over coefficients would be fake precision (1 degree of freedom).

What we CAN do: refit a quadratic through each 3-point subset of the 4 means
(drop one, exact solve through the rest) and re-run the scenario grid under
each resulting curve, substituted for winter_y1. This is a sensitivity check
on how much the optimiser's advantage shifts if any one of the four published
treatment means is excluded -- not a replication of Marahatta's own fitting
procedure, which used more than these 4 points.

SCOPE: winter_y1 cells only (3 plateaus x 3 kappa x 3 seeds per refit, 108
solves). The refit perturbs winter_y1 alone, so the spring_y2 cells of the
declared grid are unchanged by construction and are already reported in
FC_robust_raw.csv; re-solving them here would pad the table with rows that
cannot vary.

Produces results_fc_main/FC_loo_raw.csv: same schema as FC_robust_raw.csv's
scenario rows, plus loo_drop_tsi/loo_a/loo_b/loo_c identifying which mean was
dropped and the resulting coefficients.
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import build, capacity as cap, compare, response as R
from sim.phenology import DssatPhenology
from run_familyc_robust import cell, ORD, W

MAR = {1.00: 3.00, 5.00: 2.53, 6.25: 1.87, 13.75: 1.25}


def loo_fits():
    """4 exact-fit quadratics, each dropping one of the 4 published means."""
    pts = sorted(MAR.items())
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    out = []
    for i in range(4):
        mask = np.ones(4, dtype=bool)
        mask[i] = False
        c2, c1, c0 = np.polyfit(xs[mask], ys[mask], 2)
        out.append(dict(dropped_tsi=float(xs[i]), dropped_y=float(ys[i]),
                         a=float(c0), b=float(c1), c=float(c2)))
    return out


FIELDNAMES = ["n_fields", "farm_seed", "curve", "plateau", "kappa",
              "loo_drop_tsi", "loo_a", "loo_b", "loo_c", "tag", "optimal",
              "best_ordered", "b1", "best_name", "gain_vs_b1", "gain_vs_best",
              "status", "gap", "n_sown", "merged_fraction", "n_cands"]


def done_keys(out_path):
    """(loo_drop_tsi, farm_seed, curve, plateau, kappa) already on disk."""
    if not os.path.exists(out_path):
        return set()
    import pandas as pd
    d = pd.read_csv(out_path)
    return set(zip(d.loo_drop_tsi, d.farm_seed, d.curve, d.plateau, d.kappa))


def main():
    ap_seeds = 3
    ph = DssatPhenology("results/phenology_table.npz")
    m = cap.make("visits")
    m.capacity = 2.0
    ref = ph.sow_days[len(ph.sow_days) // 2]

    original = dict(R.CURVES["winter_y1"])  # a, b, c, r2
    fits = loo_fits()

    os.makedirs("results_fc_main", exist_ok=True)
    out_path = "results_fc_main/FC_loo_raw.csv"
    skip = done_keys(out_path)
    if skip:
        print(f"resuming: {len(skip)} rows already on disk, skipping those",
              flush=True)
    new_file = not os.path.exists(out_path)

    with open(out_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if new_file:
            w.writeheader()
            fh.flush()
        try:
            for fit in fits:
                R.CURVES["winter_y1"] = dict(a=fit["a"], b=fit["b"],
                                              c=fit["c"], r2=None)
                grid = R.scenario_grid(curves=("winter_y1",))
                print(f"--- LOO drop TSI={fit['dropped_tsi']:.2f} "
                      f"(y={fit['dropped_y']:.2f}): "
                      f"a={fit['a']:.4f} b={fit['b']:.4f} c={fit['c']:.4f} "
                      "---", flush=True)
                for s in range(1, ap_seeds + 1):
                    fields = None
                    for sc in grid:
                        key = (fit["dropped_tsi"], s, sc.curve, sc.plateau,
                               sc.kappa)
                        if key in skip:
                            continue
                        if fields is None:
                            fields = build.build_farm(ph, 70, s, ref)
                        t0 = time.time()
                        cands, _ = build.build_candidates(ph, fields, W, m,
                                                            scenario=sc)
                        r = cell(ph, fields, cands, m, "loo_scenario", sc,
                                 dict(n_fields=70, farm_seed=s,
                                      curve=sc.curve, plateau=sc.plateau,
                                      kappa=sc.kappa,
                                      loo_drop_tsi=fit["dropped_tsi"],
                                      loo_a=fit["a"], loo_b=fit["b"],
                                      loo_c=fit["c"]))
                        w.writerow(r)
                        fh.flush()
                        print(f"  s={s} {sc.curve} {sc.plateau} k={sc.kappa} "
                              f"({time.time()-t0:5.1f}s) "
                              f"vsB1={r['gain_vs_b1']:+6.2f} "
                              f"vsBest={r['gain_vs_best']:+6.2f}", flush=True)
        finally:
            R.CURVES["winter_y1"] = original

    import pandas as pd
    n = len(pd.read_csv(out_path))
    print(f"\n{out_path} has {n} rows (target 216)")


if __name__ == "__main__":
    main()
