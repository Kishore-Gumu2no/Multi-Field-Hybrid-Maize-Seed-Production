"""
P1 robustness re-runs on the corrected Family C engine, replacing the legacy
R7 (yield-curve sensitivity) and R8 (merging disallowed) tables.

R7 replacement: does the optimiser's advantage survive a different published
yield response? Sweeps the declared scenario grid (2 curves x 3 plateaus x
3 kappa = 18 cells) at the pressure where capacity binds hardest.

R8 replacement: is the advantage just visit merging? Rebuilds with every
merged candidate removed from the set given to EVERY planner, so the
comparison stays like-for-like.
"""
import argparse, csv, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import build, capacity as cap, compare, response as R
from sim.phenology import DssatPhenology

W = list(range(274, 335))
ORD = ("B1_sequential", "B2_earliest", "B3_largest", "B4_hardest")


def cell(ph, fields, cands, m, tag, scen, extra):
    out = {}
    for name in ORD:
        sol = compare.plan_ordered(fields, cands, m.capacity, name)
        out[name] = build.execute(ph, fields, sol, scenario=scen)["total_area_r"]
    sol = compare.plan_joint(cands, m.capacity, W, name="B6_joint", time_limit=180)
    opt = build.execute(ph, fields, sol, scenario=scen)["total_area_r"]
    best = max(out.values())
    return dict(**extra, tag=tag, optimal=opt, best_ordered=best,
                b1=out["B1_sequential"],
                best_name=max(out, key=out.get),
                gain_vs_b1=100 * (opt / out["B1_sequential"] - 1),
                gain_vs_best=100 * (opt / best - 1),
                status=sol.status, gap=(sol.gap if sol.gap is not None else ""),
                n_sown=sol.n_sown, merged_fraction=sol.merged_fraction,
                n_cands=len(cands))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results_fc_main")
    a = ap.parse_args()
    ph = DssatPhenology("results/phenology_table.npz")
    m = cap.make("visits"); m.capacity = 2.0
    ref = ph.sow_days[len(ph.sow_days) // 2]
    rows = []

    # --- R7 replacement: scenario grid at N = 70 ---
    grid = R.scenario_grid()
    print(f"scenario grid: {len(grid)} cells", flush=True)
    for s in range(1, a.seeds + 1):
        fields = build.build_farm(ph, 70, s, ref)
        for sc in grid:
            t0 = time.time()
            c, _ = build.build_candidates(ph, fields, W, m, scenario=sc)
            r = cell(ph, fields, c, m, "scenario", sc,
                     dict(n_fields=70, farm_seed=s, curve=sc.curve,
                          plateau=sc.plateau, kappa=sc.kappa))
            rows.append(r)
            print(f"  s={s} {sc.curve} {sc.plateau} k={sc.kappa} "
                  f"({time.time()-t0:5.1f}s) vsB1={r['gain_vs_b1']:+6.2f} "
                  f"vsBest={r['gain_vs_best']:+6.2f}", flush=True)

    # --- R8 replacement: merging disallowed ---
    for n in (40, 70):
        for s in range(1, a.seeds + 1):
            fields = build.build_farm(ph, n, s, ref)
            c, _ = build.build_candidates(ph, fields, W, m)
            for tag, cc in (("merge_allowed", c),
                            ("merge_disallowed", [x for x in c if not x.merged])):
                r = cell(ph, fields, cc, m, tag, R.PRIMARY_SCENARIO,
                         dict(n_fields=n, farm_seed=s, curve="winter_y1",
                              plateau="PL2", kappa=1.5))
                rows.append(r)
                print(f"  N={n} s={s} {tag:17s} vsB1={r['gain_vs_b1']:+6.2f} "
                      f"vsBest={r['gain_vs_best']:+6.2f} "
                      f"merged={r['merged_fraction']:.2f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/FC_robust_raw.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}/FC_robust_raw.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
