"""Scoring-parity evidence for all eleven planners. No experiment is run."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.phenology import DssatPhenology
from sim import build, compare, response
from sim.capacity import make

ph = DssatPhenology("results/phenology_table.npz")
rd = ph.sow_days[len(ph.sow_days) // 2]
W = list(range(300, 312))
farm = build.build_farm(ph, 8, 1, rd)
print("Scoring parity across S1-S5 and B1-B6. One shared candidate set.")
print("No strategy comparison is drawn; no performance claim is made.\n")
for mn in ("visits", "operations", "hours"):
    m = make(mn)
    c, _ = build.build_candidates(ph, farm, W, m)
    out = compare.run_all(ph, farm, W, m, c, response.PRIMARY_SCENARIO)
    print(f"--- capacity model: {mn} ({len(c)} candidates, capacity {m.capacity}) ---")
    print(f"{'planner':18s} {'objective':>10s} {'execution':>10s} {'|diff|':>9s} "
          f"{'fields':>6s} {'ha':>6s} {'merged':>7s}")
    worst = 0.0
    for k, v in out.items():
        d = abs(v["objective"] - v["execution"]); worst = max(worst, d)
        print(f"{k:18s} {v['objective']:10.4f} {v['execution']:10.4f} {d:9.1e} "
              f"{v['n_sown']:6d} {v['ha_sown']:6.2f} {v['merged_fraction']:7.2f}")
    print(f"worst |objective - execution| across all eleven: {worst:.1e}\n")
