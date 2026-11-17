"""Deterministic solver-scaling instances. NOT a research experiment."""
import time
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.phenology import DssatPhenology
from sim import build, optimise, response
from sim.capacity import make

ph = DssatPhenology("results/phenology_table.npz")
rd = ph.sow_days[len(ph.sow_days) // 2]
W = list(range(300, 331))
print("Deterministic solver-scaling check. Seeds fixed; no experiment is run.")
print(f"window {len(W)} days, scenario {response.PRIMARY_SCENARIO.curve}/"
      f"{response.PRIMARY_SCENARIO.plateau}/k={response.PRIMARY_SCENARIO.kappa}\n")
print(f"{'model':11s} {'N':>3s} {'cands':>6s} {'Z':>9s} {'LPbound':>9s} "
      f"{'gap':>8s} {'t(s)':>6s} {'status':>9s} {'T7.5':>6s} {'termination':>16s} {'lowerbound':>10s}")
for model_name in ("visits", "operations", "hours"):
    for n in (10, 20, 40):
        m = make(model_name)
        farm = build.build_farm(ph, n, 1, rd)
        c, _ = build.build_candidates(ph, farm, W, m)
        t0 = time.time()
        s = optimise.solve(c, capacity=m.capacity, days=W, objective="OBJ-A",
                           time_limit=120, gap_rel=0.002)
        dt = time.time() - t0
        r = build.execute(ph, farm, s)
        ok = abs(s.objective - r["total_area_r"]) <= 1e-9
        g = "-" if s.gap is None else f"{s.gap:.5f}"
        b = "-" if s.lp_bound is None else f"{s.lp_bound:9.4f}"
        print(f"{model_name:11s} {n:3d} {len(c):6d} {s.objective:9.4f} {b:>9s} "
              f"{g:>8s} {dt:6.2f} {s.status:>9s} {str(ok):>6s} {s.termination:>16s} {str(s.is_lower_bound):>10s}", flush=True)
print("\nAcceptance: primary results require proven Optimal or measured gap <= 0.005.")
print("A larger gap means the objective is a LOWER BOUND and must be labelled so.")
print("The gap is MEASURED against the LP relaxation, not taken from solver status.")
