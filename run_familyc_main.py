"""
P1 re-run: the main scheduling results on the CORRECTED Family C engine.

WHY THIS EXISTS
---------------
Every primary scheduling number in the manuscript (R1-R8, E1-E6, V1) was
produced by Family A runners. Family A carries defects that `sim/compare.py`
documents and excludes by construction:

  * a 10x per-field bonus in the MILP objective, so the solver optimised a
    different function from the one reported
  * `core.seed_yield`'s falsified rectangular-overlap normaliser
  * first-male-only scoring, ignoring the split male sowing
  * three divergent scoring paths between strategies, baselines and execution
  * the ensemble mean taken over TSI rather than over r(TSI)

This reruns the assignment, current-practice, greedy and capacity-scaling
results through Family C, where every planner selects from ONE candidate set
and execution re-scores through the SAME `scoring.score_plan`.

Design mirrors legacy R6: N in {10..70}, capacity 2 visits/day, the full
21-season ensemble, Rabi window DOY 274-334.

Planner map, legacy name -> Family C name:
    S1_agronomic     -> B1_sequential   (per-hybrid recommended offset, in order)
    S1_greedy        -> B5_greedy       (hardest cross first, best feasible)
    S3_static_robust -> B6_joint        (exact MILP, OBJ-A, no bonus)

Metric is ensemble-mean relative-yield-hectares (`total_area_r`), the objective
the optimiser actually maximises. Absolute t/ha is deliberately NOT reported:
it required the falsified normaliser.
"""
import argparse, csv, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import build, capacity as cap, compare, optimise, response as R
from sim.phenology import DssatPhenology
from sim.runspec import git_commit, table_sha

TABLE = "results/phenology_table.npz"
WINDOW = list(range(274, 335))
PLANNERS = ("B1_sequential", "B5_greedy", "B6_joint")
LEGACY = {"B1_sequential": "S1_agronomic", "B5_greedy": "S1_greedy",
          "B6_joint": "S3_static_robust"}


def run_cell(ph, n_fields, seed, model, time_limit, gap_rel):
    ref = ph.sow_days[len(ph.sow_days) // 2]
    fields = build.build_farm(ph, n_fields, seed, ref)
    cands, diag = build.build_candidates(ph, fields, WINDOW, model)
    rows = []
    for name in PLANNERS:
        t0 = time.time()
        if name == "B6_joint":
            sol = compare.plan_joint(cands, model.capacity, WINDOW,
                                     objective="OBJ-A", name=name,
                                     time_limit=time_limit, gap_rel=gap_rel)
        elif name == "B1_sequential":
            sol = compare.plan_ordered(fields, cands, model.capacity, name)
        elif name == "B5_greedy":
            sol = compare.plan_greedy(fields, cands, model.capacity)
        else:
            raise ValueError(name)
        res = build.execute(ph, fields, sol, scenario=R.PRIMARY_SCENARIO)
        el = time.time() - t0
        g = sol.gap
        rows.append(dict(
            n_fields=n_fields, farm_seed=seed, planner=name,
            legacy_name=LEGACY[name],
            objective=float(sol.objective), execution=float(res["total_area_r"]),
            n_sown=int(sol.n_sown), ha_sown=float(res["ha_sown"]),
            ha_total=float(sum(f.area_ha for f in fields)),
            status=sol.status, gap=("" if g is None else float(g)),
            is_lower_bound=bool(getattr(sol, "is_lower_bound", False)),
            termination=getattr(sol, "termination", ""),
            n_candidates=len(cands), solve_s=round(el, 3),
            pressure=3.0 * n_fields / (len(WINDOW) * model.capacity),
            n_field_years=int(res["reliability"].n_field_years),
            n_out_of_domain=int(res["reliability"].n_out_of_domain)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="10,20,30,35,37,40,43,45,48,55,70")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--capacity", type=int, default=2)
    ap.add_argument("--model", default="visits")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--gap-rel", type=float, default=0.005)
    ap.add_argument("--out", default="results_fc_main")
    a = ap.parse_args()

    ns = [int(x) for x in a.ns.split(",")]
    os.makedirs(a.out, exist_ok=True)
    ph = DssatPhenology(TABLE)
    model = cap.make(a.model)
    model.capacity = float(a.capacity)
    print(f"engine=FamilyC commit={git_commit()} table={table_sha(TABLE)[:12]}")
    print(f"model={a.model} capacity={a.capacity} window={WINDOW[0]}-{WINDOW[-1]} "
          f"years={len(ph.years)} ns={ns} seeds=1..{a.seeds}", flush=True)

    allrows = []
    for n in ns:
        for s in range(1, a.seeds + 1):
            t0 = time.time()
            rows = run_cell(ph, n, s, model, a.time_limit, a.gap_rel)
            allrows += rows
            d = {r["planner"]: r for r in rows}
            base = d["B1_sequential"]["execution"]
            print(f"N={n:3d} s={s} ({time.time()-t0:6.1f}s) "
                  f"practice={base:8.3f} "
                  f"greedy={100*(d['B5_greedy']['execution']/base-1):+7.2f}% "
                  f"optimal={100*(d['B6_joint']['execution']/base-1):+7.2f}% "
                  f"sown={base and d['B1_sequential']['n_sown']}/"
                  f"{d['B6_joint']['n_sown']} "
                  f"gap={d['B6_joint']['gap']} {d['B6_joint']['status']}",
                  flush=True)
            with open(f"{a.out}/FC_main_raw.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys()))
                w.writeheader(); w.writerows(allrows)
    print(f"\nwrote {a.out}/FC_main_raw.csv  ({len(allrows)} rows)")


if __name__ == "__main__":
    main()
