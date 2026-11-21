"""
Strategies S1-S5 and baselines B1-B6, all scored through `scoring.score_plan`.

WHY THIS MODULE EXISTS  (closes U4)
-----------------------------------
The Phase 3 audit found THREE independent scoring paths in the superseded
engine: two `execute` functions and an inline scorer. Any divergence
between them silently invalidates every strategy-versus-baseline comparison and
produces results that look entirely reasonable. The project has already been
caught by one instance of this class: the 10x bonus meant the solver optimised a
different objective from the one reported.

THE FIX IS STRUCTURAL, NOT A TEST
---------------------------------
Every planner here selects from ONE candidate set built by
`build.build_candidates`. Each candidate already carries:

    area_r  <- scoring.score_plan(...)        the relative-yield definition
    cost    <- build.plan_cost(model, ...)    the operational-cost definition

so all eleven planners share both definitions by construction. A planner's only
freedom is WHICH candidate it picks. Execution then re-scores through the same
`score_plan`, so the objective coefficient and the execution score are the same
number for every path, not merely equal by agreement.

The legacy strategy and baseline modules were left UNTOUCHED at the time, so
historical results stay reproducible. Nothing here imports them.

WHAT CANNOT ENTER THIS PATH
---------------------------
  * the 10x bonus          -- `optimise.solve` has no bonus constant, and a test
                              parses the module to prove it
  * first-male-only scoring-- `score_plan` takes `SowingPlan(sow_f, sow_m1,
                              sow_m2)` and credits the second sowing through the
                              split ratio
  * the superseded `seed_yield` -- absolute t/ha on a Y(0) normaliser with the
                              falsified rectangular overlap. Not imported here
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import build as _build
from . import optimise as _optimise
from . import response as _response

STRATEGIES = ("S1_conventional", "S2_static_ev", "S3_static_robust",
              "S4_dynamic", "S5_wait_and_see")
BASELINES = ("B1_sequential", "B2_earliest", "B3_largest", "B4_hardest",
             "B5_greedy", "B6_joint")
ALL_PLANNERS = BASELINES + STRATEGIES

# S1's rule-of-thumb offset. ASSUMED: a fixed number of days, applied to every
# field regardless of its parents, which is what "conventional practice" means
# in this comparison. It is deliberately NOT tuned.
CONVENTIONAL_OFFSET = -3


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------

def _fits(load, cost, capacity):
    return all(load.get(d, 0.0) + c <= capacity + 1e-9 for d, c in cost.items())


def _apply(load, cost):
    for d, c in cost.items():
        load[d] = load.get(d, 0.0) + c


def by_field(cands):
    out = {}
    for c in cands:
        out.setdefault(c.fid, []).append(c)
    return out


def offset_of(c):
    return int(c.plan.sow_m1) - int(c.plan.sow_f)


def recommended_offset(cands_for_field):
    """
    The agronomic recommendation for one field: the male offset whose mean
    relative yield per hectare is highest, averaged over the days it is
    available.

    DECLARED, not tuned. This is the information a competent supervisor would
    have -- "for this cross, sow the male about N days early" -- without the
    day-by-day optimisation the MILP performs. B1-B4 apply it and then differ
    only in the order they serve fields.
    """
    agg = {}
    for c in cands_for_field:
        agg.setdefault(offset_of(c), []).append(c.area_r / max(c.area_ha, 1e-12))
    if not agg:
        return None
    return max(agg, key=lambda o: float(np.mean(agg[o])))


def _solution(name, chosen, n_cands, status="Heuristic"):
    return _optimise.Solution(
        objective_name=name,
        objective=float(sum(c.area_r for c in chosen)),
        chosen=list(chosen), status=status, gap=None,
        stage1_value=None, n_candidates=n_cands)


# ---------------------------------------------------------------------------
# B1-B4: the agronomic offset, differing only in service order
# ---------------------------------------------------------------------------

def _order_fields(fields, which, rec_off, cands_by_field, capacity):
    fs = list(fields)
    if which == "B1_sequential":
        return sorted(fs, key=lambda f: f.fid)
    if which == "B2_earliest":
        def earliest(f):
            ds = [min(c.cost) for c in cands_by_field.get(f.fid, [])
                  if offset_of(c) == rec_off.get(f.fid)]
            return (min(ds) if ds else 10 ** 6, f.fid)
        return sorted(fs, key=earliest)
    if which == "B3_largest":
        return sorted(fs, key=lambda f: (-f.area_ha, f.fid))
    if which == "B4_hardest":
        return sorted(fs, key=lambda f: (-abs(rec_off.get(f.fid) or 0),
                                         -f.area_ha, f.fid))
    raise ValueError(which)


def plan_ordered(fields, cands, model_capacity, which):
    """Serve fields in a fixed order; each takes the earliest day its visits fit."""
    cbf = by_field(cands)
    rec_off = {f.fid: recommended_offset(cbf.get(f.fid, [])) for f in fields}
    load, chosen = {}, []
    for f in _order_fields(fields, which, rec_off, cbf, model_capacity):
        opts = [c for c in cbf.get(f.fid, []) if offset_of(c) == rec_off.get(f.fid)]
        opts.sort(key=lambda c: (min(c.cost), c.label))
        for c in opts:
            if _fits(load, c.cost, model_capacity):
                chosen.append(c)
                _apply(load, c.cost)
                break
    return _solution(which, chosen, len(cands))


# ---------------------------------------------------------------------------
# B5: greedy on the same expected-yield coefficients the MILP uses
# ---------------------------------------------------------------------------

def plan_greedy(fields, cands, model_capacity):
    """Hardest cross first; each field takes its best still-feasible candidate."""
    cbf = by_field(cands)
    rec_off = {f.fid: recommended_offset(cbf.get(f.fid, [])) for f in fields}
    order = sorted(fields, key=lambda f: (-abs(rec_off.get(f.fid) or 0),
                                          -f.area_ha, f.fid))
    load, chosen = {}, []
    for f in order:
        opts = sorted(cbf.get(f.fid, []), key=lambda c: (-c.area_r, c.label))
        for c in opts:
            if _fits(load, c.cost, model_capacity):
                chosen.append(c)
                _apply(load, c.cost)
                break
    return _solution("B5_greedy", chosen, len(cands))


# ---------------------------------------------------------------------------
# B6 / S3: exact joint optimisation
# ---------------------------------------------------------------------------

def plan_joint(cands, capacity, days, objective="OBJ-A", name="B6_joint",
               time_limit=_optimise.DEFAULT_TIME_LIMIT,
               gap_rel=_optimise.DEFAULT_GAP_REL, locked_load=None):
    sol = _optimise.solve(cands, capacity=capacity, days=days,
                          objective=objective, time_limit=time_limit,
                          gap_rel=gap_rel, locked_load=locked_load)
    sol.objective_name = name
    return sol


# ---------------------------------------------------------------------------
# S1: conventional -- fixed order, one rule-of-thumb offset for every field
# ---------------------------------------------------------------------------

def plan_conventional(fields, cands, model_capacity, offset=CONVENTIONAL_OFFSET):
    cbf = by_field(cands)
    load, chosen = {}, []
    for f in sorted(fields, key=lambda f: f.fid):
        opts = [c for c in cbf.get(f.fid, []) if offset_of(c) == offset]
        opts.sort(key=lambda c: (min(c.cost), c.label))
        for c in opts:
            if _fits(load, c.cost, model_capacity):
                chosen.append(c)
                _apply(load, c.cost)
                break
    return _solution("S1_conventional", chosen, len(cands))


# ---------------------------------------------------------------------------
# S4: dynamic -- rolling re-solve with non-anticipativity
# ---------------------------------------------------------------------------

def plan_dynamic(fields, cands, capacity, days, replan_days=None,
                 objective="OBJ-A"):
    """
    Solve on the full ensemble, then re-solve at each replan day for the fields
    not yet committed, with the capacity already consumed locked in.

    NON-ANTICIPATIVITY: a candidate whose earliest visit day is strictly before
    the replan day cannot be chosen at that replan, because that decision would
    have had to be made before the day arrived. Without this the strategy leaks
    future weather and the comparison is meaningless.

    This routine re-solves over the SAME candidate set, so it does not use any
    information the static strategies lack. It is the scheduling-side value of
    adaptivity only; the information-side value needs per-year candidate sets,
    which is a Phase 4 experimental design question, not a scoring one.
    """
    if replan_days is None:
        replan_days = days[::max(1, len(days) // 4)][1:]
    sol = plan_joint(cands, capacity, days, objective=objective, name="S4_dynamic")
    chosen = list(sol.chosen)
    for rd in replan_days:
        committed = [c for c in chosen if min(c.cost) < rd]
        locked = {}
        for c in committed:
            _apply(locked, c.cost)
        open_fids = {f.fid for f in fields} - {c.fid for c in committed}
        pool = [c for c in cands
                if c.fid in open_fids and min(c.cost) >= rd]
        if not pool:
            continue
        re = _optimise.solve(pool, capacity=capacity, days=days,
                             objective=objective, locked_load=locked)
        chosen = committed + list(re.chosen)
    out = _solution("S4_dynamic", chosen, len(cands), status="Rolling")
    out.stage1_value = float(len(replan_days))
    return out


# ---------------------------------------------------------------------------
# One entry point
# ---------------------------------------------------------------------------

def run_planner(name, ph, fields, days, model, cands, scenario, capacity=None,
                asi0=5.0, mode=None, objective="OBJ-A"):
    """
    Build the plan for one planner and execute it.

    `cands` is the single shared candidate set. Every planner receives the same
    one, so relative-yield and cost definitions are identical across all eleven.
    """
    cap = model.capacity if capacity is None else capacity
    if name in ("B1_sequential", "B2_earliest", "B3_largest", "B4_hardest"):
        sol = plan_ordered(fields, cands, cap, name)
    elif name == "B5_greedy":
        sol = plan_greedy(fields, cands, cap)
    elif name == "B6_joint":
        sol = plan_joint(cands, cap, days, objective=objective, name=name)
    elif name == "S1_conventional":
        sol = plan_conventional(fields, cands, cap)
    elif name in ("S2_static_ev", "S3_static_robust", "S5_wait_and_see"):
        # These three differ only in WHICH weather information builds the
        # candidate table; the selection rule is the same exact solve. The
        # caller supplies the appropriate `cands`, which is what makes the
        # value of the stochastic solution and of perfect information
        # measurable. Scoring is identical in all three.
        sol = plan_joint(cands, cap, days, objective=objective, name=name)
    elif name == "S4_dynamic":
        sol = plan_dynamic(fields, cands, cap, days, objective=objective)
    else:
        raise ValueError(f"unknown planner {name!r}; expected one of {ALL_PLANNERS}")

    res = _build.execute(ph, fields, sol, scenario=scenario, asi0=asi0, mode=mode)
    return sol, res


def run_all(ph, fields, days, model, cands, scenario, capacity=None, asi0=5.0,
            mode=None, objective="OBJ-A", names=ALL_PLANNERS):
    """Every planner over the one shared candidate set. No experiment is run."""
    out = {}
    for n in names:
        sol, res = run_planner(n, ph, fields, days, model, cands, scenario,
                               capacity=capacity, asi0=asi0, mode=mode,
                               objective=objective)
        out[n] = dict(solution=sol, result=res,
                      objective=sol.objective,
                      execution=res["total_area_r"],
                      n_sown=sol.n_sown, ha_sown=sol.ha_sown,
                      merged_fraction=sol.merged_fraction,
                      reliability=res["reliability"].as_row())
    return out
