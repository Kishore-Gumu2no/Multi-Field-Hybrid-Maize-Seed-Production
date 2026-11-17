"""
Six operational scheduling strategies, defined before evaluation.

The original study compared its optimiser against one invented baseline --
fields taken in an arbitrary fixed order, each started on the first day the
crew was free.  That is a weak and arbitrary competitor, and a referee is
entitled to ask whether a competent human rule would do as well.  This module
defines six strategies spanning what a supervisor could plausibly do by hand
up to the exact joint solve.

All six are given exactly the same information: the per-hybrid agronomic
offsets and, where they use one, the expected-yield table built from the
PLANNING ENSEMBLE only.  None of them sees the held-out season.  The only
difference between them is how fields are prioritised and whether the
assignment is made one field at a time or jointly.

B1  sequential          fields in registration order, first feasible day
B2  earliest-feasible   fields ordered by the earliest day they could start
B3  largest-field-first fields ordered by area, descending
B4  hardest-sync-first  fields ordered by |recommended offset|, descending
B5  greedy-yield        fields ordered as B4, each taking its best remaining
                        (day, offset) from the ensemble expected-yield table
B6  joint optimisation  exact MILP over the same table

B1-B4 apply the agronomic offset and search for a day; B5 chooses day and
offset together but one field at a time; B6 chooses for all fields at once.
The step from B5 to B6 is the value of solving jointly rather than greedily,
which is the question "could a simple script do this?" in testable form.
"""

import numpy as np
import pulp

from .capacity import choice_cost, parts_by_day

NAMES = ["B1_sequential", "B2_earliest", "B3_largest", "B4_hardest",
         "B5_greedy", "B6_joint"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fits(load, cost, cap):
    return all(load.get(d, 0.0) + c <= cap + 1e-9 for d, c in cost.items())


def _apply(load, cost):
    for d, c in cost.items():
        load[d] = load.get(d, 0.0) + c


def _feasible_days(inst, o):
    lo, hi = inst.w0, inst.w0 + inst.T - 1
    out = []
    for d in inst.days:
        v = inst.visits(d, o)
        if lo <= v[0] and v[-1] <= hi:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# B1-B4: agronomic offset, differing only in the order fields are served
# ---------------------------------------------------------------------------

def _order(inst, which, rec_off):
    fs = list(inst.fields)
    if which == "B1_sequential":
        return fs                                     # registration order
    if which == "B2_earliest":
        return sorted(fs, key=lambda f: (min(_feasible_days(inst, rec_off[f.fid]),
                                             default=10**6), f.fid))
    if which == "B3_largest":
        return sorted(fs, key=lambda f: (-f.area_ha, f.fid))
    if which == "B4_hardest":
        return sorted(fs, key=lambda f: (-abs(rec_off[f.fid]), -f.area_ha, f.fid))
    raise ValueError(which)


def plan_ordered(inst, model, rec_off, which):
    """Serve fields in a fixed order, each on the first day its visits fit."""
    load, plan = {}, {}
    for f in _order(inst, which, rec_off):
        o = rec_off[f.fid]
        for d in _feasible_days(inst, o):
            cost = choice_cost(model, f, inst, d, o)
            if _fits(load, cost, model.capacity):
                plan[f.fid] = (d, o)
                _apply(load, cost)
                break
    return plan


# ---------------------------------------------------------------------------
# B5: greedy on the ensemble expected-yield table
# ---------------------------------------------------------------------------

def plan_greedy(inst, model, tbl, rec_off):
    """
    Hardest cross first; each field takes the feasible (day, offset) with the
    highest expected yield that still fits the remaining capacity.  This is
    the best a one-field-at-a-time rule can do with the optimiser's own
    information.
    """
    load, plan = {}, {}
    by_field = {}
    for (fid, d, o), v in tbl.items():
        by_field.setdefault(fid, []).append((v, d, o))
    for lst in by_field.values():
        lst.sort(reverse=True)
    order = sorted(inst.fields,
                   key=lambda f: (-abs(rec_off[f.fid]), -f.area_ha, f.fid))
    for f in order:
        for _, d, o in by_field.get(f.fid, ()):
            cost = choice_cost(model, f, inst, d, o)
            if _fits(load, cost, model.capacity):
                plan[f.fid] = (d, o)
                _apply(load, cost)
                break
    return plan


# ---------------------------------------------------------------------------
# B6: exact joint optimisation under the same capacity model
# ---------------------------------------------------------------------------

def plan_joint(inst, model, tbl, time_limit=45, gap_rel=0.002, msg=False):
    """
    MILP: assign every field a (day, offset) under the capacity model.

    Solved to a proved relative optimality gap rather than to exact
    optimality.  Under capacity models where merging two sowings saves
    nothing, the problem has many symmetric near-optimal solutions and
    proving optimality is far more expensive than reaching it.  The objective
    The objective is total expected yield, with no per-field bonus.  A field
    left unsown contributes nothing, so yield maximisation already prefers
    sowing a field wherever it can; the large bonus used in the original
    study is unnecessary and makes the linear relaxation so loose that CBC
    converges more slowly and, in tests at N = 70, returned a WORSE incumbent
    in the same time (34 fields and 283 t against 38 fields and 347 t).

    Under the hours and hectares models the capacity coefficients are
    fractional, giving a multi-dimensional knapsack that CBC does not close
    at these sizes.  Those solves return the incumbent at the time limit,
    which makes B6 a LOWER bound on what joint optimisation achieves -- 
    conservative in the direction that matters, since B6 is the strategy
    whose advantage is under test.  Solve time and status are recorded for
    every instance and reported.
    """
    fmap = {f.fid: f for f in inst.fields}
    x, cost_cache = {}, {}
    for (fid, d, o) in inst.feasible_choices():
        if (fid, d, o) not in tbl:
            continue
        x[(fid, d, o)] = pulp.LpVariable(f"x_{fid}_{d}_{o}", cat="Binary")
        cost_cache[(fid, d, o)] = choice_cost(model, fmap[fid], inst, d, o)
    if not x:
        return {}, 0.0, "Empty", 0

    prob = pulp.LpProblem("capacity_schedule", pulp.LpMaximize)
    prob += pulp.lpSum(tbl[k] * v for k, v in x.items())

    for f in inst.fields:
        vs = [v for k, v in x.items() if k[0] == f.fid]
        if vs:
            prob += pulp.lpSum(vs) <= 1

    byday = {}
    for k, v in x.items():
        for d, c in cost_cache[k].items():
            byday.setdefault(d, []).append((c, v))
    for d, terms in byday.items():
        prob += pulp.lpSum(c * v for c, v in terms) <= model.capacity

    import time
    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit,
                                 gapRel=gap_rel))
    secs = time.time() - t0
    status = pulp.LpStatus[prob.status]
    plan = {k[0]: (k[1], k[2]) for k, v in x.items()
            if v.value() is not None and v.value() > 0.5}
    return plan, secs, status, len(x)


# ---------------------------------------------------------------------------
# Execution under a capacity model
# ---------------------------------------------------------------------------

def execute(inst, model, plan, weather, access, curve=None, sharpness=1.0):
    """
    Walk the season and carry out the plan against the realised weather,
    charging capacity through `model`.  An operation whose field is closed by
    rain, or for which the day's budget is exhausted, slips to the next day.
    """
    from .core import PRIMARY_CURVE
    curve = curve or PRIMARY_CURVE
    T_end = inst.w0 + inst.T
    fmap = {f.fid: f for f in inst.fields}
    pending = dict(plan)
    done = {}

    for day in range(inst.w0, T_end):
        due = {}
        for fid, (d_f, o) in pending.items():
            d_m = d_f + o
            rec = done.get(fid, {})
            parts = []
            if d_f <= day and "f" not in rec:
                parts.append((d_f, "f"))
            if d_m <= day and "m" not in rec:
                parts.append((d_m, "m"))
            if inst.split_male and d_m + inst.male_split_gap <= day and "m2" not in rec:
                parts.append((d_m + inst.male_split_gap, "m2"))
            if parts:
                due[fid] = sorted(parts)

        # serve fields whose operations are most overdue first
        budget = model.capacity
        for fid in sorted(due, key=lambda i: due[i][0][0]):
            if not access[fid][day]:
                continue
            parts = tuple(p for _, p in due[fid])
            # take as many of this field's due operations as the budget allows
            for take in range(len(parts), 0, -1):
                sub = parts[:take]
                c = model.serve_cost(fmap[fid], sub)
                if c <= budget + 1e-9:
                    rec = done.setdefault(fid, {})
                    for p in sub:
                        rec[p] = day
                    budget -= c
                    break

        needed = {"f", "m"} | ({"m2"} if inst.split_male else set())
        pending = {fid: v for fid, v in pending.items()
                   if not needed.issubset(done.get(fid, {}).keys())}

    total, realised = 0.0, {}
    for f in inst.fields:
        rec = done.get(f.fid, {})
        if "f" not in rec or "m" not in rec:
            realised[f.fid] = (None, None, np.nan, 0.0)
            continue
        y = f.yield_tons(weather, rec["m"], rec["f"], curve=curve,
                         sharpness=sharpness)
        realised[f.fid] = (rec["f"], rec["m"], f.tsi(weather, rec["m"], rec["f"]), y)
        total += y
    return dict(total_yield=total, detail=realised,
                n_sown=sum(1 for v in realised.values() if v[0] is not None))
