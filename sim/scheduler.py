"""
Scheduling strategies for multi-field hybrid maize seed production.

Decision structure
------------------
Each field i needs two sowing operations: the female line and the male line.
A single supervised sowing resource can make at most K field-visits per day.
If the two lines are sown on the same day the field needs one visit; if they
are staggered it needs two. A visit can only happen on a day when that field
is trafficable.

Choosing the female sowing day sets the thermal-time trajectory of the whole
field, and the male offset sets the tasseling-silking interval. Both depend on
weather that is unknown when the plan is made.

Strategies
----------
S1 CONVENTIONAL  fixed field order, uniform rule-of-thumb male offset
S2 STATIC-EV     MILP solved once against a single mean-weather forecast
S3 STATIC-ROBUST MILP solved once maximising expected yield over an ensemble
S4 DYNAMIC       S3 plan, re-optimised each day on observed weather to date
S5 WAIT-AND-SEE  MILP with perfect knowledge of the realised weather

S2 vs S3 gives the value of the stochastic solution (VSS).
S3 vs S4 gives the value of adaptivity -- the question this study asks.
S5 is the upper bound; the gap to it is the expected value of perfect
information (EVPI).
"""

import numpy as np
import pulp

from .core import (WeatherGenerator, daily_gdd, flowering_day, seed_yield,
                   trafficability, PRIMARY_CURVE)


# ----------------------------------------------------------------------------
# Problem instance
# ----------------------------------------------------------------------------

class Instance:
    """
    One scheduling instance: fields, a sowing window, and crew capacity.

    `split_male` reflects standard seed-production practice. Chassaigne-
    Ricciulli et al. (2021) recommend sowing the male line on two consecutive
    dates for most hybrids, so that pollen is available across the whole
    female receptive window rather than only at its start. With split_male
    on, a field therefore needs three visits -- two male, one female --
    rather than two, which tightens the demand on the shared crew by half as
    much again.
    """

    def __init__(self, fields, window_start=0, window_len=30,
                 capacity=2, offsets=range(-12, 3), split_male=True,
                 male_split_gap=2):
        self.fields = fields
        self.w0 = window_start
        self.T = window_len
        self.capacity = capacity
        self.offsets = list(offsets)
        self.split_male = split_male
        self.male_split_gap = male_split_gap

    @property
    def days(self):
        return list(range(self.w0, self.w0 + self.T))

    def visits(self, d_f, offset):
        """
        Days on which the crew must attend this field.

        The female is sown on d_f and the male on d_f + offset; with
        split_male the male is sown again `male_split_gap` days later. Visits
        falling on the same day are one visit.
        """
        d_m = d_f + offset
        days = {d_f, d_m}
        if self.split_male:
            days.add(d_m + self.male_split_gap)
        return tuple(sorted(days))

    def span(self, d_f, offset):
        """First and last day this field occupies."""
        v = self.visits(d_f, offset)
        return v[0], v[-1]

    def feasible_choices(self):
        """All (field, female_day, offset) triples whose visits fit the window."""
        lo, hi = self.w0, self.w0 + self.T - 1
        out = []
        for f in self.fields:
            for d in self.days:
                for o in self.offsets:
                    v = self.visits(d, o)
                    if lo <= v[0] and v[-1] <= hi:
                        out.append((f.fid, d, o))
        return out


# ----------------------------------------------------------------------------
# Yield lookup tables
# ----------------------------------------------------------------------------

def yield_table(inst, weathers, curve=PRIMARY_CURVE):
    """
    Mean yield (tonnes) for each (field, female_day, offset) averaged over the
    supplied weather realisations. `weathers` is a list of weather dicts.
    """
    tbl = {}
    lo, hi = inst.w0, inst.w0 + inst.T - 1
    for f in inst.fields:
        gdds = [f.gdd_series(w) for w in weathers]
        for d in inst.days:
            for o in inst.offsets:
                if not (lo <= d + o <= hi):
                    continue
                vals = []
                for g in gdds:
                    dt = flowering_day(g, d + o, f.tau_m)
                    ds = flowering_day(g, d, f.tau_f)
                    if not np.isfinite(dt) or not np.isfinite(ds):
                        vals.append(0.0)
                    else:
                        vals.append(float(seed_yield(dt - ds, curve=curve)))
                tbl[(f.fid, d, o)] = float(np.mean(vals)) * f.area_ha
    return tbl


def mean_weather(gen, n=200):
    """Climatological mean weather series -- the 'single forecast' for S2."""
    draws = [gen.sample() for _ in range(n)]
    tmax = np.mean([w["tmax"] for w in draws], axis=0)
    tmin = np.mean([w["tmin"] for w in draws], axis=0)
    rain = np.mean([w["rain"] for w in draws], axis=0)
    return dict(tmax=tmax, tmin=tmin, rain=rain, gdd=daily_gdd(tmax, tmin))


# ----------------------------------------------------------------------------
# MILP
# ----------------------------------------------------------------------------

def solve_milp(inst, tbl, access=None, fixed=None, only_fields=None,
               min_day=None, msg=False, time_limit=12):
    """
    Assign each field a (female_day, offset) maximising total expected yield.

    access   dict fid -> boolean array of trafficable days (None = all open)
    fixed    dict fid -> (female_day, offset) already committed; these consume
             capacity on their visit days but are not re-decided
    only_fields  restrict the decision to this set of field ids
    min_day  no visit may be scheduled before this day (used when replanning
             part-way through the season)
    """
    fids = [f.fid for f in inst.fields] if only_fields is None else list(only_fields)
    fixed = fixed or {}

    prob = pulp.LpProblem("anthesis_schedule", pulp.LpMaximize)

    x = {}
    for (fid, d, o) in inst.feasible_choices():
        if fid not in fids:
            continue
        vd = inst.visits(d, o)
        if min_day is not None and min(vd) < min_day:
            continue
        if access is not None:
            if not all(access[fid][v] for v in vd):
                continue
        x[(fid, d, o)] = pulp.LpVariable(f"x_{fid}_{d}_{o}", cat="Binary")

    if not x:
        return None

    # Objective. Each scheduled field earns its expected yield plus a large
    # constant bonus, so the solver only ever leaves a field unsown when
    # capacity genuinely runs out. This keeps the model feasible instead of
    # failing outright when the window gets tight late in the season.
    bonus = 10.0 * (max(tbl.values()) if tbl else 1.0)
    prob += pulp.lpSum((tbl[k] + bonus) * v for k, v in x.items())

    for fid in fids:
        vs = [v for k, v in x.items() if k[0] == fid]
        if vs:
            prob += pulp.lpSum(vs) <= 1

    # crew capacity per day, including visits locked in by `fixed`
    locked = {d: 0 for d in inst.days}
    for fid, (d, o) in fixed.items():
        for v in inst.visits(d, o):
            if v in locked:
                locked[v] += 1

    for day in inst.days:
        terms = [v for k, v in x.items() if day in inst.visits(k[1], k[2])]
        if terms or locked[day]:
            prob += pulp.lpSum(terms) + locked[day] <= inst.capacity

    prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit))

    if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved"):
        return None

    plan = {}
    for (fid, d, o), v in x.items():
        if v.value() is not None and v.value() > 0.5:
            plan[fid] = (d, o)
    return plan


# ----------------------------------------------------------------------------
# Execution engine (shared by all strategies)
# ----------------------------------------------------------------------------

def execute(inst, plan, weather, access, replan_fn=None, curve=PRIMARY_CURVE,
            sharpness=1.0):
    """
    Walk the season day by day and carry out the plan against real weather.

    Any operation whose field is untrafficable, or for which the crew has no
    capacity left that day, slips to the next day. If `replan_fn` is given it
    is called at the start of each day with the state so far and may return a
    revised plan for the fields not yet sown -- this is what makes a strategy
    adaptive. The revision may only use information available up to that day.

    Returns realised yield (t), the executed schedule, and diagnostics.
    """
    T_end = inst.w0 + inst.T
    pending = dict(plan)          # fid -> (female_day, offset) still to execute
    done = {}                     # fid -> {'f': day, 'm': day}
    n_replans = 0
    schedule_changes = 0

    for day in range(inst.w0, T_end):
        if replan_fn is not None and pending:
            revised = replan_fn(day, pending, done)
            if revised is not None:
                n_replans += 1
                schedule_changes += sum(
                    1 for fid in revised if fid in pending and revised[fid] != pending[fid])
                pending = revised

        used = 0
        # operations due today, earliest-planned first. With split male
        # sowing a field has three: the female, and the male on two dates.
        due = []
        for fid, (d_f, o) in pending.items():
            d_m = d_f + o
            if d_f <= day:
                due.append((d_f, fid, "f"))
            if d_m <= day:
                due.append((d_m, fid, "m"))
            if inst.split_male and d_m + inst.male_split_gap <= day:
                due.append((d_m + inst.male_split_gap, fid, "m2"))
        due.sort()

        visited_today = set()
        for _, fid, part in due:
            rec = done.setdefault(fid, {})
            if part in rec:
                continue
            if not access[fid][day]:
                continue
            new_visit = fid not in visited_today
            if new_visit and used >= inst.capacity:
                continue
            rec[part] = day
            if new_visit:
                visited_today.add(fid)
                used += 1

        needed = {"f", "m"} | ({"m2"} if inst.split_male else set())
        pending = {fid: v for fid, v in pending.items()
                   if not needed.issubset(done.get(fid, {}).keys())}

    total = 0.0
    realised = {}
    for f in inst.fields:
        rec = done.get(f.fid, {})
        if "f" not in rec or "m" not in rec:
            realised[f.fid] = (None, None, np.nan, 0.0)
            continue
        # Yield is scored from the FIRST male sowing only. The published
        # regression we score against was fitted on single-date male sowing,
        # so crediting the second date with a wider pollen window would go
        # beyond the data. We therefore charge the operational cost of the
        # second visit without claiming its agronomic benefit, which is the
        # conservative direction.
        y = f.yield_tons(weather, rec["m"], rec["f"], curve=curve,
                         sharpness=sharpness)
        tsi = f.tsi(weather, rec["m"], rec["f"])
        realised[f.fid] = (rec["f"], rec["m"], tsi, y)
        total += y

    n_sown = sum(1 for v in realised.values() if v[0] is not None)
    return dict(total_yield=total, detail=realised, n_sown=n_sown,
                n_replans=n_replans, schedule_changes=schedule_changes)
