"""
The five scheduling strategies compared in this study.

S1 CONVENTIONAL  : fields taken in order, uniform rule-of-thumb male offset
S2 STATIC-EV     : MILP against a single mean-weather forecast
S3 STATIC-ROBUST : MILP against a weather ensemble, aware of rain slippage
S4 DYNAMIC       : S3 plan, re-optimised daily on information available so far
S5 WAIT-AND-SEE  : MILP with perfect foreknowledge of the realised season

S3 - S2 is the value of the stochastic solution (VSS).
S4 - S3 is the value of adaptivity, which is what this study sets out to measure.
S5 - S4 is the expected value of perfect information (EVPI).
"""

import numpy as np

from .core import (WeatherGenerator, daily_gdd, flowering_day_fast,
                   gdd_cumulative, seed_yield, trafficability, PRIMARY_CURVE)
from .scheduler import Instance, solve_milp, execute, mean_weather


# ----------------------------------------------------------------------------
# Yield tables
# ----------------------------------------------------------------------------

def _slip(day, ok, last):
    """First trafficable day at or after `day`; None if none before `last`."""
    d = day
    while d <= last:
        if ok[d]:
            return d
        d += 1
    return None


def yield_table(inst, weathers, accesses=None, curve=PRIMARY_CURVE,
                sharpness=1.0):
    """
    Expected yield (tonnes) of each (field, female_day, offset) choice.

    If `accesses` is supplied -- a list, one per weather realisation, of
    dicts fid -> boolean trafficable array -- each visit slips to the next
    trafficable day in that realisation before yield is evaluated. This makes
    the resulting plan aware of rain-induced slippage. Crew contention is not
    represented here; it is handled by the execution engine.
    """
    tbl = {}
    lo, hi = inst.w0, inst.w0 + inst.T - 1
    last = len(weathers[0]["gdd"]) - 1
    nS = len(weathers)

    # Fields carrying cultivar names are scored with the precomputed DSSAT
    # table rather than the thermal-time model; the rest of the routine is
    # identical, so both phenology sources go through the same optimiser.
    if inst.fields and hasattr(inst.fields[0], "cult_m"):
        return _yield_table_dssat(inst, weathers, accesses, curve, sharpness)

    for f in inst.fields:
        cums = [gdd_cumulative(f.gdd_series(w)) for w in weathers]

        # cache slipped days and flowering days per scenario
        slip_cache = [{} for _ in range(nS)]
        if accesses is not None:
            for si in range(nS):
                ok = accesses[si][f.fid]
                for d in range(lo, hi + 1):
                    slip_cache[si][d] = _slip(d, ok, last)

        tas_cache = [{} for _ in range(nS)]
        sil_cache = [{} for _ in range(nS)]
        for si, cum in enumerate(cums):
            for d in range(lo, hi + 1):
                dd = slip_cache[si][d] if accesses is not None else d
                if dd is None:
                    tas_cache[si][d] = None
                    sil_cache[si][d] = None
                    continue
                tas_cache[si][d] = flowering_day_fast(cum, dd, f.tau_m)
                sil_cache[si][d] = flowering_day_fast(cum, dd, f.tau_f)

        for d in inst.days:
            for o in inst.offsets:
                if not (lo <= d + o <= hi):
                    continue
                tot = 0.0
                for si in range(nS):
                    dt = tas_cache[si][d + o]
                    ds = sil_cache[si][d]
                    if dt is None or ds is None \
                            or not np.isfinite(dt) or not np.isfinite(ds):
                        continue
                    tot += float(seed_yield(dt - ds, curve=curve, sharpness=sharpness))
                tbl[(f.fid, d, o)] = (tot / nS) * f.area_ha
    return tbl


def access_for(fields, weather):
    """
    Trafficable-day masks keyed by field.

    Trafficability depends only on the drainage rate, and a farm has very few
    distinct soil classes, so we compute one mask per class and share it.
    """
    by_class = {}
    for f in fields:
        k = f.soil_drain_days
        if k not in by_class:
            by_class[k] = trafficability(weather["rain"], k)
    return {f.fid: by_class[f.soil_drain_days] for f in fields}


# ----------------------------------------------------------------------------
# S1  CONVENTIONAL
# ----------------------------------------------------------------------------

def plan_conventional(inst, uniform_offset=-3):
    """
    What a farm manager does without an optimiser: work through the fields in
    a fixed order, start each as early as the crew allows, and apply the same
    recommended male offset everywhere.
    """
    plan, load = {}, {d: 0 for d in inst.days}
    lo, hi = inst.w0, inst.w0 + inst.T - 1
    for f in inst.fields:
        placed = False
        for d in inst.days:
            dm = d + uniform_offset
            if not (lo <= dm <= hi):
                continue
            vd = inst.visits(d, uniform_offset)
            if all(load[v] < inst.capacity for v in set(vd)):
                plan[f.fid] = (d, uniform_offset)
                for v in set(vd):
                    load[v] += 1
                placed = True
                break
        if not placed:
            plan[f.fid] = (hi, 0)
    return plan


def plan_agronomic(inst, weathers, accesses=None, ref_day=None):
    """
    Current best practice, and the baseline this paper should be judged
    against.

    A seed producer does not apply one offset to every hybrid. They look up
    each parent pair's accumulated growing degree days and sow the two lines
    the recommended number of days apart -- exactly the procedure
    Chassaigne-Ricciulli et al. (2021) and Asefa et al. (2024) advocate.
    What they do *not* do is optimise which field gets which day: fields are
    worked in whatever order they come, each started as early as the crew
    allows.

    So this baseline gets the agronomy right and the logistics naive, which
    is the honest description of the state of the art in the field. The
    offset for each field is the one minimising expected absolute TSI over
    the planning ensemble at a reference sowing date, which is what a
    climatological GDD table encodes; it never sees the realised season.
    """
    from .phenology import load as _load_pheno

    ph = _load_pheno()
    lo, hi = inst.w0, inst.w0 + inst.T - 1
    if ref_day is None:
        ref_day = inst.w0 + inst.T // 2

    preps = []
    for w in weathers:
        try:
            preps.append(ph.prepare(inst.fields[0], w))
        except KeyError:
            preps.append(None)
    live = [i for i, pr in enumerate(preps) if pr is not None]

    # one recommended offset per field, from climatology
    rec = {}
    for f in inst.fields:
        best, best_cost = None, np.inf
        for o in inst.offsets:
            if not (lo <= ref_day + o <= hi):
                continue
            vals = []
            for si in live:
                tas = ph.anthesis(preps[si], f, ref_day + o, "m")
                sil = ph.anthesis(preps[si], f, ref_day, "f")
                if np.isfinite(tas) and np.isfinite(sil):
                    vals.append(abs(tas - sil))
            if vals and np.mean(vals) < best_cost:
                best, best_cost = o, float(np.mean(vals))
        rec[f.fid] = best if best is not None else 0

    # naive sequencing: fixed field order, earliest day the crew is free
    plan, load = {}, {d: 0 for d in inst.days}
    for f in inst.fields:
        o = rec[f.fid]
        placed = False
        for d in inst.days:
            v = inst.visits(d, o)
            if v[0] < lo or v[-1] > hi:
                continue
            if all(load[x] < inst.capacity for x in set(v)):
                plan[f.fid] = (d, o)
                for x in set(v):
                    load[x] += 1
                placed = True
                break
        if not placed:
            plan[f.fid] = (hi, 0)
    return plan


# ----------------------------------------------------------------------------
# S2 / S3  STATIC PLANS
# ----------------------------------------------------------------------------

def plan_static_ev(inst, gen, n_mean=200, sharpness=1.0):
    """Single deterministic forecast: the climatological mean season."""
    mw = mean_weather(gen, n=n_mean)
    tbl = yield_table(inst, [mw], sharpness=sharpness)
    return solve_milp(inst, tbl), tbl


def plan_static_robust(inst, ensemble, ens_access, sharpness=1.0):
    """Expected yield over a weather ensemble, aware of rain slippage."""
    tbl = yield_table(inst, ensemble, accesses=ens_access, sharpness=sharpness)
    return solve_milp(inst, tbl), tbl


# ----------------------------------------------------------------------------
# S4  DYNAMIC (rolling horizon)
# ----------------------------------------------------------------------------

def make_replanner(inst, true_weather, true_access, gen, ensemble,
                   ens_access, every=3, forecast_days=0, freeze_lead=2,
                   sharpness=1.0, improve_tol=0.002, n_replan_ens=6,
                   verbose=False):
    """
    Build the callback that re-optimises the remaining schedule.

    NON-ANTICIPATIVITY. On day `today` the planner may use weather actually
    observed up to and including yesterday, plus `forecast_days` of perfect
    short-range forecast if that is being modelled. Everything beyond that
    comes from climatology. The true future is never consulted.
    """
    last = len(true_weather["gdd"]) - 1

    def conditional_ensemble(today):
        """
        Scenarios consistent with what is known on `today`: the weather that
        actually occurred up to the information cut-off, spliced onto each
        member of the climatological ensemble for the remainder of the season.
        The true future is never used beyond the cut-off.
        """
        known = min(today + forecast_days, last + 1)
        scen, accs = [], []
        for member in ensemble[:n_replan_ens]:
            w = {}
            for k in ("tmax", "tmin", "rain"):
                w[k] = np.concatenate(
                    [true_weather[k][:known], member[k][known:]])[:last + 1]
            w["gdd"] = daily_gdd(w["tmax"], w["tmin"])
            scen.append(w)
            accs.append(access_for(inst.fields, w))
        return scen, accs

    def replan(today, pending, done):
        if today % every != 0:
            return None
        if not pending:
            return None

        scen, accs = conditional_ensemble(today)
        # The yield table already prices in rain-induced slippage across the
        # conditional ensemble, so the MILP must NOT also hard-block days that
        # might be wet -- doing both counts the same risk twice and pushes the
        # schedule later than it needs to be. This matches how the static
        # robust plan is built.
        tbl = yield_table(inst, scen, accesses=accs, sharpness=sharpness)
        observed = {f.fid: true_access[f.fid] for f in inst.fields}

        # fields with the male already in the ground: only the female day is
        # still open, and it must keep the offset consistent with what was sown
        restrict = {}
        for fid in list(pending):
            rec = done.get(fid, {})
            if "m" in rec and "f" not in rec:
                restrict[fid] = ("m", rec["m"])
            elif "f" in rec and "m" not in rec:
                restrict[fid] = ("f", rec["f"])

        # Operations whose planned day is imminent are frozen: a crew cannot
        # be redirected at the last minute, and freezing is also what keeps
        # the rolling horizon from deferring a field indefinitely.
        frozen = set()
        for fid, (d, o) in pending.items():
            nxt = min(v for v in inst.visits(d, o)
                      if v >= today) if any(v >= today for v in inst.visits(d, o)) else None
            if nxt is not None and nxt - today <= freeze_lead:
                frozen.add(fid)

        allowed = set()
        for (fid, d, o) in inst.feasible_choices():
            if fid not in pending:
                continue
            if fid in frozen and (d, o) != pending[fid]:
                continue
            if fid in restrict:
                part, sown = restrict[fid]
                if part == "m" and d + o != sown:
                    continue
                if part == "f" and d != sown:
                    continue
            vd = [v for v in inst.visits(d, o)]
            if fid in restrict:
                vd = [v for v in vd if v != restrict[fid][1]]
            if vd and min(vd) < today:
                continue
            # a visit already known to be impossible (today is observed to be
            # untrafficable) is excluded; future days are left to the table
            if any(v <= today and not observed[fid][v] for v in vd):
                continue
            allowed.add((fid, d, o))

        if not allowed:
            return None

        sub = {k: tbl[k] for k in allowed}
        inst_sub = _RestrictedInstance(inst, allowed)
        new = solve_milp(inst_sub, sub, only_fields=list(pending), msg=False)
        if new is None:
            return None
        for fid in pending:
            if fid not in new:
                new[fid] = pending[fid]

        # Adopt the revision only if it is better than the plan already in
        # hand, scored on the same conditional ensemble. A rolling-horizon
        # policy that cannot tell the two apart should leave the schedule
        # alone: churning the plan has an operational cost and, as the
        # schedule-stability literature notes, can degrade performance.
        def score(p):
            return sum(tbl.get((fid,) + p[fid], 0.0) for fid in p)

        if score(new) <= score(pending) * (1.0 + improve_tol):
            return None

        if verbose:
            print(f"  day {today}: adopted revision, "
                  f"{score(pending):.1f} -> {score(new):.1f}")
        return new

    return replan


class _RestrictedInstance(Instance):
    """An Instance whose choice set has been narrowed for a replan."""

    def __init__(self, base, allowed):
        super().__init__(base.fields, base.w0, base.T, base.capacity, base.offsets)
        self._allowed = allowed

    def feasible_choices(self):
        return list(self._allowed)


# ----------------------------------------------------------------------------
# S5  WAIT-AND-SEE
# ----------------------------------------------------------------------------

def plan_wait_and_see(inst, weather, access, sharpness=1.0):
    """Perfect foreknowledge of this particular season."""
    tbl = yield_table(inst, [weather], accesses=[access], sharpness=sharpness)
    return solve_milp(inst, tbl, access=access), tbl


def _yield_table_dssat(inst, weathers, accesses=None, curve=PRIMARY_CURVE,
                       sharpness=1.0):
    """
    Expected yield per (field, female_day, offset) using the precomputed
    DSSAT CERES-Maize anthesis table.

    Structurally identical to the thermal-time version: for every scenario
    each visit is slipped to the next trafficable day, anthesis is looked up
    for both parents, and the resulting tasseling-silking interval is scored
    with the published yield response.

    The scoring is vectorised over scenarios and sowing days. This table is
    rebuilt at every replanning step of every strategy, so it dominates the
    cost of the whole study; scoring one offset at a time as an array rather
    than one (day, offset, scenario) triple at a time is what makes the
    replicated experiments affordable.
    """
    from .phenology import load as _load_pheno

    ph = _load_pheno()
    tbl = {}
    lo, hi = inst.w0, inst.w0 + inst.T - 1
    D = hi - lo + 1
    nS = len(weathers)
    last = len(weathers[0]["gdd"]) - 1
    gap = inst.male_split_gap if inst.split_male else 0

    preps = []
    for w in weathers:
        try:
            preps.append(ph.prepare(inst.fields[0], w))
        except KeyError:
            preps.append(None)
    live = [si for si in range(nS) if preps[si] is not None]
    if not live:
        return tbl

    days = np.arange(lo, hi + 1)

    for f in inst.fields:
        # anthesis[parent][scenario, sowing-day index], NaN where the day is
        # unreachable (no trafficable day left, or no anthesis simulated)
        A = np.full((nS, D), np.nan)        # male
        B = np.full((nS, D), np.nan)        # female
        for si in live:
            ok = accesses[si][f.fid] if accesses is not None else None
            for k, d in enumerate(days):
                dd = _slip(d, ok, last) if ok is not None else d
                if dd is None:
                    continue
                a = ph.anthesis(preps[si], f, dd, "m")
                b = ph.anthesis(preps[si], f, dd, "f")
                if np.isfinite(a):
                    A[si, k] = a
                if np.isfinite(b):
                    B[si, k] = b

        for o in inst.offsets:
            # window feasibility for every female day at once
            first = np.minimum(days, days + o)
            lastv = np.maximum(days, days + o + gap)
            ok_win = (first >= lo) & (lastv <= hi) & \
                     (days + o >= lo) & (days + o <= hi)
            if not ok_win.any():
                continue
            k = np.nonzero(ok_win)[0]
            tsi = A[:, k + o] - B[:, k]          # (nS, len(k))
            y = seed_yield(tsi, curve=curve, sharpness=sharpness)
            y = np.where(np.isfinite(tsi), y, 0.0)
            mean = y.sum(axis=0) / nS
            vals = mean * f.area_ha
            for j, kk in enumerate(k):
                tbl[(f.fid, int(days[kk]), o)] = float(vals[j])
    return tbl
