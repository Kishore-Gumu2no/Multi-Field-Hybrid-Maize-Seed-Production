"""
Farm construction, candidate generation and execution.

All three enter `sim.scoring.score_plan`, so the objective coefficient and the
execution score are the same number by construction (test T7.5). The audit
found three divergent scoring paths in the legacy code; this module replaces
them for the Phase 3 path and leaves them untouched for historical runs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import capacity as _cap
from . import pairs as _pairs
from . import response as _response
from . import scoring as _scoring
from .optimise import Candidate
from .runspec import Reliability


@dataclass
class FarmField:
    fid: int
    cult_m: str
    cult_f: str
    area_ha: float
    soil_drain_days: int
    pair: _pairs.PairInfo

    def coords(self):
        return dict(fid=self.fid, cult_m=self.cult_m, cult_f=self.cult_f,
                    area_ha=self.area_ha, tsi0_mean=self.pair.tsi0_mean,
                    slope=self.pair.slope,
                    single_coefficient=self.pair.single_coefficient)


def build_farm(ph, n_fields, farm_seed, sow_day_ref, asi0=_pairs.ASI0_DEFAULT,
               scope="years", rho=1.0, area_lo=1.5, area_hi=4.0):
    """
    A farm of `n_fields`, stratified on same-day TSI (P2-D3).

    Invariant SG1 is enforced on every field: a pair whose TSI(0) falls outside
    the admissible range raises rather than producing a schedule.
    """
    rng = np.random.default_rng(farm_seed)
    chosen = _pairs.stratified_pairs(ph, n_fields, rng, sow_day_ref,
                                     asi0=asi0, scope=scope, rho=rho)
    fields = []
    for i, p in enumerate(chosen):
        _pairs.check_admissible_delta(p.delta_mean, asi0)        # SG1
        fields.append(FarmField(
            fid=i, cult_m=p.cult_m, cult_f=p.cult_f,
            area_ha=float(rng.uniform(area_lo, area_hi)),
            soil_drain_days=int(rng.integers(1, 4)),
            pair=p))
    return fields


def _parts_by_day(plan):
    """day -> tuple of operations happening that day, for the capacity model."""
    out = {}
    out.setdefault(int(plan.sow_f), []).append("f")
    out.setdefault(int(plan.sow_m1), []).append("m")
    if plan.sow_m2 is not None:
        out.setdefault(int(plan.sow_m2), []).append("m2")
    return {d: tuple(v) for d, v in out.items()}


def plan_cost(model, field, plan):
    """Capacity cost of a plan, per day, under the active capacity model."""
    return {d: float(model.serve_cost(field, parts))
            for d, parts in _parts_by_day(plan).items()}


def build_candidates(ph, fields, days, model, scenario=_response.PRIMARY_SCENARIO,
                     asi0=_pairs.ASI0_DEFAULT, offsets=range(-4, 7),
                     split_male=True, split_gap=2, access=None, rho=1.0,
                     mode=None):
    """
    Enumerate admissible candidate schedules and score each one.

    A candidate enters only if ALL of:
      1. window containment  -- every sowing day lies in `days`
      2. trafficability      -- every sowing day is open for that field
      3. TSI domain          -- in [-3.75, 13.75] d for at least `rho` of
                                ensemble years; rho = 1.0 for primary runs
    Out-of-domain years are counted on the candidate, never clipped.
    """
    lo, hi = min(days), max(days)
    out, diag = [], dict(n_enumerated=0, n_window=0, n_access=0, n_domain=0)
    for f in fields:
        tsi_fn = _scoring.dssat_tsi_fn(ph, f, asi0=asi0,
                                       delta_thr=scenario.delta_thr)
        for d_f in days:
            for o in offsets:
                d_m1 = d_f + int(o)
                d_m2 = (d_m1 + int(split_gap)) if split_male else None
                plan = _scoring.SowingPlan(d_f, d_m1, d_m2)
                diag["n_enumerated"] += 1
                if any(not (lo <= d <= hi) for d in plan.visit_days):
                    diag["n_window"] += 1
                    continue
                if access is not None and not all(access[f.fid][d]
                                                  for d in plan.visit_days):
                    diag["n_access"] += 1
                    continue
                tsis = np.asarray(tsi_fn(d_m1, d_f), dtype=float)
                finite = tsis[np.isfinite(tsis)]
                if finite.size == 0:
                    diag["n_domain"] += 1
                    continue
                inside = ((finite >= _response.TSI_LO - 1e-9)
                          & (finite <= _response.TSI_HI + 1e-9)).mean()
                if inside < rho - 1e-9:
                    diag["n_domain"] += 1
                    continue
                s = _scoring.score_plan(tsi_fn, f.area_ha, plan, scenario,
                                        mode=mode)
                if not np.isfinite(s.r_mean):
                    diag["n_domain"] += 1
                    continue
                out.append(Candidate(
                    fid=f.fid, plan=plan, area_ha=f.area_ha, area_r=s.area_r,
                    cost=plan_cost(model, f, plan), tsi_mean=s.tsi_mean,
                    n_out_of_domain=s.out_of_domain,
                    label=f"f{f.fid}_d{d_f}_o{o:+d}"))
    return out, diag


def execute(ph, fields, solution, scenario=_response.PRIMARY_SCENARIO,
            asi0=_pairs.ASI0_DEFAULT, mode=None):
    """
    Score the selected schedules through `score_plan` -- the SAME function the
    objective coefficients came from.

    T7.5 therefore holds by construction. Out-of-domain field-years are flagged
    and counted here, at execution, because a plan in-domain on the ensemble
    mean can realise outside it in a given year.
    """
    by_fid = {f.fid: f for f in fields}
    rows, total, n_fy, n_ood = [], 0.0, 0, 0
    for c in solution.chosen:
        f = by_fid[c.fid]
        tsi_fn = _scoring.dssat_tsi_fn(ph, f, asi0=asi0,
                                       delta_thr=scenario.delta_thr)
        s = _scoring.score_plan(tsi_fn, f.area_ha, c.plan, scenario, mode=mode)
        total += s.area_r
        n_fy += s.n_years
        n_ood += s.out_of_domain
        rows.append(dict(**f.coords(), **c.plan.coords(), **s.coords(),
                         label=c.label, merged=c.merged,
                         n_visit_days=c.n_visit_days))
    return dict(total_area_r=total, detail=rows,
                n_sown=len(solution.chosen),
                ha_sown=float(sum(by_fid[c.fid].area_ha
                                  for c in solution.chosen)),
                reliability=Reliability(n_fy, n_ood))
