"""
THE scoring function. One implementation, used by the optimiser AND by
execution, so a plan can never be scored one way and executed another.

WHY THIS MODULE EXISTS
----------------------
The audit found THREE independent scoring paths in the repository:
`sim.scheduler.execute`, `sim.baselines.execute` and an inline scorer in
`sim.strategies`. A divergence between any two silently invalidates every
strategy comparison and produces results that look entirely reasonable. The
project has already been caught by one instance of this class -- the 10x bonus
meant the solver was optimising a different objective from the one reported.

Test T7.5 asserts that the objective coefficient equals the execution score.
Here that holds BY CONSTRUCTION rather than by agreement, because both callers
enter through `score_plan`.

HOW THE SECOND MALE SOWING IS CREDITED
--------------------------------------
Phase 2 P2-D13 said the overlap integral should REPLACE the negative branch of
r. Implementation showed that is not enough: the second sowing changes pollen
availability at every TSI, not only below zero, so replacing one branch cannot
credit it. But replacing r entirely would discard Marahatta's positive branch,
which is the only well-evidenced piece we have.

The resolution, a refinement of P2-D13 recorded in RESEARCH_DECISIONS.md:

    mode "split_ratio" (default when a second sowing exists)
        r(x) * [ O(x, gap) / O(x, single) ]

    The ratio is exactly 1.0 for a single sowing, so nothing is double
    counted; the pollen model carries ONLY the incremental effect of
    splitting, which is the single thing it is needed for. The ratio is free
    to be BELOW 1.0 -- split sowing reducing pollination is a permitted and
    expected outcome, not a defect (withdrawn SM1).

    mode "reflect"       r(x) alone. The second sowing costs capacity and earns
                         nothing. Conservative, and the legacy behaviour.
    mode "overlap_full"  O replaces r entirely. For E9 only.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import pollen as _pollen
from . import response as _response

MODES = ("split_ratio", "reflect", "overlap_full")


@dataclass(frozen=True)
class SowingPlan:
    """A field's sowing decision. `sow_m2` is None for single male sowing."""
    sow_f: int
    sow_m1: int
    sow_m2: Optional[int] = None

    @property
    def gap(self):
        return None if self.sow_m2 is None else int(self.sow_m2) - int(self.sow_m1)

    @property
    def visit_days(self):
        d = {int(self.sow_f), int(self.sow_m1)}
        if self.sow_m2 is not None:
            d.add(int(self.sow_m2))
        return tuple(sorted(d))

    def coords(self):
        return dict(sow_f=int(self.sow_f), sow_m1=int(self.sow_m1),
                    sow_m2=None if self.sow_m2 is None else int(self.sow_m2))


@dataclass(frozen=True)
class Score:
    r_mean: float             # mean relative yield over in-domain years
    area_r: float             # area_ha * r_mean -- the objective contribution
    n_in_domain: int
    n_years: int
    tsi_mean: float           # diagnostic only; never fed to r (P2-D16)
    mode: str
    credited_second_sowing: bool
    split_ratio: float        # 1.0 when no second sowing

    @property
    def out_of_domain(self):
        return self.n_years - self.n_in_domain

    def coords(self):
        return dict(mode=self.mode, r_mean=self.r_mean, area_r=self.area_r,
                    n_in_domain=self.n_in_domain, n_years=self.n_years,
                    out_of_domain=self.out_of_domain,
                    credited_second_sowing=self.credited_second_sowing,
                    split_ratio=self.split_ratio)


def score_plan(tsi_by_year, area_ha, plan, scenario=_response.PRIMARY_SCENARIO,
               mode=None, pp=_pollen.DEFAULT):
    """
    Score one field's sowing plan.

    tsi_by_year : callable (sow_m, sow_f) -> array of per-year TSI, nan where
                  the phenology table has no entry. Supplying this rather than a
                  provider keeps the scorer independent of which phenology path
                  is in use, so the optimiser and execution cannot diverge.
    area_ha     : field area, the objective weight
    plan        : SowingPlan, carrying BOTH male dates
    mode        : see module docstring. None picks "split_ratio" when a second
                  sowing exists and "reflect" otherwise.

    The ensemble averages r(TSI) per year, never TSI first (P2-D16): expected
    yield is E[r(TSI)], not r(E[TSI]). r is CONVEX on its positive branch
    (c = +0.012), so averaging TSI first overstates the penalty -- the opposite
    direction from the one Phase 2 claimed, corrected here.
    Years whose TSI falls outside the observed domain are EXCLUDED from the
    mean and COUNTED, never clipped.
    """
    gap = plan.gap
    if mode is None:
        mode = "split_ratio" if gap is not None else "reflect"
    if mode not in MODES:
        raise ValueError(f"unknown scoring mode {mode!r}; expected one of {MODES}")

    tsis = np.asarray(tsi_by_year(int(plan.sow_m1), int(plan.sow_f)), dtype=float)
    n_years = int(tsis.size)

    ratio = 1.0
    vals, n_ok = [], 0
    for t in tsis:
        if not np.isfinite(t):
            continue
        if mode == "overlap_full":
            v = _pollen.overlap_relative(float(t), gap, pp)
            ok = _response.TSI_LO - 1e-9 <= float(t) <= _response.TSI_HI + 1e-9
            if not ok:
                continue
        else:
            v, ok = _response.relative_yield_checked(float(t), scenario)
            if not ok:
                continue
            if mode == "split_ratio" and gap is not None:
                num = _pollen.overlap_fraction(float(t), gap, pp)
                den = _pollen.overlap_fraction(float(t), None, pp)
                ratio = (num / den) if (np.isfinite(den) and den > 0.0) else 1.0
                v = v * ratio
        vals.append(v)
        n_ok += 1

    r_mean = float(np.mean(vals)) if vals else float("nan")
    finite = tsis[np.isfinite(tsis)]
    return Score(
        r_mean=r_mean,
        area_r=(float(area_ha) * r_mean) if np.isfinite(r_mean) else 0.0,
        n_in_domain=n_ok,
        n_years=n_years,
        tsi_mean=float(finite.mean()) if finite.size else float("nan"),
        mode=mode,
        credited_second_sowing=(gap is not None and mode in ("split_ratio",
                                                            "overlap_full")),
        split_ratio=float(ratio),
    )


def dssat_tsi_fn(ph, field, asi0=5.0, delta_thr=0.0):
    """Adapter: per-year TSI from the DSSAT table for one field's parent pair."""
    def fn(sow_m, sow_f):
        return ph.tsi_by_year(field.cult_m, field.cult_f, sow_m, sow_f,
                              asi0=asi0, delta_thr=delta_thr)
    return fn


def constant_tsi_fn(values):
    """Deterministic fixture for tests: TSI = base + (sow_m - sow_f)."""
    arr = np.asarray(values, dtype=float)

    def fn(sow_m, sow_f):
        return arr + (int(sow_m) - int(sow_f))
    return fn
