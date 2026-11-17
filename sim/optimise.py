"""
Set-partitioning scheduler with three switchable objective formulations.

WHY SET PARTITIONING
--------------------
Every non-linearity in the problem -- phenology lookup, TSI, plateau,
kappa-reflection, split-ratio, out-of-domain rejection -- is a property of a
candidate schedule, not of the decision variables. Enumerating candidates and
selecting one per field evaluates each non-linearity exactly once and enters it
as a CONSTANT coefficient, so the programme stays a pure linear binary problem
and no biology has to be linearised. The yield model can then change without
touching the solver.

THE OBJECTIVES  (Supplement X10, X11)
-------------------------------------
OBJ-A  PRIMARY. Single stage:  max sum_i sum_s A_i * r_is * z_is
       Economically coherent: a field left unsown already contributes zero, so
       the objective values coverage correctly without being told to.

OBJ-B  SENSITIVITY, representing a CONTRACTUAL PLANTING OBLIGATION. Two stages:
       maximise the COUNT of fields sown, then yield subject to that count.
       Three undesirable preferences, recorded here so nobody re-adopts it by
       accident: it is area-blind in stage 1; it has no justification in the
       problem statement; and -- decisively -- it systematically prefers MERGED
       schedules, because merging frees a visit and a freed visit raises
       coverage. That is a thumb on the scale in favour of the very mechanism
       this project measures. DERIVED: on the Supplement's counterexample it
       gives up 12.5% of the objective to sow one extra 1.5 ha field, and the
       schedule it picks to free that capacity is the co-sown one.

OBJ-C  SENSITIVITY. As OBJ-B but stage 1 maximises HECTARES sown, which fixes
       the area-blindness. It does NOT by itself remove the merging preference.

THE 10x BONUS IS GONE
---------------------
`sim.scheduler.solve_milp` added `10.0 * max(tbl.values())` per assigned field
to make coverage dominate. Established defective: the constant must be of order
|I| * max(r) to order lexicographically, so at 10x it never worked above about
ten fields, and removing it changed the result from 34 fields / 283.0 t to
38 fields / 347.3 t at N = 70. That function is preserved unchanged for
historical reproducibility; nothing here uses it, and no constant is tuned.
"""

from dataclasses import dataclass, field as _dcfield
from typing import Dict, List, Optional

import numpy as np
import pulp

OBJECTIVES = ("OBJ-A", "OBJ-B", "OBJ-C")
PRIMARY_OBJECTIVE = "OBJ-A"

DEFAULT_TIME_LIMIT = 60
DEFAULT_GAP_REL = 0.002
REQUIRED_GAP = 0.005          # primary results must be optimal or within this


@dataclass(frozen=True)
class Candidate:
    """One schedule a field could take. All biology is already baked in."""
    fid: int
    plan: object                      # scoring.SowingPlan
    area_ha: float
    area_r: float                     # A_i * mean r(TSI) -- objective coefficient
    cost: Dict[int, float]            # day -> capacity cost under the active model
    tsi_mean: float
    n_out_of_domain: int
    label: str = ""

    @property
    def merged(self):
        """True when a male sowing shares a day with the female sowing."""
        p = self.plan
        return int(p.sow_m1) == int(p.sow_f) or (
            p.sow_m2 is not None and int(p.sow_m2) == int(p.sow_f))

    @property
    def n_visit_days(self):
        return len(self.plan.visit_days)


@dataclass
class Solution:
    objective_name: str
    objective: float
    chosen: List[Candidate]
    status: str
    gap: Optional[float]
    stage1_value: Optional[float] = None
    n_candidates: int = 0
    n_dropped_outside_window: int = 0

    @property
    def n_sown(self):
        return len(self.chosen)

    @property
    def ha_sown(self):
        return float(sum(c.area_ha for c in self.chosen))

    @property
    def merged_fraction(self):
        return (sum(1 for c in self.chosen if c.merged) / len(self.chosen)
                if self.chosen else 0.0)

    @property
    def out_of_domain_fields(self):
        return sum(1 for c in self.chosen if c.n_out_of_domain > 0)

    def coords(self):
        """Scenario coordinates for the result row."""
        return dict(objective=self.objective_name, objective_value=self.objective,
                    n_sown=self.n_sown, ha_sown=self.ha_sown,
                    merged_fraction=self.merged_fraction,
                    status=self.status, gap=self.gap,
                    n_candidates=self.n_candidates,
                    n_dropped_outside_window=self.n_dropped_outside_window,
                    out_of_domain_fields=self.out_of_domain_fields)


def _drop_outside_window(cands, days):
    """
    Discard candidates whose capacity cost falls on a day outside the declared
    window (section 6.3 rule 1, window containment).

    Without this guard a candidate's cost on an unconstrained day is silently
    ignored, so a schedule that does not fit the window appears free. Caught by
    test T8.1.
    """
    ok = set(int(d) for d in days)
    keep = [c for c in cands if all(int(d) in ok for d in c.cost)]
    return keep, len(cands) - len(keep)


def _build(cands, capacity, days, locked_load=None):
    prob = pulp.LpProblem("anthesis_setpart", pulp.LpMaximize)
    z = {k: pulp.LpVariable(f"z_{k}", cat="Binary") for k in range(len(cands))}

    by_field = {}
    for k, c in enumerate(cands):
        by_field.setdefault(c.fid, []).append(k)
    for ks in by_field.values():
        prob += pulp.lpSum(z[k] for k in ks) <= 1

    load = dict(locked_load or {})
    for d in days:
        terms = [c.cost[d] * z[k] for k, c in enumerate(cands) if d in c.cost]
        base = float(load.get(d, 0.0))
        if terms or base:
            prob += pulp.lpSum(terms) + base <= capacity
    return prob, z


def _solve_once(prob, z, cands, sense_expr, time_limit, gap_rel, msg=False):
    prob += sense_expr
    prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit,
                                 gapRel=gap_rel, threads=1))
    status = pulp.LpStatus[prob.status]
    chosen = [cands[k] for k, v in z.items()
              if v.value() is not None and v.value() > 0.5]
    try:
        gap = None if status != "Optimal" else 0.0
    except Exception:                                   # pragma: no cover
        gap = None
    return status, chosen, gap


def solve(candidates, capacity, days, objective=PRIMARY_OBJECTIVE,
          locked_load=None, time_limit=DEFAULT_TIME_LIMIT,
          gap_rel=DEFAULT_GAP_REL, msg=False):
    """
    Select at most one candidate per field subject to daily capacity.

    Returns a `Solution` carrying the objective value, the chosen candidates,
    solver status, gap, and -- for the two-stage variants -- the stage-1
    optimum, so the lexicographic preference is auditable rather than implicit.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected {OBJECTIVES}")
    cands, n_dropped = _drop_outside_window(list(candidates), days)
    if not cands:
        return Solution(objective, 0.0, [], "NoCandidates", None, None, 0)

    yield_expr = lambda z: pulp.lpSum(c.area_r * z[k] for k, c in enumerate(cands))

    stage1 = None
    if objective in ("OBJ-B", "OBJ-C"):
        prob, z = _build(cands, capacity, days, locked_load)
        if objective == "OBJ-B":
            cover = pulp.lpSum(z[k] for k in range(len(cands)))
        else:
            cover = pulp.lpSum(c.area_ha * z[k] for k, c in enumerate(cands))
        st, ch, _ = _solve_once(prob, z, cands, cover, time_limit, gap_rel, msg)
        if st not in ("Optimal", "Not Solved"):
            return Solution(objective, 0.0, [], st, None, None, len(cands))
        stage1 = (float(len(ch)) if objective == "OBJ-B"
                  else float(sum(c.area_ha for c in ch)))

    prob, z = _build(cands, capacity, days, locked_load)
    if stage1 is not None:
        if objective == "OBJ-B":
            prob += pulp.lpSum(z[k] for k in range(len(cands))) >= stage1 - 1e-6
        else:
            prob += (pulp.lpSum(c.area_ha * z[k] for k, c in enumerate(cands))
                     >= stage1 - 1e-6)
    status, chosen, gap = _solve_once(prob, z, cands, yield_expr(z),
                                      time_limit, gap_rel, msg)
    total = float(sum(c.area_r for c in chosen))
    sol = Solution(objective, total, chosen, status, gap, stage1, len(cands))
    sol.n_dropped_outside_window = n_dropped
    return sol
