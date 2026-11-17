# Canonical optimisation objective

Status: **decided for going-forward work; both implementations preserved.**
Recorded 2026-11-17 as housekeeping before Target 1. No solver code was
changed in making this decision.

## The discrepancy

Two optimisation objectives exist in this repository and produce different
schedules from identical inputs.

| Path | Objective | Used by |
| --- | --- | --- |
| `sim/scheduler.py::solve_milp` | expected yield **plus a bonus of 10x the largest field yield** for every assigned field | S2-S5; every result in `paper/paper_v2_filled.tex` and in `results/R*.csv` |
| `sim/baselines.py::plan_joint` | expected yield alone | B6; `results/E2_raw.csv` |

### Why the bonus was there

To make the solver prefer sowing a field over leaving it unsown.

### Why it is wrong

An unsown field already contributes zero to expected yield, so yield
maximisation already prefers sowing wherever a field can be served. The
bonus is therefore redundant. It is also harmful: it inflates the objective
by roughly an order of magnitude per field, which loosens the linear
relaxation and slows branch-and-bound.

Nor does the bonus achieve what it was meant to achieve. To guarantee
lexicographic behaviour -- maximise fields sown first, then yield -- the
bonus must exceed the total yield reachable by all other fields, i.e. be of
order N x max_yield. At 10x max_yield it is smaller than that for any farm
with more than ten fields, so for the farm sizes studied here it neither
guarantees the lexicographic ordering nor leaves the relaxation tight.

### Measured effect

Same solver, same instance (N = 70, hours capacity model), same 45 s limit:

| Objective | Fields sown | Plan expected yield |
| --- | --- | --- |
| with 10x bonus | 34 | 283.0 t |
| expected yield alone | **38** | **347.3 t** |

Removing the bonus produced a strictly better solution within the same
budget. Some optima reported in the current manuscript are therefore not
optimal.

## Decision

**Canonical: expected yield alone (`plan_joint`).**

`solve_milp` is **retained unchanged** for comparison and for reproducing the
existing manuscript. It must not be deleted, and the historical results it
produced remain valid as a record of what that formulation gives.

## Consequence for existing results

Numbers from the two paths **are not comparable and must never appear in the
same table**. In particular:

- `results/R3_*`, `R4_*`, `R5_*`, `R6_*`, `R7_*`, `R8_*` and every figure
  and table in the current manuscript come from `solve_milp`.
- `results/E2_raw.csv` comes from `plan_joint` and is additionally
  incomplete (visits 5/5 farms, operations 5/5, hours 3/5, hectares 0/5).

Any comparison between them requires re-running one side.

## Open question deferred to Target 4

Whether the canonical objective should remain pure expected yield, or become
expected yield subject to a biological feasibility constraint, is a Target 4
decision and is informed by the Target 1 design.
