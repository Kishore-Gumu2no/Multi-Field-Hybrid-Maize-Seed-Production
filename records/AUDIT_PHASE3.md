# Phase 3 Implementation Audit

Recorded before any modification. Every claim below was verified by direct
inspection of the repository at the commit named.

## Starting state

    branch        main
    HEAD          bde025b  Housekeeping before Target 1: canonical objective and environment record
    parents       5f73a18  Investigation additions (capacity models, six baselines, revised manuscript)
                  d57b168  Original project state as supplied
    working tree  clean
    python files  31 (7441 lines)

## Module map

| Module | Role | Touched in Phase 3 |
| --- | --- | --- |
| `sim/core.py` | GDD, trafficability, `YIELD_CURVES`, `seed_yield`, `_overlap_days`, `Field`, `make_farm` | read-only (new modules added beside it) |
| `sim/phenology.py` | `ThermalTimePhenology`, `DssatPhenology`, provider comparison | modified (step 2) |
| `sim/scheduler.py` | `Instance`, `yield_table`, `solve_milp`, `execute` | read-only; superseded by `sim/optimise.py` |
| `sim/baselines.py` | B1-B6 + a second capacity-aware `execute` | read-only in step 1-8 |
| `sim/capacity.py` | visits / operations / hours / hectares cost models | reused unchanged |
| `sim/strategies.py` | S1-S5 + a third scoring path | read-only |
| `sim/dssat.py` | experiment generation and `Summary.OUT` parsing | read-only |
| `run_real.py` | `RealField`, `pick_parent_pairs`, `build_real` | modified (step 1) |

## Claims from earlier reports, verified or corrected

| Claim | Source | Verdict against code |
| --- | --- | --- |
| `DssatPhenology.anthesis()` returns one `ADAT` for both parents | Target 0, 1A, Phase 1 | **FALSE.** `sim/phenology.py:72-73` selects `field.cult_m` / `field.cult_f`. Already corrected in the Phase 2 Supplement |
| `pick_parent_pairs` admits negative-gap pairs | Phase 2 Supplement | **TRUE.** `run_real.py:122` tests `1.5 <= abs(gap) <= 14` |
| `execute()` scores yield from the first male sowing only | Phase 1 C11 | **TRUE.** `sim/scheduler.py:287` and `sim/baselines.py:246` both use `rec["m"]`, with an explicit docstring rationale |
| `solve_milp` carries a 10x per-field bonus | Investigation Report | **TRUE.** `sim/scheduler.py:179` |
| `seed_yield` normalises on `Y(0)` | Phase 1 F15 | **TRUE.** `yield_at_optimum` returns `seed_yield(TSI_DOMAIN[0])` = `seed_yield(0.0)` = `a` |
| `TSI_DOMAIN = (0.0, 16.0)`, negative TSI governed by `_overlap_days` | Phase 1 C5 | **TRUE.** `sim/core.py:91`, `:157-158` |
| The ensemble averages TSI before applying the response | Phase 2 I4 risk | **FALSE, and already correct.** `sim/scheduler.py:131` averages `seed_yield(dt-ds)` per realisation, i.e. the factor. No change needed |
| `tau_m` targets male 75% tasseling, so the thermal path needs no ASI term | Phase 2 Supplement | **TRUE.** `sim/core.py:257-258` |

## Defects found in this audit that no earlier report names

| # | Finding |
| --- | --- |
| A1 | **Three independent scoring paths exist**, not two: `sim/scheduler.py:execute`, `sim/baselines.py:execute` and an inline scorer at `sim/strategies.py:444`. Any divergence between them silently invalidates strategy comparisons. This raises the priority of test T7.5 and T8.7 |
| A2 | `sim/scheduler.py:execute` enforces capacity as **field visits only**, while `sim/baselines.py:execute` is capacity-model aware. The two are therefore not interchangeable across capacity models |
| A3 | `make_farm` docstring claims female silking near day 70 and male tasseling near day 78 at same-day sowing, implying TSI(0) = +8. The actual `delta = U(25, 70)` GDD gives TSI(0) ~ +1.7 to +4.9 d. The docstring overstates the gap; the implemented values are inside the admissible range |
| A4 | `seed_yield` applies `YIELD_CEILING = 4.0` t/ha and an optional `sharpness` exponent anchored on `a = Y(0)`. Both interact with any change of normaliser and must not be inherited by the new relative response |

## Consequence for the implementation plan

The Phase 2 ten-step order stands. Because `sim/phenology.py` already supports
parent-specific cultivars, step 2 is a smaller change than specified: it adds the
ASI subtraction to the DSSAT provider only, and does not touch cultivar selection.

New modules are added beside the existing ones rather than rewriting them, so
historical results stay reproducible from the superseded code paths.
