# Research status

As of Phase 3 completion. No research experiment has been run and no
performance claim exists.

## Phase ledger

| Phase | Scope | State |
| --- | --- | --- |
| Target 0 | Repository and research audit | complete |
| Target 1 | Biological feasibility design | complete |
| Target 1A | Biological model verification and transferability | complete |
| Target 1B | Empirical yield-response verification (+ section 11 addendum) | complete |
| Phase 1 | Scientific verification and consolidation | complete |
| Phase 2 | Complete model redesign specification | complete |
| Phase 2 Supplement | Three challenged assumptions (X1-X14) | complete |
| **Phase 3** | **Implementation and verification** | **complete, awaiting approval** |
| Phase 4 | Research experiments E1-E9 | not started |
| Phase 5 | Manuscript revision | not started |

## Implementation state

| Step | Work | State | Gate |
| --- | --- | --- | --- |
| 1 | Pair admissibility and population generation | done | T2, T10 pass |
| 2 | ASI correction, DSSAT provider only | done | T1 passes |
| 3 | Relative response, plateau scenarios, kappa-reflection, domain rejection | done | T3, T4, T5 pass |
| 4 | Split-male interface and yield scoring | done | T7 passes, SP6 blocking test passes |
| 5 | Set-partitioning optimiser, OBJ-A primary | done | T8, T11 pass |
| 6 | Out-of-domain flagging and reliability gate | done | T6 passes |
| 7 | Scenario-grid coordinates on every result | done | T9 passes |
| 8 | Full T1-T11 suite | done | 110 of 110 pass |
| 9 | Experiments | **not started, by instruction** | - |
| 10 | Density-weighted overlap as `r`'s negative branch, and E9 | model built, reachable via `mode="overlap_full"`; E9 is a Phase 4 experiment | - |

## New modules

| Module | Role |
| --- | --- |
| `sim/pairs.py` | Pair admissibility, stratification on TSI(0), slope and population diagnostics |
| `sim/response.py` | Relative yield, scenario grid, plateau scenarios, kappa floor, domain rejection |
| `sim/pollen.py` | Density-weighted pollen-silk overlap with both male sowings |
| `sim/scoring.py` | **The** scoring function, entered by the optimiser and by execution |
| `sim/optimise.py` | Set-partitioning solver, OBJ-A / OBJ-B / OBJ-C |
| `sim/build.py` | Farm construction, candidate generation, execution |
| `sim/runspec.py` | Run coordinates, reliability gate, provenance hashes |

`sim/phenology.py` was modified (provider-specific ASI). Every other existing
module is untouched.

## Superseded, retained for reproducibility

| Path | Superseded by | Why retained |
| --- | --- | --- |
| `sim.core.seed_yield` | `sim.response.relative_yield` | Historical results |
| `sim.scheduler.solve_milp` | `sim.optimise.solve` | Historical results; carries the 10x bonus |
| `sim.scheduler.execute` | `sim.build.execute` | Historical results; scores the first male sowing only |
| `sim.baselines.execute` | `sim.build.execute` | Historical results |
| `run_real.pick_parent_pairs` | `sim.pairs.stratified_pairs` | Historical results; admits negative-Delta pairs |

Results produced by those paths are **superseded** and must be labelled so;
they are not deleted.

## Blocking items for Phase 4

None in the implementation. Three items are blocking for **submission**, not
for experiments: the manuscript's false statement that flowering is simulated
without water or nitrogen limitation (C8); the unverified published Marahatta
coefficients; and the narrowing of the paper's claim required by the synthetic
cultivars.
