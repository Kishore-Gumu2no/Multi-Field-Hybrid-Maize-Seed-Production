# Implementation log -- Phase 3

Starting state: branch `main`, HEAD `bde025b`, working tree clean.
Audit recorded in `records/AUDIT_PHASE3.md` **before** any modification.

## Commits

| Commit | Step | Content |
| --- | --- | --- |
| `4a02e7f` | 1 | Per-year pair admissibility on `TSI(0)`, stratification on `TSI(0)`, slope and population diagnostics, invariant SG1, audit record |
| `00c25e3` | 2 | Provider-specific ASI correction; legacy `anthesis()` unchanged |
| `db4a5b3` | 3 | Relative response, 18-cell scenario grid, plateau scenarios, kappa floor, domain rejection |
| `24855d8` | 4 | Density-weighted pollen model, single shared scorer, SM1 withdrawn, convexity correction |
| `16b68a5` | 5 | Set-partitioning optimiser, OBJ-A primary, window guard |
| `ceae78b` | 6-7 | Out-of-domain flagging, reliability gate, run coordinates, integration tests, pollen memoisation |

## The 14 Supplement corrections, mapped to code

| # | Correction | Where implemented | Test |
| --- | --- | --- | --- |
| X1 | SM1 withdrawn; SP1-SP4 correctness, SP5-SP6 signed predictions | `sim/pollen.py` (`split_advantage` is signed and unconstrained) | `T7.1`, `T7.2a-e` |
| X2 | SM2 renumbered SP2 | `tests/test_t07_split.py` | `T7.1` |
| X3 | E3 re-specified across strategies S1-S5 | `RESEARCH_STATUS.md`, Phase 4 scope | - |
| X4 | Split sowing is a **hedge**, not an expected-value gain | `sim/pollen.py` module docstring | `T7.2d`, `T7.2e` |
| X5 | Name what the surrogates are | `ASSUMPTIONS_AND_LIMITATIONS.md`, `sim/pairs.CULTIVAR_COEFFS` | `T10.4` |
| X6 | Admissibility **per weather year** | `sim/pairs.admissible_pairs(scope="years")` | `T2.4b` |
| X7 | Slope diagnostic reported, never gating | `sim/pairs.slope_diagnostic`, `population_report` | `T10.2` |
| X8 | Limitations L8-L11 added | `ASSUMPTIONS_AND_LIMITATIONS.md` | `T10.1`, `T10.4`, `T10.5` |
| X9 | Phenology is **not validated** against field data | `ASSUMPTIONS_AND_LIMITATIONS.md` | - |
| X10 | Single-stage objective becomes primary | `sim/optimise.PRIMARY_OBJECTIVE = "OBJ-A"` | `T11.1`, `T11.4` |
| X11 | P2-D15 amended | `RESEARCH_DECISIONS.md` (marked SUPERSEDED) | `T11.2` |
| X12 | Issue I12 added: no expected outcome as a constraint | `RESEARCH_DECISIONS.md`, and enforced by SP6 and T11.6 | `T7.2e`, `T11.6` |
| X13 | Risk R12 added: the four defects share one signature | `records/AUDIT_PHASE3.md`, `RESEARCH_DECISIONS.md` | - |
| X14 | Step 5 builds OBJ-A with B/C switchable | `sim/optimise.solve(objective=...)` | `T11.5` |

## Defects found by the tests during implementation

| # | Found by | Defect | Fix |
| --- | --- | --- | --- |
| D1 | `T8.1` | Candidate capacity costs on days **outside** the declared window were ignored, so a schedule that does not fit the window appeared free | `sim/optimise._drop_outside_window`, count reported |
| D2 | `test_ensemble_averages_the_response_not_the_tsi` | Phase 2 issue I4 and the Supplement both stated the Jensen bias **backwards**: the quadratic has `c = +0.012`, so `r` is convex and averaging TSI first **overstates** the penalty | Claim corrected in `sim/response.py`, `sim/scoring.py` and P3-D3. The requirement itself is unchanged |
| D3 | integration timeout | The pollen overlap integral was recomputed per candidate per weather year -- an 8000-point quadrature, 44k times | Memoised on `(TSI, gap, params)`; `PollenParams` is frozen and hashable so a swept set never reads another's cache |
| D4 | `T11` fixture review | My first counterexample fixture was internally inconsistent: a same-day plan carrying a two-visit cost | Rebuilt over two days with capacity 1/day |
| D5 | `T1.1` | The test asserted an identity across mismatched aggregation levels (grand-mean `Delta` against a per-day mean) | Test corrected, not the code; reason recorded in the test docstring |
| D6 | closure audit | **Per-year admissibility applied the tau range instead of the scoring domain**, which is stricter than Phase 2 section 6.3 rule 4 specifies. Cut the population from 12 pairs to 6 and left one pair in the hard stratum E4 needs | `scope="years"` corrected; stricter reading kept as `scope="years_tau"` (P3-D8); tests T2.4b, T2.4c, T2.4d |
| D7 | closure audit | The six-field smoke instance could not distinguish OBJ-A / OBJ-B / OBJ-C, because at 56% capacity utilisation the coverage constraint is slack | Three discriminating fixtures added in `tests/test_t12_objective_fixtures.py` |

## Scope held

No research experiment was run. No performance claim is made. The manuscript
was not touched. No field validation is claimed. DSSAT is absent from this
environment and no DSSAT run is claimed; the phenology table is frozen and its
content hash is recorded with every result row.

## Running the suite

    ./run_tests.sh              # all 110 tests, one line each
    ./run_tests.sh blocking     # the starred/blocking subset only
    ./run_tests.sh T7           # one group

Test groups: T1-T11 plus T12 (objective-discriminating fixtures).

Per-test evidence: `records/TEST_RESULTS.txt`. Diagnostics: `records/DIAGNOSTICS.txt`.
