# Research decisions

Append-only. A superseded decision is marked **SUPERSEDED** with what replaced
it; nothing is deleted. Labels: **VERIFIED**, *DERIVED*, **ASSUMED**,
**CONTESTED**, **UNKNOWN**.

## Phase 1 register (carried forward)

| # | Decision | Status |
| --- | --- | --- |
| P1-D1 | TSI = male anthesis - female silking | active, VERIFIED |
| P1-D2 | The ASI correction sits on the **male** parent | active, VERIFIED |
| P1-D3 | CSM-IXIM is closed and not revisited | active, VERIFIED |
| P1-D4 | The yield objective is **relative**, never absolute t/ha | active |
| P1-D5 | Normalise on the observed optimum, not `Y(0)` | active, *DERIVED* |
| P1-D6 | `winter_y2` dropped from the ensemble, reason recorded | active, VERIFIED |
| P1-D7 | `spring_y1` stays excluded, reason recorded, not silently repaired | active, SUPPORTED |
| P1-D8 | Marahatta and UAS remain separate evidence sources; no joint fit | active |
| P1-D9 | Marahatta governs TSI above +2; UAS governs sign and ordering below 0 | active |
| P1-D10 | Male-early penalty exceeds male-late at equal distance from zero | active, VERIFIED (sign) |
| P1-D11 | Search space constrained on **TSI**, derived per parent pair | active, *DERIVED* |
| P1-D12 | Schedules implying out-of-domain TSI are flagged, not silently scored | active |
| P1-D13 | Plant density is not modelled | active, VERIFIED |
| P1-D14 | `SHARE["f"] = 2/3` retained, externally justified by CIMMYT 4:2 | active, VERIFIED |
| P1-D15 | Split male sowing credited; the second sowing must affect the score | active |
| P1-D16 | Headline reported as a range across declared scenarios | active |
| P1-D17 | Same-day sowing neither preferred nor penalised by default | active |
| **C1 / BR1** | "Distinct parental coefficients before the ASI term" | **SUPERSEDED** by P2-D2. The phenology lookup is already parent-specific; the defect was pair admissibility |

## Phase 2 register

| # | Decision | Status |
| --- | --- | --- |
| P2-D1 | TSI depends on `P1`, `P2`, `PHINT` only; `P5` is post-flowering | active, VERIFIED |
| P2-D2 | Pair admissibility is `TSI(0) = Delta - ASI0` within `[0, +8]` d | active, *DERIVED* (`tau_max` ASSUMED) |
| P2-D3 | Farm stratification is on `TSI(0)`, not the absolute silking gap | active |
| P2-D4 | The ASI correction applies to the **DSSAT provider only** | active, VERIFIED |
| P2-D5 | Invariants SG1-SG3 enforced, SG1 at instance construction | active |
| P2-D6 | `ASI0` swept over {3, 5, 8} with `TSI(0)` held and `Delta` re-derived | active |
| P2-D7 | Coefficients come from Tier 3 synthetic surrogates, labelled everywhere | active, ASSUMED |
| P2-D8 | The claim is a scheduling claim over a declared pair population | active |
| P2-D9 | `r` normalised on the plateau edge; out-of-domain rejected not clipped | active |
| P2-D10 | `kappa > 1.4286` hard floor; levels {1.5, 2.0, 2.5}; never fitted | active, *DERIVED* |
| P2-D11 | Three plateau scenarios, none designated correct | active, CONTESTED |
| P2-D12 | 18-cell scenario grid; primary cell named | active |
| P2-D13 | `O` replaces the negative branch of `r`; never multiplies it | **REFINED** by P3-D1 |
| P2-D14 | Both male sowings enter the yield score | active |
| P2-D15 | Set-partitioning; **two-stage lexicographic objective** | **SUPERSEDED** by X11 / P3-D2: single-stage primary |
| P2-D16 | The ensemble averages `r(TSI)`, never TSI before `r` | active; **reason corrected** by P3-D3 |

## Phase 2 Supplement corrections (X1-X14)

All fourteen are mandatory and implemented; see `IMPLEMENTATION_LOG.md` for the
file-level mapping. The three that changed a default:

| # | Correction |
| --- | --- |
| X1 | **SM1 withdrawn.** Split sowing is not guaranteed to improve pollination. Replaced by correctness invariants SP1-SP4 and signed directional predictions SP5-SP6 |
| X6 | Pair admissibility evaluated **per weather year**, not on a grand mean |
| X10/X11 | **OBJ-A single-stage becomes primary**; OBJ-B and OBJ-C are labelled sensitivities |

## Phase 3 decisions (made during implementation)

| # | Decision | Reason | Status |
| --- | --- | --- | --- |
| **P3-D1** | The second male sowing is credited by the **split ratio** `O(x, gap) / O(x, single)` multiplying `r(x)`, not by `O` replacing one branch. Default mode `split_ratio` when a second sowing exists, `reflect` otherwise | Refines P2-D13. The second sowing changes pollen availability at **every** TSI, not only below zero, so replacing one branch cannot credit it; replacing `r` entirely would discard Marahatta's positive branch, the only well-evidenced piece. The ratio is exactly 1.0 for a single sowing, so nothing is double counted, and it is free to fall below 1.0 | active |
| **P3-D2** | Single-stage OBJ-A is the primary objective; OBJ-B and OBJ-C are switchable sensitivities, OBJ-B labelled a contractual planting obligation | Supplement X10/X11 | active |
| **P3-D3** | P2-D16 stands, but its stated reason was wrong: the published quadratic has `c = +0.012`, so `r` is **convex** on its positive branch. Averaging TSI before `r` therefore **overstates** the penalty, not understates it | Caught by test `test_ensemble_averages_the_response_not_the_tsi`. Phase 2 issue I4 and the Supplement both had the direction backwards | active, *DERIVED* |
| **P3-D4** | Candidates whose capacity cost falls on a day outside the declared window are discarded, and the count reported | Caught by test T8.1: without the guard, cost on an unconstrained day is ignored and a schedule that does not fit the window appears free | active |
| **P3-D5** | A relaxed per-year scope `rho < 1` is available and reported as a labelled variant, because `rho = 1.0` leaves only **6** admissible pairs of 56 | Phase 2 section 6.3 rule 4 already allowed it; implementation showed it is needed, since 6 pairs for a 40-field farm gives 6.67 repeats per pair | active |
| **P3-D6** | The pollen model is built at step 4 rather than step 10, because invariants SP1-SP6 cannot be tested without it. Step 10 narrows to *using* it as `r`'s negative branch (`mode="overlap_full"`) and running E9 | Phase 2's step order put the pollen model last, but experiment E3 and tests T7.2a-e both need it. Recorded as an inconsistency in the Phase 2 order | active |
| **P3-D8** | Per-year admissibility applies the **scoring domain** `[-3.75, +13.75]` per year with the tau range on the mean, matching Phase 2 section 6.3 rule 4. The stricter reading that applies the tau range per year is retained as `scope="years_tau"` | Found in the Phase 3 closure audit. The first implementation applied the tau range per year, which rejects a cross whose same-day TSI dips to -1 d in the coldest year of 21 -- an ordinary seed-production cross. It cut the population from 12 pairs to 6 and left only one pair in the hard [+5, +8] stratum that E4 needs | active |
| **P3-D7** | Legacy code paths (`sim.core.seed_yield`, `sim.scheduler.solve_milp`, both legacy `execute` functions) are left **untouched** so historical results stay reproducible. Nothing in the Phase 3 path calls them | Phase 3 engineering requirement: do not destroy the original research state | active |
