# Final audit: engine provenance, corrections, and disposition

Branch `iot-sensing`. Every claim below was checked against its source result
file, and every re-run was executed on the corrected Family C engine.

## 1. How provenance was established

`records/trace_results.py` parses every top-level runner with the `ast` module,
resolves its `sim.*` imports, and maps each result file to the engine that
wrote it. Family A = `core, scheduler, strategies, baselines`. Family C =
`build, optimise, pairs, response, runspec, scoring, capacity, pollen,
phenology, compare, sensing, soilthermal`.

Result: **every legacy result file (R1--R8, E1--E6, V1) was written by a
Family A or MIXED runner.** Only E7/E8 (the IoT work) and the new
`results_fc_main/` were produced by Family C. `run_multifarm.py` is Family A
transitively, through `from run_experiments import ...`.

## 2. What Family A got wrong

**Family A has since been deleted from the repository** (commit following this
audit), together with its thirteen runners and its result files. Only
`daily_gdd`, the cumulative-GDD helpers and the `Field` container were kept,
moved to `sim/gdd.py`, because the external validation and the thermal-time
phenology provider use them and they touch none of the defects below. The
defects are quoted here so the claim stays checkable without the code:

* the bonus: `scheduler.solve_milp` added `10.0 * max(tbl.values())` to the
  objective for every assigned field, so the solver maximised
  yield + 10*max_y*(fields sown), not yield;
* the normaliser: `core.seed_yield` divided by `Y(0)` computed through the
  rectangular pollen/silk overlap that `sim/response.py` records as
  falsified, its errors at the two observed negative-TSI points having
  opposite signs (-0.159 at -3.75 d, +0.093 at -1.75 d);
* first-male-only scoring, ignoring the second male sowing;
* three divergent scoring paths: `scheduler.execute`, `baselines.execute`
  and an inline scorer in `strategies`;
* the ensemble mean taken over TSI rather than over r(TSI).

The original wording follows.

### What Family A gets wrong

`sim/compare.py` documents the defects it excludes by construction:

1. a 10x per-field bonus in the MILP objective (`scheduler.py` lines 179--180, since deleted),
   so the solver optimised a function different from the one reported;
2. `sim.core.seed_yield`'s normaliser, built on the rectangular pollen/silk
   overlap that `sim/response.py` records as FALSIFIED (errors of opposite
   sign at the two observed negative-TSI points);
3. first-male-only scoring, ignoring the second male sowing;
4. three divergent scoring paths (`scheduler.execute`, `baselines.execute`,
   an inline scorer in `strategies`);
5. the ensemble mean taken over TSI rather than over r(TSI). r is convex on
   its positive branch (c = +0.012), so by Jensen this overstates the penalty.

In Family C every planner selects from ONE candidate set and execution
re-scores through the SAME `scoring.score_plan`. Measured in the re-run:
`max |objective - execution| = 4.3e-14` across all 165 cells.

## 3. Re-run on the corrected engine

`run_familyc_main.py`, design mirroring legacy R6: visits capacity model,
K = 2 visits/day, N in {10,20,30,35,37,40,43,45,48,55,70}, 5 farm seeds,
the full 21-season ensemble, Rabi window DOY 274--334, OBJ-A.

Planner map: `S1_agronomic -> B1_sequential`, `S1_greedy -> B5_greedy`,
`S3_static_robust -> B6_joint`. All 165 cells terminated `Optimal`; the
largest measured LP gap was 0.0032, inside the 0.005 acceptance rule; no
cell is a lower bound; 0 out-of-domain field-years in 89,418.

### A weaker baseline than the paper implied

B1_sequential --- the paper's "fields taken in order" --- uses only about
half the available visit-slots (50.8% at N = 40, 49.2% at N = 70). Three
other service orderings over the SAME recommended offsets do better, and
`B3_largest` is the strongest once capacity binds. Reporting a gain against
B1 alone would overstate the contribution, so every headline is given twice:
against B1, and against the best of the four orderings.

Two baseline families appear in this project and they must not be confused:

* **best of four** --- the four service orders over the recommended offset
  (arrival, earliest-available, largest-area, hardest-cross).
* **best of five** --- those four plus the offset-relaxing rule B1R, which
  keeps arrival order but lets each field take its best feasible offset.

The five-rule family is the stronger baseline and is what the manuscript
claims against. An earlier draft of this audit quoted +24.45% pooled; that
figure is the **four**-rule family, computed before B1R was added to the
grid, and is superseded. Both families give the same +35.20% at N = 70,
because B3_largest wins there and B1R does not.

| quantity | vs B1_sequential | vs best of four | vs best of five |
|---|---|---|---|
| optimised assignment, pooled | +42.47% [+36.98, +47.79] | +24.45% | **+20.84% [+18.48, +23.08]** |
| optimised assignment, N = 70 | +73.31% | +35.20% | **+35.20%** |
| greedy, pooled | +6.57% | -6.91% | **-9.60%** |

95% intervals are a 10,000-sample bootstrap over the 55 (N, seed) cells.
Tables `tab:curves` and `tab:nomerge` report against the **four**-rule
family, because those runs predate B1R; both carry a footnote saying so.

### Saturation

Fields actually sown, averaged over seeds: B1 plateaus at 20--24 from N = 30
onward and never exceeds 24; the optimiser climbs from 30 to 41.8. The
legacy claim of saturation "at about 39 fields" is not reproduced --- 39 is
close to the number the OPTIMISER reaches, not the number practice reaches.

### Greedy

Refined, not contradicted. Under slack (N = 10, 20) greedy matches the exact
solver to within 0.3 points. Once capacity binds it falls below the best
ordering at every N from 35 up (-6.4% to -11.3%), while staying above B1.

## 4. Replanning: NOT REPRODUCED

On the corrected engine, `compare.plan_dynamic` against the static solve over
9 cells (N = 30/45/70 x 3 seeds) gives a mean of **+0.004%** (max +0.026%),
which is solver tolerance.

This is not a contradiction of the legacy +2.26%; it is a different
experiment. `sim/strategies.make_replanner` splices weather actually observed
up to the information cut-off onto each climatological ensemble member, so
the legacy figure is the INFORMATION-side value of adaptivity.
`compare.plan_dynamic` re-solves over the same ensemble candidate set and so
measures only the SCHEDULING-side value, which is zero by construction
because the static solve is already globally optimal over that set.

Reproducing the information-side result on the corrected engine needs
conditional per-year candidate sets. `sim/compare.py` names that as a Phase 4
experimental-design question. It was not undertaken here.

Disposition: the +2.26% replanning gain and the curvature ladder that depends
on it (+0.49% flat to +9.14% sharp, legacy R7) are **withdrawn from the
results**. They are not presented as corrected results, and the superseded
figures are not carried into the abstract, tables or figures.

## 5. CIMMYT validation: RETAINED, rescoped

`validate_cimmyt.py` re-run: reproduces to 0.0e+00 on every reported
statistic. It touches none of the five defects --- it uses only
`sim.core.daily_gdd`, which is arithmetic.

It does, however, validate the thermal-time approximation, not the DSSAT
table that produces the scheduling results, and CIMMYT's published GDD
formula is the same base-10/cap-30 arithmetic, so the agreement is a
consistency check rather than independent biological validation. The two
rank statistics are not significant: spans p = 0.094, rank p = 0.111. The
manuscript now says so.

## 6. Equations against implementation

| manuscript | implementation | verdict |
|---|---|---|
| TSI = phi(male) - phi(female), positive = male late | `ADAT_m - ASI0 - ADAT_f + delta_thr`, both providers | consistent |
| g(t), base 10, cap 30, air temperature | results come from the DSSAT CERES-Maize table, not this equation | **corrected in the manuscript** |
| Y1(x) = 3.5918 - 0.3379x + 0.012x^2 | `response.CURVES["winter_y1"]`, identical | consistent |
| positive branch clipped at x = 14.1 | domain [-3.75, 13.75]; out-of-domain years EXCLUDED and counted, never clipped | **corrected** |
| negative branch = rectangular overlap, Dm = 7, Df = 6 | kappa-reflection, kappa = 1.5, plateau PL2 = (0, 2). The overlap form is retained in `response.rectangular_overlap_relative` and labelled FALSIFIED | **corrected** |
| CERES soil pathway | `sim/soilthermal.py`, pinned by 15 tests in `tests/test_t14_soilthermal.py` | consistent |
| non-anticipativity of sensing | `cfg.obs_end_doy = 273 < 274`, pinned in `tests/test_t15_sensing.py` | consistent |

## 7. Robustness of the re-run

**Scenario grid.** The assignment experiment repeated in all eighteen
declared cells (2 curves x 3 plateaus x 3 kappa) at N = 70, three farms
each. Gain over the best of four service orders runs from +28.6% to +44.9%,
mean +38.7%, **positive in all 54**. All 54 solves optimal, max gap 0.00043.

**Merging.** Removing every merged candidate from the set given to EVERY
planner costs the exact plan 13 of its 42 fields at N = 70 and cuts the gain
over the best ordering from +37.4% to +23.3% (at N = 40, +26.9% to +20.3%).
Merging therefore contributes materially and an earlier draft of this audit
understated it; it does not account for the result.

## 8. Verification

`records/verify_manuscript.py` recomputes every headline, confidence
interval, table cell and solver diagnostic directly from the raw result rows
and compares each against the value printed in the manuscript. **23 of 23
checks pass.** One rounding error was caught this way and corrected
(+20.4% printed where the rows give +20.348%).

`records/recompute_ci.py` re-derives every E7 and E8 interval from the
per-field rows; all agree with the stored statistics.

The manuscript compiles twice to an identical PDF (md5
`6199c476...` before the final two text fixes), 14 pages, zero errors, zero
undefined references.

## 9. What was retained unchanged

E7 and E8, the entire simulated-sensing study, run on Family C throughout.
`records/LIMITATIONS_IOT.md` is retained in full. The visits model remains
the only capacity model whose LP gaps satisfy the acceptance threshold; no
claim is made for operations, hours or hectares.

## 10. Manuscript condensation (final phase)

The manuscript was reduced from 14 pages to 9 (8 pages of body plus one of
references) without altering a single claim, number or caveat. What was
removed was presentation, not evidence:

* three figures --- the yield-response curve, the CIMMYT validation
  scatter, and the IoT decision-value panels. Every number each carried is
  stated in the text or a table.
* three tables --- the related-work positioning matrix, the solver timing
  table, and the sensing ablation table --- plus the scenario-grid half of
  the robustness table. All their content is in prose.
* the rolling-horizon execution algorithm, which supported the withdrawn
  replanning result.
* four references consolidated where several backed one claim
  (19 remain, all cited).
* roughly 55% of the prose, by compression rather than deletion of content.

The surviving figure is the practice-versus-optimised sowing calendar.
Reaching exactly 8 pages would require deleting a whole section --- the
external validation, the related-work positioning, or the sensing study ---
so it was not done.

## 11. Baseline definitions, stated once

Every reported gain names its denominator. B1 is practice as described.
"Best of four" is the four service orders. "Best of five" adds B1R, the
offset-relaxing rule, and is the strongest hand baseline. Table~I and all
abstract, discussion and conclusion figures use best-of-five; the robustness
table uses best-of-four and says so in its footnote, because those runs
predate B1R.

## 12. Unresolved publication risks

1. The in-season replanning contribution is withdrawn, not repaired. A
   reviewer who read the earlier version would find it missing.
2. The practice baseline is a literature-derived construction, not a surveyed
   one. No producer was observed. The gain is therefore against a described
   practice, which is why it is also reported against the best ordering.
3. Field heterogeneity in the sensing study is assumed, not measured.
4. Operations, hours and hectares capacity models remain unproven at N = 60.
5. The CIMMYT validation is a consistency check on GDD arithmetic, and its
   rank statistics are not significant.
