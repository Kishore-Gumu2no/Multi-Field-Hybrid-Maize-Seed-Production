# Decision D1 — which engine the work runs on

Resolved 2026-11-18. Evidence: `records/engine_import_map.txt`, regenerable
from the script embedded in this repository's history.

## The inconsistency

The tree contains two complete, incompatible scheduling stacks.

**Family A — legacy.** `sim/core.py`, `sim/scheduler.py`, `sim/strategies.py`,
`sim/baselines.py`. Carries the four defects found in the Phase 3 audit: a
10x solver bonus in `scheduler.py` lines 179-180, scoring on the falsified
`core.seed_yield` normaliser, single-male-only scoring, and three divergent
scoring paths.

**Family C — corrected.** `sim/build.py`, `optimise.py`, `pairs.py`,
`response.py`, `runspec.py`, `scoring.py`, `capacity.py`, `pollen.py`,
`phenology.py`, `compare.py`. Built in Phase 3 and U4 to fix all four. Every
candidate, baseline and executed plan enters the single scorer
`scoring.score_plan`, so the objective coefficient and the execution score are
the same number by construction.

**The inconsistency is not that both exist. It is that nothing produced
results on the corrected one.** Before this change, every one of the twelve
top-level experiment runners imported Family A. Family C was imported only by
`tests/` and `records/` — it was verified and never used. The manuscript's
numbers come from Family A.

## Decision

**Family C is the engine. All new work runs on it.**

`run_iot.py` is the first top-level runner on Family C and is the template for
any that follow. It reaches the optimiser only through `scoring.score_plan`,
so it cannot reintroduce a divergent scoring path.

Family A is retained, untouched, and reclassified as historical. Its runners
reproduce the provisional study and must not be used for new results. They are
not deleted: the manuscript cites numbers produced by them, and deleting the
code that produced a published figure would make that figure unreproducible.

## What this costs

Every headline figure in the current manuscript was produced on Family A and
must be either re-run on Family C or explicitly labelled as provisional. That
is a real cost and it is the reason the decision sat open for three gates.
It is cheaper now than after another round of results.

## What it does not change

The phenology table `results/phenology_table.npz`, the weather record, the
parent-pair population and the response scenarios are shared by both families.
Moving engines does not move the biology.
