# Reporting audit — the 72 / 108 discrepancy

Resolved 2026-11-18. Evidence: `records/E7_config_audit.json`,
`records/E7_ci_from_rows.json`.

## What each number was

**72 — correct.** The replicated heterogeneity grid: 6 levels of sigma
(0.0, 0.5, 0.7, 1.0, 2.0, 4.0) x 12 farm seeds. This is the set every
confidence interval in the manuscript is computed from, and it is what the
results section describes.

**108 — wrong.** It was the sum of two files' row counts,
`results_rep/E7_iot_summary.csv` (75) + `results_op/E7_iot_summary.csv` (33).
That sum is not a count of anything meaningful. It double-counts the three
preamble configurations that both files re-run (`primary`, `null_homogeneous`,
`null_sigma0`), and it silently omits the 41-row first sweep in
`results/E7_iot_summary.csv` altogether. The number should not have appeared.

## The correct accounting

| Quantity | Count |
| --- | --- |
| Result rows across all three summary files | 149 |
| Distinct experimental configurations | **123** |
| Redundant rows (same configuration, different tag) | 26 |
| Replicated heterogeneity grid (the CI basis) | **72** |
| Operational offset-freedom grid | 30 |
| First exploratory sweep, unique part | 21 |

The 26 redundant rows are not an error in the experiments; they are the same
configuration recorded under more than one tag because each invocation re-runs
a short preamble, and because the first sweep's `seed_11`...`seed_18` are the
same cells as `rep_sigma0.7_seed11`...`seed18`. Verified identical to the
sixth decimal on all eight. They were counted once in every statistic.

## Independent recomputation of every confidence interval

All intervals were recomputed from the per-field rows in
`results_iot/E7_replicated_rows.csv` -- summing `area_r` per arm per
configuration and forming the gain from scratch -- rather than from the
`gain__*` columns the runner wrote.

Maximum absolute difference across all 75 configurations: **2.4e-14**.
Floating-point identical. Every E7 figure quoted in `paper/_iot.tex` is
confirmed against the underlying rows, not against a derived summary.

## What this changes in the manuscript

Nothing numerical. The manuscript quotes only the 72-configuration grid, which
was correctly described. The "108 configurations" phrase appeared in a summary
document, not in `paper/_iot.tex`, and has been corrected to the accounting
above.
