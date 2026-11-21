# Optimised Sowing Scheduling for Multi-Field Hybrid Maize Seed Production under Anthesis-Synchrony Constraints

Code, data and results for the paper of the same name (`paper/paper_v4.pdf`).

Hybrid maize seed is grown on many isolated contract fields, but the two
parent lines are sown by one supervised crew. The day a field is sown fixes
the flowering dates of both parents, and therefore whether their pollen and
silk windows overlap at all. Deciding which field the crew starts on which
day is done by hand today. This repository formulates that decision as a
stochastic assignment problem, solves it exactly, and evaluates it on
twenty-one observed seasons at an Indian seed-production site.

## A note on what is not here

An earlier version of this work used a different implementation, which an
audit found to carry five defects: a 10x per-field bonus in the solver
objective, so it optimised a different function from the one it reported; a
yield normaliser built on a pollen/silk overlap formula that is falsified by
the data it was meant to respect; first-male-only scoring; three divergent
scoring paths; and the ensemble mean taken over the tasseling--silking
interval rather than over the yield response of it.

Every scheduling experiment was re-run on the implementation in this
repository, and two results that could not be reproduced on it were
withdrawn from the paper rather than carried forward. **The defective code
and its output have been deleted**, so nothing here can be confused for it
and nothing here can accidentally be run with it.

`records/AUDIT_FINAL.md` documents what was found, what was re-run, what was
retained and what was withdrawn, with the defects quoted. Development history
is in `records/RESEARCH_DECISIONS.md`. If you need the removed code itself,
it is in this repository's git history up to the commit that removed it.

## Reproducing the paper

```bash
pip install numpy scipy pandas matplotlib "pulp==2.9.0" pytest

python3 run_familyc_main.py   --seeds 5    # assignment, practice, greedy, capacity scaling (~25 min)
python3 run_familyc_robust.py --seeds 3    # scenario grid and merging ablation (~30 min)
python3 validate_cimmyt.py                 # external validation (seconds)
python3 run_iot_capacity.py                # simulated sensing under capacity (~1 h)
```

Then check the paper against what those runs produced:

```bash
python3 records/verify_manuscript.py   # 23 checks, recomputed from raw result rows
python3 records/recompute_ci.py        # sensing intervals, re-derived from per-field rows
python3 -m pytest tests -q             # 200 tests
```

`verify_manuscript.py` recomputes every headline, confidence interval, table
cell and solver diagnostic directly from the raw rows and compares each one
against the value printed in the manuscript. It is the quickest way to
confirm that the paper and this code agree. `records/REPRODUCE.md` has the
longer version, including the sensing experiments.

## What the paper finds

The gain is conditional, and the condition is the finding.

| | Result |
|---|---|
| Crew has slack (N = 10, 20) | Exact optimisation is worth **nothing** over the best hand rule: +0.09% and +0.21% |
| Crew over-subscribed, pooled | **+20.8%** [+18.5, +23.1] over the best of five hand rules; +42.5% over practice as described in the literature |
| Largest farm tested (N = 70) | +35.2% and +73.3% against the same two baselines |
| Greedy dispatcher | **-9.6%** against the best hand rule once capacity binds |
| Yield-response robustness | Positive in **all 54** cells of an eighteen-cell scenario grid, +28.6% to +44.9% |
| Visit merging forbidden | Gain falls by about a third, to +23.3%, but does not disappear |
| In-season replanning | **+0.004%** — nothing. An earlier figure of +2.26% is withdrawn |
| Simulated sensing, capacity binding | Sensor +0.583% [+0.435, +0.732]; perfect information +2.279% |
| Simulated sensing, no heterogeneity | Sensor **-1.410%** — noise with nothing to detect is pure cost |

Baselines are named everywhere, because the paper uses two. "Practice as
described" is the per-hybrid recommended offset with fields taken in arrival
order. "Best of five hand rules" adds three other service orders and a rule
that relaxes the recommended offset; it is much stronger, and it is the
figure the paper claims. The robustness table uses the best of **four**
orders, because those runs predate the fifth rule, and says so.

## Two results were withdrawn

In-season replanning and the curvature sweep that depended on it were
produced by the earlier defective implementation. On the current one, re-solving
in-season over the same information is worth +0.004%, which is solver
tolerance, and the reason is structural: the static solve is already
optimal over that candidate set.

That implementation measured something different and more
interesting --- it spliced observed weather onto the forecast ensemble, so
its replanner knew things the static plan did not. Measuring that properly
needs candidate sets conditioned on partially-realised seasons, which is not
implemented here. The question is open. The old numbers are not carried
forward.

`records/AUDIT_FINAL.md` gives the disposition of every number in the paper:
which engine produced it, which result file supports it, whether it
reproduces, and what was retained, replaced or withdrawn.

## Data

| What | Where | Source |
|---|---|---|
| Weather, 21 observed seasons 2005--2025 | `data/power_ranebennur.csv` | NASA POWER, Ranebennur, Karnataka (14.60N, 75.60E) |
| Flowering dates, 8 cultivars x 21 years x 61 sowing days | `results/phenology_table.npz` | Generated by `precompute_phenology.py` from CERES-Maize in DSSAT-CSM v4.8 |

Nothing is generated or synthesised. Every season evaluated is a season that
happened. Flowering dates come from the crop model, not from a degree-day
approximation.

## Biological parameters

| Quantity | Value | Source |
|---|---|---|
| Yield vs TSI (winter, year 1) | Y = 3.5918 - 0.3379x + 0.012x², R² = 0.58 | Marahatta et al. 2025, Fig. 4 |
| Observed TSI domain | [-3.75, 13.75] days | the two trials in the same paper |
| Plateau on the late side | 2 days | same-day treatments, same paper |
| Early-side steepening | kappa = 1.5, floor 1.43 | floor forced by the ordering of the two observed negative-TSI points |
| Thermal base / optimum | 8 °C / 34 °C, on estimated soil temperature | DSSAT `MZCER048.ECO`, as CERES uses them |
| Sensor accuracy | 0.5 °C | Oklahoma Mesonet thermistor specification |
| Between-field thermal spread | 0.7 °C default, swept from 0 | assumed, not measured — see below |

The 10 °C / 30 °C air-temperature degree-day form that appears in the
agronomic literature is used in this repository **only** for the external
validation against CIMMYT, because the comparison source uses that same
arithmetic. It does not produce any scheduling result.

An earlier version attenuated the early-TSI side by a rectangular overlap of
pollen shed and silk receptivity. That formula is falsified by the two
observed negative-TSI points: its errors there have opposite signs. It
remains in `sim/response.py` as `rectangular_overlap_relative`, labelled, and
is used for no number in the paper.

Marahatta, S., Basnet, R., Bastola, S., Dulal, P.R. (2025). *Optimizing male
inbred sowing for hybrid maize seed production in winter and spring.* Journal
of Agriculture and Food Research 24, 102350.
[doi:10.1016/j.jafr.2025.102350](https://doi.org/10.1016/j.jafr.2025.102350)
(CC BY 4.0)

## Limitations

No seed was produced and no sensor was deployed. Nobody has executed one of
these schedules on real fields and weighed the result.

The sensing study is **simulated throughout**. Paired soil-temperature probe
data could not be obtained: the Oklahoma Mesonet serves it behind an
interactive request form, and USCRN publishes no probe-level values. The
between-field thermal spread is therefore assumed, not measured, and the
sensing result is a statement about when such a system would be worth
building, not that it works.

Current practice is described from the literature, not observed. No producer
was interviewed. Because the literature's rule turns out to be the weakest of
several a supervisor could follow, every result is also reported against a
much stronger baseline.

The yield response comes from one site in Nepal, applied at a Karnataka site.
It is the only published quantitative seed-yield-versus-TSI response we could
find. The early-TSI branch is a declared choice, not a fit.

The external validation against CIMMYT is a consistency check on degree-day
arithmetic, not independent biological validation, and neither of its rank
statistics is significant at n = 6 (p = 0.094 and p = 0.111).

Capacity is counted in field-visits. Only that accounting model solves inside
the acceptance gap at the farm size used for the sensing study; the
operations, hours and hectares models do not, and no claim is made for them.

`records/LIMITATIONS_IOT.md` has the full list for the sensing study.

## Layout

```
sim/                  the simulator
run_familyc_main.py   assignment, practice, greedy, capacity scaling
run_familyc_robust.py scenario grid and merging ablation
run_iot*.py           simulated soil-temperature sensing
validate_cimmyt.py    external validation
precompute_phenology.py  builds the DSSAT flowering table
tests/                200 tests
records/              audit trail, verification scripts, run logs
results_fc_main/      the result rows behind the paper's scheduling numbers
results_iot/, results_cap*/  the result rows behind the sensing numbers
results/              the DSSAT flowering table and the validation output
paper/paper_v4.tex    the manuscript
figs/                 figures
```

## Licence

MIT. See `LICENSE`.
