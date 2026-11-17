# When Does Adaptive Scheduling Pay?

Simulation study of in-season replanning for multi-field hybrid maize seed
production, testing whether the limited value of adaptive scheduling reported
for irrigation persists when the biological penalty for mistimed action is
sharp rather than gentle.

## The question

Kelly et al. (2023) and Becker et al. (2026) both found that re-optimising an
agricultural plan during the season adds little over a good fixed plan. Both
studied irrigation, where being late costs a little. In hybrid seed
production, being late costs almost everything: the male parent must shed
pollen while the female parent is receptive, and published field data show
seed yield falling 29–57% as the tasseling–silking interval widens.

This repository tests whether that difference changes the conclusion.

## Layout

```
sim/core.py         phenology, yield response, weather generator, field model
sim/scheduler.py    MILP formulation and the shared execution engine
sim/strategies.py   the five strategies S1–S5
run_experiments.py  experiments E1–E5
make_figures.py     figures
fill_paper.py       injects results into the manuscript
paper/paper.tex     manuscript template (placeholders)
paper/paper_filled.tex  manuscript with results (generated)
results/            CSV output and summary.json
figs/               figures
```

## Reproducing

```bash
pip install numpy scipy pandas matplotlib "pulp==2.9.0"
python run_experiments.py      # ~35 min on 2 cores
python make_figures.py
python fill_paper.py
cd paper && pdflatex paper_filled.tex
```

Everything is seeded, so results are deterministic.

## Strategies

| | Strategy | Uses an ensemble | Adapts in season |
|---|---|---|---|
| S1 | Conventional fixed order | no | no |
| S2 | Static, mean forecast | no | no |
| S3 | Static, robust | yes | no |
| S4 | Dynamic rolling horizon | yes | yes |
| S5 | Wait-and-see (oracle) | — | perfect foresight |

Reported quantities follow stochastic programming convention: VSS = S3 − S2,
value of adaptivity = S4 − S3, EVPI = S5 − S4.

S3 is deliberately a strong baseline. Comparing an adaptive policy against
conventional practice overstates its value, because most of the gain is from
optimising at all rather than from adapting.

## Biological parameters

Every parameter traces to an open-access publication.

| Quantity | Value | Source |
|---|---|---|
| Yield vs TSI (winter y1) | Y = 3.5918 − 0.3379x + 0.012x², R² = 0.58 | Marahatta et al. 2025, Fig. 4 |
| Pollen shed duration | 7 days | Marahatta et al. 2025, §1 |
| Silk receptivity | 2–3 days per silk, 6 days stand-level | Marahatta et al. 2025, §1 |
| Days to 75% silking (winter) | 70.3–74.3 | Marahatta et al. 2025, Table 4 |
| Days to 75% tasseling (winter) | 78.5–79.0 | Marahatta et al. 2025, Table 4 |
| Base / upper temperature | 10 °C / 30 °C | CERES-Maize convention |
| Site temperature and rainfall | 17.3–34.1 °C, 6.6–24.9 °C, 386 mm | Marahatta et al. 2025, §2.1 |

Marahatta, S., Basnet, R., Bastola, S., Dulal, P.R. (2025). *Optimizing male
inbred sowing for hybrid maize seed production in winter and spring.* Journal
of Agriculture and Food Research 24, 102350.
[doi:10.1016/j.jafr.2025.102350](https://doi.org/10.1016/j.jafr.2025.102350)
(CC BY 4.0)

## Two modelling decisions worth knowing about

**The yield regression is extended on one side.** Marahatta's quadratic was
fitted almost entirely on TSI ≥ 0 and has no left-hand penalty, so taken
literally it rewards sowing the male line arbitrarily early at no cost. We
attenuate the optimum on the negative side by the fraction of the female
receptive window still overlapped by pollen shed, using the window durations
reported in the same paper. This only ever reduces yield.

**The spring year-1 regression is excluded.** As printed its linear
coefficient (−0.01532) makes yield rise as the gap widens, contradicting the
paper's own text and every other fitted season. We treat it as a likely
typographical error.

## Known limitations

This is a simulation study; no seed was produced. Phenology is thermal-time
only, not a calibrated DSSAT or APSIM run. Weather is synthetic, calibrated to
the source site's reported ranges rather than a station record. One crop, one
region, one resource type.
