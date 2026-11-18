# Reproducing the IoT experiments

All commands run from the repository root on branch `iot-sensing`.
Python 3, NumPy, PuLP. No network access and no data download required: the
weather record and the DSSAT phenology table are already in the tree.

## Tests

    python3 -m pytest tests/ -q                 # 200 tests, all must pass
    python3 -m pytest tests/test_t14_soilthermal.py tests/test_t15_sensing.py -q

T14 pins the CERES soil equations (clamp order, offset linearity, cap
attenuation). T15 pins the sensing layer (null controls, estimator bias,
missingness fallback, non-anticipativity).

## Calibration

    python3 -c "import json;print(json.load(open('records/emulator_fidelity.json'))['soil']['mean_rmse_days'])"

Regenerating the thresholds takes about 12 minutes; `records/emulator_fidelity.json`
is committed so the experiments run without it.

## E7 -- unconstrained decision value

    python3 run_iot.py                                    # primary + null controls
    python3 run_iot.py --replicate --out results_rep      # 6 sigma x 12 seeds
    python3 run_iot.py --sweep --out results              # noise, missingness, shape
    python3 run_iot.py --operational --out results_op     # offset-freedom ablation

Roughly 25 s per configuration. The replicate grid is about 30 minutes.

## E8 -- capacity-constrained decision value

    # primary: visits model, N=60, 12 seeds   (~30 s per cell, ~6 min)
    python3 run_iot_capacity.py --n-fields 60 --seeds 12 --models visits \
        --time-limit 45 --out results_cap_primary

    # unconstrained reference at the SAME N and seeds   (~4 min)
    python3 run_iot_capacity.py --n-fields 60 --seeds 12 --models visits \
        --time-limit 45 --skip-constrained --out results_cap_unc_only

    # heterogeneity sweep, 6 levels x 6 seeds           (~18 min)
    python3 run_iot_capacity.py --sigma-sweep --n-fields 60 --seeds 6 \
        --time-limit 45 --out results_cap_sigma

    # noise and missingness, 10 levels x 6 seeds        (~30 min)
    python3 run_iot_capacity.py --ablations --n-fields 60 --seeds 6 \
        --time-limit 45 --out results_cap_abl

`--quick` runs two seeds with the unconstrained reference, for a timing check.

## Statistics and figure

Confidence intervals are recomputed from the per-field rows, not from the
runner's own gain columns:

    python3 -c "exec(open('records/recompute_ci.py').read())"
    python3 make_figure_iot.py        # writes figs/fig13_iot_decision_value.{png,pdf}

## Paper

    cd paper && pdflatex paper_v3.tex && pdflatex paper_v3.tex

`paper_v3.tex` is `paper_v2_filled.tex` with `\input{_iot}` and
Figure 13 inserted before the Discussion. Note `paper_v2.tex` is the
unfilled template and does not compile.

## What each result directory holds

| Directory | Contents |
| --- | --- |
| `results_iot/` | E7 consolidated rows and summaries |
| `results_cap_visits12/` | E8 primary, visits, 12 seeds |
| `results_cap_unc_only/` | E8 unconstrained reference, N=60, 12 seeds |
| `results_cap_sigma/` | E8 heterogeneity sweep |
| `results_cap_abl/` | E8 noise and missingness sweeps |
| `records/E7_ci_from_rows.json` | E7 intervals recomputed from rows |
| `records/E8_*.json` | E8 statistics |

## Scheduling results on the corrected engine (Family C)

Every scheduling number in the paper comes from these three commands. They
replace the legacy R1--R8/E1--E6 runners, which are Family A and are kept in
the tree only so that superseded results stay inspectable.

```
python3 run_familyc_main.py --seeds 5 --out results_fc_main   # main grid, ~20 min
python3 run_familyc_robust.py --seeds 3 --out results_fc_main # scenario grid + no-merge, ~25 min
python3 validate_cimmyt.py                                     # external validation, seconds
```

`run_familyc_main.py` gives the assignment, practice and greedy results over
11 farm sizes x 5 farms. The other two hand rules are produced by the same
module (`sim.compare.plan_ordered`) and are written to
`FC_baselines_raw.csv` and `FC_relaxed_raw.csv`.

## Verifying the manuscript against the result files

```
python3 records/verify_manuscript.py   # 23 checks, all must pass
python3 records/recompute_ci.py        # E7/E8 intervals from raw rows
python3 records/trace_results.py       # result file -> runner -> engine family
```

`verify_manuscript.py` recomputes every headline, interval, table cell and
solver diagnostic directly from the raw rows and compares each against the
value printed in the manuscript.

## Figures

```
python3 make_figure_fc.py     # fig1 (implemented response), fig12 (calendars + gain)
python3 make_figure_iot.py    # fig13 (sensing decision value)
```

## Result directories, scheduling

| Directory | Contents |
|---|---|
| `results_fc_main/FC_main_raw.csv` | main grid: practice, greedy, exact |
| `results_fc_main/FC_baselines_raw.csv` | the three other service orders |
| `results_fc_main/FC_relaxed_raw.csv` | the offset-relaxed hand rule |
| `results_fc_main/FC_all_planners.csv` | the three merged; what the paper reads |
| `results_fc_main/FC_robust_raw.csv` | scenario grid and merging ablation |
| `results/` | legacy Family A output, superseded, retained for inspection |
