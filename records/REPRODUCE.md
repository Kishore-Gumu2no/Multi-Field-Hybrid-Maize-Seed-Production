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
