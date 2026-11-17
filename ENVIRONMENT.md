# Recorded environment

Captured 2026-11-17 as housekeeping before Target 1.

| Component | Version |
| --- | --- |
| Python | 3.13.18 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| pandas | 3.0.5 |
| PuLP | 2.9.0 |
| CBC (bundled with PuLP) | 2.10.3, build 2019-12-15 |
| matplotlib | 3.11.2 |

CBC binary: `pulp/solverdir/cbc/linux/64/cbc`. No system `cbc` on PATH.
MILP solutions can differ between CBC versions where optima are degenerate,
so this version should be cited in any reproducibility statement.

## DSSAT is NOT installed in this environment

`/DSSAT48` does not exist. `sim/dssat.py` and `precompute_phenology.py`
cannot run here.

Consequences:

- `results/phenology_table.npz` is a **frozen artefact**. It was produced
  elsewhere and cannot currently be regenerated, extended, or re-run with
  different output switches from this environment.
- Any Target 1 proposal that requires additional DSSAT outputs (silking
  date, daily growth stage, a different maize model) is **blocked** until
  DSSAT is available.
- The table's provenance cannot be re-verified from here. Its contents were
  checked and are internally consistent: `adat` of shape (8 cultivars,
  21 years, 61 sowing days), station RNBN, 100% of cells finite.

## Settings recorded from sim/dssat.py

These are what the frozen table was built with, read from the experiment
file template:

| Setting | Value |
| --- | --- |
| Model | `MZCER` (CERES-Maize), DSSAT v4.8 |
| WATER | **Y** (water balance ON) |
| NITRO | **Y** (nitrogen balance ON) |
| IRRIG | **N** (not irrigated -- rainfed) |
| FERTI | `R` -- single basal application, 60 kg N/ha at planting |
| Output switches | `SUMRY Y`, `OVVEW Y`, `GROUT N` |
| Parsed output | `Summary.OUT`, ADAT column only |

Note that `GROUT N` means `PlantGro.OUT` is not written, so daily growth
stage is not currently available even from the runs that were done.
