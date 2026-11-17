# Assumptions and limitations

Every value the model needs that is not VERIFIED. Each carries its bound and
how it is handled. An assumption with a quantified bound is defensible; an
undeclared one is not.

## Genetic coefficients -- the largest limitation

No calibrated coefficient set exists for RML-150, RML-18, RML-95, RML-96,
CAL 1443 or CML 451. **UNKNOWN.** The model runs on DSSAT's shipped cultivars
as **synthetic surrogates**, read from `DSSAT/Data/Genotype/MZCER048.CUL`:

| Code | Name in file | P1 | P2 | P5 | PHINT |
| --- | --- | --- | --- | --- | --- |
| PC0001 | 2500-2600 GDD | 160.0 | 0.750 | 780.0 | 49.00 |
| PC0002 | 2600-2650 GDD | 185.0 | 0.750 | 850.0 | 49.00 |
| PC0003 | 2650-2700 GDD | 212.0 | 0.750 | 850.0 | 49.00 |
| PC0004 | 2700-2750 GDD | 240.0 | 0.750 | 850.0 | 49.00 |
| PC0005 | 2750-2800 GDD | 260.0 | 0.750 | 850.0 | 49.00 |
| 990002 | MEDIUM SEASON | 200.0 | 0.300 | 800.0 | 38.90 |
| IB0012 | PIO 3382 | 160.0 | 0.700 | 950.0 | 38.90 |
| IB0013 | B59*OH43 | 162.0 | 0.800 | 685.0 | 38.90 |

These are **temperate US material and generic GDD maturity classes**. "2500-2600
GDD" is a US Corn Belt maturity rating, not a variety; PIO 3382 is a Pioneer
commercial hybrid; B59*OH43 crosses two classic US public inbreds. **None is a
tropical or subtropical inbred adapted to the Terai or to Karnataka.**

### What this permits

Claims about the scheduling **method** over a declared pair population:
that the formulation finds better schedules than named baselines under declared
resource constraints; that visit merging produces operational savings whose
size depends on the capacity model; the value of adaptivity over the weather
ensemble; how conclusions move across the scenario grid; how the benefit varies
with pair difficulty.

### What this forbids

- Any claim about MAH 14-5, Rampur Hybrid-10/16, CAL 1443 or CML 451.
- Any sowing-interval recommendation for Karnataka or Nepal.
- Any absolute seed-yield figure.
- Any statement that the phenology is **validated**: the DSSAT-versus-thermal
  comparison measures agreement between two **models**, never accuracy against
  an observation. This project holds no field phenology dataset.
- Any claim about inbred-specific behaviour: CERES is calibrated on hybrids,
  and two of these entries are themselves hybrids.
- Any reading of the chosen offsets as agronomic advice.

## Population limitations, *DERIVED* in Phase 3

| # | Limitation | Measured value |
| --- | --- | --- |
| L8 | Few admissible pairs, so fields within a farm are **not independent draws** | 6 of 56 ordered pairs at `rho = 1.0`; 13 on the grand mean; 12 at `rho = 0.8`. At 40 fields, **6.67 repeats per pair** |
| L9 | Some pairs differ in a **single** flowering coefficient | 2 of the 6; PC0001-PC0005 share `P2` 0.750 and `PHINT` 49.00 |
| L10 | Only **two** distinct `PHINT` values exist in the table | 38.90 and 49.00 |
| L11 | `dTSI/doffset` **under-covers** the observed range | synthetic 0.98-1.04 d/d; observed 1.00 (UAS) and **1.58** (Marahatta); shortfall 0.54 |
| - | Across-year SD of `TSI(0)`, which the value of replanning depends on, **is** in the right range | 1.06-1.76 d against ~1.5 d from Target 1A |

L11 is the one that touches the headline mechanism: a shallower slope means
more offset days are needed to move TSI by a given amount. It is reported by
`pairs.population_report` on every run and deliberately does **not** gate --
closing it would mean inventing coefficients.

## Model-form assumptions

| # | Assumption | Bound or levels | Handling |
| --- | --- | --- | --- |
| A1 | Shape of the response below TSI 0 | two observations only: 0.782 at -1.75 d, 0.701 at -3.75 d | `kappa`-reflection; any form must pass the two orderings |
| A2 | Response below TSI -3.75 | entirely unobserved | **excluded from the domain**, rejected not extrapolated |
| A3 | Plateau half-width | **CONTESTED** between the two trials | three scenarios PL0 / PL1 / PL2, none designated correct |
| A4 | 50%-to-75% flowering-basis correction | 0 to -0.5 d, mean -0.28 d | `delta_thr` levels {0, -0.28, -0.5}; default 0 so it shows as a sensitivity |
| A5 | `ASI0` for the male parent | 5 d VERIFIED AS REPORTED in one environment; ASI widens under stress | swept {3, 5, 8} with `TSI(0)` held and `Delta` re-derived |
| A6 | Transferability of the decline rate | UAS declines **1.68x** faster than Marahatta | reported as a range, never reconciled |
| A7 | `kappa` | floor 1.4286 *DERIVED* from the two orderings; magnitudes imply 0.97-2.56 and disagree | swept {1.5, 2.0, 2.5}, **never fitted** |
| A8 | Pollen shed shape `alpha_p`, `beta_p` | no source | mode position swept {1/4, 1/3, 1/2} |
| A9 | Saturation constant `lam` | no source | swept over a decade |
| A10 | `eta_v` within-day viability | **not identifiable** from `lam` | folded into `lam` |
| A11 | Split gap `g` | CIMMYT's criterion is phenological, not a day count | swept {2, 3, 5} |
| A12 | `tau_max` pair-difficulty bound | hardest documented cross ~ +5 d | +8 assumed; sensitivity in E4 |
| A13 | Marahatta per-hectare basis | **UNKNOWN** -- whole plot or female rows | only relative yield used |
| A14 | Published Marahatta coefficients | **UNKNOWN**; full text behind an Elsevier TDM reservation | preprint coefficients with a footnote |

## Operational assumptions -- not measurements

The brief is explicit that these are not real-world measurements, and they are
not defended as such.

| Parameter | Value | Status |
| --- | --- | --- |
| `H_TRAVEL` | 1.5 h per visit | **ASSUMED**, no source |
| `H_OP` | 2.0 h per sowing operation | **ASSUMED**, no source |
| `H_DAY` | 8.0 h working day | **ASSUMED**, plausible, no source |
| `SHARE` | f 2/3, m 1/6, m2 1/6 | **VERIFIED** -- matches CIMMYT's 4:2 ratio and its 66.7% factor. The only externally supported capacity parameter |
| hectares capacity | 3.0 ha/day | **ARTEFACT-AVOIDANCE VALUE**, not agronomic: at 1.8 only 13-15 of 40 fields could ever be sown. Reported as a diagnostic, never a result |
| visits capacity | 2 per day | **ASSUMED** |
| field areas | U(1.5, 4.0) ha | **SYNTHETIC** |
| drainage | U{1, 2, 3} d | **SYNTHETIC** |
| out-of-domain gate | 5% of field-years | **ASSUMED** by judgement; reported with the figure it gates |

## Limitations that cannot be removed within this project

| # | Limitation |
| --- | --- |
| L1 | CERES-Maize has **one** flowering stage; ASI cannot come from the model |
| L2 | "Pollen" appears nowhere in the DSSAT source tree; the pollen model is entirely outside DSSAT |
| L3 | CERES was calibrated on **hybrids**, not inbred parent lines |
| L4 | No stress effect on ASI; the `ASI0` sweep is a crude substitute |
| L5 | `ADAT` is a 50%-type plot-level date against a 75%-basis response curve |
| L6 | The phenology table is a fixed grid; new coefficients require regenerating it, which changes its hash and invalidates stored results |
| L7 | `sim/dssat.py` sets `WATER Y` and `NITRO Y` with 60 kg N basal, so flowering **is** simulated under water and nitrogen limitation. **The manuscript states the opposite and must be corrected** (C8) |
| - | **DSSAT is absent from this environment.** The phenology table is frozen and no DSSAT run is claimed anywhere |
| - | Two site-years of yield-response evidence, from two countries, on different flowering bases |
