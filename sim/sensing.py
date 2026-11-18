"""
Virtual soil-temperature sensing.

NOTHING HERE IS A MEASUREMENT
-----------------------------
Every number this module produces is synthetic. No soil-temperature
observation was retrieved for this study: the Oklahoma Mesonet's paired 5 cm
probes sit behind an interactive form and a robots-disallowed endpoint, and
USCRN publishes no probe-level soil temperature -- both readmes state the
reported soil values are "calculated from these multiple independent
measurements". That retrieval workstream is closed, and this module replaces
it with an explicitly simulated sensing layer whose every parameter is
configurable and recorded on each result row.

WHAT IS SIMULATED, AND WHY EACH PIECE
-------------------------------------
1. A per-field soil thermal offset `delta_i`, in degrees C, held constant
   through the season. This is the quantity CERES does not have: MZ_PHENOL.for
   imputes soil temperature from air temperature and radiation with fixed
   coefficients carrying no field-specific term, so every field in a season
   receives the same estimated soil temperature whatever its soil does.

2. A measurement: the probe reads the field's true soil temperature plus
   Gaussian noise, and misses days at a configurable rate.

3. An estimator: `delta_hat` is the mean over observed days of
   (measured soil temperature - the soil temperature CERES would impute).
   This is the physically correct estimator, and it is also the only one the
   information can be extracted by, since the imputation is the thing being
   corrected.

NON-ANTICIPATIVITY
------------------
Observation happens in a window that CLOSES BEFORE the sowing window opens.
A field offset is a persistent property of the field -- soil texture, colour,
drainage -- not a weather forecast, so observing it beforehand is informative
about an unsown field without any knowledge of future weather. This is the
distinction that makes the channel admissible at all: an earlier audit of this
repository established that observing past WEATHER carries no decision value,
because thermal time integrates from the sowing day and observed days fall
outside an unsown field's window.

OFFSET MAGNITUDE IS NOT CHOSEN TO SUIT THE SENSOR
-------------------------------------------------
The default scale, sigma = 0.7 degC, is the value already present in
`sim.core.make_farm` as `temp_bias=rng.normal(0.0, 0.7)`, committed long
before this workstream existed. It is used as the default so the headline
scenario cannot be accused of being tuned. Every experiment sweeps sigma from
0 upward regardless, and the sigma = 0 arm is a null scenario in which the
sensor must show no benefit.
"""

from dataclasses import dataclass, asdict

import numpy as np

from . import soilthermal as _st

# Already in sim.core.make_farm, predating this workstream. See docstring.
SIGMA_DEFAULT = 0.7

# Documented accuracy of the 5 cm thermistor probes whose data this study was
# unable to retrieve (Oklahoma Mesonet TB05/TS05, +/- 0.5 degC). Used as the
# default measurement noise so the simulated instrument is no better than a
# real one that exists.
NOISE_DEFAULT = 0.5

SCENARIOS = ("homogeneous", "normal", "bimodal", "gradient")


@dataclass(frozen=True)
class SensingConfig:
    """Everything that defines one simulated sensing layer. Recorded per row."""
    scenario: str = "normal"
    sigma_c: float = SIGMA_DEFAULT
    noise_c: float = NOISE_DEFAULT
    p_missing: float = 0.0
    n_obs_days: int = 20
    obs_end_doy: int = 273          # closes the day before the sowing window
    seed: int = 11
    estimator: str = "imputation_residual"

    def as_row(self):
        return asdict(self)


def field_offsets(n_fields, cfg):
    """
    True per-field soil thermal offsets, in degrees C.

    homogeneous : all zero. The null scenario.
    normal      : N(0, sigma). The default.
    bimodal     : two soil groups at -sigma and +sigma, split evenly.
    gradient    : a linear trend across field index, spanning +/- sigma.
    """
    rng = np.random.default_rng(cfg.seed)
    n = int(n_fields)
    if cfg.scenario == "homogeneous":
        return np.zeros(n, dtype=float)
    if cfg.scenario == "normal":
        return rng.normal(0.0, float(cfg.sigma_c), size=n)
    if cfg.scenario == "bimodal":
        half = n // 2
        v = np.concatenate([np.full(half, -float(cfg.sigma_c)),
                            np.full(n - half, float(cfg.sigma_c))])
        return rng.permutation(v)
    if cfg.scenario == "gradient":
        if n == 1:
            return np.zeros(1, dtype=float)
        return np.linspace(-float(cfg.sigma_c), float(cfg.sigma_c), n)
    raise ValueError(f"unknown scenario {cfg.scenario!r}; expected one of {SCENARIOS}")


def observe(season, offsets, cfg):
    """
    Simulate one pre-season observation campaign.

    Returns (delta_hat, n_used) with one entry per field. A field whose every
    observation day is missing yields delta_hat = nan, which the planner must
    then fall back from -- that is the point of the missingness sweep.
    """
    rng = np.random.default_rng(cfg.seed + 10_000)
    i_end = int(cfg.obs_end_doy) - 1
    i_start = max(0, i_end - int(cfg.n_obs_days) + 1)
    tmax = season.tmax[i_start:i_end + 1]
    tmin = season.tmin[i_start:i_end + 1]
    srad = season.srad[i_start:i_end + 1]
    if tmax.size == 0:
        return (np.full(len(offsets), np.nan), np.zeros(len(offsets), dtype=int))

    # What CERES would impute for these days, with no field term at all.
    td0, tn0 = _st.soil_pair(tmax, tmin, srad, 0.0)
    imputed = (td0 + tn0) / 2.0

    hats, used = [], []
    for d in np.asarray(offsets, dtype=float):
        truth = imputed + d
        meas = truth + rng.normal(0.0, float(cfg.noise_c), size=truth.size)
        keep = rng.random(truth.size) >= float(cfg.p_missing)
        if not keep.any():
            hats.append(np.nan)
            used.append(0)
            continue
        hats.append(float(np.mean(meas[keep] - imputed[keep])))
        used.append(int(keep.sum()))
    return np.asarray(hats, dtype=float), np.asarray(used, dtype=int)


def beliefs(arm, offsets, delta_hat):
    """
    What each planner believes the field offsets are.

    conventional : zero everywhere. This is exactly what CERES does today --
                   impute soil temperature from air and radiation, with no
                   field term. It is the baseline, not a strawman.
    sensor       : the estimate, falling back to zero where estimation failed.
    oracle       : the truth. Not achievable; it bounds the sensor arm and
                   says how much of the available information the sensor got.
    """
    o = np.asarray(offsets, dtype=float)
    if arm == "conventional":
        return np.zeros_like(o)
    if arm == "oracle":
        return o.copy()
    if arm == "sensor":
        h = np.asarray(delta_hat, dtype=float).copy()
        return np.where(np.isfinite(h), h, 0.0)
    raise ValueError(f"unknown arm {arm!r}")


ARMS = ("conventional", "sensor", "oracle")
