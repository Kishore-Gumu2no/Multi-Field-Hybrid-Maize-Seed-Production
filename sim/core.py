"""
Core biophysical and operational model for multi-field hybrid maize seed production
scheduling under anthesis-synchrony constraints.

All biological parameters are traceable to:
  Marahatta, S., Basnet, R., Bastola, S., Dulal, P.R. (2025).
  "Optimizing male inbred sowing for hybrid maize seed production in winter and spring."
  Journal of Agriculture and Food Research 24, 102350.
  https://doi.org/10.1016/j.jafr.2025.102350   (CC BY 4.0)
"""

import numpy as np

# ----------------------------------------------------------------------------
# 1. THERMAL TIME (phenology)
# ----------------------------------------------------------------------------

T_BASE = 10.0      # degC, standard maize base temperature
T_UPPER = 30.0     # degC, upper cutoff applied to Tmax (CERES-Maize convention)


def daily_gdd(tmax, tmin, t_base=T_BASE, t_upper=T_UPPER):
    """Growing degree days, with upper cutoff on Tmax and floor at zero."""
    tmax_c = np.minimum(tmax, t_upper)
    tmin_c = np.maximum(tmin, t_base)
    tmin_c = np.minimum(tmin_c, t_upper)
    gdd = (tmax_c + tmin_c) / 2.0 - t_base
    return np.maximum(gdd, 0.0)


def flowering_day(gdd_series, sow_day, threshold):
    """
    First day index on which cumulative GDD since sow_day reaches `threshold`.
    Returns np.inf if the threshold is not reached within the series.
    """
    if sow_day < 0 or sow_day >= len(gdd_series):
        return np.inf
    csum = np.cumsum(gdd_series[sow_day:])
    idx = np.searchsorted(csum, threshold)
    if idx >= len(csum):
        return np.inf
    return sow_day + idx


def flowering_day_fast(cum, sow_day, threshold):
    """
    Same as flowering_day but against a precomputed cumulative-GDD array.

    `cum` must be np.concatenate([[0], np.cumsum(gdd)]), so that the GDD
    accumulated from the start of day `s` through the end of day `j` is
    cum[j + 1] - cum[s]. Returns np.inf if the threshold is never reached.
    """
    n = len(cum) - 1
    if sow_day < 0 or sow_day >= n:
        return np.inf
    idx = int(np.searchsorted(cum, cum[sow_day] + threshold))
    if idx > n:
        return np.inf
    return idx - 1


def gdd_cumulative(gdd):
    """Prefix-sum array in the form expected by flowering_day_fast."""
    out = np.empty(len(gdd) + 1)
    out[0] = 0.0
    np.cumsum(gdd, out=out[1:])
    return out


# ----------------------------------------------------------------------------
# 2. YIELD RESPONSE TO TASSELING-SILKING INTERVAL
# ----------------------------------------------------------------------------
# Published quadratic regressions, Marahatta et al. 2025, Fig. 4.
# Y = seed yield (t/ha), X = tasseling-silking interval TSI (days).
#
# TSI = (day of 75% tasseling of MALE line) - (day of 75% silking of FEMALE line)
#
# NOTE ON THE SPRING YEAR-1 EQUATION: as printed, its linear coefficient
# (-0.01532) makes yield INCREASE with TSI, which contradicts the paper's own
# text and every other fitted season. We treat it as a likely typographical
# error and exclude it from the primary analysis.

YIELD_CURVES = {
    "winter_y1": dict(a=3.5918, b=-0.3379, c=0.012,  r2=0.5765, n_stars=3),
    "winter_y2": dict(a=3.2815, b=-0.1610, c=0.039,  r2=0.2907, n_stars=2),
    "spring_y2": dict(a=2.9076, b=-0.2608, c=0.0169, r2=0.3581, n_stars=2),
    # "spring_y1": excluded, see note above.
}

PRIMARY_CURVE = "winter_y1"       # best fit, and winter is the main seed season
TSI_DOMAIN = (0.0, 16.0)          # range over which the regression is used directly
YIELD_CEILING = 4.0               # t/ha; highest treatment mean reported was 3.85

# Floral window durations (Marahatta et al. 2025, Sect. 1):
#   "Pollen shedding and silk emergence may occur over 7-14 days"
#   "the viability of pollen remains for only 18-24 h from desiccation"
#   "effective pollination within the 2-3 days receptive window of the stigma"
POLLEN_SHED_DAYS = 7.0            # duration of male pollen availability
SILK_RECEPTIVE_DAYS = 6.0         # duration of female receptivity at stand level


def _overlap_days(tsi, d_m=POLLEN_SHED_DAYS, d_f=SILK_RECEPTIVE_DAYS):
    """
    Days of overlap between the male pollen-shed window and the female
    receptive window, given TSI = (male tasseling day) - (female silking day).

    Male shed spans  [TSI, TSI + d_m]  relative to female silking at 0.
    Female receptive spans [0, d_f].
    """
    x = np.asarray(tsi, dtype=float)
    return np.maximum(0.0, np.minimum(x + d_m, d_f) - np.maximum(x, 0.0))


def seed_yield(tsi, curve=PRIMARY_CURVE, floor=0.0, sharpness=1.0):
    """
    Expected seed yield (t/ha) as a function of the tasseling-silking interval.

    `sharpness` (lambda) scales how severely the system punishes mistimed
    flowering, holding the optimum fixed:

        Y_lambda(TSI) = Y_opt * (Y_1(TSI) / Y_opt) ** lambda

    lambda = 1 reproduces the published regression exactly. lambda -> 0 makes
    yield nearly insensitive to timing, which is the regime in which Kelly
    et al. (2023) and Becker et al. (2026) found adaptive replanning to be of
    little value. lambda > 1 sharpens the penalty. Sweeping lambda is how this
    study tests whether the value of adaptivity is governed by the curvature
    of the biological response.

    POSITIVE TSI (male tassels after female silks) -- the regime in which
    Marahatta et al. collected essentially all of their data. We use their
    published quadratic directly. The quadratic turns upward beyond its vertex,
    which is an artefact of the polynomial rather than biology, so we clip
    there to keep yield non-increasing in TSI.

    NEGATIVE TSI (male tassels before female silks) -- outside the fitted
    range, and the regression has no left-hand penalty, which would let an
    optimiser sow the male line absurdly early at no cost. Biologically the
    pollen is spent by the time silks emerge. We therefore scale the optimum
    yield by the fraction of the female receptive window that still overlaps
    male pollen shed, using the window durations reported in the same paper.
    This is the single place where the model extends beyond the published
    regression, and it only ever reduces yield.
    """
    p = YIELD_CURVES[curve]
    a, b, c = p["a"], p["b"], p["c"]
    vertex = -b / (2 * c)

    x = np.asarray(tsi, dtype=float)
    x_pos = np.clip(x, 0.0, TSI_DOMAIN[1])
    x_pos = np.minimum(x_pos, vertex)
    y = a + b * x_pos + c * x_pos ** 2

    # left-hand (negative TSI) attenuation by floral-window overlap
    frac = _overlap_days(x) / _overlap_days(0.0)
    y = np.where(x < 0.0, y * frac, y)

    y = np.minimum(y, YIELD_CEILING)
    y = np.maximum(y, floor)

    if sharpness != 1.0:
        y_opt = a                       # value of the regression at TSI = 0
        y = y_opt * np.power(np.clip(y / y_opt, 0.0, 1.0), sharpness)

    return y


def yield_at_optimum(curve=PRIMARY_CURVE):
    """Best attainable yield under the curve (TSI at the left edge of domain)."""
    return float(seed_yield(TSI_DOMAIN[0], curve=curve))


# ----------------------------------------------------------------------------
# 3. SYNTHETIC WEATHER
# ----------------------------------------------------------------------------
# Calibrated to the winter season reported by Marahatta et al. 2025 (Sect. 2.1):
#   Tmax 17.3-34.1 degC, Tmin 6.6-24.9 degC, seasonal rainfall ~386-396 mm,
#   central Terai, Nepal (27 37' N, 84 25' E, 256 m asl).
# Season runs mid-September onward: temperatures decline into winter.

class WeatherGenerator:
    def __init__(self,
                 n_days=200,
                 tmax_start=33.0, tmax_end=20.0,
                 tmin_start=23.0, tmin_end=8.0,
                 temp_noise_sd=2.2, temp_ar=0.70,
                 p_wet_given_dry=0.10, p_wet_given_wet=0.38,
                 rain_mean_mm=11.0,
                 seed=None):
        self.n_days = n_days
        self.tmax_start, self.tmax_end = tmax_start, tmax_end
        self.tmin_start, self.tmin_end = tmin_start, tmin_end
        self.temp_noise_sd, self.temp_ar = temp_noise_sd, temp_ar
        self.p_wd, self.p_ww = p_wet_given_dry, p_wet_given_wet
        self.rain_mean_mm = rain_mean_mm
        self.rng = np.random.default_rng(seed)

    def _seasonal(self, start, end):
        t = np.linspace(0.0, 1.0, self.n_days)
        # smooth cooling trend with a mild curve
        return start + (end - start) * (t ** 0.9)

    def _ar1_noise(self):
        e = np.zeros(self.n_days)
        innov_sd = self.temp_noise_sd * np.sqrt(1 - self.temp_ar ** 2)
        for d in range(1, self.n_days):
            e[d] = self.temp_ar * e[d - 1] + self.rng.normal(0, innov_sd)
        return e

    def sample(self):
        """Return one weather realisation as a dict of daily arrays."""
        noise = self._ar1_noise()
        tmax = self._seasonal(self.tmax_start, self.tmax_end) + noise
        tmin = self._seasonal(self.tmin_start, self.tmin_end) + 0.75 * noise \
            + self.rng.normal(0, 0.8, self.n_days)
        tmin = np.minimum(tmin, tmax - 3.0)       # keep a sane diurnal range

        # rainfall: two-state Markov chain for occurrence, exponential depth
        wet = np.zeros(self.n_days, dtype=bool)
        for d in range(self.n_days):
            p = self.p_ww if (d > 0 and wet[d - 1]) else self.p_wd
            wet[d] = self.rng.random() < p
        rain = np.where(wet, self.rng.exponential(self.rain_mean_mm, self.n_days), 0.0)

        return dict(tmax=tmax, tmin=tmin, rain=rain,
                    gdd=daily_gdd(tmax, tmin))


# ----------------------------------------------------------------------------
# 4. FIELD TRAFFICABILITY
# ----------------------------------------------------------------------------

def trafficability(rain, soil_drain_days, rain_threshold_mm=5.0):
    """
    Boolean array: can the sowing crew enter the field on each day?

    A rain event above `rain_threshold_mm` blocks the field on that day and for
    `soil_drain_days` afterwards. Heavier soils drain more slowly, which is the
    field-level variation that a soil-moisture sensor would resolve and a
    regional rain gauge would not.
    """
    n = len(rain)
    ok = np.ones(n, dtype=bool)
    for d in np.where(rain > rain_threshold_mm)[0]:
        ok[d: d + 1 + soil_drain_days] = False
    return ok


# ----------------------------------------------------------------------------
# 5. FIELD AND FARM DEFINITIONS
# ----------------------------------------------------------------------------

class Field:
    """
    One hybrid seed production field.

    tau_m : thermal time (GDD) from sowing to 75% tasseling of the MALE line
    tau_f : thermal time (GDD) from sowing to 75% silking of the FEMALE line

    Because tau_m > tau_f for the parent pairs reported by Marahatta et al.,
    the male line must generally be sown EARLIER than the female line. The
    number of calendar days that the GDD difference corresponds to depends on
    the temperature actually experienced -- which is exactly the quantity that
    is unknown at sowing time. That is the source of scheduling risk.
    """

    def __init__(self, fid, tau_m, tau_f, area_ha, soil_drain_days,
                 temp_bias=0.0):
        self.fid = fid
        self.tau_m = tau_m
        self.tau_f = tau_f
        self.area_ha = area_ha
        self.soil_drain_days = soil_drain_days
        self.temp_bias = temp_bias   # field microclimate offset, degC

    def gdd_series(self, weather):
        """Field-level GDD, applying this field's microclimate offset."""
        return daily_gdd(weather["tmax"] + self.temp_bias,
                         weather["tmin"] + self.temp_bias)

    def tsi(self, weather, sow_m, sow_f):
        """Realised tasseling-silking interval (days) for given sowing days."""
        g = self.gdd_series(weather)
        d_tassel = flowering_day(g, sow_m, self.tau_m)
        d_silk = flowering_day(g, sow_f, self.tau_f)
        if not np.isfinite(d_tassel) or not np.isfinite(d_silk):
            return np.inf
        return d_tassel - d_silk

    def yield_tons(self, weather, sow_m, sow_f, curve=PRIMARY_CURVE,
                   sharpness=1.0):
        """Total saleable seed (t) from this field."""
        t = self.tsi(weather, sow_m, sow_f)
        if not np.isfinite(t):
            return 0.0
        return float(seed_yield(t, curve=curve, sharpness=sharpness)) * self.area_ha


def make_farm(n_fields=12, seed=7):
    """
    Build a heterogeneous farm.

    Parent thermal-time requirements are set so that, under the reported winter
    temperatures, the female line reaches silking around day 70 and the male
    line reaches tasseling around day 78 when both are sown on the same day --
    consistent with Table 4 of Marahatta et al. (2025), which gives mean days
    to 75% silking of 70.3-74.3 and to 75% tasseling of 78.5-79.0 in winter.
    """
    rng = np.random.default_rng(seed)
    fields = []
    for i in range(n_fields):
        tau_f = rng.uniform(975, 1050)             # GDD to silking (female)
        delta = rng.uniform(25, 70)                # GDD gap male minus female
        fields.append(Field(
            fid=i,
            tau_m=tau_f + delta,
            tau_f=tau_f,
            area_ha=float(rng.uniform(1.5, 4.0)),
            soil_drain_days=int(rng.integers(1, 4)),
            temp_bias=float(rng.normal(0.0, 0.7)),
        ))
    return fields
