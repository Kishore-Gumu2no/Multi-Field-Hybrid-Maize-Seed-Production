"""
The air-temperature degree-day form used by the agronomic literature.

This is NOT the model behind any scheduling result in the paper. Flowering
dates there come from CERES-Maize in DSSAT, whose thermal time runs on an
estimated SOIL temperature with T_base = 8 and T_opt = 34 while fewer than
ten leaves have appeared (see `sim/soilthermal.py`).

This module exists for two consumers that legitimately need the simple form:

  * `validate_cimmyt.py`, because the comparison source reports its parent
    requirements with exactly this arithmetic, so reproducing their numbers
    requires reproducing their formula;
  * `phenology.ThermalTimePhenology`, the degree-day provider retained as a
    second, independent phenology path in the tests.

Extracted from the superseded engine when that engine was removed. The
arithmetic is unchanged, and `validate_cimmyt.py` reproduces its published
result to 0.0e+00 across it.
"""
import numpy as np

T_BASE = 10.0
T_UPPER = 30.0


def daily_gdd(tmax, tmin, t_base=T_BASE, t_upper=T_UPPER):
    """Growing degree days, with upper cutoff on Tmax and floor at zero."""
    tmax_c = np.minimum(tmax, t_upper)
    tmin_c = np.maximum(tmin, t_base)
    tmin_c = np.minimum(tmin_c, t_upper)
    gdd = (tmax_c + tmin_c) / 2.0 - t_base
    return np.maximum(gdd, 0.0)


def gdd_cumulative(gdd):
    """Prefix-sum array in the form expected by flowering_day_fast."""
    out = np.empty(len(gdd) + 1)
    out[0] = 0.0
    np.cumsum(gdd, out=out[1:])
    return out


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

