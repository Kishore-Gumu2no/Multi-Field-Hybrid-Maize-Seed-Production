"""
Phenology providers.

Two interchangeable ways to answer the only biological question the
scheduler asks -- "given this parent line, this sowing day and this season,
on which day does it flower?":

  ThermalTimePhenology   the degree-day model used in the provisional study
  DssatPhenology         a precomputed DSSAT CERES-Maize lookup table

Both expose the same call, so the optimiser is unchanged and the two can be
run against each other to quantify what the approximation costs. That
comparison is reported in the manuscript rather than assumed.
"""

import os

import numpy as np

from .core import gdd_cumulative, flowering_day_fast


class ThermalTimePhenology:
    """Degree-day accumulation to a cultivar-specific threshold."""

    name = "thermal-time"

    def prepare(self, field, weather):
        """Return an opaque per-(field, season) object for fast repeat use."""
        return gdd_cumulative(field.gdd_series(weather))

    def anthesis(self, prep, field, sow_day, parent):
        """Day of year of anthesis, or inf if never reached."""
        tau = field.tau_m if parent == "m" else field.tau_f
        return flowering_day_fast(prep, sow_day, tau)


class DssatPhenology:
    """
    DSSAT CERES-Maize anthesis dates, precomputed over the decision grid.

    The table is indexed by (cultivar, weather year, sowing day-of-year) and
    holds days from sowing to anthesis. Because anthesis depends only on
    those three things and never on the schedule, enumerating the grid once
    is exact -- this is a lookup, not an approximation.
    """

    name = "dssat"

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.adat = z["adat"]                       # (cultivar, year, sow_doy)
        self.cultivars = [str(c) for c in z["cultivars"]]
        self.years = [int(y) for y in z["years"]]
        self.sow_days = [int(d) for d in z["sow_days"]]
        self.wsta = str(z["wsta"]) if "wsta" in z else ""
        self._ci = {c: i for i, c in enumerate(self.cultivars)}
        self._yi = {y: i for i, y in enumerate(self.years)}
        self._di = {d: i for i, d in enumerate(self.sow_days)}

    def coverage(self):
        return float(np.isfinite(self.adat).mean())

    def prepare(self, field, weather):
        year = weather.get("year")
        if year not in self._yi:
            raise KeyError(
                f"no DSSAT phenology for year {year}; table covers "
                f"{self.years[0]}-{self.years[-1]}")
        return self._yi[year]

    def anthesis(self, prep, field, sow_day, parent):
        cult = field.cult_m if parent == "m" else field.cult_f
        ci = self._ci.get(cult)
        di = self._di.get(int(sow_day))
        if ci is None or di is None:
            return np.inf
        v = self.adat[ci, prep, di]
        if not np.isfinite(v):
            return np.inf
        return sow_day + float(v)


def load(path=None):
    """
    Return the DSSAT provider if its table exists, otherwise thermal time.

    Falling back rather than failing keeps the provisional pipeline runnable
    on a machine without DSSAT built.
    """
    path = path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "phenology_table.npz")
    if os.path.exists(path):
        return DssatPhenology(path)
    return ThermalTimePhenology()


def compare(dssat, thermal, fields, weathers, sow_days):
    """
    Agreement between the two providers, in days.

    Reports bias and RMSE of the thermal-time anthesis date against DSSAT,
    which is the number the manuscript needs in order to say what the
    provisional study's approximation was worth.
    """
    diffs = []
    for f in fields:
        for w in weathers:
            try:
                pd_ = dssat.prepare(f, w)
            except KeyError:
                continue
            pt = thermal.prepare(f, w)
            for d in sow_days:
                for parent in ("m", "f"):
                    a = dssat.anthesis(pd_, f, d, parent)
                    b = thermal.anthesis(pt, f, d, parent)
                    if np.isfinite(a) and np.isfinite(b):
                        diffs.append(b - a)
    if not diffs:
        return dict(n=0)
    d = np.asarray(diffs, float)
    return dict(n=int(d.size), bias=float(d.mean()),
                rmse=float(np.sqrt((d ** 2).mean())),
                p05=float(np.percentile(d, 5)),
                p95=float(np.percentile(d, 95)),
                max_abs=float(np.abs(d).max()))
