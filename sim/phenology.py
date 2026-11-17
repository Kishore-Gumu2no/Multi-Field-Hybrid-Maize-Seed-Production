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

THE ASI CORRECTION IS PROVIDER-SPECIFIC  (P2-D4)
------------------------------------------------
The two providers answer with DIFFERENT biological events, so the correction
needed to obtain the tasseling-silking interval differs between them:

  DssatPhenology   `ADAT` is the SILKING date for BOTH parents. VERIFIED from
                   five independent points in the CERES-Maize source (Target
                   1A): MZ_PHENOL.for sets ISDATE = YRDOY at exit from stage 3,
                   MZ_OPHARV.for writes YRNR1 = ISDATE out as ADAT. CERES has
                   only ONE flowering stage, so anthesis is not simulated.
                   Male anthesis must therefore be obtained as
                       male anthesis = ADAT(male) - ASI0.

  ThermalTimePhenology   `tau_m` is defined as thermal time to male 75%
                   TASSELING and `tau_f` to female 75% SILKING (see
                   sim/core.Field). It already returns the correct two events,
                   so it needs NO ASI correction and deliberately IGNORES the
                   asi0 argument.

Subtracting ASI0 on the thermal path would remove about five days from a
quantity that is already right, placing co-sowing in the male-early penalised
region -- the opposite of what both field trials measured. The uniform `tsi()`
interface below exists so callers cannot get this wrong by accident.

ASI0 itself is VERIFIED AS REPORTED (CML-451: anthesis 81 d, silking 86 d,
Marahatta et al.) from a single environment. It is environment-sensitive and is
swept over {3, 5, 8} days rather than treated as a constant.
"""

import os

import numpy as np

from .core import gdd_cumulative, flowering_day_fast


# Nominal GDD per day under the reported winter regime, used only to document
# what a GDD gap means in days. Never used in a calculation.
GDD_PER_DAY_NOMINAL = 20.0

ASI0_DEFAULT = 5.0          # see module docstring; VERIFIED AS REPORTED


class ThermalTimePhenology:
    """
    Degree-day accumulation to a cultivar-specific threshold.

    `tau_m` targets male 75% tasseling and `tau_f` female 75% silking, so this
    provider returns the two events the tasseling-silking interval is defined
    on and applies NO ASI correction.
    """

    name = "thermal-time"
    applies_asi = False

    def prepare(self, field, weather):
        """Return an opaque per-(field, season) object for fast repeat use."""
        return gdd_cumulative(field.gdd_series(weather))

    def anthesis(self, prep, field, sow_day, parent):
        """Day of year of anthesis, or inf if never reached."""
        tau = field.tau_m if parent == "m" else field.tau_f
        return flowering_day_fast(prep, sow_day, tau)

    def male_anthesis(self, prep, field, sow_day, asi0=ASI0_DEFAULT):
        """
        Male anthesis. `asi0` is accepted for interface symmetry and IGNORED:
        tau_m already targets male 75% tasseling (P2-D4).
        """
        del asi0                                    # deliberately unused
        return flowering_day_fast(prep, sow_day, field.tau_m)

    def female_silking(self, prep, field, sow_day):
        """Female silking; tau_f already targets 75% silking."""
        return flowering_day_fast(prep, sow_day, field.tau_f)

    def tsi(self, prep, field, sow_m, sow_f, asi0=ASI0_DEFAULT, delta_thr=0.0):
        """TSI = male anthesis - female silking, plus the basis correction."""
        a = self.male_anthesis(prep, field, sow_m, asi0)
        b = self.female_silking(prep, field, sow_f)
        if not (np.isfinite(a) and np.isfinite(b)):
            return np.inf
        return float(a - b + delta_thr)


class DssatPhenology:
    """
    DSSAT CERES-Maize anthesis dates, precomputed over the decision grid.

    The table is indexed by (cultivar, weather year, sowing day-of-year) and
    holds days from sowing to anthesis. Because anthesis depends only on
    those three things and never on the schedule, enumerating the grid once
    is exact -- this is a lookup, not an approximation.
    """

    name = "dssat"
    applies_asi = True

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

    # ---------------------------------------------------------------- new API
    def _adat_day(self, cult, year_i, sow_day):
        """Calendar day of ADAT (= SILKING) for one cultivar, year, sowing day."""
        ci = self._ci.get(cult)
        di = self._di.get(int(sow_day))
        if ci is None or di is None:
            return np.inf
        v = self.adat[ci, year_i, di]
        if not np.isfinite(v):
            return np.inf
        return int(sow_day) + float(v)

    def male_anthesis(self, prep, field, sow_day, asi0=ASI0_DEFAULT):
        """
        Male anthesis = ADAT(male) - ASI0.

        ADAT is silking (see module docstring), and CERES does not simulate
        anthesis, so the interval must be supplied externally.
        """
        d = self._adat_day(field.cult_m, prep, sow_day)
        return d - float(asi0) if np.isfinite(d) else np.inf

    def female_silking(self, prep, field, sow_day):
        """Female silking = ADAT(female), used directly."""
        return self._adat_day(field.cult_f, prep, sow_day)

    def tsi(self, prep, field, sow_m, sow_f, asi0=ASI0_DEFAULT, delta_thr=0.0):
        """TSI = male anthesis - female silking, plus the basis correction."""
        a = self.male_anthesis(prep, field, sow_m, asi0)
        b = self.female_silking(prep, field, sow_f)
        if not (np.isfinite(a) and np.isfinite(b)):
            return np.inf
        return float(a - b + delta_thr)

    def tsi_by_year(self, cult_m, cult_f, sow_m, sow_f,
                    asi0=ASI0_DEFAULT, delta_thr=0.0):
        """TSI for every weather year in the table, as an array (nan where absent)."""
        out = []
        for yi in range(self.adat.shape[1]):
            a = self._adat_day(cult_m, yi, sow_m)
            b = self._adat_day(cult_f, yi, sow_f)
            out.append(a - float(asi0) - b + delta_thr
                       if np.isfinite(a) and np.isfinite(b) else np.nan)
        return np.asarray(out, dtype=float)

    def tsi_mean(self, cult_m, cult_f, sow_m, sow_f,
                 asi0=ASI0_DEFAULT, delta_thr=0.0):
        """
        Ensemble-mean TSI. Note this averages TSI, which is correct ONLY for
        candidate admissibility. The yield coefficient must average the
        RESPONSE r(TSI) per year, never TSI first (P2-D16).
        """
        v = self.tsi_by_year(cult_m, cult_f, sow_m, sow_f, asi0, delta_thr)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else np.inf


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
