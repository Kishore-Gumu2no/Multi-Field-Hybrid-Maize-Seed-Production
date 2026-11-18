"""
CERES-Maize soil-temperature thermal pathway, and a field-level offset.

WHY THIS MODULE EXISTS
----------------------
The phenology providers in `sim.phenology` answer flowering from air
temperature (`ThermalTimePhenology`) or from a precomputed DSSAT lookup
(`DssatPhenology`). Neither carries a FIELD-LEVEL thermal environment: the
lookup is keyed by (cultivar, year, sowing day) only, so two fields with the
same parent pair sown on the same day flower on the same day by construction.

That is not what CERES-Maize does. While `LEAFNO .LE. 10` -- the whole leaf
appearance phase that sets flowering date -- CERES drives thermal time from
SOIL temperature, which it ESTIMATES from air temperature and solar radiation
with fixed coefficients carrying no field-specific term (MZ_PHENOL.for lines
457-496). The model needs a soil temperature it does not have, and imputes it.

This module supplies that pathway explicitly so a per-field soil thermal
offset has somewhere to enter. It is the substrate for the sensing experiment
in `sim.sensing`: nothing here asserts that any field has any particular
offset.

VERIFIED EQUATIONS  (MZ_PHENOL.for, DSSAT source read directly)
---------------------------------------------------------------
  line 470  ACOEF  = 0.01061 * SRAD + 0.5902
  line 471  TDSOIL = ACOEF * TMAX + (1 - ACOEF) * TMIN
  line 472  TNSOIL = 0.36354 * TMAX + 0.63646 * TMIN
  473-494   TNSOIL floored at TBASE and TDSOIL capped at DOPT **before** the
            two are averaged; the daily result is then capped at DOPT - TBASE.
            Clamp order is load-bearing: a 40/20 pair gives 19.0, whereas
            averaging first and clamping after would give 22.0.

  TBASE = 8.0, TOPT = ROPT = 34.0 for every standard maize ecotype in the
  shipped MZCER048.ECO. DOPT = TOPT during vegetative growth.

NOTE these differ from `sim.core`, which uses T_BASE = 10.0 and T_UPPER = 30.0
on AIR temperature. That discrepancy is real and is reported in the manuscript
rather than silently reconciled.

FIDELITY, STATED PLAINLY
------------------------
`calibrate_thresholds` fits one thermal threshold per cultivar so this pathway
reproduces the existing DSSAT table as closely as a single-parameter thermal
model can. Measured over 8 cultivars x 21 years x 61 sowing days, that is
about 2.6 days RMSE with near-zero bias (`records/emulator_fidelity.json`).

2.6 days is LARGER than the one-day decision unit. This pathway is therefore
used as a self-consistent simulation substrate -- truth and both planners run
on it -- and never as a claim to reproduce DSSAT exactly. The comparison in
`run_iot.py` is paired within this model, so the emulator's absolute error
cancels; what would not cancel is a biased offset, and the bias is ~0.
"""

from dataclasses import dataclass

import numpy as np

# MZCER048.ECO, every standard maize ecotype shipped with DSSAT
TBASE = 8.0
TOPT = 34.0
ROPT = 34.0
DOPT_VEG = TOPT                 # DOPT = TOPT during vegetative growth
DTT_CAP = DOPT_VEG - TBASE      # 26.0, the AMIN1 ceiling


def acoef(srad_mj):
    """MZ_PHENOL.for line 470. SRAD in MJ/m^2/day."""
    return 0.01061 * np.asarray(srad_mj, dtype=float) + 0.5902


def soil_pair(tmax, tmin, srad_mj, offset_c=0.0):
    """
    CERES's estimated daytime and night-time soil temperature, plus a field
    offset applied to BOTH. The offset is the only thing here that is not in
    the DSSAT source; it is the quantity a buried probe would measure.
    """
    tmax = np.asarray(tmax, dtype=float)
    tmin = np.asarray(tmin, dtype=float)
    a = acoef(srad_mj)
    td = a * tmax + (1.0 - a) * tmin + offset_c
    tn = 0.36354 * tmax + 0.63646 * tmin + offset_c
    return td, tn


def dtt_from_pair(t_day, t_night):
    """
    Daily thermal time from a day/night soil pair, MZ_PHENOL.for 473-494.
    Clamp each component, THEN average, THEN cap. Order is load-bearing.
    """
    td = np.asarray(t_day, dtype=float)
    tn = np.asarray(t_night, dtype=float)
    out = np.zeros_like(td)
    live = td >= TBASE                       # TDSOIL < TBASE -> the day is zero
    tn_c = np.maximum(tn, TBASE)
    td_c = np.minimum(td, DOPT_VEG)
    out[live] = (tn_c[live] + td_c[live]) / 2.0 - TBASE
    return np.minimum(np.maximum(out, 0.0), DTT_CAP)


def soil_dtt(tmax, tmin, srad_mj, offset_c=0.0):
    """Daily thermal time on the CERES soil pathway, with a field offset."""
    return dtt_from_pair(*soil_pair(tmax, tmin, srad_mj, offset_c))


def air_dtt(tmax, tmin, offset_c=0.0):
    """
    The same clamp arithmetic applied to AIR temperature, for the ablation
    that asks how much of the result is the soil pathway rather than the
    CERES constants.
    """
    return dtt_from_pair(np.asarray(tmax, float) + offset_c,
                         np.asarray(tmin, float) + offset_c)


# --------------------------------------------------------------------------
# Season weather
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SeasonWeather:
    """
    One sowing year's daily record, running from 1 January of that year into
    the following year so that a late sowing can still reach flowering.
    Index i corresponds to day-of-year i+1 of `year`.
    """
    year: int
    tmax: np.ndarray
    tmin: np.ndarray
    srad: np.ndarray

    def __len__(self):
        return int(self.tmax.size)


def seasons_from_power(df, years):
    """Build one SeasonWeather per year from a parsed NASA POWER frame."""
    out = {}
    for y in years:
        cur = df[df.year == y].sort_values("doy")
        nxt = df[df.year == y + 1].sort_values("doy")
        def cat(col):
            a = cur[col].to_numpy(dtype=float)
            b = nxt[col].to_numpy(dtype=float) if len(nxt) else np.empty(0)
            return np.concatenate([a, b])
        out[y] = SeasonWeather(year=int(y), tmax=cat("tmax"),
                               tmin=cat("tmin"), srad=cat("srad"))
    return out


# --------------------------------------------------------------------------
# Flowering from accumulated thermal time
# --------------------------------------------------------------------------

def _cum(dtt):
    """Prefix sum in the form `days_to_threshold` expects."""
    out = np.empty(dtt.size + 1, dtype=float)
    out[0] = 0.0
    np.cumsum(dtt, out=out[1:])
    return out


def days_to_threshold(cum, sow_index, thresh):
    """
    Days from sowing until accumulated thermal time reaches `thresh`.

    `cum[i]` is thermal time accumulated through the end of day i-1, so the
    accumulation from the start of day `sow_index` is cum[j] - cum[sow_index].
    Returns nan if the threshold is never reached inside the record.
    """
    target = cum[sow_index] + float(thresh)
    j = int(np.searchsorted(cum, target))
    if j > cum.size - 1:
        return float("nan")
    return float(j - 1 - sow_index)


class SoilThermalPhenology:
    """
    Flowering from CERES soil-pathway thermal time, with a per-field offset.

    `thresholds` maps cultivar -> thermal time from sowing to ADAT, calibrated
    by `calibrate_thresholds`. As in `sim.phenology.DssatPhenology`, ADAT is
    the SILKING date for both parents, so male anthesis needs the ASI0
    correction and this provider sets `applies_asi = True`.
    """

    name = "soil-thermal"
    applies_asi = True

    def __init__(self, thresholds, seasons, pathway="soil"):
        if pathway not in ("soil", "air"):
            raise ValueError("pathway must be 'soil' or 'air'")
        self.thresholds = dict(thresholds)
        self.seasons = dict(seasons)
        self.pathway = pathway
        self.years = sorted(self.seasons)
        self._yi = {y: i for i, y in enumerate(self.years)}
        self._cache = {}

    def cum_for(self, year, offset_c=0.0):
        """Cumulative thermal time for one season at one field offset."""
        key = (int(year), round(float(offset_c), 6))
        hit = self._cache.get(key)
        if hit is None:
            w = self.seasons[int(year)]
            d = (soil_dtt(w.tmax, w.tmin, w.srad, offset_c) if self.pathway == "soil"
                 else air_dtt(w.tmax, w.tmin, offset_c))
            hit = _cum(d)
            self._cache[key] = hit
        return hit

    def adat_day(self, cult, year, sow_day, offset_c=0.0):
        """Calendar day of ADAT (silking) for one cultivar, year, sowing day."""
        S = self.thresholds.get(cult)
        if S is None:
            return float("nan")
        cum = self.cum_for(year, offset_c)
        i0 = int(sow_day) - 1
        if i0 < 0 or i0 >= cum.size - 1:
            return float("nan")
        n = days_to_threshold(cum, i0, S)
        return float(sow_day) + n if np.isfinite(n) else float("nan")

    def tsi_one_year(self, cult_m, cult_f, sow_m, sow_f, year,
                     offset_c=0.0, asi0=5.0, delta_thr=0.0):
        """TSI = male anthesis - female silking, both at this field's offset."""
        a = self.adat_day(cult_m, year, sow_m, offset_c)
        b = self.adat_day(cult_f, year, sow_f, offset_c)
        if not (np.isfinite(a) and np.isfinite(b)):
            return float("nan")
        return float(a - float(asi0) - b + float(delta_thr))

    def tsi_by_year(self, cult_m, cult_f, sow_m, sow_f,
                    offset_c=0.0, asi0=5.0, delta_thr=0.0):
        """Per-year TSI array, in `self.years` order. Matches the Family C API."""
        return np.array([
            self.tsi_one_year(cult_m, cult_f, sow_m, sow_f, y,
                              offset_c, asi0, delta_thr)
            for y in self.years], dtype=float)

    def tsi_callable(self, cult_m, cult_f, offset_c=0.0, asi0=5.0, delta_thr=0.0):
        """
        A `(sow_m, sow_f) -> per-year TSI` callable, which is exactly what
        `sim.scoring.score_plan` consumes. This is the whole integration seam:
        belief and truth differ only in the `offset_c` passed here.
        """
        def fn(sow_m, sow_f):
            return self.tsi_by_year(cult_m, cult_f, sow_m, sow_f,
                                    offset_c, asi0, delta_thr)
        return fn


# --------------------------------------------------------------------------
# Calibration against the existing DSSAT table
# --------------------------------------------------------------------------

def calibrate_thresholds(adat, cultivars, years, sow_days, seasons,
                         pathway="soil", lo=600.0, hi=1400.0, step=5.0,
                         refine=0.5):
    """
    Fit one thermal threshold per cultivar to the precomputed DSSAT table.

    Coarse grid then a local refinement. Returns (thresholds, fidelity) where
    fidelity carries per-cultivar RMSE and bias in days, so the cost of the
    approximation is reported rather than assumed.
    """
    prov = SoilThermalPhenology({}, seasons, pathway=pathway)
    cums = {y: prov.cum_for(y, 0.0) for y in years}

    def err_for(ci, S):
        errs = []
        for yi, y in enumerate(years):
            cum = cums[y]
            for di, d0 in enumerate(sow_days):
                t = adat[ci, yi, di]
                if not np.isfinite(t):
                    continue
                n = days_to_threshold(cum, int(d0) - 1, S)
                if np.isfinite(n):
                    errs.append(n - float(t))
        if not errs:
            return None
        e = np.asarray(errs, dtype=float)
        return float(np.sqrt((e ** 2).mean())), float(e.mean()), int(e.size)

    thresholds, fidelity = {}, {}
    for ci, c in enumerate(cultivars):
        best = None
        for S in np.arange(lo, hi, step):
            r = err_for(ci, S)
            if r and (best is None or r[0] < best[1]):
                best = (float(S), r[0], r[1], r[2])
        if best is None:
            continue
        for S in np.arange(best[0] - step, best[0] + step, refine):
            r = err_for(ci, float(S))
            if r and r[0] < best[1]:
                best = (float(S), r[0], r[1], r[2])
        thresholds[c] = best[0]
        fidelity[c] = dict(threshold_gdd=best[0], rmse_days=best[1],
                           bias_days=best[2], n=best[3])
    return thresholds, fidelity
