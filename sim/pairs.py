"""
Parent-pair admissibility and population characterisation.

WHY THIS MODULE EXISTS
----------------------
The superseded rule in `run_real.pick_parent_pairs` admitted any cultivar pair
whose flowering gap satisfied ``1.5 <= abs(gap) <= 14``. The absolute value
admits pairs in which the MALE silks before the female, which is not a seed
production cross. Across 20 pairs the admitted gap ran -9.97 to +13.45 days,
so with the ASI correction applied the population's same-day TSI landed at
-3.24 days -- deep in the male-early penalised region and the opposite of what
both field trials measured as optimal.

A seed-production pair is chosen so that male ANTHESIS coincides with female
SILKING when the two are co-sown. Because DSSAT's ``ADAT`` is the silking date
for both parents (VERIFIED from CERES-Maize source, Target 1A), the same-day
tasseling-silking interval is

    TSI(0) = Delta - ASI0,     Delta = ADAT(male) - ADAT(female)

and a pair is admissible when ``TSI(0)`` lies in a declared agronomic range.

BASIS FOR THE RANGE  (nothing stronger is claimed)
  tau_min = 0   VERIFIED  MAH 14-5 co-sown reaches TSI(0) = 0.00 (UAS Bengaluru)
                VERIFIED  Marahatta's pairs reach TSI(0) = +1.00 to +2.25
  tau_max = 8   ASSUMED   CIMMYT's T33 x T10 needs the female sown ~5 days
                          first, implying TSI(0) ~ +5 (SUPPORTED). +8 extends
                          modestly past the hardest documented cross so the
                          farm population contains genuinely difficult fields.
                          Belongs in the sensitivity set.

Supplement correction X6: admissibility is evaluated PER WEATHER YEAR, not on
a grand mean, to agree with the per-year candidate rule in the optimiser.
Supplement correction X7: ``dTSI/doffset`` is reported per pair against the two
observed values. It is a DISCLOSED LIMITATION, never a gate -- closing the gap
would mean inventing coefficients.
"""

from dataclasses import dataclass, field as _dcfield

import numpy as np

# --------------------------------------------------------------------------
# Declared constants
# --------------------------------------------------------------------------

ASI0_DEFAULT = 5.0      # VERIFIED AS REPORTED: CML-451 anthesis 81 d, silking
                        # 86 d (Marahatta et al.). Environment-sensitive; swept
                        # over {3, 5, 8} per Phase 2 section 3.6.
ASI0_LEVELS = (3.0, 5.0, 8.0)

TAU_MIN = 0.0           # VERIFIED lower edge of observed same-day TSI
TAU_MAX = 8.0           # ASSUMED upper edge; sensitivity factor in E4

# Observed dTSI/doffset, for comparison only (Phase 1 section 3, DERIVED from
# the two published trials). Never used to accept or reject a pair.
SLOPE_OBSERVED = {"uas_bengaluru": 1.00, "marahatta": 1.58}

# Across-year SD of TSI(0) carried from Target 1A, for comparison only.
TSI0_SD_TARGET = 1.5

# DSSAT genetic coefficients for the cultivars in the precomputed table.
# VERIFIED by direct read of DSSAT/Data/Genotype/MZCER048.CUL in the
# dssat-csm-os source tree. NOT invented, NOT calibrated to any parent line of
# the hybrids this project discusses. See ASSUMPTIONS_AND_LIMITATIONS.md.
#   (name, P1, P2, P5, PHINT)
CULTIVAR_COEFFS = {
    "PC0001": ("2500-2600 GDD", 160.0, 0.750, 780.0, 49.00),
    "PC0002": ("2600-2650 GDD", 185.0, 0.750, 850.0, 49.00),
    "PC0003": ("2650-2700 GDD", 212.0, 0.750, 850.0, 49.00),
    "PC0004": ("2700-2750 GDD", 240.0, 0.750, 850.0, 49.00),
    "PC0005": ("2750-2800 GDD", 260.0, 0.750, 850.0, 49.00),
    "990002": ("MEDIUM SEASON", 200.0, 0.300, 800.0, 38.90),
    "IB0012": ("PIO 3382",      160.0, 0.700, 950.0, 38.90),
    "IB0013": ("B59*OH43",      162.0, 0.800, 685.0, 38.90),
}

# Coefficients that affect ADAT. P5 is silking -> physiological maturity and
# cannot change flowering (P2-D1, VERIFIED from CERES source).
FLOWERING_COEFFS = ("P1", "P2", "PHINT")


class InadmissiblePair(ValueError):
    """Raised by invariant SG1 when a pair cannot be a seed-production cross."""


@dataclass(frozen=True)
class PairInfo:
    cult_m: str
    cult_f: str
    delta_mean: float
    tsi0_mean: float
    tsi0_min: float
    tsi0_max: float
    tsi0_sd: float
    slope: float
    asi0: float
    single_coefficient: bool
    n_years: int

    def label(self):
        return f"{self.cult_m}x{self.cult_f}"


# --------------------------------------------------------------------------
# Scalar relations -- the arithmetic the invariants test
# --------------------------------------------------------------------------

def same_day_tsi_from_delta(delta, asi0=ASI0_DEFAULT):
    """TSI at offset 0, given the nicking gap and the male ASI."""
    return float(delta) - float(asi0)


def delta_for_target_tsi0(tsi0, asi0=ASI0_DEFAULT):
    """
    Inverse of `same_day_tsi_from_delta`.

    Used by the ASI sweep: TSI(0) is held at its stratified value and Delta is
    re-derived, so sweeping ASI0 does not shift the whole population's same-day
    TSI in lockstep (Phase 2 section 3.6).
    """
    return float(tsi0) + float(asi0)


def check_admissible_delta(delta, asi0=ASI0_DEFAULT,
                           tau_min=TAU_MIN, tau_max=TAU_MAX):
    """
    Invariant SG1. Raises `InadmissiblePair` with a diagnostic naming Delta,
    ASI0 and the resulting TSI(0) -- never returns a silently clipped value.
    """
    tsi0 = same_day_tsi_from_delta(delta, asi0)
    if not (tau_min - 1e-9 <= tsi0 <= tau_max + 1e-9):
        raise InadmissiblePair(
            f"pair rejected by SG1: delta={delta:.3f} d, asi0={asi0:.3f} d, "
            f"tsi0={tsi0:.3f} d outside the admissible range "
            f"[{tau_min:.2f}, {tau_max:.2f}] d")
    return tsi0


def single_coefficient_pair(cult_m, cult_f):
    """
    True when the two cultivars differ in exactly one flowering coefficient.

    Reported because PC0001-PC0005 share P2 (0.750) and PHINT (49.00), so pairs
    drawn within that family differ in P1 alone -- a pure thermal-time shift.
    Such a pair cannot express a difference in how the parents' development
    RATE responds to sowing date, which is what produces a steep dTSI/doffset.
    """
    a, b = CULTIVAR_COEFFS.get(cult_m), CULTIVAR_COEFFS.get(cult_f)
    if a is None or b is None:
        return False
    idx = {"P1": 1, "P2": 2, "PHINT": 4}
    diff = sum(1 for k in FLOWERING_COEFFS if abs(a[idx[k]] - b[idx[k]]) > 1e-12)
    return diff == 1


# --------------------------------------------------------------------------
# Table-driven quantities
# --------------------------------------------------------------------------

def _adat(ph, cult, year_i, sow_day):
    ci = ph._ci.get(cult)
    di = ph._di.get(int(sow_day))
    if ci is None or di is None:
        return np.nan
    v = ph.adat[ci, year_i, di]
    return float(v) if np.isfinite(v) else np.nan


def delta_series(ph, cult_m, cult_f, sow_day):
    """Delta = ADAT(m) - ADAT(f) at one sowing day, one value per weather year."""
    out = []
    for yi in range(ph.adat.shape[1]):
        a, b = _adat(ph, cult_m, yi, sow_day), _adat(ph, cult_f, yi, sow_day)
        out.append(a - b if np.isfinite(a) and np.isfinite(b) else np.nan)
    return np.asarray(out, dtype=float)


def grand_mean_delta(ph, cult_m, cult_f):
    """Delta averaged over every year and sowing day in the table."""
    ci, cj = ph._ci.get(cult_m), ph._ci.get(cult_f)
    if ci is None or cj is None:
        return np.nan
    return float(np.nanmean(ph.adat[ci]) - np.nanmean(ph.adat[cj]))


def slope_diagnostic(ph, cult_m, cult_f, sow_day, asi0=ASI0_DEFAULT,
                     offsets=range(-4, 7)):
    """
    dTSI/doffset for one pair: how fast TSI moves per day of male offset.

    DISCLOSED LIMITATION, not a gate (X7). Observed values are 1.00 d/d
    (UAS Bengaluru) and 1.58 d/d (Marahatta). A shallow slope means more offset
    days are needed to move TSI by a given amount, which changes how much
    staggering the optimiser must buy.
    """
    offs, tsis = [], []
    nyr = ph.adat.shape[1]
    for o in offsets:
        dm = int(sow_day) + int(o)
        if dm not in ph._di:
            continue
        vals = []
        for yi in range(nyr):
            a, b = _adat(ph, cult_m, yi, dm), _adat(ph, cult_f, yi, sow_day)
            if np.isfinite(a) and np.isfinite(b):
                vals.append((dm + a - asi0) - (int(sow_day) + b))
        if vals:
            offs.append(o)
            tsis.append(float(np.mean(vals)))
    if len(offs) < 3:
        return np.nan
    return float(np.polyfit(offs, tsis, 1)[0])


def pair_info(ph, cult_m, cult_f, sow_day, asi0=ASI0_DEFAULT):
    """Everything needed to admit, stratify and report one pair."""
    d_mean = grand_mean_delta(ph, cult_m, cult_f)
    d_years = delta_series(ph, cult_m, cult_f, sow_day)
    ok = np.isfinite(d_years)
    if not ok.any() or not np.isfinite(d_mean):
        return None
    t_years = d_years[ok] - asi0
    return PairInfo(
        cult_m=cult_m, cult_f=cult_f,
        delta_mean=d_mean,
        tsi0_mean=same_day_tsi_from_delta(d_mean, asi0),
        tsi0_min=float(t_years.min()),
        tsi0_max=float(t_years.max()),
        tsi0_sd=float(t_years.std()),
        slope=slope_diagnostic(ph, cult_m, cult_f, sow_day, asi0),
        asi0=float(asi0),
        single_coefficient=single_coefficient_pair(cult_m, cult_f),
        n_years=int(ok.sum()),
    )


def admissible_pairs(ph, sow_day, asi0=ASI0_DEFAULT, scope="years",
                     tau_min=TAU_MIN, tau_max=TAU_MAX):
    """
    Admissible ordered pairs, sorted by same-day TSI.

    scope "mean"  -- grand-mean TSI(0) within range (the superseded weaker test)
    scope "years" -- TSI(0) within range in EVERY weather year  (X6, required)
    """
    if scope not in ("mean", "years"):
        raise ValueError(f"unknown scope {scope!r}")
    out = []
    for cm in ph.cultivars:
        for cf in ph.cultivars:
            if cm == cf:
                continue
            p = pair_info(ph, cm, cf, sow_day, asi0)
            if p is None:
                continue
            if scope == "mean":
                ok = tau_min - 1e-9 <= p.tsi0_mean <= tau_max + 1e-9
            else:
                ok = (tau_min - 1e-9 <= p.tsi0_min
                      and p.tsi0_max <= tau_max + 1e-9)
            if ok:
                out.append(p)
    return sorted(out, key=lambda q: q.tsi0_mean)


def legacy_rule_violations(ph, lo=1.5, hi=14.0):
    """
    How many pairs the superseded ``1.5 <= abs(gap) <= 14`` rule admits that
    have a NEGATIVE nicking gap, i.e. the male silks before the female.

    Kept so the defect it fixes stays visible and testable rather than becoming
    folklore.
    """
    n = 0
    for cm in ph.cultivars:
        for cf in ph.cultivars:
            if cm == cf:
                continue
            g = grand_mean_delta(ph, cm, cf)
            if np.isfinite(g) and lo <= abs(g) <= hi and g < 0:
                n += 1
    return n


def stratified_pairs(ph, n_pairs, rng, sow_day, asi0=ASI0_DEFAULT,
                     scope="years", with_replacement=True):
    """
    Draw `n_pairs` pairs stratified on TSI(0) (P2-D3).

    Strata are equal-width slices of the admissible set ordered by same-day TSI,
    so every farm spans easy and hard crosses in the same proportions while two
    farm seeds give genuinely different farms.

    `with_replacement` defaults True because few pairs are admissible: with 13
    pairs a 40-field farm must reuse them. The reuse factor is reported by
    `population_report` as the independence caveat for the bootstrap.
    """
    cands = admissible_pairs(ph, sow_day, asi0, scope)
    if not cands:
        raise InadmissiblePair(
            f"no admissible pairs at sow_day={sow_day}, asi0={asi0}, scope={scope}")
    if not with_replacement and n_pairs > len(cands):
        raise InadmissiblePair(
            f"{n_pairs} pairs requested without replacement but only "
            f"{len(cands)} are admissible")
    edges = np.linspace(0, len(cands), n_pairs + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        b = max(b, a + 1)
        out.append(cands[int(rng.integers(a, min(b, len(cands))))])
    return out


def population_report(ph, sow_day, asi0=ASI0_DEFAULT, n_fields=20):
    """
    T10 characterisation table. Must be produced for every run: a population
    whose properties are unreported can manufacture a result.
    """
    n_c = len(ph.cultivars)
    mean_set = admissible_pairs(ph, sow_day, asi0, "mean")
    year_set = admissible_pairs(ph, sow_day, asi0, "years")
    n_adm = len(year_set) or len(mean_set)
    slopes = [p.slope for p in year_set if np.isfinite(p.slope)]
    return {
        "sow_day": int(sow_day),
        "asi0": float(asi0),
        "tau_range": (TAU_MIN, TAU_MAX),
        "n_cultivars": n_c,
        "n_ordered_pairs": n_c * (n_c - 1),
        "n_admissible_mean": len(mean_set),
        "n_admissible_years": len(year_set),
        "n_single_coefficient_pairs": sum(1 for p in year_set if p.single_coefficient),
        "slope_observed": dict(SLOPE_OBSERVED),
        "slope_range": (min(slopes), max(slopes)) if slopes else (np.nan, np.nan),
        "slope_shortfall_vs_marahatta": (
            SLOPE_OBSERVED["marahatta"] - max(slopes) if slopes else np.nan),
        "tsi0_sd_target_target_1a": TSI0_SD_TARGET,
        "expected_repeats_per_pair": (n_fields / n_adm) if n_adm else np.nan,
        "distinct_P1": sorted({CULTIVAR_COEFFS[c][1] for c in ph.cultivars
                               if c in CULTIVAR_COEFFS}),
        "distinct_P2": sorted({CULTIVAR_COEFFS[c][2] for c in ph.cultivars
                               if c in CULTIVAR_COEFFS}),
        "distinct_PHINT": sorted({CULTIVAR_COEFFS[c][4] for c in ph.cultivars
                                  if c in CULTIVAR_COEFFS}),
        "pairs": [
            {"label": p.label(), "cult_m": p.cult_m, "cult_f": p.cult_f,
             "name_m": CULTIVAR_COEFFS.get(p.cult_m, ("?",))[0],
             "name_f": CULTIVAR_COEFFS.get(p.cult_f, ("?",))[0],
             "delta_mean": p.delta_mean, "tsi0_mean": p.tsi0_mean,
             "tsi0_min": p.tsi0_min, "tsi0_max": p.tsi0_max,
             "tsi0_sd": p.tsi0_sd, "slope": p.slope,
             "single_coefficient": p.single_coefficient}
            for p in year_set],
    }
