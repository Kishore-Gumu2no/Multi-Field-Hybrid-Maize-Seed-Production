"""
Relative yield response to the tasseling-silking interval.

WHAT THIS REPLACES
------------------
`sim.core.seed_yield` returns ABSOLUTE t/ha, normalises implicitly on Y(0),
clips at a 4.0 t/ha ceiling, and governs negative TSI with a rectangular
pollen-window overlap. Each of those is a defect established in Phase 1 and
Phase 2:

  * absolute yield is not comparable across field geometries. Marahatta's field
    was 4:1 female:male and ours is 2:1, and the per-hectare basis of the
    published figures is UNKNOWN. Only RELATIVE yield transfers.
  * Y(0) sits 12.8% above the best observed treatment mean, so normalising on
    it inflates every relative loss by about a sixth.
  * the rectangular overlap is falsified: its errors at the two observed
    negative-TSI points have OPPOSITE signs, so the curvature is wrong, and
    its asymmetry gap is 0.279 against an observed 0.036.

`sim.core.seed_yield` is left untouched so historical results stay reproducible.

FORM
----
    r(x) = reject                      x < -3.75
           r+(kappa * |x|)             -3.75 <= x < -p_minus
           1                           -p_minus <= x <= p_plus
           Y(x) / Y(p_plus)            p_plus < x <= 13.75
           reject                      x > 13.75

with Y the published quadratic clipped at its vertex, and r+ its plateau-
normalised positive branch.

EVIDENTIAL STATUS OF EACH PIECE
-------------------------------
  Y coefficients        VERIFIED. Transcribed correctly, and the curve
                        reproduces its own treatment means to RMSE 0.216 t/ha
                        (R2 0.893 on means) when checked against the TSI values
                        reconstructed from Marahatta Table 4.
  domain [-3.75, 13.75] VERIFIED as the union of observed TSI across the two
                        trials. Outside it the function REJECTS; it never clips.
  plateau width         CONTESTED between the two trials. Three scenarios, none
                        designated correct.
  kappa                 BOUNDED, never fitted. The ordering of the two UAS
                        observations requires kappa > 1.429; their magnitudes
                        imply 0.97 to 2.56 depending on plateau and disagree
                        with each other under every scenario, so a point
                        estimate would require pooling two datasets that differ
                        1.68x in decline rate.
  delta_thr             ASSUMED, bounded 0 to -0.5 d. Our TSI is on a 50%
                        flowering basis (DSSAT ISDATE) and Y was fitted on a
                        75% basis.

NOT DONE HERE, deliberately: no pooled regression, no curve fitted to the two
negative observations, no point estimate for kappa, no absolute tonnage.
"""

from dataclasses import dataclass, replace
from itertools import product

import numpy as np

# ---------------------------------------------------------------------------
# Published coefficients (Marahatta et al., Figure 4). VERIFIED transcription.
# ---------------------------------------------------------------------------

CURVES = {
    "winter_y1": dict(a=3.5918, b=-0.3379, c=0.012,  r2=0.5765),
    "spring_y2": dict(a=2.9076, b=-0.2608, c=0.0169, r2=0.3581),
}

EXCLUSION_REASONS = {
    "winter_y2": (
        "removed from the ensemble: vertex at 2.06 d makes the curve flat at "
        "0.949 of optimum for every TSI above 2, asserting that a ten-day "
        "desynchronisation costs 5% of yield. Its own source data (Marahatta "
        "Table 6) shows winter yield falling from 3.00 to 1.25 t/ha between "
        "TSI 1.00 and 13.75 d, a 58% loss. R2 0.2907 with a large curvature "
        "term: an artefact of fitting a quadratic to scattered plot data."),
    "spring_y1": (
        "excluded: as published, b = -0.01532 with c = +0.0086 puts the vertex "
        "at 0.89 d, so the curve RISES across almost its whole observed range, "
        "contradicting the paper's own text (line 444: a consistent decline in "
        "seed yield with increasing TSI). A single-character correction to "
        "-0.1532 would place the vertex at 8.91 d and make it consistent, "
        "which is SUPPORTED but unconfirmed. Excluded rather than silently "
        "repaired."),
}

# Observed TSI domain: union across the two trials. DERIVED.
TSI_LO, TSI_HI = -3.75, 13.75

# Plateau scenarios. CONTESTED; none designated correct.
PLATEAUS = {
    "PL0": (0.0, 0.0),   # no plateau; closest to the UAS data
    "PL1": (1.0, 1.0),   # symmetric +/-1 d; a compromise with no direct support
    "PL2": (0.0, 2.0),   # one-sided, from Marahatta's same-day treatments
}

# kappa: hard floor from the ORDERING of the two UAS observations. DERIVED.
#   kappa * 1.75 > 2.50  ->  kappa > 1.4286
#   kappa * 3.75 > 4.00  ->  kappa > 1.0667
KAPPA_FLOOR = 2.50 / 1.75
KAPPA_LEVELS = (1.5, 2.0, 2.5)

# Flowering-basis correction, 50% (our TSI) to 75% (the fitted curve). ASSUMED.
DELTA_THR_LEVELS = (0.0, -0.28, -0.5)

# Falsified rectangular-overlap parameters, retained ONLY for the E9 comparison.
POLLEN_SHED_DAYS = 7.0
SILK_RECEPTIVE_DAYS = 6.0


class CurveNotAvailable(KeyError):
    """Raised for a curve that was removed or excluded, with the reason."""


class OutOfDomain(ValueError):
    """Raised when TSI falls outside the observed domain. Never clipped."""


class KappaBelowFloor(ValueError):
    """Raised when kappa would break an observed ordering."""


# ---------------------------------------------------------------------------
# Scenario coordinates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    curve: str = "winter_y1"
    plateau: str = "PL2"
    kappa: float = 1.5
    delta_thr: float = 0.0
    _unchecked: bool = False        # test-only escape hatch; never used in runs

    def validate(self):
        if self.curve not in CURVES:
            reason = EXCLUSION_REASONS.get(self.curve, "unknown curve")
            raise CurveNotAvailable(f"{self.curve}: {reason}")
        if self.plateau not in PLATEAUS:
            raise ValueError(f"unknown plateau scenario {self.plateau!r}")
        if not self._unchecked and self.kappa <= KAPPA_FLOOR:
            raise KappaBelowFloor(
                f"kappa={self.kappa} is at or below the floor "
                f"{KAPPA_FLOOR:.4f} implied by the ordering of the two UAS "
                f"observations; it would permit r(-1.75) >= r(+2.50)")
        return self

    def coords(self):
        """Scenario coordinates for the result row (P2-D12, X-corrections)."""
        return dict(curve=self.curve, plateau=self.plateau,
                    kappa=float(self.kappa), delta_thr=float(self.delta_thr))


PRIMARY_SCENARIO = Scenario(curve="winter_y1", plateau="PL2", kappa=1.5,
                            delta_thr=0.0)


def scenario_grid(curves=tuple(CURVES), plateaus=tuple(PLATEAUS),
                  kappas=KAPPA_LEVELS, delta_thrs=(0.0,)):
    """
    The declared scenario grid. 2 curves x 3 plateaus x 3 kappa = 18 cells.
    Every reported result carries its cell; no result is a single number.
    """
    return [Scenario(c, p, k, d).validate()
            for c, p, k, d in product(curves, plateaus, kappas, delta_thrs)]


# ---------------------------------------------------------------------------
# The curve
# ---------------------------------------------------------------------------

def _coeffs(curve):
    if curve not in CURVES:
        raise CurveNotAvailable(
            f"{curve}: {EXCLUSION_REASONS.get(curve, 'unknown curve')}")
    p = CURVES[curve]
    return p["a"], p["b"], p["c"]


def vertex(curve="winter_y1"):
    a, b, c = _coeffs(curve)
    return -b / (2.0 * c)


def curve_absolute(x, curve="winter_y1"):
    """
    The published quadratic in t/ha, clipped at its vertex so yield is
    non-increasing in TSI. Exposed for verification against the paper's own
    numbers ONLY -- never used as an objective, because the per-hectare basis
    is UNKNOWN and the field geometry differs from ours.
    """
    a, b, c = _coeffs(curve)
    xx = min(max(float(x), 0.0), vertex(curve))
    return a + b * xx + c * xx * xx


def _positive_branch(u, curve, p_plus):
    """r+(u) for u >= 0: 1 on the plateau, then normalised on its edge."""
    if u <= p_plus:
        return 1.0
    return curve_absolute(u, curve) / curve_absolute(p_plus, curve)


def relative_yield(x, scenario=PRIMARY_SCENARIO):
    """
    Relative yield factor in (0, 1]. Raises `OutOfDomain` outside the observed
    TSI range -- a silently clipped value would score a schedule in a region no
    experiment has visited.
    """
    sc = scenario
    x = float(x) + float(sc.delta_thr)
    if not (TSI_LO - 1e-9 <= x <= TSI_HI + 1e-9):
        raise OutOfDomain(
            f"TSI {x:.3f} d is outside the observed domain "
            f"[{TSI_LO}, {TSI_HI}] d; no experiment covers it, so the "
            f"response is undefined rather than extrapolated")
    p_minus, p_plus = PLATEAUS[sc.plateau]
    if x >= -p_minus:
        return _positive_branch(x, sc.curve, p_plus)
    return _positive_branch(sc.kappa * abs(x), sc.curve, p_plus)


def relative_yield_checked(x, scenario=PRIMARY_SCENARIO):
    """
    Non-raising variant for the execution path, which must FLAG rather than
    abort: returns (value, in_domain). Out of domain gives (nan, False) so the
    caller records an out-of-domain field-year instead of scoring it.
    """
    try:
        return relative_yield(x, scenario), True
    except OutOfDomain:
        return float("nan"), False


def ensemble_relative_yield(tsi_values, scenario=PRIMARY_SCENARIO):
    """
    Mean of r(TSI) over weather years -- the FACTOR is averaged, never TSI
    (P2-D16). Expected yield is E[r(TSI)], not r(E[TSI]).

    CORRECTION made in Phase 3: the published quadratic has c = +0.012, so r is
    CONVEX on its positive branch. Phase 2 issue I4 and the Phase 2 Supplement
    both gave the bias direction backwards. By Jensen, averaging TSI first
    OVERSTATES the penalty (r(E[TSI]) <= E[r(TSI)]). The requirement is
    unchanged; only the stated reason was wrong.

    Returns (mean_r_over_in_domain_years, n_in_domain, n_total).
    """
    vals, n_ok = [], 0
    arr = np.asarray(tsi_values, dtype=float)
    for t in arr:
        if not np.isfinite(t):
            continue
        v, ok = relative_yield_checked(float(t), scenario)
        if ok:
            vals.append(v)
            n_ok += 1
    mean = float(np.mean(vals)) if vals else float("nan")
    return mean, n_ok, int(arr.size)


# ---------------------------------------------------------------------------
# Superseded form, retained for the E9 comparison only
# ---------------------------------------------------------------------------

def rectangular_overlap_relative(x, d_m=POLLEN_SHED_DAYS,
                                 d_f=SILK_RECEPTIVE_DAYS):
    """
    The superseded rectangular overlap, normalised as `sim.core.seed_yield`
    normalised it. FALSIFIED: errors at the two observed negative-TSI points
    have opposite signs (-0.159 at -3.75 d, +0.093 at -1.75 d), and the
    asymmetry gap is 0.279 against an observed 0.036.

    Kept so the E9 comparison reports what the old model actually gave rather
    than a reconstruction from memory.
    """
    def ov(t):
        return max(0.0, min(t + d_m, d_f) - max(t, 0.0))
    base = ov(0.0)
    return ov(float(x)) / base if base > 0 else float("nan")
