"""
Density-weighted pollen-silk overlap, with both male sowings.

WHY NOT THE RECTANGULAR MODEL
-----------------------------
`sim.core._overlap_days` intersects two uniform windows. Tested against the two
observed negative-TSI points its errors have OPPOSITE signs (-0.159 at
TSI -3.75 d, +0.093 at -1.75 d), so the defect is curvature, not calibration,
and retuning the two widths cannot fix it. A uniform window treats the first
and last day of shed as worth as much as the peak, which under-penalises small
misalignments and over-penalises large ones -- exactly the observed pattern.

WHY SM1 WAS WITHDRAWN  (Supplement X1)
--------------------------------------
Splitting the male across two dates CONSERVES total pollen but halves its peak.
The pollination term 1 - exp(-lambda*P) is concave, so its marginal value is
highest where pollen is scarce. Splitting therefore moves pollen from wherever
the single peak sat to a point `g` days later:

  * if the single peak already coincided with silking, pollen moves from the
    highest-value region to a lower one and O FALLS;
  * if the male was early, or the silking window is much wider than the shed
    window, the second pulse lands where pollen was scarce and O RISES.

Splitting is a mean-preserving spread of P(t), so by Jensen's inequality no
parameterisation makes "split >= single" a theorem, and a model in which it
were one would assert that a well-nicked pair benefits from having its pollen
peak halved. DERIVED: the proposed model violates the old SM1 in 16 of 24
configurations, worst at TSI = 0 with g = 5 where split is 38% worse.

Split sowing is therefore a HEDGE against phenological uncertainty, which is
what CIMMYT's conclusions actually argue for ("to guarantee that in the seed
production field there will be pollen of T10 during the entire female flowering
period"). Its value should shrink or reverse as information improves -- a
falsifiable prediction that the old SM1 would have made untestable.

PARAMETER STATUS
----------------
  w_m   7-14 d, use 7   VERIFIED AS REPORTED  Marahatta section 1
  w_f   7-14 d, use 7   VERIFIED AS REPORTED  same sentence
  d_s   2-3 d, use 3    VERIFIED AS REPORTED  "2-3 days receptive window of
                                              the stigma". Note this is the
                                              PER-SILK window; stand-level
                                              receptivity is the convolution of
                                              emergence with it, which is why
                                              the superseded 6.0 d constant was
                                              defensible but undocumented.
  phi   0.5             VERIFIED  CIMMYT 4:2 with one male row per date
  g     {2, 3, 5} d     ASSUMED   CIMMYT's criterion is phenological
                                  (half-emerged coleoptile), not a day count
  alpha_p, beta_p       ASSUMED   no source gives the shed shape; sweep the
                                  mode position over {1/4, 1/3, 1/2}
  lam                   ASSUMED   no source; sweep over a decade
  eta_v                 NOT IDENTIFIABLE from lam; folded into it

Four of these have no empirical value and one is not identifiable. That is why
`sim.response` keeps the simple kappa-reflection as the DEFAULT negative branch
and this model is the robustness alternative (P2-D13, E9).
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as _beta


@dataclass(frozen=True)
class PollenParams:
    w_m: float = 7.0          # pollen shed duration, d
    w_f: float = 7.0          # silk emergence spread, d
    d_s: float = 3.0          # per-silk receptive duration, d
    alpha_p: float = 2.0      # shed shape; mode at (a-1)/(a+b-2) = 1/3
    beta_p: float = 3.0
    lam: float = 3.0          # saturation constant, folds in eta_v
    phi: float = 0.5          # first male sowing's share of male area
    step: float = 0.01        # integration step, d
    span: float = 40.0        # half-width of the integration window, d

    def axis(self):
        return np.arange(-self.span, self.span, self.step)

    def coords(self):
        return dict(w_m=self.w_m, w_f=self.w_f, d_s=self.d_s,
                    alpha_p=self.alpha_p, beta_p=self.beta_p,
                    lam=self.lam, phi=self.phi)


DEFAULT = PollenParams()


def _shed(t, t0, pp):
    """Unit-mass pollen density from one male sowing whose anthesis is at t0."""
    u = (t - t0) / pp.w_m
    out = np.zeros_like(t)
    m = (u > 0.0) & (u < 1.0)
    out[m] = _beta.pdf(u[m], pp.alpha_p, pp.beta_p) / pp.w_m
    return out


def silk_receptivity(t, pp=DEFAULT):
    """
    Stand receptivity: silk emergence spread over w_f, convolved with the
    per-silk receptive duration d_s. This separates the two quantities the
    superseded single 6.0 d constant conflated.
    """
    u = t / pp.w_f
    e = np.zeros_like(t)
    m = (u > 0.0) & (u < 1.0)
    e[m] = _beta.pdf(u[m], pp.alpha_p, pp.beta_p) / pp.w_f
    k = np.ones(max(1, int(round(pp.d_s / pp.step))))
    return np.convolve(e, k, mode="same") * pp.step


def total_pollen(t, tsi, gap=None, pp=DEFAULT):
    """
    Pollen available at time t, origin at female silking onset.

    `tsi` positions the FIRST male anthesis. `gap` adds a second sowing phi/
    (1-phi) days later; None means a single sowing carrying all the male area.
    """
    if gap is None:
        return _shed(t, tsi, pp)
    return (pp.phi * _shed(t, tsi, pp)
            + (1.0 - pp.phi) * _shed(t, tsi + gap, pp))


def pollen_mass(tsi, gap=None, pp=DEFAULT):
    """Total pollen mass. Invariant SP1: identical for split and single."""
    t = pp.axis()
    return float(np.trapezoid(total_pollen(t, tsi, gap, pp), t))


def overlap_fraction(tsi, gap=None, pp=DEFAULT):
    """
    Fraction of silks pollinated, in [0, 1].

    Saturating: beyond a certain load more pollen fertilises no more ovules.
        O = int s(t) [1 - exp(-lam P(t))] dt  /  int s(t) dt
    """
    t = pp.axis()
    s = silk_receptivity(t, pp)
    den = float(np.trapezoid(s, t))
    if den <= 0.0:
        return float("nan")
    P = total_pollen(t, tsi, gap, pp)
    num = float(np.trapezoid(s * (1.0 - np.exp(-pp.lam * P)), t))
    return num / den


def overlap_relative(tsi, gap=None, pp=DEFAULT):
    """
    Overlap normalised on the perfectly synchronous single-sowing case, so it
    is comparable with `sim.response.relative_yield`.
    """
    base = overlap_fraction(0.0, None, pp)
    if not np.isfinite(base) or base <= 0.0:
        return float("nan")
    return overlap_fraction(tsi, gap, pp) / base


def split_advantage(tsi, gap, pp=DEFAULT):
    """
    O(split) - O(single) at the same first male anthesis.

    SIGNED, and expected to be negative for a well-positioned male. Never
    constrained to be non-negative: that was the withdrawn SM1.
    """
    return overlap_fraction(tsi, gap, pp) - overlap_fraction(tsi, None, pp)
