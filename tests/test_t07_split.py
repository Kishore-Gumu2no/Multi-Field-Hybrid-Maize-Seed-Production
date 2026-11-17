"""
T7 -- split male sowing. SM1 is WITHDRAWN (Supplement X1).

SP1-SP4 are correctness invariants: they hold by construction and test the
implementation, not an outcome. SP5-SP6 are SIGNED directional predictions with
fixed configurations, verified rather than imposed. T7.2e (SP6) is BLOCKING: a
model that cannot show split sowing to be harmful has experiment E3's answer
built into it.
"""
import numpy as np
import pytest

from sim import pollen as P
from sim import response as R
from sim.scoring import SowingPlan, constant_tsi_fn, score_plan


# ------------------------------------------------------------------- SP2 / T7.1

@pytest.mark.parametrize("gap", [2.0, 3.0, 5.0])
def test_t7_1_sp2_phi_one_reproduces_single_sowing(gap):
    pp = P.PollenParams(phi=1.0)
    for tsi in (-3.0, -1.0, 0.0, 2.0, 4.0):
        a = P.overlap_fraction(tsi, gap, pp)
        b = P.overlap_fraction(tsi, None, pp)
        assert abs(a - b) <= 1e-12, (tsi, gap, a, b)


# ------------------------------------------------------------------ SP1 / T7.2a

@pytest.mark.parametrize("gap", [0.0, 2.0, 3.0, 5.0, 12.0])
@pytest.mark.parametrize("phi", [0.25, 0.5, 0.75])
def test_t7_2a_sp1_mass_conservation(gap, phi):
    """Splitting divides the same male area; total pollen must not change."""
    pp = P.PollenParams(phi=phi)
    single = P.pollen_mass(0.0, None, pp)
    split = P.pollen_mass(0.0, gap, pp)
    assert abs(split - single) / single <= 1e-6, (gap, phi, single, split)


# ------------------------------------------------------------------ SP3 / T7.2b

def test_t7_2b_sp3_zero_gap_reproduces_single_sowing():
    pp = P.DEFAULT
    for tsi in (-3.0, -1.0, 0.0, 2.0, 4.0):
        a = P.overlap_fraction(tsi, 0.0, pp)
        b = P.overlap_fraction(tsi, None, pp)
        assert abs(a - b) <= 1e-9, (tsi, a, b)


# ------------------------------------------------------------------ SP4 / T7.2c

def test_t7_2c_sp4_large_gap_converges_to_phi_weighted_single():
    """
    As the second pulse leaves the receptive window, O must converge to the
    value from the first sowing's phi share alone, monotonically.
    """
    pp = P.DEFAULT
    tsi = 0.0
    target = P.overlap_fraction(tsi, None, P.PollenParams(
        phi=1.0, lam=pp.lam * pp.phi))
    gaps = [10.0, 20.0, 30.0, 40.0, 60.0]
    vals = [P.overlap_fraction(tsi, g, pp) for g in gaps]
    diffs = [abs(v - target) for v in vals]
    assert all(b <= a + 1e-9 for a, b in zip(diffs, diffs[1:])), diffs
    assert diffs[-1] <= 1e-3, (vals[-1], target)


# ------------------------------------------------------------------ SP5 / T7.2d

def test_t7_2d_sp5_split_helps_when_the_male_is_early():
    """TSI = -2, w_f = 7, g = 2: the second pulse arrives in time for the silks."""
    adv = P.split_advantage(-2.0, 2.0, P.DEFAULT)
    assert adv > 0.0, f"expected split to help at TSI -2, got {adv:+.4f}"


# ------------------------------------------------------------------ SP6 / T7.2e

def test_t7_2e_sp6_split_HURTS_when_the_male_is_aligned():
    """
    BLOCKING. TSI = 0, w_f = 7, g = 5: the single peak already coincides with
    silking, so halving it and delivering the rest after the silks have closed
    must REDUCE pollination.

    If this fails, either pollen mass is not conserved or the pollination
    response is not concave -- and the model would have E3's answer built in.
    """
    adv = P.split_advantage(0.0, 5.0, P.DEFAULT)
    assert adv < 0.0, f"model cannot represent split sowing as harmful: {adv:+.4f}"
    assert adv < -0.10, f"expected a large penalty, got {adv:+.4f}"


def test_sp5_sp6_are_a_matched_pair_both_signs_reachable():
    signs = {np.sign(P.split_advantage(t, g, P.DEFAULT))
             for t, g in ((-2.0, 2.0), (0.0, 5.0))}
    assert signs == {1.0, -1.0}, signs


# ------------------------------------------------------------------------ T7.3

def test_t7_3_visit_days_merge_when_male_coincides_with_female():
    assert SowingPlan(10, 10, 12).visit_days == (10, 12)        # 2 visit-days
    assert SowingPlan(10, 13, 15).visit_days == (10, 13, 15)    # 3 visit-days
    assert SowingPlan(10, 10, None).visit_days == (10,)         # 1 visit-day
    assert SowingPlan(10, 8, 10).visit_days == (8, 10)          # m2 merges


# ------------------------------------------------------------------------ T7.4

def test_t7_4_both_male_dates_enter_the_score():
    """
    C11: the superseded execute() scored from the first male sowing only, so
    split sowing was costed and not credited.
    """
    fn = constant_tsi_fn([0.0] * 8)
    single = score_plan(fn, 2.0, SowingPlan(10, 10, None))
    split = score_plan(fn, 2.0, SowingPlan(10, 10, 15))
    assert single.credited_second_sowing is False
    assert split.credited_second_sowing is True
    assert split.r_mean != single.r_mean, "second sowing changed nothing"
    assert split.split_ratio != 1.0


def test_t7_4b_split_is_permitted_to_reduce_yield():
    """The withdrawn SM1 would have forbidden this. It must be reachable."""
    fn = constant_tsi_fn([0.0] * 8)
    single = score_plan(fn, 2.0, SowingPlan(10, 10, None))
    split = score_plan(fn, 2.0, SowingPlan(10, 10, 15))
    assert split.r_mean < single.r_mean


def test_t7_4c_reflect_mode_does_not_credit_the_second_sowing():
    fn = constant_tsi_fn([0.0] * 8)
    a = score_plan(fn, 2.0, SowingPlan(10, 10, 15), mode="reflect")
    b = score_plan(fn, 2.0, SowingPlan(10, 10, None), mode="reflect")
    assert a.r_mean == b.r_mean
    assert a.credited_second_sowing is False


# ------------------------------------------------- ensemble and domain handling

def test_ensemble_averages_the_response_not_the_tsi():
    """
    P2-D16. Expected yield is E[r(TSI)], never r(E[TSI]); the two differ
    whenever the ensemble is spread and r is non-linear.

    DIRECTION, corrected in Phase 3: the published quadratic has c = +0.012,
    so Y and therefore r are CONVEX on the positive branch, not concave.
    Phase 2 issue I4 and the Supplement both stated the bias direction
    backwards. By Jensen, for convex r:  E[r(TSI)] >= r(E[TSI]), so averaging
    TSI first OVERSTATES the penalty rather than understating it. The
    requirement to average the response is unchanged -- only the stated reason
    was wrong.
    """
    tsis = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    sc = R.PRIMARY_SCENARIO
    mean_of_r = float(np.mean([R.relative_yield(t, sc) for t in tsis]))
    r_of_mean = R.relative_yield(float(np.mean(tsis)), sc)
    assert mean_of_r != pytest.approx(r_of_mean, abs=1e-6)
    assert mean_of_r > r_of_mean, (mean_of_r, r_of_mean)


def test_response_is_convex_on_the_positive_branch():
    """Records the corrected curvature claim as a standing fact."""
    sc = R.Scenario(curve="winter_y1", plateau="PL0", kappa=1.5)
    xs = np.arange(0.0, R.TSI_HI - 0.5, 0.25)
    ys = np.array([R.relative_yield(float(x), sc) for x in xs])
    second = np.diff(ys, 2)
    assert (second >= -1e-12).all(), second[second < 0][:5]


def test_out_of_domain_years_are_excluded_and_counted_not_clipped():
    fn = constant_tsi_fn([0.0, 0.0, -9.0, 20.0, 1.0])
    s = score_plan(fn, 2.0, SowingPlan(10, 10, None))
    assert s.n_years == 5
    assert s.n_in_domain == 3
    assert s.out_of_domain == 2
    assert np.isfinite(s.r_mean)
