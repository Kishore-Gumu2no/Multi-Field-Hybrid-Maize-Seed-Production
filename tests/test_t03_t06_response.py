"""
T3 -- UAS Bengaluru regression tests (starred: failure means the model
      contradicts published data).
T4 -- Marahatta response-curve checks.
T5 -- male-early asymmetry.
T6 -- hard rejection of out-of-domain TSI.
"""
import numpy as np
import pytest

from sim import response as R

# Observed relative seed yield, UAS Bengaluru 2019, MAH 14-5 (Mysore J. Agric.
# Sci. 53(3):59-63). DERIVED TSI from the paper's own-sowing 50% dates.
UAS = {-3.75: 0.701, -1.75: 0.782, 0.00: 1.000, 2.50: 0.849, 4.00: 0.737}

# Marahatta winter year 1: observed hybrid-mean seed yield (t/ha) against
# DERIVED observed TSI reconstructed from Table 4.
MAR = {1.00: 3.00, 5.00: 2.53, 6.25: 1.87, 13.75: 1.25}


# ============================================================= T3  (starred)

def test_t3_1_r_at_zero_is_one():
    for sc in R.scenario_grid():
        assert abs(R.relative_yield(0.0, sc) - 1.0) <= 1e-12, sc


def test_t3_2_ordering_minus1p75_below_plus2p50_every_scenario():
    for sc in R.scenario_grid():
        lo, hi = R.relative_yield(-1.75, sc), R.relative_yield(2.50, sc)
        assert lo < hi, f"{sc}: r(-1.75)={lo:.4f} not < r(+2.50)={hi:.4f}"


def test_t3_3_ordering_minus3p75_below_plus4p00_every_scenario():
    for sc in R.scenario_grid():
        lo, hi = R.relative_yield(-3.75, sc), R.relative_yield(4.00, sc)
        assert lo < hi, f"{sc}: r(-3.75)={lo:.4f} not < r(+4.00)={hi:.4f}"


def test_t3_4_kappa_below_floor_breaks_the_ordering():
    """
    The 1.429 floor must be load-bearing, not decorative. At kappa = 1.40 the
    first ordering must FAIL, which is what justifies the floor.
    """
    with pytest.raises(R.KappaBelowFloor):
        R.Scenario(curve="winter_y1", plateau="PL0", kappa=1.40).validate()
    sc = R.Scenario(curve="winter_y1", plateau="PL0", kappa=1.40, _unchecked=True)
    assert R.relative_yield(-1.75, sc) >= R.relative_yield(2.50, sc)


def test_t3_5_some_scenario_is_within_tolerance_of_all_five_uas_points():
    """
    Deliberately weak (+/-0.10). The two datasets disagree 1.68x in decline
    rate, so a tight criterion would force the pooling P1-D8 forbids. The test
    only asks that the functional family can reach the observations at all.
    """
    best, best_err = None, np.inf
    for sc in R.scenario_grid():
        err = max(abs(R.relative_yield(x, sc) - y) for x, y in UAS.items())
        if err < best_err:
            best, best_err = sc, err
    assert best_err <= 0.10, f"closest scenario {best} is off by {best_err:.3f}"


# ============================================================= T4  (starred)

def test_t4_1_absolute_curve_values():
    exp = {1.00: 3.266, 5.00: 2.202, 6.25: 1.949, 13.75: 1.214}
    for x, y in exp.items():
        assert abs(R.curve_absolute(x, "winter_y1") - y) <= 0.001, x


def test_t4_2_curve_reproduces_its_own_treatment_means():
    pred = np.array([R.curve_absolute(x, "winter_y1") for x in MAR])
    obs = np.array(list(MAR.values()))
    rmse = float(np.sqrt(((obs - pred) ** 2).mean()))
    r2 = 1.0 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
    assert abs(rmse - 0.216) <= 0.005, rmse
    assert abs(r2 - 0.893) <= 0.005, r2


def test_t4_3_vertex():
    assert abs(R.vertex("winter_y1") - 14.08) <= 0.01


def test_t4_4_plateau_edge_is_the_normaliser_not_y0():
    sc = R.Scenario(curve="winter_y1", plateau="PL2", kappa=1.5)
    assert abs(R.relative_yield(2.0, sc) - 1.0) <= 1e-12
    assert abs(R.curve_absolute(2.0, "winter_y1") - 2.964) <= 0.001
    # Y(0) is 12.8% above the best observed treatment mean, so it must NOT be
    # the normaliser (Phase 1 F15).
    assert abs(R.curve_absolute(0.0, "winter_y1") - 3.5918) <= 1e-9


@pytest.mark.parametrize("bad", ["winter_y2", "spring_y1"])
def test_t4_5_t4_6_removed_curves_raise(bad):
    with pytest.raises(R.CurveNotAvailable) as e:
        R.curve_absolute(1.0, bad)
    assert R.EXCLUSION_REASONS[bad][:20].lower() in str(e.value).lower()
    with pytest.raises(R.CurveNotAvailable):
        R.Scenario(curve=bad, plateau="PL0", kappa=1.5).validate()


def test_t4_7_r_is_monotone_non_increasing_above_the_plateau():
    for sc in R.scenario_grid():
        xs = np.arange(R.PLATEAUS[sc.plateau][1], R.TSI_HI + 1e-9, 0.25)
        ys = [R.relative_yield(float(x), sc) for x in xs]
        assert all(b <= a + 1e-12 for a, b in zip(ys, ys[1:])), sc


# ===================================================================== T5

def test_t5_1_male_early_never_better_than_male_late():
    for sc in R.scenario_grid():
        for u in np.arange(0.25, 3.75 + 1e-9, 0.25):
            lo = R.relative_yield(-float(u), sc)
            hi = R.relative_yield(float(u), sc)
            assert lo <= hi + 1e-12, f"{sc} u={u}: r(-u)={lo} > r(+u)={hi}"


def test_t5_2_kappa_one_gives_symmetry():
    sc = R.Scenario(curve="winter_y1", plateau="PL0", kappa=1.0, _unchecked=True)
    for u in (0.5, 1.0, 2.0, 3.5):
        assert abs(R.relative_yield(-u, sc) - R.relative_yield(u, sc)) <= 1e-12


def test_t5_3_rectangular_overlap_reproduces_the_falsified_values():
    """
    Kept so the E9 comparison is honest: these are the values the superseded
    rectangular model gives, against observed 0.701 and 0.782.
    """
    assert abs(R.rectangular_overlap_relative(-3.75) - 0.542) <= 0.001
    assert abs(R.rectangular_overlap_relative(-1.75) - 0.875) <= 0.001


# ===================================================================== T6

@pytest.mark.parametrize("x", [-3.76, -4.5, -10.0, 13.76, 14.5, 20.0])
def test_t6_3_out_of_domain_raises_and_never_clips(x):
    sc = R.Scenario(curve="winter_y1", plateau="PL2", kappa=1.5)
    with pytest.raises(R.OutOfDomain) as e:
        R.relative_yield(x, sc)
    assert "-3.75" in str(e.value) and "13.75" in str(e.value)


def test_t6_3b_in_domain_edges_are_accepted():
    sc = R.Scenario(curve="winter_y1", plateau="PL2", kappa=1.5)
    for x in (R.TSI_LO, R.TSI_HI):
        v = R.relative_yield(x, sc)
        assert 0.0 < v <= 1.0


def test_t6_3c_safe_variant_reports_instead_of_raising():
    sc = R.Scenario(curve="winter_y1", plateau="PL2", kappa=1.5)
    v, ok = R.relative_yield_checked(-9.0, sc)
    assert ok is False and not np.isfinite(v)
    v, ok = R.relative_yield_checked(1.0, sc)
    assert ok is True and np.isfinite(v)


def test_grid_size_is_eighteen_cells():
    g = R.scenario_grid()
    assert len(g) == 18, len(g)
    assert {s.curve for s in g} == {"winter_y1", "spring_y2"}
    assert {s.plateau for s in g} == {"PL0", "PL1", "PL2"}
    assert {s.kappa for s in g} == {1.5, 2.0, 2.5}


def test_primary_scenario_is_named_and_in_the_grid():
    assert R.PRIMARY_SCENARIO in R.scenario_grid()
    assert R.PRIMARY_SCENARIO.curve == "winter_y1"
    assert R.PRIMARY_SCENARIO.plateau == "PL2"
    assert R.PRIMARY_SCENARIO.kappa == 1.5
