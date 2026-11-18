"""
T14 -- the CERES soil-temperature pathway.

These pin the arithmetic read out of MZ_PHENOL.for, because the whole sensing
experiment rests on it. The clamp-order test is the one that matters: getting
it wrong is silent and changes every number downstream.
"""

import numpy as np
import pytest

from sim import soilthermal as ST


def test_constants_match_the_shipped_ecotype_file():
    # MZCER048.ECO, every standard maize ecotype
    assert ST.TBASE == 8.0
    assert ST.TOPT == 34.0 and ST.ROPT == 34.0
    assert ST.DTT_CAP == pytest.approx(26.0)


def test_acoef_matches_the_source_line():
    # MZ_PHENOL.for line 470
    assert float(ST.acoef(0.0)) == pytest.approx(0.5902)
    assert float(ST.acoef(18.0)) == pytest.approx(0.01061 * 18.0 + 0.5902)


def test_soil_pair_blends_match_the_source():
    td, tn = ST.soil_pair(np.array([30.0]), np.array([20.0]), np.array([18.0]))
    td, tn = td[0], tn[0]
    a = 0.01061 * 18.0 + 0.5902
    assert float(td) == pytest.approx(a * 30.0 + (1 - a) * 20.0)
    assert float(tn) == pytest.approx(0.36354 * 30.0 + 0.63646 * 20.0)


def test_clamp_order_is_component_then_average():
    """
    The distinction that is silent if wrong. A 40/20 pair: TDSOIL capped at
    DOPT=34 first gives (20+34)/2 - 8 = 19.0. Averaging first would give
    min((40+20)/2, 34) - 8 = 22.0, which must NOT happen.
    """
    got = float(ST.dtt_from_pair(np.array([40.0]), np.array([20.0]))[0])
    assert got == pytest.approx(19.0)
    assert got != pytest.approx(22.0)


def test_night_is_floored_before_averaging():
    got = float(ST.dtt_from_pair(np.array([20.0]), np.array([2.0]))[0])
    assert got == pytest.approx((8.0 + 20.0) / 2.0 - 8.0)


def test_cold_day_scores_zero():
    assert float(ST.dtt_from_pair(np.array([5.0]), np.array([1.0]))[0]) == 0.0


def test_daily_cap():
    assert float(ST.dtt_from_pair(np.array([60.0]), np.array([60.0]))[0]) == pytest.approx(ST.DTT_CAP)


def test_offset_raises_thermal_time_one_for_one_when_no_clamp_binds():
    """
    The relation the whole experiment depends on: a sustained offset of d degC
    adds d degree-days a day, because DTT is the mean of the two clamped
    components less TBASE.
    """
    args = (np.array([30.0]), np.array([20.0]), np.array([18.0]))
    base = float(ST.soil_dtt(*args, offset_c=0.0)[0])
    for d in (0.5, 1.0, 2.0):
        assert float(ST.soil_dtt(*args, offset_c=d)[0]) == pytest.approx(base + d)


def test_offset_is_attenuated_once_the_cap_binds():
    """Above DOPT the cap eats the signal. This is a real limitation, not a bug."""
    args = (np.array([40.0]), np.array([30.0]), np.array([25.0]))
    base = float(ST.soil_dtt(*args, offset_c=0.0)[0])
    assert float(ST.soil_dtt(*args, offset_c=3.0)[0]) - base < 3.0


def test_days_to_threshold_counts_from_the_sowing_day():
    dtt = np.full(50, 10.0)
    cum = np.concatenate([[0.0], np.cumsum(dtt)])
    # from day index 0, 100 GDD at 10/day is reached at the end of day 9
    assert ST.days_to_threshold(cum, 0, 100.0) == pytest.approx(9.0)
    # the same accumulation starting later takes the same number of days
    assert ST.days_to_threshold(cum, 10, 100.0) == pytest.approx(9.0)


def test_days_to_threshold_is_nan_when_never_reached():
    cum = np.concatenate([[0.0], np.cumsum(np.full(10, 1.0))])
    assert np.isnan(ST.days_to_threshold(cum, 0, 1e6))


def _toy_provider(offset_years=(2020,)):
    n = 400
    seasons = {y: ST.SeasonWeather(year=y,
                                   tmax=np.full(n, 30.0),
                                   tmin=np.full(n, 20.0),
                                   srad=np.full(n, 18.0))
               for y in offset_years}
    return ST.SoilThermalPhenology({"M": 500.0, "F": 450.0}, seasons)


def test_warmer_field_flowers_earlier():
    p = _toy_provider()
    cool = p.adat_day("F", 2020, 100, 0.0)
    warm = p.adat_day("F", 2020, 100, 1.5)
    assert warm < cool


def test_tsi_callable_matches_tsi_by_year():
    p = _toy_provider()
    fn = p.tsi_callable("M", "F", offset_c=0.8, asi0=5.0)
    a = fn(100, 103)
    b = p.tsi_by_year("M", "F", 100, 103, offset_c=0.8, asi0=5.0)
    assert np.allclose(a, b, equal_nan=True)


def test_unknown_cultivar_is_nan_not_an_exception():
    p = _toy_provider()
    assert np.isnan(p.adat_day("NOPE", 2020, 100, 0.0))


def test_pathway_argument_is_validated():
    with pytest.raises(ValueError):
        ST.SoilThermalPhenology({}, {}, pathway="magnetic")
