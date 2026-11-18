"""
T15 -- the simulated sensing layer.

The controls matter more than the capabilities here: a sensing layer that
cannot be shown to do nothing when there is nothing to detect cannot be
trusted when it reports something.
"""

import numpy as np
import pytest

from sim import sensing as SE
from sim import soilthermal as ST


def _season(n=400):
    return ST.SeasonWeather(year=2020, tmax=np.full(n, 30.0),
                            tmin=np.full(n, 20.0), srad=np.full(n, 18.0))


def test_default_sigma_is_the_value_already_in_the_repository():
    """
    sim.core.make_farm carries temp_bias=rng.normal(0.0, 0.7), committed long
    before this workstream. Using it as the default is what stops the headline
    scenario being tuned to suit the sensor.
    """
    assert SE.SIGMA_DEFAULT == 0.7


def test_default_noise_is_a_real_instrument_accuracy():
    # Oklahoma Mesonet TB05/TS05 thermistors, +/- 0.5 degC
    assert SE.NOISE_DEFAULT == 0.5


def test_homogeneous_scenario_is_exactly_zero():
    o = SE.field_offsets(12, SE.SensingConfig(scenario="homogeneous"))
    assert np.all(o == 0.0)


def test_sigma_zero_is_exactly_zero():
    o = SE.field_offsets(12, SE.SensingConfig(scenario="normal", sigma_c=0.0))
    assert np.all(o == 0.0)


def test_normal_scenario_has_about_the_requested_spread():
    o = SE.field_offsets(4000, SE.SensingConfig(sigma_c=1.0))
    assert abs(float(o.std()) - 1.0) < 0.05


def test_bimodal_has_two_groups_at_plus_and_minus_sigma():
    o = SE.field_offsets(12, SE.SensingConfig(scenario="bimodal", sigma_c=1.3))
    assert set(np.round(np.unique(o), 6)) == {-1.3, 1.3}


def test_gradient_spans_the_requested_range_monotonically():
    o = SE.field_offsets(5, SE.SensingConfig(scenario="gradient", sigma_c=2.0))
    assert o[0] == pytest.approx(-2.0) and o[-1] == pytest.approx(2.0)
    assert np.all(np.diff(o) > 0)


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        SE.field_offsets(3, SE.SensingConfig(scenario="telepathy"))


def test_noiseless_estimator_recovers_the_offset_exactly():
    truth = np.array([-1.5, 0.0, 0.8, 2.2])
    hat, n = SE.observe(_season(), truth,
                        SE.SensingConfig(noise_c=0.0, p_missing=0.0, n_obs_days=20))
    assert np.allclose(hat, truth, atol=1e-9)
    assert np.all(n == 20)


def test_estimator_is_unbiased_under_noise():
    truth = np.full(200, 1.0)
    hat, _ = SE.observe(_season(), truth,
                        SE.SensingConfig(noise_c=1.0, n_obs_days=20, seed=5))
    assert abs(float(np.mean(hat)) - 1.0) < 0.05


def test_estimator_error_shrinks_with_more_observation_days():
    truth = np.full(400, 1.0)
    errs = []
    for nd in (2, 40):
        hat, _ = SE.observe(_season(), truth,
                            SE.SensingConfig(noise_c=1.0, n_obs_days=nd, seed=3))
        errs.append(float(np.std(hat - truth)))
    assert errs[1] < errs[0] / 2.0


def test_total_missingness_yields_nan_not_a_fabricated_estimate():
    truth = np.array([1.0, -1.0])
    hat, n = SE.observe(_season(), truth, SE.SensingConfig(p_missing=1.0))
    assert np.all(np.isnan(hat)) and np.all(n == 0)


def test_partial_missingness_reduces_the_usable_count():
    truth = np.zeros(50)
    _, n = SE.observe(_season(), truth,
                      SE.SensingConfig(p_missing=0.5, n_obs_days=20, seed=9))
    assert 0 < float(n.mean()) < 20


def test_observation_window_closes_before_the_sowing_window_opens():
    """Non-anticipativity: nothing observed on or after the first sowing day."""
    cfg = SE.SensingConfig()
    assert cfg.obs_end_doy < 274          # the table's first sowing day


def test_conventional_belief_is_zero_which_is_what_ceres_does():
    o = np.array([1.0, -2.0, 0.5])
    assert np.all(SE.beliefs("conventional", o, o) == 0.0)


def test_oracle_belief_is_the_truth():
    o = np.array([1.0, -2.0, 0.5])
    assert np.allclose(SE.beliefs("oracle", o, np.zeros(3)), o)


def test_sensor_belief_falls_back_to_zero_where_estimation_failed():
    o = np.array([1.0, -2.0, 0.5])
    hat = np.array([0.9, np.nan, 0.4])
    b = SE.beliefs("sensor", o, hat)
    assert b[0] == pytest.approx(0.9) and b[1] == 0.0 and b[2] == pytest.approx(0.4)


def test_unknown_arm_raises():
    with pytest.raises(ValueError):
        SE.beliefs("clairvoyant", np.zeros(2), np.zeros(2))


def test_config_is_fully_serialisable_onto_a_result_row():
    row = SE.SensingConfig().as_row()
    for k in ("scenario", "sigma_c", "noise_c", "p_missing", "n_obs_days",
              "obs_end_doy", "seed", "estimator"):
        assert k in row
