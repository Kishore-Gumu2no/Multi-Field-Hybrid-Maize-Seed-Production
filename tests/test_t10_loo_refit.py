"""
T10 -- leave-one-out refit of the winter_y1 yield curve.

Pins the four three-point refits reported in the Limitations paragraph and the
finding that two of them are degenerate: the exact-fit quadratic flips
concavity, the vertex clip then flattens the response, and asynchrony becomes
costless. Those two are excluded from the sensitivity claim; these tests fail
if that exclusion ever becomes silent.
"""
import numpy as np
import pytest

from sim import response as R
from run_loo_sensitivity import MAR, loo_fits

XS = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.75]
SC = R.Scenario(curve="winter_y1", plateau="PL2", kappa=1.5)

# a, b, c per dropped treatment mean. Pinned from loo_fits().
EXPECTED = {
    1.00: (6.7605, -1.1006, 0.0509),
    5.00: (3.2802, -0.2906, 0.0104),
    6.25: (3.1062, -0.1040, -0.0023),
    13.75: (2.7265, 0.3516, -0.0782),
}
DEGENERATE = (6.25, 13.75)
VALID = (1.00, 5.00)


def test_mar_is_four_points_three_parameters():
    """The premise of the whole check: 4 means, 3 free parameters."""
    assert len(MAR) == 4


def test_loo_fits_reproduce_pinned_coefficients():
    got = {f["dropped_tsi"]: (f["a"], f["b"], f["c"]) for f in loo_fits()}
    assert set(got) == set(EXPECTED)
    for k, (a, b, c) in EXPECTED.items():
        assert got[k] == pytest.approx((a, b, c), abs=5e-4), k


def test_each_refit_passes_through_its_three_retained_means():
    """An exact fit through 3 points must reproduce those 3 points."""
    for f in loo_fits():
        a, b, c = f["a"], f["b"], f["c"]
        for x, y in MAR.items():
            if x == f["dropped_tsi"]:
                continue
            assert a + b * x + c * x * x == pytest.approx(y, abs=1e-6)


def _response_spread(a, b, c):
    original = dict(R.CURVES["winter_y1"])
    R.CURVES["winter_y1"] = dict(a=a, b=b, c=c, r2=None)
    try:
        vals = [R.relative_yield(x, SC) for x in XS]
    finally:
        R.CURVES["winter_y1"] = original
    return max(vals) - min(vals)


def test_published_curve_response_declines():
    p = R.CURVES["winter_y1"]
    assert p["c"] > 0
    assert _response_spread(p["a"], p["b"], p["c"]) > 0.5


@pytest.mark.parametrize("drop", DEGENERATE)
def test_degenerate_refits_flatten_the_response(drop):
    """Concave refit + vertex clip => no synchrony penalty. Must stay visible."""
    a, b, c = EXPECTED[drop]
    assert c < 0, drop
    assert _response_spread(a, b, c) < 0.01, drop


@pytest.mark.parametrize("drop", VALID)
def test_valid_refits_keep_a_declining_convex_response(drop):
    a, b, c = EXPECTED[drop]
    assert c > 0, drop
    assert _response_spread(a, b, c) > 0.4, drop
