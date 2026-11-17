"""
T1 -- TSI calculation, and the provider-specific ASI rule (P2-D4).

DSSAT's ADAT is the SILKING date for both parents (VERIFIED from CERES-Maize
source, Target 1A), so male anthesis needs ASI0 subtracted. The thermal-time
provider's tau_m already targets male 75% TASSELING, so it needs no correction.
Applying the correction to both would subtract five days from a path that is
already right -- the error the Phase 2 Supplement caught.
"""
import numpy as np
import pytest

from sim import pairs
from sim.core import Field
from sim.phenology import ThermalTimePhenology, GDD_PER_DAY_NOMINAL


class _FakeTable:
    """Deterministic stand-in: ADAT is constant per cultivar, so TSI is exact."""
    def __init__(self, days):
        self.days = days          # cultivar -> days from sowing to ADAT (silking)


@pytest.fixture
def synth(ph):
    """A real table plus a known-Delta synthetic pair helper."""
    return ph


# ------------------------------------------------------------------ T1.1, T1.2

def test_t1_1_delta_equals_asi_gives_zero_tsi_at_offset_zero(ph, ref_day):
    """
    SG2 for the DSSAT provider: Delta = ASI0 at offset 0 must give TSI = 0.00.

    Delta must be taken at the SAME aggregation level the comparison uses --
    the mean over weather years at this sowing day. PairInfo.delta_mean is a
    grand mean over years AND sowing days, which differs by ~0.6 d; using it
    here would test an identity that does not hold.
    """
    sel = pairs.admissible_pairs(ph, ref_day, scope="years")
    assert sel
    p = sel[0]
    d = pairs.delta_series(ph, p.cult_m, p.cult_f, ref_day)
    asi0 = float(np.nanmean(d))              # choose ASI0 so Delta - ASI0 == 0
    got = ph.tsi_mean(p.cult_m, p.cult_f, ref_day, ref_day, asi0=asi0)
    assert abs(got) <= 0.01, got


def test_t1_2_tsi_increases_monotonically_with_male_offset(ph, ref_day):
    sel = pairs.admissible_pairs(ph, ref_day, scope="years")
    p = sel[0]
    vals = []
    for k in range(0, 7):
        d = ref_day + k
        if d not in ph._di:
            continue
        vals.append(ph.tsi_mean(p.cult_m, p.cult_f, d, ref_day))
    assert len(vals) >= 4
    assert all(b > a for a, b in zip(vals, vals[1:])), vals


# ------------------------------------------------------------------------ T1.3

def test_t1_3_thermal_provider_ignores_asi0():
    """
    The thermal-time provider must NOT subtract ASI0: tau_m already targets
    male tasseling. Passing asi0 must change nothing.
    """
    prov = ThermalTimePhenology()
    f = Field(fid=0, tau_m=1030.0, tau_f=1000.0, area_ha=2.0, soil_drain_days=1)
    w = dict(tmax=np.full(200, 28.0), tmin=np.full(200, 12.0))
    prep = prov.prepare(f, w)
    base = prov.tsi(prep, f, sow_m=5, sow_f=5, asi0=0.0)
    for asi0 in (3.0, 5.0, 8.0):
        assert prov.tsi(prep, f, 5, 5, asi0=asi0) == base, (
            "thermal provider must ignore asi0")
    # tau_m - tau_f = 30 GDD at 20 GDD/d is about 1.5 d
    assert 0.5 <= base <= 3.5, base


def test_t1_3b_dssat_provider_does_apply_asi0(ph, ref_day):
    sel = pairs.admissible_pairs(ph, ref_day, scope="years")
    p = sel[0]
    a = ph.tsi_mean(p.cult_m, p.cult_f, ref_day, ref_day, asi0=0.0)
    b = ph.tsi_mean(p.cult_m, p.cult_f, ref_day, ref_day, asi0=5.0)
    assert abs((a - b) - 5.0) <= 1e-9, (a, b)


# ------------------------------------------------------------------------ T1.4

def test_t1_4_both_providers_give_non_negative_same_day_tsi(ph, ref_day):
    """
    C1b: the two providers must agree in the SIGN of the parental gap. The
    thermal path forces tau_m > tau_f; the DSSAT path must do the same through
    pair admissibility.
    """
    prov = ThermalTimePhenology()
    f = Field(fid=0, tau_m=1040.0, tau_f=1000.0, area_ha=2.0, soil_drain_days=1)
    w = dict(tmax=np.full(200, 28.0), tmin=np.full(200, 12.0))
    assert prov.tsi(prov.prepare(f, w), f, 5, 5) > 0.0

    for p in pairs.admissible_pairs(ph, ref_day, scope="years"):
        assert ph.tsi_mean(p.cult_m, p.cult_f, ref_day, ref_day) >= -1e-9, p


# ------------------------------------------------------------------------ T1.5

def test_t1_5_threshold_correction_shifts_tsi_exactly(ph, ref_day):
    """delta_thr (the 50%-to-75% flowering-basis correction) is additive."""
    sel = pairs.admissible_pairs(ph, ref_day, scope="years")
    p = sel[0]
    base = ph.tsi_mean(p.cult_m, p.cult_f, ref_day, ref_day)
    for dt in (-0.28, -0.5):
        got = ph.tsi_mean(p.cult_m, p.cult_f, ref_day, ref_day, delta_thr=dt)
        assert abs((got - base) - dt) <= 1e-9


# ------------------------------------------------------------------------ T1.6

def test_t1_6_missing_cultivar_or_day_is_non_finite_not_zero(ph, ref_day):
    assert not np.isfinite(ph.tsi_mean("NOPE01", ph.cultivars[0], ref_day, ref_day))
    assert not np.isfinite(ph.tsi_mean(ph.cultivars[0], "NOPE01", ref_day, ref_day))
    far = max(ph.sow_days) + 500
    assert not np.isfinite(ph.tsi_mean(ph.cultivars[0], ph.cultivars[1], far, ref_day))


def test_t1_7_legacy_anthesis_is_unchanged(ph, ref_day):
    """
    Historical scripts call anthesis() and must keep reproducing their old
    numbers: it still returns ADAT, i.e. SILKING, with no ASI subtraction.
    """
    f = type("F", (), {"cult_m": ph.cultivars[0], "cult_f": ph.cultivars[1]})()
    yi = ph._yi[ph.years[0]]
    v = ph.anthesis(yi, f, ref_day, "m")
    raw = ph.adat[ph._ci[ph.cultivars[0]], yi, ph._di[ref_day]]
    assert abs(v - (ref_day + float(raw))) <= 1e-12
