"""
T2  -- same-day sowing invariants SG1-SG3, with per-year admissibility (X6).
T10 -- pair-population characterisation (new in the Phase 2 Supplement).

These tests verify the model's treatment of the single most important point in
the search space: co-sowing both parents. Phase 1 established that same-day
sowing is the observed optimum in all four Marahatta season-years and in the
UAS Bengaluru trial, so a model that mis-places it is wrong where it matters.
"""
import numpy as np
import pytest

from sim import pairs


# ---------------------------------------------------------------- T2.1 - T2.3

def test_t2_1_sg2_delta_equals_asi_gives_zero_tsi():
    """SG2: a pair with Delta = ASI0 must score TSI(0) = 0.00 (the UAS case)."""
    got = pairs.same_day_tsi_from_delta(delta=5.0, asi0=5.0)
    assert abs(got - 0.0) <= 0.01


def test_t2_2_sg3_delta_plus_1p5_gives_1p5():
    """SG3: Delta = ASI0 + 1.5 must score TSI(0) = 1.50 (Marahatta's range)."""
    got = pairs.same_day_tsi_from_delta(delta=6.5, asi0=5.0)
    assert abs(got - 1.5) <= 0.01


def test_t2_3_sg1_rejects_male_silking_with_female():
    """SG1: a pair with Delta = 0 must be rejected at construction, not scheduled."""
    with pytest.raises(pairs.InadmissiblePair) as e:
        pairs.check_admissible_delta(delta=0.0, asi0=5.0)
    msg = str(e.value)
    for token in ("delta", "asi0", "tsi0"):
        assert token in msg.lower(), f"diagnostic must name {token}: {msg}"


# ---------------------------------------------------------------------- T2.4

def test_t2_4a_grand_mean_admissibility(ph, ref_day):
    """Every pair admitted on the grand mean has mean TSI(0) in [0, +8]."""
    sel = pairs.admissible_pairs(ph, ref_day, scope="mean")
    assert sel, "no pairs admissible on the grand mean"
    for p in sel:
        assert pairs.TAU_MIN - 1e-9 <= p.tsi0_mean <= pairs.TAU_MAX + 1e-9


def test_t2_4b_per_year_admissibility_is_strictly_stronger(ph, ref_day):
    """
    X6: admissibility must hold in EVERY ensemble year, not on average.

    This is the regression test for Supplement gap 3. Pairs admitted on the
    grand mean fall outside [0, +8] in individual years, so the per-year set
    must be a strict subset.
    """
    mean_set = {(p.cult_m, p.cult_f) for p in pairs.admissible_pairs(ph, ref_day, scope="mean")}
    year_set = {(p.cult_m, p.cult_f) for p in pairs.admissible_pairs(ph, ref_day, scope="years")}
    assert year_set <= mean_set
    assert year_set != mean_set, (
        "per-year admissibility rejected nothing the mean admitted; "
        "the Supplement found PC0005xPC0001 and 990002xIB0012 should fail")
    for p in pairs.admissible_pairs(ph, ref_day, scope="years"):
        assert pairs.TAU_MIN - 1e-9 <= p.tsi0_min, p
        assert p.tsi0_max <= pairs.TAU_MAX + 1e-9, p


def test_t2_4b_old_rule_would_admit_negative_delta(ph):
    """The superseded rule 1.5 <= abs(gap) <= 14 admits male-silks-first pairs."""
    bad = pairs.legacy_rule_violations(ph)
    assert bad > 0, "expected the legacy abs(gap) rule to admit negative-Delta pairs"


def test_t2_5_sweeping_asi_holds_tsi0_and_moves_delta():
    """
    Phase 2 section 3.6: the ASI sweep must hold TSI(0) at its stratified value
    and re-derive Delta. Holding Delta fixed instead shifts every pair's
    same-day TSI in lockstep, which is the artefact Phase 1 section 2 found.
    """
    for asi0 in (3.0, 5.0, 8.0):
        d = pairs.delta_for_target_tsi0(tsi0=2.0, asi0=asi0)
        assert abs(pairs.same_day_tsi_from_delta(d, asi0) - 2.0) <= 1e-9
        assert abs(d - (asi0 + 2.0)) <= 1e-9


# ----------------------------------------------------------------------- T10

def test_t10_1_admissible_pair_count_is_reported(ph, ref_day):
    rep = pairs.population_report(ph, ref_day)
    assert rep["n_ordered_pairs"] == len(ph.cultivars) * (len(ph.cultivars) - 1)
    assert 0 < rep["n_admissible_mean"] <= rep["n_ordered_pairs"]
    assert rep["n_admissible_years"] <= rep["n_admissible_mean"]


def test_t10_2_slope_diagnostic_present_and_not_gated(ph, ref_day):
    """
    dTSI/doffset must be reported for every admissible pair and compared with
    the two observed values. It must NOT gate: the shortfall is a limitation
    to disclose, not a bug to fix, and fixing it would mean inventing
    coefficients.
    """
    rep = pairs.population_report(ph, ref_day)
    assert rep["slope_observed"] == {"uas_bengaluru": 1.00, "marahatta": 1.58}
    for p in rep["pairs"]:
        assert np.isfinite(p["slope"])
    assert "slope_shortfall_vs_marahatta" in rep


def test_t10_3_across_year_tsi_sd_reported(ph, ref_day):
    rep = pairs.population_report(ph, ref_day)
    assert rep["tsi0_sd_target_target_1a"] == 1.5
    for p in rep["pairs"]:
        assert np.isfinite(p["tsi0_sd"]) and p["tsi0_sd"] >= 0.0


def test_t10_4_single_coefficient_pairs_are_flagged(ph, ref_day):
    """PC0001-PC0005 share P2 and PHINT, so pairs within that family differ in P1 alone."""
    rep = pairs.population_report(ph, ref_day)
    assert "n_single_coefficient_pairs" in rep
    for p in rep["pairs"]:
        assert isinstance(p["single_coefficient"], bool)


def test_t10_5_expected_repeats_per_pair(ph, ref_day):
    """Fields in a farm are not independent draws when few pairs are admissible."""
    rep = pairs.population_report(ph, ref_day, n_fields=40)
    n = rep["n_admissible_years"] or rep["n_admissible_mean"]
    assert abs(rep["expected_repeats_per_pair"] - 40.0 / n) < 1e-9


# ------------------------------------------------- stratification is on TSI(0)

def test_stratification_is_on_tsi0_and_within_range(ph, ref_day):
    """P2-D3: strata are on TSI(0), not on the absolute silking gap."""
    rng = np.random.default_rng(1)
    sel = pairs.stratified_pairs(ph, 12, rng, ref_day)
    assert len(sel) == 12
    for p in sel:
        assert pairs.TAU_MIN - 1e-9 <= p.tsi0_mean <= pairs.TAU_MAX + 1e-9
    assert sorted(p.tsi0_mean for p in sel) == [p.tsi0_mean for p in
                                                sorted(sel, key=lambda q: q.tsi0_mean)]


def test_no_admissible_pair_has_negative_delta(ph, ref_day):
    for p in pairs.admissible_pairs(ph, ref_day, scope="years"):
        assert p.delta_mean > 0.0, p
