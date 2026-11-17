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
    X6: admissibility must be checked per weather year, not on average.

    The per-year scope applies the SCORING DOMAIN per year (Phase 2 s6.3
    rule 4) on top of the tau range on the mean, so it is a strict subset of
    the mean-only set: a pair whose TSI leaves [-3.75, +13.75] in some year
    cannot be scored in that year.
    """
    from sim import response
    mean_set = {(p.cult_m, p.cult_f) for p in pairs.admissible_pairs(ph, ref_day, scope="mean")}
    year_set = {(p.cult_m, p.cult_f) for p in pairs.admissible_pairs(ph, ref_day, scope="years")}
    assert year_set <= mean_set
    assert year_set != mean_set, "per-year admissibility rejected nothing"
    for p in pairs.admissible_pairs(ph, ref_day, scope="years"):
        assert pairs.TAU_MIN - 1e-9 <= p.tsi0_mean <= pairs.TAU_MAX + 1e-9, p
        assert p.tsi0_min >= response.TSI_LO - 1e-9, p
        assert p.tsi0_max <= response.TSI_HI + 1e-9, p


def test_t2_4c_years_tau_variant_is_stronger_still(ph, ref_day):
    """
    The conservative variant applies the TAU RANGE per year. It is strictly
    stronger than the spec requires and is retained as a labelled variant.

    Recorded because the first Phase 3 implementation used it by mistake,
    cutting the population from 12 pairs to 6 and leaving only one pair in the
    hard [+5, +8] stratum that experiment E4 needs.
    """
    spec = {(p.cult_m, p.cult_f) for p in pairs.admissible_pairs(ph, ref_day, scope="years")}
    tau = {(p.cult_m, p.cult_f) for p in pairs.admissible_pairs(ph, ref_day, scope="years_tau")}
    assert tau < spec, (len(tau), len(spec))
    for p in pairs.admissible_pairs(ph, ref_day, scope="years_tau"):
        assert pairs.TAU_MIN - 1e-9 <= p.tsi0_min
        assert p.tsi0_max <= pairs.TAU_MAX + 1e-9


def test_t2_4d_both_e4_strata_are_populated(ph, ref_day):
    """
    E4 needs an easy-only [0, +2] and a hard-only [+5, +8] subpopulation.
    A stratum with one pair is not a subpopulation.
    """
    t = [p.tsi0_mean for p in pairs.admissible_pairs(ph, ref_day, scope="years")]
    assert sum(1 for v in t if 0 <= v <= 2) >= 2, "easy stratum too thin"
    assert sum(1 for v in t if 5 <= v <= 8) >= 2, "hard stratum too thin"


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
