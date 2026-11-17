"""
T8  -- solver optimality, determinism, scoring parity.
T11 -- the three objective formulations (Supplement X10, X11).

The T11 instance is the counterexample from Phase 2 Supplement section 4:
one sowing day, capacity 2 visits, field A 4.0 ha and field B 1.5 ha.
"""
import numpy as np
import pytest

from sim import optimise as O
from sim.scoring import SowingPlan


DAYS = [1, 2]
CAP = 1.0        # one field-visit per day


def counterexample():
    """
    Phase 2 Supplement section 4, made internally consistent: two days, one
    visit per day, field A 4.0 ha and field B 1.5 ha.

      A staggered  female day 1, male day 2  -> occupies BOTH days, r 1.00
      A co-sown    female and male day 1     -> one day,            r 0.50
      B co-sown    female and male, day 1 or 2 -> one day,          r 1.00

    A staggered therefore blocks B entirely. Taking it gives coverage 1 and
    Z = 4.00; merging A to free a day gives coverage 2 and Z = 3.50.
    """
    return [
        O.Candidate(fid=0, plan=SowingPlan(sow_f=1, sow_m1=2), area_ha=4.0,
                    area_r=4.0, cost={1: 1.0, 2: 1.0}, tsi_mean=0.0,
                    n_out_of_domain=0, label="A staggered"),
        O.Candidate(fid=0, plan=SowingPlan(sow_f=1, sow_m1=1), area_ha=4.0,
                    area_r=2.0, cost={1: 1.0}, tsi_mean=0.0,
                    n_out_of_domain=0, label="A co-sown"),
        O.Candidate(fid=1, plan=SowingPlan(sow_f=1, sow_m1=1), area_ha=1.5,
                    area_r=1.5, cost={1: 1.0}, tsi_mean=0.0,
                    n_out_of_domain=0, label="B co-sown d1"),
        O.Candidate(fid=1, plan=SowingPlan(sow_f=2, sow_m1=2), area_ha=1.5,
                    area_r=1.5, cost={2: 1.0}, tsi_mean=0.0,
                    n_out_of_domain=0, label="B co-sown d2"),
    ]


# ======================================================================= T11

def test_t11_1_obj_a_takes_the_higher_objective_with_lower_coverage():
    s = O.solve(counterexample(), capacity=CAP, days=DAYS, objective="OBJ-A")
    assert abs(s.objective - 4.00) <= 1e-6, s.objective
    assert s.n_sown == 1
    assert s.chosen[0].label == "A staggered"


def test_t11_2_obj_b_gives_up_objective_to_raise_field_count():
    s = O.solve(counterexample(), capacity=CAP, days=DAYS, objective="OBJ-B")
    assert abs(s.objective - 3.50) <= 1e-6, s.objective
    assert s.n_sown == 2
    assert s.chosen[0].label == "A co-sown", "OBJ-B must pick the MERGED schedule"


def test_t11_2b_obj_b_loss_is_quantified():
    a = O.solve(counterexample(), capacity=CAP, days=DAYS, objective="OBJ-A")
    b = O.solve(counterexample(), capacity=CAP, days=DAYS, objective="OBJ-B")
    loss = (a.objective - b.objective) / a.objective
    assert abs(loss - 0.125) <= 1e-6, loss


def test_t11_3_obj_c_area_coverage_is_reported():
    s = O.solve(counterexample(), capacity=CAP, days=DAYS, objective="OBJ-C")
    assert np.isfinite(s.objective)
    assert s.ha_sown >= 4.0


def test_t11_4_obj_a_dominates_both_constrained_variants():
    a = O.solve(counterexample(), capacity=CAP, days=DAYS, objective="OBJ-A")
    for other in ("OBJ-B", "OBJ-C"):
        s = O.solve(counterexample(), capacity=CAP, days=DAYS, objective=other)
        assert a.objective >= s.objective - 1e-9, (other, a.objective, s.objective)


def test_t11_5_fields_and_hectares_sown_reported_for_all_three():
    for obj in O.OBJECTIVES:
        s = O.solve(counterexample(), capacity=CAP, days=DAYS, objective=obj)
        assert isinstance(s.n_sown, int)
        assert np.isfinite(s.ha_sown)
        assert s.coords()["objective"] == obj


def test_t11_6_merging_bias_is_measurable():
    a = O.solve(counterexample(), capacity=CAP, days=DAYS, objective="OBJ-A")
    b = O.solve(counterexample(), capacity=CAP, days=DAYS, objective="OBJ-B")
    assert b.merged_fraction > a.merged_fraction


def test_no_bonus_constant_anywhere():
    """
    The 10x per-field bonus must not reappear in EXECUTABLE code. The module
    docstring names it deliberately, to record why it was removed, so the
    check strips docstrings before looking.
    """
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(O))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    assert "bonus" not in ast.unparse(tree).lower()


# ======================================================================== T8

def test_t8_1_single_field_matches_the_hand_computed_optimum():
    cands = [c for c in counterexample() if c.fid == 0]
    s = O.solve(cands, capacity=1.0, days=[1], objective="OBJ-A")
    assert abs(s.objective - 2.00) <= 1e-6   # only the 1-visit option fits
    s2 = O.solve(cands, capacity=CAP, days=DAYS, objective="OBJ-A")
    assert abs(s2.objective - 4.00) <= 1e-6


def test_t8_2_infinite_capacity_gives_each_field_its_own_best():
    s = O.solve(counterexample(), capacity=1e6, days=DAYS, objective="OBJ-A")
    assert abs(s.objective - (4.0 + 1.5)) <= 1e-6
    assert s.n_sown == 2


def test_t8_3_stage_two_coverage_equals_stage_one_optimum():
    s = O.solve(counterexample(), capacity=CAP, days=DAYS, objective="OBJ-B")
    assert s.stage1_value is not None
    assert s.n_sown == int(round(s.stage1_value))


def test_t8_5_gap_is_reported_and_primary_results_are_tight():
    s = O.solve(counterexample(), capacity=CAP, days=DAYS, objective="OBJ-A")
    assert s.status == "Optimal"
    assert s.gap is not None and s.gap <= 0.005


def test_t8_6_determinism_identical_objective_and_selection():
    runs = [O.solve(counterexample(), capacity=CAP, days=DAYS, objective="OBJ-A")
            for _ in range(3)]
    assert len({round(r.objective, 12) for r in runs}) == 1
    assert len({tuple(sorted(c.label for c in r.chosen)) for r in runs}) == 1


def test_t8_at_most_one_schedule_per_field():
    s = O.solve(counterexample(), capacity=1e6, days=DAYS, objective="OBJ-A")
    fids = [c.fid for c in s.chosen]
    assert len(fids) == len(set(fids))


def test_unknown_objective_rejected():
    with pytest.raises(ValueError):
        O.solve(counterexample(), capacity=CAP, days=DAYS, objective="OBJ-Z")
