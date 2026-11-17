"""
T12 -- deterministic fixtures that make the three objective formulations
       give DIFFERENT answers.

The six-field smoke instance in `records/DIAGNOSTICS.txt` has all three
returning 16.4248 with six fields sown. That is not evidence the choice is
immaterial: at 56% capacity utilisation every field is sown under every
formulation, so the coverage constraint is slack and all three reduce to the
same single-stage problem. These fixtures create the pressure that instance
lacks, and every number below is hand-checkable.
"""
import pytest

from sim import build, optimise
from sim.capacity import make as make_capacity
from sim.scoring import SowingPlan


def _c(fid, label, sow_f, sow_m1, area_ha, r, cost, sow_m2=None):
    return optimise.Candidate(
        fid=fid, plan=SowingPlan(sow_f, sow_m1, sow_m2), area_ha=area_ha,
        area_r=area_ha * r, cost=cost, tsi_mean=0.0, n_out_of_domain=0,
        label=label)


# ===================================================== F1: the merging bias

F1_DAYS, F1_CAP = [1, 2], 1.0


def fixture_f1():
    """
    A 4.0 ha, B 1.5 ha; one visit per day over two days.
      A staggered  occupies both days, r 1.00 -> area_r 4.00
      A co-sown    one day,             r 0.50 -> area_r 2.00
      B co-sown    one day,             r 1.00 -> area_r 1.50
    Taking A staggered blocks B. Merging A frees a day for B.
    """
    return [
        _c(0, "A staggered", 1, 2, 4.0, 1.00, {1: 1.0, 2: 1.0}),
        _c(0, "A co-sown",   1, 1, 4.0, 0.50, {1: 1.0}),
        _c(1, "B co-sown d1", 1, 1, 1.5, 1.00, {1: 1.0}),
        _c(1, "B co-sown d2", 2, 2, 1.5, 1.00, {2: 1.0}),
    ]


F1_EXPECTED = {"OBJ-A": (4.00, 1, 4.0, 0.0),
               "OBJ-B": (3.50, 2, 5.5, 1.0),
               "OBJ-C": (3.50, 2, 5.5, 1.0)}


@pytest.mark.parametrize("obj", list(optimise.OBJECTIVES))
def test_f1_merging_bias(obj):
    z, n, ha, merged = F1_EXPECTED[obj]
    s = optimise.solve(fixture_f1(), capacity=F1_CAP, days=F1_DAYS, objective=obj)
    assert abs(s.objective - z) <= 1e-6, (obj, s.objective)
    assert s.n_sown == n
    assert abs(s.ha_sown - ha) <= 1e-6
    assert abs(s.merged_fraction - merged) <= 1e-6


def test_f1_obj_c_does_NOT_fix_the_merging_bias():
    """
    Area coverage fixes area-blindness (F2) but not the merging preference:
    sowing A co-sown plus B covers 5.5 ha against A staggered's 4.0, so
    stage 1 still prefers the merged schedule.
    """
    a = optimise.solve(fixture_f1(), capacity=F1_CAP, days=F1_DAYS, objective="OBJ-A")
    c = optimise.solve(fixture_f1(), capacity=F1_CAP, days=F1_DAYS, objective="OBJ-C")
    assert c.objective < a.objective
    assert c.merged_fraction > a.merged_fraction


# ================================================= F2: area-blindness of OBJ-B

F2_DAYS, F2_CAP = [1, 2], 1.0


def fixture_f2():
    """
    L 10.0 ha needs BOTH days; S1 and S2 are 1.0 ha and need one day each.
      L staggered -> area_r 10.00, coverage 1 field, 10.0 ha
      S1 + S2     -> area_r  2.00, coverage 2 fields, 2.0 ha
    OBJ-B counts fields and takes the two small ones. OBJ-C counts hectares
    and takes L, agreeing with OBJ-A.
    """
    return [
        _c(0, "L staggered", 1, 2, 10.0, 1.00, {1: 1.0, 2: 1.0}),
        _c(1, "S1 d1", 1, 1, 1.0, 1.00, {1: 1.0}),
        _c(2, "S2 d2", 2, 2, 1.0, 1.00, {2: 1.0}),
    ]


F2_EXPECTED = {"OBJ-A": (10.00, 1), "OBJ-B": (2.00, 2), "OBJ-C": (10.00, 1)}


@pytest.mark.parametrize("obj", list(optimise.OBJECTIVES))
def test_f2_area_blindness(obj):
    z, n = F2_EXPECTED[obj]
    s = optimise.solve(fixture_f2(), capacity=F2_CAP, days=F2_DAYS, objective=obj)
    assert abs(s.objective - z) <= 1e-6, (obj, s.objective)
    assert s.n_sown == n


def test_f2_obj_b_gives_up_eighty_percent_of_the_objective():
    a = optimise.solve(fixture_f2(), capacity=F2_CAP, days=F2_DAYS, objective="OBJ-A")
    b = optimise.solve(fixture_f2(), capacity=F2_CAP, days=F2_DAYS, objective="OBJ-B")
    assert abs((a.objective - b.objective) / a.objective - 0.80) <= 1e-6


def test_f2_obj_c_repairs_what_obj_b_breaks():
    a = optimise.solve(fixture_f2(), capacity=F2_CAP, days=F2_DAYS, objective="OBJ-A")
    c = optimise.solve(fixture_f2(), capacity=F2_CAP, days=F2_DAYS, objective="OBJ-C")
    assert abs(a.objective - c.objective) <= 1e-6


def test_f1_and_f2_together_discriminate_all_three():
    """
    Neither fixture alone separates the three. Together they do:
      F1  OBJ-A > OBJ-B = OBJ-C   (merging bias; OBJ-C does not help)
      F2  OBJ-A = OBJ-C > OBJ-B   (area-blindness; OBJ-C does help)
    """
    f1 = {o: optimise.solve(fixture_f1(), capacity=F1_CAP, days=F1_DAYS,
                            objective=o).objective for o in optimise.OBJECTIVES}
    f2 = {o: optimise.solve(fixture_f2(), capacity=F2_CAP, days=F2_DAYS,
                            objective=o).objective for o in optimise.OBJECTIVES}
    assert f1["OBJ-A"] > f1["OBJ-B"] and abs(f1["OBJ-B"] - f1["OBJ-C"]) <= 1e-6
    assert abs(f2["OBJ-A"] - f2["OBJ-C"]) <= 1e-6 and f2["OBJ-C"] > f2["OBJ-B"]


# ======================================== F3: real data under capacity pressure

def test_f3_real_farm_under_pressure_separates_the_objectives(ph, ref_day):
    """
    The same six-field farm, but a window short enough that capacity binds.
    No research experiment: this is a solver-behaviour fixture.
    """
    farm = build.build_farm(ph, 6, 1, ref_day)
    short = list(range(300, 306))                   # 6 days instead of 16
    cands, _ = build.build_candidates(ph, farm, short, make_capacity("visits"))
    if not cands:
        pytest.skip("no candidates in the short window")
    res = {}
    for obj in optimise.OBJECTIVES:
        s = optimise.solve(cands, capacity=2.0, days=short, objective=obj)
        r = build.execute(ph, farm, s)
        assert abs(s.objective - r["total_area_r"]) <= 1e-9, obj   # T7.5 holds
        res[obj] = (s.objective, s.n_sown, s.ha_sown, s.merged_fraction)
    assert res["OBJ-A"][0] >= res["OBJ-B"][0] - 1e-9
    assert res["OBJ-A"][0] >= res["OBJ-C"][0] - 1e-9
    assert res["OBJ-B"][1] >= res["OBJ-A"][1]       # coverage never lower
