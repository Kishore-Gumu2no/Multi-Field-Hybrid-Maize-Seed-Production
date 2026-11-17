"""
End-to-end integration on a real (synthetic-cultivar) farm.

T6   -- out-of-domain candidates never become variables; realised out-of-domain
        field-years are flagged at execution and gate the result.
T7.5 -- BLOCKING. The objective coefficient must equal the execution score.
T8.7 -- every scored path uses the same scorer and capacity accounting.
T9   -- reproducibility: coordinates, determinism, table hash.
"""
import numpy as np
import pytest

from sim import build, optimise, pairs, response, runspec, scoring
from sim.capacity import make as make_capacity

WINDOW = list(range(300, 316))


@pytest.fixture
def farm(ph, ref_day):
    return build.build_farm(ph, n_fields=6, farm_seed=1, sow_day_ref=ref_day)


@pytest.fixture
def cands(ph, farm):
    model = make_capacity("visits")
    c, diag = build.build_candidates(ph, farm, WINDOW, model,
                                     offsets=range(-4, 7), split_gap=2)
    assert c, f"no candidates generated; diagnostics {diag}"
    return c, diag, model


# ======================================================================= T6

def test_t6_1_t6_2_out_of_domain_candidates_are_absent(cands):
    c, _, _ = cands
    for cand in c:
        assert response.TSI_LO - 1e-6 <= cand.tsi_mean <= response.TSI_HI + 1e-6, cand


def test_t6_rho_one_admits_only_fully_in_domain_candidates(ph, farm):
    model = make_capacity("visits")
    c, _ = build.build_candidates(ph, farm, WINDOW, model, rho=1.0)
    assert all(cand.n_out_of_domain == 0 for cand in c)


def test_t6_rho_relaxed_admits_more_and_flags_them(ph, farm):
    model = make_capacity("visits")
    strict, _ = build.build_candidates(ph, farm, WINDOW, model, rho=1.0)
    loose, _ = build.build_candidates(ph, farm, WINDOW, model, rho=0.6)
    assert len(loose) >= len(strict)
    if len(loose) > len(strict):
        assert any(cand.n_out_of_domain > 0 for cand in loose)


def test_t6_4_execution_flags_realised_out_of_domain_field_years(ph, farm, cands):
    c, _, model = cands
    sol = optimise.solve(c, capacity=2.0, days=WINDOW, objective="OBJ-A")
    res = build.execute(ph, farm, sol)
    rel = res["reliability"]
    assert rel.n_field_years > 0
    assert rel.n_out_of_domain >= 0
    assert isinstance(rel.reliable, bool)


def test_t6_5_reliability_gate_trips_above_threshold():
    rel = runspec.Reliability(n_field_years=100, n_out_of_domain=6)
    assert not rel.reliable
    assert "UNRELIABLE" in rel.verdict()
    ok = runspec.Reliability(n_field_years=100, n_out_of_domain=5)
    assert ok.reliable and ok.verdict() == "reliable"


# ================================================================ T7.5 BLOCKING

@pytest.mark.parametrize("objective", list(optimise.OBJECTIVES))
def test_t7_5_objective_coefficient_equals_execution_score(ph, farm, cands,
                                                           objective):
    """
    BLOCKING. A divergence here invalidates every strategy comparison and
    produces results that look entirely reasonable. The project was already
    caught by one instance of this class (the 10x bonus).
    """
    c, _, _ = cands
    sol = optimise.solve(c, capacity=2.0, days=WINDOW, objective=objective)
    res = build.execute(ph, farm, sol)
    assert abs(sol.objective - res["total_area_r"]) <= 1e-9, (
        f"{objective}: objective {sol.objective!r} != execution "
        f"{res['total_area_r']!r}")
    for cand, row in zip(sol.chosen, res["detail"]):
        assert abs(cand.area_r - row["area_r"]) <= 1e-9, (cand.label, row)


def test_t7_5b_scoring_is_a_single_function(ph, farm):
    """Both the candidate builder and execution enter scoring.score_plan."""
    import inspect
    for mod in (build,):
        src = inspect.getsource(mod)
        assert "score_plan" in src
    assert build._scoring.score_plan is scoring.score_plan


# ======================================================================= T8.7

@pytest.mark.parametrize("model_name", ["visits", "operations", "hours"])
def test_t8_7_every_capacity_model_uses_the_same_scorer(ph, farm, model_name):
    model = make_capacity(model_name)
    c, _ = build.build_candidates(ph, farm, WINDOW, model)
    assert c
    sol = optimise.solve(c, capacity=model.capacity, days=WINDOW,
                         objective="OBJ-A")
    res = build.execute(ph, farm, sol)
    assert abs(sol.objective - res["total_area_r"]) <= 1e-9


def test_t8_objectives_ordered_on_a_real_farm(ph, farm, cands):
    c, _, _ = cands
    a = optimise.solve(c, capacity=2.0, days=WINDOW, objective="OBJ-A")
    for other in ("OBJ-B", "OBJ-C"):
        s = optimise.solve(c, capacity=2.0, days=WINDOW, objective=other)
        assert a.objective >= s.objective - 1e-6, (other, a.objective, s.objective)


def test_t8_window_guard_drops_nothing_when_offsets_fit(ph, farm, cands):
    c, _, _ = cands
    sol = optimise.solve(c, capacity=2.0, days=WINDOW, objective="OBJ-A")
    assert sol.n_dropped_outside_window == 0


# ========================================================================= T9

def test_t9_run_coordinates_are_complete(ph):
    rc = runspec.RunCoords(
        curve="winter_y1", plateau="PL2", kappa=1.5, delta_thr=0.0, asi0=5.0,
        scoring_mode="split_ratio", pair_scope="years", pair_rho=1.0,
        tau_min=pairs.TAU_MIN, tau_max=pairs.TAU_MAX, n_fields=6, farm_seed=1,
        objective="OBJ-A", capacity_model="visits", capacity=2.0,
        split_male=True, split_gap=2,
        phenology_table_sha=runspec.table_sha("results/phenology_table.npz"),
        git_commit=runspec.git_commit("."))
    row = rc.as_row()
    for k in ("curve", "plateau", "kappa", "delta_thr", "asi0", "objective",
              "capacity_model", "pair_scope", "farm_seed", "split_gap",
              "phenology_table_sha"):
        assert k in row, k
    assert len(row["phenology_table_sha"]) == 16


def test_t9_farm_is_deterministic_in_its_seed(ph, ref_day):
    a = build.build_farm(ph, 6, 1, ref_day)
    b = build.build_farm(ph, 6, 1, ref_day)
    c = build.build_farm(ph, 6, 2, ref_day)
    assert [f.coords() for f in a] == [f.coords() for f in b]
    assert [f.coords() for f in a] != [f.coords() for f in c]


def test_t9_solve_is_deterministic(ph, farm, cands):
    c, _, _ = cands
    runs = [optimise.solve(c, capacity=2.0, days=WINDOW, objective="OBJ-A")
            for _ in range(2)]
    assert len({round(r.objective, 12) for r in runs}) == 1
    assert len({tuple(sorted(x.label for x in r.chosen)) for r in runs}) == 1


def test_t9_every_scenario_cell_runs_end_to_end(ph, farm):
    """Smoke test over the declared grid. No performance claim is made."""
    model = make_capacity("visits")
    for sc in response.scenario_grid()[:6]:
        c, _ = build.build_candidates(ph, farm, WINDOW, model, scenario=sc)
        sol = optimise.solve(c, capacity=2.0, days=WINDOW, objective="OBJ-A")
        res = build.execute(ph, farm, sol, scenario=sc)
        assert abs(sol.objective - res["total_area_r"]) <= 1e-9, sc
