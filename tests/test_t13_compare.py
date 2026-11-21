"""
T13 -- U4 closure. Every strategy and baseline scored through the one shared
       scorer, with identical relative-yield and operational-cost definitions.

The Phase 3 audit found three divergent scoring paths. These tests assert the
new path has exactly one, that it agrees with the legacy COST definition, and
that no superseded scoring rule can enter it.
"""
import ast
import inspect

import numpy as np
import pytest

from sim import build, capacity, compare, optimise, pollen, response, scoring
from sim.capacity import make as make_capacity

WINDOW = list(range(300, 312))


@pytest.fixture(scope="module")
def farm(ph, ref_day):
    return build.build_farm(ph, n_fields=8, farm_seed=1, sow_day_ref=ref_day)


@pytest.fixture(scope="module")
def shared(ph, farm):
    model = make_capacity("visits")
    cands, diag = build.build_candidates(ph, farm, WINDOW, model, split_gap=2)
    assert cands, diag
    return model, cands


# ============================================ every planner, objective parity

@pytest.mark.parametrize("name", list(compare.ALL_PLANNERS))
def test_t13_1_every_planner_objective_equals_execution(ph, farm, shared, name):
    """
    The U4 requirement. A divergence here invalidates every strategy-versus-
    baseline comparison and leaves no other symptom.
    """
    model, cands = shared
    sol, res = compare.run_planner(name, ph, farm, WINDOW, model, cands,
                                   response.PRIMARY_SCENARIO)
    assert abs(sol.objective - res["total_area_r"]) <= 1e-9, (
        f"{name}: objective {sol.objective!r} != execution {res['total_area_r']!r}")
    for c, row in zip(sol.chosen, res["detail"]):
        assert abs(c.area_r - row["area_r"]) <= 1e-9, (name, c.label)


@pytest.mark.parametrize("name", list(compare.ALL_PLANNERS))
def test_t13_2_every_planner_produces_a_feasible_plan(ph, farm, shared, name):
    model, cands = shared
    sol, _ = compare.run_planner(name, ph, farm, WINDOW, model, cands,
                                 response.PRIMARY_SCENARIO)
    load = {}
    for c in sol.chosen:
        for d, v in c.cost.items():
            load[d] = load.get(d, 0.0) + v
    assert max(load.values(), default=0.0) <= model.capacity + 1e-9, (name, load)
    fids = [c.fid for c in sol.chosen]
    assert len(fids) == len(set(fids)), f"{name} sowed a field twice"
    assert all(d in WINDOW for c in sol.chosen for d in c.cost), name


# ============================================== the same schedule, same score

def test_t13_3_same_candidate_scores_the_same_in_every_path(ph, farm, shared):
    """
    Requirement 5. Take one candidate and score it three ways: the coefficient
    the builder attached, a direct `score_plan` call, and the value execution
    reports. All three must be the same number.
    """
    _, cands = shared
    by_fid = {f.fid: f for f in farm}
    for c in cands[:40]:
        f = by_fid[c.fid]
        fn = scoring.dssat_tsi_fn(ph, f, asi0=5.0,
                                  delta_thr=response.PRIMARY_SCENARIO.delta_thr)
        direct = scoring.score_plan(fn, f.area_ha, c.plan,
                                    response.PRIMARY_SCENARIO)
        assert abs(direct.area_r - c.area_r) <= 1e-12, c.label
    sol = optimise.solve(cands[:40], capacity=1e6, days=WINDOW, objective="OBJ-A")
    res = build.execute(ph, farm, sol)
    assert abs(sol.objective - res["total_area_r"]) <= 1e-12


def test_t13_4_all_planners_share_one_candidate_set(ph, farm, shared):
    """Identical relative-yield AND cost definitions, by construction."""
    model, cands = shared
    out = compare.run_all(ph, farm, WINDOW, model, cands,
                          response.PRIMARY_SCENARIO)
    pool = {c.label: (c.area_r, tuple(sorted(c.cost.items()))) for c in cands}
    for name, v in out.items():
        for c in v["solution"].chosen:
            assert c.label in pool, (name, c.label)
            assert pool[c.label] == (c.area_r, tuple(sorted(c.cost.items())))


# ===================================== cost parity with the legacy definition

@pytest.mark.parametrize("model_name", ["visits", "operations", "hours", "hectares"])
def test_t13_5_new_cost_matches_the_legacy_capacity_definition(farm, model_name):
    """
    Requirement 6. `build.plan_cost` must give the same day-by-day cost as the
    legacy `capacity.choice_cost`, for every capacity model. If these ever
    diverge, new and historical results stop being comparable.
    """
    model = make_capacity(model_name)

    class _Inst:
        split_male = True
        male_split_gap = 2

    inst = _Inst()
    for f in farm[:4]:
        for d_f in (300, 305):
            for o in (-4, -1, 0, 2, 5):
                plan = scoring.SowingPlan(d_f, d_f + o, d_f + o + 2)
                new = build.plan_cost(model, f, plan)
                old = capacity.choice_cost(model, f, inst, d_f, o)
                assert set(new) == set(old), (model_name, d_f, o)
                for d in new:
                    assert abs(new[d] - old[d]) <= 1e-12, (model_name, d_f, o, d)


@pytest.mark.parametrize("model_name", ["visits", "operations", "hours"])
def test_t13_6_parity_holds_under_every_capacity_model(ph, farm, model_name):
    model = make_capacity(model_name)
    cands, _ = build.build_candidates(ph, farm, WINDOW, model)
    assert cands
    for name in ("B1_sequential", "B5_greedy", "B6_joint", "S1_conventional"):
        sol, res = compare.run_planner(name, ph, farm, WINDOW, model, cands,
                                       response.PRIMARY_SCENARIO)
        assert abs(sol.objective - res["total_area_r"]) <= 1e-9, (model_name, name)


# ============================================================ split-male sowing

def test_t13_7_split_male_is_carried_through_every_planner(ph, farm, shared):
    """Requirement 7: split-male sowing must survive the whole path."""
    _, cands = shared
    out = compare.run_all(ph, farm, WINDOW, shared[0], cands,
                          response.PRIMARY_SCENARIO)
    for name, v in out.items():
        for row in v["result"]["detail"]:
            assert row["sow_m2"] is not None, f"{name} lost the second male date"
            assert row["credited_second_sowing"] is True, name


def test_t13_8_single_vs_split_both_run_and_can_differ(ph, farm):
    model = make_capacity("visits")
    split, _ = build.build_candidates(ph, farm, WINDOW, model, split_male=True)
    single, _ = build.build_candidates(ph, farm, WINDOW, model, split_male=False)
    assert split and single
    a, _ = compare.run_planner("B6_joint", ph, farm, WINDOW, model, split,
                               response.PRIMARY_SCENARIO)
    b, _ = compare.run_planner("B6_joint", ph, farm, WINDOW, model, single,
                               response.PRIMARY_SCENARIO)
    assert a.objective != b.objective, (
        "split and single gave identical objectives; the second sowing is "
        "either uncosted or uncredited")


# ================================ no superseded rule can enter the new path

def test_t13_9_no_legacy_imports_in_the_phase4_path():
    """
    Requirement 8. The comparison path must not reach the 10x bonus, the
    first-male-only scorer, or the superseded `seed_yield`.
    """
    for mod in (compare, build, optimise, scoring):
        tree = ast.parse(inspect.getsource(mod))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.lstrip("."))
            elif isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
        for banned in ("strategies", "baselines", "scheduler"):
            assert banned not in imported, f"{mod.__name__} imports {banned}"
        # Docstrings name the superseded functions deliberately, to record why
        # they are not used. Strip them before checking executable code.
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = ""
        code = ast.unparse(tree)
        for banned in ("seed_yield", "solve_milp", "yield_tons"):
            assert banned not in code, f"{mod.__name__} calls {banned}"


def test_t13_10_no_bonus_constant_in_the_comparison_path():
    for mod in (compare, optimise):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = ""
        assert "bonus" not in ast.unparse(tree).lower(), mod.__name__


def test_t13_11_every_scored_row_credits_both_male_dates(ph, farm, shared):
    _, cands = shared
    sol, res = compare.run_planner("B6_joint", ph, farm, WINDOW, shared[0],
                                   cands, response.PRIMARY_SCENARIO)
    for row in res["detail"]:
        assert row["mode"] == "split_ratio"
        assert row["sow_m1"] != row["sow_m2"]


# ===================================================== documented equivalences

def test_t13_12_s2_s3_s5_coincide_on_one_candidate_set_by_design(ph, farm, shared):
    """
    NOT a bug. S2, S3 and S5 use the same exact solve and differ only in WHICH
    weather information builds the candidate table. Given one table they must
    agree; the value of the stochastic solution and of perfect information
    come from supplying different tables, which is a Phase 4 design question.
    """
    model, cands = shared
    vals = {}
    for n in ("S2_static_ev", "S3_static_robust", "S5_wait_and_see", "B6_joint"):
        sol, _ = compare.run_planner(n, ph, farm, WINDOW, model, cands,
                                     response.PRIMARY_SCENARIO)
        vals[n] = sol.objective
    assert len({round(v, 9) for v in vals.values()}) == 1, vals


def test_t13_13_conventional_is_not_tuned(ph, farm, shared):
    """S1 applies one fixed offset to every field and should not win."""
    model, cands = shared
    s1, _ = compare.run_planner("S1_conventional", ph, farm, WINDOW, model,
                                cands, response.PRIMARY_SCENARIO)
    b6, _ = compare.run_planner("B6_joint", ph, farm, WINDOW, model, cands,
                                response.PRIMARY_SCENARIO)
    assert s1.objective <= b6.objective + 1e-9


def test_t13_14_joint_is_at_least_as_good_as_every_heuristic(ph, farm, shared):
    """
    B6 solves exactly over the same candidates the heuristics choose from, so
    it cannot be beaten by any of them. A violation means a heuristic is
    selecting something the MILP could not, i.e. the pools differ.
    """
    model, cands = shared
    out = compare.run_all(ph, farm, WINDOW, model, cands,
                          response.PRIMARY_SCENARIO)
    best = out["B6_joint"]["objective"]
    for name, v in out.items():
        assert v["objective"] <= best + 1e-6, (name, v["objective"], best)


# ================================================ solver optimality reporting

def test_t13_15_gap_is_measured_against_the_lp_relaxation(ph, farm, shared):
    """
    The reported gap must be MEASURED, not inferred from the solver's status
    string. CBC through PuLP reports "Optimal" even after stopping on the time
    limit, so the status alone cannot mean proven optimality.
    """
    _, cands = shared
    s = optimise.solve(cands, capacity=2.0, days=WINDOW, objective="OBJ-A")
    assert s.lp_bound is not None and s.lp_bound >= s.objective - 1e-9
    assert s.gap is not None and 0.0 <= s.gap <= 1.0
    assert abs(s.gap - (s.lp_bound - s.objective) / s.lp_bound) <= 1e-12


def test_t13_16_time_limit_is_detected_and_blocks_proven_optimality(ph, farm, shared):
    _, cands = shared
    s = optimise.solve(cands, capacity=2.0, days=WINDOW, objective="OBJ-A",
                       time_limit=0.01)
    assert s.elapsed_s is not None
    if s.hit_time_limit:
        assert s.proven_optimal is False


def test_t13_17_acceptance_rule_is_enforceable(ph, farm, shared):
    """A result that fails `meets_primary_gap` must be labelled a lower bound."""
    _, cands = shared
    s = optimise.solve(cands, capacity=2.0, days=WINDOW, objective="OBJ-A")
    assert isinstance(s.meets_primary_gap(), bool)
    assert s.termination in ("proven", "solver_converged", "time_limited")
    assert s.meets_primary_gap() is not s.is_lower_bound
    row = s.coords()
    for k in ("gap", "lp_bound", "hit_time_limit", "proven_optimal",
              "termination", "meets_primary_gap", "is_lower_bound"):
        assert k in row, k


def test_t13_18_time_limited_result_is_labelled_a_lower_bound(ph, farm, shared):
    """
    A solve cut off by the time limit with a wide measured gap must report
    itself as a lower bound, whatever CBC's status string says.
    """
    _, cands = shared
    s = optimise.solve(cands, capacity=2.0, days=WINDOW, objective="OBJ-A",
                       time_limit=0.01)
    if s.hit_time_limit and (s.gap is None or s.gap > optimise.REQUIRED_GAP):
        assert s.termination == "time_limited"
        assert s.is_lower_bound is True
        assert s.proven_optimal is False
