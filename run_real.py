"""
R1 -- the real-data run.

Everything that was synthetic in the provisional study is replaced here:

  weather    21 years of NASA POWER daily records for Ranebennur, Karnataka
             (14.60 N, 75.60 E, 597 m), a hybrid maize seed production area
  season     rabi, sown October-November, so that flowering falls in the dry
             December-January window as commercial practice requires
  phenology  DSSAT CERES-Maize with published cultivar coefficients,
             precomputed over the decision grid
  operations three crew visits per field -- female once, male on two dates,
             following Chassaigne-Ricciulli et al. (2021)

It also answers the question the provisional study could not: at what farm
size and crew capacity does the shared resource actually constrain the
schedule? If it never binds, the scheduling problem dissolves and the paper
should say so.

Writes results/R1_*.csv and results/R1_summary.json
"""

import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from sim.core import Field, seed_yield, trafficability
from sim.phenology import DssatPhenology
from sim.scheduler import Instance, execute
from sim.strategies import (access_for, plan_conventional, plan_static_ev,
                            plan_static_robust, plan_wait_and_see,
                            make_replanner)
from sim.weather_real import read_power_csv, years_available, to_season

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
POWER_CSV = os.path.join(HERE, "data", "power_ranebennur.csv")
PHENO = os.path.join(RESULTS, "phenology_table.npz")

# Rabi sowing window: 1 October (doy 274) to 30 November (doy 334).
SOW_START, SOW_END = 274, 334
N_PROC = max(1, min(2, os.cpu_count() or 1))


# ---------------------------------------------------------------------------
# Instance construction from real data
# ---------------------------------------------------------------------------

_PH = None


def pheno():
    """Lazily loaded DSSAT phenology table, shared by all fields."""
    global _PH
    if _PH is None:
        _PH = DssatPhenology(PHENO)
    return _PH


class RealField(Field):
    """
    A field whose parents are real DSSAT cultivars.

    Flowering comes from the precomputed CERES-Maize table rather than the
    thermal-time approximation, so tau_m and tau_f are unused.
    """

    def __init__(self, fid, cult_m, cult_f, area_ha, soil_drain_days):
        super().__init__(fid, tau_m=0, tau_f=0, area_ha=area_ha,
                         soil_drain_days=soil_drain_days)
        self.cult_m = cult_m
        self.cult_f = cult_f

    def tsi(self, weather, sow_m, sow_f):
        ph = pheno()
        try:
            yi = ph.prepare(self, weather)
        except KeyError:
            return np.inf
        a = ph.anthesis(yi, self, int(sow_m), "m")
        b = ph.anthesis(yi, self, int(sow_f), "f")
        if not (np.isfinite(a) and np.isfinite(b)):
            return np.inf
        return a - b

    def yield_tons(self, weather, sow_m, sow_f, curve=None, sharpness=1.0):
        t = self.tsi(weather, sow_m, sow_f)
        if not np.isfinite(t):
            return 0.0
        kw = dict(sharpness=sharpness)
        if curve:
            kw["curve"] = curve
        return float(seed_yield(t, **kw)) * self.area_ha


def pick_parent_pairs(ph, n_pairs, rng):
    """
    Choose male/female cultivar pairs whose flowering differs enough to need
    staggering, as real parent pairs do.

    Candidate pairs are ordered by how far apart the two parents flower, cut
    into `n_pairs` strata of equal width, and one pair drawn at random from
    each stratum. Stratifying keeps the difficulty profile of a farm
    comparable across farm sizes -- every farm spans easy and hard crosses in
    the same proportions -- while the draw within each stratum makes two farms
    with different seeds genuinely different farms rather than the same farm
    relabelled. An earlier version selected the stratum representatives
    deterministically, which made replicate farms differ only in field area
    and so understated the variance of every estimate.
    """
    adat = np.nanmean(ph.adat, axis=(1, 2))        # mean days to anthesis
    order = np.argsort(adat)
    cands = []
    for i in range(len(order)):
        for j in range(len(order)):
            if i == j:
                continue
            gap = adat[order[j]] - adat[order[i]]
            if 1.5 <= abs(gap) <= 14:
                cands.append((ph.cultivars[order[i]], ph.cultivars[order[j]],
                              float(gap)))
    cands.sort(key=lambda t: abs(t[2]))

    edges = np.linspace(0, len(cands), n_pairs + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        b = max(b, a + 1)
        out.append(cands[int(rng.integers(a, min(b, len(cands))))])
    return out


def build_real(n_fields=20, capacity=2, farm_seed=1, split_male=True):
    ph = DssatPhenology(PHENO)
    rng = np.random.default_rng(farm_seed)
    pairs = pick_parent_pairs(ph, n_fields, rng)
    rng.shuffle(pairs)

    fields = []
    for i, (cm, cf, _) in enumerate(pairs):
        fields.append(RealField(
            fid=i, cult_m=cm, cult_f=cf,
            area_ha=float(rng.uniform(1.5, 4.0)),
            soil_drain_days=int(rng.integers(1, 4)),
        ))

    inst = Instance(fields, window_start=SOW_START,
                    window_len=SOW_END - SOW_START + 1,
                    capacity=capacity, offsets=range(-12, 3),
                    split_male=split_male)
    return ph, fields, inst


def load_seasons():
    w = read_power_csv(POWER_CSV)
    yrs = [y for y in years_available(w) if y in
           set(DssatPhenology(PHENO).years)]
    return [to_season(w, y) for y in yrs], yrs


# ---------------------------------------------------------------------------
# Resource pressure
# ---------------------------------------------------------------------------

def pressure(inst):
    """
    Visit-days demanded divided by visit-days available.

    Below 1 the crew has slack and the shared resource cannot bind; near or
    above 1 it is the binding constraint and the scheduling problem is real.
    """
    v = 3 if inst.split_male else 2
    demand = len(inst.fields) * v
    supply = inst.T * inst.capacity
    return demand / supply


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_block(ph, fields, inst, seasons, ens_idx, eval_idx, sharpness=1.0,
              replan_every=5, tag=""):
    """
    Evaluate the five strategies. The ensemble used for planning and the
    seasons used for evaluation are disjoint sets of real years, so no
    strategy is ever planned on the year it is scored against.
    """
    ens = [seasons[i] for i in ens_idx]
    ens_acc = [access_for(fields, w) for w in ens]

    # The expected-value plan needs a SINGLE deterministic forecast. With a
    # year-indexed crop model we cannot average weather into a synthetic year
    # -- the crop model has no phenology for it -- so the forecast is the one
    # historical year whose thermal accumulation is closest to the ensemble
    # median. That is the natural analogue of a mean season here.
    tot = [float(np.nansum(s["gdd"][inst.w0:inst.w0 + inst.T])) for s in ens]
    typical = ens[int(np.argsort(tot)[len(tot) // 2])]

    p1 = plan_conventional(inst)
    from sim.strategies import yield_table as _yt
    from sim.scheduler import solve_milp as _solve
    _tbl_ev = _yt(inst, [typical], sharpness=sharpness)
    p2 = _solve(inst, _tbl_ev) if _tbl_ev else None
    p3, _ = plan_static_robust(inst, ens, ens_acc, sharpness=sharpness)

    rows = []
    for k in eval_idx:
        w = seasons[k]
        acc = access_for(fields, w)
        p5, _ = plan_wait_and_see(inst, w, acc, sharpness=sharpness)
        rp = make_analogue_replanner(inst, w, acc, ens,
                                     every=replan_every, sharpness=sharpness)
        for name, plan, fn in [("S1_conventional", p1, None),
                               ("S2_static_ev", p2, None),
                               ("S3_static_robust", p3, None),
                               ("S4_dynamic", p3, rp),
                               ("S5_wait_and_see", p5, None)]:
            if plan is None:
                continue
            r = execute(inst, plan, w, acc, replan_fn=fn, sharpness=sharpness)
            tsis = [v[2] for v in r["detail"].values() if v[2] == v[2]]
            rows.append(dict(
                tag=tag, year=w.get("year"), strategy=name,
                sharpness=sharpness, n_fields=len(fields),
                capacity=inst.capacity, pressure=pressure(inst),
                yield_t=r["total_yield"], n_sown=r["n_sown"],
                n_replans=r["n_replans"], changes=r["schedule_changes"],
                mean_abs_tsi=float(np.mean(np.abs(tsis))) if tsis else np.nan,
            ))
    return pd.DataFrame(rows)


def gains(df):
    out = []
    for key, sub in df.groupby(["tag", "n_fields", "capacity", "sharpness"]):
        piv = sub.pivot_table(index="year", columns="strategy", values="yield_t")
        if "S4_dynamic" not in piv or "S3_static_robust" not in piv:
            continue
        a, b = piv["S3_static_robust"], piv["S4_dynamic"]
        d = (b - a).values
        t, p = stats.ttest_rel(b.values, a.values) if len(d) > 2 else (np.nan, np.nan)
        out.append(dict(
            tag=key[0], n_fields=key[1], capacity=key[2], sharpness=key[3],
            pressure=float(sub.pressure.iloc[0]),
            S1=float(piv["S1_conventional"].mean()) if "S1_conventional" in piv else np.nan,
            S3=float(a.mean()), S4=float(b.mean()),
            S5=float(piv["S5_wait_and_see"].mean()) if "S5_wait_and_see" in piv else np.nan,
            gain_t=float(d.mean()),
            gain_pct=float(100 * d.mean() / a.mean()) if a.mean() else np.nan,
            p=float(p), n=len(d),
        ))
    return pd.DataFrame(out)


def _scale_job(args):
    n_fields, capacity, seasons, ens_idx, eval_idx = args
    ph, fields, inst = build_real(n_fields=n_fields, capacity=capacity)
    df = run_block(ph, fields, inst, seasons, ens_idx, eval_idx,
                   tag=f"N{n_fields}_K{capacity}")
    return df


def main():
    t0 = time.time()
    seasons, yrs = load_seasons()
    print(f"real seasons available: {len(yrs)} ({yrs[0]}-{yrs[-1]})")

    # disjoint planning / evaluation years
    ens_idx = list(range(0, len(seasons), 2))        # even years for planning
    eval_idx = [i for i in range(len(seasons)) if i not in ens_idx][:6]
    print(f"planning on {len(ens_idx)} years, evaluating on {len(eval_idx)} "
          f"disjoint years")

    ph, fields, inst = build_real()
    print(f"phenology table: {len(ph.cultivars)} cultivars, "
          f"{len(ph.years)} years, sowing days "
          f"{ph.sow_days[0]}-{ph.sow_days[-1]}")
    print(f"base instance: {len(fields)} fields, capacity {inst.capacity}, "
          f"window {inst.T} d, 3 visits/field, "
          f"resource pressure {pressure(inst):.2f}")
    print()

    # ---------------------------------------------------------------- scale
    print("R1a resource-pressure sweep ...", flush=True)
    jobs = []
    for n_fields, capacity in [(10, 2), (20, 2), (30, 2), (40, 2), (55, 2)]:
        jobs.append((n_fields, capacity, seasons, ens_idx, eval_idx))
    with Pool(N_PROC) as pool:
        parts = pool.map(_scale_job, jobs)
    r1 = pd.concat([p for p in parts if len(p)], ignore_index=True)
    r1.to_csv(f"{RESULTS}/R1_scale.csv", index=False)

    g = gains(r1).sort_values("pressure")
    g.to_csv(f"{RESULTS}/R1_scale_gains.csv", index=False)
    print(g[["n_fields", "capacity", "pressure", "S3", "S4",
             "gain_t", "gain_pct", "p"]].round(3).to_string(index=False))

    # ------------------------------------------------- sharpness on real data
    print("\nR1b sharpness sweep at realistic scale ...", flush=True)
    sweep = []
    for lam in [0.5, 1.0, 2.0, 3.0]:
        ph2, f2, i2 = build_real(n_fields=40, capacity=2)
        d = run_block(ph2, f2, i2, seasons, ens_idx, eval_idx,
                      sharpness=lam, tag=f"L{lam}")
        sweep.append(d)
    r2 = pd.concat([d for d in sweep if len(d)], ignore_index=True)
    r2.to_csv(f"{RESULTS}/R1_sharpness.csv", index=False)
    g2 = gains(r2).sort_values("sharpness")
    g2.to_csv(f"{RESULTS}/R1_sharpness_gains.csv", index=False)
    print(g2[["sharpness", "pressure", "S3", "S4", "gain_t", "gain_pct", "p"]]
          .round(3).to_string(index=False))

    with open(f"{RESULTS}/R1_summary.json", "w") as fh:
        json.dump(dict(
            site="Ranebennur, Karnataka (14.60 N, 75.60 E, 597 m)",
            years=[int(y) for y in yrs],
            n_plan_years=len(ens_idx), n_eval_years=len(eval_idx),
            sow_window=[SOW_START, SOW_END],
            visits_per_field=3,
            scale=g.to_dict("records"),
            sharpness=g2.to_dict("records"),
        ), fh, indent=2)
    print(f"\ndone in {time.time()-t0:.0f}s -> {RESULTS}/R1_*")




# ---------------------------------------------------------------------------
# Analogue-based replanning
# ---------------------------------------------------------------------------

def make_analogue_replanner(inst, true_season, true_access, ens_seasons,
                            every=3, freeze_lead=2, keep=5, sharpness=1.0,
                            improve_tol=0.002):
    """
    Rolling-horizon replanning by historical analogue.

    With a year-indexed crop model the planner cannot invent hybrid weather
    years, so the conditional ensemble is formed differently: each day, the
    season so far is compared with the same stretch of every historical year,
    and the closest `keep` years become the forecast. Early in the window the
    observed record is short and almost every year is a plausible analogue;
    as the season progresses the set narrows. That narrowing is precisely the
    information an adaptive policy has and a static plan does not.

    Non-anticipativity holds: similarity is computed only from days already
    elapsed, and the true season's future is never read.
    """
    from sim.strategies import yield_table, access_for
    from sim.scheduler import solve_milp
    from sim.strategies import _RestrictedInstance

    def analogues(today):
        lo = inst.w0
        if today <= lo:
            return ens_seasons
        a = true_season["gdd"][lo:today]
        b = true_season["rain"][lo:today]
        scored = []
        for s in ens_seasons:
            ga = s["gdd"][lo:today]
            rb = s["rain"][lo:today]
            n = min(len(a), len(ga))
            if n == 0:
                continue
            d = (np.abs(a[:n].cumsum() - ga[:n].cumsum())[-1] / max(n, 1)
                 + 0.15 * abs(b[:n].sum() - rb[:n].sum()) / max(n, 1))
            scored.append((d, s))
        scored.sort(key=lambda t: t[0])
        return [s for _, s in scored[:keep]] or ens_seasons

    def replan(today, pending, done):
        if today % every != 0 or not pending:
            return None

        scen = analogues(today)
        accs = [access_for(inst.fields, s) for s in scen]
        tbl = yield_table(inst, scen, accesses=accs, sharpness=sharpness)
        if not tbl:
            return None

        frozen = set()
        for fid, (d, o) in pending.items():
            nxt = [v for v in inst.visits(d, o) if v >= today]
            if nxt and min(nxt) - today <= freeze_lead:
                frozen.add(fid)

        allowed = set()
        for (fid, d, o) in inst.feasible_choices():
            if fid not in pending:
                continue
            if fid in frozen and (d, o) != pending[fid]:
                continue
            v = [x for x in inst.visits(d, o)]
            rec = done.get(fid, {})
            if rec:                      # part-sown: keep what is in the ground
                continue
            if v and min(v) < today:
                continue
            if any(x <= today and not true_access[fid][x] for x in v):
                continue
            allowed.add((fid, d, o))
        if not allowed:
            return None

        sub = {k: tbl[k] for k in allowed if k in tbl}
        if not sub:
            return None
        new = solve_milp(_RestrictedInstance(inst, allowed), sub,
                         only_fields=list(pending), msg=False)
        if new is None:
            return None
        for fid in pending:
            new.setdefault(fid, pending[fid])

        def score(p):
            return sum(tbl.get((fid,) + p[fid], 0.0) for fid in p)

        if score(new) <= score(pending) * (1.0 + improve_tol):
            return None
        return new

    return replan

if __name__ == "__main__":
    main()
