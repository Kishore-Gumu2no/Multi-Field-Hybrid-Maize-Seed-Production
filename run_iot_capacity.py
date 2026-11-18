"""
E8 -- the capacity-constrained version of E7.

E7 asked what per-field soil-temperature sensing is worth when each field is
free to take its own best schedule. Those figures are upper bounds: a crew with
finite daily capacity cannot give every field its preferred plan, and the
information can only be acted on where the constraint leaves room.

This runs the SAME question through the actual Family C pipeline --
`build.build_candidates` -> `optimise.solve` -> `scoring.score_plan` -- so the
objective coefficient and the execution score are the same number, and the
only thing that differs between arms is what the planner believes each field's
soil offset to be.

Held identical across arms, by construction rather than by assertion:
fields, parent pairs, areas, drainage, weather years, capacity model and
budget, response scenario, objective, candidate enumeration rule, TSI domain
filter. Only `beliefs` differs.

Every sensor reading is simulated. See sim/sensing.py.

Usage:
    python3 run_iot_capacity.py --quick        # one model, 3 seeds, timing
    python3 run_iot_capacity.py                # 4 capacity models x 12 seeds
    python3 run_iot_capacity.py --sigma-sweep  # heterogeneity levels, one model
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim import build, capacity as cap, optimise, response as R, scoring as SC
from sim import sensing as SE, soilthermal as ST
from sim import pairs as P
from sim.phenology import DssatPhenology
from sim.runspec import git_commit, table_sha
from sim.weather_real import read_power_csv

TABLE = "results/phenology_table.npz"
WEATHER = "data/power_ranebennur.csv"
FIDELITY = "records/emulator_fidelity.json"


class OffsetView:
    """
    The Family C phenology interface with one soil offset baked in.

    `scoring.dssat_tsi_fn` needs only `tsi_by_year`, so presenting that is
    enough to drive the whole unmodified pipeline. ADAT cubes are cached per
    (cultivar, offset) because the candidate enumeration asks for the same
    cultivar at the same offset across all 61 sowing days.
    """

    applies_asi = True

    def __init__(self, prov, offset_c, years, sows, cache):
        self.prov, self.offset_c = prov, float(offset_c)
        self.years, self.sows = years, sows
        self._idx = {int(d): i for i, d in enumerate(sows)}
        self._cache = cache
        self.name = f"soil-thermal@{self.offset_c:+.3f}"

    def _cube(self, cult):
        key = (cult, round(self.offset_c, 6))
        hit = self._cache.get(key)
        if hit is None:
            hit = np.array([[self.prov.adat_day(cult, y, d, self.offset_c)
                             for d in self.sows] for y in self.years],
                           dtype=float)
            self._cache[key] = hit
        return hit

    def tsi_by_year(self, cult_m, cult_f, sow_m, sow_f, asi0=5.0, delta_thr=0.0):
        im, if_ = self._idx.get(int(sow_m)), self._idx.get(int(sow_f))
        if im is None or if_ is None:
            return np.full(len(self.years), np.nan)
        return (self._cube(cult_m)[:, im] - float(asi0)
                - self._cube(cult_f)[:, if_] + float(delta_thr))


def load_world():
    df = read_power_csv(WEATHER)
    z = np.load(TABLE, allow_pickle=True)
    years = [int(y) for y in z["years"]]
    sows = [int(d) for d in z["sow_days"]]
    seasons = ST.seasons_from_power(df, years)
    fid = json.load(open(FIDELITY))["soil"]
    prov = ST.SoilThermalPhenology(fid["thresholds"], seasons, pathway="soil")
    return prov, years, sows, fid


def run_one(cfg, prov, years, sows, model_name, n_fields=12, farm_seed=7,
            asi0=P.ASI0_DEFAULT, scenario=R.PRIMARY_SCENARIO,
            objective=optimise.PRIMARY_OBJECTIVE, ph_dssat=None,
            time_limit=60, unconstrained=False):
    """
    One (sensing config, capacity model) cell: three arms through the real
    optimiser, all scored against the same truth.
    """
    days = list(sows)
    model = cap.make(model_name)
    # Field population is built on the DSSAT table so it is identical to E7's
    # and unchanged by this workstream. It is the SAME object for all arms.
    fields = build.build_farm(ph_dssat, n_fields=n_fields, farm_seed=farm_seed,
                              sow_day_ref=sows[len(sows) // 2], asi0=asi0)
    truth = SE.field_offsets(len(fields), cfg)
    hat, n_used = SE.observe(prov.seasons[years[0]], truth, cfg)

    cube_cache = {}
    truth_views = {f.fid: OffsetView(prov, truth[i], years, sows, cube_cache)
                   for i, f in enumerate(fields)}

    rows, per_arm = [], {}
    for arm in SE.ARMS:
        bel = SE.beliefs(arm, truth, hat)
        cands, diag = [], dict(n_enumerated=0, n_window=0, n_access=0, n_domain=0)
        for i, f in enumerate(fields):
            view = OffsetView(prov, bel[i], years, sows, cube_cache)
            c, d = build.build_candidates(view, [f], days, model,
                                          scenario=scenario, asi0=asi0,
                                          split_male=True, split_gap=2, rho=1.0)
            cands.extend(c)
            for k in diag:
                diag[k] += d[k]
        if unconstrained:
            # one best candidate per field, no capacity coupling: the E7 bound
            best = {}
            for c in cands:
                if c.fid not in best or c.area_r > best[c.fid].area_r:
                    best[c.fid] = c
            chosen = list(best.values())
            sol = optimise.Solution(objective, float(sum(c.area_r for c in chosen)),
                                    chosen, "Unconstrained", 0.0, None, len(cands))
        else:
            sol = optimise.solve(cands, model.capacity, days,
                                 objective=objective, time_limit=time_limit)

        # EXECUTION under truth, through the same scorer the coefficients used.
        true_area_r, n_ood, tsi_abs = 0.0, 0, []
        for c in sol.chosen:
            f = fields[c.fid]
            tfn = SC.dssat_tsi_fn(truth_views[f.fid], f, asi0=asi0,
                                  delta_thr=scenario.delta_thr)
            s = SC.score_plan(tfn, f.area_ha, c.plan, scenario)
            true_area_r += float(s.area_r)
            n_ood += int(s.out_of_domain)
            t = np.asarray(tfn(c.plan.sow_m1, c.plan.sow_f), dtype=float)
            t = t[np.isfinite(t)]
            if t.size:
                tsi_abs.append(float(np.mean(np.abs(t))))
            rows.append(dict(arm=arm, fid=c.fid, label=c.label,
                             sow_f=c.plan.sow_f, sow_m1=c.plan.sow_m1,
                             sow_m2=c.plan.sow_m2, area_ha=c.area_ha,
                             believed_area_r=c.area_r, true_area_r=float(s.area_r),
                             true_r_mean=float(s.r_mean),
                             true_tsi_mean=float(s.tsi_mean),
                             out_of_domain=int(s.out_of_domain),
                             merged=bool(c.merged),
                             true_offset_c=float(truth[c.fid]),
                             belief_offset_c=float(bel[c.fid])))
        per_arm[arm] = dict(
            believed_objective=float(sol.objective), true_objective=true_area_r,
            n_sown=int(sol.n_sown), ha_sown=float(sol.ha_sown),
            merged_fraction=float(sol.merged_fraction),
            n_candidates=int(sol.n_candidates), n_dropped=int(sol.n_dropped_outside_window),
            status=str(sol.status), gap=(None if sol.gap is None else float(sol.gap)),
            termination=str(sol.termination), is_lower_bound=bool(sol.is_lower_bound),
            mean_abs_tsi=(float(np.mean(tsi_abs)) if tsi_abs else float("nan")),
            out_of_domain_field_years=int(n_ood),
            n_enumerated=diag["n_enumerated"], n_rejected_domain=diag["n_domain"])

    base = per_arm["conventional"]["true_objective"]
    gain = {}
    for arm in ("sensor", "oracle"):
        v = per_arm[arm]["true_objective"]
        gain[f"{arm}_pct"] = float(100.0 * (v - base) / base) if base else float("nan")
    d_sen = per_arm["sensor"]["true_objective"] - base
    d_ora = per_arm["oracle"]["true_objective"] - base
    gain["sensor_share_of_oracle"] = (float(d_sen / d_ora) if abs(d_ora) > 1e-12
                                      else float("nan"))
    conv_plans = {(r["fid"], r["sow_f"], r["sow_m1"]) for r in rows if r["arm"] == "conventional"}
    for arm in ("sensor", "oracle"):
        ap = {(r["fid"], r["sow_f"], r["sow_m1"]) for r in rows if r["arm"] == arm}
        gain[f"plans_changed_{arm}"] = int(len(conv_plans - ap))
    return rows, per_arm, gain


def write_rows(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sigma-sweep", action="store_true")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--skip-constrained", action="store_true")
    ap.add_argument("--models", default="visits,operations,hours,hectares")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--n-fields", type=int, default=60)
    ap.add_argument("--time-limit", type=int, default=60)
    ap.add_argument("--out", default="results_cap")
    a = ap.parse_args()

    prov, years, sows, fid = load_world()
    ph_dssat = DssatPhenology(TABLE)
    all_rows, all_sum = [], []

    def do(tag, cfg, model_name, unconstrained=False):
        t0 = time.time()
        rows, per_arm, gain = run_one(cfg, prov, years, sows, model_name,
                                      n_fields=a.n_fields, ph_dssat=ph_dssat,
                                      time_limit=a.time_limit,
                                      unconstrained=unconstrained)
        el = time.time() - t0
        for r in rows:
            r.update(tag=tag, capacity_model=model_name,
                     unconstrained=unconstrained, **cfg.as_row())
        all_rows.extend(rows)
        rec = dict(tag=tag, capacity_model=model_name,
                   unconstrained=unconstrained, elapsed_s=round(el, 2),
                   **cfg.as_row())
        for arm, d in per_arm.items():
            rec.update({f"{arm}__{k}": v for k, v in d.items()})
        rec.update({f"gain__{k}": v for k, v in gain.items()})
        all_sum.append(rec)
        print(f"  {tag:34s} {model_name:11s} sensor {gain['sensor_pct']:+7.3f}%  "
              f"oracle {gain['oracle_pct']:+7.3f}%  sown "
              f"{per_arm['conventional']['n_sown']:2d}/{a.n_fields}  "
              f"[{el:.0f}s]")
        write_rows(os.path.join(a.out, "E8_capacity_summary.csv"), all_sum)
        write_rows(os.path.join(a.out, "E8_capacity_rows.csv"), all_rows)

    models = [m.strip() for m in a.models.split(",") if m.strip()]

    if a.quick:
        for m in models[:1]:
            for sd in (11, 12):
                do(f"quick_seed{sd}", SE.SensingConfig(seed=sd), m)
                do(f"quick_unc_seed{sd}", SE.SensingConfig(seed=sd), m,
                   unconstrained=True)
        return

    if a.ablations:
        print("NOISE SWEEP, capacity-constrained, visits model")
        for nz in (0.0, 0.25, 0.5, 1.0, 2.0):
            for sd in range(11, 11 + a.seeds):
                do(f"abl_noise{nz}_seed{sd}", SE.SensingConfig(noise_c=nz, seed=sd),
                   "visits")
        print("MISSINGNESS SWEEP, capacity-constrained, visits model")
        for pm in (0.0, 0.25, 0.5, 0.75, 1.0):
            for sd in range(11, 11 + a.seeds):
                do(f"abl_miss{pm}_seed{sd}", SE.SensingConfig(p_missing=pm, seed=sd),
                   "visits")
        return

    if a.sigma_sweep:
        print("HETEROGENEITY SWEEP, capacity-constrained, visits model")
        for s in (0.0, 0.5, 0.7, 1.0, 2.0, 4.0):
            for sd in range(11, 11 + a.seeds):
                do(f"cap_sigma{s}_seed{sd}", SE.SensingConfig(sigma_c=s, seed=sd),
                   "visits")
        return

    print("PRIMARY: sigma = 0.7, capacity models, constrained vs unconstrained")
    if not a.skip_constrained:
        for m in models:
            for sd in range(11, 11 + a.seeds):
                do(f"cap_{m}_seed{sd}", SE.SensingConfig(seed=sd), m)
    print("UNCONSTRAINED REFERENCE (the E7 bound, same pipeline)")
    for sd in range(11, 11 + a.seeds):
        do(f"unc_seed{sd}", SE.SensingConfig(seed=sd), models[0],
           unconstrained=True)

    with open(os.path.join(a.out, "E8_provenance.json"), "w") as fh:
        json.dump(dict(git_commit=git_commit("."), table_sha=table_sha(TABLE),
                       weather=WEATHER, objective=optimise.PRIMARY_OBJECTIVE,
                       scenario=R.PRIMARY_SCENARIO.name,
                       emulator_mean_rmse_days=fid["mean_rmse_days"],
                       simulated_sensing=True, n_years=len(years)), fh, indent=2)
    print(f"\nwrote {a.out}/E8_capacity_summary.csv ({len(all_sum)} cells)")


if __name__ == "__main__":
    main()
