"""
E7 -- does simulated soil-temperature sensing change the sowing schedule?

Three planners differing ONLY in what they believe each field's soil thermal
offset to be. All three are scored against the same truth.

  conventional  believes every offset is zero, which is exactly what CERES
                does: MZ_PHENOL.for imputes soil temperature from air and
                radiation with no field-specific term.
  sensor        believes a noisy estimate from a simulated pre-season probe.
  oracle        believes the truth. Not achievable; it bounds the gain.

Every number produced here is simulated. See sim/sensing.py.

Usage:
    python3 run_iot.py                 # primary configuration
    python3 run_iot.py --sweep         # full sensitivity and ablation grid
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim import pairs as P
from sim import response as R
from sim import scoring as SC
from sim import sensing as SE
from sim import soilthermal as ST
from sim.phenology import DssatPhenology
from sim.runspec import git_commit, table_sha
from sim.weather_real import read_power_csv

TABLE = "results/phenology_table.npz"
WEATHER = "data/power_ranebennur.csv"
FIDELITY = "records/emulator_fidelity.json"


# --------------------------------------------------------------------------

def load_world(pathway="soil"):
    """Phenology provider, season weather and the decision grid."""
    df = read_power_csv(WEATHER)
    z = np.load(TABLE, allow_pickle=True)
    cults = [str(c) for c in z["cultivars"]]
    years = [int(y) for y in z["years"]]
    sows = [int(d) for d in z["sow_days"]]
    seasons = ST.seasons_from_power(df, years)
    with open(FIDELITY) as fh:
        fid = json.load(fh)
    prov = ST.SoilThermalPhenology(fid[pathway]["thresholds"], seasons,
                                   pathway=pathway)
    return prov, cults, years, sows, fid[pathway]


def adat_cube(prov, cult, years, sows, offset_c):
    """(n_years, n_sow_days) calendar ADAT days at one field offset."""
    out = np.empty((len(years), len(sows)), dtype=float)
    for yi, y in enumerate(years):
        for di, d in enumerate(sows):
            out[yi, di] = prov.adat_day(cult, y, d, offset_c)
    return out


class FieldPhenoCache:
    """ADAT cubes for one field's two cultivars at the offsets it needs."""

    def __init__(self, prov, cult_m, cult_f, years, sows):
        self.prov, self.cult_m, self.cult_f = prov, cult_m, cult_f
        self.years, self.sows = years, sows
        self._c = {}
        self._idx_set = set(int(d) for d in sows)

    def cubes(self, offset_c):
        key = round(float(offset_c), 6)
        hit = self._c.get(key)
        if hit is None:
            hit = (adat_cube(self.prov, self.cult_m, self.years, self.sows, key),
                   adat_cube(self.prov, self.cult_f, self.years, self.sows, key))
            self._c[key] = hit
        return hit

    def tsi_fn(self, offset_c, asi0, delta_thr=0.0):
        """`(sow_m, sow_f) -> per-year TSI`, the callable score_plan consumes."""
        am, af = self.cubes(offset_c)
        idx = {d: i for i, d in enumerate(self.sows)}

        def fn(sow_m, sow_f):
            im, if_ = idx.get(int(sow_m)), idx.get(int(sow_f))
            if im is None or if_ is None:
                return np.full(len(self.years), np.nan)
            return am[:, im] - float(asi0) - af[:, if_] + float(delta_thr)
        return fn


def best_plan(cache, sows, offsets, area_ha, belief_offset, asi0,
              scenario=R.PRIMARY_SCENARIO):
    """Highest-scoring (sow_f, sow_m) under the planner's BELIEF."""
    fn = cache.tsi_fn(belief_offset, asi0)
    best, best_v = None, -np.inf
    for sf in sows:
        for off in offsets:
            sm = sf + off
            if sm not in cache._idx_set:
                continue
            plan = SC.SowingPlan(sow_f=int(sf), sow_m1=int(sm))
            s = SC.score_plan(fn, area_ha, plan, scenario=scenario, mode="reflect")
            v = s.area_r
            if np.isfinite(v) and v > best_v:
                best, best_v = plan, float(v)
    return best, best_v


# --------------------------------------------------------------------------

def run_one(cfg, prov, cults, years, sows, n_fields=12, farm_seed=7,
            asi0=P.ASI0_DEFAULT, offsets=tuple(range(-4, 7)),
            scenario=R.PRIMARY_SCENARIO, ph_dssat=None):
    """
    One configuration: build the farm, run all three arms, score against truth.
    Returns (rows, summary).
    """
    # Population: the manuscript's own stratified pairs, chosen on the DSSAT
    # table so the field population is identical across arms and unchanged by
    # this workstream.
    rng = np.random.default_rng(farm_seed)
    chosen = P.stratified_pairs(ph_dssat, n_fields, rng, sows[len(sows)//2],
                                asi0=asi0, scope="years", rho=1.0)
    areas = np.array([float(rng.uniform(1.5, 4.0)) for _ in chosen])

    truth = SE.field_offsets(len(chosen), cfg)
    season_ref = prov.seasons[years[0]]
    hat, n_used = SE.observe(season_ref, truth, cfg)

    caches = [FieldPhenoCache(prov, p.cult_m, p.cult_f, years, sows)
              for p in chosen]

    rows = []
    for arm in SE.ARMS:
        bel = SE.beliefs(arm, truth, hat)
        for i, (p, cache) in enumerate(zip(chosen, caches)):
            plan, believed = best_plan(cache, sows, offsets, float(areas[i]),
                                       float(bel[i]), asi0, scenario)
            if plan is None:
                continue
            # Score the SAME plan against the truth.
            tfn = cache.tsi_fn(float(truth[i]), asi0)
            s_true = SC.score_plan(tfn, float(areas[i]), plan,
                                   scenario=scenario, mode="reflect")
            tsis = tfn(plan.sow_m1, plan.sow_f)
            fin = tsis[np.isfinite(tsis)]
            rows.append(dict(
                arm=arm, fid=i, pair=p.label(), area_ha=float(areas[i]),
                true_offset_c=float(truth[i]), belief_offset_c=float(bel[i]),
                delta_hat_c=float(hat[i]) if np.isfinite(hat[i]) else np.nan,
                n_obs_used=int(n_used[i]),
                sow_f=int(plan.sow_f), sow_m1=int(plan.sow_m1),
                offset_days=int(plan.sow_m1 - plan.sow_f),
                believed_area_r=float(believed),
                area_r=float(s_true.area_r), r_mean=float(s_true.r_mean),
                tsi_mean=float(s_true.tsi_mean),
                abs_tsi_err=float(np.mean(np.abs(fin))) if fin.size else np.nan,
                n_in_domain=int(s_true.n_in_domain), n_years=int(s_true.n_years),
            ))

    summary = {}
    for arm in SE.ARMS:
        sel = [r for r in rows if r["arm"] == arm]
        if not sel:
            continue
        tot_area = sum(r["area_ha"] for r in sel)
        summary[arm] = dict(
            n_fields=len(sel),
            total_area_r=float(sum(r["area_r"] for r in sel)),
            mean_r=float(np.mean([r["r_mean"] for r in sel])),
            area_weighted_r=float(sum(r["area_r"] for r in sel) / tot_area),
            mean_abs_tsi=float(np.nanmean([r["abs_tsi_err"] for r in sel])),
            in_domain_frac=float(sum(r["n_in_domain"] for r in sel) /
                                 max(1, sum(r["n_years"] for r in sel))),
            mean_offset_days=float(np.mean([r["offset_days"] for r in sel])),
        )
    if "conventional" in summary and "sensor" in summary:
        c, s_, o = (summary["conventional"], summary["sensor"],
                    summary.get("oracle", summary["sensor"]))
        base = c["total_area_r"]
        summary["gain"] = dict(
            sensor_vs_conventional_pct=float(100.0 * (s_["total_area_r"] - base) / base)
            if base else float("nan"),
            oracle_vs_conventional_pct=float(100.0 * (o["total_area_r"] - base) / base)
            if base else float("nan"),
            sensor_share_of_oracle=float(
                (s_["total_area_r"] - base) / (o["total_area_r"] - base))
            if abs(o["total_area_r"] - base) > 1e-12 else float("nan"),
            plans_changed_sensor=int(sum(
                1 for a, b in zip([r for r in rows if r["arm"] == "conventional"],
                                  [r for r in rows if r["arm"] == "sensor"])
                if (a["sow_f"], a["sow_m1"]) != (b["sow_f"], b["sow_m1"]))),
            plans_changed_oracle=int(sum(
                1 for a, b in zip([r for r in rows if r["arm"] == "conventional"],
                                  [r for r in rows if r["arm"] == "oracle"])
                if (a["sow_f"], a["sow_m1"]) != (b["sow_f"], b["sow_m1"]))),
        )
    return rows, summary


# --------------------------------------------------------------------------

def write_rows(path, rows):
    import csv
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--replicate", action="store_true")
    ap.add_argument("--sigmas", default="")
    ap.add_argument("--operational", action="store_true")
    ap.add_argument("--n-fields", type=int, default=12)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    prov, cults, years, sows, fid = load_world("soil")
    ph_dssat = DssatPhenology(TABLE)
    prov_air, *_ = load_world("air")

    provenance = dict(git_commit=git_commit("."), table_sha=table_sha(TABLE),
                      weather=WEATHER, pathway="soil",
                      emulator_mean_rmse_days=fid["mean_rmse_days"],
                      emulator_mean_abs_bias_days=fid["mean_abs_bias_days"],
                      n_years=len(years), n_sow_days=len(sows),
                      simulated_sensing=True)

    all_rows, all_sum = [], []

    def do(tag, cfg, provider=prov, **kw):
        rows, summ = run_one(cfg, provider, cults, years, sows,
                             n_fields=a.n_fields, ph_dssat=ph_dssat, **kw)
        for r in rows:
            r.update(tag=tag, **cfg.as_row())
        all_rows.extend(rows)
        rec = dict(tag=tag, **cfg.as_row())
        rec.update({f"{k}__{kk}": vv for k, v in summ.items()
                    if isinstance(v, dict) for kk, vv in v.items()})
        all_sum.append(rec)
        g = summ.get("gain", {})
        print(f"  {tag:34s} sensor {g.get('sensor_vs_conventional_pct', float('nan')):+6.3f}%  "
              f"oracle {g.get('oracle_vs_conventional_pct', float('nan')):+6.3f}%  "
              f"plans changed {g.get('plans_changed_sensor', 0):2d}/{a.n_fields}")
        return summ

    print("PRIMARY")
    primary = SE.SensingConfig()
    do("primary", primary)

    print("NULL SCENARIO (sigma = 0: the sensor must show nothing)")
    do("null_homogeneous", SE.SensingConfig(scenario="homogeneous"))
    do("null_sigma0", SE.SensingConfig(scenario="normal", sigma_c=0.0))

    if a.replicate:
        print("REPLICATED SIGMA GRID (12 farm/offset seeds per level)")
        levels = ([float(x) for x in a.sigmas.split(",")] if a.sigmas
                  else [0.0, 0.5, 0.7, 1.0, 2.0, 4.0])
        for s in levels:
            for sd in range(11, 23):
                do(f"rep_sigma{s}_seed{sd}", SE.SensingConfig(sigma_c=s, seed=sd))
                # write incrementally so a stopped run keeps its results
                write_rows(os.path.join(a.out, "E7_iot_summary.csv"), all_sum)

    if a.operational:
        print("OPERATIONAL FEASIBILITY: narrowing the male-female offset freedom")
        for lo, hi, name in ((-4, 7, "full"), (-2, 5, "tight"), (-1, 3, "tighter"),
                             (0, 2, "minimal"), (0, 1, "co-sown only")):
            for sd in (11, 12, 13, 14, 15, 16):
                do(f"op_{name}_seed{sd}", SE.SensingConfig(seed=sd),
                   offsets=tuple(range(lo, hi)))
                write_rows(os.path.join(a.out, "E7_iot_summary.csv"), all_sum)

    if a.sweep:
        print("SIGMA SWEEP")
        for s in (0.0, 0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0):
            do(f"sigma_{s}", SE.SensingConfig(sigma_c=s))
        print("NOISE SWEEP (sigma held at the default)")
        for nz in (0.0, 0.25, 0.5, 1.0, 2.0):
            do(f"noise_{nz}", SE.SensingConfig(noise_c=nz))
        print("MISSINGNESS SWEEP")
        for pm in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
            do(f"missing_{pm}", SE.SensingConfig(p_missing=pm))
        print("OBSERVATION LENGTH SWEEP")
        for nd in (1, 3, 5, 10, 20, 40):
            do(f"nobs_{nd}", SE.SensingConfig(n_obs_days=nd))
        print("HETEROGENEITY SHAPE")
        for sc in ("normal", "bimodal", "gradient"):
            do(f"shape_{sc}", SE.SensingConfig(scenario=sc))
        print("SEED REPLICATES")
        for sd in (11, 12, 13, 14, 15, 16, 17, 18):
            do(f"seed_{sd}", SE.SensingConfig(seed=sd))
        print("ABLATION: air pathway instead of soil")
        do("ablation_air_pathway", SE.SensingConfig(), provider=prov_air)

    write_rows(os.path.join(a.out, "E7_iot_rows.csv"), all_rows)
    write_rows(os.path.join(a.out, "E7_iot_summary.csv"), all_sum)
    with open(os.path.join(a.out, "E7_provenance.json"), "w") as fh:
        json.dump(provenance, fh, indent=2)
    print(f"\nwrote {a.out}/E7_iot_rows.csv ({len(all_rows)} rows), "
          f"{a.out}/E7_iot_summary.csv ({len(all_sum)} configs)")


if __name__ == "__main__":
    main()
