"""Recompute every confidence interval from per-field rows, not from summaries."""
import csv, re, json, collections
import numpy as np

def gains_from_rows(path):
    tot = collections.defaultdict(float)
    for r in csv.DictReader(open(path)):
        col = 'true_area_r' if 'true_area_r' in r else 'area_r'
        tot[(r['tag'], r['arm'])] += float(r[col])
    out = {}
    for t in sorted({k[0] for k in tot}):
        base = tot[(t, 'conventional')]
        if base:
            out[t] = {a: 100.0 * (tot[(t, a)] - base) / base
                      for a in ('sensor', 'oracle')}
    return out

def ci(v):
    v = np.asarray(v, float)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    return m, m - 1.96 * se, m + 1.96 * se

def table(gains, pattern, label):
    by = collections.defaultdict(list)
    for t, v in gains.items():
        m = re.match(pattern, t)
        if m:
            by[float(m.group(1))].append((v['sensor'], v['oracle']))
    if not by:
        return {}
    print(f"\n{label}")
    print(f"{'level':>8} {'n':>3} | {'sensor':>8} {'95% CI':>20} {'>0':>6} | {'oracle':>8}")
    out = {}
    for k in sorted(by):
        a = np.array(by[k]); ms, lo, hi = ci(a[:, 0]); mo = a[:, 1].mean()
        print(f"{k:8.2f} {len(a):3d} | {ms:+8.3f} [{lo:+7.3f},{hi:+7.3f}] "
              f"{int((a[:,0]>0).sum()):3d}/{len(a)} | {mo:+8.3f}")
        out[f"{k:g}"] = dict(n=len(a), sensor_mean=float(ms),
                             sensor_ci=[float(lo), float(hi)],
                             seeds_positive=int((a[:, 0] > 0).sum()),
                             oracle_mean=float(mo))
    return out

if __name__ == "__main__":
    import os
    res = {}
    if os.path.exists('results_iot/E7_replicated_rows.csv'):
        g = gains_from_rows('results_iot/E7_replicated_rows.csv')
        res['e7_sigma'] = table(g, r'rep_sigma([\d.]+)_seed', 'E7 unconstrained, N=12')
    for path, pat, lab, key in (
            ('results_cap_sigma/E8_capacity_rows.csv', r'cap_sigma([\d.]+)_seed',
             'E8 capacity-constrained, N=60', 'e8_sigma'),
            ('results_cap_abl/E8_capacity_rows.csv', r'abl_noise([\d.]+)_seed',
             'E8 noise', 'e8_noise'),
            ('results_cap_abl/E8_capacity_rows.csv', r'abl_miss([\d.]+)_seed',
             'E8 missingness', 'e8_missing')):
        if os.path.exists(path):
            res[key] = table(gains_from_rows(path), pat, lab)
    json.dump(res, open('records/ci_recomputed_all.json', 'w'), indent=2)
    print("\nwrote records/ci_recomputed_all.json")
