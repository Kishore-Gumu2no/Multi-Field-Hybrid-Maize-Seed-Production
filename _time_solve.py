import time, numpy as np, sys
from run_real import build_real, load_seasons
from run_real_rep import ENS_CAP
from sim.strategies import access_for, plan_static_robust, plan_agronomic, yield_table
from sim.scheduler import solve_milp, execute
seasons, yrs = load_seasons()
print("seasons", len(seasons), yrs)
k = 0
others = [s for i,s in enumerate(seasons) if i!=k]
rs = np.random.default_rng(1000+k)
ens = [others[i] for i in sorted(rs.choice(len(others), size=ENS_CAP, replace=False))]
for N in [10, 40, 70]:
    _, fields, inst = build_real(n_fields=N, capacity=2, farm_seed=1)
    ens_acc = [access_for(fields, s) for s in ens]
    t0=time.time(); tbl = yield_table(inst, ens, accesses=ens_acc); t1=time.time()
    plan = solve_milp(inst, tbl); t2=time.time()
    print(f"N={N}: table {t1-t0:.1f}s ({len(tbl)} entries), CBC {t2-t1:.1f}s, placed {len(plan)}")
