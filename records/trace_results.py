"""Mechanical provenance: result file -> writing runner -> engine family."""
import ast, os, re, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = {"core", "scheduler", "strategies", "baselines"}
C = {"build", "optimise", "pairs", "response", "runspec", "scoring",
     "capacity", "pollen", "phenology", "compare", "sensing", "soilthermal"}

def mods(path):
    try: t = ast.parse(open(path).read())
    except Exception: return set()
    out = set()
    for n in ast.walk(t):
        if isinstance(n, ast.ImportFrom) and n.module and "sim" in (n.module or ""):
            for al in n.names: out.add(al.name)
            if n.module.startswith("sim."): out.add(n.module.split(".")[-1])
        if isinstance(n, ast.Import):
            for al in n.names:
                if al.name.startswith("sim."): out.add(al.name.split(".")[-1])
    return out

def family(ms):
    a, c = ms & A, ms & C
    if a and c: return "MIXED"
    if a: return "FamilyA"
    if c: return "FamilyC"
    return "none"

pat = re.compile(r"((?:R\d|E\d|V1)_[A-Za-z0-9_]*\.(?:csv|json))")
rows = []
for f in sorted(os.listdir(ROOT)):
    if not f.endswith(".py"): continue
    p = os.path.join(ROOT, f)
    ms = mods(p); fam = family(ms)
    for m in pat.finditer(open(p).read()):
        rows.append((m.group(1), f, fam))
# Family C runners write into results_* dirs, not R*/E* names
for f in ("run_iot.py", "run_iot_capacity.py", "run_familyc_main.py"):
    p = os.path.join(ROOT, f)
    if os.path.exists(p):
        rows.append((f"<{f.replace('run_','').replace('.py','')} outputs>", f, family(mods(p))))

seen, out = set(), []
for r in sorted(set(rows)):
    out.append(r)
print(f"{'result file':34s} {'written by':24s} engine")
print("-" * 72)
for rf, w, fam in out:
    print(f"{rf:34s} {w:24s} {fam}")
cnt = {}
for _, _, fam in out: cnt[fam] = cnt.get(fam, 0) + 1
print("\ncounts:", cnt)
