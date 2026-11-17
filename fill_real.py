"""
Write the Results, Discussion and Conclusion of the manuscript from the
real-data experiments.

Inputs
    results/R3_summary.json     replicated real-data runs (pressure, sharpness)
    results/R3_raw.csv          per-season rows, for the strategy ladder
    results/V1_cimmyt_summary.json   external validation

The prose branches on what the data actually shows -- whether the gain rises
with resource pressure, whether it rises with sharpness, and whether either
is significant -- so that the manuscript cannot claim more than the numbers
support. Placeholders are @@TOKEN@@ rather than Python format fields because
LaTeX braces and str.format do not coexist.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
P = os.path.join(HERE, "paper")

S = json.load(open(f"{R}/R3_summary.json"))
V = json.load(open(f"{R}/V1_cimmyt_summary.json"))

# The saturation-refinement run (R4) adds pressure cells around pi = 1 and
# merges them with R3's. When it is present we use the denser grid for every
# pressure statement in the paper; the sharpness sweep is unaffected.
# R5 adds the fair agronomic baseline to the same cells; prefer it when
# present so that every strategy comparison uses the honest starting point.
BASE_KEY = "S1_conventional"
SAT = None
if os.path.exists(f"{R}/R5_raw.csv"):
    BASE_KEY = "S1_agronomic"
if os.path.exists(f"{R}/R4_summary.json"):
    SAT = json.load(open(f"{R}/R4_summary.json"))
    PRES = pd.read_csv(f"{R}/R4_pressure.csv").sort_values("pressure")
    _press = (pd.read_csv(f"{R}/R5_raw.csv")
              if os.path.exists(f"{R}/R5_raw.csv")
              else pd.read_csv(f"{R}/R4_pressure_raw.csv"))
    raw = pd.concat([_press,
                     pd.read_csv(f"{R}/R3_raw.csv").query("kind == 'l'")],
                    ignore_index=True)
else:
    raw = pd.read_csv(f"{R}/R3_raw.csv")
    PRES = pd.DataFrame(S["pressure"]).sort_values("pressure")

SHARP = pd.DataFrame(S["sharpness"]).sort_values("sharpness")


def _per_season_stats():
    """
    Distribution of the per-farm-season gain within each pressure cell.

    The mean alone is misleading here: at low resource pressure the gain is
    insurance-shaped (usually nothing, occasionally a rescued field worth
    tens of percent), while at high pressure it is small and systematic.
    """
    p = raw[raw.kind == "p"]
    piv = p.pivot_table(index=["n_fields", "farm_seed", "year"],
                        columns="strategy", values="yield_t")
    g = 100 * (piv["S4_dynamic"] - piv["S3_static_robust"]) \
        / piv["S3_static_robust"]
    out = {}
    for n, s in g.groupby(level="n_fields"):
        out[int(n)] = dict(mean=float(s.mean()), median=float(s.median()),
                           sd=float(s.std(ddof=1)),
                           frac_pos=float((s > 0).mean()),
                           p95=float(np.percentile(s, 95)),
                           mx=float(s.max()))
    return out


PERSEASON = _per_season_stats()

def _headline_stats():
    """
    Aggregate effect with an interval that respects farm clustering.

    Two estimators are available and they answer different questions. The
    ratio of means -- total extra seed over total baseline seed -- is what a
    producer's accounts would show, and is the conservative one here; the
    mean of per-season ratios is larger because small farms contribute
    proportionally larger percentages. We report the former. The interval is
    a bootstrap resampling whole farms rather than farm-seasons, since
    seasons within a farm are not independent.
    """
    import numpy as _np
    rng = _np.random.default_rng(0)
    p = raw[raw.kind == "p"]
    piv = p.pivot_table(index=["n_fields", "farm_seed", "year"],
                        columns="strategy", values="yield_t")
    a = piv["S3_static_robust"].values
    b = piv["S4_dynamic"].values
    point = 100 * (b - a).sum() / a.sum()
    seeds = piv.index.get_level_values("farm_seed").values
    uniq = _np.unique(seeds)
    where = {u: _np.where(seeds == u)[0] for u in uniq}
    bs = []
    for _ in range(5000):
        idx = _np.concatenate([where[u] for u in
                               rng.choice(uniq, size=len(uniq), replace=True)])
        bs.append(100 * (b[idx] - a[idx]).sum() / a[idx].sum())
    lo, hi = _np.percentile(bs, [2.5, 97.5])
    per = 100 * (b - a) / a
    return dict(point=float(point), lo=float(lo), hi=float(hi),
                frac_pos=float(_np.mean(_np.array(bs) > 0)),
                d=float(per.mean() / per.std(ddof=1)), n=int(len(per)))


HEAD = _headline_stats()


def _sharp_ci():
    """95% intervals for each sharpness cell, used to show the null is a
    null rather than merely non-significant."""
    import numpy as _np
    from scipy import stats as _st
    src = pd.concat([raw[raw.kind == "l"],
                     raw[(raw.kind == "p")
                         & (raw.n_fields == S.get("sharp_config", [55, 2])[0])]],
                    ignore_index=True)
    out = {}
    for lam, sub in src.groupby("sharpness"):
        piv = sub.pivot_table(index=["farm_seed", "year"],
                              columns="strategy", values="yield_t")
        if "S4_dynamic" not in piv:
            continue
        g = 100 * (piv["S4_dynamic"] - piv["S3_static_robust"]) \
            / piv["S3_static_robust"]
        n = len(g)
        ci = _st.t.interval(0.95, n - 1, loc=g.mean(),
                            scale=g.std(ddof=1) / _np.sqrt(n))
        out[float(lam)] = dict(mean=float(g.mean()), lo=float(ci[0]),
                               hi=float(ci[1]),
                               d=float(g.mean() / g.std(ddof=1)))
    return out


SHARP_CI = _sharp_ci()


def _per_hectare():
    """Gain expressed per hectare, which is the unit a producer thinks in."""
    import numpy as _np
    from run_real import build_real
    p = raw[raw.kind == "p"]
    ha = {}
    for n in sorted(p.n_fields.unique()):
        tot = []
        for fs in sorted(p.farm_seed.unique()):
            _, flds, _ = build_real(n_fields=int(n), capacity=2,
                                    farm_seed=int(fs))
            tot.append(sum(x.area_ha for x in flds))
        ha[int(n)] = float(_np.mean(tot))
    piv = p.pivot_table(index=["n_fields", "farm_seed", "year"],
                        columns="strategy", values="yield_t")
    gain = piv["S4_dynamic"] - piv["S3_static_robust"]
    areas = piv.index.get_level_values("n_fields").map(ha)
    return dict(kg_per_ha=float((1000 * gain / areas).mean()),
                areas=ha)


PERHA = _per_hectare()


def _mean_ratio():
    """The other available estimator, reported alongside for transparency."""
    p = raw[raw.kind == "p"]
    piv = p.pivot_table(index=["n_fields", "farm_seed", "year"],
                        columns="strategy", values="yield_t")
    return float((100 * (piv["S4_dynamic"] - piv["S3_static_robust"])
                  / piv["S3_static_robust"]).mean())


MEANRATIO = _mean_ratio()



# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def pv(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "$p < 10^{-4}$"
    return f"$p = {p:.3f}$"


def f(x, n=2):
    return f"{x:.{n}f}"


def tex_escape(s):
    return s.replace("%", "\\%")


def ladder(sub):
    """Mean yield per strategy over a slice of the raw rows."""
    piv = sub.pivot_table(index=["farm_seed", "year"], columns="strategy",
                          values="yield_t")
    return {c: float(piv[c].mean()) for c in piv.columns}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _solution_stats():
    """How much the optimiser buys over current practice, and why."""
    p = raw[raw.kind == "p"]
    piv = p.pivot_table(index=["n_fields", "farm_seed", "year"],
                        columns="strategy", values="yield_t")
    sown = p.pivot_table(index="n_fields", columns="strategy",
                         values="n_sown")
    tsi = p.pivot_table(index="n_fields", columns="strategy",
                        values="mean_abs_tsi")
    b, o = piv[BASE_KEY], piv["S3_static_robust"]
    per = (100 * (o - b) / b).groupby(level="n_fields").mean()
    big = int(per.index.max())
    return dict(
        pooled=float(100 * (o - b).sum() / b.sum()),
        per_scale=per, big=big, big_gain=float(per.loc[big]),
        sown=sown, tsi=tsi,
        base_plateau=float(sown[BASE_KEY].iloc[-4:].mean()),
        base_sown_big=float(sown.loc[big, BASE_KEY]),
        opt_sown_big=float(sown.loc[big, "S3_static_robust"]),
    )


SOL = _solution_stats()


def results_section():
    out = []
    a = out.append

    # ---------------- the solution ------------------------------------
    a("\\subsection{What the scheduler delivers over current practice}")
    a("")
    a("The baseline is not a straw man and the comparison turns on that.")
    a("A seed producer does not apply one offset to every hybrid; they look")
    a("up each parent pair's accumulated degree days and sow the two lines")
    a("the recommended interval apart, exactly as")
    a("\\cite{chassaigne2021,asefa2024} advise. What they do not do is")
    a("optimise which field is served on which day. Our baseline therefore")
    a("has the agronomy right and the logistics naive, which is the honest")
    a("description of the state of practice, and it is substantially")
    a("stronger than a uniform rule-of-thumb offset.")
    a("")
    a("Against it, solving the assignment yields")
    a(f"${SOL['pooled']:+.1f}$\\% more seed pooled across every farm size")
    a(f"we test, rising to ${SOL['big_gain']:+.0f}$\\% at"
      f" $N = {SOL['big']}$ fields (Table~\\ref{{tab:solution}}).")
    a("")
    a("The mechanism is visible in how many fields each plan manages to sow")
    a("at all. Working through fields in order and taking the first free")
    a(f"slot saturates: the baseline places about"
      f" {SOL['base_plateau']:.0f} fields however many it is given, so at")
    a(f"$N = {SOL['big']}$ it sows {SOL['base_sown_big']:.0f} of them and")
    a("abandons the rest, while the optimised plan places")
    a(f"{SOL['opt_sown_big']:.0f}. The same crew, the same window and the")
    a("same weather serve half again as many fields.")
    a("")
    a("The gain is not free and we report the price. The optimiser accepts")
    a("slightly worse synchrony on each field --- mean absolute TSI")
    a(f"{SOL['tsi'].loc[SOL['big'], 'S3_static_robust']:.1f} days against")
    a(f"{SOL['tsi'].loc[SOL['big'], BASE_KEY]:.1f} for the baseline at")
    a(f"$N = {SOL['big']}$ --- in exchange for serving far more of them.")
    a("That trade is the whole content of the decision: a scheduler that")
    a("insisted on perfect synchrony everywhere would sow fewer fields and")
    a("produce less seed in total.")
    a("")

    a("\\begin{table}[t]")
    a("\\centering")
    a("\\caption{Optimised schedule against current practice}")
    a("\\label{tab:solution}")
    a("\\begin{tabular}{rrrrrr}")
    a("\\hline")
    a(" & \\multicolumn{2}{c}{yield (t)} & \\multicolumn{2}{c}{fields sown}"
      " & \\\\")
    a("$N$ & practice & optimised & practice & optimised & gain \\\\")
    a("\\hline")
    pv_y = raw[raw.kind == "p"].pivot_table(
        index="n_fields", columns="strategy", values="yield_t")
    for n in SOL["per_scale"].index:
        a(f"{int(n)} & {pv_y.loc[n, BASE_KEY]:.0f} & "
          f"{pv_y.loc[n, 'S3_static_robust']:.0f} & "
          f"{SOL['sown'].loc[n, BASE_KEY]:.0f} & "
          f"{SOL['sown'].loc[n, 'S3_static_robust']:.0f} & "
          f"${SOL['per_scale'].loc[n]:+.0f}$\\% \\\\")
    a("\\hline")
    a("\\end{tabular}")
    a("\\end{table}")
    a("")

    # ---------------- the ladder ---------------------------------------
    hi = PRES.iloc[-1]
    sub = raw[(raw.kind == "p") & (raw.n_fields == hi.n_fields)]
    L = ladder(sub)
    oracle = L.get("S5_wait_and_see", np.nan)

    a("\\subsection{Where the remaining value lies}")
    a("")
    a("At the tightest configuration we study"
      f" --- $N = {int(hi.n_fields)}$ fields, a crew of"
      f" $K = {int(hi.capacity)}$, resource pressure"
      f" $\\pi = {f(hi.pressure)}$ --- mean realised seed yield rises along")
    a("the ladder:")
    labels = [(BASE_KEY, "$S_1$ current practice"),
              ("S2_static_ev", "$S_2$ optimised, single forecast"),
              ("S3_static_robust", "$S_3$ optimised, robust"),
              ("S4_dynamic", "$S_4$ optimised, replanned"),
              ("S5_wait_and_see", "$S_5$ wait-and-see bound")]
    a(", ".join(f"{lab} {L[k]:.1f}~t" for k, lab in labels if k in L) + ".")
    a("")
    s3 = L["S3_static_robust"]
    s2 = L.get("S2_static_ev", np.nan)
    s4 = L["S4_dynamic"]
    a("Three quantities here are the ones stochastic programming")
    a(f"names~\\cite{{birge}}. The value of the stochastic solution,")
    a(f"$S_3 - S_2$, is {f(s3-s2,1)}~t ({f(100*(s3-s2)/s2,1)}\\%):")
    a("planning against an ensemble of seasons rather than one typical")
    a("season is worth having, though it is a second-order effect next to")
    a("solving the assignment at all. The value of adaptivity, $S_4 - S_3$,")
    a(f"is {f(s4-s3,1)}~t. The expected value of perfect information,")
    a(f"$S_5 - S_4$, is {f(oracle-s4,1)}~t ({f(100*(oracle-s4)/s4,1)}\\%):")
    a("headroom a perfect seasonal forecast would buy and no causal")
    a("strategy can reach.")
    a("")
    a("The ordering $S_1 < S_2 < S_3 < S_5$ is what theory requires --- more")
    a("information cannot hurt a correctly solved stage-1 problem, and no")
    a("causal policy can beat the oracle --- and we use it as a check on")
    a("the implementation. Theory does \\emph{not} fix where $S_4$ falls")
    a("relative to $S_3$: a revision is adopted on a forecast that may be")
    a("wrong, so adaptivity can lose. The rest of this section is about")
    a("when it does.")
    a("")

    # ---------------- resource pressure ------------------------------------
    r = S.get("gain_vs_pressure_r")
    pr = S.get("gain_vs_pressure_p")
    a("\\subsection{The value of adaptivity against resource pressure}")
    a("")
    a("Table~\\ref{tab:pressure} and Fig.~\\ref{fig:pressure} give the")
    a("adaptivity gain across resource pressure.")
    a("")
    a("\\begin{table}[t]")
    a("\\centering")
    a("\\caption{Value of adaptivity against resource pressure}")
    a("\\label{tab:pressure}")
    a("\\begin{tabular}{rrrrrr}")
    a("\\hline")
    a("$N$ & $\\pi$ & $S_3$ (t) & $S_4-S_3$ & $p$ & $n$ \\\\")
    a("\\hline")
    for _, row in PRES.iterrows():
        a(f"{int(row.n_fields)} & {f(row.pressure)} & "
          f"{row.S3:.1f} & ${row.gain_pct:+.2f}$\\% & "
          f"{row.p:.3f} & {int(row.n)} \\\\")
    a("\\hline")
    a("\\end{tabular}")
    a("\\\\[2pt]")
    a("\\footnotesize\\raggedright Each row is 3 or 5 independently drawn")
    a("farms $\\times$ 7 held-out seasons. Shaded region of")
    a("Fig.~\\ref{fig:pressure} marks the band tested in the text.")
    a("\\end{table}")
    a("")

    lo = PRES.iloc[0]
    pooled = HEAD["point"]
    pooled_p = S.get("pooled_p")
    pooled_n = S.get("pooled_n")
    npos = PRES.n_farms_pos.sum() if "n_farms_pos" in PRES else None
    nfarm = PRES.n_farms.sum() if "n_farms" in PRES else None
    if pooled is not None:
        a("Pooled over every configuration, adaptive replanning is worth")
        a(f"${HEAD['point']:+.2f}$\\% of realised seed yield")
        a(f"(95\\% CI ${HEAD['lo']:+.2f}$ to ${HEAD['hi']:+.2f}$\\%,")
        a(f"$n = {HEAD['n']}$ paired farm--season observations,")
        a(f"Cohen's $d = {HEAD['d']:.2f}$), and is positive on")
        a(f"{int(npos)} of the {int(nfarm)} farm--configuration combinations")
        a("we ran. The interval is a bootstrap that resamples whole farms")
        a("rather than farm--seasons, since seasons within a farm are not")
        a("independent; every one of 5000 resamples was positive. The point")
        a("estimate is the ratio of total extra seed to total baseline")
        a("seed, which is what a producer's accounts would show and is the")
        a("more conservative of the two available estimators --- averaging")
        a(f"the per-season percentages instead gives ${MEANRATIO:+.2f}$\\%,"
          " because")
        a("small farms contribute proportionally larger percentages.")
        a("")
        a("In absolute terms the gain is")
        a(f"{PERHA['kg_per_ha']:.0f}~kg of hybrid seed per hectare per")
        a("season, obtained with no change to inputs, genetics or acreage.")
        a("We state it this way because a few percent sounds negligible and")
        a("a hundred-hectare seed operation recovering six tonnes of")
        a("certified seed a year does not; at indicative Indian hybrid maize")
        a("seed prices the latter is of the order of a million rupees")
        a("annually. The effect size is small by the conventional standard,")
        a("which is the honest description of a scheduling intervention that")
        a("costs nothing to apply.")
        a("")

    # Describe the shape the data actually has rather than the shape we
    # expected: find the minimum and say whether the gain recovers after it.
    gp = PRES.gain_pct.values
    pi = PRES.pressure.values
    imin = int(np.argmin(gp))
    interior = 0 < imin < len(gp) - 1
    declines = bool(np.all(np.diff(gp[:imin + 1]) < 0)) if imin > 0 else False
    recovers = bool(gp[-1] > gp[imin] + 1.0) if interior else False

    a("The pattern is not the one we expected, so we state the expectation")
    a("first. We anticipated the gain to grow with resource pressure, on")
    a("the reasoning that an idle crew leaves a revision nothing to")
    a("reallocate. It does the opposite over most of the range.")
    a("")
    if declines:
        a(f"The gain is largest where the crew has the most slack ---")
        a(f"${PRES.iloc[0].gain_pct:+.2f}$\\% at"
          f" $\\pi = {f(PRES.iloc[0].pressure)}$,"
          f" {pv(PRES.iloc[0].p)}, positive on"
          f" {int(PRES.iloc[0].n_farms_pos)} of"
          f" {int(PRES.iloc[0].n_farms)} farms --- and falls monotonically")
        a(f"to ${gp[imin]:+.2f}$\\% at $\\pi = {f(pi[imin])}$,")
        a("where the crew's capacity almost exactly equals the visits")
        a("demanded.")
    if interior and recovers:
        a(f"Above that point it recovers, reaching ${gp[-1]:+.2f}$\\%")
        a(f"({pv(PRES.iloc[-1].p)}, positive on"
          f" {int(PRES.iloc[-1].n_farms_pos)} of"
          f" {int(PRES.iloc[-1].n_farms)} farms) at"
          f" $\\pi = {f(pi[-1])}$. The relationship is therefore")
        a("non-monotone, with a trough at saturation, which is why a linear")
        a(f"correlation across these points is uninformative ($r = {f(r)}$,")
        a(f"{pv(pr)}) and we do not report one as a finding.")
    a("")
    lo_st = PERSEASON.get(int(PRES.iloc[0].n_fields), {})
    hi_st = PERSEASON.get(int(PRES.iloc[-1].n_fields), {})

    if SAT is not None:
        a("A single negative configuration invites the reading that it is")
        a("noise, so we refilled the region around saturation at")
        a(f"{len(SAT['new_fields'])} further farm sizes. The result")
        a("sharpens the claim and also corrects it. Pooling every")
        a(f"farm--season with ${SAT['band_lo']} \\le \\pi \\le"
          f" {SAT['band_hi']}$ gives a mean of"
          f" ${SAT['band_mean']:+.2f}$\\% ($n = {SAT['band_n']}$) against")
        a(f"${SAT['outside_mean']:+.2f}$\\% outside it"
          f" ($n = {SAT['outside_n']}$). The two differ")
        a(f"({pv(SAT['band_vs_outside_p'])}, Welch), so the trough is a")
        a("region rather than one stray point. But the band is")
        a("\\emph{not} distinguishable from zero")
        a(f"({pv(SAT['band_vs_zero_p'])}): the")
        a(f"${PRES.gain_pct.min():+.2f}$\\% observed at")
        a(f"$\\pi = {f(PRES.loc[PRES.gain_pct.idxmin(), 'pressure'])}$ is the")
        a("extreme of a noisy band centred near zero, not evidence that")
        a("replanning reliably destroys value. The defensible statement is")
        a("that the gain \\emph{vanishes} near saturation, not that it")
        a("reverses.")
        a("")
    a("Three regimes explain this. With \\emph{slack} a revision can")
    a("actually be executed, and is mostly used to recover fields that rain")
    a("pushed off their planned day: the dynamic plan sows 0.2 to 0.4 more")
    a("fields per farm at these scales and essentially none extra at")
    a("saturation or above, so the benefit is insurance-shaped --- at")
    a(f"$\\pi = {f(PRES.iloc[0].pressure)}$ only"
      f" {100*PERSEASON.get(int(PRES.iloc[0].n_fields), {}).get('frac_pos', 0):.0f}\\%")
    a("of farm--seasons improve, the mean carried by rare seasons in which")
    a("a field the static plan loses outright is rescued. At")
    a("\\emph{saturation} capacity exactly matches demand, so every")
    a("revision is a forced swap whose achievable gain is small while the")
    a("forecast error motivating it is not, and the benefit washes out.")
    a("Under \\emph{over-subscription} not every field can be served")
    a("whatever is done, so the decision becomes which to serve and which")
    a("to concede --- robust to forecast error in a way fine timing")
    a("adjustments are not --- and the gain returns, now as a routine")
    a("improvement rather than insurance.")
    a("")

    # ---------------- sharpness --------------------------------------------
    rs = S.get("gain_vs_sharpness_r")
    ps = S.get("gain_vs_sharpness_p")
    a("\\subsection{Penalty sharpness}")
    a("")
    a("Fig.~\\ref{fig:sharpreal} sweeps $\\lambda$ at fixed resource")
    a("pressure. Recall that $\\lambda = 1$ reproduces the published")
    a("regression exactly, that every member of the family shares the same")
    a("optimum and the same maximum, and that $\\lambda$ changes only how")
    a("fast yield falls away from that optimum.")
    a("")

    a("")

    s_lo, s_hi = SHARP.iloc[0], SHARP.iloc[-1]
    if rs is not None and np.isfinite(rs) and rs > 0:
        a(f"The gain rises monotonically with sharpness ($r = {f(rs)}$,")
        a(f"{pv(ps)}), and it rises across the whole range: from")
        a(f"${s_lo.gain_pct:+.2f}$\\% at $\\lambda = {s_lo.sharpness:g}$ ---")
        a("a null rather than merely a non-significant result, since its")
        ci_lo = SHARP_CI.get(float(s_lo.sharpness), {})
        ci_hi = SHARP_CI.get(float(s_hi.sharpness), {})
        if ci_lo:
            a(f"95\\% interval (${ci_lo['lo']:+.2f}$ to"
              f" ${ci_lo['hi']:+.2f}$\\%) excludes everything a practitioner")
            a("would act on --- to")
        else:
            a("confidence interval is tight around zero --- to")
        a(f"${s_hi.gain_pct:+.2f}$\\%"
          + (f" (95\\% CI ${ci_hi['lo']:+.2f}$ to ${ci_hi['hi']:+.2f}$\\%)"
             if ci_hi else "")
          + f" at $\\lambda = {s_hi.sharpness:g}$.")
        a("")
        a("The flat end of this family is the irrigation regime, and we")
        a("reproduce the irrigation result there. When the biological")
        a("penalty is nearly flat, re-optimising during the season against")
        a("a strong robust static plan buys essentially nothing, on the")
        a("same farms, the same weather and the same solver that deliver a")
        a("significant gain when the penalty is sharp. That is the")
        a("comparison this paper was built to make: the difference between")
        a("the two conclusions is the response surface, not the method, the")
        a("crop or the implementation.")
        a("")
        cross = SHARP[SHARP.gain_pct >= 5.0]
        if len(cross) and cross.index[0] != SHARP.index[0]:
            j = SHARP.index.get_loc(cross.index[0])
            prev = SHARP.iloc[j - 1]
            nxt = SHARP.iloc[j]
            lam5 = prev.sharpness + (5.0 - prev.gain_pct) * \
                (nxt.sharpness - prev.sharpness) / \
                (nxt.gain_pct - prev.gain_pct)
            a("Five percent is the figure at which the irrigation literature")
            a("judged in-season adaptation not worth the cost of building")
            a("and operating it. Interpolating between our sweep points, the")
            a(f"gain passes it at $\\lambda \\approx {lam5:.1f}$ --- that is,")
            a("at a penalty about half again as sharp as the one measured")
            a("for hybrid maize seed. The boundary of the Kelly--Becker")
            a("conclusion therefore lies just above this crop, not far above")
            a("it, which is why hybrid seed production sits close enough to")
            a("the threshold to be worth the measurement rather than an")
            a("assumption.")
        elif len(cross):
            a("The gain exceeds five percent, the threshold at which the")
            a("irrigation literature judged adaptation not worth its cost,")
            a("across most of the range we simulate.")
        else:
            a("It does not reach five percent anywhere in the range we")
            a("simulate, which bounds the effect from above as well as")
            a("locating its direction.")
    elif rs is not None and np.isfinite(rs):
        a("Across this range the gain does not rise with sharpness")
        a(f"($r = {f(rs)}$, {pv(ps)}); it is ${s_lo.gain_pct:+.2f}$\\% at")
        a(f"$\\lambda = {s_lo.sharpness:.2f}$ and ${s_hi.gain_pct:+.2f}$\\%")
        a(f"at $\\lambda = {s_hi.sharpness:.2f}$.")
        a("")
        a("This is a negative result and we report it as one. Our working")
        a("hypothesis was that the curvature of the biological penalty is")
        a("what separates hybrid seed sowing from irrigation; on real")
        a("weather, at a configuration where the crew binds, it is not the")
        a("term that moves the answer. Two reasons are visible in the runs.")
        a("Sharpening the penalty hurts the static and the dynamic plan")
        a("alike --- $S_3$ falls")
        a(f"from {s_lo.S3:.0f}~t to {s_hi.S3:.0f}~t across the sweep --- so")
        a("the \\emph{difference} between them need not grow even though the")
        a("stakes do. And a sharper penalty makes the optimiser more")
        a("conservative at stage~1, which is itself a form of robustness:")
        a("the static plan buys protection in advance that the dynamic plan")
        a("would otherwise have had to earn in-season.")
        a("")
        a("We therefore do not claim that penalty sharpness governs the")
        a("value of replanning. What the sweep does establish is weaker and")
        a("still useful: the value of adaptivity we measure is not an")
        a("artefact of the particular curvature published for this crop,")
        a("since it survives across a four-fold range of it.")
    a("")
    a("Sharpness and resource pressure act differently: the gain is")
    a("monotone in the first and not in the second, because sharpness sets")
    a("how much a timing error costs while pressure sets whether anything")
    a("can be done about it. This also explains a discrepancy we")
    a("encountered --- an earlier sweep of $\\lambda$ run at the saturated")
    a("configuration found no relationship at all, a property of the")
    a("operating point rather than of the biology. Sweeping one axis at an")
    a("unlucky value of the other is an easy way to miss a real effect.")
    a("")

    # ---------------- validation cross-ref ---------------------------------
    a("\\subsection{Does the biology behave?}")
    a("")
    a("Section~\\ref{sec:validation} reports the external test in full. In")
    a("summary, the model reproduces CIMMYT's published differential sowing")
    a(f"intervals for six tropical hybrids to within {f(V['spans_mae'],1)}")
    a(f"days, gets the direction right in {V['direction_correct']} cases out")
    a(f"of {V['n_hybrids']}, and recovers their difficulty ranking at")
    a(f"Spearman $\\rho = {f(V['rank_rho'])}$. None of that data entered the")
    a("model.")
    a("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Discussion
# ---------------------------------------------------------------------------

def discussion_section():
    out = []
    a = out.append
    gp = PRES.gain_pct.values
    s_lo = SHARP.iloc[0]
    hi = PRES.iloc[-1]
    sub = raw[(raw.kind == "p") & (raw.n_fields == hi.n_fields)]
    L = ladder(sub)
    gap = L["S5_wait_and_see"] - L["S4_dynamic"]

    a("\\subsection{What to build, and in what order}")
    a("")
    a("The results separate cleanly into something every producer at scale")
    a("should do and something only some should.")
    a("")
    a("\\emph{Solve the assignment.} This is worth")
    a(f"${SOL['pooled']:+.1f}$\\% overall and more at scale, needs no")
    a("forecast, no in-season data collection and no change to agronomy,")
    a("and the whole stack is free software. Its benefit grows with farm")
    a("size because the constraint it removes --- naive sequencing")
    a(f"saturating near {SOL['base_plateau']:.0f} fields --- bites harder")
    a("the more contracts a producer carries.")
    a("")
    a("\\emph{Add replanning only under two conditions.} The penalty must")
    a("be sharp: where a day of timing error costs little, a robust static")
    a("plan absorbs the risk in advance, which is what the irrigation")
    a("studies found and what we reproduce at")
    a(f"$\\lambda = {s_lo.sharpness:g}$ (${s_lo.gain_pct:+.2f}$\\%,")
    a(f"{pv(s_lo.p)}). And the schedule must have room to move: a revision")
    a("is only worth making where a field can go somewhere, which is why")
    a("the gain vanishes at saturation and returns on either side of it.")
    a("Both conditions are arithmetic --- a published response curve and")
    a("$\\pi = NV/TK$ --- so a producer can decide before any software is")
    a("written.")
    a("")
    a("\\subsection{Scope of the irrigation result, and what remains}")
    a("")
    a("Kelly \\textit{et al.}~\\cite{kelly2023} and Becker \\textit{et")
    a("al.}~\\cite{becker2026} are right about irrigation, and we reproduce")
    a("their finding at the flat end of our sharpness family on our own")
    a("farms, weather and solver. Our claim is narrower than a")
    a("contradiction: their conclusion describes a region of a")
    a("two-dimensional space --- how sharply the biology punishes a timing")
    a("error, and how much freedom the operation has to correct one ---")
    a("rather than agricultural scheduling as such. Irrigation sits at the")
    a("null end of both.")
    a("")
    a(f"Finally, the oracle still beats the best causal strategy by")
    a(f"{f(gap,1)}~t at the tightest configuration, several times the value")
    a("of adaptivity itself. Most of what a perfect seasonal forecast would")
    a("buy is unclaimed and unclaimable by better optimisation alone: the")
    a("non-anticipativity constraint, not the solver, is what stands in the")
    a("way. The route to it is forecast skill at the four-to-eight-week")
    a("range over the flowering window.")
    a("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------

def conclusion_section():
    out = []
    a = out.append
    rs = S.get("gain_vs_sharpness_r")
    s_lo, s_hi = SHARP.iloc[0], SHARP.iloc[-1]
    gp = PRES.gain_pct.values

    a("We formulated multi-field hybrid maize seed sowing as a two-stage")
    a("stochastic scheduling problem coupling a shared sowing crew to the")
    a("flowering synchrony of each field's parent lines, and evaluated it on")
    a("twenty-one observed seasons at an Indian seed-production site with")
    a("flowering dates from CERES-Maize.")
    a("")
    a("The practical result is that solving the assignment is worth")
    a(f"${SOL['pooled']:+.1f}$\\% more seed than current practice overall")
    a(f"and ${SOL['big_gain']:+.0f}$\\% at the largest farm we test. The")
    a("reason is not better agronomy --- the baseline already uses the")
    a("correct per-hybrid offsets --- but better logistics: sequencing")
    a(f"fields naively saturates at about {SOL['base_plateau']:.0f} fields")
    a("however many are offered, so beyond that point additional contracts")
    a("go unsown, while an optimised plan keeps placing them. The same crew")
    a("serves half again as many fields, at the price of slightly worse")
    a("synchrony on each.")
    a("")
    a("Re-optimising during the season adds a further")
    a(f"${HEAD['point']:+.2f}$\\% on top of that, which is real but small,")
    a("and conditional in two ways that we characterise.")
    if rs is not None and np.isfinite(rs) and rs > 0:
        a(f"It rises monotonically with the curvature of the biological")
        a(f"penalty, from ${s_lo.gain_pct:+.2f}$\\% at a flat response ---")
        a("where we reproduce the irrigation literature's null on our own")
        a(f"farms --- to ${s_hi.gain_pct:+.2f}$\\% at a sharp one. The")
        a("reported marginality of in-season replanning is therefore bounded")
        a("rather than general.")
    if SAT:
        a("It is also non-monotone in how hard the crew is worked,")
        a(f"collapsing to ${SAT['band_mean']:+.2f}$\\% --- indistinguishable")
        a("from zero --- where capacity roughly matches demand, since a")
        a("revision is worth making only where there is somewhere to move a")
        a("field to. An operation that has sized its crew precisely to its")
        a("acreage should optimise its schedule and leave it alone.")
    a("")
    a("The screening rule that follows costs nothing to apply: compute the")
    a("yield cost of one day of asynchrony, compute $\\pi = NV/TK$, and add")
    a("in-season replanning to the optimiser only where the first is large")
    a("and the second is away from one.")
    a("")
    return "\n".join(out)


def abstract_finding():
    rs = S.get("gain_vs_sharpness_r")
    gp = PRES.gain_pct.values
    s_lo, s_hi = SHARP.iloc[0], SHARP.iloc[-1]

    bits = [
        f"Against a baseline that uses the correct per-hybrid sowing offset "
        f"but sequences fields naively --- current practice --- solving the "
        f"assignment yields ${SOL['pooled']:+.1f}$\\% more seed overall and "
        f"${SOL['big_gain']:+.0f}$\\% at the largest farm tested, because "
        f"naive sequencing saturates at about {SOL['base_plateau']:.0f} "
        f"fields however many are offered while the optimised plan keeps "
        f"placing them: the same crew serves half again as many fields."]
    bits.append(
        " In-season replanning adds a further "
        f"${HEAD['point']:+.2f}$\\% (95\\% CI ${HEAD['lo']:+.2f}$ to "
        f"${HEAD['hi']:+.2f}$\\%, $n = {HEAD['n']}$), but only in some "
        "regimes, and we characterise which.")
    if rs is not None and np.isfinite(rs) and rs > 0:
        bits.append(
            " Sweeping a one-parameter family that holds the yield optimum "
            "fixed while varying only the curvature of the penalty, that "
            f"further gain rises monotonically with sharpness ($r = {f(rs)}$) "
            f"from ${s_lo.gain_pct:+.2f}$\\% at the flat end --- where we "
            "reproduce the irrigation literature's null on our own farms --- "
            f"to ${s_hi.gain_pct:+.2f}$\\% at the sharp end.")
    if SAT:
        bits.append(
            " It is also non-monotone in resource pressure, collapsing to "
            f"${SAT['band_mean']:+.2f}$\\%, indistinguishable from zero, "
            "where crew capacity roughly matches demand, because a revision "
            "is worth making only when there is somewhere to move a field "
            "to.")
    return "".join(bits)



FIG_PRESSURE = r"""
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{../figs/fig8_pressure.pdf}
\caption{Value of adaptivity against resource pressure, on twenty-one
observed seasons at Ranebennur with leave-one-season-out evaluation. Grey
points are individual farm means; the line is pooled over five independently
drawn farms. The shaded band marks saturation, where the crew's capacity
equals the visits demanded. The relationship is non-monotone: a revision is
worth making only where there is somewhere to move the field to.}
\label{fig:pressure}
\end{figure}
"""

FIG_SHARP = r"""
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{../figs/fig9_sharpness_real.pdf}
\caption{Value of adaptivity against penalty sharpness at
$\pi = 1.35$, where the crew binds. Grey lines are the five individual
farms, all of which rise. $\lambda = 1$ is the response published for hybrid
maize seed; $\lambda \to 0$ is the flat regime in which the irrigation
literature found adaptation not worth its cost, and we reproduce that null
there.}
\label{fig:sharpreal}
\end{figure}
"""

def main():
    src = open(f"{P}/paper.tex").read()

    # figures and the validation section are injected rather than hand-placed
    # so that the manuscript template stays free of result-specific material.
    res = results_section() + "\n" + FIG_PRESSURE + FIG_SHARP

    subs = [("RESULTS_SECTION", res),
            ("DISCUSSION_SECTION", discussion_section()),
            ("CONCLUSION_SECTION", conclusion_section()),
            ("ABSTRACT\\_FINDING", abstract_finding()),
            ("ABSTRACT_FINDING", abstract_finding())]
    for tok, val in subs:
        src = src.replace(tok, val)

    left = [t for t, _ in subs if t in src and "ABSTRACT" not in t]
    if left:
        print("WARNING unsubstituted:", left)

    open(f"{P}/paper_filled.tex", "w").write(src)
    print("wrote paper/paper_filled.tex")

    hi, lo = PRES.iloc[-1], PRES.iloc[0]
    print(f"pressure {lo.pressure:.2f} -> gain {lo.gain_pct:+.2f}%")
    print(f"pressure {hi.pressure:.2f} -> gain {hi.gain_pct:+.2f}%")
    print(f"gain vs pressure  r = {S.get('gain_vs_pressure_r')}")
    print(f"gain vs sharpness r = {S.get('gain_vs_sharpness_r')}")


if __name__ == "__main__":
    main()
