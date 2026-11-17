"""
Rewrite the results, discussion and conclusion of the manuscript around the
multi-farm evidence, replacing the single-instance numbers.

Two findings of the provisional study change here:

  1. The headline adaptivity gain roughly halves (3.9% -> 1.9%), because the
     single farm it was measured on ranks second of ten.
  2. The claim that the gain saturates below five percent does not survive.
     It rises on every farm and crosses five percent at about twice the
     measured curvature.

As in fill_paper.py, prose is plain text with @@TOKEN@@ placeholders so that
LaTeX braces never meet Python format strings.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
P = os.path.join(HERE, "paper")

S6 = json.load(open(f"{R}/E6_summary.json"))
gains = pd.read_csv(f"{R}/E6_gains.csv")
sg = pd.read_csv(f"{R}/E6_sharpness_gains.csv")
e1 = pd.read_csv(f"{R}/E1_main.csv")

M = S6["main"]
W = S6["sweep"]
BY = {float(k): v for k, v in W["by_lambda"].items()}
LAMS = sorted(BY)


def pv(p):
    if p < 1e-6:
        return "$p < 10^{-6}$"
    if p < 1e-3:
        return "$p < 0.001$"
    return f"$p = {p:.3f}$"


# ---------------------------------------------------------------- derived
r_base, p_base = stats.pearsonr(gains.S3, gains.gain_pct)
rho_base, prho = stats.spearmanr(gains.S3, gains.gain_pct)
rank7 = sorted(gains.gain_pct, reverse=True).index(
    float(gains[gains.farm_seed == 7].gain_pct.iloc[0])) + 1

# sharpness at which each farm's gain reaches 5%
cross = []
for fs, sub in sg.groupby("farm_seed"):
    sub = sub.sort_values("sharpness")
    x, y = sub.sharpness.values, sub.gain_pct.values
    above = np.where(y >= 5.0)[0]
    if len(above) and above[0] > 0:
        i = above[0]
        cross.append(x[i - 1] + (5.0 - y[i - 1]) * (x[i] - x[i - 1])
                     / (y[i] - y[i - 1]))
lam_med = float(np.median(cross)) if cross else float("nan")
lam_lo, lam_hi = (float(min(cross)), float(max(cross))) if cross else (0, 0)

single = float(e1[e1.strategy == "S4_dynamic"].yield_t.mean()
               - e1[e1.strategy == "S3_static_robust"].yield_t.mean())
single_pct = 100 * single / e1[e1.strategy == "S3_static_robust"].yield_t.mean()

table_farms = "\n".join(
    f"{int(r.farm_seed)} & {r.S3:.1f} & {r.S4:.1f} & {r.gain_t:+.2f} & "
    f"{r.gain_pct:+.2f} & {pv(r.p)} \\\\"
    for _, r in gains.sort_values("farm_seed").iterrows())

table_sweep = "\n".join(
    f"{lam:g} & {BY[lam]['mean']:.2f} & {BY[lam]['std']:.2f} & "
    f"{BY[lam]['min']:+.2f} & {BY[lam]['max']:+.2f} \\\\" for lam in LAMS)


RESULTS6 = r"""
\subsection{Across farm instances}
\label{sec:farms}

Every number reported above comes from one farm. Because field areas, soil
drainage, microclimate offsets and parent thermal-time requirements are all
drawn at random, a single instance can be unrepresentative, so we repeated
the comparison on @@NFARMS@@ independently generated farms.

It is. Table~\ref{tab:farms} gives the per-farm result and
Fig.~\ref{fig:farms}(a) orders them. The gain from adaptivity averages
@@GMEAN@@\% across farms (SD @@GSD@@, range @@GMIN@@\% to @@GMAX@@\%),
positive on @@NPOS@@ of @@NFARMS@@ farms and individually significant on only
@@NSIG@@. Treating the farm as the unit of analysis, the mean gain differs
from zero (@@FARMP@@, Wilcoxon @@FARMWP@@), so the effect is real.

Its size, however, is roughly half what the single-instance study reported.
The farm used there ranks @@RANK7@@ of @@NFARMS@@, and the @@SINGLEPCT@@\%
measured on it sits near the top of the distribution. On the best-performing
farm the gain is slightly negative. We report this because it is the kind of
error that a single-instance design cannot reveal, and because the corrected
value strengthens rather than weakens the paper's conclusion: at
@@GMEAN@@\%, adaptive replanning is further below the threshold at which the
irrigation literature judged it not worth its cost.

A second pattern explains much of the spread. The gain is strongly
anti-correlated with how well the farm's static plan already performs
($r = @@RBASE@@$, @@PBASE@@; Spearman $\rho = @@RHOBASE@@$),
Fig.~\ref{fig:farms}(b). A farm whose fields and parent lines happen to line
up well needs no adaptation, because a good fixed plan already reaches the
synchrony optimum; adaptivity recovers losses rather than creating gains.
This is directly usable: a producer can estimate in advance whether adaptive
scheduling is worth installing by checking how close a static plan already
gets.

\begin{table}[t]
\caption{Adaptivity gain by farm instance ($\lambda = 1$)}
\label{tab:farms}
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{rrrrrl}
\toprule
Farm & S3 (t) & S4 (t) & $\Delta$ (t) & $\Delta$ (\%) & Sig. \\
\midrule
@@TABLEFARMS@@
\midrule
Mean & & & @@GTMEAN@@ & @@GMEAN@@ & @@FARMP@@ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{../figs/fig5_multifarm.pdf}
\caption{(a) Adaptivity gain on each of @@NFARMS@@ independently generated
farms; the outlined bar is the instance used for the single-farm analysis.
(b) The gain is largest on farms whose static plan performs worst.}
\label{fig:farms}
\end{figure}

\subsection{Sharpness, re-examined across farms}

Repeating the sharpness sweep on every farm overturns the saturation result
obtained from one instance. Table~\ref{tab:sweep6} and
Fig.~\ref{fig:sharp6}(a) show the gain rising monotonically with $\lambda$ on
@@NRISING@@ of @@NFARMS@@ farms, with a mean within-farm correlation of
$r = @@MEANR@@$ and a mean slope of @@SLOPE@@ percentage points per unit of
sharpness. It does not level off: the peak exceeds five percent on
@@NABOVE5@@ of @@NFARMS@@ farms, reaching @@MAXPEAK@@\% at $\lambda = 3$.

The single-instance analysis reported that the gain saturated below five
percent and concluded that the Kelly--Becker finding held even for sharply
responding outcomes. That conclusion was an artefact of one farm that
happened to flatten out. It does not survive replication.

What replaces it is more precise and more useful. Interpolating each farm's
curve, the gain reaches five percent at $\lambda = @@LAMMED@@$ on average
(range @@LAMLO@@ to @@LAMHI@@), Fig.~\ref{fig:sharp6}(b) --- that is, at
roughly @@LAMMED@@ times the curvature actually measured in the field. Below
that boundary adaptive replanning is not worth its operational cost; above
it, it is. Hybrid maize, at $\lambda = 1$, sits below the boundary. Operations
with a steeper timing penalty do not.

\begin{table}[t]
\caption{Adaptivity gain (\%) by sharpness, across @@NFARMS@@ farms}
\label{tab:sweep6}
\centering\footnotesize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{rrrrr}
\toprule
$\lambda$ & Mean & SD & Min & Max \\
\midrule
@@TABLESWEEP@@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{../figs/fig6_sharpness_multifarm.pdf}
\caption{(a) Adaptivity gain against penalty sharpness; grey lines are
individual farms, the marked line their mean and standard deviation.
(b) Sharpness at which each farm's gain reaches five percent.}
\label{fig:sharp6}
\end{figure}
"""


DISC6 = r"""
We set out to ask whether the limited value of in-season replanning reported
for irrigation survives when the biological outcome responds sharply to
timing error. It does not, and the boundary can be located.

The gain from adaptive replanning rises with the curvature of the yield
response on every farm tested ($r = @@MEANR@@$), crossing the five percent
mark at about @@LAMMED@@ times the curvature measured for hybrid maize.
Below that the static plan absorbs the risk; above it, it cannot. The
mechanism is hedging. A sowing offset is a decision taken once with a tunable
margin, so when the penalty steepens the static optimiser responds by
choosing an offset that stays tolerable across the plausible range of
thermal-time outcomes rather than aiming at the peak. That substitution works
until the penalty becomes steep enough that no single margin is tolerable
everywhere, and it is at that point that acting on new information starts to
pay. An irrigation depth, which must be chosen repeatedly and cannot be
hedged in the same way, sits in a different part of this space, which is why
the irrigation literature saw only the flat regime.

Hybrid maize, at the sharpness its own field data imply, falls below the
boundary: adaptivity is worth @@GMEAN@@\% (@@FARMP@@), real but modest, and
concentrated in the seasons and on the farms that go worst. For this crop the
Kelly--Becker recommendation transfers. The contribution is not that it
transfers, but that we can now say how far it transfers and what takes an
operation outside it.

\subsection{What single-instance designs hide}

Both of the corrections above came from the same fix. Measured on one farm,
the adaptivity gain was @@SINGLEPCT@@\% and appeared to saturate below five
percent; measured on ten, it is @@GMEAN@@\% and saturates nowhere. One
instance produced an effect roughly twice its true size and a qualitative
conclusion that reversed on replication.

We draw attention to this because simulation studies in this area routinely
report results from a single synthetic instance, and because neither error
was visible from inside that instance: the single-farm result was internally
consistent, adequately powered across seasons, and statistically significant.
Replication across instances is not a robustness check to be added if space
allows. It is what determines whether the headline number means anything.

\subsection{A screening rule}

Three questions decide whether an adaptive scheduler is worth building.

\emph{How sharp is the timing penalty?} Below roughly @@LAMMED@@ times the
curvature of the hybrid maize response, a well-built static plan captures
nearly everything available. This is the dominant factor.

\emph{Can the decision be hedged in advance?} A choice made once with a
tunable margin, such as a sowing offset, can absorb uncertainty before the
season; one that must be made repeatedly cannot. Hedging substitutes for
adapting.

\emph{How well does a static plan already do?} The gain was strongly
anti-correlated with baseline performance ($r = @@RBASE@@$). An operation
already close to its synchrony optimum has little for adaptation to recover.

\subsection{What the benchmark choice decides}

Had we compared adaptive replanning against conventional practice rather than
against a robust static plan, we would have reported a substantially larger
figure and claimed a strong result. Most of the apparent benefit of adaptive
scheduling in this problem is the benefit of optimising at all, which a static
solver delivers once, before the season, at a fraction of the operational
cost. We also note that both failure modes encountered in development ---
indefinite deferral under free reassignment, and double-counting rain risk in
objective and constraints --- caused the adaptive policy to \emph{underperform}
the static one. An adaptive system is better than a fixed one only if these
are handled.
"""


CONCL6 = r"""
We asked whether the reported marginality of in-season replanning survives
when the biological outcome responds sharply to timing error. Across
@@NFARMS@@ independently generated farms it does not: the gain from adaptive
replanning rises with the curvature of the yield response on every farm
($r = @@MEANR@@$) and reaches five percent at roughly @@LAMMED@@ times the
curvature measured for hybrid maize. The irrigation finding is therefore
bounded rather than general, and we locate the boundary.

For multi-field hybrid maize seed production at its measured response,
adaptive replanning is worth @@GMEAN@@\% (SD @@GSD@@, @@FARMP@@) against a
robust static plan --- positive on @@NPOS@@ of @@NFARMS@@ farms, significant
on @@NSIG@@, and largest where the static plan performs worst. Below the
boundary, a well-built static plan is the better engineering choice.

This is the first formulation of hybrid seed sowing as a resource-constrained
stochastic scheduling problem coupled to parent flowering synchrony, scored
against a published field regression and solved to proven optimality so that
the measured difference is attributable to adaptivity rather than to search.

A methodological point stands alongside the biological one. Measured on a
single farm, the same system showed an effect roughly twice its size and a
saturation result that did not replicate. Instance-level replication decided
both.
"""


TOK = {
    "@@NFARMS@@": str(M["n_farms"]),
    "@@NPOS@@": str(M["n_positive"]),
    "@@NSIG@@": str(M["n_significant"]),
    "@@GMEAN@@": f"{M['gain_pct_mean']:.2f}",
    "@@GSD@@": f"{M['gain_pct_sd']:.2f}",
    "@@GMIN@@": f"{M['gain_pct_min']:+.2f}",
    "@@GMAX@@": f"{M['gain_pct_max']:+.2f}",
    "@@GTMEAN@@": f"{M['gain_t_mean']:+.2f}",
    "@@FARMP@@": pv(M["farm_level_t_p"]),
    "@@FARMWP@@": pv(M["farm_level_wilcoxon_p"]),
    "@@RANK7@@": str(rank7),
    "@@SINGLEPCT@@": f"{single_pct:.1f}",
    "@@RBASE@@": f"{r_base:.2f}",
    "@@PBASE@@": pv(p_base),
    "@@RHOBASE@@": f"{rho_base:.2f}",
    "@@NRISING@@": str(W["n_rising"]),
    "@@MEANR@@": f"{W['mean_r']:.2f}",
    "@@SLOPE@@": f"{W['mean_slope']:.1f}",
    "@@NABOVE5@@": str(M["n_farms"] - W["n_peak_below_5pct"]),
    "@@MAXPEAK@@": f"{W['max_peak_pct']:.1f}",
    "@@LAMMED@@": f"{lam_med:.1f}",
    "@@LAMLO@@": f"{lam_lo:.1f}",
    "@@LAMHI@@": f"{lam_hi:.1f}",
    "@@TABLEFARMS@@": table_farms,
    "@@TABLESWEEP@@": table_sweep,
}


def render(text):
    for k, v in TOK.items():
        text = text.replace(k, v)
    return text


if __name__ == "__main__":
    for name, block in [("results", RESULTS6), ("discussion", DISC6),
                        ("conclusion", CONCL6)]:
        out = os.path.join(P, f"_multifarm_{name}.tex")
        with open(out, "w") as fh:
            fh.write(render(block))
        print(f"wrote {out}")
    print()
    print(f"headline gain   : {M['gain_pct_mean']:.2f}% "
          f"(was {single_pct:.1f}% on one farm)")
    print(f"farms positive  : {M['n_positive']}/{M['n_farms']}, "
          f"significant {M['n_significant']}/{M['n_farms']}")
    print(f"rises with lam  : {W['n_rising']}/{M['n_farms']} farms, r={W['mean_r']:.2f}")
    print(f"crosses 5% at   : lambda {lam_med:.2f} ({lam_lo:.2f}-{lam_hi:.2f})")
    print(f"gain vs baseline: r={r_base:.2f} (p={p_base:.4f})")
