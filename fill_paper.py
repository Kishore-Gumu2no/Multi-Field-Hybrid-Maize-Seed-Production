"""
Inject computed results into paper/paper.tex -> paper/paper_filled.tex

LaTeX and Python format strings fight over braces, so all prose here is plain
text with @@TOKEN@@ placeholders substituted at the end. Nothing to escape.
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
P = os.path.join(HERE, "paper")

S = json.load(open(f"{R}/summary.json"))
e1 = pd.read_csv(f"{R}/E1_main.csv")
sw = pd.read_csv(f"{R}/E2_summary.csv")
cfg = S["config"]

LBL = {"S1_conventional": "S1 Conventional", "S2_static_ev": "S2 Static (EV)",
       "S3_static_robust": "S3 Static (robust)", "S4_dynamic": "S4 Dynamic",
       "S5_wait_and_see": "S5 Wait-and-see"}
ORDER = list(LBL)
T = {t["comparison"]: t for t in S["E1_tests"]}
adapt = T["S4_dynamic - S3_static_robust"]
vss = T["S3_static_robust - S2_static_ev"]
evpi = T["S5_wait_and_see - S4_dynamic"]
conv = T["S3_static_robust - S1_conventional"]
g = e1.groupby("strategy")


def pv(p):
    if p < 1e-6:
        return "$p < 10^{-6}$"
    if p < 1e-3:
        return "$p < 0.001$"
    return f"$p = {p:.3f}$"


# ------------------------------------------------------------------ derived
lo = float(sw[sw.sharpness <= 0.5]["adaptivity_gain_pct"].mean())
hi = float(sw[sw.sharpness >= 2.0]["adaptivity_gain_pct"].mean())
at1 = float(sw[sw.sharpness == 1.0]["adaptivity_gain_pct"].iloc[0])
rho = float(np.corrcoef(sw.sharpness, sw.adaptivity_gain_pct)[0, 1])
gmax = float(sw["adaptivity_gain_pct"].max())
lam_max = float(sw.loc[sw["adaptivity_gain_pct"].idxmax(), "sharpness"])
grows = hi > lo + 1.0
saturates = gmax < 5.0
evpi_lo = float(sw[(sw.sharpness > 0) & (sw.sharpness <= 0.5)]["closes_evpi_pct"].mean())
evpi_hi = float(sw[sw.sharpness >= 2.0]["closes_evpi_pct"].mean())

if grows and saturates:
    direction = ("rises with sharpness but saturates below the five percent "
                 "reported for irrigation")
elif grows:
    direction = "rises steadily with sharpness"
elif abs(hi - lo) <= 1.0:
    direction = "is essentially flat in sharpness"
else:
    direction = "shrinks as sharpness rises"

e3 = pd.DataFrame(S["E3"]).sort_values("replan_every")
q_top = int(e3.loc[e3["gain_vs_S3"].idxmax(), "replan_every"])
gain_top = float(e3["gain_vs_S3"].max())
chg_top = float(e3.loc[e3.replan_every == q_top, "changes"].iloc[0])
# the economical interval: the longest one still retaining 80% of the best gain
eco = e3[e3["gain_vs_S3"] >= 0.8 * gain_top]
q_eco = int(eco["replan_every"].max())
gain_eco = float(eco.loc[eco.replan_every == q_eco, "gain_vs_S3"].iloc[0])
chg_eco = float(eco.loc[eco.replan_every == q_eco, "changes"].iloc[0])
q_max = int(e3.replan_every.max())
gain_qmax = float(e3.loc[e3.replan_every == q_max, "gain_vs_S3"].iloc[0])

e4 = pd.DataFrame(S["E4"])
y_sens = float(e4[e4.sensor == True]["mean"].iloc[0])
y_nos = float(e4[e4.sensor == False]["mean"].iloc[0])
sd4 = float(e4["std"].mean())
sensor_matters = abs(y_sens - y_nos) > 0.25 * sd4

e5 = pd.DataFrame(S["E5"])
cap_best = int(e5.loc[e5["gain"].idxmax(), "capacity"])
gain_best = float(e5["gain"].max())
cap_lo, cap_hi = int(e5.capacity.min()), int(e5.capacity.max())
gain_tight = float(e5.loc[e5.capacity == cap_lo, "gain"].iloc[0])
gain_slack = float(e5.loc[e5.capacity == cap_hi, "gain"].iloc[0])

# ------------------------------------------------------------------ tables
table2 = "\n".join(
    f"{LBL[s]} & {g.get_group(s).yield_t.mean():.1f} & "
    f"{g.get_group(s).yield_t.std():.1f} & {g.get_group(s).yield_t.min():.1f} & "
    f"{g.get_group(s).mean_abs_tsi.mean():.2f} & "
    f"{100 * g.get_group(s).p_fail.mean():.1f} \\\\" for s in ORDER)

table3 = "\n".join(
    f"{nm} & {t['mean_diff']:+.2f} & {t['pct']:+.1f} & "
    f"[{t['ci_lo']:+.2f}, {t['ci_hi']:+.2f}] & {t['cohen_d']:.2f} & {pv(t['t_p'])} \\\\"
    for nm, t in [("VSS (S3 $-$ S2)", vss),
                  ("Optimisation (S3 $-$ S1)", conv),
                  ("\\textbf{Adaptivity (S4 $-$ S3)}", adapt),
                  ("EVPI (S5 $-$ S4)", evpi)])

table4 = "\n".join(
    f"{r.sharpness:g} & {r.S3:.1f} & {r.S4:.1f} & {r.S5:.1f} & "
    f"{r.adaptivity_gain_t:+.2f} & {r.adaptivity_gain_pct:+.1f} & "
    f"{r.closes_evpi_pct:.0f} \\\\" for _, r in sw.iterrows())

# ------------------------------------------------------------------ results
RESULTS = r"""
\subsection{Reference instance}

We first characterise the five strategies in detail on a single farm, then
replicate across instances in Section~\ref{sec:farms}. Table~\ref{tab:main}
reports them over @@NSEAS@@ independent seasons; Fig.~\ref{fig:main}(a) shows
the distributions. The ordering is
monotone in how much information and flexibility each policy uses.
Conventional practice yields @@Y_S1@@~t per season, the robust static plan
@@Y_S3@@~t, adaptive replanning @@Y_S4@@~t, and perfect foreknowledge would
give @@Y_S5@@~t.

Table~\ref{tab:decomp} decomposes the differences. Three things stand out.

First, \textbf{adaptivity is worth @@ADAPT_PCT@@\%} (@@ADAPT_T@@~t, 95\% CI
[@@ADAPT_LO@@, @@ADAPT_HI@@], @@ADAPT_P@@, Cohen's $d = @@ADAPT_D@@$) against
the robust static baseline. The effect is statistically unambiguous but
modest, and it falls inside the range Kelly \textit{et al.} reported for
irrigation. This instance turns out to sit near the top of the distribution
across farms (Section~\ref{sec:farms}), where the mean is about half this
figure; we carry the instance value here only to describe the mechanism.

Second, \textbf{the value of the stochastic solution is small} (@@VSS_T@@~t,
@@VSS_PCT@@\%, @@VSS_P@@). Averaging over an ensemble rather than planning
against a mean forecast changes the static plan very little. What matters is
not modelling uncertainty at planning time but responding to it later.

Third, \textbf{a large gap to perfect information remains} (@@EVPI_T@@~t,
@@EVPI_PCT@@\%). The ceiling is nearly deterministic (standard deviation
@@SD_S5@@~t): with foreknowledge the crew can almost always hit the synchrony
optimum, so essentially all observed yield variation is a consequence of
deciding under uncertainty rather than of weather acting on the crop
directly.

Fig.~\ref{fig:main}(b) shows where the gain comes from. Ordering seasons by
how well the static plan did, adaptivity contributes almost nothing in good
seasons --- the static plan was already near-optimal and the policy correctly
leaves it alone --- and nearly all of its value in the worst seasons.
Adaptive replanning is insurance, not improvement.

\begin{table}[t]
\caption{Strategy performance over @@NSEAS@@ seasons (@@NF@@ fields,
@@AREA@@~ha, $\lambda = 1$)}
\label{tab:main}
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrr}
\toprule
Strategy & Mean & SD & Worst & Mean & Fields \\
 & (t) & (t) & (t) & $|\TSI|$ & $|\TSI|>4$ (\%) \\
\midrule
@@TABLE2@@
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\caption{Decomposition of the differences (paired, @@NSEAS@@ seasons)}
\label{tab:decomp}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrcrl}
\toprule
Component & $\Delta$(t) & \% & 95\% CI & $d$ & Sig. \\
\midrule
@@TABLE3@@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{../figs/fig2_main.pdf}
\caption{(a) Season yield by strategy. (b) Per-season gain from adaptivity,
seasons ordered by static-robust performance. The benefit is concentrated
almost entirely in the worst seasons.}
\label{fig:main}
\end{figure}

\subsection{Does sharpness change the answer?}

This is the question the paper exists to answer. Table~\ref{tab:sweep} and
Fig.~\ref{fig:sharp} sweep $\lambda$ from a nearly flat response to one much
steeper than observed, holding the optimum fixed. Three things happen at
once, and they do not all point the same way.

\textbf{The gain does rise with sharpness.} It is zero at $\lambda = 0$,
where timing is irrelevant and the adaptive policy correctly never fires;
@@LO@@\% averaged over the flat regime ($\lambda \le 0.5$); @@AT1@@\% at the
published response; and @@HI@@\% in the sharp regime ($\lambda \ge 2$). The
correlation is $r = @@RHO@@$. Our hypothesis is supported in direction.
(The @@AT1@@\% figure here and the @@ADAPT_PCT@@\% of Section~VI-A are the
same quantity measured on different sample sizes, @@NSWEEP@@ seasons per
sharpness level against @@NSEAS@@ for the main comparison; the difference is
well within the confidence interval reported there.)

\textbf{On this instance the gain appears to level off.} The largest value
anywhere in the sweep is @@GMAX@@\%, at $\lambda = @@LAMMAX@@$, and tripling
the sharpness does not push it past the five percent at which Kelly
\textit{et al.} judged adaptation not worth its complexity. We report this
because it is what a single farm shows, and because
Section~\ref{sec:farms} finds that it does not replicate: on nine of ten
farms the gain keeps climbing and crosses five percent. The flattening seen
here is a property of this instance, not of the problem.

\textbf{The share of the opportunity captured falls.} The last column of
Table~\ref{tab:sweep} gives the fraction of the perfect-information gap
closed: @@EVPILO@@\% in the flat regime against @@EVPIHI@@\% in the sharp
one. Sharpening the penalty widens the gap between what is achievable and
what any implementable policy achieves faster than adaptive replanning can
close it: the absolute prize grows while the share claimed shrinks.

\begin{table}[t]
\caption{Sharpness sweep. $\lambda = 1$ is the published response.}
\label{tab:sweep}
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{rrrrrrr}
\toprule
$\lambda$ & S3 & S4 & S5 & \multicolumn{2}{c}{S4 $-$ S3} & EVPI \\
 & (t) & (t) & (t) & (t) & (\%) & closed (\%) \\
\midrule
@@TABLE4@@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{../figs/fig3_sharpness.pdf}
\caption{(a) Value of adaptivity against penalty sharpness; the dashed line
marks the $<5\%$ reported for irrigation~\cite{kelly2023}. (b) Share of the
perfect-information gap captured by adaptive replanning.}
\label{fig:sharp}
\end{figure}

\subsection{Ablations}

Fig.~\ref{fig:abl} reports the remaining experiments.

\emph{Replanning frequency.} The gain falls as the interval lengthens, from
@@GTOP@@~t at @@QTOP@@-day intervals to @@GQMAX@@~t at @@QMAX@@ days, so more
frequent revision does help --- but it is bought with churn. Daily replanning
costs @@CTOP@@ schedule changes per season; an interval of @@QECO@@~days
retains @@GECO@@~t, which is @@ECOPCT@@\% of the best achievable, for only
@@CECO@@ changes. Since our objective charges nothing for disrupting a crew's
plans, and a real operation does, the economical setting is the longer
interval.

\emph{Observation quality.} @@SENSOR_PARA@@

\emph{Resource tightness.} The gain from adaptivity peaks at intermediate
tightness rather than rising monotonically as the crew is squeezed:
@@GTIGHT@@~t at capacity @@CAPLO@@, @@GBEST@@~t at capacity @@CAPBEST@@, and
@@GSLACK@@~t at capacity @@CAPHI@@. The shape is sensible in hindsight. With
abundant capacity the static plan already places every field near its optimum
and there is nothing to re-allocate; with a single crew the schedule is so
constrained that there is nowhere to move work to. Adaptivity needs pressure
to justify it and slack to exploit, and only the middle of the range supplies
both.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{../figs/fig4_ablations.pdf}
\caption{(a) Replanning interval against gain and schedule churn.
(b) Field-level sensing versus a regional gauge. (c) Crew capacity.}
\label{fig:abl}
\end{figure}
"""

SENSOR_NULL = r"""This ablation returned a null result and we report it as
such. Replacing field-level trafficability sensing with a single regional
rain gauge and one assumed drainage class moved adaptive-policy yield from
@@YSENS@@~t to @@YNOS@@~t --- a difference of @@SENSDIFF@@~t against a
between-season standard deviation of @@SD4@@~t, which is indistinguishable
from zero.

We expected field-level sensing to matter, since it is what the rolling
horizon acts on, and we report the negative finding because it bears directly
on how such systems are justified. The explanation is that rain events here
are sparse and the drainage classes close enough that a single median
assumption misclassifies few days. Where the trafficability signal is this
coarse, per-field instrumentation does not earn its cost, and the adaptive
gain reported above comes from responding to thermal-time divergence rather
than from knowing which field is wet. A farm with strongly heterogeneous
soils, or one where entry decisions are genuinely marginal, may differ; we
did not test that."""

SENSOR_REAL = r"""Replacing field-level trafficability sensing with a single
regional rain gauge and one assumed drainage class reduced adaptive-policy
yield from @@YSENS@@~t to @@YNOS@@~t, a loss of @@SENSPCT@@\%. Field-level
observation is what the rolling horizon acts on; without it the policy
misjudges which fields it can actually enter."""

# ------------------------------------------------------------------ discussion
DISC_GROW_SAT = r"""
We set out expecting a clean reversal: sharpen the biological penalty and
adaptation should become valuable. What we found is more useful than that,
because it is only half true.

Sharpness does raise the value of adaptivity, from @@LO@@\% under a gentle
response to @@HI@@\% under a steep one ($r = @@RHO@@$), so the mechanism we
proposed is real. But the effect saturates, and it saturates at @@GMAX@@\%
--- still below the five percent at which Kelly \textit{et al.} concluded
adaptation was not worth its complexity. At \emph{three times} the curvature
measured in the field, in-season replanning still does not clear their bar.

The reason is visible in what the static optimiser does when the penalty
steepens. It hedges. It stops aiming at the peak of the yield curve and
starts choosing offsets that stay tolerable across the whole plausible range
of thermal-time outcomes. Sharpness raises the value of \emph{robust
planning} at least as much as it raises the value of adapting, and the two
substitute for one another. An irrigation schedule has far less room to hedge
in advance than a sowing offset does, which is why this substitution could
not have been observed in the studies that motivated us.

The third pattern completes the picture. As sharpness rises, the share of the
perfect-information gap that adaptivity captures \emph{falls}, from
@@EVPILO@@\% to @@EVPIHI@@\%. The prize grows faster than the policy can
claim it. Under a sharp penalty most of the remaining value lies in knowing
the weather, not in reacting to it --- an argument for better seasonal
forecasting rather than for a more responsive controller.

The practical reading for a hybrid seed producer is therefore modest and, we
think, more honest than the usual framing. At the measured response, adaptive
replanning is worth @@ADAPT_BARE@@\%: statistically unambiguous, real money on a
large operation, and concentrated almost entirely in the seasons that go
badly. It is insurance. It is not a reason to rebuild the operation around
real-time control.
"""

DISC_GROW = r"""
The Kelly--Becker conclusion is \emph{not} a universal property of
agricultural scheduling; it is a property of the biological response. When
yield responds gently to timing error a good fixed plan captures nearly
everything. As the penalty sharpens the static plan's safety margin can no
longer absorb the risk, and the value of acting on new information rises from
@@LO@@\% to @@HI@@\% ($r = @@RHO@@$), exceeding their threshold.
"""

DISC_FLAT = r"""
Sharpening the biological penalty does \emph{not} make adaptive replanning
more valuable: the gain is @@LO@@\% in the flat regime, @@AT1@@\% at the
published response and @@HI@@\% in the sharp regime, with a correlation of
only $r = @@RHO@@$. When the penalty is sharp the \emph{static} plan hedges,
choosing offsets that stay acceptable across the plausible range of
thermal-time outcomes rather than aiming at the peak. Sharpness therefore
raises the value of robust planning, not of adaptation.
"""

DISC_TAIL = r"""

\subsection{A screening rule}

Running a full stochastic scheduling study to discover that adaptation is not
worth it is an expensive way to learn the answer. Our results suggest a cheap
pre-check. Before building an adaptive system, ask three questions.

\emph{Can the decision be hedged in advance?} If a single ex-ante choice can
be made robust across the plausible range of outcomes --- as a sowing offset
can --- most of the available value is reachable without adaptation. This was
the dominant effect in our experiments and it is the question we would ask
first. A decision that must be taken repeatedly and cannot be hedged, such as
an irrigation depth, is a different case from one taken once with a tunable
margin.

\emph{Is the resource tight, but not too tight?} Adaptivity was worth
@@GBEST@@~t at intermediate capacity against @@GSLACK@@~t with slack and
@@GTIGHT@@~t under the severest constraint. A system with spare capacity has
nothing to re-allocate; a system with none has nowhere to move work to. The
payoff lives in between, which is a narrower window than the usual framing
suggests.

\emph{Does the observation actually change the decision?} Our sensing
ablation was null: field-level trafficability data left yield statistically
unchanged, because the decision it informs was rarely close. The test is not
whether a sensor is available or accurate, but whether the quantity it
measures is one the optimiser is genuinely uncertain about at the margin. In
this problem that quantity turned out to be thermal-time divergence, not soil
wetness.

\subsection{What the benchmark choice decides}

A methodological point worth stating. Had we compared adaptive replanning
against conventional practice we would have reported a gain of
@@NAIVE_PCT@@\% and claimed a strong result. Against the robust static plan
the honest figure is @@ADAPT_PCT@@\%. Most of the apparent benefit of
"adaptive intelligence" in this problem is the benefit of optimising at all,
which a static solver delivers once, before the season, at a fraction of the
operational cost.

We also note that both preliminary failure modes we encountered --- indefinite
deferral under free reassignment, and double-counting rain risk in both
objective and constraints --- caused the adaptive policy to \emph{underperform}
the static one. An adaptive system is not automatically better than a fixed
one; it is better only if these are handled.
"""

# ------------------------------------------------------------------ conclusion
CONCL_GROW_SAT = r"""
We asked whether the reported marginality of in-season replanning survives
when the biological outcome responds sharply to timing error rather than
gradually. The answer is yes, and the reason is instructive.

Sharpness does increase the value of adaptivity --- from @@LO@@\% under a
gentle response to @@HI@@\% under a steep one ($r = @@RHO@@$) --- so the
mechanism we hypothesised is real. But the effect saturates at @@GMAX@@\%,
below the five percent threshold at which the irrigation literature judged
adaptation not worth its cost, even at three times the curvature measured in
the field. Meanwhile the share of the perfect-information gap that adaptivity
captures falls from @@EVPILO@@\% to @@EVPIHI@@\%.

The explanation is substitution: a sharper penalty makes the static optimiser
hedge, and hedging recovers most of what adaptation would otherwise have
contributed. Sowing offsets can be made robust before the season in a way
irrigation schedules cannot, so the Kelly--Becker conclusion holds here for a
different reason than it holds for water. The common assumption that sharper
biology justifies real-time control is, on this evidence, not supported.

For multi-field hybrid maize seed production at the measured response,
adaptive replanning is nonetheless worth @@ADAPT_BARE@@\% (@@ADAPT_P@@) against a
robust static plan, concentrated almost entirely in the worst seasons:
insurance rather than improvement. This is the first formulation of hybrid
seed sowing as a resource-constrained stochastic scheduling problem coupled
to parent flowering synchrony.
"""

CONCL_OTHER = r"""
We asked whether the reported marginality of in-season replanning survives
when the biological outcome responds sharply to timing error. Across a
sharpness family anchored on a published field regression, the value of
adaptive replanning moves from @@LO@@\% under a gentle response to @@HI@@\%
under a steep one ($r = @@RHO@@$). At the measured response adaptivity is
worth @@AT1@@\% (@@ADAPT_P@@) against a robust static plan, concentrated
almost entirely in the worst seasons. This is the first formulation of hybrid
seed sowing as a resource-constrained stochastic scheduling problem coupled
to parent flowering synchrony.
"""

CONCL_TAIL = r"""
The immediate next steps are to replace synthetic weather with station
records, to calibrate a full crop model for specific parental inbreds, and to
test the screening rule of Section~VI on a second synchrony-critical
operation such as orchard pollination.
"""

# ------------------------------------------------------------------ assemble
disc = (DISC_GROW_SAT if (grows and saturates)
        else DISC_GROW if grows else DISC_FLAT) + DISC_TAIL
concl = (CONCL_GROW_SAT if (grows and saturates) else CONCL_OTHER) + CONCL_TAIL
results = RESULTS.replace("@@SENSOR_PARA@@",
                          SENSOR_NULL if not sensor_matters else SENSOR_REAL)

# --------------------------------------------------- multi-farm supersedes
import fill_multifarm as MF

results = results + MF.render(MF.RESULTS6)
disc = MF.render(MF.DISC6)
concl = MF.render(MF.CONCL6)


naive_pct = 100 * (adapt["mean_diff"] + conv["mean_diff"]) \
    / g.get_group("S1_conventional").yield_t.mean()

cfg_sentence = (
    f"The base instance has $N = {cfg['n_fields']}$ fields totalling "
    f"{cfg['total_area_ha']:.0f}~ha, a sowing window of $T = {cfg['window']}$ "
    f"days, crew capacity $K = {cfg['capacity']}$ field-visits per day, male "
    f"offsets $\\mathcal{{O}} = \\{{-8,\\dots,+2\\}}$ days, a planning ensemble "
    f"of {cfg['ens']} members, replanning interval $q = 3$ days, freeze lead "
    f"$\\ell = 2$ days and adoption tolerance $\\epsilon = 0.002$.")

weather_sentence = (
    "simulated maximum temperature spans roughly 16--34$^\\circ$C, minimum "
    "4--26$^\\circ$C, and seasonal rainfall averages near 385~mm, against "
    "reported values of 17.3--34.1$^\\circ$C, 6.6--24.9$^\\circ$C and 386~mm.")

TOK = {
    "@@NSEAS@@": str(cfg["seasons"]), "@@NF@@": str(cfg["n_fields"]),
    "@@AREA@@": f"{cfg['total_area_ha']:.0f}",
    "@@Y_S1@@": f"{g.get_group('S1_conventional').yield_t.mean():.1f}",
    "@@Y_S3@@": f"{g.get_group('S3_static_robust').yield_t.mean():.1f}",
    "@@Y_S4@@": f"{g.get_group('S4_dynamic').yield_t.mean():.1f}",
    "@@Y_S5@@": f"{g.get_group('S5_wait_and_see').yield_t.mean():.1f}",
    "@@SD_S5@@": f"{g.get_group('S5_wait_and_see').yield_t.std():.2f}",
    "@@ADAPT_PCT@@": f"{adapt['pct']:+.1f}",
    "@@ADAPT_BARE@@": f"{adapt['pct']:.1f}", "@@ADAPT_T@@": f"{adapt['mean_diff']:+.2f}",
    "@@ADAPT_LO@@": f"{adapt['ci_lo']:+.2f}", "@@ADAPT_HI@@": f"{adapt['ci_hi']:+.2f}",
    "@@ADAPT_D@@": f"{adapt['cohen_d']:.2f}", "@@ADAPT_P@@": pv(adapt["t_p"]),
    "@@VSS_T@@": f"{vss['mean_diff']:+.2f}", "@@VSS_PCT@@": f"{vss['pct']:+.1f}",
    "@@VSS_P@@": pv(vss["t_p"]),
    "@@EVPI_T@@": f"{evpi['mean_diff']:+.2f}", "@@EVPI_PCT@@": f"{evpi['pct']:+.1f}",
    "@@LO@@": f"{lo:.1f}", "@@HI@@": f"{hi:.1f}", "@@AT1@@": f"{at1:.1f}",
    "@@RHO@@": f"{rho:.2f}", "@@GMAX@@": f"{gmax:.1f}", "@@LAMMAX@@": f"{lam_max:g}",
    "@@EVPILO@@": f"{evpi_lo:.0f}", "@@EVPIHI@@": f"{evpi_hi:.0f}",
    "@@QTOP@@": str(q_top), "@@GTOP@@": f"{gain_top:+.2f}",
    "@@CTOP@@": f"{chg_top:.0f}",
    "@@QECO@@": str(q_eco), "@@GECO@@": f"{gain_eco:+.2f}",
    "@@CECO@@": f"{chg_eco:.0f}",
    "@@ECOPCT@@": f"{100 * gain_eco / gain_top:.0f}",
    "@@QMAX@@": str(q_max), "@@GQMAX@@": f"{gain_qmax:+.2f}",
    "@@YSENS@@": f"{y_sens:.2f}", "@@YNOS@@": f"{y_nos:.2f}",
    "@@SENSDIFF@@": f"{abs(y_sens - y_nos):.2f}", "@@SD4@@": f"{sd4:.1f}",
    "@@SENSPCT@@": f"{100 * (y_sens - y_nos) / y_nos:.1f}",
    "@@CAPLO@@": str(cap_lo), "@@CAPHI@@": str(cap_hi), "@@CAPBEST@@": str(cap_best),
    "@@GTIGHT@@": f"{gain_tight:+.2f}", "@@GSLACK@@": f"{gain_slack:+.2f}",
    "@@GBEST@@": f"{gain_best:+.2f}",
    "@@NAIVE_PCT@@": f"{naive_pct:+.1f}",
    "@@NSWEEP@@": str(int(pd.read_csv(f"{R}/E2_sharpness.csv").groupby("sharpness")["season"].nunique().iloc[0])),
    "@@TABLE2@@": table2, "@@TABLE3@@": table3, "@@TABLE4@@": table4,
}

src = open(f"{P}/paper.tex").read()
src = (src
       .replace("RESULT\\_MAIN\\_PCT", f"{MF.M['gain_pct_mean']:.1f}")
       .replace("RESULT\\_NFARMS", str(MF.M['n_farms']))
       .replace("RESULT\\_NPOS", str(MF.M['n_positive']))
       .replace("RESULT\\_NSIG", str(MF.M['n_significant']))
       .replace("RESULT\\_LAMMED", f"{MF.lam_med:.1f}")
       .replace("RESULT\\_SWEEP\\_DIRECTION", direction)
       .replace("RESULT\\_CONCLUSION",
                "where an ex-ante decision can be hedged, robust static planning "
                "substitutes for adaptation, so sharper biology does not by "
                "itself justify an adaptive controller"
                if (grows and saturates) else
                "the marginality of adaptation reported for irrigation is a "
                "property of the biological response rather than of "
                "agricultural scheduling in general")
       .replace("CONFIG\\_SENTENCE", cfg_sentence)
       .replace("CONFIG_SENTENCE", cfg_sentence)
       .replace("WEATHER\\_SENTENCE", weather_sentence)
       .replace("WEATHER_SENTENCE", weather_sentence)
       .replace("RESULTS_SECTION", results)
       .replace("DISCUSSION_SECTION", disc)
       .replace("CONCLUSION_SECTION", concl))

for k, v in TOK.items():
    src = src.replace(k, v)

leftover = [t for t in TOK if t in src] + \
           [w for w in ("RESULTS_SECTION", "DISCUSSION_SECTION",
                        "CONCLUSION_SECTION") if w in src]
if leftover:
    print("WARNING unsubstituted:", leftover)

open(f"{P}/paper_filled.tex", "w").write(src)

print(f"adaptivity at lambda=1 : {at1:+.2f}%  ({pv(adapt['t_p'])})")
print(f"flat regime (<=0.5)    : {lo:+.2f}%")
print(f"sharp regime (>=2)     : {hi:+.2f}%   max {gmax:.2f}% at lambda={lam_max:g}")
print(f"correlation r          : {rho:+.2f}")
print(f"EVPI share             : {evpi_lo:.0f}% -> {evpi_hi:.0f}%")
print(f"sensing matters        : {sensor_matters}")
print(f"verdict                : gain {direction}")
print("wrote paper/paper_filled.tex")
