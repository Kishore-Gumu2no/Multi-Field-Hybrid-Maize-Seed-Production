# Remaining scientific limitations — simulated sensing

Ordered by how much each could change the conclusion.

1. **The real between-field heterogeneity is unknown.** The study reports
   decision value as a function of sigma; it does not establish sigma for a
   commercial seed-production block. Published same-site contrasts at 5 cm are
   1–6 degC between deliberately imposed treatments, but those treatments are
   not applied differentially across a uniformly managed block. The one
   continuous relationship available (0.0112–0.0293 degC per point of residue
   cover) implies 0.11–0.59 degC between similarly managed fields. The
   conclusion flips sign inside that range.

2. **Robustness to capacity accounting is not established.** Only the visits
   model solved within the 0.005 acceptance gap at N=60. Operations, hours and
   hectares all terminated time-limited with gaps of 0.011–0.034 at a 45 s
   limit — wider than the effect — and one produced a negative oracle gain that
   cannot be real. Raising the limit to 300 s left those gaps unchanged to five
   decimals, so this is LP bound looseness for these cost structures, not
   search time. Settling it needs a stronger formulation or valid inequalities,
   not a longer budget.

3. **No measurement of any kind was taken.** Every sensor reading is
   synthetic. No probe was built, deployed or tested, and no claim is made
   about real instrument behaviour beyond adopting a documented +/- 0.5 degC
   thermistor accuracy as the noise scale.

4. **The emulator's error exceeds the decision unit.** 2.62 d RMSE against the
   DSSAT table is larger than the one-day integer sowing step. The comparison
   is paired so this cancels; the bias that would not cancel is 0.14 d. No
   absolute claim about a flowering date rests on this model.

5. **sigma = 0.7 degC is an inherited assumption, not a measurement.** It was
   already in `sim.core.make_farm` before this workstream, which is what makes
   it defensible as a default rather than a tuned value. It is still an
   assumption.

6. **One farm design, one site.** Ranebennur weather, the study's own
   stratified parent pairs, a Rabi window of day 274–334, N = 60 for the
   constrained results and N = 12 for the unconstrained ones. Seeds vary the
   draw, not the design. Results at the two farm sizes are not directly
   comparable; only the matched-N comparison at N = 60 isolates the capacity
   effect.

7. **No seed-yield claim.** The objective is the study's own area-weighted
   relative-yield response to anthesis–silking interval. Nothing here measures
   seed yield, germination or seed quality.

8. **Heterogeneity is static and additive.** Each field's offset is constant
   through the season and applied equally to the day and night soil
   temperatures. Real offsets vary with wetting, canopy closure and depth, and
   the DOPT cap at 34 degC attenuates large offsets asymmetrically — an effect
   the code models but the experiments do not explore.
