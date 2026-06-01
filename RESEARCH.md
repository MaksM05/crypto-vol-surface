# RESEARCH.md — Intraday Vol Around the NY Equity Open

**Status: PRE-REGISTRATION. Locked at commit time. Do not edit the
hypotheses, windows, metrics, sample rule, or statistical method after the
git timestamp — that timestamp is the entire point.** Anything learned
during analysis that would have changed the design goes in a *new* dated
section at the bottom (an amendment log), never as a silent edit above.

This note pre-registers a single, small-sample intraday study before any of
its data exists. The motivation, the exact quantities, the stopping rule,
and the test are all fixed here first so the later results section cannot be
the product of choosing the analysis that happened to produce a p-value.

## Motivation

A prior spot backtest on this project found a consistent short bias at
13:30 UTC — the New York equity cash open (09:30 ET). This note asks the
options-market analogue: around the NY open, does BTC front-month ATM
implied vol move, and does the 25-delta risk reversal tilt further toward
downside protection? If the spot-down bias is real and the options market
prices it, both effects should show up as a within-day shift from a quiet
baseline to the NY-open window.

This is an observational study on a few weeks of 5-minute snapshots. It is
powered to detect a within-day shift, not to make a causal claim about why
one exists. Mechanism (US macro releases clustering near the open, US-hours
flow, hedging demand) is explicitly out of scope.

## Design: within-day paired comparison

Each US equity **trading day** contributes at most **one pair** of
observations on the *same* front-month instrument:

- **Baseline window:** 03:00–04:00 UTC (quiet Asia-hours block).
- **Event window:** 12:30–13:30 UTC. This is the hour of **lead-up to** the
  NY open; the window *ends at* 13:30 UTC (the open), it does not straddle
  it. This matches the 13:30-UTC anchor of the prior spot finding.

The unit of analysis is the **within-day difference** `event − baseline`,
not the two windows pooled across days. BTC ATM IV moves over a wide range
day to day (regularly 40s to 70s in vol points); pooling event and baseline
across days would let that between-day regime variance dominate and bury a
within-day NY-open effect of a fraction of a vol point to a few vol points.
Pairing differences out the daily regime by construction and is where the
power is.

### Why EDT matters (and why it is fine here)

`13:30 UTC = 09:30 ET` holds **only under EDT** (US daylight time, UTC−4).
Under EST (UTC−5) the NY open is 14:30 UTC and these windows would be
mis-aligned by one hour. The planned collection window (June 2026 onward,
~4 weeks) lies entirely within EDT, so no DST boundary is crossed and the
UTC-pinned windows are correct as written. **A winter re-run, or any run
crossing the early-November EDT→EST transition, must shift both windows
+1 hour and is a separate pre-registration.**

## Metrics

Both metrics are computed per snapshot, then aggregated per window (see
"Window aggregation"). All vol quantities are in **vol points** (percent),
matching Deribit's `mark_iv` units and the existing validation harness.

### Metric 1 — front-month ATM implied vol

- **Instrument:** the nearest expiry with **≥ 7 days** to expiry at the
  snapshot time ("front month" = nearest *non-sub-week* expiry). The ≥7-day
  floor avoids the sub-week microstructure noise the surface code already
  excludes, and avoids the front expiry rolling to near-zero T mid-study.
- **ATM IV definition:** `σ_ATM = √(w_svi(0, T) / T)`, where `w_svi(0, T)`
  is the per-expiry SVI total-variance fit evaluated at log-moneyness
  `k = 0`. This reuses `calibration.svi.fit_svi` /
  `svi_total_variance` and is exactly the SSVI backbone `θ_T` the surface
  code already builds — no new ATM-extraction code path, no new convention.
- **Liquidity gate:** the SVI fit uses the same filter as
  `analytics/smile_fit.py` (`open_interest > 10`,
  `(best_ask − best_bid)/mark_price < 0.05`, OTM-only, `|k| < 3`) and the
  same **≥ 6 liquid OTM points** minimum. A snapshot whose front-month fit
  fails the gate or fails to converge yields **no Metric-1 value** for that
  snapshot.

### Metric 2 — 25-delta risk reversal

- **Definition (locked):** `RR_25 = σ(25Δ call) − σ(25Δ put)`.
  This is the standard desk convention. Under BTC's documented downside
  skew (puts richer than equidistant calls, `ρ ≈ −0.25`), `RR_25` is
  **negative**, and a *stronger* downside tilt makes it **more negative**.
  (The convention in the earlier NOTES draft, put − call, has the opposite
  sign and made the H2 direction read backwards; corrected here.)
- **Delta used (locked):** the **standard Black-76 forward delta**
  (`pricer.black76.standard_delta`), *not* the inverse/coin-settled delta.
  The 25-delta strikes here are a vol-surface **moneyness coordinate**, and
  the market quotes RR in standard-delta terms. The project's inverse delta
  (`pricer.inverse.inverse_delta`) is the *hedge ratio* — a different object
  with a different purpose — and is deliberately not used for locating the
  RR strikes. Stated explicitly precisely because this project otherwise
  leans hard on the inverse-delta correction.
- **Strike location:** on the **same fitted SVI smile** as Metric 1, solve
  for the strikes where `|standard_delta| = 0.25` (one call side, one put
  side), and read `σ_BS` from the fit at those strikes. Same expiry, same
  liquidity gate, and same snapshot as Metric 1. If either 25Δ strike cannot
  be located on the fitted smile, the snapshot yields **no Metric-2 value**.

## Window aggregation

A 60-minute window holds ~12 five-minute snapshots.

- **Window value** = the **mean** of the per-snapshot metric across all
  valid snapshots in the window (Metric 1 and Metric 2 aggregated
  independently).
- **Minimum coverage:** a window must contain **≥ 6 valid snapshots** for
  that metric to count.
- **Pairing rule:** a trading day forms a usable **pair** for a given metric
  only if *both* its windows qualify (≥ 6 valid snapshots each, same
  front-month expiry on both windows of that day). Days missing either
  window — laptop-sleep gaps, fit failures, holiday — are dropped, not
  partially imputed.

## Sample and stopping rule

- **Minimum sample: 20 complete day-pairs** (days where both windows
  qualify), accumulated **before any analysis is run**. The 20-pair count
  is per metric; if one metric reaches 20 usable pairs before the other,
  collection continues until both do.
- **No peeking.** No test statistic, plot, or summary of the event-vs-
  baseline difference is computed until the 20-pair threshold is met. The
  pipeline (`analytics/intraday_metrics.py`) may be built and unit-tested
  against *existing historical* snapshots for correctness, but the
  event/baseline difference series itself is not inspected before the
  threshold.
- **Trading-day filter (locked):** event days are **US equity trading days
  only**. US market holidays are excluded — there is no NY open on those
  days, so a "13:30 UTC event" on a holiday is a null event that would
  dilute a real effect. Weekends drop out under the same rule. The holiday
  calendar is fixed to the standard US equity market holiday schedule for
  the collection year and is not adjusted after the fact.

## Hypotheses

- **H1 (two-sided):** the mean within-day difference in front-month ATM IV
  (event − baseline) is **non-zero**.
  - H0: mean difference = 0.
- **H2 (one-sided):** the mean within-day difference in `RR_25`
  (event − baseline) is **negative** — i.e. the risk reversal tilts further
  toward downside protection in the NY-open window, consistent with the
  prior spot-down bias.
  - H0: mean difference ≥ 0.

Both hypotheses are tested at **α = 0.05**. There is no multiplicity
correction across the two metrics; they are pre-registered as two distinct
primary hypotheses, and both are reported regardless of outcome.

## Statistical method

- **Unit:** the within-day difference `d_i = event_i − baseline_i` for each
  usable pair `i`.
- **Primary test:** **paired t-test** on `{d_i}` (one-sample t-test of the
  differences against 0; one-sided for H2, two-sided for H1).
- **Robustness test:** **Wilcoxon signed-rank** on `{d_i}` (distribution-
  free, same pairing). Reported alongside the t-test, not as a fallback
  chosen after seeing results.
- **Reported for each metric:** n pairs, mean difference, 95% CI on the
  mean difference, effect size (Cohen's `d_z` for the paired differences),
  and the p-value from both tests.
- Paired t + Wilcoxon **replace** the earlier Welch / Mann-Whitney plan,
  which were unpaired and discarded the within-day pairing that is the
  source of power here.

## Known limitations (documented, not fixed)

- **Serial correlation.** Daily differences may be mildly autocorrelated
  via vol clustering (today's regime resembles yesterday's). With n ≥ 20
  over ~4 weeks we report the paired tests as-is and flag this rather than
  modeling AR structure; explicit AR/HAC treatment is out of v1 scope. This
  may make the nominal CIs slightly optimistic.
- **Capture gaps are non-random.** Laptop-sleep gaps are not a random
  sample of snapshots — they cluster at night (overlapping the baseline
  window) and whenever the machine is closed. The ≥6-snapshot coverage rule
  and the both-windows-qualify pairing rule are the guard; residual bias
  from systematically thinner coverage in one window is a caveat, not a
  correction.
- **Single venue, single asset, one season.** Deribit BTC only, EDT only,
  ~4 weeks. No claim generalizes beyond that. Any result is a localized
  observation, not a stylized fact.
- **Observational, no mechanism.** A detected shift is association around a
  clock time, not evidence of a specific cause.

## Sequencing (operational, not part of the lock)

1. Commit this file to git — the credibility timestamp. **(this step)**
2. Turn on continuous 5-min capture (launchd/cron; accept laptop-sleep
   gaps per the limitation above).
3. Build `analytics/intraday_metrics.py`: a pure-ish orchestrator that, for
   a given snapshot, returns `(front_month_atm_iv, rr_25)` using
   `pricer/` + `calibration/` exactly as the surface code does. Unit-test
   against existing historical snapshots. Pipeline ready before the data is.
4. Accumulate until **both** metrics have ≥ 20 usable day-pairs.
5. Run the pre-registered analysis once, write the results section in a new
   dated block below, ship the note.

## Amendment log

*(Empty at lock time. Any post-timestamp change to the design is recorded
here with a date and a reason, and the original text above is left intact.)*
