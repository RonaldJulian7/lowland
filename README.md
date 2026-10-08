# Lowland

Three linked studies of the Dutch power system, built on **103,200 hours (2015–2026) of
real open data**. They are designed to compose into one pipeline — **forecast → decide →
explain** — rather than to stand as three unrelated demos.

| | Project | Question | Core methods |
|---|---|---|---|
| **1** | **Bellwether** *(the lead sheep whose bell tells you where the flock is going)* | How much residual load, at what price, 24 hours out — and how confident can we be? | Quantile gradient boosting · sequence-to-sequence quantile network · **conformal prediction** · congestion risk |
| **2** | **Flywheel** *(stores energy, releases it at the right moment)* | What is that forecast worth to a grid-scale battery? | Perfect-foresight **MILP** · deterministic & **stochastic MPC** · Gaussian-copula scenarios · **PPO** |
| **3** | **Vantage** *(a position from which you can see far)* | Why do prices move, and where does build-out take them? | **IV / Double ML** causal identification · **HMM** regimes · **conditional VAE** · scenario simulation |

Read the signs, act on them, see further.

Project 2 consumes project 1's predictive distributions. Project 3 cross-checks its
structural simulator against its own causal estimate. Nothing here is a toy: every number
below comes from a rolling-origin backtest on out-of-sample data.

---

## Headline results

### 1 · Bellwether — probabilistic forecasting

Residual load, 24 h ahead, pooled over 4 rolling-origin folds (n = 2,880 scored hours):

| Model | MAE (MW) | CRPS | 90% coverage |
|---|---|---|---|
| Climatology | 1278.6 | 871.9 | 0.865 |
| Seasonal naive | 1565.4 | 1096.6 | 0.834 |
| Linear quantile AR | 1177.2 | 791.2 | 0.912 |
| LightGBM **without weather** | 1118.4 | 746.6 | 0.883 |
| LightGBM | 496.4 | 356.6 | **0.794** |
| LightGBM + conformal | 496.4 | 343.4 | **0.926** |
| Deep TFT + conformal | 444.5 | 301.3 | 0.918 |
| **Ensemble** | **426.4** | **295.3** | 0.950 |

**73% lower MAE than seasonal naive**, at 4.0% MAPE on a mean residual load of ~10.5 GW.

Three findings worth more than the headline number:

- **Conformal calibration fixed coverage *and* improved the proper score.** The raw
  LightGBM 90% interval covered 79.4%; conformalised it covers 92.6%, and CRPS *fell*
  from 356.6 to 343.4. Sharper and better calibrated at once, which is the behaviour the
  theory predicts and the reason it is worth the extra machinery.
- **Weather is worth 56% of the error.** Same model, same hyperparameters, weather
  features removed: MAE 1118 → 496. Holding model class fixed is what makes that
  attributable to the meteorology rather than to a change of estimator.
- **Model families win on different targets.** The sequence model beats LightGBM on
  residual load (CRPS 301 vs 343) and *loses* on price (15.9 vs 14.6). Residual load is a
  smooth physical trajectory; price is spiky and dominated by autoregressive fuel-cost
  dynamics (70% of split gain). Reporting both is more useful than declaring a winner.

Day-ahead price, same protocol: **Ensemble MAE €20.64/MWh, CRPS 14.14, coverage 0.937**,
significantly ahead of the runner-up (Diebold–Mariano p = 0.043).

> **Negative result, reported.** On residual load the ensemble's edge over Deep TFT +
> conformal is **not** statistically significant (DM p = 0.68). The ensemble is the better
> point estimate; the claim that it is a better *model* is not supported by this sample.

### Tested again on winter

The splitter takes the most recent windows, so the default backtest lands in summer. For
a project about congestion that is the wrong season — peak residual load is a winter
problem. Truncating the panel at 1 March re-runs the identical protocol on
**Nov 2025 – Feb 2026**:

| Residual load | Summer | Winter |
|---|---|---|
| Ensemble MAE | 426 MW | 622 MW |
| Ensemble CRPS | 295 | 420 |
| Skill vs seasonal naive | 73% | 72% |
| LightGBM, raw coverage | 0.794 | **0.684** |
| LightGBM + conformal | 0.926 | 0.858 |
| LightGBM + **Mondrian** | 0.917 | **0.880** |
| Ensemble coverage | 0.950 | 0.889 |

Three things fall out of this:

**Errors rise ~46%, but skill barely moves** (72% vs 73% over seasonal naive). The model
is not just riding easy conditions.

**Calibration degrades faster than accuracy.** Raw coverage falls to 0.684 — the
uncalibrated model is badly overconfident in winter, considerably worse than in summer.
That is the honest weakness, and it only appears if you test both seasons.

**Group-conditional calibration earns its place.** In summer, Mondrian and plain conformal
were indistinguishable (0.917 vs 0.926) and the extra machinery looked unjustified. In
winter it wins, 0.880 against 0.858. The argument for it was always that marginal coverage
can hide failure on winter evening peaks; summer could not test that claim and winter did.

Price runs the other way — winter is *easier* (MAE €11.89 against €20.64), because the
summer window covers the volatile August–September 2026 period. The model ranking also
flips: LightGBM beats the sequence model on price in summer, and loses to it in winter.
No single architecture dominates, which is a better argument for the ensemble than any
amount of averaging theory.

### 2 · Flywheel — what the forecast is worth

25 MW / 50 MWh battery, 2,880 hours of genuine out-of-sample forecasts:

| Controller | Profit | % of optimum | Worst day | CVaR 5% | Cycles |
|---|---|---|---|---|---|
| Perfect foresight (MILP) | €903,060 | 100.0% | — | — | 154.7 |
| **Stochastic MPC** | €863,249 | **95.6%** | **+€1,687** | **€2,032** | 147.5 |
| Deterministic MPC | €855,791 | 94.8% | −€479 | €1,496 | 174.4 |
| PPO agent | €777,124 | 86.1% | −€23 | €1,277 | 103.5 |
| Threshold (no forecast) | €390,540 | 43.3% | −€7,017 | −€4,288 | 89.1 |

**Forecasting more than doubles a no-forecast heuristic** (43% → 95% of the achievable
optimum). That is the economic case for project 1, denominated in euros.

The more interesting result is what the *distribution* buys over the *point*. Stochastic
MPC beats deterministic MPC by only 0.9 percentage points of profit — but it does so with
**fewer cycles (147 vs 174)** and a materially better downside: its worst day is +€1,687
against −€479, and its 5% CVaR is 36% higher. **The value of probabilistic forecasting
here is risk management, not mean return.** A paper reporting only mean profit would have
concluded the extra machinery barely pays.

The congestion Pareto sweep prices grid-friendly operation directly: moving from λ=0 to
λ=200 buys ~1,000 MWh of net congestion relief for ~€95k of forgone profit — roughly
**€95 per MWh of relief**, which is the number a flexible-connection contract negotiation
needs.

### 3 · Vantage — causal effect and build-out scenarios

**The merit-order effect**, EUR/MWh per GW of renewable output, 102,073 hours:

| Method | Estimate | 95% CI |
|---|---|---|
| OLS (naive) | −11.98 | [−12.97, −10.99] |
| **IV-2SLS (weather instruments)** | **−13.34** | [−14.42, −12.25] |
| Double ML (cross-fitted) | −16.75 | [−18.18, −15.31] |

First-stage F = **18,313** (weak-instrument threshold ≈ 10). OLS is biased toward zero
relative to IV, exactly as curtailment (low prices *cause* low output) and the feed's
missing behind-the-meter solar would predict.

Estimated **by year**, the coefficient recovers textbook economics with no supervision:

| 2019 | 2020 | 2021 | **2022** | 2023 | 2025 |
|---|---|---|---|---|---|
| −5.66 | −6.52 | −16.81 | **−34.66** | −13.38 | −10.69 |

The effect *is* the slope of the residual supply curve: it steepens when gas is expensive
and the merit order is stretched, and compresses as the system normalises. Any 2030
revenue model that holds this parameter fixed is wrong.

**Price regimes** from a hand-written Gaussian HMM (Baum–Welch, log-space), given no dates
and no labels, across 4,241 days:

| State | Mean price | Volatility | Negative hours/day | Character |
|---|---|---|---|---|
| 0 | €34.82 | 6.5 | 0.00 | pre-2021 calm |
| 2 | €60.12 | 51.0 | **4.64** | renewable-surplus, high-volatility |
| 4 | €214.60 | 58.9 | 0.00 | gas crisis (from Sept 2021) |

It found the crisis and the negative-price regime unsupervised — a genuine external
validity check.

**Build-out scenarios** on a learned residual supply curve:

| Scenario | Mean price | Negative-price share | **Wind capture rate** | Curtailment |
|---|---|---|---|---|
| Base (observed) | €83.77 | 2.6% | **0.913** | 0.0% |
| 2030 offshore push | €75.75 | 3.0% | **0.710** | 0.5% |
| 2035 high renewables | €78.14 | 4.5% | **0.634** | 3.1% |

The capture rate — what a wind farm earns per MWh relative to the average price — falls
from 0.91 to 0.63. **Cannibalisation, not the average price, is what decides whether an
unsubsidised project is financeable.**

The structural simulator agrees with the causal estimate to within 68–77%, and the
agreement ratio *declines* as scenarios grow (0.77 → 0.70 → 0.68). That is the convexity
of the merit order appearing in the data: extrapolating a marginal effect linearly
overstates it, because the system moves onto a flatter part of the supply curve.

---

## Things that went wrong

**The most recent month of data was quietly wrong.** ENTSO-E actuals are published fast
and settled slowly, so the last few weeks under-report. On this snapshot, Dutch load drops
from ~12,500 MW to 8,000–10,500 MW overnight while the day-of-year reference is unchanged.
Training through it is survivable; *scoring* on it inflated test MAE **4.6×** and collapsed
90% coverage from 0.93 to **0.41**. It looked exactly like a broken model.
`detect_provisional_tail` finds the cutoff automatically by comparing daily load against a
climatology built from history older than the test window, and requiring the shortfall to
run contiguously to the end of the sample — so a Christmas week is not mistaken for a
settlement tail. It independently recovered 2026-08-28, the date found by hand.

**Splitting on the target timestamp leaks the future.** For direct multi-horizon
forecasting, a training row whose *origin* sits just before the split has its *target*
inside the test window. The backtester splits on origin and purges one horizon between
train and test; the conformal calibration block is carved off the end of training with its
own purge. `tests/test_correctness.py` asserts every lag is ≥ the horizon by reconstructing
it from the raw series.

**Three solvers failed before the linear baseline worked.** scikit-learn's
`QuantileRegressor` is a linear program that does not finish at 10⁵ rows; statsmodels'
IRLS does not converge on a design this collinear and returned a "baseline" with an MAE of
**294 GW**. Minimising pinball loss directly by gradient descent is convex, converges
monotonically, and cannot diverge.

**A generative model that is never checked is decoration.** The conditional VAE needed four
rounds of honest evaluation:

1. it generated *training-era* price levels on a post-regime-shift holdout → added a
   fuel-cost proxy to the conditioning;
2. `sample()` returned the decoder output, i.e. the conditional **mean**, never adding the
   observation noise back — a real bug that strips variance by construction;
3. i.i.d. noise restored dispersion and destroyed within-day autocorrelation (0.86 → 0.69)
   → estimated the full 24×24 residual covariance and sampled through its Cholesky factor;
4. asking one network to learn level *and* shape across a 7× price-regime change failed →
   decomposed into a level (which the supply curve predicts well) and a shape (which is
   what storage is actually paid for).

**It still under-disperses, and that is reported as a result.** The final generator matches
the level exactly, the median to 4.1% and the 95th percentile to 6.7%, but reproduces only
**58% of the real daily spread** and 17% of negative-price hours. Ex-post sampling from the
aggregate posterior (Ghosh et al., 2020) helped marginally. With 4,241 training days over a
24-dimensional profile, this is the sample size talking. A heavier-tailed likelihood or a
diffusion model would be the next step. **The honest version of this table is more useful
than a version that hides it.**

**The causal cross-check was comparing different questions.** The scenarios grow demand by
up to 28% *and* renewables; the IV coefficient is identified holding load constant. Netting
a merit-order effect against a demand effect gave agreement ratios of 0.03–0.22 and looked
like a broken simulator. Building a demand-neutral twin of each scenario purely for the
comparison moved it to 0.68–0.77 — and revealed the convexity result above.

---

## The interface

The apps are not a default framework theme with the numbers dropped in. The visual system
is deliberate and documented in `lowland/viz.py` and `apps/_theme.py`:

**Type.** Fraunces — a variable serif with optical-size and "wonk" axes — for headings;
Inter Tight for interface text, which sets more compactly than Inter at small sizes;
JetBrains Mono for every figure, axis tick and label, so numbers never jitter as they
update and a value reads as something to be checked rather than glanced at. Hierarchy is
carried by weight, case and letterspacing across a deliberately narrow scale, because a
small scale used consistently reads as designed and a large one rarely does.

**Colour.** Warm paper (`#F7F4ED`) and warm ink (`#131510`) rather than neutral grey, and a
teal-led categorical palette. Both modes were validated with a colour-blindness checker
rather than chosen by eye — worst adjacent CVD ΔE 11.0 light / 8.5 dark against a ≥8 target,
normal-vision ΔE 21.7 in both. Ochre falls below 3:1 on the light surface, so every chart
that uses it carries direct labels and a table view.

The slot *ordering* is part of that safety result, not taste. A semantically nicer ordering
for the generation mix — wind in teal and green, solar in ochre — was tried and **rejected**:
it put rose beside rust at ΔE 11.1, below the floor at which full-colour vision can separate
them. The validated ordering stayed.

**Chrome.** Hairline rules and near-square corners instead of shadows and pills; numbered
monospaced section markers; Streamlit's own header, menu and footer removed. One
implementation note worth recording: Streamlit renders injected HTML through a Markdown
parser, which mangles a `<style>` block two different ways — four-space indentation becomes
a code block, and a blank line terminates the HTML block, silently truncating the
stylesheet at the first paragraph break. Both were happening. The fix is to collapse the
stylesheet to a single line before injection.

---

## Reproducing it

```bash
uv venv --python 3.12 && uv pip install -e .
python -m scripts.build_dataset                       # ~50 API calls, resumable, cached
python -m projects.bellwether.train --all --folds 4  # ~35 min (GPU)
python -m projects.bellwether.train --all --folds 4 --until 2026-03-01 --label winter
python -m projects.flywheel.run --model Ensemble    # ~40 min (MILP + PPO)
python -m projects.vantage.run              # ~25 min
python -m pytest tests -q                             # 24 tests
streamlit run apps/Home.py
```

Everything reruns **offline** from committed parquet snapshots. Ingestion is chunk-cached
and resumable — a long backfill against a rate-limited public API fails partway through as
a matter of course, and without per-chunk persistence the retry hits the same quota wall.

### Data

| Source | Licence | Used for |
|---|---|---|
| [energy-charts.info](https://api.energy-charts.info) (Fraunhofer ISE, from ENTSO-E) | CC BY 4.0 | Load, residual load, generation by type, day-ahead price |
| [Open-Meteo](https://open-meteo.com) | CC BY 4.0 | ERA5 reanalysis + operational NWP at 14 Dutch sites |

No API keys. Every cached dataset records its source, licence, retrieval time and covered
range in `data/cache/_manifest.json`.

Weather is sampled at the locations that matter rather than at a national centroid: four
offshore wind clusters (Borssele, Hollandse Kust, Gemini, Egmond), four onshore clusters,
and six demand centres, each weighted by capacity or population share. Hub-height (100 m)
wind speed is used, not the 10 m standard, and it enters through a normalised turbine power
curve — output is cubic in wind speed below rated and flat above it, so a linear wind
feature is badly mis-specified.

> **A note on Dutch solar.** Metered solar in this feed averages ~85 MW, which looks absurd
> against ~25 GW of installed capacity. It is correct: most Dutch PV is behind-the-meter
> rooftop, invisible to ENTSO-E generation data, and appears as *suppressed load* instead.
> It is consistent across all eleven years, so it is fine for modelling — but it means
> "renewable share" here is metered share, and the solar capacity multipliers in project 3
> move less than they appear to.

---

## Layout

```
lowland/                      shared library
  config.py                paths, Dutch power-system parameters, weather sites
  io/                      provenance-tracked, chunk-cached API clients
  dataset.py               master hourly panel + provisional-data detection
  features.py              leakage-safe direct multi-horizon features
  metrics.py               proper scoring rules, coverage, Diebold–Mariano
  backtest.py              rolling-origin splitter with purging
  conformal.py             split / Mondrian / adaptive conformal prediction
  viz.py                   validated colour system, Plotly chart layer
projects/
  bellwether/             baselines, LightGBM, deep TFT, congestion risk
  flywheel/              MILP dispatch, copula scenarios, RL env, PPO
  vantage/       IV/DML, HMM regimes, conditional VAE, simulator
apps/                      Streamlit: overview + one page per project
tests/                     24 correctness tests
artifacts/reports/         every number in this README
```

### Notes on the implementation

Several components are written from first principles rather than imported, where the
details are the point: the **Baum–Welch recursions** (log-space, to avoid the underflow
that silently corrupts long sequences), **PPO** (so the state-dependent action bounds and
the tanh log-probability correction are visible rather than buried in a wrapper), the
**two-stage least squares** with Newey–West standard errors, and the **interpretable
multi-head attention** in the sequence model, which shares a single value projection so
the averaged attention weights can honestly be read as temporal focus.

The MILP keeps its on/off binaries rather than relaxing them. With strictly positive prices
the relaxation is exact, but Dutch prices are now negative for several hundred hours a year
— and in those hours it genuinely pays to burn energy through the round-trip loss, so the
relaxation stops being exact precisely in the hours that matter most for a battery.

Charts follow a colour system validated with a colour-blindness checker in both light and
dark mode (worst adjacent pair ΔE 9.1 / 8.4 against a ≥8 target). Three light-mode slots
fall below 3:1 contrast, so every chart ships a legend, direct labels and a table view.

---

## Known limitations

- **The conditional VAE under-disperses** (58% of real daily spread). Diagnosed, four
  fixes applied, honestly reported above. Sample size is the binding constraint.
- **The Sargan overidentification test rejects** (p < 0.001). At n = 102,073 this test has
  enormous power and will reject economically trivial violations, but it is a genuine
  warning: irradiance plausibly reaches price through behind-the-meter solar suppressing
  *net* load, not only through metered generation. The IV estimate should be read as
  well-identified in direction and magnitude, not as assumption-free.
- **The RL agent is trained on surrogate forecasts** — realised prices perturbed by an
  autocorrelated error calibrated to project 1's out-of-sample residuals — because genuine
  forecasts exist only for the 2,880-hour evaluation window. It is *evaluated* on genuine
  forecasts. Read the 86.1% as indicative; the MPC results carry no such caveat.
- **Scenarios are not a market model.** No unit commitment, no cross-border flows, no
  interconnector constraints, no strategic bidding. The `extrapolation_share` column states
  how often each scenario leaves the residual-load range ever observed (24% for 2035).
- **One bidding zone, one horizon.** Everything is NL at h=24. The machinery is
  horizon-agnostic but nothing here demonstrates it at h=1 or h=168.
- **Two seasonal windows, not a full year.** Summer and winter are both tested, which is
  enough to show the calibration story changes between them, but spring and autumn are
  untested and the winter run is a single year.

---

Built with Python 3.12, LightGBM, PyTorch (CUDA), PuLP/CBC, statsmodels and Streamlit.
