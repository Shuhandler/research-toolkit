# Reusable notebook research — implemented

Use `import research_toolkit as rt`. These extensions retain Polars tables and
existing default behavior. The runnable [synthetic workflow](../examples/research_workflow.py)
combines allocation, liquidity, nonlinear costs, a ledger, dated risk-free returns,
benchmark statistics and comparisons. No assignment-specific selection/scoring,
private prose, market caches or provider clients are included.

## Inspection and migration

The reference notebook already calls the toolkit for returns, dividend accounting,
financing and portfolio simulation. Its reusable local additions were capped
weights, trailing liquidity estimates, square-root costs and entry sizing, dated
risk-free reporting, rolling benchmark measures, cumulative plots and comparisons.
They now have library entry points; notebook cells can define assumptions, call
these functions, display numerical results and write their own interpretation.

Two distinctions matter when migrating:

- The notebook converts estimated nonlinear costs to fixed per-asset impact bps
  for a sized entry basket. Pass `SquareRootImpactCosts` directly instead: the
  ledger solves sizing and charges the actual orders once. A sizing preview is
  informational; do not subtract its fees from capital or add them as another fee.
- The notebook computes Sharpe using the sample standard deviation of **excess**
  returns. Existing toolkit calculations use sample standard deviation of
  **portfolio** returns. Defaults are preserved. Select
  `sharpe_denominator="excess_returns"` to reproduce the notebook convention.
  These denominators agree for a constant risk-free return but generally differ
  for a changing series. No subtraction of borrowing spread is inferred.

Historical allocation cannot be applied retroactively as independent evidence:
choose the universe and weights using information available before the decision,
then start execution afterward. Assignment-specific selection and retrospective
scoring remain outside the library. Modern adjusted histories and reconstructed
availability are not point-in-time vintages.

## Capped inverse volatility

```python
allocation = rt.inverse_volatility_weights(
    asset_returns, decision_session=decision, lookback=lookback,
    periods_per_year=annualization, max_asset_weight=cap,
    cap_policy="redistribute",  # Default remains "raise".
)
weights = allocation.weights
```

All estimates end strictly before the decision session. `redistribute` implements
`w_i = min(cap, k / sigma_i)`, redistributing the remainder in inverse-volatility
proportions until weights sum to one. It requires `asset_count * cap >= 1`;
zero volatility, inadequate history and infeasible caps raise. This constrains
risky proportions, not net-equity exposure or subsequent weight drift. The
existing scheduled target validator still rejects breaches; it never adjusts a
submitted basket. The result retains `weights`, annualized `estimates`, metadata,
and `diagnostics` (`asset`, `uncapped_weight`, `weight_change`, `at_cap`).

## Liquidity estimation and execution costs

```python
liquidity = rt.estimate_liquidity(
    asset_returns, dollar_volume, decision_session=decision,
    lookback=lookback, metadata=volume_metadata,
)
costs = rt.SquareRootImpactCosts(
    liquidity=liquidity, commission_bps=commissions,
    commission_per_share=per_share_fees, half_spread_bps=half_spreads,
    impact_coefficient=coefficients,
)
preview = rt.estimate_trade_costs(
    orders, costs=costs, execution_session=entry, decision_session=decision,
)
sized = rt.size_entry_orders(
    weights=weights, initial_capital=capital, gross_leverage=leverage,
    prices=entry_marks, costs=costs,
    execution_session=entry, decision_session=decision,
)
# Use the SAME model in buy_and_hold(..., costs=costs) or
# scheduled_rebalance(..., costs=costs). Do not pre-deduct preview costs.
```

`dollar_volume` has exactly `session: Date`, `asset: String`,
`dollar_volume: Float64`, `available_at: Datetime("us", "UTC")`. Metadata requires
nonblank `source` and matching `currency`. The selected return window must have
all assets on all dates, nonnegative volumes and positive mean dollar volume;
availability cannot exceed local midnight of the decision session. Returns use
sample daily standard deviation (`ddof=1`), not annualized volatility. Their
availability inherits the existing prior-session-close assumption; revised inputs
need separate vintage verification. No share-volume conversion is guessed.

`LiquidityResult.estimates` has exactly `asset: String`,
`daily_volatility: Float64`, `dollar_adv: Float64`, `n_obs: Int64` (at least two).
A caller can construct this result from independently estimated inputs. Its
metadata must declare `source`, `currency`, `volatility_unit="daily_decimal"`,
`adv_unit="currency_per_trading_day"`, ISO `sample_end` and `decision_session`,
with sample end strictly before decision. A declaration is not independent proof
of availability. The model copies this snapshot and detects later mutation of it.
All model parameters are finite, nonnegative scalars or exact asset mappings;
explicit zeros are required when a component is absent.

For absolute order notional `Q`, daily decimal volatility `sigma`, daily dollar ADV
`V`, coefficient `eta`, raw reference price `P`, and per-share fee `f`:

- `order_adv_ratio = Q / V`. This is a daily-volume ratio, **not intraday
  participation**, a fill probability or a capacity guarantee; it may exceed one.
- `impact_bps = 10000 * eta * sigma * sqrt(Q / V)` is **impact only**.
- `impact = Q * eta * sigma * sqrt(Q / V)`; `half_spread = Q * half_spread_bps / 10000`.
- `commission = Q * commission_bps / 10000 + (Q/P) * f`.
- `total_cost` sums these components; `total_cost_bps = 10000 * total_cost / Q`
  for nonzero orders. All costs and ratios are zero for zero orders.

`orders` has exactly `asset: String`, `signed_notional: Float64`,
`reference_price: Float64`; `entry_marks` has `asset`, `reference_price`. The
universe must exactly match the model, including zero orders. Reference prices
must be positive. Sizing weights follow existing risky-proportion contracts.
`ExecutionResult.orders` adds `signed_quantity`, `daily_volatility`, `dollar_adv`,
`order_adv_ratio`, `impact_bps`, `commission`, `half_spread`, `impact`,
`total_cost`, `total_cost_bps` (Float64). Costs are currency; bps are basis points.
Metadata retains estimates, coefficients, source identity and timing. Sizing also
returns gross notional, post-cost equity, cash/debt, fees and numerical residual.
The preview does not itself validate a financing contract or margin policy.

The monotone entry solve is `N = L * (C - sum(cost_i(w_i*N)))`. Per-share fees use
fractional quantities at the supplied reference price. The simulated trade stays
at that reference price and costs are cash expenses, with no second adverse fill.
The first holding return continues to include entry costs relative to initial
pre-cost capital. Static weights must already be known before entry; closing
marks size an idealized predetermined order, not a guaranteed auction fill.

Scheduled execution solves the existing post-cost equity equation using costs
of **changed** notionals, including sales and funding constraints for receivables.
Its conservative monotonicity guard bounds marginal expenses over possible
orders: each bound must be below one and the leverage-weighted sum below one.
Unsupported extreme inputs raise; the solver does not guess among ambiguous
solutions. An unchanged basket has no fees. Entry and scheduled calls require
liquidity estimates dated no later than their decision/execution cutoff and the
same currency. One model uses one frozen liquidity snapshot throughout a run;
there is no implicit refresh. Dated model refresh, minimum/fixed-ticket fees,
volume-constrained execution, shorts and order splitting remain unsupported.

`BacktestResult.execution_costs` adds `trade_id` and `session` to the cost-breakdown
schema for actual square-root-model ordinary fills. It is typed empty for fixed
bps models. It reconciles with existing `trades`, `costs` and cash events; it is an
audit view, not an additional debit. Automatic dividend reinvestments retain their
existing zero-cost trade/payment records and are excluded from this model's audit.
A scheduled trade funded by released dividends is still an ordinary costed trade.

## Dated risk-free performance benchmark

```python
rf = rt.risk_free_returns(
    daily_rates, intervals=result.daily.select("period_start", "session"),
    day_count="ACT/360", compounding="daily",
    rate_timing="known_at_accrual_start", timezone_name="America/New_York",
    metadata=rf_metadata,
)
report = rt.performance(
    result, periods_per_year=annualization,
    risk_free_returns=rf, sharpe_denominator="excess_returns",
    minimum_acceptable_return_annual_effective=mar,
    benchmark=benchmark_returns, benchmark_metadata=benchmark_metadata,
)
```

`performance` and `rolling_risk` require exactly one of
`risk_free_annual_effective` and `risk_free_returns`. A dated input is either a
`RiskFreeResult` or a Polars table plus `risk_free_metadata`. The table is exactly
`period_start: Date`, `session: Date`, `simple_return: Float64`. Sort order is free;
both interval endpoints must match the actual portfolio coverage exactly.
Duplicates, nulls, nonfinite values and returns at/below −1 raise. Metadata requires
nonblank `source`, `basis`, `currency`, `frequency="1d"`; currency must match.
Do not supply separate metadata with a `RiskFreeResult`. Scalar effective-annual
conversion remains `expm1(log1p(rate)/periods_per_year)`.

The converter accepts **already assigned calendar-day** rate observations, exactly
`date: Date`, `annual_rate: Float64`, `available_at: Datetime("us", "UTC")`.
Rates are nominal annual **decimals**, not percent values or annual effective
rates. All dates in each `(period_start, session]`, including weekends, are
required; no holiday carry, missing-date fill or spread stripping is implicit.
Each rate must be known by local midnight on its assigned accrual date. Explicit
`ACT/360` or `ACT/365F` divides by 360 or 365. `compounding="daily"` multiplies
daily factors; `"simple"` sums daily accruals. Negative rates are allowed if each
factor and the resulting interval wealth stay positive. Result tables retain the
assigned daily inputs and interval returns with convention metadata.

A saved `SOFRFinancing` run's `financing_accruals` can supply its explicitly
selected `date`, `sofr` renamed `annual_rate`, and `available_at`, with separately
supplied performance metadata. Select the SOFR observation, not the loan's spread-
inclusive rate. This is an explicit reuse of dated source observations, never an
automatic coupling of financing, cash interest and performance benchmarks. Other
reference rates are accepted under the same declared contract.

Sharpe is `sqrt(A) * mean(r - rf) / sample_std(selected_series)` with
`sharpe_denominator="portfolio_returns"` (default) or `"excess_returns"`.
Undefined cases retain null values and `insufficient_samples`, `zero_volatility`
or `zero_excess_volatility`. Annualization does not adjust serial correlation.
Sortino continues to use the separately specified constant MAR; dated RF does not
silently replace it. Beta/correlation continue to use raw aligned simple returns,
not risk-free-subtracted returns.

## Rolling benchmark estimates and cumulative plots

```python
rolling = rt.rolling_risk(
    result, window=window, periods_per_year=annualization,
    risk_free_returns=rf, sharpe_denominator="excess_returns",
    benchmark=benchmark_returns, benchmark_metadata=benchmark_metadata,
)
fig, ax = rt.plots.rolling_risk(rolling, metric="beta")
fig, ax = rt.plots.cumulative_returns(report, method="sum")
fig, ax = rt.plots.cumulative_returns(report, method="compound")
fig, ax = rt.plots.cumulative_returns(report, method="wealth", benchmark=False)
```

Rolling results add `beta`, `correlation`, `beta_status`, `correlation_status`.
A full explicit window of at least two matching intervals is required; earlier
rows stay null with `insufficient_samples`. A constant benchmark gives
`zero_benchmark_variance` for beta; correlation follows existing zero-volatility
statuses. Without a benchmark these fields are null with `no_benchmark`.
Windows include the current close and are descriptive, not same-close signals.

`PerformanceResult.cumulative` has `session`, `series` (`portfolio`/`benchmark`),
`cumulative_simple_return`, `compounded_return`, `wealth`, starting at 0, 0, 1.
The plot explicitly selects summed simple returns, compounded return, or wealth
multiple; these are never labeled interchangeably. Benchmark inclusion defaults
to true when present. It consumes prepared tables without recomputation and
returns `(Figure, Axes)`, accepting an existing `ax`. No show/save/simulation
occurs. Plot titles preserve actual dates and any stopped-run reason.

## Numerical scenario comparisons

```python
comparison = rt.compare_performance(
    {"Unlevered": report_one, "Levered": report_two}, benchmark_label="Benchmark",
)
comparison.values  # Long-form numerical Polars table; formatting belongs outside.
comparison.diagnostics
```

`performance` now retains `benchmark_summary` using the same RF/MAR, volatility,
Sharpe, Sortino and drawdown definitions, with equity normalized to initial
capital and no separately modeled benchmark trading or financing. Benchmark cost
rows are zero with `not_modeled` status. `summary` also includes `entry_cost`,
`financing_cost` (borrowing expenses), and `ending_leverage`.

Comparison rows contain `scenario`, `kind` (portfolio/benchmark), `metric`,
nullable `value`, `unit`, `n_obs`, `metric_status`, `run_status`, `stop_reason`,
`period_start`, `actual_end_session`, `requested_end_session` (Date columns).
Each report's metadata remains in `scenario_metadata`. Identical benchmark
summaries are included once; benchmark labels must not collide with portfolios.
No rounding, percent strings, rankings or written conclusions are produced.

Default `coverage="strict", assumptions="strict"` rejects unequal holding
intervals/requested coverage and incompatible currency, capital normalization,
annualization, return basis, RF series, benchmark or metric conventions. Cost,
leverage and financing differences are scenario inputs, retained in metadata.
Explicit `coverage="separate"` and/or `assumptions="separate"` retain each
scenario's own sample and conventions, with difference diagnostics; nothing is
intersected, resampled or recomputed. Different dated RF/benchmark rows also
require separate assumptions handling even if they result from unequal coverage.
When separate handling is selected, differing benchmark summaries are qualified
by scenario. A stopped report additionally requires `allow_partial=True` in both
`performance` and the comparison. Partial observations never masquerade as the
full requested horizon.

## Offline saved Yahoo chart adapter

```python
converted = rt.adapters.yahoo_chart(
    saved_responses, sessions=sessions, splits=splits, dividends=dividends,
    metadata=market_metadata, volume_basis="unknown",
    availability="session_close_reconstructed",
)
market = converted.market
```

`saved_responses` maps caller asset identifiers to decoded, saved Yahoo chart JSON
objects (`chart.result[0]`, daily timestamps, `indicators.quote`, optional
`indicators.adjclose`, and `events`). No files are downloaded or opened implicitly.
Only one successful daily result per asset, one currency/timezone and the exact
supplied calendar are supported. Extra/missing/duplicate sessions, null selected
prices, unsupported events, malformed arrays or mismatched provider metadata raise.
A hash of each supplied response and its provider symbol are recorded in source
metadata. This small adapter uses synthetic format fixtures; the reference
notebook's folder contained derived tables/provenance, not original chart payloads,
so it was not validated against those original private responses.

Yahoo labels Close as split-adjusted and Adjusted Close as adjusted for splits and
distributions. See [Yahoo historical prices](https://finance.yahoo.com/quote/AAPL/history/)
and [Yahoo adjustment documentation](https://help.yahoo.com/kb/SLN28256.html).
`market_metadata.price_basis` therefore explicitly selects:

- `split_adjusted`: use provider Close, keeping that label; not executable raw history.
- `total_return_adjusted`: require provider Adj Close; no separate dividend addition.
- `raw`: multiply Close by supplied `raw_adjustment_factors`, exactly
  `session: Date`, `asset: String`, `raw_price_factor: Float64`. Factors must be
  positive and cover the entire panel. Their ratios between sessions must agree
  with supplied splits. Factors include **all later splits through retrieval**,
  even splits outside the selected sample; lack of in-window splits is not proof
  that factors are one. Required `factor_metadata` keys are nonblank `source`,
  `basis="cumulative_splits_after_session_through_retrieval"` and
  `verified_through` equal to the response `retrieved_at`. This is a caller assertion
  supported by their independent split history; the adapter cannot verify omitted
  later actions from a truncated response. Factors are rejected for other bases.

The authoritative sessions, splits and dividends use existing market schemas,
including actual dividend **payment dates** and raw post-split-share amounts.
Provider event dates must be represented in supplied tables; split ratios are
cross-checked. Yahoo dividend amounts do not replace authoritative amounts because
adjustment units may differ. Missing provider event blocks do not prove absence of
actions; the caller still asserts `actions_complete=True`. Pay dates, holidays and
unusual distributions are never invented. Unsupported action types raise.

`volume_basis` must be declared as `unknown`, `raw_shares`, or
`split_adjusted_shares` from verified provider semantics. Unknown volume produces
no dollar-volume estimate. For raw reconstruction, declared split-adjusted share
volume is divided by the raw price factor; declared raw volume is retained.
`dollar_volume` is produced only from raw prices and identified raw share volume.
Never substitute total-return-adjusted prices into executable dollar ADV.

`ProviderDataResult` contains validated `market`, `bars`, metadata and diagnostics.
Bars retain session/asset, provider timestamp/Close/Adj Close/volume, optional raw
factor/share volume/dollar volume, and `available_at`. Availability is **explicitly
reconstructed from the supplied session close**, not an observed publication time.
It is unsuitable as proof of a historical data vintage or intraday availability.
Missing unknown volume and unrequested Adj Close may remain null in these audit
bars; selected market prices cannot. No Yahoo client, credentials, broad provider
framework or additional runtime dependency was added.
