# Implemented API: milestones 1A–1D and 2

Use `import research_toolkit as rt`. Core functions are `prepare_market_data`,
`returns`, `cumulative_returns`, `equal_weights`, `buy_and_hold`, `performance`,
`correlation`, `save_snapshot`, `load_snapshot`, `inverse_volatility_weights`,
`scheduled_rebalance`, `risk_contributions`, and `rolling_risk`, with optional
`rt.plots` views.
Result/configuration objects are concrete dataclasses; their tables are Polars
DataFrames. See the [acceptance notebook](../examples/buy_and_hold_equities.ipynb)
for a complete offline workflow and the smaller [ledger example](../examples/unlevered_buy_and_hold.py).

Short positions, signal-driven trading, richer cost models, provider adapters, and
chronological model research remain unimplemented. Future
extensions in [architecture](architecture.md) are labeled separately.

## Validate market data

```python
market = rt.prepare_market_data(
    prices=prices, sessions=sessions, splits=splits, dividends=dividends,
    metadata=metadata, missing="raise",
)
```

All inputs must be eager Polars tables with exactly the following columns/types.
Create empty action tables with these schemas; an untyped empty DataFrame fails.
Numeric columns must be Float64, dates must be Date, and close times must be
Datetime with microsecond precision and UTC timezone. No automatic type conversion,
deduplication, missing-price filling, or row dropping occurs.

| Table | Schema | Key |
| --- | --- | --- |
| `prices` | `session: Date`, `asset: String`, `close: Float64` | `(session, asset)` |
| `sessions` | `session: Date`, `close_at: Datetime("us", "UTC")` | `session` |
| `splits` | `action_id: String`, `asset: String`, `effective_session: Date`, `ratio: Float64` | `action_id` |
| `dividends` | `action_id: String`, `asset: String`, `ex_session: Date`, `pay_date: Date`, `cash_per_share: Float64` | `action_id` |

Prices must be positive and finite. All price assets require a row on **every**
supplied session, including assets later assigned zero portfolio weight. Asset
currencies must be declared for exactly that universe. Session close times must
increase, and their local dates must agree with the session labels. The library
checks against the supplied calendar; it does not independently verify exchange
holidays or detect sessions omitted from that calendar. Use an authoritative
calendar when preparing real inputs.

Ratios are new/old shares. Cash dividend amounts are per post-split share; a split
precedes a dividend when both are effective on one session. Action identifiers
must be globally unique; multiple splits on one asset/session are rejected as
ambiguous. Pay dates cannot precede ex-dates and may fall outside the price window
or on nontrading dates. All effective/ex-sessions must lie inside the supplied
calendar. Unsupported actions cannot be encoded as extra columns and ignored.

Required metadata is a JSON-compatible mapping (ISO dates/timestamps as strings):

| Key | Contract |
| --- | --- |
| `source` | Nonblank source/snapshot description |
| `retrieved_at` | ISO timestamp with explicit UTC offset or `Z` |
| `currency`, `asset_currencies` | Base currency and exact `{asset: same_currency}` mapping |
| `calendar`, `calendar_version`, `timezone` | Nonblank calendar provenance/version and valid IANA zone |
| `price_basis` | `raw`, `split_adjusted`, or `total_return_adjusted` |
| `frequency` | Exactly `1d` |
| `coverage_start`, `coverage_end` | ISO dates matching first and last supplied sessions |
| `actions_complete` | Explicit `True`, including when both action tables are empty |
| `dividend_basis` | Exactly `post_split_share` |

Add source identifiers, transformation descriptions, file hashes, redistribution
restrictions, and source code revision as additional JSON metadata when available.
`actions_complete=True` is the caller's assertion that supported actions and their
payment dates are complete; validation cannot discover omitted source events.

The returned `MarketData` has owned table copies, copied metadata, a `diagnostics`
table (`code`, `table`, `count`), and `snapshot_id`. Rows are canonically sorted and
sorting is reported. The SHA-256 snapshot identity covers normalized table schemas,
values, and metadata; it is invariant to input row order. Source metadata changes
change the identity. Snapshot persistence is implemented below.

Polars tables and dictionaries remain mutable even inside frozen dataclasses.
Consumers revalidate `MarketData` and reject changes that no longer match its
identity. Call `prepare_market_data` again to intentionally change a snapshot.

## Return arithmetic

```python
price_returns = rt.returns(market, method="simple", basis="price")
summed = rt.cumulative_returns(price_returns, method="sum")
compounded = rt.cumulative_returns(price_returns, method="compound")
wealth = rt.cumulative_returns(price_returns, method="wealth")
```

`returns` accepts validated `MarketData`, rather than a bare price table, so it
can check the calendar and adjustment basis. `method` is `simple` or `log`.
`basis="price"` accepts raw/split-adjusted inputs; `basis="total_return"` accepts
a declared total-return-adjusted series or raw inputs with the explicit
`dividend_policy="reinvest_ex_close"` analytical convention described in M2 below.
Raw `basis="price"` retains split jumps and produces a
`raw_price_split_discontinuity` diagnostic when splits are present. Cash-held
portfolio returns still come from the ledger, not the analytical reinvested series.

`ReturnResult.values` has `session`, `asset`, `period_start`, and `simple_return`
or `log_return`. The first return and period start for each asset remain null.
One observation is valid and produces only this structural null; empty market
panels fail validation. No annualization is inferred or performed.

`cumulative_returns` preserves the original return columns and adds:

| Method | Simple input | Log input |
| --- | --- | --- |
| `sum` | `cumulative_simple_return` | `cumulative_log_return` |
| `compound` | `compounded_return = product(1+r)-1` | `compounded_return = exp(sum(log_r))-1` |
| `wealth` | `wealth_multiple = product(1+r)` | `wealth_multiple = exp(sum(log_r))` |

Leading nulls remain null. Result metadata stores the method, basis, frequency,
units, source identity, sessions, and assets. Cumulative transforms reject interior
nulls, missing/duplicate keys, broken interval links, invalid return kinds, and
nonfinite or unrepresentable wealth. Standalone simple returns must exceed −1;
default/bankruptcy handling is not implemented by these positive-price utilities.

## Allocation and simulation

```python
weights = rt.equal_weights(["A", "B"])  # Or {"A": 0.4, "B": 0.6}.
result = rt.buy_and_hold(
    market, weights=weights, initial_capital=1_000.0,
    entry_session=entry_date, end_session=end_date,  # datetime.date values
    policy=rt.BuyHoldPolicy(
        execution="entry_close", sizing="post_cost_equity",
        fractional_shares=True, initial_gross_leverage=1.0,
        terminal_action="mark_only",
    ),
    costs=rt.TradeCosts(commission_bps=5.0, half_spread_bps=2.0, impact_bps=0.0),
    cash_rate=0.0, cash_day_count="ACT/365F",
)
```

Policy fields and cost/rate inputs are required, including explicit zero rates.
Initial exposure must be nonnegative: zero is all cash, one is fully invested
after costs, intermediate values retain cash, and values above one borrow through
an explicit `Financing` configuration. The `cash_rate`/`cash_day_count` pair shown
above remains supported for unlevered calls. Omitting all rate configuration fails;
there is no inferred cash or loan rate.

## Financing and leverage

Keep the same weights, data, and entry/exit sessions, set the desired initial
exposure in `BuyHoldPolicy`, and replace the cash-rate pair with `financing`:

```python
result = rt.buy_and_hold(
    market, weights=weights, initial_capital=1_000.0,
    entry_session=entry_date, end_session=end_date,
    policy=rt.BuyHoldPolicy(
        execution="entry_close", sizing="post_cost_equity", fractional_shares=True,
        initial_gross_leverage=2.0, terminal_action="mark_only",
    ),
    costs=rt.TradeCosts(commission_bps=5.0, half_spread_bps=2.0, impact_bps=0.0),
    financing=rt.Financing(
        cash_rate=0.02, borrowing_rate=0.08, day_count="ACT/365F",
        maintenance_equity_ratio=0.4, on_breach="stop", cash_sweep="repay_debt",
    ),
)
result.require_complete()  # Raises if the run stopped, including on the final date.
result.daily.select("session", "equity", "debt", "cash", "gross_leverage", "equity_ratio")
```

These example rates and threshold are modeling assumptions, not library defaults
or broker rules. Every `Financing` field is required. Cash/borrowing rates must be
finite nonnegative annual nominal fractions; only `ACT/365F` is supported.
`maintenance_equity_ratio` must lie in `(0, 1]`; it may be explicit `None` only
when initial exposure does not exceed 1. `on_breach="stop"` and
`cash_sweep="repay_debt"` are the supported policies. Supplying `financing` together
with either non-None legacy cash-rate argument raises rather than choosing one.

Gross initial notional `N=L*C/(1+L*k)` targets exposure relative to post-cost equity.
Here C is starting equity, L initial leverage, and k the weighted proportional
entry cost. The actual order basket's funding need determines the loan; a borrowing
event credits cash and debt **before** entry trades and costs. Borrowing is not P&L.
Initial funding must leave positive equity and satisfy the maintenance threshold;
otherwise the call raises instead of returning an invalid entry.

Once per calendar date after entry, credit opening cash at `cash_rate/365` and
capitalize opening debt at `borrowing_rate/365`. Weekend/holiday dates are included
without fabricating market-return rows. Borrowing-interest cost rows have null
trade/asset IDs because financing is not a trade fee. Debt can increase from posted
interest; there are no implicit purchases or daily leverage resets. The optional
dividend policy below adds explicitly funded purchases.

After that date's interest and corporate-action payments, available cash repays
debt. Receivables cannot fund a repayment before payment. Repayment reduces cash
and debt equally and is not a second expense. Excess cash remains available and
earns interest from the following date. Without opt-in dividend reinvestment,
buy-and-hold quantities change only through splits.

At each session close, compare `equity_ratio = equity/gross_exposure` with the
configured threshold. Equality is compliant; a relative 1e-12 tolerance handles
roundoff at the boundary. All-cash equity ratio is null and no margin test applies.
The ratio uses receivables in equity and risky holdings in gross exposure, exactly
as the balance sheet defines them. This is daily research margin monitoring,
not intraday or broker-specific margin enforcement.

If maintenance fails or equity is nonpositive, the run retains that close and
stops. `status="stopped"`, `stop_reason` (`maintenance_margin_breach` or
`nonpositive_equity`), `stop_session`, and `stop_time` identify the failure.
Insolvency takes precedence if both conditions occur. Metadata retains requested
`end_session` and adds `actual_end_session`; a failure at the requested end is still
a stopped run. No later charge, payment, mark, or automatic liquidation occurs.

The final P&L and simple return can represent losses at or beyond −100%; they are
not clipped. For nonpositive equity, log return, gross leverage, and position
equity weights are null. The signed compounded return still reconciles to ending
equity/initial capital minus one. Completed results have null stop fields.
`require_complete()` returns the result when complete and raises `ValueError`
otherwise. Call it before full-period comparisons. Partial analysis must explicitly
show actual coverage and the stop reason. Full performance reports are implemented below.

## Shared allocation and accounting rules

Weights are risky-asset proportions, not weights of net equity. Accept either an
asset mapping or a Polars table (`asset: String`, `weight: Float64`). They may
select a subset of the market universe; unknown assets, negative weights, duplicates,
and nonfinite values fail. Weights must sum to 1 within absolute tolerance 1e-12.
Only representational roundoff within that tolerance is normalized; both supplied
and resolved values are saved in metadata. There is no economic renormalization
of arbitrary sums. Zero weights create no trades. Even an all-cash run takes a
valid allocation template. `equal_weights` returns a sorted Polars weight table.

Cost components are nonnegative scalar bps or mappings covering exactly the
allocation assets (including explicit zero-weight assets). They are recorded as
**modeled** proportional per-side cash expenses, not measured liquidity data.
Half-spread is already the one-sided cost. No adverse fill-price adjustment is
added on top. Fixed/minimum fees and nonlinear impact are not supported.

The implementation follows [financial conventions](financial-conventions.md):

- Predetermined dollar allocations execute at the chosen entry close; post-cost
  sizing funds entry expenses and any explicitly configured loan. The caller is responsible for
  choosing weights before entry; the library cannot verify how a static mapping
  was researched.
- Without the optional dividend policy, share quantities stay fixed except for
  splits. Dividends accrue on ex-date and transfer to cash on pay date. Entry on an
  ex-date does not receive that dividend. Unpaid receivables remain in equity.
- Cash interest accrues once per calendar date after entry, including weekends,
  from the previous date's cash balance at nominal `cash_rate/365`. It is credited
  before that date's dividend payments. There is no interest before cash receipt.
- The first daily interval opens at original capital **before fees**. Its P&L
  includes entry costs; later intervals open at the previous closing equity.
  Pre/post-entry valuations are recorded separately. No terminal sale is assumed.
- The ledger checks shares against event deltas, cash/debt/receivable balances against
  events, P&L against attribution, cumulative P&L against equity change, and
  compounded returns against equity. Material residuals raise `ArithmeticError`.

## Backtest outputs and limits

`BacktestResult` exposes the following Polars tables:

| Table | Main fields / meaning |
| --- | --- |
| `daily` | Session/period start, opening and ending equity, dollar P&L, simple/log returns, cumulative P&L and simple-return sum, compounded return, cash, debt, receivables, exposure, leverage, equity ratio, margin-breach flag, closing drawdown, entry-cost flag |
| `positions` | Session/asset, quantity, raw mark, value, net-equity weight; includes entry session |
| `trades` | Trade ID, session/UTC close, asset, signed quantity/notional, reference price, execution policy, cost |
| `costs` | Cost ID, date/time, nullable trade ID/asset, component, amount, modeled basis; nonzero trade costs and calendar-date borrowing interest |
| `events` | Stable event ID/sequence, calendar date, nullable UTC time, phase, type, linked IDs, quantity/cash/debt/receivable deltas |
| `valuations` | Pre-entry, post-entry, applicable pre-trade, and closing balance sheets |
| `dividend_reinvestments` | Per-paid-action funding disposition and linked reinvestment trade; typed empty if disabled |
| `attribution` | Session, component, asset (null for account interest), dollar P&L; sums to daily P&L |
| `receivables` | Session/action, ex/pay dates, entitled shares, original amount, outstanding amount; includes paid entitlements with zero outstanding |
| `diagnostics` | Session, reconciliation code, residual, currency tolerance |

Date-only interest/action events have null `time` and `phase="before_close"`;
they are modeled daily events, not invented observed timestamps. Sequence follows
cash/borrowing interest → splits → ex-date accrual → pay-date cash → debt repayment.
Entry funding/trades and stop events have their supplied closing timestamp;
stop events have `phase="close"` and zero balance deltas. No market-return rows
are created on weekends.

Currency tolerances are `min(0.009, 1e-8 + 1e-12 * scale)`; quantity checks use
1e-12 absolute/relative tolerance. Tiny entry cash roundoff is explicitly reported,
and fully invested cash is set to zero only after passing that check. All other
balances remain auditable; no material cash deficits are hidden.

Metadata records source/snapshot identity, resolved weights/policy/cost rates,
cash/borrowing rates, day count, financing configuration, stop status/coverage,
dividend/return conventions, currency, dates, package/Python/
Polars versions, immediate settlement and excluded taxes. No Git command or
filesystem/network access occurs during simulation. Record a commit/dirty state
externally alongside saved outputs if needed for development-run reproduction.

Daily drawdown uses a peak starting at original capital. The post-entry loss is
visible in `valuations` even if it recovers before the first interval close;
`performance()` includes all these valuations when reporting maximum drawdown. Numerical overflow,
invalid input/funding, or failed reconciliations raise; they are not disguised as
ordinary margin stops. Stopped runs preserve real finite negative/zero equity.

Required runtime dependency: Polars only. Simulation invokes no plotting, provider,
ML, annualization, or benchmark code. Reports and optional plots are separate calls.
All tests and examples run offline.

## Performance reports — milestone 1D

```python
report = rt.performance(
    result,
    periods_per_year=252,  # Required assumption; never inferred from data/calendar.
    risk_free_annual_effective=0.03,
    minimum_acceptable_return_annual_effective=0.0,
    benchmark=benchmark_returns,  # Optional Polars table, schema below.
    benchmark_metadata={
        "source": "saved benchmark source", "basis": "total return; reinvested; no costs",
        "currency": "USD", "frequency": "1d", "snapshot_id": benchmark_snapshot_id,
    },
    alignment="strict", allow_partial=False,
)
```

`result` is a `BacktestResult`. Its daily arithmetic and valuation coverage are
validated before reporting; inputs are not mutated. `periods_per_year` must be
positive and finite. Effective annual risk-free and MAR rates must be finite and
strictly greater than −1. Each converts via `expm1(log1p(annual)/periods_per_year)`.
These metric rates are separate from nominal calendar-day financing rates.

Benchmark schema is **exactly** `period_start: Date`, `session: Date`,
`simple_return: Float64`. Rows may arrive out of order; keys are sorted and both
interval endpoints must match the portfolio exactly. Nulls, duplicates, nonfinite
values, missing/extra intervals, returns ≤ −1, currency/frequency mismatches, and
unsupported alignment modes raise. There is no silent intersection. A supplied
benchmark requires nonblank `source`, `basis`, `currency`, and `frequency` strings
in `benchmark_metadata`; additional provenance is preserved. Benchmark equity starts
at portfolio initial capital and compounds benchmark returns without inserting
portfolio costs. With no benchmark, comparison/benchmark-series tables are typed empty.

A complete-period report calls `result.require_complete()`. For a stopped run,
`allow_partial=True` is mandatory; metadata retains actual/requested dates, stop
reason and status. An optional partial benchmark must be explicitly sliced by the
caller to those exact intervals. Plots display partial status and stop reason.
No metric annualizes a stopped result into a fabricated full-period total return.
The retained last simple return/drawdown may be below −100%; it is never clipped.

### `PerformanceResult`

| Table | Columns and meaning |
| --- | --- |
| `summary` | `metric: String`, `value: Float64` (nullable), `unit: String`, `n_obs: Int64`, `status: String` |
| `benchmark_comparison` | Same schema; strict common-sample comparisons |
| `benchmark_series` | `session: Date`, `equity: Float64`; entry capital plus each compounded benchmark close |
| `daily` | Owned copy of the validated daily ledger, including P&L, simple returns and distinct cumulative columns |
| `equity` | `session: Date`, `phase: String`, `equity: Float64`; ordered pre-entry, post-entry, then closes |
| `drawdowns` | `equity` columns plus `drawdown: Float64`; running peak includes both entry valuations |
| `allocation` | `session: Date`, `component: String`, `weight: Float64`; asset components prefixed `asset:`, plus `account:cash`, `account:debt` (negative) and `account:dividend_receivable` |
| `attribution` | `component: String`, `pnl: Float64`; summed ledger dollar contributions over actual coverage |

Position/allocation weights use net equity, not gross risky value, and are null at
nonpositive equity. Pre-entry is excluded from allocation; post-entry account
balances and held positions are included. Positive-equity weights sum to one even
for leveraged portfolios. Dollar attribution is realized ledger attribution, not
covariance risk contributions. Result tables/dictionaries remain mutable: treat
reports as immutable inputs to plots, and create a new report after changing a run.

Summary metrics: `ending_equity`, `cumulative_pnl` (declared currency),
`cumulative_simple_return`, `compounded_return`, `max_drawdown` (fractions),
`annualized_arithmetic_mean` (fraction/year), `annualized_volatility`
(fraction/sqrt(year)), `sharpe`, `sortino` (ratios). `n_obs` always counts daily
holding intervals; drawdown additionally examines the two entry valuations.
Volatility uses sample standard deviation (`ddof=1`). Sharpe uses periodic excess
returns; Sortino uses the root-mean-square negative MAR shortfall over **all**
observations, not only losing observations. Ratios use square-root annualization.
This is a sampling assumption, not a correction for serial correlation. CAGR and
drawdown durations are not implemented.

Benchmark metrics: `return_correlation` (Pearson), `beta` (intercept regression
slope of portfolio on benchmark), `benchmark_compounded_return` (fraction),
`compounded_return_difference` (**percentage points**, multiplied by 100), and
`relative_wealth_return` (fraction). No implicit risk-free subtraction for beta.

Volatility and ratios require at least two observations. Undefined results have
null values with `insufficient_samples`, `zero_volatility`,
`zero_downside_risk`, or `zero_benchmark_variance`; finite zero volatility itself
is a valid reported zero. Invalid data raises. An empty backtest is invalid because
simulation requires a holding interval. Neither status codes nor calculations
produce narrative investment conclusions.

Metadata includes original run assumptions/identities and status, annualization,
effective annual/converted periodic hurdles, `ddof`, Sortino denominator, benchmark
provenance, actual sample size, alignment, partial-report opt-in, and drawdown basis.

## Asset correlations

```python
price_returns = rt.returns(market, method="simple", basis="price")
correlations = rt.correlation(price_returns)
```

`correlation(ReturnResult) -> CorrelationResult` validates a complete common daily
panel using the same interval contract as cumulative returns. It requires simple
returns and excludes only the structural leading null per asset. A single input
session produces null correlations with `empty_sample`; one interval produces
`insufficient_samples`; a constant member produces `zero_volatility`.

`values` has `asset: String`, `other_asset: String`, `correlation: Float64` (nullable),
`n_obs: Int64`, `status: String`, with all ordered pairs. Metadata and diagnostics
preserve return basis/source, assets, sessions and snapshot identity. Raw price
returns retain split discontinuities and are **not** total returns. The acceptance
notebook constructs explicit single-asset ledger return panels instead, including
splits and dividends held in cash; it labels their distinct basis.

## Local snapshot persistence

```python
rt.save_snapshot(market, "data/local/study-v1")  # Must be a NEW directory.
market = rt.load_snapshot("data/local/study-v1")
```

`save_snapshot(MarketData, path) -> pathlib.Path` revalidates input identity and
writes `prices.parquet`, `sessions.parquet`, `splits.parquet`, `dividends.parquet`,
and `manifest.json`. Polars handles Parquet directly; no PyArrow dependency.
The manifest records `format_version=1`, canonical `snapshot_id`, complete source
metadata, original preparation diagnostics, and each file's SHA-256, row count and
schema. Include transformations and usage/redistribution restrictions in source
metadata. No existing path is overwritten. A failed write removes only the newly
created snapshot directory; use a single writer per destination.

`load_snapshot(path) -> MarketData` checks all file hashes, schemas and row counts,
revalidates the financial contracts, and matches canonical identity. It never
fetches data. The reconstructed diagnostics describe loading the saved canonical
order; original preparation diagnostics remain in the manifest. Canonical identity
is independent of Parquet byte encoding. File hashes detect accidental corruption;
they are not signatures authenticating an external source. The canonical snapshot
identity is stored in simulation metadata. A separate benchmark snapshot carries
its own identity. Run parameters and environment/revision provenance belong in the
notebook/report, not in market-data files.

## Optional Matplotlib plots

Install `.[plot]` (or `.[notebook]`) from this checkout. `import research_toolkit as rt`
still imports no Matplotlib, NumPy, pandas, network client or ML framework.
Matplotlib loads only when a plotting function is called.

```python
fig, ax = rt.plots.equity(report)
ax.set_title("My study")
fig.savefig("artifacts/equity.png", dpi=150)
```

- `prices(market, *, ax=None)`: supplied, **unnormalized** prices, labeled adjustment
  basis and currency, with split-date markers. No hidden rebasing calculation.
- `pnl(report, *, ax=None)`, `returns(report, *, ax=None)`: daily net dollar P&L and
  net simple returns, respectively.
- `equity(report, *, ax=None)`: prepared equity with pre/post-entry valuations and
  optional precomputed benchmark on the same initial capital.
- `drawdown(report, *, ax=None)`: prepared entry-aware drawdown.
- `distribution(report, *, bins=30, ax=None)`: histogram of supplied daily returns;
  binning is display-only, with no new performance calculation.
- `allocation(report, *, ax=None)`: drifting net-equity weights, including cash,
  receivables and negative debt weights.
- `attribution(report, *, ax=None)`: prepared cumulative dollar contributions.
- `correlation(correlations, *, ax=None)`: prepared asset-return heatmap; undefined
  values remain blank. The title carries basis, coverage and sample size.

Every function returns `(Figure, Axes)`, supports an existing axis, and never calls
`show()`, changes global styles, fetches inputs or reruns the simulator/metrics.
Portfolio plots label actual coverage and stopped status/reason. Percentage axes
format stored fractions without changing data. Cash/loan attribution remains
separate from asset price P&L. Milestone 2 adds rolling risk below; interactive
backends remain later work.

## Scheduled allocation and rebalancing — milestone 2

### Historical allocation inputs

Inverse-volatility and covariance estimates consume an explicit `ReturnResult`
with simple returns. Existing price-return and supplied adjusted-index calls
remain supported. Raw equity inputs also support this **explicit analytical**
corporate-action convention:

```python
asset_returns = rt.returns(
    market, method="simple", basis="total_return",
    dividend_policy="reinvest_ex_close",
)
allocation = rt.inverse_volatility_weights(
    asset_returns, decision_session=decision_session, lookback=40,
    periods_per_year=252, max_asset_weight=0.40,
)
allocation.weights  # asset: String, weight: Float64; risky proportions sum to 1.
```

For raw total returns, the holding interval's gross multiplier is
`split_ratio * (close + ex_date_cash_dividend_per_post_split_share) / prior_close`.
It assumes the ex-date entitlement is reinvested at that close; actual payment
liquidity is **not** modeled by this analytical series. Multiple ordinary dividends
on the same ex-date are summed. Entry on an ex-date earns no prior entitlement.
The ledger still uses receivables and actual payment dates, so its cash-held returns
can differ. Supplying an adjusted index with `dividend_policy` raises to prevent
counting actions twice. Log and cumulative transformations retain the chosen policy.
Raw `basis="price"` continues to retain split jumps; choose that basis deliberately,
not as a substitute for action-aware allocation inputs.

`inverse_volatility_weights` requires an integer `lookback >= 2`, positive explicit
annualization, and a decision session in the supplied return calendar. It uses the
last `lookback` nonstructural intervals whose **ending session is strictly earlier
than the decision session**. It never uses the decision day's close or a later
observation. Insufficient history, missing inputs, non-simple returns, or zero
asset sample volatility raise. Weights are proportional to `1 / sample_volatility`;
normalization uses only those estimates. No risk-free subtraction occurs.

`max_asset_weight` is a finite scalar in `(0, 1]`, measured against gross risky
notional. Breaches **raise**, with no clipping, optimization or silent redistribution.
A relative 1e-12 tolerance handles equality roundoff. The returned `AllocationResult`
contains `weights`, `estimates` (`asset`, `annualized_volatility`, `n_obs`), and metadata
with source/basis, snapshot identity, sample interval endpoints, decision date,
lookback, annualization, `ddof=1`, concentration limit and denominator. It is not a
model-parameter vector or a portfolio equity-weight table.

### Dated targets and execution

```python
# Construct one complete basket per execution session from chosen weights.
# This is table construction, not a call that chooses dates implicitly.
targets = allocation.weights.with_columns(
    pl.lit(decision_session).alias("decision_session"),
    pl.lit(execution_session).alias("session"),
    pl.lit(1.0).alias("gross_leverage"),
).select("decision_session", "session", "asset", "weight", "gross_leverage")

result = rt.scheduled_rebalance(
    market, targets=targets, initial_capital=starting_equity,
    entry_session=execution_session, end_session=final_session,
    policy=rt.RebalancePolicy(
        execution="scheduled_close", sizing="post_cost_equity",
        fractional_shares=True, terminal_action="mark_only", non_session="raise",
        receivable_policy="reserve", max_asset_weight=0.40,
    ),
    costs=costs, financing=financing,
)
```

Targets have the exact schema `decision_session: Date`, `session: Date`,
`asset: String`, `weight: Float64`, `gross_leverage: Float64`. The first basket
executes on `entry_session`. Later execution dates lie strictly before `end_session`
so the terminal session remains mark-only. Every decision/execution date must be
in the supplied calendar, with `decision_session < session`; missing/non-session
dates raise. The engine does not infer monthly dates, shift holidays, or carry a
missing target forward into a new trade. The example explicitly materializes the
first supplied session of each month; other schedules can supply their own dates.

Every basket must cover the same asset universe, use explicit zeros for exits,
have nonnegative weights summing to one (1e-12 tolerance, with only that roundoff
normalized by the recorded basket-sum convention), and declare one leverage and decision
session. A later entrant must already have an explicit zero entry weight and full
price history in the market snapshot. Leverage is finite/nonnegative and can change
at scheduled targets. All-cash targets use leverage zero and retain a valid risky
allocation template. Concentration limits apply to **requested risky proportions**,
not subsequent weight drift; equity weights may exceed those limits when leveraged.
User-supplied targets assert their own information provenance; dated columns cannot
prove arbitrary weights were chosen without future information. Use the trailing
allocation helper and retain its metadata for generated targets.

`Financing` is required, including explicit zero rates. Every leveraged target
needs a valid maintenance threshold and must be initially compatible with it.
The shared ledger preserves corporate-action order, financing on calendar dates,
receivable entitlement from pre-ex holdings, actual dividend payments, debt sweeps,
and complete trade/cost/P&L reconciliation. Holdings before the close earn that
interval's price move; the scheduled close changes future holdings and books its
own transaction costs in the current interval. The first interval still includes
entry costs relative to original capital. Buy-and-hold behavior is preserved.

### Funding, limits and turnover

At a scheduled mark, let `E` be pre-trade equity, `R` unpaid receivables, `v_i` the
current marked holdings, `w_i` risky proportions, `L` target leverage, and `k_i`
the sum of proportional commission, half-spread and impact rates. Solve
`E_after + sum(k_i * abs(w_i * N(E_after) - v_i)) = E`.

- `receivable_policy="reserve"`: for `L <= 1`,
  `N(x) = max(0, min(L*x, x-R))`. This explicitly reserves unavailable receivables
  instead of funding purchases with an unstated loan. Actual gross leverage can
  be below the request; the shortfall is recorded. For `L > 1`, `N(x)=L*x` and
  borrowing is explicit under the supplied financing policy.
- `receivable_policy="require_target"`: `N(x)=L*x`. A basket with `L <= 1` that
  would leave debt raises; the caller can choose reserve, leave cash, or explicitly
  request leverage. Leveraged targets still use supplied financing.

Existing debt can remain when a reserve-policy liquidation leaves only unpaid
receivables securing the loan; reserve never increases debt for an unlevered target.
Later payments repay that debt. Receivables remain part of net equity and margin
accounting even though they are not spendable cash.

The proportional-cost solver requires each asset's total rate `< 1` and
`L * sum(w_i*k_i) < 1`; unsupported extreme rates or insolvent cost funding raise.
Bisection solves the monotone post-cost equation; resulting cash, quantities,
post-trade equity and attribution are independently reconciled. Required borrowing
is funded explicitly, sales execute before buys, and remaining cash repays excess
debt. Costs apply to **absolute changed notional**, not the whole target portfolio.
Relative notional differences no larger than 1e-13 are treated as representational
roundoff; this is recorded, not a user-facing minimum-trade threshold. No quantities
changed means no trades and no fees. Fixed/per-share/minimum fees and nonlinear
capacity models remain unsupported and need their own sizing tests before addition.

Before a scheduled trade, nonpositive equity or a maintenance breach stops the run
**before any rebalance can conceal it**. Otherwise the completed close is checked
again after costs. Failure status and all actual records are preserved. Targets
remain the requested plan, including any future unexecuted rows in a stopped run.
No forced liquidation, automatic breach-triggered rebalance or position clipping
is added. Full-period reports still require complete status.

### Added result records

Every `BacktestResult` now also contains typed tables:

| Table | Fields |
| --- | --- |
| `targets` | The validated, sorted requested target schema above; empty for buy-and-hold |
| `rebalances` | `session`, `decision_session`, `equity_before`, `equity_after`, `target_gross_leverage`, `actual_gross_leverage`, `trade_cost`, `gross_traded_notional`, `receivable_reserved` |
| `turnover` | `session`, `phase` (`entry`/`rebalance`/`dividend_reinvestment`), `gross_traded_notional`, `equity_before`, `turnover` |

Dates are `Date`, labels `String`, numerics `Float64`. `receivable_reserved` is the
**dollar reduction in requested risky notional** caused by reserve sizing, not an
additional expense or a second receivable balance. Entry is included only in
`turnover`; `rebalances` records subsequent processed baskets, including zero-trade
ones. Turnover is gross buys **plus** gross sells divided by equity just before the
basket; it is not half-turnover and excludes financing/split events. No-trade
holding sessions are omitted; scheduled zero-trade baskets explicitly report zero.

Scheduled sessions add a `pre_rebalance` valuation before the usual `close`.
Performance drawdown/equity includes these marks, so a price peak just before a
fee is visible. Closing positions/allocation exclude those intermediate marks.
`execution="scheduled_close"` identifies later fills; their events use
`phase="rebalance_close"`. Entry keeps its existing records and semantics.
Metadata records the full policy, concentration denominator, decision/execution
separation, turnover denominator, margin-monitoring order and roundoff threshold.

### Estimated risk and realized rolling risk

```python
risk = rt.risk_contributions(
    asset_returns, weights=allocation.weights, gross_leverage=1.0,
    decision_session=decision_session, lookback=40, periods_per_year=252,
)
rolling = rt.rolling_risk(
    result, window=40, periods_per_year=252, risk_free_annual_effective=0.03,
    allow_partial=False,
)
```

`risk_contributions` uses the same strictly pre-decision common sample as allocation.
Weights must cover all return assets, including zeros. Risky proportions are scaled
by explicit leverage into net-equity exposures `u`; cash/financing are assumed
deterministic for this estimate. Annualized covariance is sample covariance times
`periods_per_year`, using `ddof=1`. Portfolio volatility is `sqrt(u' Sigma u)`;
marginal contribution is `(Sigma u)_i / volatility`; component contribution is
`u_i * marginal`. Components reconcile to estimated volatility and may be negative
when an asset offsets others. This is not realized dollar attribution or a complete
model of stochastic financing/margin risk.

`RiskResult.values` has `asset`, `equity_weight`, `marginal_volatility`,
`volatility_contribution`, `fraction_of_total`, `n_obs`, `status`.
`covariance` has `asset`, `other_asset`, `annualized_covariance`. Metadata records
window/basis/source, explicit leverage, annualization and `portfolio_volatility`.
Zero portfolio volatility returns null contributions with
`zero_portfolio_volatility`; insufficient history raises. Contributions use
fraction/sqrt(year), covariance fraction-squared/year; fractions of total are ratios.

`rolling_risk` uses realized net portfolio returns **through each closing session**
for reporting, not for a same-close trading decision. `window >= 2` is an integer;
a full window is required. Early rows stay null with `insufficient_samples` and
the actual count. `RollingRiskResult.values` has `session`, `period_start` (the
window's first interval start), `n_obs`, `annualized_volatility`, `sharpe`,
`volatility_status`, `sharpe_status`. Flat windows have zero volatility and null
Sharpe with `zero_volatility`. Effective annual risk-free conversion, sample std,
and annualization match `performance`. Metadata retains status/actual coverage;
stopped runs require `allow_partial=True`.

### Additional prepared-data plots

`rt.plots.rolling_risk(rolling, metric="annualized_volatility", ax=None)` also
accepts `metric="sharpe"`. `rt.plots.risk_contributions(risk, ax=None)` labels
estimated volatility contributions separately from `plots.attribution` dollars.
`rt.plots.turnover(result, allow_partial=False, ax=None)` separates entry from later
baskets; stopped results need explicit opt-in. `rt.plots.exposures(report, ax=None)`
shows risky exposure, cash, debt and receivables. Existing `plots.allocation(report)`
shows scheduled changes and intervening drift. All return `(Figure, Axes)`, preserve
null gaps, and consume prepared numerical tables without rerunning allocation or
simulation. See [the runnable example](../examples/scheduled_rebalancing.py).

## Automatic dividend reinvestment

Both `buy_and_hold` and `scheduled_rebalance` accept the additional keyword
`dividend_reinvestment: DividendReinvestment | None = None`. Omit it to preserve
existing cash/debt behavior. All policy fields are required:

```python
reinvestment = rt.DividendReinvestment(
    execution="first_close_on_or_after_payment",
    funding="before_debt_repayment",  # Or "after_debt_repayment".
    scheduled_collision="rebalance_only",
    terminal_action="hold_cash",
)
# Supply dividend_reinvestment=reinvestment in either simulator call.
```

- **Timing and asset:** a standing instruction buys the paying asset at the first
  supplied session close on/after actual payment, assuming cash is available before
  that close. Weekend payments wait; unpaid receivables are never spent. Raw close
  and fractional shares are used. The policy applies to all entitled assets and
  can reopen a payer sold by an earlier scheduled basket; there is no per-asset
  enrollment/cancellation model yet.
- **Funding:** `before_debt_repayment` reserves paid dividend principal before the
  sweep; existing debt continues accruing interest. `after_debt_repayment` repays
  debt first, using unearmarked cash before proportionately reducing pending
  dividend budgets. Only the remainder buys shares. Neither borrows for DRIP or
  spends unrelated cash. Interest on reserved cash is not reinvested.
- **Costs:** automatic dividend purchases have zero commission, half-spread and
  impact: `notional = budget`, `quantity = budget / raw_close`. Each paid action
  retains its trade and audit row with `trade_cost=0`; no cost rows or fee events
  are posted for that trade. Entry and scheduled rebalancing keep their configured
  costs, including when a scheduled basket uses released dividend cash. This is
  the user's modeling convention, not a claim about every broker's DRIP fees.
  Discounts, tax withholding and measured fill prices are not modeled.
- **Collisions:** `rebalance_only` releases the pending budget into account cash
  before the sweep and scheduled basket. No extra DRIP trade runs on that close.
  `hold_cash` preserves the final mark-only session; cash may still repay debt.
  These are currently the only supported collision/terminal policies.
- **Risk:** check maintenance/nonpositive equity before purchases and again after
  execution/marks; stopped runs retain actual coverage. DRIP never restores target
  weights/leverage or enforces a target concentration limit between scheduled dates.
  Old holdings earn the move into the fill; new shares earn only later moves and
  later ex-date entitlements. Analytical `returns(..., dividend_policy="reinvest_ex_close")`
  remains separate and never configures the simulator.

`result.dividend_reinvestments` has one row per nonzero entitled payment processed
while the policy is enabled (not per announced or unpaid action):

| Column | Type and meaning |
| --- | --- |
| `action_id`, `asset` | String; original action and paying asset |
| `pay_date` | Date; actual receipt date, possibly a non-session |
| `session` | Date, nullable; execution/release/check session; null if entirely consumed by the ordinary debt sweep |
| `paid_amount` | Float64; total entitled cash received |
| `debt_repaid` | Float64; earmarked principal consumed by the ordinary sweep before disposition |
| `cash_released` | Float64; earmark released to normal account funding, potentially used for debt or a scheduled basket |
| `signed_notional`, `trade_cost` | Float64; actual DRIP purchase (zero without a fill) and cost (always zero) |
| `trade_id` | String, nullable; join to `trades` and `costs` |
| `status` | String: `reinvested`, `debt_repaid`, `scheduled_rebalance`, `terminal_cash`, or `stopped_before_trade` |

The four disposition amounts sum to `paid_amount`. Audit rows do not themselves
post additional cash events. `cash_released` need not equal retained closing cash.
A pre-trade stop releases its earmark in the audit and freezes ledger balances;
there is no additional post-stop sweep. All normal accounting reconciliations run.

Purchases use trade `execution="dividend_reinvestment_close"`; related events have
`phase="reinvestment_close"`, a UTC close time and the original `action_id`.
A funded execution/check adds a `pre_reinvestment` valuation, retained in reports
and drawdown calculations. `turnover` adds `phase="dividend_reinvestment"` with
combined purchase notional divided by that close's pre-purchase equity. The
turnover plot displays it separately. Existing performance, rolling risk, exposure,
allocation and attribution plots consume these results without rerunning simulation.

Metadata stores the full policy under `dividend_reinvestment`, plus payment
availability, standing payer instruction, budget, margin and turnover conventions.
`reinvestment_cost_policy="zero_commission_spread_impact"` explicitly records the
zero-cost assumption independently of the ordinary `cost_rates`.
No new runtime dependencies or network access are required. See
[the runnable example](../examples/dividend_reinvestment.py).

## Historical SOFR financing

`rt.SOFRFinancing` is an alternative to `rt.Financing`, accepted as `financing=` by
both `buy_and_hold` and `scheduled_rebalance`, including dividend-reinvestment runs.
All fields below are explicit. The old fixed-rate mode and legacy cash-rate pair
retain their behavior. Do not mix a financing object with legacy cash arguments.

```python
financing = rt.SOFRFinancing(
    rates=sofr_rates,
    publication_calendar=sofr_publications,
    metadata=sofr_source_metadata,
    borrowing_spread_bps=100.0,      # Example modeled spread, not a default.
    cash_rate=0.0,                  # Fixed cash rate, independent of SOFR.
    day_count="ACT/360",
    cash_day_count="ACT/365F",
    rate_timing="known_at_accrual_start",
    max_rate_age_days=7,
    maintenance_equity_ratio=0.25,
    on_breach="stop",
    cash_sweep="repay_debt",
)
# Supply financing=financing in either existing backtest call.
```

### Rate inputs and provenance

Both tables are eager Polars DataFrames with exact schemas and unique observation
dates; no null/nonfinite values or implicit casts. Inputs are copied and sorted.

| Table | Columns |
| --- | --- |
| `rates` | `observation_date: Date`, `sofr: Float64` (annual decimal; 5% = 0.05) |
| `publication_calendar` | `observation_date: Date`, `available_at: Datetime(us, UTC)` |

The calendar must be supplied independently as complete for the declared coverage,
with **exactly the same observation keys** as the rate table. This rejects a
missing expected observation rather than silently carrying an older quote through
it. Include enough history before entry to seed the first cutoff. Availability
must fall after the observation date in New York, increase strictly with
observation dates, and not follow `retrieved_at`. No holiday dates, publication
hours or revision vintages are guessed. Later revisions must not be presented as
known earlier; reconstruct verified vintages before using this interface.

`metadata` is a finite JSON dictionary requiring:

- Nonblank `source`, `calendar`, `calendar_version`.
- UTC ISO `retrieved_at`; ISO `coverage_start`/`coverage_end` covering all requested
  calendar accrual dates (first day after entry through ending session).
- `currency="USD"`, `rate_units="decimal"`, `timezone="America/New_York"`.
- `calendar_complete=True`, `vintage="point_in_time"` source assertions.

Other provenance fields may be added. The library checks consistency, not the
truth of a provider's asserted completeness or vintage. Only USD portfolios are
supported. Negative SOFR/spreads/cash rates raise; no automatic rate floor exists.
`max_rate_age_days` is a required positive integer, excluding bool.

`financing.snapshot_id` is a deterministic SHA-256 identity of the canonical input
tables and metadata. Mutating owned tables/metadata after construction is detected
at simulation; construct a new configuration to change inputs.

### Rate selection and accounting

The only initial `rate_timing` is `known_at_accrual_start`: at New York midnight
on each posted accrual date, use the latest supplied observation available by that
time (equality allowed). This is a deliberately lagged, known-rate loan convention.
A Monday morning publication first affects Tuesday's charge. Missing initial data,
exceeded age limits (accrual date minus observation date), and incomplete declared
coverage raise before any simulation. Weekend/holiday carry is permitted only
under this explicit timing policy and age limit. No market prices are filled.

Borrowing uses `sofr + borrowing_spread_bps/10_000`; cash continues to use fixed
`cash_rate`. `day_count` and `cash_day_count` independently accept `ACT/360` or
`ACT/365F`. Opening calendar-day balances accrue once before actions/payments and
sweeps; interest capitalizes daily, including non-session dates. This is not the
SOFR Index or a broker-specific billing model. Interest on a weekend is attributed
to the next supplied market session, without an invented daily market-return row.

Margin, debt sweeps, dividend reservation, scheduled funding and stopped-run rules
remain those of the shared ledger. Requested rate coverage is validated even for
zero borrowing or an eventually stopped run; posted audit rows end at the actual
stop. Cash and loan interest remain distinct from performance risk-free rates.

### Audit and replay

`result.financing_accruals` is populated for SOFR mode and typed empty otherwise.
It contains one row per processed calendar date, including zero-balance dates:

| Columns | Type / meaning |
| --- | --- |
| `date` | Date; posted accrual date |
| `cutoff_at`, `available_at` | Datetime(us, UTC); selection cutoff and selected vintage availability |
| `observation_date`, `rate_age_days` | Date, Int64; selected observation and age in calendar days |
| `sofr`, `borrowing_spread_bps`, `borrowing_rate`, `cash_rate` | Float64; annual decimal rates except spread in basis points |
| `borrowing_day_fraction`, `cash_day_fraction` | Float64; 1/360 or 1/365 |
| `opening_debt`, `opening_cash` | Float64; balances before that calendar date's accrual/actions |
| `borrowing_interest`, `cash_interest` | Float64; posted monetary amounts, including explicit zeros |

Join `date` to borrowing-interest cost rows or cash-interest events to reconcile
postings. Nonzero loan charges still have `component="borrowing_interest"`, with
null trade/asset IDs. All normal position, cash/debt, attribution, return and equity
reconciliations apply. Performance and plots consume these results normally.

`result.metadata["borrowing_rate"]` is **None** in variable-rate mode, not a final
or average rate presented as constant. `metadata["financing"]` contains the full
configuration, source metadata, snapshot identity, and exact rate/publication rows
with ISO dates/timestamps, all JSON-serializable. Market snapshot persistence is
unchanged; save separate SOFR tables with Polars Parquet and source metadata as
JSON, or reconstruct them from the embedded run records. No network access or
provider dependency occurs in financing/simulation. See the
[offline runnable example](../examples/historical_sofr.py), which uses synthetic
rates rather than claiming historical market results.
