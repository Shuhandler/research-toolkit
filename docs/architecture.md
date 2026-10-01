# Architecture and extension proposal

Milestones 1A–1D implement data, returns, allocation, costs, financing, the ledger,
performance, local snapshots, and optional plotting. [The implemented API](api.md) is authoritative for
current signatures and schemas. This document also retains later design targets;
the full financed/performance/plotting acceptance notebook is implemented. Add each
future module only with working, tested behavior, without stub hierarchies.

## Boundaries and dependencies

| Planned module | Responsibility | Earliest phase |
| --- | --- | --- |
| `__init__.py` | Small documented facade; no network or plotting side effects | Setup |
| `_data.py` | Polars schemas, key validation, session alignment, provenance | 1A |
| `_returns.py` | Simple/log returns and cumulative transformations | 1A |
| `_results.py` | Concrete validated data/result containers | As their consumers arrive |
| `_portfolio.py` | Equal/custom weights, exposure and drift calculations | 1B |
| `_costs.py` | Trade-level commissions/spread assumptions | 1B |
| `_financing.py` | Explicit cash/loan rates, day count, sweep and margin configuration | 1C |
| `_backtest.py` | Ordered events, quantities, cash, receivables, debt, reconciliation | 1B–1C |
| `_snapshots.py` | Immutable local Parquet/JSON snapshots with file hashes and data identity | 1D |
| `_metrics.py` | Performance tables, benchmark comparisons, rolling calculations | 1D onward |
| `plots.py` | Public plotting namespace consuming prepared numerical results | 1D onward |
| `_allocation.py` | Inverse-volatility and later constrained allocations | 2 |
| `_research.py` | Lagged features and chronological splits without ML requirements | 3 |
| `data_sources/` | Optional acquisition adapters producing the standard contracts | When needed |
| `models/` | Optional model comparisons; never a dependency of simulation | 4 |

Dependency direction: input validation and arithmetic → portfolio/cost rules →
simulation → performance → plotting. Providers only produce inputs. Plotting does
not call the simulator. Numerical code does not import plotting, providers, or ML.
Start with Polars, adding NumPy only where useful, and Matplotlib in a plotting
extra when implemented. Use ordinary concrete dataclasses for actual result
containers; no plugin registry or generic engine framework in milestone 1.

## Input contracts

Public tables are eager `polars.DataFrame` objects. Use explicit columns, not row
position or an implicit index. Numerical columns are Float64; decimal rates are
fractions (`0.01` means 1%), shares can be fractional, and monetary columns are in
the declared base currency. Preserve input objects; return new tables. Sort
validated keys into canonical order and report that normalization; never deduplicate
or aggregate conflicting rows automatically.

| Table | Required columns / key | Semantics |
| --- | --- | --- |
| Daily prices | `session: Date`, `asset: String`, `close: Float64`; unique `(session, asset)` | Positive finite raw close for execution and valuation; one currency and shared calendar in M1 |
| Sessions | `session: Date`, `close_at: Datetime(us, UTC)`; unique `session` | Complete expected exchange sessions, including holidays/early closes through explicit UTC closes |
| Splits | `action_id: String`, `asset`, `effective_session: Date`, `ratio: Float64` | New shares per old share, positive ratio, effective before session trading |
| Dividends | `action_id`, `asset`, `ex_session: Date`, `pay_date: Date`, `cash_per_share: Float64` | Ordinary cash dividend in base currency per post-split share; entitlement from pre-ex holdings adjusted for same-day split |
| Initial weights | `asset: String`, `weight: Float64`, or explicit mapping | Nonnegative, finite risky-asset proportions summing to 1 within tolerance; separate initial gross leverage sets cash/debt |
| Benchmark returns | `period_start: Date`, `session: Date`, `simple_return: Float64` | Returns over exactly the same holding intervals, with source and price/total-return basis metadata |

Empty, typed action tables require an explicit source assertion of complete action
coverage. An empty table must not mean "we did not check." Metadata must also
declare coverage sufficient for receivables still unpaid at the end. M1 rejects
shorts, multi-currency inputs, nonpositive equity, unsupported corporate actions,
and ambiguous dividend adjustment bases. It supports all-cash via initial leverage
zero; no trades or debt are created. Weights remain a declared allocation template.

Raw price returns are **price-only** and split-discontinuous. Total returns must be
explicitly labeled: a supplied documented total-return series, or returns derived
from a corporate-action-aware ledger. Adjusted prices may be used for explicitly
labeled analytics; M1 rejects them as execution input. Do not apply actions twice.

Missing prices on an expected session, duplicates, nonfinite values, unknown
assets, dates outside declared coverage, or currency/timezone inconsistencies
raise contextual validation errors. A non-session date is not a missing price.
Never forward-fill. Price gaps must not become a false multi-day "daily return."
Reject missing held-asset marks during simulation. For benchmark comparisons,
`alignment="strict"` rejects unmatched intervals. A later explicitly requested
intersection mode must report every excluded interval and changed sample count;
it must not change the underlying portfolio's performance history.

Daily `session` is an exchange-local date label, not UTC midnight. The supplied
session table and metadata specify the exchange timezone/calendar version. For
intraday later, introduce explicit UTC `bar_start`, `bar_end`, `available_at`, and
execution timestamps; do not overload daily session dates or infer bar availability
from labels. Multi-exchange calendars need an explicit policy before support.

## Snapshot and run metadata

A local snapshot consists of tables plus a JSON manifest: schema version, frequency
(`1d` in M1), return interval convention where relevant, source
and retrieval timestamp (UTC), source identifiers, asset/currency mapping, price
adjustment semantics, action coverage, timezone, session calendar provenance,
requested/actual coverage, missing-data diagnostics, transformations, file SHA-256
hashes, and usage/redistribution restrictions. Keep original and transformed data
separately. Store the canonical data identity in the run result; per-file hashes and the
manifest remain in the snapshot directory.

A run stores resolved policy/cost/rate settings, code version or revision, dirty-tree
status where available, environment versions, input hashes, and start/end coverage.
Identical saved inputs and configuration must run without acquisition code. Provider
caches alone are not reproducible snapshots. No hidden "latest" date or network
fetch in a calculation. A simple manifest reader/writer is enough when needed.

## Result objects and numerical tables

Implemented `MarketData` holds validated tables, metadata, diagnostics, and snapshot identity.
`ReturnResult` holds a `values` table keyed by `(session, asset)`, with
`period_start` and explicitly named `simple_return` or `log_return` columns, plus
return-kind, frequency, basis, and source metadata. Cumulative transformations
preserve this information and add distinctly named cumulative columns. The leading
undefined interval remains null. This keeps standalone calculations auditable
without depending on a simulation result.

Implemented `BacktestResult` contains the tables below plus a per-action
`receivables` table. Exact current fields are documented in [the API](api.md).

| Table | Key and important fields |
| --- | --- |
| `daily` | `session`; `period_start`, `opening_equity`, `equity`, `pnl`, `simple_return`, `log_return`, `cumulative_pnl`, `cumulative_simple_return`, `compounded_return`, `cash`, `debt`, `dividend_receivable`, `gross_exposure`, `net_exposure`, `gross_leverage`, `drawdown`, `equity_ratio`, `margin_breached` |
| `positions` | `(session, asset)`; quantity, raw mark, market value, weight relative to equity |
| `trades` | `trade_id`; session/time, asset, signed quantity, reference price, signed notional, execution policy, trade cost |
| `costs` | `cost_id`; date/time, nullable `trade_id`/asset for financing charges, component, amount, `basis="modeled"` |
| `events` | `event_id`; time and deterministic sequence, type, asset/action/trade identifiers, quantity/cash/debt/receivable deltas |
| `valuations` | `(time, phase)`; pre-entry, post-entry, and subsequent closing balance-sheet values |
| `attribution` | `(session, component, asset if applicable)`; dollar contribution, including financing and cost rows |
| `diagnostics` | session, code, currency reconciliation residual and tolerance |

`daily` contains completed holding intervals, not an extra zero-duration entry
return. The first interval includes entry costs; pre/post-entry valuations remain
visible separately. See [timing conventions](financial-conventions.md).

Implemented `PerformanceResult` contains `summary` (`metric`, `value`, `unit`,
`n_obs`, `status`), `benchmark_comparison`, `benchmark_series`, `drawdowns`, and metadata for frequency,
annualization, sample window, risk-free convention, and return basis. Undefined
metrics are null with a reason code. Invalid inputs raise; they do not become null
metrics. `CorrelationResult` uses the same null/status convention; rolling risk remains later work.
Status codes are diagnostics, not strategy recommendations.

Results carry `complete` or `stopped` status and stop reason/session/time. The
failure close is retained, with requested and actual coverage in metadata.
`require_complete()` rejects stopped runs; 1D whole-period reports must call it
unless an explicitly requested partial report labels coverage and stop reason.
Numerical failures still raise. Portfolio inputs are not mutated.

## Public workflow — implemented, supply your input tables and configuration

The first five facade operations `prepare_market_data`, `returns`,
`cumulative_returns`, `equal_weights`, and `buy_and_hold` are implemented.
`Financing`, `performance`, `correlation`, snapshots and the plotting namespace are
implemented. The workflow below uses a few
concrete policy/result types and the `plots` namespace. Avoid exporting internal
helpers. Keyword-only policy arguments should carry meaningful names and units.

```python
import research_toolkit as rt

# All tables/configuration below come from the notebook's local snapshot.
market = rt.prepare_market_data(
    prices=raw_prices, splits=splits, dividends=dividends,
    sessions=sessions, metadata=source_metadata, missing="raise",
)
weights = rt.equal_weights(asset_ids)  # Or {asset: proportion, ...}.
result = rt.buy_and_hold(
    market,
    weights=weights,
    initial_capital=100_000_000.0,  # Example equity, not a library default.
    entry_session=entry_session,
    end_session=end_session,
    policy=rt.BuyHoldPolicy(
        execution="entry_close", sizing="post_cost_equity",
        fractional_shares=True, initial_gross_leverage=1.0,
        terminal_action="mark_only",
    ),
    costs=rt.TradeCosts(
        commission_bps=commission_bps,
        half_spread_bps=half_spread_bps, impact_bps=impact_bps,
    ),
    financing=rt.Financing(
        cash_rate=cash_rate, borrowing_rate=borrowing_rate,
        day_count="ACT/365F", maintenance_equity_ratio=margin_ratio,
        on_breach="stop", cash_sweep="repay_debt",
    ),
)
report = rt.performance(
    result, benchmark=benchmark_returns, benchmark_metadata=benchmark_metadata,
    alignment="strict",
    periods_per_year=252, risk_free_annual_effective=0.0,
    minimum_acceptable_return_annual_effective=0.0,
)
result.daily.select(
    "session", "pnl", "simple_return", "cumulative_simple_return",
    "compounded_return", "equity",
)
report.summary
report.benchmark_comparison  # Return correlation, beta, n_obs, status.
benchmark_equity = report.benchmark_series.select("session", "equity")
fig, ax = rt.plots.equity(report)
fig.savefig("artifacts/equity.png", dpi=150)
```

All costs/rates must be supplied, including explicit zeros. The example's 252 and
zero risk-free/MAR values are visible choices, not universal inferred settings.
`benchmark_equity` is a precomputed table from performance; plots take the report
to preserve its status, coverage and benchmark basis. Repeat
the simulation with another `initial_gross_leverage` to compare financed cases;
report the resulting debt and drift, not only scaled returns.

Implemented standalone calls `rt.returns(market, method="simple", basis="price")` and
`rt.cumulative_returns(return_result, method="compound")` require validated keys
and explicit return-kind metadata. The validated `market` supplies the calendar
and source metadata; a bare price table is not accepted. They return `ReturnResult`.
`method="sum"` produces a distinctly named
column; a log-return input is never silently interpreted as simple returns.

## Plotting and later extensions

Matplotlib functions accept prepared reports/market data/correlations plus an optional `ax`, return
`(Figure, Axes)` (or a documented axes array for multi-panel figures), and do not
call `show()` or change global styles. Labels state dates, currency, percent versus
decimal units, gross/net status, and return basis. No fetching or recomputing
metrics in plotting; direct display normalization must be explicit.

Implemented views: price, returns, dollar P&L, equity, drawdown, distribution,
correlation, allocation, and attribution. Rolling risk remains planned. Add a view
only after its underlying numerical result exists. Interactive backends can be optional later.

Future rebalancing consumes dated target weights and creates **quantity changes**
through the same ledger; scheduled and signal policies do not replace accounting.
Inverse-volatility weights use only a stated trailing window ending before the
decision; insufficient/zero volatility is an explicit error or named policy.
Concentration limits apply to declared equity or gross-exposure denominators;
violations cannot silently renormalize targets. Risk contributions use a declared
covariance window and denominator, with zero-risk handling.

Lagged features retain availability times. Chronological train/validation/test
splits enforce nonoverlap of label horizons at boundaries; purge/embargo rules
depend on those horizons. Fit transforms on training data only. Choose parameters
on validation folds, and evaluate the reserved test once. Model parameters live
separately from dated portfolio weights. Walk-forward and model comparisons are
later consumers of these contracts, not prerequisites for the first release.

Packaging follows the [PyPA package tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
The distinction between distribution and import names follows
[PyPA's terminology](https://packaging.python.org/en/latest/discussions/distribution-package-vs-import-package/).
