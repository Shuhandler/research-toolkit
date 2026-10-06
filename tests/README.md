# Tests

Run `python -m pytest -q` after installing `.[test]`. Tests use small synthetic
Polars tables and block network connections. `conftest.py` supplies explicit input
schemas, provenance, sessions, and portfolio policies.

- `test_data.py`: strict schemas, metadata, calendar/price alignment, action checks,
  snapshot identity, and input preservation.
- `test_returns.py`: simple/log returns, cumulative sums versus compounding,
  adjustment basis, leading nulls, and invalid interval handling.
- `test_backtest.py`: hand-calculated entry costs, splits, dividend entitlements,
  unpaid receivables, weekend payments/interest, cash-only/partial allocations,
  drifting weights, scaling, and independent ledger reconstruction.
- `test_financing.py`: financed entry sizing, calendar-day debt capitalization,
  dividend repayment, 1× equivalence, leverage drift, margin/insolvency stops,
  coverage guards, and financed multi-asset reconstruction at $100 million.

- `test_metrics.py`: hand-calculated metrics, benchmark alignment, null reasons,
  corporate-action-aware entry drawdown, stopped reports and asset correlations.
- `test_snapshots.py`: replay, immutable writes, hash/schema/identity corruption.
- `test_series_performance.py`: NAV accounting, compounding, initial-capital
  drawdown, strict benchmark alignment, interval/NAV validation, plot and
  comparison compatibility, VaR ranks, ties and independent dollar tails.
- `test_yahoo_download.py`: mocked downloads, inclusive exchange-local dates,
  retries/timeouts (closing every HTTP error), failures, duplicates, reported missing
  values, request-ordered rows, retained payloads and bars still open at retrieval.
- `test_calendars.py`: real exchange calendars offline: skipped sessions (even
  when every asset lacks them), holidays, early closes, DST, before/after close,
  listing limits, suspensions, multi-exchange and multi-session intervals, assets
  with no bars, returned session tables, calendar argument errors, and a download
  converted by `yahoo_chart` into raw prices and run through `buy_and_hold`.
  Skipped unless the `calendar` extra is installed.
- `test_performance_diagnostics.py`: calendar-time and trading-period CAGR,
  initial-capital drawdown and Calmar, negative CAGR, zero drawdown, G1/G2 hand
  values and a pandas reference, degenerate samples and nonpositive wealth.
- `test_factor_regression.py`: known coefficients, joint versus standalone betas,
  residual degrees of freedom and idiosyncratic volatility, perfect fits, singular
  and constant designs, strict alignment, excess-return conversion rules and a
  numpy least-squares reference.
- `test_signal_diagnostics.py`: positive/negative rank IC, average-rank ties,
  per-date (not pooled) means, constant/insufficient cross-sections, uneven and
  extreme groups, group tie rules, changing universes and exclusions, forward
  timing and overlapping horizons.
- `test_public_api.py`: every `rt.*` call named in the README and API docs exists
  and is exported; `__version__`, `__all__`, and plot views without other tests
  (risk contributions, histogram counts, split markers, ledger-only errors, guards).
- `test_validation_guards.py`: documented rejections across the public API, cost
  solver guard rails, comparison guards, restored research audits, malformed Yahoo
  payloads, numpy number handling and tail risk on a report with no intervals.
- `test_plots.py`: lazy optional imports, plotted values/dates, reusable axes,
  stopped labels and rendering; plotting tests skip when the extra is absent.
- `test_acceptance.py`: saved one-year inputs, five holdings, $100 million,
  all-cash/financed cases and independent balance/attribution reconstruction.

Install `.[test,plot,calendar]` to run every unit/integration test (calendar tests
skip without the `calendar` extra; the notebook test stubs IPython's `display`). The acceptance notebook
complements these checks; run `python examples/run_acceptance.py` with the notebook
extra to render it and inspect the figures. See [the testing plan](../docs/testing-plan.md).

Milestone 2 adds `test_allocation.py` (pre-decision windows, inverse-volatility
oracles, concentration rejection, covariance contributions, explicit analytical
actions and realized rolling risk) and `test_rebalancing.py` (drift versus target
trades, financing changes, asymmetric costs, exits, zero turnover, split/dividend
timing, receivable funding, pre-trade stops and independent event reconstruction).
The five-asset monthly example is also tested against its complete exported ledger.
Existing M1 tests still run against the shared engine.

`test_dividend_reinvestment.py` covers opt-in payment-funded DRIP, both debt
priorities, timing/splits/entitlements, actual costs, scheduled/terminal collisions,
margin stops, per-payment audits and independent ledger reconciliations.

`test_sofr.py` covers historical-rate input contracts, publication-aware selection,
calendar/day-count accrual, source identity, offline replay and shared ledger
integration. Rates in these tests are synthetic, not historical SOFR fixtures.


`test_long_short.py` checks signed price P&L, equity-neutral short entry,
restricted collateral, exact-quantity hedges, fixed/nonlinear actual-trade costs,
partial covers and sign crossings, weekend/dated borrow fees, historical SOFR,
short dividends/splits, protected zero-cost long DRIP, side-specific margin stops,
reporting and zero-benchmark information ratios with small numerical oracles.
