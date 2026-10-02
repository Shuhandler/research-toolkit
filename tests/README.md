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
- `test_plots.py`: lazy optional imports, plotted values/dates, reusable axes,
  stopped labels and rendering; plotting tests skip when the extra is absent.
- `test_acceptance.py`: saved one-year inputs, five holdings, $100 million,
  all-cash/financed cases and independent balance/attribution reconstruction.

Install `.[test,plot]` to run every unit/integration test. The acceptance notebook
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
