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

Borrowing, performance ratios, plots, and research workflow checks will be added
with their implementations; see [the testing plan](../docs/testing-plan.md).
The future five-stock notebook complements, but does not replace, accounting tests.
