# Examples and acceptance replay

`unlevered_buy_and_hold.py` is the small 1A/1B hand-checkable example;
`financed_buy_and_hold.py` adds 1×/2× comparisons and a separate margin-stop case.
Both run offline after installing the checkout.

[buy_and_hold_equities.ipynb](buy_and_hold_equities.ipynb) is the completed milestone
1 acceptance notebook. Five synthetic equity holdings start at $100 million and
run from 2024-01-02 through 2025-01-02: 263 session observations and 262 holding
intervals. The fixture uses **every weekday**, including real exchange holidays,
at 21:00 UTC; it is deliberately not represented as an exchange calendar. Explicit
252-period annualization is an example assumption, not inferred from row count.

The notebook loads [saved snapshots](snapshots/v1), displays provenance/actions,
configures weights, entry costs and financing, runs 1×/1.5×/2× ledgers, checks
balances/attribution, and reports daily P&L/returns, cumulative sum versus compounding,
benchmark correlation/beta, drawdown and risk ratios. A benchmark snapshot supplies
a hypothetical reinvested total-return index without costs; portfolio dividends
remain in cash/receivables or repay debt. Stopped scenarios are excluded from full
comparisons and receive explicit partial reporting. The default scenarios complete.

Nine plots show raw prices with split markers, P&L, simple returns, equity,
drawdown, distribution, drifting allocation, dollar attribution, and correlations.
Asset correlations use separately simulated, zero-cost, zero-interest single-asset
portfolios with dividends held in cash. They do not use split-discontinuous raw
price returns. The final Markdown observations were written after inspecting the
actual default tables and rendered plots; rewrite them when changing inputs.

## Run

From the repository root, using Python 3.12+:

```sh
python -m pip install -e '.[test,notebook]'
python -m pytest -q
python examples/run_acceptance.py
```

Or open the source notebook with a Jupyter interface using the same installed
environment. The headless runner launches this Python as the local kernel;
localhost sockets are required for Jupyter, but data acquisition is blocked inside
the notebook. No Internet access or credentials are needed for execution after
installation. The optional notebook extra includes the execution kernel/tools,
not a JupyterLab user interface.

Generated output goes to ignored `artifacts/acceptance/`: an executed notebook,
nine PNG figures, daily CSV, exact environment versions, and Git revision/status
provenance. The committed source notebook stays clean. To reproduce the exact
CPython 3.14.6/macOS arm64 acceptance environment in a fresh virtual environment:

```sh
python -m pip install -r examples/requirements-acceptance.txt
python -m pip install --no-deps --no-build-isolation -e .
python examples/run_acceptance.py
```

Consumer dependencies remain ranged. Compatibility tests also cover Python 3.12.11,
Polars 1.30.0 and Matplotlib 3.9.0, including notebook execution. These checks are
not a guarantee for every operating system or future dependency version.

## Fixture provenance and regeneration

`snapshots/v1/equities` and `snapshots/v1/benchmark` contain self-authored, small
synthetic Parquet tables and manifests with SHA-256 hashes and canonical identities.
No reference-project code or third-party market data was copied. The fixture has
no third-party data redistribution restrictions; the repository's own license
remains undecided. All rates, prices, calendars and costs are modeling assumptions.

`make_acceptance_snapshot.py` records the deterministic generation method. To
inspect a fresh generation without replacing the canonical saved inputs:

```sh
python examples/make_acceptance_snapshot.py artifacts/new-synthetic-snapshot
```

The destination must not exist. Canonical replay uses saved bytes, because last-bit
libm rounding during generation can vary across platforms. Never replace these
fixtures with credentials, private data, or licensed data without reviewing its
redistribution terms. Real-data research requires its own supplied calendar,
source/action coverage and clearly identified benchmark convention.

## Milestone 2: scheduled allocation

[scheduled_rebalancing.ipynb](scheduled_rebalancing.ipynb) and
[scheduled_rebalancing.py](scheduled_rebalancing.py) use the same committed synthetic
snapshot. The default starts with $10 million after a warm-up, uses 40 pre-decision
returns, materializes first-supplied-session monthly targets, caps each risky
proportion at 40%, and requests 1.25× post-cost exposure. Target dates are explicit;
non-session dates raise. The January 1 synthetic-calendar target is deliberately
not presented as a real-exchange execution date.

The shared ledger produces actual purchases/sales, component costs, debt changes,
turnover and attribution; no return-series multiplication or daily target-weight
multiplication substitutes for accounting. The script compares a scheduled run to
buy-and-hold from the identical initial basket. Plots show allocations, turnover,
exposures, equity, rolling risk and estimated covariance risk contributions.

```sh
python examples/scheduled_rebalancing.py
python examples/run_acceptance.py --milestone 2
```

The latter writes an executed notebook with six embedded figures under ignored
`artifacts/milestone2/`, alongside numerical CSVs and environment/revision provenance.
The notebook's configuration exposes window, warm-up, annualization, leverage,
concentration limit and capital; the short example workflow holds explicit fee,
funding and margin settings. Its Markdown observations were written after inspecting
the default tables and figures. Inputs are synthetic; no provider, license choice or
new dependency is needed to run it offline after installation.

## Payment-date dividend reinvestment

[dividend_reinvestment.py](dividend_reinvestment.py) compares default cash sweeping,
debt-first reinvestment and reinvestment-first reservation on the same saved
five-asset synthetic snapshot. All use identical initial weights, capital, costs
and financing. Import `run_example(initial_capital=..., gross_leverage=...)` into a
notebook for the numerical runs, reports and comparison table. No plotting import
is required for that function.

```sh
python examples/dividend_reinvestment.py
```

The script writes daily/payment audits, a comparison CSV and two PNG figures under
ignored `artifacts/dividend_reinvestment/`. The standing instruction buys the paying
stock at the first supplied close on/after payment, using the full available
dividend budget with no trading costs. Entry/rebalancing costs still apply. Payments are assumed available before the close. It does not model broker
DRIP discounts, exact fills, withholding or per-asset enrollment. The final session
remains mark-only. This example does not change the existing acceptance notebooks'
cash-held dividend assumptions or their interpretation.

## Historical SOFR interface

[historical_sofr.py](historical_sofr.py) is an offline, hand-checkable demonstration
of the historical-rate interface. **Both SOFR values and stock prices are
synthetic**, chosen for simple arithmetic; it is not a historical performance claim.
It prints each selected fixing and charge, plus daily debt/equity/P&L:

```sh
python examples/historical_sofr.py
```

Outputs go to ignored `artifacts/historical_sofr/`: daily/accrual CSV tables and
full JSON run metadata, including rate/calendar source records. Import
`run_example()` in a notebook to inspect the result without writing files.

For real research, replace `rates` with saved, permitted annual-decimal observations
and `publication_calendar` with complete verified observation/availability records.
Do not create that calendar by dropping missing rows from the rate file. Document
units, coverage and point-in-time vintage; current revised history alone is not
proof of past availability. The source metadata assertions are the caller's
responsibility. The example and library never download rates. Publication/day-count
background is linked in [financial conventions](../docs/financial-conventions.md#historical-sofr-loan-convention).
