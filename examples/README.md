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
