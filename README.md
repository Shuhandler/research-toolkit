# research-toolkit

A notebook-friendly Python library for quantitative trading research: prepare
market data, construct portfolios, simulate trades, measure performance, and plot
results through a small, consistent interface.

**Status: milestones 1 (1A–1D), 2 and 3 implemented.** Strict Polars inputs, a financed
buy-and-hold ledger, performance and benchmark tables, optional Matplotlib plots,
replayable snapshots, inverse-volatility allocation, and scheduled rebalancing
are implemented. The [acceptance notebook](examples/buy_and_hold_equities.ipynb)
runs offline on clearly labeled synthetic data. See the [implemented API](docs/api.md).

## Purpose and scope

Built for learning and practical research in Hedge Fund Strategies and Algorithmic
Trading classes. Calculations should remain understandable, reproducible, and
auditable while reducing repetitive notebook code. Polars tables are the chosen
data interface. Machine learning will be optional; ordinary plotting and
backtesting must never require a model framework.

The current library supports daily, single-currency equity buy-and-hold portfolios:
raw prices, explicit splits and cash dividends, equal or custom initial weights,
entry costs, cash holdings, initial leverage, calendar-day financing, and dividend
cash repayment of debt, plus opt-in payment-date dividend reinvestment. Saved inputs
run entirely offline. Buy-and-hold never rebalances; scheduled strategies use supplied
target dates. Either can add explicitly configured dividend purchases. Neither performs
an implicit terminal sale. This is a research simulator, not a broker execution system.

Scheduled portfolios use dated targets, explicit funding policies, concentration
checks and actual quantity-changing trades. Risk contributions, rolling risk,
turnover and exposure views are available. Notebook extensions add capped inverse
volatility, square-root impact and cost-aware sizing, dated risk-free reporting,
rolling beta/correlation, cumulative plots, comparison tables and an offline Yahoo
chart adapter. See the [migration guide and contracts](docs/notebook-extensions.md)
and [runnable workflow](examples/research_workflow.py).

Milestone 3 adds lagged features, purged chronological splits, training-only
standardization, validation/final-test audits and dated long-only signals with
explicit next-session-close execution. See the [research guide](docs/chronological-research.md)
and [notebook](examples/chronological_research.ipynb). Later work covers
short positions, fixed/minimum ticket fees, intraday inputs and walk-forward model
workflows. See the [roadmap](docs/roadmap.md) for scope boundaries.

## Usage

The distribution name is `research-toolkit`; the Python import is:

```python
import research_toolkit as rt
```

Use `rt.prepare_market_data(...)` to validate prices, sessions, action tables, and
source metadata. `rt.returns(market, method="simple", basis="price")` and
`rt.cumulative_returns(...)` provide explicitly labeled return arithmetic.
`rt.equal_weights(...)` or custom weights feed `rt.buy_and_hold(...)`, which returns
positions, trades, costs, corporate-action events, receivables, daily P&L, equity,
and reconciliation diagnostics. The first return includes entry costs once.
Supply `rt.Financing(...)` for borrowing, with explicit rates, day count, cash sweep,
and maintenance threshold. Runs stop on a breached threshold or nonpositive equity;
use `result.require_complete()` before treating a result as a full-period run.

The [offline example](examples/unlevered_buy_and_hold.py) supplies a complete small
synthetic portfolio with a split, dividend, and entry costs. All quantities and
financial policies are explicit. The [financed example](examples/financed_buy_and_hold.py)
compares 1× and 2× initial leverage and demonstrates a stopped run.

The [acceptance notebook](examples/buy_and_hold_equities.ipynb) uses five synthetic
stocks, $100 million starting equity, one calendar year, entry costs, a benchmark,
and 1×/1.5×/2× financed scenarios. It reports P&L, simple returns, their cumulative
sum, compounded return, ending equity, return correlation/beta, risk metrics and
nine chart types. These are example parameters, never library constants.

```python
report = rt.performance(
    result, periods_per_year=252, risk_free_annual_effective=0.03,
    minimum_acceptable_return_annual_effective=0.0,
)
report.summary
fig, ax = rt.plots.equity(report)  # Optional plot extra.
```

Supply an optional strictly matched benchmark table and its source/basis metadata
for comparisons. Reports reject stopped runs unless `allow_partial=True` is
explicit, and plots label their actual coverage and stop reason. Save validated
inputs with `rt.save_snapshot(market, new_directory)` and replay using
`rt.load_snapshot(directory)`; loading verifies file hashes and input identity.

The [Milestone 2 notebook](examples/scheduled_rebalancing.ipynb) demonstrates monthly
inverse-volatility targets, concentration checks, 1.25× financing, actual trade
costs, turnover, allocation drift and risk diagnostics. Its [short Python workflow](examples/scheduled_rebalancing.py)
shows how to assemble calls without notebook-specific library logic. Allocation
estimates exclude the decision session and all future returns; execution occurs
later. Unpaid dividend funding is an explicit policy. See [the exact API](docs/api.md#scheduled-allocation-and-rebalancing--milestone-2).

## Automatic dividend reinvestment

Pass this optional policy to either `rt.buy_and_hold(...)` or
`rt.scheduled_rebalance(...)`:

```python
dividend_reinvestment = rt.DividendReinvestment(
    execution="first_close_on_or_after_payment",
    funding="before_debt_repayment",
    scheduled_collision="rebalance_only",
    terminal_action="hold_cash",
)
# Add dividend_reinvestment=dividend_reinvestment to your backtest call.
```

This reserves the actual paid dividend to buy fractional shares of the paying
stock with no commission, spread or impact deducted. Entry and scheduled trades
retain their configured costs. Choose `"after_debt_repayment"`
to repay debt first and reinvest only the remaining dividend cash. Neither option
borrows to fund the purchase. Omitting the policy preserves existing behavior.

Execution assumes payment is available before that day's close; weekend payments
wait for the next supplied session. Scheduled targets take priority on overlapping
dates, and no purchase occurs on the final session. A standing instruction can
reopen a previously sold payer. These are explicit research assumptions, not a
broker DRIP fill model. `result.dividend_reinvestments` links each paid dividend
to its trade or other disposition. See the [full contract](docs/api.md#automatic-dividend-reinvestment)
and the [runnable comparison](examples/dividend_reinvestment.py).

## Historical SOFR financing

Pass `rt.SOFRFinancing(...)` as the `financing` argument to either simulator.
It accepts saved annual-decimal SOFR observations plus a supplied publication
calendar, adds an explicit borrowing spread, and supports ACT/360 or ACT/365F.
Rates become eligible only once published; each accrual date uses the latest rate
known at midnight in New York. Holiday carry is explicit and age-limited; missing
expected observations raise. Cash interest remains a separately configured fixed
rate and day count. Existing `rt.Financing` calls retain fixed-rate behavior.

`result.financing_accruals` records the rate, publication timestamp, opening
balances and interest for every processed calendar date, including weekends and
zero-debt dates. Rate data, source metadata and identity are retained for replay.
See [the complete contract](docs/api.md#historical-sofr-financing) and the
[runnable offline example](examples/historical_sofr.py). Its rates are deliberately
synthetic; replace them with permitted, verified historical inputs for research.
The library does not fetch rates or reconstruct publication vintages automatically.

This model capitalizes interest daily, including weekends. It does not reproduce
the official SOFR Index, retrospective overnight fixings, or a broker's exact
margin-loan billing rules.

## Local development

Python 3.12+ is the target. From this checkout:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,notebook]'
python -m pytest -q
python examples/unlevered_buy_and_hold.py
python examples/financed_buy_and_hold.py
python examples/run_acceptance.py
python examples/run_acceptance.py --milestone 2
```

Run the notebook kernel from that environment. The commands install the local
checkout; do not install an unrelated package from an index by name. The build
backend may need downloading. Polars is the only runtime dependency; pytest is an
optional test dependency. Use `.[plot]` for figures or `.[notebook]` to execute the
notebook. Core imports do not load Matplotlib, NumPy, pandas, providers, or ML.
Tests and the example use synthetic inputs and run offline. The implementation was
validated on Python 3.14.6 with Polars 1.44.2 and Python 3.12.11 with the declared
minimum Polars 1.30.0 and Matplotlib 3.9.0. The current environment uses Matplotlib
3.11.2. Both execute the acceptance notebook. Exact replay dependencies are in
[examples/requirements-acceptance.txt](examples/requirements-acceptance.txt).
`run_acceptance.py` writes the executed notebook, PNG figures, daily CSV and
environment/revision provenance under ignored `artifacts/acceptance/` or
`artifacts/milestone2/`. No new dependencies were added for milestone 2.

## Repository guide

| Location | Purpose |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Durable instructions for coding agents |
| [Project context](docs/project-context.md) | Goals, confirmed choices, and open decisions |
| [Implemented API](docs/api.md) | Supported calls, schemas, units, result tables, and limits |
| [Architecture](docs/architecture.md) | Module boundaries and proposals for later phases |
| [Financial conventions](docs/financial-conventions.md) | Accounting, timing, units, costs, metrics |
| [Roadmap](docs/roadmap.md) | Phases and first-release acceptance criteria |
| [Testing plan](docs/testing-plan.md) | Hand calculations and reconciliation checks |
| [Reference review](docs/reference-review.md) | Verified findings and license provenance |
| [examples/](examples/README.md) | Runnable offline examples, snapshots and acceptance notebook |
| [tests/](tests/README.md) | Offline validation, return arithmetic, and ledger tests |
| [data/](data/README.md) | Local snapshot policy; no downloaded data |
| `src/research_toolkit/` | Small public facade backed by focused internal modules |

No reference code has been copied. The reference project's MIT license does not
automatically license this project. This project's license remains an owner
decision before external distribution. Nothing has been published or pushed.
