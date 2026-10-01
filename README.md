# research-toolkit

A notebook-friendly Python library for quantitative trading research: prepare
market data, construct portfolios, simulate trades, measure performance, and plot
results through a small, consistent interface.

**Status: milestones 1A and 1B implemented.** Strict Polars input validation,
return arithmetic, and an unlevered buy-and-hold ledger are available. Borrowing,
performance ratios, benchmark comparisons, plotting, and data adapters remain
unimplemented. See the [implemented API](docs/api.md) for the supported contracts.

## Purpose and scope

Built for learning and practical research in Hedge Fund Strategies and Algorithmic
Trading classes. Calculations should remain understandable, reproducible, and
auditable while reducing repetitive notebook code. Polars tables are the chosen
data interface. Machine learning will be optional; ordinary plotting and
backtesting must never require a model framework.

The current library supports daily, single-currency equity buy-and-hold portfolios:
raw prices, explicit splits and cash dividends, equal or custom initial weights,
entry costs, cash holdings, and calendar-day cash interest. Borrowing and leverage
comparisons follow in milestone 1C. Saved
inputs must run entirely offline. There is no automatic rebalancing or terminal
sale. This is a research simulator, not a broker execution system.

Later work covers inverse-volatility allocation, scheduled rebalancing, signals,
short positions, richer trading costs, intraday inputs, and chronological research
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

The [offline example](examples/unlevered_buy_and_hold.py) supplies a complete small
synthetic portfolio with a split, dividend, and entry costs. All quantities and
financial policies are explicit. The larger [architecture](docs/architecture.md)
also contains planned APIs; `rt.performance` and `rt.plots` are not available yet.

The acceptance notebook will use five stocks, $100 million starting equity, one
year of daily observations, a benchmark, entry costs, and optional leverage
comparisons. These are example parameters, never library constants. It must report
daily dollar P&L, daily simple returns, the cumulative sum of simple returns,
compounded return, ending equity, return correlation, and beta, with appropriate
plots. Cumulative sums and compounded returns will have distinct labels.

## Local development

Python 3.12+ is the target. From this checkout:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python examples/unlevered_buy_and_hold.py
```

Run the notebook kernel from that environment. The commands install the local
checkout; do not install an unrelated package from an index by name. The build
backend may need downloading. Polars is the only runtime dependency; pytest is an
optional test dependency. Core calculations do not import a plotting or ML stack.
Tests and the example use synthetic inputs and run offline. The implementation was
validated on Python 3.14.6 with Polars 1.44.2 and Python 3.12.11 with the declared
minimum Polars 1.30.0. A broader platform matrix remains a 1D release check.

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
| [examples/](examples/README.md) | Runnable offline example and future acceptance notebook specification |
| [tests/](tests/README.md) | Offline validation, return arithmetic, and ledger tests |
| [data/](data/README.md) | Local snapshot policy; no downloaded data |
| `src/research_toolkit/` | Small public facade backed by focused internal modules |

No reference code has been copied. The reference project's MIT license does not
automatically license this project. This project's license remains an owner
decision before external distribution. Nothing has been published or pushed.
