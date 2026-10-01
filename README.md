# research-toolkit

A notebook-friendly Python library for quantitative trading research: prepare
market data, construct portfolios, simulate trades, measure performance, and plot
results through a small, consistent interface.

**Status: design and repository setup only. No financial functions, result
classes, plots, or data adapters are implemented.** The package currently contains
only an importable namespace. All financial API examples below and in the design
documents are proposals, not runnable examples.

## Purpose and scope

Built for learning and practical research in Hedge Fund Strategies and Algorithmic
Trading classes. Calculations should remain understandable, reproducible, and
auditable while reducing repetitive notebook code. Polars tables are the chosen
data interface. Machine learning will be optional; ordinary plotting and
backtesting must never require a model framework.

The first release targets daily, single-currency equity buy-and-hold portfolios:
raw prices, explicit splits and cash dividends, equal or custom initial weights,
entry costs, cash, borrowing, and optional initial-leverage comparisons. Saved
inputs must run entirely offline. There is no automatic rebalancing or terminal
sale. This is a research simulator, not a broker execution system.

Later work covers inverse-volatility allocation, scheduled rebalancing, signals,
short positions, richer trading costs, intraday inputs, and chronological research
workflows. See the [roadmap](docs/roadmap.md) for scope boundaries.

## Intended usage — unimplemented

The distribution name is `research-toolkit`; the Python import is:

```python
import research_toolkit as rt
```

The intended workflow is explicit data validation → initial allocation → simulation
→ numerical analysis → plots → notebook interpretation. For example, **after
implementation**, `rt.buy_and_hold(...)` will return position, trade, cost, P&L,
and equity tables, and `rt.performance(...)` will produce reusable metric tables.
The [architecture proposal](docs/architecture.md) contains the full illustrative
calls and data contracts.

The acceptance notebook will use five stocks, $100 million starting equity, one
year of daily observations, a benchmark, entry costs, and optional leverage
comparisons. These are example parameters, never library constants. It must report
daily dollar P&L, daily simple returns, the cumulative sum of simple returns,
compounded return, ending equity, return correlation, and beta, with appropriate
plots. Cumulative sums and compounded returns will have distinct labels.

## Local development

Python 3.12+ is the initial target. From this checkout, the intended installation is:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -c "import research_toolkit as rt; print(rt.__name__)"
```

Run the notebook kernel from that environment. The commands install the local
checkout; do not install an unrelated package from an index by name. The build
backend may need downloading. No runtime dependencies are declared yet because
there are no financial functions. Add Polars with the first calculation slice,
pytest with its tests, and Matplotlib with plotting; do not install an ML stack
in anticipation of future work. Compatibility beyond the local smoke checks must
be established when implementation begins.

Setup verification: an editable install and import succeeded in a temporary Python
3.14.6 environment. Documentation links and package metadata were checked. There
is no implemented financial behavior to test yet.

## Repository guide

| Location | Purpose |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Durable instructions for coding agents |
| [Project context](docs/project-context.md) | Goals, confirmed choices, and open decisions |
| [Architecture](docs/architecture.md) | Proposed modules, contracts, result tables, API |
| [Financial conventions](docs/financial-conventions.md) | Accounting, timing, units, costs, metrics |
| [Roadmap](docs/roadmap.md) | Phases and first-release acceptance criteria |
| [Testing plan](docs/testing-plan.md) | Hand calculations and reconciliation checks |
| [Reference review](docs/reference-review.md) | Verified findings and license provenance |
| [examples/](examples/README.md) | Acceptance notebook specification; no notebook yet |
| [tests/](tests/README.md) | Test organization; no financial tests yet |
| [data/](data/README.md) | Local snapshot policy; no downloaded data |
| `src/research_toolkit/` | Import namespace; internal modules added when implemented |

No reference code has been copied. The reference project's MIT license does not
automatically license this project. This project's license remains an owner
decision before external distribution. Nothing has been published or pushed.
