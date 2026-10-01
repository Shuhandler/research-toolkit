# Project context

## Goals and preferences

Jordan is learning quantitative trading through Hedge Fund Strategies and
Algorithmic Trading classes. The toolkit should replace repetitive notebook code
with simple calls while keeping every calculation understandable and auditable.
Support discretionary portfolios first and systematic strategies later. Machine
learning is optional. Data acquisition and simulation must be separable so saved
inputs can run offline.

The reference's convenient `research` import is the inspiration, not a source of
financial authority. Inspect its assumptions independently. Never modify it.

## Confirmed decisions

| Decision | Source |
| --- | --- |
| Distribution `research-toolkit`, import `research_toolkit as rt` | User explicitly accepted the import |
| Polars public table interface, not pandas | User chose 1B |
| Raw prices plus explicit splits and dividend accounting in the first backtest | User chose 2A |
| Borrowing and leverage included in the first milestone | User chose 3B |
| Setup phase creates documentation and minimal scaffolding only; no financial functions | Initial request |
| Implement milestones 1A and 1B; continue to defer 1C/1D | Follow-up user request after setup |
| No publishing, GitHub push, or changes to the reference | Initial request |
| Ask consequential questions in ordinary chat with recommended multiple-choice options | Initial request |
| Narrative interpretation belongs in Markdown after inspecting outputs | Initial request |

## First acceptance application

Five equity holdings, $100 million starting **equity**, one year of daily returns,
a benchmark, entry trading costs, and optional comparisons of initial leverage.
Report daily dollar P&L, daily simple returns, their cumulative sum, compounded
return, ending equity, return correlation, and beta. Produce price, P&L/return,
equity, drawdown, distribution, and allocation views. Weights, capital, assets,
dates, benchmark, and leverage levels are notebook inputs, never library constants.

## Engineering proposals adopted for planning

These are explicit, revisable design choices, not additional user confirmations:

- Python 3.12+, `src/` package layout, setuptools build backend; Polars is now the
  only runtime dependency, with pytest in the test extra.
- Long-form Polars tables with explicit session/asset keys; no implicit index.
- First release: single currency, fractional shares, long positions plus a cash
  loan, fixed initial allocation, ordinary splits and cash dividends. No external
  contributions/withdrawals or implicit liquidation at the end.
- Entry at an explicitly chosen session close from weights decided beforehand.
  Cost-aware sizing targets gross exposure relative to post-entry net equity.
  Portfolio weights and leverage then drift.
- Dividends accrue as receivables on ex-date and become cash on pay date; no
  automatic stock reinvestment. Cash first repays debt under the named sweep policy.
- Fixed nominal financing rates with actual elapsed days/365; configurable
  research margin threshold with stop-on-breach. This is not a broker margin model.
- Strict date alignment by default. Explicit supplied session calendars and source
  metadata. Matplotlib figures initially; interactive plots are a later option.
- Synthetic fixtures first; local snapshots plus manifests for real-data examples.
  Detailed choices and limits live in the conventions and architecture documents.

## Open and deferred choices

- **Before real-data acceptance:** choose assets, dates, benchmark and its return
  basis, data source/redistribution permission, rates, costs, and margin threshold.
  These are example configuration choices, not blockers for synthetic development.
- **Before external distribution:** choose this project's license and verify package
  name availability. Do not assume the reference's license applies to new work.
- **Before later phases:** broker-specific margin/liquidation, lot rules, short
  borrowing, complex corporate actions, multi-currency/FX, provider adapters,
  intraday execution, and interactive plotting. Do not prebuild abstractions for them.

## Status and next step

Milestones 1A and 1B are implemented: strict metadata/calendar/price/action
validation, deterministic input identities, simple/log and cumulative returns,
equal/custom weights, and an unlevered fractional-share ledger with entry costs,
splits, dividend receivables/payments, cash interest, and reconciliations.
An offline synthetic Python example and financial tests are included. No reference
code was reused, market data downloaded, notebook created, package published, or
GitHub changes pushed.

Concrete interface refinements: `returns` takes validated `MarketData` so session
gaps cannot bypass checks; `buy_and_hold` takes explicit `cash_rate` and
`cash_day_count` until financed accounts are added. No placeholder `Financing`,
`performance`, or plotting objects exist. See [the API](api.md).

Next is milestone 1C: borrowing, financed initial leverage, debt sweeps, and
stop-on-margin-breach behavior through the same ledger. 1D will add performance
metrics, benchmarks, plots, snapshot persistence, and the full acceptance notebook.
