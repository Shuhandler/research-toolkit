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
| Implement milestone 1C | Subsequent user request |
| Implement milestone 1D, including reports, plots and offline acceptance notebook | Prior user request |
| Implement milestone 2 | Latest user request |
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
  only required runtime dependency, with pytest in the test extra and optional
  Matplotlib/notebook extras.
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

Milestones 1A–1D and 2 are implemented: strict metadata/calendar/price/action
validation, deterministic input identities, simple/log and cumulative returns,
equal/custom weights, and a fractional-share ledger with entry costs, splits,
dividend receivables/payments, cash interest, borrowing, financing charges,
debt repayment, margin/insolvency stops, and reconciliations.
Performance/benchmark tables, optional Matplotlib plots, hashed local snapshot
persistence, and the executed five-stock acceptance notebook are included. The
self-authored fixture covers one calendar year of synthetic weekdays, with five
raw equity series, splits/dividends and a separate hypothetical total-return index.
It is not historical market data or an actual exchange calendar. The 1×/1.5×/2×
scenarios complete and reconcile. No reference code was reused, market data
downloaded, package published, or GitHub changes pushed.

Concrete interface refinements: `returns` takes validated `MarketData` so session
gaps cannot bypass checks. `buy_and_hold` accepts an explicit `Financing` object;
the earlier `cash_rate`/`cash_day_count` pair remains valid for unlevered calls,
but mixing the two configurations raises. The named `cash_sweep` belongs to
`Financing`, rather than changing the existing execution policy. Stopped results
retain the failure close, reason/time, and requested versus actual coverage;
`require_complete()` guards full-period consumers. Margin equality uses a relative
1e-12 roundoff tolerance, separate from currency reconciliation tolerances.
The 1D report requires explicit annualization, risk-free/MAR rates and benchmark
metadata; strict alignment matches both endpoints. Reports require completion or
explicit partial opt-in, and plots consume prepared result objects so coverage and
stop reasons stay visible. Snapshot persistence uses Polars Parquet plus a versioned
JSON manifest; the source data identity is distinct from file hashes. These are
engineering refinements of the accepted design. See [the API](api.md).

Milestone 2 adds inverse-volatility weights, target concentration rejection,
scheduled baskets through the same ledger, turnover, exposures, covariance risk
contributions, and rolling risk. The example uses the existing synthetic snapshot
for actual monthly trades versus buy-and-hold; both runs complete and reconcile.
No additional dependency, reference code, provider access or publication was needed.

Engineering decisions made within M2 (not additional user confirmations):
- Allocation/risk windows end strictly before their declared decision session;
  scheduled close executions follow the decision. All dates must be supplied
  sessions; there is no inferred holiday roll. The terminal session is mark-only.
- Concentration is a scalar maximum risky proportion at each target. Breaches
  raise without clipping; later drift does not itself cause trades.
- Required receivable policy: `reserve` explicitly caps unlevered risky spending
  when equity includes unpaid dividends; `require_target` rejects an unlevered
  basket requiring debt. Leveraged baskets use declared financing. Pre-trade
  breaches stop before a schedule can conceal them.
- Add explicit analytical ex-date-reinvested raw total returns for allocation;
  executable positions retain actual receivable/payment accounting.
- Turnover counts gross buys plus sells over pre-trade equity, with entry separate.
  Fixed/per-share/minimum fees and nonlinear impact remain cost-model extensions;
  they are not approximated by this proportional-cost sizing solver.

Next is milestone 3: dated signals and chronological research boundaries, retaining
these timing and accounting contracts. Real-data research can also begin after
choosing permitted sources and assumptions.
