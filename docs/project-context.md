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
| Implement milestone 2 | Prior user request |
| Implement realistic automatic dividend reinvestment after payment | Prior user request |
| Implement historical daily SOFR financing (interpreting “SOFT” as SOFR) | Prior user request |
| Exclude commission, spread and impact from automatic dividend reinvestment; preserve normal trading costs | Latest user request |
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
- Dividends accrue as receivables on ex-date and become cash on pay date. By
  default, cash repays debt and any excess remains cash. The implemented opt-in
  reinvestment policy can instead reserve paid principal for same-asset purchases;
  the caller explicitly chooses funding priority.
- Fixed nominal financing rates with actual elapsed days/365, or optional
  historical SOFR-plus-spread borrowing with explicit availability/day count; configurable
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
  borrowing, complex corporate actions, multi-currency/FX, additional provider/download adapters,
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
  M2 originally deferred per-share fees and nonlinear impact; the notebook
  extension below now implements them in a separate sizing solver. Fixed/minimum
  ticket fees remain deferred.

The subsequent milestone 3 request below implements dated signals and chronological
research boundaries while retaining these timing and accounting contracts. Real-data research can also begin after
choosing permitted sources and assumptions.

## Payment-date reinvestment extension — implemented

The user requested realistic automatic reinvestment. `DividendReinvestment` now
works with both simulators through the shared ledger. Existing calls retain their
cash/debt behavior. These detailed policies are engineering choices exposed in the
API, not additional user confirmations:

- A standing same-asset instruction executes at the first supplied close on or
  after payment, assuming payment is available before that close. No ex-date credit
  funds purchases. Fractional quantities apply. The subsequent user decision
  exempts these purchases from commission, spread and impact; ordinary trades
  continue to use configured costs.
- Choose `before_debt_repayment` to reserve paid principal (existing debt continues
  accruing), or `after_debt_repayment` to reinvest residual dividend cash. No extra
  loan is taken for reinvestment. Interest on held cash is not added to its budget.
- Scheduled baskets take priority at a coincident close; the dividend earmark is
  released to normal account funding. The final session stays mark-only.
- Reinvestment remains enabled for the payer even if an earlier scheduled basket
  sold that asset; it can reopen a position. Per-asset enrollment/cancellation and
  broker-specific payment timestamps/fills remain future features.
- Per-payment records link cash receipt, earmarked debt repayment, released cash,
  purchases, zero DRIP fees and status. The zero-cost assumption is recorded in
  result metadata. Pre/post-trade margin checks and all reconciliations
  apply. Analytical `reinvest_ex_close` returns remain a separate convention.

## Historical SOFR extension — implemented

The user requested a historical daily SOFR financing option. `SOFRFinancing` works
with both simulators and dividend reinvestment; fixed-rate financing is preserved.
Engineering choices exposed/documented by the implementation (not extra user
confirmations):

- USD borrowing uses a supplied SOFR table plus a nonnegative spread in bps.
  Cash earns a separately configured fixed rate. Both day counts are explicit;
  the new mode supports ACT/360 and ACT/365F.
- The only initial timing policy is `known_at_accrual_start`: choose the latest
  supplied rate available by 00:00 America/New_York on the posted accrual date.
  This intentionally lags observations. Never select on observation date alone.
- Require a separate complete publication calendar with matching rate keys,
  declared calendar coverage and an explicit maximum observation age. Missing
  rows, stale rates, missing initial history and ambiguous units raise.
- Existing calendar-day balance/event order and daily capitalization apply,
  including weekends. This is a configurable research loan convention, not the
  official SOFR Index, retrospective in-arrears fixing or broker billing engine.
- Save all source rate/calendar rows, metadata and their deterministic identity in
  run metadata; a new accrual table ties selected rates/balances to posted interest.
  The example uses synthetic rates; no live provider or historical rate download
  is built into the library. Real inputs need verified point-in-time vintages.

Possible follow-ups are source adapters, vintage reconstruction, SOFR-linked cash,
signed rates/spreads, and other loan compounding/billing conventions. They are not
implicitly supported by this release.

## Notebook reuse extension — user request and implemented choices

The user requested reusable functionality from the private Challenge 1 notebook:
capped inverse volatility, square-root execution costs and sizing, dated risk-free
performance, rolling beta/correlation, cumulative plots, scenario tables and a
small justified offline Yahoo adapter. They require the notebook remain unchanged,
no private prose/caches in the repository, no assignment constants in the library,
Polars tables and no hidden downloads. No push or publication is authorized.

Engineering choices within that request (not extra user confirmations):

- Preserve cap rejection and constant-rate reporting; opt into redistribution or
  supply exactly matched dated RF returns. The notebook uses excess-return Sharpe
  volatility; the toolkit retains portfolio-return volatility by default and
  exposes an explicit denominator choice.
- One frozen daily-volatility/dollar-ADV snapshot feeds a square-root cost object.
  Costs use actual orders and raw reference prices, with per-share commissions
  supported. The engine solves funding and records costs once; a preview is not
  an extra debit. Automatic dividend reinvestment stays cost-free.
- Explicit comparison mismatch modes preserve separate samples/assumptions and
  status, never silently intersecting dates. Plots use prepared numerical tables.
- The small Yahoo adapter consumes supplied decoded chart JSON. Provider Close is
  labeled split-adjusted; raw reconstruction requires explicit cumulative factors
  through retrieval. Authoritative sessions/actions/payment dates are required;
  availability is disclosed as reconstructed and volume units must be declared.
  No original Yahoo payload was present alongside the reference notebook; adapter
  verification uses self-authored synthetic format fixtures.

[Implementation contracts and migration notes](notebook-extensions.md) and the
[synthetic workflow](../examples/research_workflow.py) record these choices. All
implementation is independently written; no private notebook code/prose/data is
copied. Later signal/ML work, dated cost-snapshot refresh, fixed/minimum ticket fees,
live provider acquisition and point-in-time vintage reconstruction remain open.

## Milestone 3 — user request and implemented choices

The user requested implementation of roadmap milestone 3. This adds lagged features,
chronological train/validation/test partitions, label-horizon/availability purging,
training-only standardization, audited validation selection/final-test use, and
dated long-only signal instructions. No short-stock accounting or ML dependency
was requested in milestone 3; the subsequent signed-accounting extension is recorded below.

Engineering choices exposed by this release (not additional user confirmations):

- Positive lags count supplied sessions; feature decisions are at session closes.
  Missing input panels raise. Warmup and late source availability remain visible,
  and later splitting retains all excluded rows with reasons.
- End dates are inclusive outcome/availability cutoffs, with actual per-row label
  horizons and optional boundary gaps. The standardizer pools retained training
  asset rows and uses sample variance; constant features require an explicit policy.
- Basic supplied prediction selection supports MSE/MAE on validation. Final test is
  evaluated only for the selected candidate, with an exportable/resumable audit.
  Repeated test use/reselection stays exploratory. External test access and fresh
  objects without previous audit history cannot be detected.
- Signals explicitly choose next supplied close and each-signal or on-change
  rebalancing. They are already declared weights/leverage, not arbitrary scores.
  The same scheduled ledger executes them and retains processing/stop provenance.
- The new example independently fits two tiny models outside the library, fixes a
  forecast-to-allocation rule, and reports synthetic outputs. No private assignment
  notebook, provider cache, reference code, push or publication is involved.

[Milestone 3 contracts](chronological-research.md) and the
[example notebook](../examples/chronological_research.ipynb) record these decisions.
Next roadmap work is optional walk-forward/model workflows; next-open/intraday
execution and richer execution constraints remain open. The subsequent long/short
accounting extension is recorded below.


## Long/short extension — user decisions and implementation

The user requested signed portfolios, supplied hedges, short accounting, stock-loan
fees, short dividends, existing cost/SOFR support, margin stops and information
ratios, with offline tests and a runnable example. The user confirmed **1A: daily
marked restricted collateral** and **2A: separate long/short maintenance fractions**.
These supersede the earlier deferral of short accounting. No private notebook or
assignment data is changed, and no publishing/push is authorized.

Implemented engineering choices, exposed and documented rather than additional
user confirmations:

- Extend buy-and-hold and scheduled rebalancing on the shared ledger. Signed
  post-cost equity exposures and exact quantities are separate from legacy long
  allocation weights. Exact quantities preserve the original longs when adding a
  caller-specified hedge; incremental fees/collateral can increase the loan.
- Segregate a caller-supplied multiple (at least one) of short market value at
  supplied closes. Released collateral follows the explicit debt sweep; opening
  proceeds do not automatically repay the loan. Side-specific margin replaces the
  old long-only threshold in signed mode.
- Explicit fixed asset borrow assumptions or complete dated calendar-day rates,
  available by New York midnight. Accrue on the prior supplied close's short value.
  Separate gross interest on all restricted collateral from fees, loans and free
  cash interest. Net rebate quotes and borrow availability/recall modeling remain
  unsupported.
- Short dividends accrue liabilities and settle on the provided dates, including
  after covering. Long reinvestment stays cost-free; if the payer has become short,
  release its old long dividend to cash rather than auto-covering.
- Signed scheduled targets require exact-target funding through the explicit loan;
  automatic reserve-based shrinking is rejected. Information ratio extends existing
  performance reporting with aligned active returns and sample standard deviation.
- The new example uses only existing self-authored synthetic data. Signal conversion,
  risk allocation helpers, borrow locates, recalls, broker margin, intraday execution
  and hedge selection/optimization are outside this signed-accounting extension.

See [contracts and notebook migration](long-short.md) and
[the runnable comparison](../examples/long_short_hedge.py).

## Fixed dollar baskets and dated execution models — implemented

User request: support predetermined signed dollar targets and changing pre-decision
liquidity models in one scheduled ledger. The next notebook may rank stocks by
momentum and supply a beta hedge, then scale the **entire** basket, including the
hedge, to $2 million gross against $1 million initial equity. These capital values,
signal rules, assets and hedge selection belong to the application, never defaults.
Signals are intended for next-session-close execution.

Routine engineering choices made within that request:

- Extend the existing signed target schema with `target_notional: Float64`; retain
  `session` for execution and `decision_session` for instruction time. Divide by
  execution raw prices in the engine. Costs affect equity/funding without scaling
  the target. Quantities and equity exposures retain their separate meanings.
- Accept `costs={decision_date: SquareRootImpactCosts(...)}` with exact complete
  decision keys and full asset coverage, including zero exits and hedges. Static
  models are unchanged. Dated models require explicit sample endpoints and cutoff
  metadata and are bound once per basket for sizing and execution.
- Keep `estimate_liquidity` strictly prior-session and its local-midnight volume
  availability cutoff. Do not quietly extend it to decision-day data for an
  after-close signal. Record both model and liquidity identities.
- Preserve explicit `require_target` financing, segregated collateral, side-specific
  research margin, zero-cost long DRIP, SOFR and stopped runs. Plain tables still
  allow an explicitly chosen later supplied session; the notebook chooses the next
  one. The final session is still mark-only.
- Add numerical target/model audits and an offline synthetic notebook. No momentum,
  regression, hedge optimization, downloads or export behavior enters the core.

No new financial choice needed user clarification. See [API and migration](fixed-dollar-rebalancing.md)
and [the runnable notebook](../examples/fixed_dollar_rebalancing.ipynb).


## Independent liquidity-history calendar — authorized adjustment

The user authorized using estimation history that predates the execution market.
Return metadata now carries the validated source session closes, propagated through
liquidity estimation. Dated cost binding validates sample endpoints against that
calendar and checks overlap with execution closes. Existing availability cutoffs,
strictly prior samples, exact decision keys and immutable model identities remain.
Legacy manual estimates without a source calendar retain the original validation.
Offline tests cover truncated execution history and malformed/late calendars.


## Same-session closing execution — user-confirmed change

The user explicitly requested the assignment-style assumption of forming daily
signals and trading at that same close. Added an opt-in RebalancePolicy setting,
`decision_timing="same_session_close_assumed"`, while retaining strict prior-session
decisions by default. No future decisions or preceding-return attribution are
allowed. This supersedes strict precedence only for explicitly opted-in schedules;
source estimation/availability cutoffs and all accounting remain unchanged.


## P&L series, tail risk, Yahoo downloads and exchange calendars — user requests

User-requested: `series_performance` (NAV = initial capital + cumulative net P&L;
returns over previous NAV), signed historical `tail_risk` with independent dollar
tails, and a network `download_yahoo` adapter (standard-library HTTP, bounded
retries/timeouts, inclusive exchange-local dates). User-confirmed: optional
calendar validation through the maintained `exchange_calendars` library, declared
as the `calendar` extra and imported only when a calendar is supplied. Calendars
are explicit identifiers per asset; multi-exchange P&L needs an explicit
union/intersection reporting calendar. Engineering choices: missing completed
sessions raise by default (`incomplete="report"` returns status `incomplete`);
unknown listing dates make only pre-first-bar gaps `coverage_uncertain`;
suspensions need caller-supplied exclusions.

## Audit follow-up — user requests

After a full audit, the user asked for the Medium findings, the integration gaps
and the Low items to be fixed, with two explicit choices: `download_yahoo` keeps
rows in the order the assets were requested (the docs now say so instead of
"sorted by asset"), and `compare_performance` keeps its behaviour for unequal
coverage while its error now names coverage as the cause. Engineering choices made
within that request:

- An asset with no bars raises unless calendar validation explains every session
  (declared coverage, exclusions, or closes after the validation time).
- `ProviderDownloadResult` gains `sessions` (completed sessions per calendar) and
  `responses` (decoded payloads); `yahoo_chart` accepts a download and limits its
  payloads to the requested inclusive sessions before the usual strict checks.
- The `calendar` extra requires `exchange_calendars>=4.5.6`, the first release that
  imports under pandas 3.
- Ledger-only plots raise `ValueError` for P&L-series reports. `rt.__version__`,
  `__all__` for `plots`/`adapters`, one numeric validator family, one local-midnight
  cutoff helper, and the `_Ledger` class (with bit-identical outputs) were added.


## Performance diagnostics, factor regression and signal evaluation — user request

User-requested: general-purpose report diagnostics (CAGR, bias-corrected skewness
and excess kurtosis, Calmar), descriptive factor regressions with residual risk,
and cross-sectional signal evaluation; no assignment-specific assumptions, engine
changes, plots, downloads or commentary. User-specified conventions: explicit
calendar-time versus trading-period CAGR starting at initial capital; idiosyncratic
volatility `sqrt(A*SSE/(n-k-1))`; risk-free subtraction never applied to long/short
factors; no pseudoinverse coefficients; per-date Spearman ICs with average ranks;
equal-weight quantile returns; optional equal-size extreme groups of `floor(n/Q)`;
unexplained signal/outcome mismatches rejected by default.

Engineering choices within that request:

- Names `performance_diagnostics`, `factor_regression`, `signal_diagnostics`, with
  small result dataclasses. Calendar CAGR requires an explicit `days_per_year`;
  neither convention has a default.
- Factor bases are declared per factor (`total_return`, `excess_return`,
  `long_short`); `risk_free="report"` reuses a report's resolved RF returns.
  Pure-Python pivoted Householder QR keeps Polars the only dependency. No standard
  errors are reported.
- Outcomes carry their own `signal_date` key and one forward interval per date.
  Unmatched rows need an explicit `exclusions` reason. Group ties are broken by
  asset identifier or the date is rejected. In `extremes` grouping the middle assets
  form the interior groups (unassigned when Q=2).
