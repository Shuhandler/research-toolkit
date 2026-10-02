# Implementation roadmap

Milestones 1A–1D, 2 and 3 are implemented and tested. The offline synthetic acceptance
notebook passes the complete financed, corporate-action-aware workflow. This is a
research release; real-data selection/licensing remains a separate user decision.
See [the API](api.md) and [examples](../examples/README.md).

## Milestone 1 — daily equity buy-and-hold with financing

Current validation covers Python 3.12.11/Polars 1.30.0 and Python 3.14.6/Polars
1.44.2, including editable installation, optional plotting, and the offline acceptance notebook.
The minimum Matplotlib 3.9.0 is checked with Python 3.12; current 3.11.2 with Python 3.14.

Limit scope to one currency/calendar, fractional long shares, equal/custom initial
weights, raw closes, ordinary splits/cash dividends, proportional entry costs,
fixed financing rates, and explicit initial gross leverage. Include single-asset
and all-cash cases. Use a caller-supplied calendar and offline inputs. Do not add
signals, scheduled rebalancing, shorts, provider dependencies, optimization,
intraday support, or ML to this milestone.

### 1A. Data contracts and return arithmetic — implemented

- Add Polars and pytest only as needed. Implement strict daily table/key validation,
  metadata/action-coverage checks, expected-session alignment, and informative
  errors on missing marks, duplicates, and incompatible bases.
- Implement simple/log returns and separately named cumulative sum, compounded
  return, and wealth transformations on explicitly labeled input series.
- Add tiny hand-calculated fixtures for +10%/−10%, initial nulls, single observations,
  flat prices, mismatched dates, and invalid values. Do not use the reference code
  as an expected-value oracle.

**Exit:** correct documented Polars schemas; inputs unchanged; invalid cases fail
explicitly; return identities pass; no acquisition, plotting, or model imports.
An action table can be validated here before any simulator exists.

### 1B. Unlevered ledger, actions, and entry costs — implemented

- Implement equal/custom allocation, post-cost sizing, fractional share quantities,
  pre/post-entry valuations, positions, trades, and costs. Specify the first-return
  denominator before implementing it.
- Implement splits, ex-date receivables, pay-date cash, cash interest, drift in
  weights, daily P&L, and full balance-sheet/attribution reconciliations.
- Keep terminal holdings marked, not liquidated. Add no implicit stock dividend
  reinvestment. Store resolved assumptions and input manifest identity.

**Exit:** all-cash and multi-asset cases, entry costs, action timing, unpaid
receivables, and no-silent-rebalance tests pass with currency residual tolerances.
This is an internal slice, not completion of the user's first milestone.

### 1C. Borrowing, leverage, and financing — implemented

- Extend the same ledger with explicit debt, initial leveraged sizing, daily
  calendar-date financing, debt-repayment sweep, and leverage drift.
- Require explicit research margin settings; stop on maintenance breach or
  nonpositive equity and preserve the ledger with incomplete status.
- Test interest through weekends/pay dates, the initial fee funding equation,
  1× equivalence, and losses that make the requested full run impossible.

**Exit:** leveraged runs reconcile and cannot be produced by scaling an unlevered
return series. Financing charges and debt changes are separately inspectable.
No successful full-period report can conceal a stopped run.

### 1D. Performance, plots, and acceptance notebook — implemented

- Add performance tables for P&L/returns, drawdowns, volatility, Sharpe, Sortino,
  return correlation and beta; record units, annualization, sample counts, and
  undefined-metric reasons.
- Add Matplotlib support when implementing plots. Plot precomputed price,
  P&L/return, equity, drawdown, distribution, allocation, asset-return correlation,
  and dollar attribution tables. Keep formatting separate from calculations.
- Create a replayable local snapshot manifest and the acceptance notebook described
  in [examples](../examples/README.md). Add small synthetic permitted inputs for an
  offline end-to-end test; do not require redistribution of licensed market data.
- Recheck install/import and tests on Python 3.12 and the chosen current development
  version for the complete 1D release. Pin the acceptance
  environment for replay without freezing every future consumer's environment.

**Milestone 1 acceptance criteria:**

1. Five stocks, $100 million initial equity, entry valuation plus one calendar year
   of daily holding intervals, benchmark, entry costs, and optional financed
   leverage comparisons are notebook configuration, not library constants.
2. Raw closes plus supplied splits/dividends drive actual share, cash, receivable,
   and debt records; benchmark return/reinvestment basis is visible.
3. Reports include daily dollar P&L, daily simple returns, their cumulative sum,
   compounded return, ending equity, return correlation and beta, with readable
   figures and inspectable result tables.
4. Every valuation reconciles. Cumulative P&L equals ending equity minus starting
   capital, and compounded returns reproduce the same ending equity. The first
   interval includes entry costs once; a pure split or dividend payment creates
   no artificial profit. Holding without trades creates no transaction charges.
5. Different initial leverage scenarios rerun the ledger with stated financing
   and margin assumptions, not return multiplication. Every accepted full-year
   comparison completes; stopped scenarios are shown separately with actual coverage.
6. Portfolio and benchmark intervals join by dates; missing data fails clearly.
   A different number of assets/capital amount works with the same functions.
7. Saved inputs run offline deterministically. Core imports need no ML/network
   credentials. Plots return customizable objects without rerunning analysis.
8. Notebook narrative is written after inspecting actual tables and plots. The
   README/API documentation describe only implemented behavior as runnable.

## Milestone 2 — allocation and scheduled rebalancing — implemented

Implemented pre-decision inverse-volatility allocation with concentration rejection,
explicit dated target baskets through the shared financed ledger, gross turnover,
exposure/allocation views, covariance risk contributions and rolling net-return
risk. The [offline notebook](../examples/scheduled_rebalancing.ipynb) and
[Python workflow](../examples/scheduled_rebalancing.py) compare actual monthly trades
with unchanged initial holdings. No new required dependencies.

M2 originally deferred per-share commissions and nonlinear impact. The notebook
reuse extension now supports these with corresponding sizing tests. Fixed/minimum
ticket fees and volume-constrained execution remain unsupported.
M2 uses the existing configurable proportional commission/spread/impact models
on actual changed quantities, including sales.

**Acceptance:** no trade/cost when quantities do not change; drift differs from
target weights; schedule/holiday policy is explicit; cash/cost funding reconciles;
inverse-volatility inputs exclude future observations; allocation limits cannot
silently alter user weights. Estimated risk contributions and realized attribution
remain distinct. These checks pass, including initial-only equality with
buy-and-hold, future-observation perturbation, independent event reconstruction,
pre-trade margin stops, and unpaid-dividend reserve/rejection behavior.

## Milestone 3 — signals and chronological research — implemented

Daily lagged features retain availability and structural warmup. Explicit
train/validation/test boundaries purge actual label horizons and delayed label
publication, with optional session gaps and inspectable exclusions. The concrete
standardizer fits retained training rows only. `ResearchStudy` selects supplied
predictions on validation, evaluates only the selected candidate on test, and
records repeated test use or post-test reselection as exploratory in an exportable
and resumable audit. It is not a model-training/search framework.

`signal_targets` maps dated long-only instructions to the next supplied session
close. Each-signal versus on-change rebalancing is explicit. Passing its result to
`scheduled_rebalance` preserves timing/source audits and uses the existing ledger,
costs, financing, corporate actions, DRIP and stop rules. Long/short remains a
separate subrelease requiring stock loan, dividend liabilities and collateral;
negative weights still raise.

**Acceptance:** future-value perturbations cannot alter earlier features, fitted
training parameters or prior trades; no signal earns the move into its fill;
source/label publication cutoffs and horizon purges are checked; validation/test
keys cannot be interchanged; resumed audit history cannot reset test use. The
[synthetic workflow](../examples/chronological_research.py) and
[notebook](../examples/chronological_research.ipynb) demonstrate the complete path.
See [implemented contracts and limitations](chronological-research.md).

## Milestone 4 — optional extensions

Walk-forward/model comparisons with an untouched final holdout; optional provider
adapters; intraday UTC bars with availability/execution times; advanced allocations;
multi-currency and complex corporate actions; richer measured spread/impact models;
optional interactive plotting. Prioritize an actual research need before adding a
dependency or framework. Each extension retains the same reconciliation and
provenance obligations. None is required to plot or backtest a basic portfolio.

## User-requested post-M2 extension — implemented

Payment-funded automatic dividend reinvestment now works in both simulators.
Explicit policies select debt priority, first-close timing, scheduled-basket
priority and terminal cash treatment. Per user decision, automatic fractional
dividend purchases exclude trading costs; ordinary entry/rebalancing remain costed;
per-payment audits and pre/post-trade margin checks reconcile through the existing
ledger. Broker-specific DRIP fills/timestamps, tax withholding, per-asset enrollment
and fixed/minimum fee models remain outside this extension. Milestone 3 adds
explicit signal-driven target instructions.

## User-requested historical SOFR extension — implemented

The optional `SOFRFinancing` policy supplies USD historical benchmark-plus-spread
borrowing to both simulators, with explicit publication timing, calendar coverage,
rate age, day counts and per-calendar-day accrual audits. Fixed-rate financing
remains supported. Offline tests verify known-rate selection, actual posting and
reconciliations. Source adapters, historical vintage reconstruction, SOFR-linked
cash rates, signed rates/spreads and alternate interest billing are deferred.

## Notebook research reuse extension — implemented

Capped inverse-volatility redistribution, pre-decision liquidity estimates,
square-root impact/per-share costs and funded sizing, dated RF reporting, explicit
Sharpe denominator, rolling beta/correlation, prepared cumulative plots, numerical
scenario comparisons and a small offline Yahoo chart adapter are implemented.
See [the contracts](notebook-extensions.md) and
[synthetic workflow](../examples/research_workflow.py).

Acceptance focuses on feasible weights and timing, independent entry-cost oracles,
actual-trade reconciliation (including nonlinear scheduled orders), free automatic
dividend purchases, strict interval matching, calendar-day rate availability,
undefined metric statuses, partial-run/mismatch guards and provider adjustment
semantics. Assignment notebook migration is intentionally a separate task; the
private reference remains unchanged. Milestone 3 signals were implemented separately
under the subsequent user request.
