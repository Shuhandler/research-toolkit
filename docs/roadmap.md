# Implementation roadmap

Milestones 1A–1D are implemented and tested. The offline synthetic acceptance
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

## Milestone 2 — allocation and scheduled rebalancing

Add inverse-volatility weights from pre-decision trailing windows, concentration
limits, scheduled target weights, turnover, exposures, covariance risk contributions,
and rolling-risk/allocation views. Generate real trades from quantity changes and
keep all financing/action accounting from M1. Add fixed/per-share/minimum fees or
capacity-sensitive impact only alongside the corresponding sizing tests.

**Acceptance:** no trade/cost when quantities do not change; drift differs from
target weights; schedule/holiday policy is explicit; cash/cost funding reconciles;
inverse-volatility inputs exclude future observations; allocation limits cannot
silently alter user weights. Estimated risk contributions and realized attribution
remain distinct.

## Milestone 3 — signals and chronological research

Add lagged features, chronological train/validation/test boundaries, horizon-aware
purging, dated signals, and an explicit next-supported-execution policy. Do not
couple signal generation to position accounting. Add long/short only with stock
loan, dividend obligations, collateral, and financing tests; it can be a separate
subrelease. Unsupported negative weights still raise until then.

**Acceptance:** changing future observations cannot alter earlier decisions/trades;
no same-bar close signal earns a past return; transforms fit only training data;
selection uses validation, and final-test reuse is recorded as exploratory.

## Milestone 4 — optional extensions

Walk-forward/model comparisons with an untouched final holdout; optional provider
adapters; intraday UTC bars with availability/execution times; advanced allocations;
multi-currency and complex corporate actions; richer measured spread/impact models;
optional interactive plotting. Prioritize an actual research need before adding a
dependency or framework. Each extension retains the same reconciliation and
provenance obligations. None is required to plot or backtest a basic portfolio.
