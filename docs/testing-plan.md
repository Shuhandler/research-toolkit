# Testing plan and implemented checks

Prioritize independent financial examples and ledger identities over coverage
percentages or tests that mirror implementation. Milestone 1A–1D and 2 tests now cover
data validation, return arithmetic, entry costs, actions, cash interest, drift,
scaling, leverage, financing, debt repayment, stopped runs, and accounting identities
using synthetic Polars tables and independent expected values. Performance,
benchmark, snapshot and plotting checks are implemented; chronological research
checks remain targets for later phases. Run `python -m pytest -q`; network connections are
blocked by the unit-test fixture.

## Hand-checkable oracles

| Case | Expected result / bug prevented |
| --- | --- |
| Prices 100 → 110 → 99; one share | Simple returns +0.10, −0.10; sum 0; compound −0.01; P&L +10, −11; ending equity 99 |
| Flat price, no costs/interest | P&L and returns zero; equity constant; volatility zero; Sharpe/Sortino undefined with status, not fabricated zero skill |
| $100 equity, price 100, 1% entry cost, no loan | Notional 99.0099009901, quantity 0.9900990099, cost 0.9900990099, cash/debt zero; flat first holding interval returns −0.0099009901 |
| Same entry, next price 110 | Equity 108.9108910891; first P&L 8.9108910891 relative to original $100; fee counted once |
| $100 equity, 2× leverage, 1% fee | Notional 196.0784313725; fee 1.9607843137; debt = equity = 98.0392156863; post-entry leverage exactly 2 |
| $100 equity, $200 stock, $100 debt, no costs/interest; price +10% | Equity 120, P&L 20, return 20%, leverage 220/120; unchanged quantities |
| $100 loan, nominal rate 0.365, three calendar days | Debt 100.3003001, financing 0.3003001 under daily capitalization; no fictitious trading sessions |
| Ten shares at 100, 2-for-1 split, new price 50 | Twenty shares, value 1,000, zero cash flow/turnover/cost/P&L from split |
| Ten shares at 100; dividend 2; ex-date price 98 | Stock 980 + receivable 20 = equity 1,000; pay date transfers 20 to cash, no second P&L |
| Entry on dividend ex-date close | No entitlement; entry shares do not create a receivable |
| Dividend pays after end date | Receivable stays in ending equity; never silently disappears or becomes premature spendable cash |
| Equal $50/$50 holdings; A 100 → 200 → 100, B flat at 100 | Buy-and-hold ends at 100; weights drift to 2/3 and 1/3 mid-run. Daily rebalancing would end at 112.5, so it must not happen silently |
| All-cash, rates explicitly zero | No trades, no costs, no debt, constant equity; no divide-by-zero from gross exposure |
| Benchmark x = [−0.01, 0.02, 0.03], portfolio = 2x + 0.001 | Pearson correlation 1 and beta 2; test the intercept, not a return ratio |
| $200 stock/$100 debt falls to $150 stock | Equity 50, equity/gross ratio 1/3; a 0.4 threshold stops the run, preserves balances, and prevents a completed full-year report |

Add combined-event cases: split and dividend on the same date with the declared
post-split share basis; pay-date cash repays debt; dividend payment on a weekend;
entry costs plus financing; insolvency with a last simple return at/below −100%.
Costs must be component-level traceable and never repeated on a no-trade day.

## Invariants and reconciliation

- At each valuation, sum of marked positions + cash + receivables − debt equals
  equity. Trades explain quantity changes apart from split events.
- Cumulative daily P&L equals ending equity minus starting equity. Compound growth
  reproduces the same equity, including entry costs. Log/simple conversion agrees
  only on valid positive wealth paths.
- Attribution components sum to daily P&L; cost component sums match ledger
  outflows; every trade cost references a real trade. Financing has its own events.
- Split value neutrality, dividend receivable-to-cash neutrality, and loan
  funding/repayment neutrality hold independently of price gains/losses.
- 1× with zero borrowing rate agrees with the unlevered path; cash-only and zero
  weight assets create no phantom trades. Changing initial weights changes initial
  quantities; changing prices changes realized weights without causing trades.
- With fractional shares and proportional costs, scaling starting capital scales
  dollar outputs but preserves returns (subject to the same margin policy). Test
  more than five assets and capital other than $100 million.
- Permuting input row order does not change canonical results; asset labels and
  timestamp order are never matched by row number. Inputs remain unchanged.

Set and document tolerances by unit and magnitude. Initially use Float64 with
relative tolerance around `1e-12` and currency absolute tolerance around `1e-8`
for tiny fixtures; at $100 million require residuals below one cent, with a tighter
scale-aware check where feasible. Rounded oracle numbers in this document are not
an excuse to weaken actual tests; use rational/Decimal hand calculations for exact
expected values. Fail loudly on reconciliation errors, not with a printed warning.

## Invalid data and metric boundaries

Test duplicate keys, missing held-asset prices, missing benchmark intervals,
different period starts despite equal end dates, unsorted inputs, timezone/calendar
mismatch, non-session rows, nonfinite and nonpositive prices, invalid rates,
negative/sum-mismatched weights, unknown assets/actions, inconsistent adjustment
basis, and missing action coverage metadata. Assert descriptive failures.

Test empty, one-observation, and two-observation inputs; initial structural nulls
versus interior gaps; zero return variance, zero benchmark variance, no downside
observations, and nonpositive ending wealth. Undefined metrics must retain units,
sample counts, and a specific null reason. Never hide an invalid input as a null
metric. Test net simple-return metrics separately from log-series utilities.

Test annualization explicitly: variance scaling, effective-to-period risk-free/MAR
conversion, and rejection of a frequency change with an incompatible convention.
Drawdown must include initial capital and post-entry fees, even before the first
holding-interval close. Do not use a price-level correlation as a return oracle.

## Integration, plotting, and research safeguards

Run a tiny end-to-end offline ledger test before the one-year five-stock notebook.
Replay the same immutable snapshot/configuration and compare numerical tables,
manifest identity, and assumptions. Use one-asset and different-size portfolio
integration cases so the acceptance example cannot hide hard-coded dimensions.

Plot tests inspect returned figure/axes, series values, unit labels, dates, and
customization. Use a headless backend. Do not rely on brittle pixel equality.
Assert plotting does not fetch data, run simulations, change inputs, or import ML.

When chronological research is added, perturb future observations and assert no
earlier feature/decision/trade changes. Verify positive lag direction, availability
times, chronological boundaries, purged overlapping labels, training-only fitting,
and model selection on validation rather than final test. Scheduled rebalancing
must test trade deltas, zero-trade schedules, costs, and holiday execution rules.

Separate network/provider integration checks from unit tests, opt in explicitly,
and never require credentials in CI. Check any redistributed fixture's license.
Run relevant checks after each slice; add broader checks when a new interaction
justifies them. Update this plan and runnable examples when contracts change.

## Implemented milestone 1D checks

The suite now includes independent +10%/−10% P&L and benchmark beta oracles,
effective-rate conversion, Sortino's all-observation denominator, entry loss
recovered before the first closing return, flat/short samples, zero benchmark
variance, invalid conventions, edited ledger rejection, and explicit stopped-run
opt-in. Benchmark date reordering is harmless; missing/duplicate/changed endpoints,
nulls and nonfinite values fail. Correlations preserve the basis and structural-null
contract rather than silently deleting data.

Snapshot tests verify immutable destinations, exact typed round trips, identical
replayed numerical ledgers, file corruption, and altered manifest identity/schema.
Plot tests run headlessly with optional Matplotlib, inspect plotted dates and
numbers, preserve caller axes, check partial labels and negative-equity visibility,
and save every view. A fresh process verifies core import does not load the optional
plotting/NumPy/ML/network stack.

The committed year-long synthetic inputs drive four end-to-end cases (all cash,
1×, 1.5×, 2×). Independent exported-record checks reconstruct market values, balance
sheets, daily dollar attribution, cumulative P&L and complete allocation weights;
entry fees occur only on trades, and unpaid final dividends remain receivables.
The notebook is executed with network acquisition blocked and plots inspected
visually. Its qualitative discussion is manually written Markdown.

Validation environments: Python 3.14.6 / Polars 1.44.2 / Matplotlib 3.11.2 and Python
3.12.11 / Polars 1.30.0 / Matplotlib 3.9.0. Both install the checkout and run the full
suite and notebook. The source notebook stays output-free; the runner writes an
executed artifact with figures and environment/revision provenance. Exact current
acceptance dependencies are pinned separately from consumer requirements.

## Implemented milestone 2 checks

Use independent two-asset hand calculations before trusting the larger monthly
example. A $60/$50 drifted basket rebalanced to 50/50 sells/buys $5 and reports
`10/110` gross turnover; a later 10% A rise earns $5.50. A complete switch with
1% proportional fees leaves `100*0.99/(1.01**2)` after entry/sale/purchase.
Unchanged quantities and pure splits create no spurious trades or repeated fees.
Asymmetric commission/spread rates match actual changed notionals.

Tests reconstruct each closing quantity, cash, debt, receivable, market value and
attribution from independent exported events/records, including a $10 million
monthly five-asset run. They cover leverage increases/decreases, all-cash targets,
concentration drift without hidden trades, dividends earned before selling, and
reserve versus rejected unfunded targets. A pre-trade margin breach must stop
before scheduled deleveraging, and full/rolling reports preserve the stop guard.
An initial-only schedule matches buy-and-hold's financial tables exactly.

Inverse-volatility tests use known 1:2 volatilities and expect 2/3:1/3 proportions;
concentration breaches raise without clipping. Zero-volatility assets and inadequate
windows raise. Covariance contributions reconcile to portfolio volatility, preserve
negative offsets, and return explicit nulls for zero portfolio volatility. Changing
returns on/after a decision cannot change that allocation or risk estimate; changing
future marks cannot change past scheduled fills. Raw total-return tests verify
splits/dividends, explicit reinvestment semantics and the cash-held ledger difference.

Rolling tests check full-window warm-up, sample counts, flat windows, and partial
coverage. Headless plot tests inspect dates/values and reuse caller axes. The M2
notebook is executed with acquisition blocked and six embedded charts, with manual
visual inspection and separately written Markdown interpretation. Both supported
validation environments continue to run M1 regression checks.

## Automatic reinvestment checks — implemented

`test_dividend_reinvestment.py` verifies payment-date fractional shares and full dividend
budgets with zero DRIP fees against small hand calculations, price P&L before/after a fill, same-close
ex-date eligibility and subsequent dividend compounding, weekend receipt followed
by a split, multiple paying assets and pro-rata debt allocation, both funding
priorities, financing/cash-interest separation, terminal/unpaid distributions,
scheduled collisions and reopening a sold payer. It independently reconstructs
positions/cash/debt/receivables and attribution from events, checks zero DRIP costs
while preserving entry and later rebalancing fees, checks trade-cost links
and turnover, perturbs future marks, and tests pre/post-purchase margin stops and
partial reports. Disabled/no-dividend behavior preserves existing financial tables.
The saved five-asset example also reconciles; plotting remains optional and offline.

## Historical SOFR checks — implemented

`test_sofr.py` uses offline synthetic rates and Decimal loan oracles: ACT/360
spread accrual, calendar-day capitalization, publication lag, explicit holiday
carry, DST cutoff conversion, equality at the publication cutoff, separate cash
day count, and constant-curve equality with existing ACT/365F financing. Perturbing
future published rates cannot change prior rows. Missing expected fixings, missing
seed history, stale observations, bad coverage/schema/units/currency/vintage,
nonfinite/negative inputs and mutated snapshots raise. Scheduled deleveraging,
dividend reinvestment and stops retain shared accounting. Each accrual is matched
to actual cost/events and balance-sheet reconciliation. Embedded source metadata
reconstructs an identical input identity and replay. No tests download rates.

## Notebook reuse extension checks

New offline tests cover:

- Water filling with one/multiple binding caps, equal feasible caps, infeasible
  capacity, permutation invariance and unchanged pre-decision cutoffs.
- Sample daily volatility and mean dollar ADV on exact windows, availability
  rejection, explicit units and liquidity mutation detection.
- Independent cost oracle: a $100 order at $10/share, 2 bps commission plus
  $0.01/share, 3 bps half-spread, daily sigma 0.02, dollar ADV 10,000, eta 0.5
  yields $0.12 commission + $0.03 spread + $0.10 impact = $0.25 total. Quadrupling
  notional multiplies impact dollars by eight; sells cost the same and zero orders
  cost zero.
- A $100 purchase with sigma 0.1, ADV 100, eta 1 costs $10: capital 110 at 1x,
  capital 60 at 2x, and capital 210 at 0.5x all buy $100. Ledger cash, debt, fees,
  positions, attribution and first-return denominators reconcile independently.
- Nonlinear scheduled drift, switches, leverage changes, liquidation, receivable
  reserve, split-only/no-trade baskets, all-cash and stopped runs, and free DRIP.
- Returns 10%, 20%, with RF 0%, 5%, A=2 give Sharpe 2.5 with portfolio volatility
  and 5 with excess volatility. Matching RF yields undefined zero excess risk.
- Three calendar rates .36/.72/.36 under ACT/360 give compounded return .004005002
  or simple .004; missing weekend dates and late observations raise.
- Rolling beta 2/correlation 1, insufficient history, flat benchmark, exact endpoint
  alignment, cumulative sum versus compounding/wealth, prepared-data-only plots.
- Comparison units, one shared benchmark, incompatible conventions, unequal and
  stopped coverage, explicit mismatch diagnostics, and retained actual stop dates.
- Synthetic Yahoo payloads: split-adjusted/total-return/raw conversion, volume
  declarations, later splits outside the sample, explicit pay dates, source hashes,
  reconstructed availability, missing/null/duplicate bars and unsupported actions.

No original provider cache is a fixture. The adapter's provider-format tests are
synthetic; these do not validate a real vendor's historical revisions or omitted
corporate actions. Core imports remain free of optional plotting/provider/ML stacks.

Verification for this extension: 277 offline tests pass on Python 3.14 / Polars
1.44.2 / Matplotlib 3.11.2 and Python 3.12 / Polars 1.30 / Matplotlib 3.9.
The older plotting stack emits dependency deprecation warnings. The synthetic
workflow executes, its preview costs match ledger costs, and exported cumulative
and rolling-beta figures were inspected. The private reference notebook's SHA-256
is unchanged. No private data fixtures, downloads, pushes or package publication
were used.

## Milestone 3 validation

Use synthetic daily calendars, explicit publication times and hand-checkable labels.
Lagged values cross weekends by session count, preserve null warmup and delayed
availability, and never mix assets. Missing/null/duplicate inputs and noncausal
publication timestamps raise. Future-value perturbations leave earlier features
and training-only means/scales unchanged. An independent transform oracle uses
training values 1,2,3,3,4,5: mean 3, sample scale sqrt(2).

Chronological tests cover variable label horizons, delayed label publication,
inclusive cutoff equality, explicit session gaps, trailing test purges, empty
post-purge partitions and immutable-result validation. Validation predictions must
match only validation keys. Test keys cannot select a winner; candidate changes,
unavailable forecasts, incomplete panels and default tied losses raise. Audit
restore preserves prior test use; repeated/reselected/outside-inspected tests
stay exploratory, including across copied result snapshots.

Signal oracles include a Friday decision at price 100, Monday fill at 200 and
Tuesday mark 220: $100 capital buys 0.5 shares and earns $10, never the prior jump.
Explicit each-signal versus on-change baskets differ under price drift. Cash
entries/exits, nonlinear costs, unchanged quantities, concentration breaches,
calendar mismatch, missing/late decisions and the final mark-only session are
checked. Signal and plain-target ledgers are identical apart from audit metadata;
stopped signal plans retain blocked/unreached states. Existing event/accounting
reconciliations and free automatic dividend purchases also apply.

Run the offline end-to-end Python example and milestone 3 notebook, inspect plotted
outputs and audit tables, and retain core-import checks for optional dependencies.
No fitted model/backend dependency or network fixture is required.

Milestone 3 verification: 325 offline tests pass on Python 3.14/Polars 1.44.2
and Python 3.12/Polars 1.30 (48 new checks plus all previous regressions). The
older Matplotlib stack emits existing dependency deprecation warnings. The Python
workflow and notebook execute; the executed notebook contains three figures.
Equity, allocation and drawdown charts were inspected against the numerical output.
Notebook execution required local Jupyter kernel sockets; no external data access
was used. `git diff --check` passes, and no new dependencies were introduced.
