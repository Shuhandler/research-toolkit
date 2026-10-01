# Testing plan and implemented checks

Prioritize independent financial examples and ledger identities over coverage
percentages or tests that mirror implementation. Milestone 1A–1D tests now cover
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
