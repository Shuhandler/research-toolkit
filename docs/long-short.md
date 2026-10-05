# Long and short portfolios

Implemented through the same ledger as long-only buy-and-hold and scheduled
rebalancing. All prices remain raw, calendars explicit, and tables Polars. There
are no downloads, broker connections, automatic exports, or hedge optimizers.

## Notebook migration

Existing `weights=...` calls are unchanged: those weights are nonnegative
proportions within a long allocation. For signed portfolios, use **one** of:

- `equity_exposures={"LONG": 1.5, "HEDGE": -0.3}`: signed asset values divided by
  equity **after trade costs**. These are neither normalized nor multiplied by a
  separate leverage setting. Gross exposure is 1.8 times post-cost equity; net is 1.2.
- `quantities={"LONG": 150., "HEDGE": -30.}`: exact signed shares. Costs affect
  equity and funding, never resize the supplied positions. This is the option for
  adding a hedge while preserving existing long holdings. Both forms also accept
  a Polars table with `asset: String` and `equity_exposure: Float64` or
  `quantity: Float64` respectively. Include zero positions for later scheduled exits.

Both modes require explicit `long_short`, `stock_borrow`, and `financing` objects.
Keep `BuyHoldPolicy.initial_gross_leverage=1.0`; the signed inputs themselves set
exposure. Set the financing object's `maintenance_equity_ratio=None` because the
new side-specific margin rule replaces it. No conflicting margin policy is ignored.

```python
import research_toolkit as rt

short_policy = rt.LongShortPolicy(
    collateral_multiple=1.02,
    long_margin=0.25,
    short_margin=0.30,
    rebate_rate=0.0,                 # Gross annual nominal interest on collateral.
    rebate_day_count="ACT/360",
)
borrow = rt.StockBorrow(
    rates={"HEDGE": 0.02},          # Annual nominal decimal fee, not a verified quote.
    day_count="ACT/360",
    metadata={"source": "my explicit borrow-fee assumption", "basis": "modeled"},
)
financing = rt.Financing(
    cash_rate=0.0, borrowing_rate=0.06, day_count="ACT/365F",
    maintenance_equity_ratio=None, on_breach="stop", cash_sweep="repay_debt",
)
result = rt.buy_and_hold(
    market, quantities={"LONG": 150.0, "HEDGE": -30.0},
    initial_capital=10_000.0, entry_session=start, end_session=end,
    policy=rt.BuyHoldPolicy(
        execution="entry_close", sizing="post_cost_equity", fractional_shares=True,
        initial_gross_leverage=1.0, terminal_action="mark_only",
    ),
    costs=rt.TradeCosts(commission_bps=2.0, half_spread_bps=3.0, impact_bps=1.0),
    financing=financing, long_short=short_policy, stock_borrow=borrow,
)
result.require_complete()
```

The numbers above are example assumptions, not defaults. All policy fields are
required. `collateral_multiple >= 1`; each maintenance fraction is in `(0, 1]`.
Rates must be finite nonnegative decimals. Both new day-count fields accept
`ACT/360` or `ACT/365F`. Historical `SOFRFinancing` remains supported as `financing=`
with its existing publication, age, coverage, cash-rate, and accrual contracts.

To preserve a previously simulated long entry, extract its entry-session
`asset, quantity` rows, add the caller's hedge quantity, and run the same capital
and dates. The [runnable example](../examples/long_short_hedge.py) does this. Additional
hedge costs can increase borrowing: identical long quantities do **not** imply
identical post-cost equity, debt, or leverage. This simulates a new comparison run;
it does not splice a hedge into an existing result or treat short proceeds as capital.

## Accounting and collateral

Let `L` be positive long market value, `S` absolute short market value, `C` free
cash, `K` restricted collateral, `R` dividend receivables, `D` dividend liabilities,
and `B` debit borrowing. Then:

```
equity = L - S + C + K + R - D - B
gross exposure = L + S
net exposure = L - S
gross leverage = (L + S) / equity
required collateral = collateral_multiple * S
required margin equity = long_margin * L + short_margin * S
```

A negative quantity already contributes `-S` to signed market value. The reported
`short_liability` is a disclosure of that same amount, **not** another deduction.
Restricted collateral is an asset included once. Opening a short exchanges a share
liability for cash and creates no equity; only its trading costs reduce equity.

At entry and after every supplied session close, segregate the required marked
collateral. A price rise requires a transfer from free cash, with any deficit
funded by an explicit debit loan. A price fall or cover releases collateral; the
released cash follows the existing `repay_debt` sweep. The sale proceeds themselves
never enter a debt sweep before the required collateral is set aside. A multiple
above one locks additional account capital and can increase the loan.

Trade baskets are atomic close transactions: release the old collateral within
the basket, execute sales before buys, charge actual costs, establish the new
collateral requirement, then sweep free cash. Intermediate basket cash is not
available for another strategy. There is immediate settlement and no intraday
margin model. Between sessions, collateral stays at the previous supplied close;
weekends do not create new price observations.

With $100 equity, a $200 long and $100 short, a collateral multiple of one leaves
$100 restricted cash, $100 debt, and zero free cash. Equity is
`200 - 100 + 100 - 100 = 100`. Adding the short has not repaid the long loan.
If the short falls to $90 and the long stays flat, equity rises to $110. Releasing
$10 collateral then reduces debt to $90 without creating a second profit.

## Sizing, costs, and schedules

For signed equity targets `u_i`, current signed values `v_i`, and pre-trade equity
`E`, solve `x + sum(cost_i(u_i*x - v_i)) = E` for positive post-cost equity `x`.
For exact quantities, trade the difference between supplied shares and actual
holdings; pay costs separately. Fixed `target_notional` amounts instead divide by
the execution raw close; costs are funded separately. No signed mode normalizes
the basket to sum to one.
Exposure sizing retains a monotonic-cost condition; extreme configurations that
cannot guarantee a unique solve are rejected. Insolvent cost funding also raises.

Commission, spread, square-root impact, and per-share commissions use the absolute
quantity/notional actually traded. Crossing from -10 to +10 shares is a 20-share
purchase. Reference fills stay at the raw close; costs are separate expenses, not
an additional adverse fill adjustment. Preview calculations never post a debit.
Cost-estimate information cutoffs remain strictly separate from execution prices.

For `rt.scheduled_rebalance(..., long_short=..., stock_borrow=...)`, supply exactly:

| Column | Type |
| --- | --- |
| `decision_session` | Date |
| `session` | Date |
| `asset` | String |
| `equity_exposure`, `quantity` **or** `target_notional` | Float64 |

Every basket covers the same universe, including zero exits. Decision sessions
strictly precede execution sessions. All dates must be supplied sessions, the
first basket executes at entry, and no basket executes at the final session.
Quantity targets refer to shares in the units at their execution date; splits do
not rewrite the caller's future instructions.

Use the existing `RebalancePolicy` with `receivable_policy="require_target"`.
In signed mode this explicitly executes the requested target using the declared
loan, including funding against unavailable dividend receivables; `reserve` is
rejected rather than silently shrinking a signed basket. The target concentration
limit remains absolute asset value divided by gross exposure. It does not constrain
later drift. Quantity and dollar baskets report null `target_gross_leverage`, since they do
not target a post-cost leverage ratio. Existing long-only schedules and their
reserve/require-target semantics are unchanged. `SignalResult` still supports
long-only signals; signed callers supply the dated target table directly.

## Borrow fees, financing, and rebates

Four separate streams are retained:

1. Debit financing on opening loan balances, using fixed rates or historical SOFR.
2. Cash interest on unrestricted cash only, under the existing financing policy.
3. Stock-borrow fees on each asset's outstanding absolute short value.
4. Gross collateral interest/rebate on opening restricted collateral, under
   `LongShortPolicy.rebate_rate`; paid into free cash.

The rebate is **gross, never net of borrow fees**. A broker's net rebate quote
cannot be supplied as gross interest while also charging the embedded borrow fee.
Both amounts must be explicitly decomposed for this model. Rebate applies to the
entire collateral balance, including any amount above the short value; the library
does not model a separate interest tier for that extra collateral.

Borrow fees post on every calendar date after entry through the actual ending
session, including weekends. Each uses the previous supplied closing short market
value and that calendar date's rate, divided by 360 or 365. The closing trade first
changes the next calendar day's fee basis. Splits preserve the basis economically.
No current session close is used before it is observed. Fees reduce free cash;
shortfalls increase the explicit loan. Interest on that additional debt starts on
the following calendar date. Gross rebates are paid after borrow fees, followed by
actions, payments, collateral marks on sessions, and cash sweeps. Reserved long
reinvestment principal cannot silently fund these obligations.

`StockBorrow.rates` is either a mapping covering **exactly all potentially short
assets** in the requested run, or a dated table:

| Column | Type / meaning |
| --- | --- |
| `date` | Date; assigned calendar accrual date |
| `asset` | String |
| `annual_rate` | Float64; nonnegative nominal annual decimal fee |
| `available_at` | Datetime(us, UTC); availability of this supplied rate |

The dated table must cover every calendar accrual date for every potentially
short asset, including dates when its actual short balance is zero. Availability
must be no later than New York midnight on that date. No weekend fill, rate carry,
publication calendar inference, or percentage conversion occurs. Supply assigned
weekend rates with their real earlier availability. Missing dates, duplicates,
extra dates/assets, late availability, and null/nonfinite rates raise. Full requested
coverage is checked even for a subsequently stopped run. An all-long signed run
uses an explicit empty mapping. Source metadata requires nonblank `source` and
`basis="modeled"` or `"measured"`; it is retained with a deterministic input identity.
Mutation after construction is rejected. No rate input verifies stock-loan availability.

## Dividends, margin, and stops

Long entitlements keep the existing receivable/payment/reinvestment behavior.
Short entitlements instead recognize a positive payment-in-lieu liability and
negative P&L on the ex-date. Payment debits cash and removes that liability, with
no second expense—even after a cover. Unpaid obligations remain in ending equity.
Splits multiply signed quantities by the positive split ratio without a trade.

Short dividend liabilities never generate reinvestment. A long dividend earned
before an asset becomes short is still paid, but its standing reinvestment is
released to cash with audit status `payer_now_short`, rather than automatically
covering the short. Other eligible long purchases retain zero commission, spread,
and impact. Scheduled collisions and terminal mark-only behavior are unchanged.

The chosen margin rule compares equity with the sum of separate long and short
requirements. Equality is compliant with the existing relative roundoff tolerance.
Entry must comply; subsequent closes stop on nonpositive equity or a margin
breach. Pre-trade breaches cannot be concealed by scheduled covers or reinvestment.
Checks follow that date's obligations/collateral funding and run again after trades.
This is an explicit research assumption, not a broker, regulatory, or intraday rule.
No forced liquidation, borrow recall, or additional events occur after a stop.

Returns continue to divide P&L by portfolio equity: original capital for the first
interval (including entry costs), then previous closing equity. The failing close
is retained even for losses beyond 100%; log returns, position weights and leverage
are null when equity is nonpositive. `require_complete()` and partial-report opt-in
remain mandatory for their respective uses.

## Results, reports, and plots

Existing records retain their meaning and gain these fields:

- `daily`: `long_exposure`, `short_exposure` (absolute), `short_liability`,
  `restricted_collateral`, `dividend_liability`, and `margin_required`.
  `gross_exposure` is absolute; `net_exposure` is signed. `equity_ratio` still
  describes equity/gross exposure but does not replace the side-specific margin rule.
- `valuations`: `restricted_collateral`, `short_liability`, `dividend_liability`.
- `events`: `collateral_delta`, `dividend_liability_delta`. Positive short sale cash
  is visible in the linked signed trade event; collateral transfers are separate.
- `dividend_liabilities`: same columns as `receivables`, with negative entitled
  quantity and positive original/outstanding liability amounts. Paid rows remain
  with zero outstanding. Settlement is linked by action ID in `events`.
- `stock_borrow_accruals`: `date`, `asset`, `mark_session`, `cutoff_at`,
  `available_at` (null for fixed assumptions), `short_market_value`, `annual_rate`,
  `day_fraction`, `amount`. Zero positions retain audit rows; nonzero fees also
  appear in `costs` as `stock_borrow_fee`.
- `short_financing_accruals`: `date`, `opening_collateral`, `rebate_rate`,
  `day_fraction`, `rebate`. Rebate credits also appear in events and attribution.

Date fields are `Date`; timestamps are `Datetime(us, UTC)`; monetary/rate fields
are `Float64`; identifiers are strings. New tables are typed empty for legacy
long-only runs. Fees, dividend expenses, rebates, and price P&L reconcile to equity.
Short dividends appear in liabilities and attribution, not as a second trade cost.

`performance()` adds short borrow cost, short dividend expense, and gross collateral
rebate to signed-run summaries. Allocation includes collateral as an asset and
dividend liabilities as negative weights, so it still sums to one at positive equity.
Equity, returns, cumulative returns, drawdown, attribution, allocation, exposures,
turnover, and rolling-risk reporting consume the same result objects.

With a supplied benchmark, `performance()` now returns `information_ratio` and
`annualized_tracking_error` in `benchmark_comparison`:

```
active_return = portfolio_simple_return - benchmark_simple_return
information_ratio = sqrt(periods_per_year) * mean(active_return) / std(active_return)
annualized_tracking_error = sqrt(periods_per_year) * std(active_return)
```

The standard deviation is sample `ddof=1`; both interval endpoints must align.
Risk-free returns are not subtracted a second time. Fewer than two observations
produce null values with `insufficient_samples`; zero tracking error is reported
as zero while the ratio is null with `zero_tracking_error`. A zero-return benchmark
is valid: beta remains undefined for zero benchmark variance, while information
ratio uses the portfolio's active-return variation. `compare_performance()` includes
these metrics and retains its coverage, capital, units, assumption, and stop checks.
Scenario-specific holdings, fees, and financing policies remain in scenario metadata.

## Limits

Daily close marking and immediate settlement only. No borrow locate verification,
recalls, forced buy-ins, negative/net rebates, intraday margin, broker haircuts,
stock-loan collateral settlement lags, taxes, options, futures, or multiple currencies.
The ordinary allocation helpers and signal conversion still produce long-only
weights. This extension accepts caller-supplied signed positions; it does not select,
regress, or optimize hedges. Detailed documentation describes modeled economics,
not a promise that any real broker will finance or execute the portfolio.

Fixed currency targets and exact decision-dated cost models are implemented. See
[schemas, timing, audit tables and notebook migration](fixed-dollar-rebalancing.md).
