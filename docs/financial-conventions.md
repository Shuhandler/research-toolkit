# Financial conventions

Milestones 1A–1D and 2 implement the input/return conventions, account funding,
entry costs, splits, dividends, cash/borrowing interest, debt repayment, and margin
stops, performance ratios, benchmark comparisons, scheduled targets, allocation
and risk estimates below. Advanced allocation, CAGR, drawdown durations and
per-period risk-free curves remain proposed. See [the API](api.md) for the exact supported subset. User-confirmed
scope is recorded in [project context](project-context.md); numerical examples here
are independent test oracles, not market-data backtest outputs.

## Units and return definitions

Money is in the declared base currency, shares are quantities, and returns/rates
are decimal fractions. Basis points are converted by division by 10,000. Display
formatting may show percentages without changing stored values.

For positive comparable prices, price-only simple return is
`r_t = P_t / P_(t-1) - 1`; log return is `l_t = log(P_t / P_(t-1))`.
Raw prices across splits are not comparable without explicit action treatment.
A total-return series must identify its action/reinvestment convention; it is not
interchangeable with a raw-price series or a dividends-held-in-cash portfolio.

For a portfolio with no external flows and positive opening equity:

| Quantity | Definition | Unit |
| --- | --- | --- |
| Daily net P&L | `E_t - opening_equity_t` | Currency |
| Daily simple return | `pnl_t / opening_equity_t` | Fraction |
| Daily log return | `log1p(r_t)`, only when `r_t > -1` | Log fraction |
| Cumulative dollar P&L | `sum(pnl_t) = E_T - initial_capital` | Currency |
| Cumulative simple-return sum | `sum(r_t)` | Fraction; **not** total portfolio return |
| Compounded return | `product(1 + r_t) - 1 = E_T / initial_capital - 1` | Fraction |
| Cumulative log return | `sum(l_t)` | Log fraction |
| Wealth multiple | `product(1 + r_t)` | Multiple |
| Equity curve | `initial_capital * product(1 + r_t)` | Currency |

Example: +10%, then −10% gives a zero cumulative simple-return sum, a −1%
compounded return, and a 0.99 wealth multiple. On $100 starting equity, daily P&L
is +$10 then −$11. Never label a wealth multiple "compound return" or a log-return
sum "equity." Short-position simple return is not obtained by negating a log
return and exponentiating; use signed share P&L and the account ledger.

The first price-only observation has no prior price and a null return; do not fill
it with zero. Price-derived return functions preserve this structural null with a
diagnostic. Metric functions exclude only documented structural leading nulls;
interior missing data is an error, not automatic sample deletion.

External cash flows are rejected in M1. Later support must distinguish investment
P&L from contributions/withdrawals and specify time-weighted or money-weighted
returns before changing the denominator.

## Balance sheet and reconciliation

At each recorded valuation:

`equity = sum(quantity_i * raw_price_i) + cash + dividend_receivables - debt`.

Cash and debt are separately visible nonnegative balances. There is no hidden
negative-cash loan. Receivables are assets but are not spendable cash. Debt includes
posted financing charges. Entry capital is equity, not gross asset exposure.

Every quantity change links to a trade or a corporate action. Trade cash flow is
`-signed_quantity * reference_price - trade_cost`. Borrowing increases both cash
and debt; repayment reduces both. Neither is P&L. Dividend payment transfers a
receivable to cash; it is not a second dividend profit. Net P&L reconciles to market
value changes adjusted for trades/splits, dividend accrual, cash interest, minus
commissions, spread/impact expenses, and borrowing charges. Dollar attribution
including unattributed financing must sum to portfolio P&L.

M1 uses fractional shares, one currency, no taxes, no settlement lag, no external
flows, and ordinary cash dividends and splits only. These are recorded modeling
assumptions. Reject mergers, spin-offs, special distributions, unsupported
delistings, or other unmodeled events; do not fill through them or discard assets.
Raw prices must be positive finite values. A zero-price default/delisting requires
an explicit future terminal-event model, not a fabricated last price.

## Entry, holding periods, and execution

The caller chooses entry session `t0`, final session `tN`, and the named execution
policy. Static weights must be decided before entry. The first implementation
supports only `entry_close`: idealized dollar-sized orders at that session's raw
close, with modeled execution costs. Using that close to convert predetermined
dollar allocations into fractional shares is a stated fill assumption. Selecting
weights from that same close is not permitted. There is no claimed guarantee of a
real closing-auction fill or capacity at the requested size.

Record `initial_capital` as a pre-entry valuation, then post-entry balances and
entry trades/costs at `t0`. The daily return table starts at `t1`: its opening equity
is **initial capital before entry costs**, and its P&L includes entry costs plus
the first holding interval. Later opening equity is the preceding daily closing
equity. Thus entry costs are never lost and no extra zero-duration observation
inflates annualized metrics. The first row carries `includes_entry_costs=true`
in its metadata/diagnostics. The post-entry balance at `t0` remains separately
available in `valuations`, not substituted as the performance denominator.

Require at least one holding interval for a full backtest; entry-only accounting
can be tested internally. All returns compare identical interval endpoints with
the benchmark. The ending valuation is mark-to-market, without an assumed sale
or exit fee. Ordinary buy-and-hold quantities remain fixed except for splits.
Dividend reinvestment, repeated target-weight multiplication, and daily leverage
resetting would require trades and are not this strategy.

Later signal strategies must carry observation/availability, decision, order,
execution, and mark times. A close-derived signal cannot earn the return ending
at that close; a next-open policy needs next-open data. Rebalancing schedules and
holiday rules must be explicit. Never infer an execution policy from data frequency.

## Cost-aware initial allocation and leverage

Risky-asset weights `w_i` are nonnegative and sum to 1. Initial gross leverage
`L` means gross asset value divided by **post-entry net equity**. `L=0` is all
cash; `0<L<1` leaves cash; `L=1` is fully invested without a loan; `L>1` borrows.
Concentration is checked on the declared denominator; equity weights can exceed
100% with leverage. Initial allocation limits do not trigger automatic later trades.

For capital `C`, a uniform proportional entry cost `k`, and gross notional `N`:

`N = L * (C - k*N) = L*C / (1 + L*k)`.

Then `cost = k*N`, `E_post = C-cost`,
`debt = max(N + cost - C, 0)`,
`cash = C + debt - N - cost`, and `quantity_i = w_i*N/P_i`.
For asset-specific proportional rates use `k = sum(w_i*k_i)`.
Verify nonnegative cash/debt within stated numerical tolerance; do not mask a
funding deficit by silently borrowing. Fixed/minimum commissions, integer lots,
and nonlinear impact need a tested sizing solver in a later phase.

Example: `C=100`, `L=2`, `k=0.01` gives `N=196.0784313725`,
cost `1.9607843137`, equity and debt each `98.0392156863`, and zero cash.
With no price move or financing the net return relative to initial capital is
−1.9607843137%. With no costs or interest, a $100 account buying $200 of stock
with $100 debt earns $20 on a 10% price rise: equity is $120 and leverage drifts
to `220/120`, not 2. Maintaining 2 would require another trade.

## Corporate actions and daily event order

1. Accrue financing/cash interest for the elapsed calendar date as described below.
2. Apply splits effective before that session's trading: multiply quantity by the
   new/old share ratio. The raw mark changes independently. A pure 2-for-1 split
   with the price halved changes neither wealth nor cash and incurs no trade cost.
3. Accrue ordinary dividends on the ex-session for eligible pre-ex holdings,
   expressed in post-split shares if a split occurs on the same date. Increase
   dividend receivables. An entry at the ex-date close does not earn that dividend.
4. Transfer due receivables to cash on their supplied payment dates, including
   nontrading dates. Apply the explicitly selected debt-repayment cash sweep.
5. Execute any permitted trades (only initial entry in M1), charge their costs,
   mark at the session close, and check equity and maintenance limits.

For nontrading dates process financing and cash events without inventing new
market prices or daily market-return observations. Aggregate these events into
the next session's P&L. Preserve their actual calendar dates in the event ledger.
Opening holdings at entry are zero; dividends with ex-dates before entry create
no entitlement. A dividend earned before the final session but paid afterward
remains a receivable in ending equity. The snapshot must include its pay date.

No automatic reinvestment. A benchmark total-return index may assume reinvestment;
label that difference. Split-adjusted-only, split-and-dividend-adjusted, and raw
prices have distinct meanings. Snapshot metadata must specify which was supplied.
The chosen first implementation executes only against raw prices.

The distinction between dividend entitlement and payment follows
[Investor.gov's ex-dividend explanation](https://www.investor.gov/introduction-investing/investing-basics/glossary/ex-dividend-dates-when-are-you-entitled-stock-and).
M1 deliberately covers ordinary distributions only; unusual entitlement rules
need explicit additional support.

## Costs, financing, and limits

Commission, half-spread, and impact are separate nonnegative expense components
on actual absolute traded notional. M1 supports user-supplied proportional bps
per component (uniform or by asset); an assumed impact bps is a scenario, not a
liquidity/capacity model. All values, including zeros, must be explicit. Keep
observed quotes/fees separate from modeled assumptions in the cost records.

M1 books spread/impact as cash expenses against a reference-price trade. Do not
also embed those expenses in an adverse fill price. Later fill-price models must
report the same economic cost once. No repeated cost on unchanged holdings, and
no automatic round-trip fee per observation. A sale incurs its own cost only if
a sale is actually simulated. Financing can accrue even when there are no trades.

Financing rates are stated **nominal annual** rates with ACT/365F calendar-day
accrual. At each calendar date after entry through the ending session date,
charge `opening_debt * borrowing_rate / 365` and credit
`opening_cash * cash_rate / 365`; post the charge into debt and interest into cash
before that date's action/payment events. Opening balances are the preceding
calendar date's ending balances. This is daily capitalization, including weekends
and holidays; three days with unchanged other balances grow debt by
`(1 + borrowing_rate/365)^3`, not one trading-day charge. No charge precedes entry.
Intraday partial-day accrual is outside M1. Rates are nonnegative in M1; future
negative-rate support needs explicit signed-expense rules.

The named `repay_debt` cash sweep uses available cash to repay debt after payments
and interest, with any excess kept as cash. It does not sell shares, reinvest
dividends, or maintain leverage. Constant rates are M1 assumptions; dated rate
curves and different day counts are later extensions. Risk-free rates used for
metrics are independent of these cash/borrowing rates.

For leveraged runs require a caller-supplied maintenance equity/gross-exposure
threshold, validate initial funding and compliance, and monitor at each session
close. Require finite `L >= 0`, finite nonnegative cost/rate inputs, and a
maintenance ratio in `(0, 1]` when borrowing; reject initial leverage that already
violates the configured threshold. If equity becomes nonpositive or maintenance is breached, stop with the
failing valuation/event ledger and incomplete status. Do not silently liquidate,
continue borrowing, or advertise a full-year result. All-cash exposure has no
margin-ratio denominator. A configurable research threshold is not a representation
of broker margin rules, intraday monitoring, or guaranteed liquidation prices.
Real margin accounts involve borrowing and additional requirements, as described
by [FINRA](https://www.finra.org/rules-guidance/key-topics/margin-accounts).

Equality at the maintenance threshold is compliant. A relative tolerance of 1e-12
handles floating-point roundoff in that comparison; it is not a currency tolerance.
Stopped results include the failing session's balances, P&L, and costs, plus stop
reason/time and actual ending session. No subsequent event, charge, or trade is
processed. A breach on the requested final session still has stopped status.
`BacktestResult.require_complete()` rejects such results for full-period use.

At bankruptcy retain the last computable P&L and simple return from positive
opening equity, even if the return is at or below −100%; log return and CAGR are
undefined. Never clip losses to make charts or logarithms work. Short sales,
stock-loan availability/fees, collateral, recalls, and payments in lieu of dividends
are deferred. Reject negative weights until that separate accounting is supported.

## Metrics, annualization, and undefined cases

Annualized metrics require an explicit `periods_per_year=A`; 252 is an example
for daily equities, 365 a different assumption for daily continuous markets.
The calendar determines expected sessions, not a universal annualization constant.
Changing sampling frequency requires a new declared convention. No hidden square
root multiplier inherited from an earlier notebook cell. Irregular/overlapping
return intervals are rejected by regular-frequency metrics until explicitly modeled.

Use daily **net simple portfolio returns** for ordinary risk metrics. Sample
standard deviations/covariances use `ddof=1`. Volatility is `std(r)*sqrt(A)`;
arithmetic annualized mean is `mean(r)*A` and is not CAGR. CAGR, if requested, is
`(E_T/C)^(365.25/elapsed_calendar_days)-1` for positive wealth and elapsed time,
with that year-length convention recorded. Square-root scaling assumes a regular
sampling convention and is not a serial-correlation adjustment.

Risk-free input is an explicit effective annual scalar or an aligned per-period
simple-return table. The initial scalar convention is
`rf_period=(1+rf_annual)^(1/A)-1`. This sampling-based metric convention is separate
from calendar-day loan accrual. Record the choice. Annual risk-free/MAR inputs must
exceed −1; never divide an effective annual rate by A silently.

- Sharpe: `sqrt(A)*mean(r-rf)/std(r-rf)` on matched periods.
- Sortino: convert the explicit annual minimum acceptable return (MAR) by the same
  effective-rate rule, let `x=r-MAR`, then use
  `sqrt(A)*mean(x)/sqrt(mean(min(x,0)^2))`. The downside denominator uses **all**
  observations, not only negative ones. State this convention in results.
- Drawdown: `E_t / running_peak_equity_t - 1`; include pre-entry capital and the
  post-entry valuation so initial losses are visible. Maximum drawdown is the
  minimum, a nonpositive fraction. Durations use recorded sessions/dates explicitly.
- Correlation: Pearson correlation of aligned simple-return intervals, never
  price-level correlation by default. Asset correlations require an explicitly
  chosen price or total-return basis and declared common sample.
- Beta: `cov(portfolio_return, benchmark_return)/var(benchmark_return)` on the
  same aligned net-return sample with `ddof=1`; report n and coverage. It is the
  slope with an intercept, not a ratio of total returns or prices. An excess-return
  beta would be a separately named option, not a silent substitution.
- Benchmark comparisons also show benchmark compounded return and the difference
  in compounded returns (percentage points), distinct from relative wealth
  `(1+R_portfolio)/(1+R_benchmark)-1`.

Require at least two observations for sample volatility, Sharpe, Sortino, beta,
and correlation in the reporting contract. Undefined metrics return null plus a
status such as `insufficient_samples`, `zero_volatility`, `zero_downside_risk`,
or `zero_benchmark_variance`; never invent zero Sharpe or infinite skill. Empty
series produce an empty-sample diagnostic. Nonfinite inputs, invalid denominators,
duplicates, or missing interior observations raise before metric calculation.
Short samples may be computable but must show n; computability is not reliability.

## Allocation and turnover conventions

Gross exposure is `sum(abs(asset_value))`; net exposure is `sum(asset_value)`;
leverage is gross exposure/equity. Position weights use net equity, so a leveraged
long portfolio can sum above 1. Cash, receivables, and negative debt weights make
the complete balance-sheet weights sum to 1. Initial risky proportions have a
different denominator (gross risky exposure) and must be named accordingly.

Turnover reports gross traded notional/pre-trade equity, with entry separately
labeled. If a half-turnover statistic is added it must have a different name.
Drifting weights alone are not turnover. Estimated volatility risk contribution is
`w_i*(Sigma*w)_i / sqrt(w' Sigma w)`, with covariance sample/window/annualization
recorded; contributions sum to portfolio volatility under that stated model.
Zero variance gives undefined contributions. These are analytical estimates,
not a replacement for realized ledger attribution or financed cash-flow accounting.

## Scheduled portfolios and historical estimates (implemented M2)

The first release's buy-and-hold policy remains unchanged. Scheduled portfolios
supply complete dated target baskets, explicit zero-weight exits, and a concentration
limit on **risky proportions**, separate from target leverage against post-cost net
equity. Actual proportions and leverage may drift between scheduled dates. A breach
of the requested concentration limit raises, never clips weights or forces a later
trade. Every decision precedes its execution; non-session dates raise and the final
session is mark-only. Dates come from the declared calendar, not inferred holidays.

For a scheduled close, old quantities (after any effective split) earn that day's
price movement and ex-date entitlement. Financing and due payments precede the
trade. The engine checks pre-trade solvency/margin; a breached account stops before
a scheduled sale could conceal the breach. Otherwise it solves post-cost sizing,
funds borrowing, executes sales before purchases, charges each proportional cost on
absolute changed notional, and repays excess debt. The final close is checked again.
No fill earns a price move that already happened. Shares/events, cash/debt,
receivables, costs, P&L and equity reconcile through the same ledger as buy-and-hold.

For pre-trade equity `E`, receivables `R`, current risky values `v`, target proportions
`w`, leverage `L` and total proportional cost rates `k`, solve
`x + sum(k_i * abs(w_i*N(x) - v_i)) = E` for positive post-cost equity `x`.
Ordinary target notional is `N(x)=L*x`. The explicit `reserve` policy at `L<=1`
uses `max(0, min(L*x, x-R))`, so unspendable dividend assets cannot cause an implicit
new loan; any reduction in requested exposure is recorded. Existing debt can remain
against unpaid receivables after liquidating all risky holdings and is repaid when
cash arrives. `require_target` instead rejects a target at `L<=1` if it would leave
debt. Leveraged targets use the supplied financing policy, including for receivable
funding. All dividends still follow actual pay dates; reserve is not a second cost.

M2's monotone proportional-cost solver requires per-asset total rates below 100%
and `L*sum(w_i*k_i)<1`; unsupported extreme rates/funding fail explicitly. These are
numerical/model scope limits, not market fee recommendations. Fixed, per-share,
minimum and nonlinear costs require another tested sizing method before support.
A 1e-13 relative notional tolerance suppresses only representation-level differences;
other changed quantities create actual trades. No arbitrary minimum turnover or
fee is inserted. The existing sub-cent currency reconciliation tolerance still applies.

Gross turnover is `(sum(buy_notional) + sum(abs(sell_notional))) / pre_trade_equity`,
with entry separately labeled. Unchanged quantities have zero trade turnover/cost.
Financing and split events are not trades. Holding days do not implicitly rebalance.
Pre-rebalance valuations are included in equity/drawdown reporting but do not add
zero-duration observations to daily returns, volatility or Sharpe.

Independent example without costs: a $100 portfolio owns $50 of each of A/B. A rises
20%, B is flat, giving values $60/$50 and equity $110. A scheduled 50/50 target sells
$5 A and buys $5 B, gross turnover `10/110`. If A then rises 10%, the new basket earns
$5.50, while unchanged holdings would earn $6. With 1% costs, a complete switch from
A to flat-priced B leaves `100*0.99/(1.01**2)` of equity after the original A entry,
the A sale, and B purchase. It does not charge three full-portfolio round trips.

Inverse-volatility and covariance calculations use common simple-return intervals
ending strictly before the declared decision session. An integer window of at least
two returns and explicit annualization are required; covariance/std use `ddof=1`.
A zero-volatility asset cannot receive an inverse-volatility weight; raise rather
than assigning infinity or dropping it. Concentration failure also raises. The
result retains the sample start/end, prior interval start, count, basis and source.

Raw analytical total returns can explicitly assume `reinvest_ex_close`, using
`split_ratio*(close+ex_dividend_per_post_split_share)/prior_close-1`. This labels a
reinvested entitlement index, not executable use of unpaid dividends. Portfolio
simulation continues to distinguish entitlement and payment; users may instead
supply their own documented return panel for allocation. Raw price-only returns
remain split-discontinuous and should not be mistaken for total returns.

Estimated Euler contributions use net-equity exposures `u=L*w`, annualized sample
covariance `Sigma`, and portfolio volatility `sigma=sqrt(u' Sigma u)`:
`contribution_i=u_i*(Sigma*u)_i/sigma`. They sum to sigma and can be negative when
an asset offsets others. Null contributions with `zero_portfolio_volatility` replace
undefined zero-risk ratios. This fixed-exposure covariance model treats cash and
financing as deterministic; it is distinct from realized dollar attribution and
from the path-dependent costs and margin behavior of the ledger.

Rolling portfolio volatility and Sharpe use realized **net** returns through each
closing session. They retain all early rows as null until the full window exists,
with actual counts and reason codes. These are reporting diagnostics, not inputs
available to place an order at that same close. Stopped runs require explicit
partial-report opt-in and retain the actual dates and stop reason in plotted output.
