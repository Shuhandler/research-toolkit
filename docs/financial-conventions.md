# Financial conventions

Milestones 1A–1D and 2 implement the input/return conventions, account funding,
entry costs, splits, dividends, cash/borrowing interest, debt repayment, and margin
stops, performance ratios, benchmark comparisons, scheduled targets, allocation
and risk estimates below. Capped allocation, nonlinear square-root costs and
dated risk-free reporting are also implemented, as are CAGR/Calmar/higher-moment
diagnostics, factor regressions and cross-sectional signal diagnostics, as are
two-strategy information-ratio weights and analytical strategy combinations. Drawdown
durations and advanced allocation remain proposed. See [the API](api.md) for the exact supported subset. User-confirmed
scope is recorded in [project context](project-context.md); numerical examples here
are independent test oracles, not market-data backtest outputs.

The original sections below describe the long-only account. The implemented
[long/short convention](#longshort-accounting-extension) extends that account with
restricted collateral, signed holdings and short dividend obligations.

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
weights from that same close requires the explicit scheduled
`same_session_close_assumed` research opt-in described below. There is no claimed guarantee of a
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
Without an explicit dividend policy, there is no stock reinvestment. Opt-in
`DividendReinvestment` adds payment-funded purchases, recorded as actual trades.
Repeated target-weight multiplication and daily leverage resetting remain separate
strategies requiring explicit trades.

Implemented signal instructions carry observation/availability, decision, order
and next-session-close execution times; the ledger retains actual raw mark times. A close-derived signal cannot earn the return ending
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
   nontrading dates. Allocate optional reinvestment budgets, release them on
   scheduled/terminal dates, and apply the selected funding priority and debt sweep.
5. Check pre-trade margin, execute any scheduled basket or dividend purchases, charge their costs,
   mark at the session close, and check equity and maintenance limits.

For nontrading dates process financing and cash events without inventing new
market prices or daily market-return observations. Aggregate these events into
the next session's P&L. Preserve their actual calendar dates in the event ledger.
Opening holdings at entry are zero; dividends with ex-dates before entry create
no entitlement. A dividend earned before the final session but paid afterward
remains a receivable in ending equity. The snapshot must include its pay date.

Automatic reinvestment requires an explicit `DividendReinvestment` policy. A
benchmark total-return index may use a different reinvestment convention; label
that difference even when portfolio reinvestment is enabled. Split-adjusted-only, split-and-dividend-adjusted, and raw
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

The original fixed-rate `Financing` rates are **nominal annual** rates with ACT/365F calendar-day
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
and interest, with any excess kept as cash. With reinvestment enabled,
`before_debt_repayment` reserves paid dividend principal from this sweep until
execution or release; `after_debt_repayment` leaves the sweep unchanged. The sweep
itself does not sell shares or maintain leverage. Constant rates remain the M1 mode. The historical SOFR extension below supplies
dated loan rates with explicit day counts; arbitrary curves and other billing
conventions remain future work. Risk-free rates used for
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
stock-loan fees, collateral and payments in lieu of dividends are now supported
only through the explicit signed interface described below. Negative long-allocation
`weights` remain invalid; recalls and borrow-availability verification remain deferred.

## Metrics, annualization, and undefined cases

Annualized metrics require an explicit `periods_per_year=A`; 252 is an example
for daily equities, 365 a different assumption for daily continuous markets.
The calendar determines expected sessions, not a universal annualization constant.
Changing sampling frequency requires a new declared convention. No hidden square
root multiplier inherited from an earlier notebook cell. Irregular/overlapping
return intervals are rejected by regular-frequency metrics until explicitly modeled.

Use daily **net simple portfolio returns** for ordinary risk metrics. Sample
standard deviations/covariances use `ddof=1`. Volatility is `std(r)*sqrt(A)`;
arithmetic annualized mean is `mean(r)*A` and is not CAGR. CAGR is implemented in
`performance_diagnostics` with an explicitly chosen convention; see
[diagnostics conventions](#performance-diagnostics-factor-regression-and-signal-evaluation). Square-root scaling assumes a regular
sampling convention and is not a serial-correlation adjustment.

Risk-free input is an explicit effective annual scalar or an aligned per-period
simple-return table. The initial scalar convention is
`rf_period=(1+rf_annual)^(1/A)-1`. This sampling-based metric convention is separate
from calendar-day loan accrual. Record the choice. Annual risk-free/MAR inputs must
exceed −1; never divide an effective annual rate by A silently.

- Sharpe: `sqrt(A)*mean(r-rf)/std(r)` by default, preserving the existing
  implementation; explicitly select `sharpe_denominator="excess_returns"` for
  `std(r-rf)`, the reference notebook convention. Both use sample `ddof=1`
  and exactly matched intervals. They differ when the risk-free return changes.
  Earlier documentation wrote only `std(r-rf)` while scalar-rate code used
  `std(r)`; that ambiguity is now resolved explicitly.
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
numerical/model scope limits, not market fee recommendations. The implemented
`SquareRootImpactCosts` extension uses a separate tested nonlinear solve, including
per-share commissions; fixed/minimum ticket fees remain unsupported.
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
than assigning infinity or dropping it. Concentration failure raises by default;
explicit `cap_policy="redistribute"` water-fills weights within a feasible cap. The
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

## Payment-funded automatic reinvestment

A standing `DividendReinvestment` instruction purchases the paying asset with
actual paid cash, never an unpaid entitlement. It assumes cash is available before
the first supplied session close on or after the payment date. This is an explicit
daily execution assumption; data contain no intraday payment timestamp and no
broker DRIP execution price. Missing required prices still raise. A non-session
payment earns no fabricated stock return before the next actual session.

For available dividend budget `B`, purchase `B` of stock and `B/P_close` shares.
Automatic dividend purchases have zero commission, spread and impact, as requested
by the user. No external cash or new debt tops up this budget. Each paid action
retains a linked fill with `trade_cost=0`, and generates no fee events/cost rows.
The exchange of cash for shares at the raw mark is equity-neutral. Ordinary entry
and scheduled rebalancing retain their configured costs, even when a scheduled
basket uses released dividend cash. The zero-DRIP-cost assumption is recorded in
metadata as `reinvestment_cost_policy="zero_commission_spread_impact"`; it is a
research convention, not a universal statement about broker fees.

With `after_debt_repayment`, unearmarked cash repays debt first and any required
remainder reduces pending dividend budgets pro rata. With `before_debt_repayment`,
paid principal stays earmarked through non-session dates; other cash still sweeps.
Existing debt continues accruing financing costs, and reserved cash earns the
configured cash rate. Cash interest is not added to the purchase budget. Both
policies retain original ex-date entitlements regardless of later splits/sales;
purchases use the execution date's raw mark and post-split share units.

Scheduled rebalancing takes priority at an overlapping close. Release the earmark
before the ordinary sweep/basket; do not buy and then immediately sell the same
stock as an extra DRIP leg. The final session is mark-only: release any pending
budget to the account and apply the ordinary cash sweep. Thus `hold_cash` means
no final stock purchase, and released cash may repay debt. Dividends paid after
ending remain receivables, with no reinvestment record until actual receipt.

A standing instruction can reopen a payer sold at an earlier close. It does not
retarget portfolio weights or enforce a target concentration limit between
scheduled baskets. This behavior, timing, funding and cost assumptions are saved
in run metadata. Disable reinvestment to retain the original quantity path.

Old holdings earn the price move into the execution close. The cost-free purchase
creates no immediate P&L; new shares first earn subsequent price moves and subsequent
ex-date dividends. Pre-reinvestment equity is included in drawdowns. A pre-trade
margin/insolvency failure prevents purchases; post-trade failures stop at the same
close without liquidation. Stops freeze account balances; any blocked paid budget
is released in the audit record without additional post-stop debt transactions.

For every paid action the audit must reconcile:
`paid_amount = debt_repaid + cash_released + signed_notional + trade_cost`.
`trade_cost` is retained for schema consistency and is always zero for DRIP.
`cash_released` is cash returned to unrestricted account funding, not a promise
that it remains in closing cash. These audit allocations are not extra ledger
cash movements. Events, positions, all costs, P&L and equity still reconcile.

## Historical SOFR loan convention

SOFR observations describe overnight transactions and are published subsequently.
The New York Fed describes actual-calendar-day/360 calculations and publication
and revision timing in its [reference-rate methodology](https://www.newyorkfed.org/markets/reference-rates/additional-information-about-reference-rates).
These facts motivate separate observation and availability fields; they do not
uniquely specify a brokerage financing agreement.

`SOFRFinancing` implements this explicit research model:

1. For every calendar date after entry through the requested ending session,
   establish the cutoff at **00:00 America/New_York on that date**, stored in UTC.
   Select the latest observation whose supplied availability timestamp is at or
   before that cutoff. A rate published later that morning cannot affect that
   day's charge. Rates known at the cutoff are deliberately lagged relative to
   overnight transaction observation dates. No intra-day repricing occurs.
2. Carry the latest available observation only under this named policy, with an
   explicit maximum age measured as accrual date minus observation date in calendar
   days. The supplied publication calendar must contain every expected observation;
   each must have a rate. A missing rate is not treated as a holiday. Coverage must
   include the whole requested run, even if it later stops or has no debt.
3. `loan_rate = sofr + borrowing_spread_bps / 10_000`. Inputs are annual decimal
   rates, not percentages. For `ACT/360`, charge `opening_debt * loan_rate / 360`;
   `ACT/365F` instead divides by 365. Cash uses its separately declared fixed rate
   and day count. No risk-free metric rate is inferred from SOFR.
4. Post interest to debt/cash before that date's actions, payments, sweep or trades.
   Capitalize every calendar day, including weekends. Day fractions remain 1/360
   or 1/365 on daylight-saving transitions; no hourly proration is implied. Entry
   receives no charge; its first subsequent date receives one whole daily charge,
   consistent with the existing daily ledger. Stops retain charges through the
   failure close only, with no later postings.

Example: a $100 loan, synthetic annual SOFR 3.6%, 36bps spread and ACT/360 grows
by factor 1.00011 per calendar day. If Thursday's observation is published Friday
morning and Friday's is published Monday morning, the Thursday observation funds
Saturday, Sunday and Monday accrual rows; Friday's first funds Tuesday. If the new
SOFR is 7.2%, Tuesday's factor becomes 1.00021. These are synthetic arithmetic
examples, not reported historical rate values.

The official SOFR Index's treatment of overnight observation periods and
nonbusiness-day simple interest differs from this loan's daily capitalization and
availability-based reset. Do not label this model an exact index replication or
an in-arrears overnight contract. Matching a broker requires its reset, spread,
settlement and billing terms. This version supports nonnegative SOFR/spreads and
fixed nonnegative cash rates; negative inputs raise, with no implicit floor.

Publication timestamps must reflect availability of the supplied rate vintage.
A contemporary download containing later revisions must not be labeled
point-in-time history without verification. Multiple intraday vintages and their
reconstruction are not supported; one chronological available vintage per
observation is required. The complete publication calendar and declared metadata
remain the caller's responsibility; no exchange holiday rules are inferred.

`financing_accruals` ties each selected observation/cutoff, rate, spread, day fraction
and opening balance to cash interest and borrowing costs. Zero-balance dates still
get audit rows; no zero-value ledger cost event is invented. Source input identity,
all supplied observations/calendar rows and policy are embedded in run metadata.
The loan spread/cash rate and balance/capitalization rules are modeling assumptions;
the benchmark observations retain separate source provenance.

## Notebook extension conventions — implemented

[The extension contracts](notebook-extensions.md) specify the financial details:

- Capped inverse volatility preserves relative inverse-volatility scores among
  uncapped assets, requires feasible capacity, and never extends the sample cutoff.
- Square-root impact uses daily volatility, absolute order/dollar ADV, and a
  nonnegative modeled coefficient. It is a cash expense, separate from spread and
  commissions. Post-cost equity determines leverage; ordinary actual orders are
  charged once, while automatic dividend reinvestment remains cost-free.
- Dated RF returns align both interval endpoints. Daily nominal decimal rates must
  already be assigned to every calendar accrual day and available by its local
  midnight. ACT/360 or ACT/365F and simple versus daily compounding are explicit.
  Missing dates raise. Loan rates/spreads, cash interest and the performance RF
  benchmark remain distinct. Sortino's constant MAR is unaffected.
- Rolling benchmark beta/correlation use common complete windows and sample
  covariance/variance, through the current close, for descriptive reporting.
- Cumulative sum, compounded return and wealth multiple have different columns
  and plot labels. Comparison tables retain numerical units, requested/actual
  coverage, undefined statuses and declared differences without sample intersections.
- Yahoo Close is split-adjusted. Raw reconstruction needs explicit complete split
  factors through retrieval, authoritative actions/payment dates, and a calendar.
  Volume basis and reconstructed close-time availability are disclosed separately.

## Chronological research and signal conventions — implemented M3

Lagged features use positive supplied-session offsets, with per-source publication
lineage and explicit warmup/unavailable statuses. Research end dates are inclusive
outcome/availability cutoffs. Labels crossing a partition's end or published after
its cutoff are purged and retained as exclusions; explicit boundary gaps are
additional decision-session exclusions. Past history may supply held-out lagged
features, but only retained training rows fit means/sample standard deviations.

Validation selects the lowest declared prediction loss on exactly matching keys.
Only that candidate can be evaluated on the final-test partition. Repeated test
use and reselection after test use require an exploratory audit status. Persisted
history must be supplied on resume; the toolkit cannot detect external test access
or prove externally computed predictions/universes were constructed causally.
These loss scores are not net trading returns or performance ratios.

Dated signal weights remain nonnegative risky proportions, separately from gross
leverage and model coefficients. Explicit next-session-close execution cannot earn
the move into that close. Each-signal rebalancing may restore drifted exposures;
on-change rebalancing holds quantities until the supplied instruction changes.
Neither convention silently chooses the other. Calendar gaps do not create bars,
and the terminal close remains mark-only. Signal-driven trades share all existing
cost/funding/receivable/margin/DRIP accounting. See [the full guide](chronological-research.md).


## Long/short accounting extension

User-confirmed conventions: daily marked restricted collateral (choice 1A) and
separate long/short maintenance requirements (choice 2A). These are research
assumptions, not broker rules. Exact contracts and worked examples are in the
[long/short guide](long-short.md).

For long value `L`, absolute short value `S`, unrestricted cash `C`, restricted
collateral `K`, dividend receivables `R`, dividend liabilities `D`, and borrowing `B`:

- `E = L - S + C + K + R - D - B`.
- `gross = L + S`, `net = L - S`, `leverage = gross/E` for positive equity.
- `K = collateral_multiple * S` at each supplied close and after baskets;
  the multiple is explicitly supplied and at least one.
- Maintenance requires `E >= long_margin*L + short_margin*S`, with independently
  supplied fractions. Equity/gross remains descriptive, not a second margin rule.

Opening a short increases cash and the signed share liability equally, creating
no P&L. Cash is segregated before any debt sweep. Daily collateral increases use
free cash then the explicit loan; decreases release cash to the existing debt
sweep. Extra collateral is an account asset, not an expense. A cover removes
shares and releases collateral. The reported short liability is already included
in signed market value; never subtract it twice. The same engine reconciles cash,
restricted collateral, loans, quantities, dividend assets/liabilities, and P&L.

Signed equity targets scale with positive post-cost equity; exact signed quantities
never scale. Entry expenses reduce equity and can increase borrowing without
altering exact long quantities. Fixed-bps and square-root/per-share costs apply to
absolute actual trades once. Zero trades have no fees; crossing signs trades the
full quantity difference. Existing `weights` and long-only sizing retain their rules.

Every calendar date uses the previous supplied closing short market value for
borrow fees, including weekends. Fixed annual asset assumptions or complete dated
rates explicitly assigned to each date are required. Dated availability must be
no later than New York midnight. ACT/360 or ACT/365F is explicit. Debit interest,
free-cash interest, borrow fees, and gross collateral rebate are separate postings.
Rebate applies to opening restricted collateral, including any excess above S;
net-of-borrow-fee rebates are unsupported. No short-loan availability is inferred.

Short dividends reduce equity and create a liability on the ex-date; payment
reduces cash and the liability without another expense. They never reinvest.
Signed splits preserve economic value. Existing long dividend reinvestment stays
free of trading costs. If a previously entitled long payer is now short, its
payment is released to cash rather than automatically covering the position.

Session-close margin/insolvency checks occur before scheduled trades and again
after execution. Stops retain the failure close and actual coverage; subsequent
events stop. Returns divide by equity, not exposure or short proceeds. The failing
simple return can be at/below -100%, while log returns/weights/leverage are null
at nonpositive equity. No forced liquidation or broker-specific rescue is assumed.

Information ratio uses `sqrt(A)*mean(r-b)/sample_std(r-b)` on exactly aligned
holding intervals. Tracking error is `sqrt(A)*sample_std(r-b)`. No second risk-free
subtraction occurs. At least two intervals are required; zero tracking error yields
zero reported tracking error and null IR with its reason. A caller-supplied zero
benchmark is valid and does not make beta against a constant benchmark defined.

## Fixed currency targets and dated execution estimates

Scheduled signed `target_notional` is a predetermined currency amount. At its
execution close, divide by the raw price and trade against actual shares after
splits, dividend processing and earlier transactions. Positive/negative/zero mean
long/short/exit. The absolute amounts, including any hedge, determine gross asset
exposure; neither capital nor post-cost equity normalizes them. Costs reduce
equity and affect cash/debt separately. `equity_exposure` still sizes a multiple
of post-cost equity and `quantity` still specifies exact shares.

No dollar basket itself establishes beta neutrality or portfolio leverage. It
restores gross dollars only at scheduled executions, within the existing bounded
currency reconciliation tolerance; exposure drifts afterward. Receivable funding
must remain `require_target` under the explicit loan. Restricted collateral cannot
fund longs. Existing entry validation, pre-trade checks, post-trade margin stops
and stopped coverage are unchanged; costs never cause a substitute smaller target.

Decisions strictly precede supplied execution sessions. Next-session close is an
explicit caller mapping for ordinary target tables, not a newly inferred policy.
The final supplied run session remains mark-only. An order cannot capture the move
ending at its own execution price.

Scheduled `costs` may be one static model or an exact decision-date mapping of
`SquareRootImpactCosts`. Validate the entire requested mapping, currency, model
identity, and the full target universe (including exits and hedges) before the run.
Each dated snapshot must identify that decision, have sample endpoints on supplied
sessions strictly before it, and declare an availability cutoff between sample-end
close and decision close. `estimate_liquidity` retains the stricter prior-session
sample and decision-day local-midnight volume availability rule even when signals
are formed after the close. No decision-day or execution-day return/volume enters
that estimate. External snapshot vintage remains a caller declaration.

Select one binding per basket for sizing and fills. Commission, per-share fees,
spread and nonlinear impact apply to actual absolute changed positions once.
Daily volatility, dollar ADV and order/ADV retain their existing units; order/ADV
is not intraday participation. Debit financing, stock borrowing and collateral
rebate remain separate. Automatic long dividend reinvestment is still free.
Audit identities and source windows are retained alongside actual cost components;
no model refresh restarts or stitches the ledger. See
[exact contracts and numerical tolerances](fixed-dollar-rebalancing.md).


## Same-close research assumption

`RebalancePolicy(decision_timing="same_session_close_assumed", ...)` explicitly
permits decision and execution on the same supplied session. The default
`decision_timing="prior_session"` still requires strict precedence. Future
decisions and terminal-date trades remain invalid. The opt-in is an idealized
research assumption that close-derived signals can fill at that close, not a
claim about executable auction information. Holdings first earn the subsequent
interval; existing positions earn the move ending at a rebalance close.
Trailing beta/liquidity windows and cost availability checks are unchanged.
The setting is retained in run metadata and applies to signed and long-only
schedules; signal-instruction helpers still require next-session-close execution.


## External P&L series and historical tail risk

- `series_performance` sets NAV to initial capital plus cumulative net P&L, with
  no external flows, and each simple return to interval P&L over the previous NAV.
  NAV must stay positive. Initial capital is the first drawdown observation.
  Intervals must be ordered and contiguous; weekends and holidays are not gaps, but
  without a calendar a skipped trading session cannot be detected.
- Historical VaR is the signed value at descending rank `ceil(c*n)`; ETL is the
  mean of every value at or below it, including ties. Negative values are losses
  and are not converted to positive magnitudes. Dollar VaR/ETL apply the same
  rule to interval P&L independently; they are not return VaR times a NAV.
- Calendar validation is opt-in through explicit exchange identifiers. Daily
  intervals run from one session to the next session of ONE reporting calendar;
  several exchanges need an explicit union or intersection. Downloaded bars are
  checked against each asset's own calendar, expecting only sessions whose
  scheduled close precedes the validation time. Unknown listing dates make leading
  gaps uncertain rather than missing; suspensions require supplied explanations.
- `download_yahoo` keeps Yahoo Close (split-adjusted), Adj Close (split and
  distribution adjusted) and reported volume separate, uses inclusive
  exchange-local session dates, and never forward-fills or drops requested assets.


## Performance diagnostics, factor regression and signal evaluation

- CAGR uses the ending wealth multiple `W = E_T/C` from the report's compounded
  equity, where `C` is initial capital. Calendar time: `W^(D/days) - 1`, with
  `days` from the first `period_start` (when initial capital is invested) to the last
  `session`, and an explicit calendar-year length `D` (for example 365 or 365.25).
  Trading periods: `W^(A/n) - 1` for `n` holding intervals and explicit `A`. It is
  never the arithmetic annualized mean. `W <= 0` makes CAGR (and Calmar) undefined.
- Calmar is `CAGR/|max drawdown|` using the report's drawdown table, whose first
  observation is initial capital; a zero drawdown leaves it undefined.
- Skewness is the adjusted Fisher-Pearson `G1 = sqrt(n(n-1))/(n-2) * m3/m2^1.5`
  (n >= 3); excess kurtosis is `G2 = (n-1)/((n-2)(n-3)) * ((n+1)(m4/m2^2-3)+6)`
  (n >= 4), zero for a normal distribution. Both use periodic simple net returns and
  population central moments `m_k`. Constant returns are `zero_variance`.
- Short samples are computed and labeled with interval count and elapsed time; no
  minimum history is imposed and no statistical reliability is implied.
- Factor regressions are descriptive OLS fits with an intercept on exactly aligned
  holding intervals. Total-return series subtract a matched risk-free return only
  under the explicit `excess_returns` model; excess-return and long/short factors are
  never adjusted. Joint coefficients are conditional on the other factors and are
  reported separately from standalone single-factor betas and correlations.
- Annualized idiosyncratic volatility is `sqrt(A*SSE/(n-k-1))` for k factors and an
  intercept: the degrees-of-freedom-adjusted residual standard error, not the
  `ddof=1` standard deviation of fitted residuals (which divides by n-1).
- Rank-deficient designs (collinear or constant factors) return null coefficients;
  no pseudoinverse picks one of infinitely many solutions.
- Signal evaluation matches signals and supplied forward returns by signal date and
  asset. A forward interval may start at the signal session's close only under the
  declared same-close assumption; it must otherwise start later and always end after
  it starts. A later start excludes the intervening return by construction.
- Rank IC is a per-date Spearman correlation with average ranks for ties. Summaries
  average per-date values over evaluated dates; asset-date pairs are never pooled or
  counted as independent time observations. Overlapping horizons are flagged.
- Quantile returns are equal-weight means of supplied simple forward returns. Group
  ties are broken by asset identifier or the date is rejected, separately from the
  average-rank IC rule. Spreads are analytical, not executable portfolio returns.

## Two-strategy information-ratio allocation and analytical combination

- Inputs are periodic simple returns on net equity of two existing strategies.
  Log returns are never converted silently, and dollar P&L from differently sized
  accounts is never pooled; source capital is provenance only.
- Strategies and the common benchmark match on both holding-interval endpoints
  inside an explicitly declared window. Nothing is intersected, forward-filled or
  truncated; rows outside the declared window are counted, and an interval crossing
  its boundary raises.
- Active return `a_i,t = r_i,t - b_t`. Objective
  `IR(w) = sqrt(A) * w'mean(a) / sqrt(w' Sigma w)` with `ddof=1` sample covariance and
  explicit `A`. Weights are nonnegative, sum to one and respect explicit per-strategy
  bounds; infeasible bounds raise. The benchmark is subtracted once from the
  combined return because weights sum to one; a zero benchmark makes active returns
  equal strategy returns.
- `zero_correlation` keeps active variances and sets the covariance to zero. It is a
  counterfactual assumption about the objective's series (active returns, not
  necessarily raw returns). Weights chosen under it are also evaluated under the
  empirical covariance, i.e. on the observed combined returns, and both values are
  reported separately.
- The maximum is global over the feasible interval: endpoints plus the single
  interior stationary point of the linear first-order condition. IR itself is
  maximized, never squared or absolute IR, so negative means are handled correctly.
  Ties choose the smallest weight on the alphabetically first strategy.
- Zero tracking error (relative threshold `1e-9` of `w1*vol1 + w2*vol2`) at a
  feasible weight with positive mean active return makes IR unbounded; zero
  tracking error everywhere leaves it undefined. Both return null weights with a
  status. Undefined IR is never zero and no shrinkage or regularization is added.
  Near-singular covariances are flagged but remain valid finite estimates.
- A combination applies fixed weights every period: a periodically rebalanced
  allocation between already-costed strategy sleeves, not drifting buy-and-hold
  sleeves. No reallocation costs, netting, shared collateral or financing offsets
  are modeled, and scaled returns do not recompute impact or borrowing, so fixed
  dollar targets and nonlinear costs may not scale proportionally. It is not an
  executable combined-account backtest. Illustrative equity is labeled as such.
- Fitted weights are applied to an evaluation window without re-estimation. A
  window equal to the estimation sample is labeled in-sample; partial overlap is
  rejected unless explicitly identified; earlier windows raise. Later windows are
  never claimed out-of-sample, since prior inspection cannot be detected.

