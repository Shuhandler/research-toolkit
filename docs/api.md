# Implemented API: milestones 1A–1C

Use `import research_toolkit as rt`. The five implemented functions are
`prepare_market_data`, `returns`, `cumulative_returns`, `equal_weights`, and
`buy_and_hold`. Result and configuration objects are concrete dataclasses; their
tables are Polars DataFrames. See the runnable
[offline example](../examples/unlevered_buy_and_hold.py) for complete inputs.

Short positions, scheduled trading, performance ratios,
benchmark comparisons, and plotting are **not implemented**. The larger API in
[architecture](architecture.md) remains a design proposal where explicitly labeled.

## Validate market data

```python
market = rt.prepare_market_data(
    prices=prices, sessions=sessions, splits=splits, dividends=dividends,
    metadata=metadata, missing="raise",
)
```

All inputs must be eager Polars tables with exactly the following columns/types.
Create empty action tables with these schemas; an untyped empty DataFrame fails.
Numeric columns must be Float64, dates must be Date, and close times must be
Datetime with microsecond precision and UTC timezone. No automatic type conversion,
deduplication, missing-price filling, or row dropping occurs.

| Table | Schema | Key |
| --- | --- | --- |
| `prices` | `session: Date`, `asset: String`, `close: Float64` | `(session, asset)` |
| `sessions` | `session: Date`, `close_at: Datetime("us", "UTC")` | `session` |
| `splits` | `action_id: String`, `asset: String`, `effective_session: Date`, `ratio: Float64` | `action_id` |
| `dividends` | `action_id: String`, `asset: String`, `ex_session: Date`, `pay_date: Date`, `cash_per_share: Float64` | `action_id` |

Prices must be positive and finite. All price assets require a row on **every**
supplied session, including assets later assigned zero portfolio weight. Asset
currencies must be declared for exactly that universe. Session close times must
increase, and their local dates must agree with the session labels. The library
checks against the supplied calendar; it does not independently verify exchange
holidays or detect sessions omitted from that calendar. Use an authoritative
calendar when preparing real inputs.

Ratios are new/old shares. Cash dividend amounts are per post-split share; a split
precedes a dividend when both are effective on one session. Action identifiers
must be globally unique; multiple splits on one asset/session are rejected as
ambiguous. Pay dates cannot precede ex-dates and may fall outside the price window
or on nontrading dates. All effective/ex-sessions must lie inside the supplied
calendar. Unsupported actions cannot be encoded as extra columns and ignored.

Required metadata is a JSON-compatible mapping (ISO dates/timestamps as strings):

| Key | Contract |
| --- | --- |
| `source` | Nonblank source/snapshot description |
| `retrieved_at` | ISO timestamp with explicit UTC offset or `Z` |
| `currency`, `asset_currencies` | Base currency and exact `{asset: same_currency}` mapping |
| `calendar`, `calendar_version`, `timezone` | Nonblank calendar provenance/version and valid IANA zone |
| `price_basis` | `raw`, `split_adjusted`, or `total_return_adjusted` |
| `frequency` | Exactly `1d` |
| `coverage_start`, `coverage_end` | ISO dates matching first and last supplied sessions |
| `actions_complete` | Explicit `True`, including when both action tables are empty |
| `dividend_basis` | Exactly `post_split_share` |

Add source identifiers, transformation descriptions, file hashes, redistribution
restrictions, and source code revision as additional JSON metadata when available.
`actions_complete=True` is the caller's assertion that supported actions and their
payment dates are complete; validation cannot discover omitted source events.

The returned `MarketData` has owned table copies, copied metadata, a `diagnostics`
table (`code`, `table`, `count`), and `snapshot_id`. Rows are canonically sorted and
sorting is reported. The SHA-256 snapshot identity covers normalized table schemas,
values, and metadata; it is invariant to input row order. Source metadata changes
change the identity. Snapshot file persistence is deferred to 1D.

Polars tables and dictionaries remain mutable even inside frozen dataclasses.
Consumers revalidate `MarketData` and reject changes that no longer match its
identity. Call `prepare_market_data` again to intentionally change a snapshot.

## Return arithmetic

```python
price_returns = rt.returns(market, method="simple", basis="price")
summed = rt.cumulative_returns(price_returns, method="sum")
compounded = rt.cumulative_returns(price_returns, method="compound")
wealth = rt.cumulative_returns(price_returns, method="wealth")
```

`returns` accepts validated `MarketData`, rather than a bare price table, so it
can check the calendar and adjustment basis. `method` is `simple` or `log`.
`basis="price"` accepts raw/split-adjusted inputs; `basis="total_return"` requires
a declared total-return-adjusted series. Raw price returns retain split jumps and
produce a `raw_price_split_discontinuity` diagnostic when splits are present.
They do not calculate corporate-action-aware asset total returns; use the ledger
or a correctly documented adjusted input for those economics.

`ReturnResult.values` has `session`, `asset`, `period_start`, and `simple_return`
or `log_return`. The first return and period start for each asset remain null.
One observation is valid and produces only this structural null; empty market
panels fail validation. No annualization is inferred or performed.

`cumulative_returns` preserves the original return columns and adds:

| Method | Simple input | Log input |
| --- | --- | --- |
| `sum` | `cumulative_simple_return` | `cumulative_log_return` |
| `compound` | `compounded_return = product(1+r)-1` | `compounded_return = exp(sum(log_r))-1` |
| `wealth` | `wealth_multiple = product(1+r)` | `wealth_multiple = exp(sum(log_r))` |

Leading nulls remain null. Result metadata stores the method, basis, frequency,
units, source identity, sessions, and assets. Cumulative transforms reject interior
nulls, missing/duplicate keys, broken interval links, invalid return kinds, and
nonfinite or unrepresentable wealth. Standalone simple returns must exceed −1;
default/bankruptcy handling is not implemented by these positive-price utilities.

## Allocation and simulation

```python
weights = rt.equal_weights(["A", "B"])  # Or {"A": 0.4, "B": 0.6}.
result = rt.buy_and_hold(
    market, weights=weights, initial_capital=1_000.0,
    entry_session=entry_date, end_session=end_date,  # datetime.date values
    policy=rt.BuyHoldPolicy(
        execution="entry_close", sizing="post_cost_equity",
        fractional_shares=True, initial_gross_leverage=1.0,
        terminal_action="mark_only",
    ),
    costs=rt.TradeCosts(commission_bps=5.0, half_spread_bps=2.0, impact_bps=0.0),
    cash_rate=0.0, cash_day_count="ACT/365F",
)
```

Policy fields and cost/rate inputs are required, including explicit zero rates.
Initial exposure must be nonnegative: zero is all cash, one is fully invested
after costs, intermediate values retain cash, and values above one borrow through
an explicit `Financing` configuration. The `cash_rate`/`cash_day_count` pair shown
above remains supported for unlevered calls. Omitting all rate configuration fails;
there is no inferred cash or loan rate.

## Financing and leverage

Keep the same weights, data, and entry/exit sessions, set the desired initial
exposure in `BuyHoldPolicy`, and replace the cash-rate pair with `financing`:

```python
result = rt.buy_and_hold(
    market, weights=weights, initial_capital=1_000.0,
    entry_session=entry_date, end_session=end_date,
    policy=rt.BuyHoldPolicy(
        execution="entry_close", sizing="post_cost_equity", fractional_shares=True,
        initial_gross_leverage=2.0, terminal_action="mark_only",
    ),
    costs=rt.TradeCosts(commission_bps=5.0, half_spread_bps=2.0, impact_bps=0.0),
    financing=rt.Financing(
        cash_rate=0.02, borrowing_rate=0.08, day_count="ACT/365F",
        maintenance_equity_ratio=0.4, on_breach="stop", cash_sweep="repay_debt",
    ),
)
result.require_complete()  # Raises if the run stopped, including on the final date.
result.daily.select("session", "equity", "debt", "cash", "gross_leverage", "equity_ratio")
```

These example rates and threshold are modeling assumptions, not library defaults
or broker rules. Every `Financing` field is required. Cash/borrowing rates must be
finite nonnegative annual nominal fractions; only `ACT/365F` is supported.
`maintenance_equity_ratio` must lie in `(0, 1]`; it may be explicit `None` only
when initial exposure does not exceed 1. `on_breach="stop"` and
`cash_sweep="repay_debt"` are the supported policies. Supplying `financing` together
with either non-None legacy cash-rate argument raises rather than choosing one.

Gross initial notional `N=L*C/(1+L*k)` targets exposure relative to post-cost equity.
Here C is starting equity, L initial leverage, and k the weighted proportional
entry cost. The actual order basket's funding need determines the loan; a borrowing
event credits cash and debt **before** entry trades and costs. Borrowing is not P&L.
Initial funding must leave positive equity and satisfy the maintenance threshold;
otherwise the call raises instead of returning an invalid entry.

Once per calendar date after entry, credit opening cash at `cash_rate/365` and
capitalize opening debt at `borrowing_rate/365`. Weekend/holiday dates are included
without fabricating market-return rows. Borrowing-interest cost rows have null
trade/asset IDs because financing is not a trade fee. Debt can increase from posted
interest; no new purchases or daily leverage resetting occur.

After that date's interest and corporate-action payments, available cash repays
debt. Receivables cannot fund a repayment before payment. Repayment reduces cash
and debt equally and is not a second expense. Excess cash remains available and
earns interest from the following date. Quantities change only through splits.

At each session close, compare `equity_ratio = equity/gross_exposure` with the
configured threshold. Equality is compliant; a relative 1e-12 tolerance handles
roundoff at the boundary. All-cash equity ratio is null and no margin test applies.
The ratio uses receivables in equity and risky holdings in gross exposure, exactly
as the balance sheet defines them. This is daily research margin monitoring,
not intraday or broker-specific margin enforcement.

If maintenance fails or equity is nonpositive, the run retains that close and
stops. `status="stopped"`, `stop_reason` (`maintenance_margin_breach` or
`nonpositive_equity`), `stop_session`, and `stop_time` identify the failure.
Insolvency takes precedence if both conditions occur. Metadata retains requested
`end_session` and adds `actual_end_session`; a failure at the requested end is still
a stopped run. No later charge, payment, mark, or automatic liquidation occurs.

The final P&L and simple return can represent losses at or beyond −100%; they are
not clipped. For nonpositive equity, log return, gross leverage, and position
equity weights are null. The signed compounded return still reconciles to ending
equity/initial capital minus one. Completed results have null stop fields.
`require_complete()` returns the result when complete and raises `ValueError`
otherwise. Call it before full-period comparisons. Partial analysis must explicitly
show actual coverage and the stop reason. Full performance reports remain for 1D.

## Shared allocation and accounting rules

Weights are risky-asset proportions, not weights of net equity. Accept either an
asset mapping or a Polars table (`asset: String`, `weight: Float64`). They may
select a subset of the market universe; unknown assets, negative weights, duplicates,
and nonfinite values fail. Weights must sum to 1 within absolute tolerance 1e-12.
Only representational roundoff within that tolerance is normalized; both supplied
and resolved values are saved in metadata. There is no economic renormalization
of arbitrary sums. Zero weights create no trades. Even an all-cash run takes a
valid allocation template. `equal_weights` returns a sorted Polars weight table.

Cost components are nonnegative scalar bps or mappings covering exactly the
allocation assets (including explicit zero-weight assets). They are recorded as
**modeled** proportional per-side cash expenses, not measured liquidity data.
Half-spread is already the one-sided cost. No adverse fill-price adjustment is
added on top. Fixed/minimum fees and nonlinear impact are not supported.

The implementation follows [financial conventions](financial-conventions.md):

- Predetermined dollar allocations execute at the chosen entry close; post-cost
  sizing funds entry expenses and any explicitly configured loan. The caller is responsible for
  choosing weights before entry; the library cannot verify how a static mapping
  was researched.
- Share quantities stay fixed except for splits. Dividends accrue on ex-date,
  transfer to cash on pay date, and are not automatically reinvested. Entry on an
  ex-date does not receive that dividend. Unpaid receivables remain in equity.
- Cash interest accrues once per calendar date after entry, including weekends,
  from the previous date's cash balance at nominal `cash_rate/365`. It is credited
  before that date's dividend payments. There is no interest before cash receipt.
- The first daily interval opens at original capital **before fees**. Its P&L
  includes entry costs; later intervals open at the previous closing equity.
  Pre/post-entry valuations are recorded separately. No terminal sale is assumed.
- The ledger checks shares against event deltas, cash/debt/receivable balances against
  events, P&L against attribution, cumulative P&L against equity change, and
  compounded returns against equity. Material residuals raise `ArithmeticError`.

## Backtest outputs and limits

`BacktestResult` exposes the following Polars tables:

| Table | Main fields / meaning |
| --- | --- |
| `daily` | Session/period start, opening and ending equity, dollar P&L, simple/log returns, cumulative P&L and simple-return sum, compounded return, cash, debt, receivables, exposure, leverage, equity ratio, margin-breach flag, closing drawdown, entry-cost flag |
| `positions` | Session/asset, quantity, raw mark, value, net-equity weight; includes entry session |
| `trades` | Trade ID, session/UTC close, asset, signed quantity/notional, reference price, execution policy, cost |
| `costs` | Cost ID, date/time, nullable trade ID/asset, component, amount, modeled basis; nonzero trade costs and calendar-date borrowing interest |
| `events` | Stable event ID/sequence, calendar date, nullable UTC time, phase, type, linked IDs, quantity/cash/debt/receivable deltas |
| `valuations` | Pre-entry, post-entry, and subsequent close balance sheets |
| `attribution` | Session, component, asset (null for account interest), dollar P&L; sums to daily P&L |
| `receivables` | Session/action, ex/pay dates, entitled shares, original amount, outstanding amount; includes paid entitlements with zero outstanding |
| `diagnostics` | Session, reconciliation code, residual, currency tolerance |

Date-only interest/action events have null `time` and `phase="before_close"`;
they are modeled daily events, not invented observed timestamps. Sequence follows
cash/borrowing interest → splits → ex-date accrual → pay-date cash → debt repayment.
Entry funding/trades and stop events have their supplied closing timestamp;
stop events have `phase="close"` and zero balance deltas. No market-return rows
are created on weekends.

Currency tolerances are `min(0.009, 1e-8 + 1e-12 * scale)`; quantity checks use
1e-12 absolute/relative tolerance. Tiny entry cash roundoff is explicitly reported,
and fully invested cash is set to zero only after passing that check. All other
balances remain auditable; no material cash deficits are hidden.

Metadata records source/snapshot identity, resolved weights/policy/cost rates,
cash/borrowing rates, day count, financing configuration, stop status/coverage,
dividend/return conventions, currency, dates, package/Python/
Polars versions, immediate settlement and excluded taxes. No Git command or
filesystem/network access occurs during simulation. Record a commit/dirty state
externally alongside saved outputs if needed for development-run reproduction.

Daily drawdown uses a peak starting at original capital. The post-entry loss is
visible in `valuations` even if it recovers before the first interval close;
full drawdown reporting over all valuations belongs to 1D. Numerical overflow,
invalid input/funding, or failed reconciliations raise; they are not disguised as
ordinary margin stops. Stopped runs preserve real finite negative/zero equity.

Runtime dependency: Polars only. No plotting, provider, ML, annualization, or
benchmark code is imported or invoked. All tests and the example run offline.
