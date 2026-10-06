# Fixed dollar targets and dated costs

Implemented through `rt.scheduled_rebalance`; no new simulator is needed. See the
[offline notebook](../examples/fixed_dollar_rebalancing.ipynb) for a complete
three-basket example with numerical audits and plots.

## Target amounts

For signed portfolios, supply exactly these Polars columns, in this order:

| Column | Type | Meaning |
| --- | --- | --- |
| `decision_session` | Date | Session when the basket was determined |
| `session` | Date | Execution session |
| `asset` | String | Stock, ETF, or other supported asset identifier |
| `target_notional` | Float64 | Signed currency amount at execution |

Positive means long, negative means short, and zero means exit. Every basket must
include the same asset universe, including old holdings, explicit zero exits, and
any hedge. Amounts are finite and use the market's declared currency. Do not also
supply `quantity`, `equity_exposure`, `weight`, or `gross_leverage` columns.

The initial basket executes at `entry_session`. Decision and execution dates must
be supplied sessions and decision must precede execution by default; the explicit
`decision_timing="same_session_close_assumed"` policy permits equality. The table supports any
later supplied execution session; to execute at the next close, explicitly choose
the next row of the supplied calendar. No orders execute on `end_session`.
That final session remains mark-only, including when its price would be useful
for an intended rebalance. Supply a later end session if that trade is needed.

```python
import polars as pl
import research_toolkit as rt

# decisions and baskets are supplied by the notebook; amounts are already fixed.
sessions = market.sessions["session"].to_list()
targets = pl.DataFrame([
    {"decision_session": d, "session": sessions[sessions.index(d) + 1],
     "asset": asset, "target_notional": float(amount)}
    for d, basket in baskets.items() for asset, amount in basket.items()
], schema={"decision_session": pl.Date, "session": pl.Date,
           "asset": pl.String, "target_notional": pl.Float64})
```

For each asset, the engine sizes `target_notional / raw_execution_close`, and
trades against its actual pre-trade shares. Drift, splits, prior trades and
completed dividend reinvestments are already in those holdings. A change from
-$100 to +$100 at a fixed price trades $200. A genuinely unchanged holding trades
zero. Prices between decision and execution do not change the requested dollars,
and the new shares cannot earn returns before execution.

These three signed modes have different meanings:

| Target column | Sizing at execution | Do costs shrink the target? |
| --- | --- | --- |
| `equity_exposure` | Multiple of post-cost equity | Equity changes the dollar position |
| `quantity` | Exact signed shares in execution-date units | No |
| `target_notional` | Fixed signed currency amount divided by raw close | No |

The `RebalancePolicy.sizing="post_cost_equity"` field remains required for API
compatibility. The target column selects signed sizing; dollar targets bypass the
equity-target solve. `receivable_policy="require_target"`, explicit financing,
`LongShortPolicy` and `StockBorrow` are required even for an all-long dollar basket.
Use empty borrow rates for a universe that is never short. `max_asset_weight`
limits each absolute target value divided by gross exposure, not equity.

Costs reduce equity and use cash or the declared loan. They never rescale the
dollar target. Restricted short collateral, receivables, dividend liabilities,
borrow fees, gross rebates and SOFR/fixed debit financing retain their existing
[accounting](long-short.md). This research financing contract does not impose a
broker credit limit or verify borrow availability. Unsupported inputs, insufficient
equity to pay costs, and an entry margin violation raise. An existing pre-trade
margin breach stops before trading; a breach after a scheduled basket stops with
those actual fills retained. Targets are never replaced to avoid a stop.

Immediately after a successfully executed basket, gross asset exposure equals the
sum of absolute requested dollars within reconciliation tolerance. This is separate
from equity, collateral and debt, and does not guarantee beta neutrality. Gross
exposure and leverage can drift between scheduled trades.

## One cost model per decision

Pass a mapping of exact Python `date` keys to `rt.SquareRootImpactCosts` as `costs=`.
Its keys must equal the set of all basket decision dates, including initial entry;
missing **or extra** keys raise. There is no stale-model fallback or automatic refresh.
The same decision can serve multiple later execution baskets with the same model.

```python
asset_returns = rt.returns(market, method="simple", basis="price")
models = {
    d: rt.SquareRootImpactCosts(
        liquidity=rt.estimate_liquidity(
            asset_returns, dollar_volume, decision_session=d, lookback=lookback,
            metadata=volume_metadata,
        ),
        commission_bps=commission_bps,
        commission_per_share=commission_per_share,
        half_spread_bps=half_spread_bps,
        impact_coefficient=impact_coefficient,
    ) for d in baskets
}
run = rt.scheduled_rebalance(
    market, targets=targets, initial_capital=capital,
    entry_session=targets["session"].min(), end_session=end,
    policy=rebalance_policy, costs=models, financing=financing,
    long_short=long_short_policy, stock_borrow=stock_borrow,
).require_complete()
```

Use an explicitly selected analytical return basis when raw prices contain splits
or dividends; a corporate-action price jump must not masquerade as daily volatility.
`estimate_liquidity` preserves the chosen return basis in its provenance.

Each model covers exactly the complete target universe, **including assets being
closed, zero rows, and the hedge**. Its currency must match the market. Estimation
uses returns ending strictly before `decision_session` and dollar volume published
by that session's local midnight. An after-close decision therefore still excludes
the decision day's observations: this is a deliberate conservative lag. Execution
prices are used only for quantities and per-share fees, never as input to the
frozen daily volatility or dollar ADV estimates.

For a manually supplied `LiquidityResult`, dated selection additionally requires
ISO `sample_start`, `sample_end`, `decision_session`, and timezone-aware
`decision_at` metadata. `returns` now retains `source_session_closes`, which `estimate_liquidity` carries
into its provenance. This permits estimation history before the execution market
starts without requiring historical action coverage in the execution ledger.
The source calendar must match the declared return sessions, contain timezone-aware
chronological closes, and agree with execution closes wherever they overlap.
Legacy manually supplied estimates without this metadata use the execution calendar.
Sample endpoints must belong to that source calendar (or the legacy execution calendar), with
`sample_start <= sample_end < decision_session`. The declared availability cutoff
must follow the sample-end close and be no later than the decision close. The
model's `decision_session` must exactly equal its mapping key. These checks cannot
independently prove the vintage of externally supplied estimates; retain honest
source provenance and use the estimator's availability checks for supplied inputs.

All requested models are validated before simulation, even if the run later stops.
A frozen binding is selected once per execution basket and used for all its sizing
and actual costs. Nonlinear costs use absolute **changed** notionals, including
sales and covers; a preview does not add another charge. Financing and borrow fees
are separate. Long dividend reinvestment remains free and requires no model for its
payment day. Ordinary scheduled trades funded by dividends still incur costs.

`daily_volatility` is a daily decimal sample standard deviation; `dollar_adv` is
currency per trading day, not shares. `order_adv_ratio = abs(order)/dollar_adv` is
not intraday participation. Impact-only bps exclude spread and commissions.

One static `TradeCosts` or `SquareRootImpactCosts` continues to work unchanged;
static snapshots are not refreshed. Dated mappings also work with existing signed
quantity/equity targets and long-only scheduled weights. Buy-and-hold and standalone
`estimate_trade_costs` / `size_entry_orders` still take a single model; select
`models[decision]` explicitly for a preview. Mutating a constructed model's nested
estimates or parameter mappings raises; construct a new model for changed inputs.

## Audit tables

All existing results remain available. Additional fields are additive:

- `targets` retains requested decision/execution dates and dollar targets, including
  baskets not executed because of a stop.
- `target_executions` contains one row per asset per executed dollar basket:
  `decision_session`, `session`, `asset`, `target_notional`, `pre_trade_quantity`,
  `pre_trade_notional`, `reference_price`, traded `signed_quantity`, final `quantity`,
  `actual_notional`, `notional_residual`, `tolerance`, `trade_id`, `trade_cost`.
  Unchanged holdings have zero trade quantity/cost and a null trade ID. Other modes
  receive a typed empty table.
- `turnover` includes initial and subsequent gross absolute traded notional divided
  by pre-trade equity. `rebalances.target_gross_leverage` is null for dollar/share
  targets; `actual_gross_leverage` still reports exposure / post-cost equity.
- `cost_model_selections` contains each requested `decision_session`, execution
  `session`, `model_id`, `snapshot_id`, `sample_start`, `sample_end`, and
  `status` (`executed` or `not_executed`). A zero-trade basket is executed. Fixed-bps
  costs receive a typed empty table; a legacy static snapshot can have null sample
  start. Buy-and-hold static selection has a null decision date.
- `execution_costs` adds `decision_session`, `snapshot_id`, and `model_id` to its
  existing per-trade numerical components. Snapshot identity covers estimates and
  provenance; model identity additionally covers normalized cost assumptions.
- `metadata["cost_models"]` records each binding's full estimates, source metadata,
  sample/decision/execution dates, units and assumptions. `metadata["cost_model"]`
  remains the initial binding for backward compatibility. `cost_model_selection`
  states `static` or `exact_decision_date`.

Dollar residual is `actual_notional - target_notional`. For each asset, tolerance
is `min(0.009, 1e-8 + 1e-12 * max(initial_capital, abs(target_notional)))` in currency.
Gross targets and ledger balances also reconcile using the existing bounded dollar
tolerance; the existing 1e-13 relative no-trade filter is retained. A residual
outside tolerance raises, rather than returning a silently different target.

Existing notebooks need no migration. For fixed dollars, replace the signed target
value column with `target_notional`; do not precompute future share quantities or
normalize by equity. For changing costs, replace the static `costs` argument with
the decision mapping. Select columns by name if consuming the expanded audit tables.
Always preserve stopped-run coverage; call `require_complete()` for a full report
or explicitly opt into partial reporting.


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
