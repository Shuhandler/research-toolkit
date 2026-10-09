# Security lifecycles and corporate actions

Implemented on the existing shared ledger: `buy_and_hold`, `scheduled_rebalance`
and signal-driven schedules apply the same action code. Inputs are provider-
independent Polars tables validated by `prepare_market_data`; nothing is
downloaded or inferred from ticker strings. Markets prepared **without** these
tables keep every earlier rule, identity and numerical result.

The [runnable example](../examples/corporate_actions.py) holds a parent through a
spin-off with delayed delivery, a target through a stock-and-cash merger, and a
short through a cash acquisition with delayed settlement.

## Notebook workflow

```python
import research_toolkit as rt

lifecycle = rt.corporate_action_inputs(
    securities=[dict(asset="PARENT", security_type="equity", first_session=start,
                     unquoted_valuation="none", source="security master"), ...],
    aliases=[dict(asset="PARENT", ticker="PRNT", exchange="XNYS", valid_from=start,
                  valid_to=rename_eve, source="listing notice"), ...],
    actions=[dict(action_id="SPIN", action_type="distribution", source_asset="PARENT",
                  announced_at=announced_utc, effective_date=ex_date, record_date=record,
                  entitlement_basis="ex_date", classification_source="issuer notice"), ...],
    legs=[dict(action_id="SPIN", leg_id="child", leg_type="security", asset="CHILD",
               units_per_source_share=0.25, due_date=distribution_date,
               fraction_policy="cash_in_lieu", cash_in_lieu_price=cil_price), ...],
)
market = rt.prepare_market_data(prices=prices, sessions=sessions, splits=splits,
                                dividends=dividends, metadata=metadata, **lifecycle)
result = rt.buy_and_hold(market, weights=weights, ..., corporate_actions=rt.CorporateActionPolicy(
    distributed_securities="retain", short_obligations="cash_at_delivery_close",
    pending_claim_margin=1.0, obligation_margin=1.0, warrant_margin=1.0,
    reorganization_fee=0.0))
result.require_complete()
result.corporate_actions      # Per-event audit.
result.security_claims        # Pending consideration and undelivered securities by close.
rt.security_status(market)    # Dated eligibility, aliases and announced actions.
rt.security_returns(market)   # Quoted-price versus economic total returns.
```

`corporate_action_inputs` only converts lists of row dictionaries into typed tables
(omitted nullable fields become null). All validation happens in
`prepare_market_data`. Every value above is a caller input, never a default.

## Input tables

`asset` is the **stable security identity** used by prices, positions and actions.
Tickers and exchanges are dated aliases and never move holdings.

| Table | Columns (nullable in *italics*) | Rules |
| --- | --- | --- |
| `securities` (required) | `asset`, `security_type` (`equity`/`warrant`), `first_session`, *`last_session`*, *`terminated_date`*, `unquoted_valuation` (`none`/`supplied_mark`), `source` | Must cover every priced asset and every `asset_currencies` key. A terminated security needs `last_session < terminated_date` and a terminal action (acquisition, or warrant expiry). |
| `suspensions` | `asset`, `first_session`, `last_session`, `reason`, `source` | Supplied sessions; non-overlapping per asset. |
| `aliases` | `asset`, `ticker`, `exchange`, `valid_from`, *`valid_to`* | Non-overlapping per security **and** per (ticker, exchange), so a reused ticker cannot join two securities. |
| `corporate_actions` | `action_id`, `action_type`, `source_asset`, `announced_at` (UTC), `effective_date`, *`record_date`*, `entitlement_basis`, *`provider_label`*, `classification_source` | Types `cash_acquisition`, `stock_acquisition`, `distribution`. IDs unique across splits, dividends and actions. |
| `action_legs` | `action_id`, `leg_id`, `leg_type` (`cash`/`security`), *`asset`*, *`units_per_source_share`*, *`cash_per_source_share`*, *`accrual_per_day`*, *`accrual_start`*, *`accrual_end`*, `due_date`, *`fraction_policy`*, *`cash_in_lieu_price`* | Cash legs: cash per source share plus an optional explicit accrual. Security legs: units per source share, `fractional` or `cash_in_lieu` (with an explicit price). |
| `warrants` | `asset`, `underlying_asset`, `units_per_warrant`, `exercise_price` (per warrant), `exercisable_from`, `expiry_date`, `exercise_style` (`american`/`european`), `settlement` (`physical`), `source` | One row per warrant security; `terminated_date` must be the day after expiry. |
| `valuation_marks` | `session`, `asset`, `mark`, `method`, `source` | Only for securities with `unquoted_valuation="supplied_mark"`, only on sessions without a market quote. Positive, or zero for warrants. |

Run-level input `warrant_exercises`: `exercise_id`, `decision_session`, `session`,
`warrant`, `warrants`, `delivery_date`, `fee` (schema
`rt._lifecycle.EXERCISE_SCHEMA`). The decision precedes the exercise session, which
must be inside the warrant's exercise window and before the terminal session.

### Dates are not interchangeable

- `announced_at` is when the terms were available; it must precede the start of
  `effective_date`. `security_status` uses it so universes never see later news.
- `effective_date` is the entitlement/effectiveness date, applied at the start of
  that calendar date before its session. For acquisitions the source is
  extinguished then. For distributions it is the **ex-date**.
- `record_date` is informational and validated: regular-way (`entitlement_basis=
  "ex_date"`) needs `record_date >= ex-date`; `due_bill` needs `record_date <
  ex-date`. In both cases the holder at the start of the ex-date is entitled;
  pre-ex quotes include the entitlement. `entitlement_basis="record_date"` is
  rejected: record-date holdings do not determine entitlement safely.
- Each leg's `due_date` is its own cash settlement or security delivery date.
  For a due bill, supply the redemption/delivery date the holder actually receives.

### Lifecycle-aware price validation

A quote is expected for an asset exactly when the session is on or after
`first_session`, on or before `last_session` (if any), outside suspensions and
before `terminated_date`. A supplied price anywhere else raises with the cell's
status (`not_listed`, `suspended`, `unquoted_after_last_session`, `terminated`); an
expected but absent quote raises as **missing data**. Prices stay strictly positive;
nothing is forward-filled, invented, dropped or replaced with zero. Lifecycle inputs
require `price_basis="raw"`, because adjusted series already embed an unknown action
treatment.

Further validation rejects: unknown or unsupported action types, duplicate IDs,
two actions for one source on one date, splits coinciding with an action's
effective or delivery date for the affected securities, actions after an
acquisition, dividends or splits after termination, cash legs on distributions,
security legs without assets, missing cash-in-lieu prices, incomplete accruals,
deliveries of securities that terminate first, splits or dividend ex-dates of a
delivered security between effectiveness and delivery (pending claims are not
re-denominated; the same is enforced at run time for exercise deliveries),
underlying splits while a warrant is outstanding (adjusted terms are unsupported),
and warrant features other than physical cash-funded American or European exercise.

## Run policy

`CorporateActionPolicy` is required whenever an action, warrant or exercise can
affect the run's reachable securities (the strategy universe plus everything its
actions can deliver):

| Field | Meaning |
| --- | --- |
| `distributed_securities` | `retain` keeps delivered securities outside the target universe. `liquidate_at_first_permitted_close` sells (or covers) them at the first quoted, non-terminal close after delivery, with ordinary trade costs (proportional `TradeCosts` only). |
| `short_obligations` | `cash_at_delivery_close`: a short holder pays the delivered securities' value at the first quoted close on/after delivery. `borrowed_short_position`: the obligation becomes an ordinary borrowed short (requires `StockBorrow` coverage of the successor; never for warrants). |
| `pending_claim_margin` | Signed margin fraction on positive pending claims (1 = no margin credit). |
| `obligation_margin` | Signed margin fraction on pending cash or delivery obligations. |
| `warrant_margin` | Signed margin fraction on long warrant holdings (instead of `long_margin`). |
| `reorganization_fee` | Explicit cash fee per mandatory action applied to a nonzero position (0 for none). |

## Daily event order

For each calendar date after entry:

1. Cash interest and loan interest (fixed or SOFR), as before.
2. Stock-borrow fees on the prior close's short values, **skipping securities
   terminated on or before that date**; rebate on restricted collateral.
3. Splits, then dividend ex-date entitlements (unchanged).
4. Warrant expiry (start of the day after `expiry_date`), then acquisitions and
   distributions in `action_id` order. Entitlement uses holdings at that moment.
   Each action is applied exactly once.
5. Dividend payments, then claim cash settlements and security deliveries due that
   date (deliveries due on a date are not entitled to that date's actions).
6. Reinvestment release and the debt sweep.

At a session close:

7. If any held security or pending security claim has neither a market quote nor
   a supplied mark, roll back to the last valued close and **stop** (below).
8. Price P&L on holdings, marks on pending security claims.
9. Short obligations due for cash settlement settle at this close.
10. Signed portfolios re-mark restricted collateral, then sweep.
11. Explicit warrant exercises, then the selected liquidation of distributed
    securities (both skipped if the account already breaches margin).
12. Scheduled basket or dividend reinvestment, then the closing margin check and
    reporting, as before.

## Accounting

**Claims.** Pending consideration is a *cash claim* carried at **face value**:
undiscounted, with no credit risk and no interest. Undelivered securities are
*security claims* carried at the security's market close, or at an explicitly
supplied mark. Positive claims are assets; negative claims are obligations of
short holders. Claims are separate from holdings: they cannot be traded, sold or
exercised before delivery. Equity is

`holdings + cash + collateral + dividend receivables - dividend liabilities - debt
+ cash claims + security claims`,

and every close reconciles cash, debt, collateral, receivables, liabilities, claim
cash and per-security claim quantities against their posted events.

**Acquisitions.** At the effective date the source position (long or short) is
extinguished. Each cash leg creates a claim of `quantity × final cash per share`
(including any explicit accrual, `cash_per_source_share + accrual_per_day ×
days(accrual_start, accrual_end)`), settled on its due date. Each security leg
creates a claim of `quantity × units` successor shares (whole units plus a
cash-in-lieu claim when selected), delivered on its due date into the successor
holding. No trade, commission, spread or impact is booked. Borrow fees stop on the
effective date. Earned unpaid dividends remain receivables or liabilities.

**Distributions** (spin-offs, security distributions, warrant distributions)
retain the parent and create child claims, delivered later or the same day. The
parent's raw quote drops; the child value appears as a separate claim or holding.
It is never a split, a cash dividend or extra wealth: with value-conserving prices
equity is unchanged.

**P&L attribution.** The derecognized source (its previous close value) and the
recognized claims (cash at face, securities at their previous close mark) give a
`corporate_action` component; a child with no earlier quote is recognized at its
first mark under the same component. Later changes in a claim's mark are
`claim_valuation`; delivered shares then earn ordinary `price_pnl`. Warrant
expiry (`warrant_expiry`), exercise (`warrant_exercise`, plus any
`warrant_exercise_fee`) and `reorganization_fee` have their own components. The
real economic effect of a deal premium or discount therefore appears once, and a
pure conversion at consistent prices adds nothing.

**Warrants** are separate securities. Distributions create warrant claims; delivered
warrants are valued at market or supplied marks only (never silently at zero or
intrinsic value) and can be sold through ordinary scheduled targets or the
liquidation policy. An explicit exercise pays `warrants × exercise_price` (free
cash first, then the loan; long-only runs without a maintenance ratio cannot
borrow and raise), removes the warrants at that close's mark and creates an
underlying claim (or immediate delivery) valued at the underlying's close.
Unexercised warrants lapse at zero after expiry. Automatic exercise or sale only
happens when selected. Short warrants are unsupported.

**Shorts.** Short holders owe the consideration: a negative cash claim, and a
security delivery obligation settled under the selected treatment. Restricted
collateral covers `collateral_multiple × (short values + obligations)` at each
close, so collateral changes when an obligation is created or settled, not when a
ticker disappears. Obligations are not borrowed shares and incur no borrow fee;
`borrowed_short_position` converts them into a borrowed short that does.

**Margin.** Signed portfolios use `long_margin × long equities + warrant_margin ×
long warrants + short_margin × shorts + pending_claim_margin × positive claims +
obligation_margin × obligations`. Long-only maintenance ratios treat undelivered
securities as risky exposure and cash claims like dividend receivables.

**Dividend reinvestment** keeps its zero-cost convention, but a payment for a
payer that is not quoted at the execution close (terminated, suspended) is released
to cash with status `payer_not_quoted`.

## Strategies and universes

- Mandatory actions happen when economically effective; strategy targets only use
  what the caller supplies by their decision cutoff. `security_status` returns, per
  session, `status`, `tradable`, the valid alias and `announced_actions` known by
  that close, for building dated universes without later information.
- A target that requires trading a security not quoted at its execution close
  (terminated, suspended, not listed) is rejected as stale, before the run for
  nonzero targets and at execution otherwise.
- Scheduled and signed baskets trade only quoted strategy assets. Retained
  distributed securities and pending claims form an untraded sleeve: weight and
  equity-exposure targets size against post-cost equity **minus** a positive (long)
  sleeve; a net short sleeve (obligations, borrowed successors) never enlarges the base.
  Including a distributed security in the target universe makes it an ordinary
  strategy asset (an explicit zero target sells it).
- Proportional `TradeCosts` given as per-asset mappings must cover every reachable
  security (strategy assets plus everything their actions can deliver).
- Square-root cost models are bound to the strategy assets quoted at each
  execution; supply dated models whose liquidity covers exactly those assets when
  the quoted universe changes.

## Unvalued positions

If a held position or pending claim cannot be valued at a session close (no quote,
no supplied mark), the run rolls back that interval and stops at the last valued
close: `status="stopped"`, `stop_reason="unvalued_position"`, `stop_session` = last
valued session, and `metadata["lifecycle"]["unvalued"]` names the session and
securities. `require_complete()` rejects it; partial reports need
`allow_partial=True`. No NAV is invented.

## Results

New tables: `corporate_actions` (date, action, stage, source and affected asset,
entitled quantity, quantity and cash amounts, claim ID, value, valuation method,
P&L, fee, trade ID, provenance) and `security_claims` (claims pending at each
close with mark, value, due date and settlement). New zero-filled columns:
`pending_cash_receivable`, `pending_cash_payable`, `pending_security_receivable`
and `pending_security_obligation` in `daily` and `valuations`; `claim_cash_delta`
and `claim_quantity_delta` in `events`. `positions` include every reachable
security, with a null `raw_mark` while unquoted. `metadata["lifecycle"]` records
the policy, event order, valuation and recognition conventions.
`performance` adds the four claim balances to its balance-sheet allocation, and the
exposure/allocation plots label them; gross exposure and leverage remain holdings
only, so nothing is counted twice.

## Analytical returns

`rt.security_returns(market)` returns `ReturnResult` rows with `price_return` (raw
quoted change) and `total_return`, which reinvests ex-date dividends and distributed
securities at the ex-date close and values acquisition consideration at face cash
plus successor closes on the first session on/after effectiveness. Statuses:
`ok`, `first_observation`, `quote_gap` (after a suspension; null return),
`unvalued_distribution`, `supplied_mark_valuation`, `acquisition_consideration`,
`unvalued_consideration`, `consideration_after_coverage`. It requires raw prices.
`rt.returns` still serves complete panels; on lifecycle markets with actions or
gaps it raises and points here. These are analytical series, not executable ledgers.

## Provider data

- `rt.adapters.resolve_aliases(observations, market_or_alias_table, ...)` maps
  provider rows keyed by dated ticker (and optionally exchange) to stable IDs; an
  unmatched or ambiguous row raises.
- `rt.adapters.yahoo_chart(..., provider_split_overrides=..., lifecycle=...)`: a
  sourced override reclassifies a provider "split" (for example a spin-off) so it
  is neither a ledger split nor a validation error, and raw factors must undo the
  provider's adjustment. With `lifecycle`, bars are expected only on quoted
  sessions, so supplied pre-delisting history is kept and later gaps stay visible.
  Downloads remain outside the simulator; unavailable history is not recoverable.

## Snapshots

`save_snapshot` writes format 1 (four tables) for markets without lifecycle inputs,
exactly as before, and format 2 (eleven tables) otherwise. `load_snapshot` reads
both and checks the same hashes, schemas and canonical identity. Existing saved
snapshots need no migration; legacy market identities are unchanged.

## Not supported

Rights offerings, tender offers, elections, proration, appraisal rights, CVRs,
spin-offs of non-equity assets other than warrants, cash distributions in a
distribution action (use dividends), cashless/net-share/cash-settled warrants,
strike or ratio resets, short warrants, multi-currency consideration, taxes,
intraday timing, discounting or credit risk of pending claims, interest on deal
cash, broker-specific reorganization fees or stock-loan buy-ins, and automatic
recognition of provider labels. Unsupported terms raise rather than being
approximated.
