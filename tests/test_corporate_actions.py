"""Hand-checkable lifecycle and corporate-action accounting on small synthetic markets."""
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
import math

import polars as pl
import pytest

import research_toolkit as rt

D = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5),
     date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)]
ZERO = rt.TradeCosts(commission_bps=0., half_spread_bps=0., impact_bps=0.)
ANNOUNCED = datetime(2023, 12, 1, tzinfo=timezone.utc)


def equity(first_session=D[0], last_session=None, terminated_date=None, unquoted_valuation="none"):
    return dict(security_type="equity", first_session=first_session, last_session=last_session,
                terminated_date=terminated_date, unquoted_valuation=unquoted_valuation, source="synthetic test lifecycle")


def make_market(series, securities, dates=D, splits=(), dividends=(), **tables):
    prices = pl.DataFrame([{"session": d, "asset": a, "close": float(p)}
                           for a, values in series.items() for d, p in zip(dates, values) if p is not None],
                          schema={"session": pl.Date, "asset": pl.String, "close": pl.Float64})
    sessions = pl.DataFrame({"session": dates, "close_at": [datetime.combine(d, time(21), tzinfo=timezone.utc) for d in dates]},
                            schema={"session": pl.Date, "close_at": pl.Datetime("us", "UTC")})
    rows = [{"asset": a, **row} for a, row in securities.items()]
    meta = {"source": "synthetic corporate-action fixture", "retrieved_at": "2024-02-01T00:00:00Z",
            "currency": "USD", "asset_currencies": {a: "USD" for a in securities},
            "calendar": "supplied_test_sessions", "calendar_version": "1", "timezone": "America/New_York",
            "price_basis": "raw", "frequency": "1d", "coverage_start": dates[0].isoformat(),
            "coverage_end": dates[-1].isoformat(), "actions_complete": True, "dividend_basis": "post_split_share"}
    return rt.prepare_market_data(
        prices=prices, sessions=sessions, metadata=meta,
        splits=pl.DataFrame(list(splits), schema={"action_id": pl.String, "asset": pl.String,
                            "effective_session": pl.Date, "ratio": pl.Float64}, orient="row"),
        dividends=pl.DataFrame(list(dividends), schema={"action_id": pl.String, "asset": pl.String,
                               "ex_session": pl.Date, "pay_date": pl.Date, "cash_per_share": pl.Float64}, orient="row"),
        **rt.corporate_action_inputs(securities=rows, **tables))


def action(action_id, kind, source, effective, *, basis=None, record=None, label=None):
    return dict(action_id=action_id, action_type=kind, source_asset=source, announced_at=ANNOUNCED,
                effective_date=effective, record_date=record,
                entitlement_basis=basis or ("effective_date" if kind != "distribution" else "ex_date"),
                provider_label=label, classification_source="synthetic test terms")


def cash_leg(action_id, amount, due, leg_id="cash", **accrual):
    return dict(action_id=action_id, leg_id=leg_id, leg_type="cash", cash_per_source_share=amount, due_date=due, **accrual)


def stock_leg(action_id, asset, units, due, leg_id=None, fraction="fractional", price=None):
    return dict(action_id=action_id, leg_id=leg_id or f"stock_{asset}", leg_type="security", asset=asset,
                units_per_source_share=units, due_date=due, fraction_policy=fraction, cash_in_lieu_price=price)


POLICY = rt.CorporateActionPolicy(distributed_securities="retain", short_obligations="cash_at_delivery_close",
                                  pending_claim_margin=1., obligation_margin=1., warrant_margin=1., reorganization_fee=0.)
HOLD = rt.BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity", fractional_shares=True,
                        initial_gross_leverage=1., terminal_action="mark_only")
SIGNED = dict(long_short=rt.LongShortPolicy(collateral_multiple=1., long_margin=.5, short_margin=.5,
                                            rebate_rate=0., rebate_day_count="ACT/360"),
              financing=rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F",
                                     maintenance_equity_ratio=None, on_breach="stop", cash_sweep="repay_debt"))


def borrow(rates):
    return rt.StockBorrow(rates=rates, day_count="ACT/365F", metadata={"source": "test assumption", "basis": "modeled"})


def hold(market, end=D[-1], **kwargs):
    defaults = dict(initial_capital=1000., entry_session=D[0], end_session=end, policy=HOLD, costs=ZERO,
                    cash_rate=0., cash_day_count="ACT/365F", corporate_actions=POLICY)
    return rt.buy_and_hold(market, **(defaults | kwargs))


def equity_on(result, day):
    return result.daily.filter(pl.col("session") == day)["equity"].item()


def cash_merger_market(cash=105., settle=D[5]):
    return make_market({"T": [100, 100, 100, None, None, None, None], "A": [50]*7},
                       {"T": equity(last_session=D[2], terminated_date=D[3]), "A": equity()},
                       actions=[action("M1", "cash_acquisition", "T", D[3])], legs=[cash_leg("M1", cash, settle)])


def test_pure_ticker_change_preserves_wealth_and_creates_no_trade():
    secs = {"S": equity()}
    aliases = [dict(asset="S", ticker="OLD", exchange="XNYS", valid_from=D[0], valid_to=D[2], source="test"),
               dict(asset="S", ticker="NEW", exchange="XNAS", valid_from=D[3], valid_to=None, source="test")]
    market = make_market({"S": [10]*7}, secs, aliases=aliases)
    plain = make_market({"S": [10]*7}, secs)
    result = hold(market, weights={"S": 1.}, corporate_actions=None)
    assert result.trades.height == 1 and result.daily["pnl"].to_list() == [0.]*6
    assert result.daily.equals(hold(plain, weights={"S": 1.}, corporate_actions=None).daily)
    status = rt.security_status(market)
    assert status.filter(pl.col("session") == D[2])["ticker"].item() == "OLD"
    assert status.filter(pl.col("session") == D[3]).select("ticker", "exchange").row(0) == ("NEW", "XNAS")


def test_alias_cannot_join_two_securities():
    aliases = [dict(asset="S", ticker="X", exchange="XNYS", valid_from=D[0], valid_to=None, source="t"),
               dict(asset="R", ticker="X", exchange="XNYS", valid_from=D[3], valid_to=None, source="t")]
    with pytest.raises(ValueError, match="overlapping alias"):
        make_market({"S": [1]*7, "R": [1]*7}, {"S": equity(), "R": equity()}, aliases=aliases)


def test_long_cash_acquisition_with_delayed_settlement():
    market = cash_merger_market()
    result = hold(market, weights={"T": .5, "A": .5}).require_complete()
    # 5 T at 100 become a 525 receivable on the effective date; cash arrives later.
    assert equity_on(result, D[3]) == pytest.approx(1025.)
    day = result.daily.filter(pl.col("session") == D[4])
    assert day["pending_cash_receivable"].item() == pytest.approx(525.) and day["cash"].item() == 0.
    settled = result.daily.filter(pl.col("session") == D[5])
    assert settled["cash"].item() == pytest.approx(525.) and settled["pending_cash_receivable"].item() == 0.
    assert result.positions.filter((pl.col("asset") == "T") & (pl.col("session") >= D[3]))["quantity"].to_list() == [0.]*4
    assert result.trades.height == 2  # Entry only; the conversion is not a sale.
    stages = result.corporate_actions["stage"].to_list()
    assert stages == ["entitlement", "extinguishment", "cash_claim", "conversion_pnl", "cash_settlement"]
    attribution = result.attribution.filter(pl.col("component") == "corporate_action")["pnl"].sum()
    assert attribution == pytest.approx(25.)


def test_value_conserving_cash_merger_creates_no_equity():
    result = hold(cash_merger_market(cash=100.), weights={"T": .5, "A": .5}).require_complete()
    assert result.daily["equity"].to_list() == pytest.approx([1000.]*6)


def test_date_dependent_consideration_is_resolved_and_audited():
    market = make_market({"T": [100, 100, 100, None, None, None, None]},
                         {"T": equity(last_session=D[2], terminated_date=D[3])},
                         actions=[action("M1", "cash_acquisition", "T", D[3])],
                         legs=[cash_leg("M1", 100., D[5], accrual_per_day=.5, accrual_start=date(2023, 12, 25),
                                        accrual_end=D[3])])
    result = hold(market, weights={"T": 1.}).require_complete()
    claim = result.corporate_actions.filter(pl.col("stage") == "cash_claim")
    assert claim["cash_amount"].item() == pytest.approx(10*(100+.5*11))


def test_short_cash_acquisition_stops_borrow_fees_and_keeps_collateral_until_settlement():
    market = cash_merger_market()
    result = rt.buy_and_hold(market, quantities={"T": -5., "A": 20.}, initial_capital=1000., entry_session=D[0],
                             end_session=D[-1], policy=HOLD, costs=ZERO, stock_borrow=borrow({"T": .365}),
                             corporate_actions=replace(POLICY, obligation_margin=.5), **SIGNED).require_complete()
    fees = result.stock_borrow_accruals
    assert fees["date"].max() == date(2024, 1, 4)  # No fee on or after the effective date.
    assert fees.filter(pl.col("date") == date(2024, 1, 4))["amount"].item() == pytest.approx(500*.365/365)
    after = result.daily.filter(pl.col("session") == D[4])
    # The short now owes 525 cash; collateral follows the obligation, not the vanished ticker.
    assert after["pending_cash_payable"].item() == pytest.approx(525.)
    assert after["restricted_collateral"].item() == pytest.approx(525.)
    settled = result.daily.filter(pl.col("session") == D[5])
    assert settled["pending_cash_payable"].item() == 0. and settled["restricted_collateral"].item() == 0.
    # Short loses the 25 deal premium; fees are separate.
    assert result.attribution.filter(pl.col("component") == "corporate_action")["pnl"].sum() == pytest.approx(-25.)
    assert result.daily["equity"][-1] == pytest.approx(1000.-25.-fees["amount"].sum())


def test_pending_dividend_survives_merger():
    market = make_market({"T": [100, 100, 100, None, None, None, None]},
                         {"T": equity(last_session=D[2], terminated_date=D[3])},
                         dividends=[("DV", "T", D[2], D[5], 2.)],
                         actions=[action("M1", "cash_acquisition", "T", D[3])], legs=[cash_leg("M1", 100., D[4])])
    result = hold(market, weights={"T": 1.}).require_complete()
    assert result.daily.filter(pl.col("session") == D[4])["dividend_receivable"].item() == pytest.approx(20.)
    assert result.daily["cash"][-1] == pytest.approx(1020.)


def stock_merger_market(units=1.5, cash=None, fraction="fractional", price=None, delivery=D[4]):
    legs = [stock_leg("M2", "B", units, delivery, fraction=fraction, price=price)]
    if cash is not None:
        legs.append(cash_leg("M2", cash, D[5]))
    return make_market({"T": [60, 60, 60, None, None, None, None], "B": [40]*7},
                       {"T": equity(last_session=D[2], terminated_date=D[3]), "B": equity()},
                       actions=[action("M2", "stock_acquisition", "T", D[3])], legs=legs)


def test_stock_acquisition_with_delayed_delivery_is_value_neutral():
    result = hold(stock_merger_market(), weights={"T": 1.}).require_complete()
    assert result.daily["equity"].to_list() == pytest.approx([1000.]*6)
    pending = result.daily.filter(pl.col("session") == D[3])
    assert pending["pending_security_receivable"].item() == pytest.approx(1000.)
    assert result.positions.filter((pl.col("asset") == "B") & (pl.col("session") == D[4]))["quantity"].item() == pytest.approx(25.)
    assert result.trades.height == 1
    assert result.attribution.filter(pl.col("component") == "corporate_action")["pnl"].sum() == pytest.approx(0.)


def test_mixed_consideration_with_cash_in_lieu():
    market = make_market({"T": [60, 60, 60, None, None, None, None], "B": [40]*7},
                         {"T": equity(last_session=D[2], terminated_date=D[3]), "B": equity()},
                         actions=[action("M2", "stock_acquisition", "T", D[3])],
                         legs=[stock_leg("M2", "B", .7, D[4], fraction="cash_in_lieu", price=40.),
                               cash_leg("M2", 30., D[5])])
    # 1000/60 = 16.666... T shares -> 11.666... B, delivered 11 plus 0.666... in cash.
    result = hold(market, weights={"T": 1.}).require_complete()
    b = result.positions.filter((pl.col("asset") == "B") & (pl.col("session") == D[4]))["quantity"].item()
    assert b == 11.
    lieu = result.corporate_actions.filter(pl.col("stage") == "cash_in_lieu_claim")
    assert lieu["cash_amount"].item() == pytest.approx((1000/60*.7-11)*40)
    assert result.daily["equity"][-1] == pytest.approx(1000/60*(.7*40+30))


def test_spin_off_retains_parent_and_delays_child_delivery():
    market = make_market({"P": [100, 100, 100, 80, 80, 80, 80], "C": [None, None, None, 40, 40, 40, 40]},
                         {"P": equity(), "C": equity(first_session=D[3])},
                         actions=[action("SP", "distribution", "P", D[3], record=D[3], label="split")],
                         legs=[stock_leg("SP", "C", .5, D[5])])
    result = hold(market, weights={"P": 1.}).require_complete()
    assert result.daily["equity"].to_list() == pytest.approx([1000.]*6)
    ex = result.daily.filter(pl.col("session") == D[3])
    assert ex["pending_security_receivable"].item() == pytest.approx(200.)
    assert result.positions.filter((pl.col("asset") == "P") & (pl.col("session") == D[-1]))["quantity"].item() == 10.
    assert result.positions.filter((pl.col("asset") == "C") & (pl.col("session") == D[4]))["quantity"].item() == 0.
    assert result.positions.filter((pl.col("asset") == "C") & (pl.col("session") == D[5]))["quantity"].item() == 5.
    # Child was unquoted before the ex-date: its value is recognized at the first mark.
    stages = result.corporate_actions["stage"].to_list()
    assert "recognition_at_first_mark" in stages
    components = dict(result.attribution.group_by("component").agg(pl.col("pnl").sum()).iter_rows())
    assert components["price_pnl"] == pytest.approx(-200.) and components["corporate_action"] == pytest.approx(200.)


def test_immediate_spin_off_delivery_and_entry_on_ex_date_has_no_entitlement():
    market = make_market({"P": [100, 100, 100, 80, 80, 80, 80], "C": [None, None, None, 40, 40, 40, 40]},
                         {"P": equity(), "C": equity(first_session=D[3])},
                         actions=[action("SP", "distribution", "P", D[3], record=D[3])], legs=[stock_leg("SP", "C", .5, D[3])])
    result = hold(market, weights={"P": 1.}).require_complete()
    assert result.positions.filter((pl.col("asset") == "C") & (pl.col("session") == D[3]))["quantity"].item() == 5.
    late = hold(market, weights={"P": 1.}, entry_session=D[3]).require_complete()
    assert late.corporate_actions.height == 0 and late.daily["equity"].to_list() == pytest.approx([1000.]*3)


def test_trading_around_the_ex_date_boundary():
    market = make_market({"P": [100, 100, 100, 80, 80, 80, 80], "C": [None, None, None, 40, 40, 40, 40], "Q": [10]*7},
                         {"P": equity(), "C": equity(first_session=D[3]), "Q": equity()},
                         actions=[action("SP", "distribution", "P", D[3], record=D[3])], legs=[stock_leg("SP", "C", .5, D[4])])
    policy = rt.RebalancePolicy(execution="scheduled_close", sizing="post_cost_equity", fractional_shares=True,
                                terminal_action="mark_only", non_session="raise", receivable_policy="reserve",
                                max_asset_weight=1.)
    financing = rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F", maintenance_equity_ratio=None,
                             on_breach="stop", cash_sweep="repay_debt")

    def run(sell_day):
        rows = [(D[0], D[1], "P", 1., 1.), (D[0], D[1], "Q", 0., 1.), (D[0], D[1], "C", 0., 1.),
                (D[1], sell_day, "P", 0., 1.), (D[1], sell_day, "Q", 1., 1.), (D[1], sell_day, "C", 0., 1.)]
        targets = pl.DataFrame(rows, schema={"decision_session": pl.Date, "session": pl.Date, "asset": pl.String,
                                             "weight": pl.Float64, "gross_leverage": pl.Float64}, orient="row")
        return rt.scheduled_rebalance(market, targets=targets, initial_capital=1000., entry_session=D[1],
                                      end_session=D[-1], policy=policy, costs=ZERO, financing=financing,
                                      corporate_actions=POLICY).require_complete()
    sold_before = run(D[2])  # Sold at the last cum-entitlement close: no child.
    assert sold_before.corporate_actions.filter(pl.col("stage") == "entitlement")["entitled_quantity"].item() == 0.
    sold_on = run(D[3])  # Held into the ex-date: entitled, sells the parent at the ex close.
    assert sold_on.corporate_actions.filter(pl.col("stage") == "entitlement")["entitled_quantity"].item() == 10.
    assert sold_on.positions.filter((pl.col("asset") == "C") & (pl.col("session") == D[-1]))["quantity"].item() == 5.


# --------------------------------------------------------------------- warrants

def warrant_market(*, mark_on_ex=True, settlement="physical", w_first=D[3]):
    w = dict(security_type="warrant", first_session=w_first, last_session=D[5], terminated_date=D[6],
             unquoted_valuation="supplied_mark", source="synthetic warrant")
    marks = [dict(session=D[2], asset="W", mark=8., method="model:assumed_fair_value", source="test")] if mark_on_ex else []
    return make_market({"P": [100, 100, 96, 96, 96, 96, 96], "W": [None, None, None, 8, 8, 8, None]},
                       {"P": equity(), "W": w},
                       actions=[action("WD", "distribution", "P", D[2], record=D[2])],
                       legs=[stock_leg("WD", "W", .5, D[3])],
                       warrants=[dict(asset="W", underlying_asset="P", units_per_warrant=1., exercise_price=90.,
                                      exercisable_from=D[3], expiry_date=D[5], exercise_style="american",
                                      settlement=settlement, source="synthetic terms")],
                       valuation_marks=marks)


def test_unvalued_warrant_entitlement_stops_without_inventing_nav():
    market = warrant_market(mark_on_ex=False)
    result = hold(market, weights={"P": 1.})
    assert result.status == "stopped" and result.stop_reason == "unvalued_position"
    assert result.stop_session == D[1] and result.daily["session"].to_list() == [D[1]]
    assert result.metadata["lifecycle"]["unvalued"] == {"session": "2024-01-04", "assets": ["W"],
                                                        "last_valued_session": "2024-01-03"}
    with pytest.raises(ValueError, match="unvalued_position"):
        result.require_complete()
    with pytest.raises(ValueError, match="stopped"):
        rt.performance(result, periods_per_year=252, risk_free_annual_effective=0.,
                       minimum_acceptable_return_annual_effective=0.)


def test_supplied_mark_delivery_and_worthless_expiry():
    result = hold(warrant_market(), weights={"P": 1.}).require_complete()
    assert result.daily["equity"].to_list() == pytest.approx([1000., 1000., 1000., 1000., 1000., 960.])
    ex = result.security_claims.filter(pl.col("session") == D[2])
    assert ex.select("asset", "quantity", "mark", "value").row(0) == ("W", 5., 8., 40.)
    assert result.corporate_actions.filter(pl.col("stage") == "warrant_expiry")["pnl"].item() == pytest.approx(-40.)
    assert result.positions.filter((pl.col("asset") == "W") & (pl.col("session") == D[6]))["quantity"].item() == 0.


def test_explicit_cash_exercise_borrows_and_delivers_underlying():
    exercises = pl.DataFrame([("X1", D[3], D[4], "W", 5., D[5], 1.)], schema=rt._lifecycle.EXERCISE_SCHEMA, orient="row")
    financing = rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F", maintenance_equity_ratio=.25,
                             on_breach="stop", cash_sweep="repay_debt")
    result = hold(warrant_market(), weights={"P": 1.}, financing=financing, cash_rate=None, cash_day_count=None,
                  warrant_exercises=exercises).require_complete()
    row = result.corporate_actions.filter(pl.col("stage") == "warrant_exercise")
    # Five warrants worth 8 plus 450 strike become 480 of stock: the exercise forfeits 10 of time value.
    assert row["pnl"].item() == pytest.approx(-10.) and row["fee"].item() == 1.
    day = result.daily.filter(pl.col("session") == D[4])
    assert day["debt"].item() == pytest.approx(451.) and day["pending_security_receivable"].item() == pytest.approx(480.)
    assert result.positions.filter((pl.col("asset") == "P") & (pl.col("session") == D[5]))["quantity"].item() == 15.
    assert result.daily["equity"][-1] == pytest.approx(1000.-10.-1.)


def test_exercise_without_cash_or_financing_is_rejected():
    exercises = pl.DataFrame([("X1", D[3], D[4], "W", 5., D[5], 0.)], schema=rt._lifecycle.EXERCISE_SCHEMA, orient="row")
    with pytest.raises(ValueError, match="needs borrowing"):
        hold(warrant_market(), weights={"P": 1.}, warrant_exercises=exercises)


def test_exercise_before_delivery_is_rejected():
    exercises = pl.DataFrame([("X1", D[1], D[2], "W", 5., D[3], 0.)], schema=rt._lifecycle.EXERCISE_SCHEMA, orient="row")
    with pytest.raises(ValueError, match="exercise window"):
        hold(warrant_market(), weights={"P": 1.}, warrant_exercises=exercises)
    market = make_market({"P": [100, 100, 96, 96, 96, 96, 96], "W": [None, 9, 8, 8, 8, 8, None]},
                         {"P": equity(), "W": dict(security_type="warrant", first_session=D[1], last_session=D[5],
                          terminated_date=D[6], unquoted_valuation="none", source="t")},
                         actions=[action("WD", "distribution", "P", D[2], record=D[2])],
                         legs=[stock_leg("WD", "W", .5, D[4])],
                         warrants=[dict(asset="W", underlying_asset="P", units_per_warrant=1., exercise_price=90.,
                                        exercisable_from=D[1], expiry_date=D[5], exercise_style="american",
                                        settlement="physical", source="t")])
    early = pl.DataFrame([("X1", D[2], D[3], "W", 5., D[4], 0.)], schema=rt._lifecycle.EXERCISE_SCHEMA, orient="row")
    with pytest.raises(ValueError, match="exceeds"):
        hold(market, weights={"P": 1.}, warrant_exercises=early)


def test_selected_policy_sells_delivered_warrants_with_ordinary_costs():
    costs = rt.TradeCosts(commission_bps=10., half_spread_bps=0., impact_bps=0.)
    result = hold(warrant_market(), weights={"P": 1.}, costs=costs,
                  corporate_actions=replace(POLICY, distributed_securities="liquidate_at_first_permitted_close"))
    sale = result.trades.filter(pl.col("execution") == "corporate_action_liquidation_close")
    warrants = 1000/1.001/100*.5  # Entry costs reduce the parent quantity and hence the entitlement.
    assert sale.select("session", "asset").row(0) == (D[3], "W")
    assert sale["signed_quantity"].item() == pytest.approx(-warrants)
    assert sale["trade_cost"].item() == pytest.approx(warrants*8*.001)
    assert result.positions.filter((pl.col("asset") == "W") & (pl.col("session") == D[5]))["quantity"].item() == 0.
    assert result.corporate_actions.filter(pl.col("stage") == "warrant_expiry").height == 0


def test_unsupported_warrant_terms_are_rejected():
    with pytest.raises(ValueError, match="only physical"):
        warrant_market(settlement="net_share")


def test_retained_distribution_waits_for_explicit_policy():
    with pytest.raises(ValueError, match="CorporateActionPolicy"):
        hold(warrant_market(), weights={"P": 1.}, corporate_actions=None)


# ------------------------------------------------------------------ shorts

def spin_market(delivery=D[4]):
    return make_market({"P": [100, 100, 100, 80, 80, 80, 80], "C": [None, None, None, 40, 40, 40, 40], "Q": [10]*7},
                       {"P": equity(), "C": equity(first_session=D[3]), "Q": equity()},
                       actions=[action("SP", "distribution", "P", D[3], record=D[3])], legs=[stock_leg("SP", "C", .5, delivery)])


def test_short_spin_off_obligation_is_collateralized_then_cash_settled():
    result = rt.buy_and_hold(spin_market(), quantities={"P": -5., "Q": 100.}, initial_capital=1000., entry_session=D[0],
                             end_session=D[-1], policy=HOLD, costs=ZERO, stock_borrow=borrow({"P": 0.}),
                             corporate_actions=POLICY, **SIGNED).require_complete()
    ex = result.daily.filter(pl.col("session") == D[3])
    # Short 5 P at 80 (400) plus an obligation to deliver 2.5 C (100), all collateralized.
    assert ex["pending_security_obligation"].item() == pytest.approx(100.)
    assert ex["restricted_collateral"].item() == pytest.approx(500.)
    settled = result.daily.filter(pl.col("session") == D[4])
    assert settled["pending_security_obligation"].item() == 0. and settled["restricted_collateral"].item() == pytest.approx(400.)
    assert result.corporate_actions.filter(pl.col("stage") == "obligation_cash_settlement")["cash_amount"].item() == pytest.approx(-100.)
    assert result.daily["equity"].to_list() == pytest.approx([1000.]*6)
    assert result.positions.filter(pl.col("asset") == "C")["quantity"].to_list() == [0.]*7


def test_short_stock_merger_becomes_borrowed_successor_short():
    market = stock_merger_market(units=1.5, delivery=D[3])
    result = rt.buy_and_hold(market, quantities={"T": -10., "B": 0.}, initial_capital=1000., entry_session=D[0],
                             end_session=D[-1], policy=HOLD, costs=ZERO,
                             stock_borrow=borrow({"T": .365, "B": .73}),
                             corporate_actions=replace(POLICY, short_obligations="borrowed_short_position"),
                             **SIGNED).require_complete()
    assert result.positions.filter((pl.col("asset") == "B") & (pl.col("session") == D[3]))["quantity"].item() == -15.
    fees = result.stock_borrow_accruals
    assert fees.filter(pl.col("asset") == "T")["date"].max() == date(2024, 1, 4)
    # B is borrowed from the conversion date onward: 600 short at 0.2% per day.
    assert fees.filter((pl.col("asset") == "B") & (pl.col("date") == date(2024, 1, 6)))["amount"].item() == pytest.approx(1.2)
    assert result.trades.height == 1


def test_borrowed_short_successor_needs_explicit_borrow_terms():
    with pytest.raises(ValueError, match="borrow rates"):
        rt.buy_and_hold(stock_merger_market(delivery=D[3]), quantities={"T": -10., "B": 0.}, initial_capital=1000.,
                        entry_session=D[0], end_session=D[-1], policy=HOLD, costs=ZERO, stock_borrow=borrow({"T": 0.}),
                        corporate_actions=replace(POLICY, short_obligations="borrowed_short_position"), **SIGNED)


# --------------------------------------------------------- strategy integration

def schedule(rows):
    return pl.DataFrame(rows, schema={"decision_session": pl.Date, "session": pl.Date, "asset": pl.String,
                                      "weight": pl.Float64, "gross_leverage": pl.Float64}, orient="row")


REBALANCE = rt.RebalancePolicy(execution="scheduled_close", sizing="post_cost_equity", fractional_shares=True,
                               terminal_action="mark_only", non_session="raise", receivable_policy="reserve",
                               max_asset_weight=1.)
NO_LOAN = rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F", maintenance_equity_ratio=.5,
                       on_breach="stop", cash_sweep="repay_debt")


def test_merger_collides_with_scheduled_rebalance():
    market = cash_merger_market()
    targets = schedule([(D[0], D[1], "T", .5, 1.), (D[0], D[1], "A", .5, 1.),
                        (D[2], D[3], "T", 0., 1.), (D[2], D[3], "A", 1., 1.)])
    result = rt.scheduled_rebalance(market, targets=targets, initial_capital=1000., entry_session=D[1],
                                    end_session=D[-1], policy=REBALANCE, costs=ZERO, financing=NO_LOAN,
                                    corporate_actions=POLICY).require_complete()
    # The 525 claim is unspendable on D[3]: the reserve policy invests only the remaining 500.
    assert result.trades.filter(pl.col("session") == D[3]).height == 0
    assert result.positions.filter((pl.col("asset") == "A") & (pl.col("session") == D[3]))["market_value"].item() == pytest.approx(500.)
    assert result.daily["equity"][-1] == pytest.approx(1025.)


def test_stale_target_for_terminated_security_is_rejected():
    targets = schedule([(D[0], D[1], "T", .5, 1.), (D[0], D[1], "A", .5, 1.),
                        (D[3], D[4], "T", .5, 1.), (D[3], D[4], "A", .5, 1.)])
    with pytest.raises(ValueError, match="stale target: T is terminated"):
        rt.scheduled_rebalance(cash_merger_market(), targets=targets, initial_capital=1000., entry_session=D[1],
                               end_session=D[-1], policy=REBALANCE, costs=ZERO, financing=NO_LOAN, corporate_actions=POLICY)


def test_flat_holding_at_event_is_audited_without_effect():
    result = hold(cash_merger_market(), weights={"T": 0., "A": 1.}).require_complete()
    entitlement = result.corporate_actions.filter(pl.col("stage") == "entitlement")
    assert entitlement.select("entitled_quantity", "valuation").row(0) == (0., "no_position")
    assert result.daily["equity"].to_list() == pytest.approx([1000.]*6)


def test_dividend_reinvestment_never_buys_a_terminated_payer():
    market = make_market({"T": [100, 100, 100, None, None, None, None], "A": [50]*7},
                         {"T": equity(last_session=D[2], terminated_date=D[3]), "A": equity()},
                         dividends=[("DV", "T", D[2], D[5], 2.)],
                         actions=[action("M1", "cash_acquisition", "T", D[3])], legs=[cash_leg("M1", 100., D[4])])
    drip = rt.DividendReinvestment(execution="first_close_on_or_after_payment", funding="after_debt_repayment",
                                   scheduled_collision="rebalance_only", terminal_action="hold_cash")
    result = hold(market, weights={"T": .5, "A": .5}, dividend_reinvestment=drip).require_complete()
    assert result.dividend_reinvestments["status"].to_list() == ["payer_not_quoted"]
    assert result.trades.height == 2
    report = rt.performance(result, periods_per_year=252, risk_free_annual_effective=0.,
                            minimum_acceptable_return_annual_effective=0.)
    assert report.summary.filter(pl.col("metric") == "ending_equity")["value"].item() == pytest.approx(1010.)


def test_leveraged_long_merger_with_financing_and_reporting():
    market = cash_merger_market()
    financing = rt.Financing(cash_rate=0., borrowing_rate=.0365, day_count="ACT/365F",
                             maintenance_equity_ratio=.25, on_breach="stop", cash_sweep="repay_debt")
    result = hold(market, weights={"T": .5, "A": .5}, financing=financing, cash_rate=None, cash_day_count=None,
                  policy=replace(HOLD, initial_gross_leverage=2.)).require_complete()
    # The 1050 settlement repays the loan only when the cash actually arrives.
    assert result.daily.filter(pl.col("session") == D[4])["debt"].item() > 1000.
    assert result.daily.filter(pl.col("session") == D[5])["debt"].item() == 0.
    report = rt.performance(result, periods_per_year=252, risk_free_annual_effective=0.,
                            minimum_acceptable_return_annual_effective=0.)
    claims = report.allocation.filter(pl.col("component") == "account:pending_cash_receivable")
    assert claims.filter(pl.col("session") == D[4])["weight"].item() > 0
    totals = report.allocation.filter(pl.col("session") == D[4])["weight"].sum()
    assert totals == pytest.approx(1.)


# --------------------------------------------------------- validation and data

def test_missing_and_unexpected_quotes_are_distinguished():
    secs = {"S": equity()}
    with pytest.raises(ValueError, match="missing data"):
        make_market({"S": [1, 1, None, 1, 1, 1, 1]}, secs)
    with pytest.raises(ValueError, match="suspended"):
        make_market({"S": [1]*7}, secs, suspensions=[dict(asset="S", first_session=D[2], last_session=D[3],
                                                         reason="halt", source="t")])
    with pytest.raises(ValueError, match="not_listed"):
        make_market({"S": [1]*7}, {"S": equity(first_session=D[1])})


def test_suspended_holding_stops_unless_explicitly_marked():
    halt = [dict(asset="S", first_session=D[2], last_session=D[3], reason="regulatory halt", source="t")]
    series = {"S": [10, 10, None, None, 10, 10, 10]}
    result = hold(make_market(series, {"S": equity()}, suspensions=halt), weights={"S": 1.}, corporate_actions=None)
    assert result.status == "stopped" and result.stop_session == D[1]
    marked = make_market(series, {"S": equity(unquoted_valuation="supplied_mark")}, suspensions=halt,
                         valuation_marks=[dict(session=d, asset="S", mark=9., method="broker_mark", source="t")
                                          for d in D[2:4]])
    ok = hold(marked, weights={"S": 1.}, corporate_actions=None).require_complete()
    assert ok.daily["equity"].to_list() == pytest.approx([1000., 900., 900., 1000., 1000., 1000.])


def test_action_validation_rejects_unsafe_inputs():
    base = dict(series={"T": [100, 100, 100, None, None, None, None]},
                securities={"T": equity(last_session=D[2], terminated_date=D[3])})
    with pytest.raises(ValueError, match="record-date entitlement"):
        make_market(**base, actions=[action("M1", "cash_acquisition", "T", D[3], basis="record_date")],
                    legs=[cash_leg("M1", 1., D[3])])
    with pytest.raises(ValueError, match="unsupported action_type"):
        make_market(**base, actions=[action("M1", "tender_offer", "T", D[3])], legs=[cash_leg("M1", 1., D[3])])
    with pytest.raises(ValueError, match="duplicate keys"):
        make_market(**base, actions=[action("M1", "cash_acquisition", "T", D[3])]*2, legs=[cash_leg("M1", 1., D[3])])
    with pytest.raises(ValueError, match="terminates without"):
        make_market(**base)
    with pytest.raises(ValueError, match="no consideration"):
        make_market(**base, actions=[action("M1", "cash_acquisition", "T", D[3])])
    with pytest.raises(ValueError, match="unique across"):
        make_market(**base, dividends=[("M1", "T", D[1], D[2], 1.)],
                    actions=[action("M1", "cash_acquisition", "T", D[3])], legs=[cash_leg("M1", 1., D[3])])
    spin = dict(series={"P": [1]*7, "C": [1]*7}, securities={"P": equity(), "C": equity()})
    with pytest.raises(ValueError, match="conflicting actions"):
        make_market(**spin, actions=[action("S1", "distribution", "P", D[3], record=D[3]),
                                     action("S2", "distribution", "P", D[3], record=D[3])],
                    legs=[stock_leg("S1", "C", 1., D[3]), stock_leg("S2", "C", 1., D[3])])
    with pytest.raises(ValueError, match="due bill"):
        make_market(**spin, actions=[action("S1", "distribution", "P", D[3], basis="due_bill", record=D[3])],
                    legs=[stock_leg("S1", "C", 1., D[4])])
    late = dict(action("S1", "distribution", "P", D[3], record=D[3]), announced_at=datetime(2024, 1, 8, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="announcement"):
        make_market(**spin, actions=[late], legs=[stock_leg("S1", "C", 1., D[3])])


def test_due_bill_distribution_entitles_holders_at_the_ex_date():
    market = make_market({"P": [100, 100, 100, 80, 80, 80, 80], "C": [None, None, None, 40, 40, 40, 40]},
                         {"P": equity(), "C": equity(first_session=D[3])},
                         actions=[action("SP", "distribution", "P", D[3], basis="due_bill", record=D[1])],
                         legs=[stock_leg("SP", "C", .5, D[4])])
    result = hold(market, weights={"P": 1.}).require_complete()
    assert result.corporate_actions.filter(pl.col("stage") == "entitlement")["entitled_quantity"].item() == 10.
    assert result.daily["equity"].to_list() == pytest.approx([1000.]*6)


def test_adjusted_prices_cannot_carry_ledger_actions():
    with pytest.raises(ValueError, match="require raw prices"):
        market = cash_merger_market()
        rt.prepare_market_data(prices=market.prices, sessions=market.sessions, splits=market.splits,
                               dividends=market.dividends, metadata={**market.metadata, "price_basis": "total_return_adjusted"},
                               **{k: getattr(market, k) for k in rt._lifecycle.LIFECYCLE_TABLES})


def test_security_returns_separate_quoted_and_economic_returns():
    values = rt.security_returns(spin_market()).values
    parent = values.filter((pl.col("asset") == "P") & (pl.col("session") == D[3]))
    assert parent.select("price_return", "total_return", "status").row(0) == (pytest.approx(-.2), pytest.approx(0.), "ok")
    merger = rt.security_returns(cash_merger_market()).values.filter(pl.col("asset") == "T")
    assert merger.filter(pl.col("status") == "acquisition_consideration").select(
        "session", "period_start", "total_return").row(0) == (D[3], D[2], pytest.approx(.05))
    unvalued = rt.security_returns(warrant_market(mark_on_ex=False)).values
    assert unvalued.filter((pl.col("asset") == "P") & (pl.col("session") == D[2]))["status"].item() == "unvalued_distribution"
    with pytest.raises(ValueError, match="security_returns"):
        rt.returns(spin_market(), method="simple", basis="price")


def test_lifecycle_snapshot_round_trip_and_legacy_snapshot(tmp_path):
    market = warrant_market()
    loaded = rt.load_snapshot(rt.save_snapshot(market, tmp_path / "v2"))
    assert loaded.snapshot_id == market.snapshot_id and loaded.warrants.equals(market.warrants)
    from pathlib import Path
    legacy = rt.load_snapshot(Path(__file__).resolve().parents[1] / "examples/snapshots/v1/equities")
    assert legacy.securities is None
    import json
    assert json.loads((rt.save_snapshot(legacy, tmp_path / "v1") / "manifest.json").read_text())["format_version"] == 1


def test_status_table_uses_only_announced_information():
    status = rt.security_status(cash_merger_market())
    assert status.filter((pl.col("asset") == "T") & (pl.col("session") == D[0]))["announced_actions"].item() == "M1"
    assert status.filter((pl.col("asset") == "T") & (pl.col("session") == D[3]))["status"].item() == "terminated"
    late = make_market({"T": [100, 100, 100, None, None, None, None]},
                       {"T": equity(last_session=D[2], terminated_date=D[3])},
                       actions=[dict(action("M1", "cash_acquisition", "T", D[3]),
                                     announced_at=datetime(2024, 1, 3, 20, tzinfo=timezone.utc))],
                       legs=[cash_leg("M1", 1., D[3])])
    known = rt.security_status(late).filter(pl.col("asset") == "T")["announced_actions"].to_list()
    assert known[:3] == [None, "M1", "M1"]


def test_signal_workflow_shares_the_action_ledger():
    market = cash_merger_market()
    closes = dict(market.sessions.iter_rows())
    rows = [(d, a, w, 1., closes[d], closes[d]) for d, basket in ((D[0], {"T": .5, "A": .5}), (D[3], {"T": 0., "A": 1.}))
            for a, w in basket.items()]
    signals = pl.DataFrame(rows, schema={"decision_session": pl.Date, "asset": pl.String, "weight": pl.Float64,
                                         "gross_leverage": pl.Float64, "observed_at": pl.Datetime("us", "UTC"),
                                         "available_at": pl.Datetime("us", "UTC")}, orient="row")
    instructions = rt.signal_targets(signals, sessions=market.sessions, execution="next_session_close", rebalance="each_signal",
                                     metadata={**market.metadata, "signal_definition": "fixed synthetic instructions"})
    result = rt.scheduled_rebalance(market, targets=instructions, initial_capital=1000., entry_session=D[1],
                                    end_session=D[-1], policy=REBALANCE, costs=ZERO, financing=NO_LOAN,
                                    corporate_actions=POLICY).require_complete()
    assert result.signal_audit["status"].to_list() == ["processed"]*4
    # The deal cash is still a pending claim at the D[4] execution, so only free equity is invested.
    day = result.daily.filter(pl.col("session") == D[4])
    assert day["pending_cash_receivable"].item() == pytest.approx(525.)
    assert result.positions.filter((pl.col("asset") == "A") & (pl.col("session") == D[4]))["market_value"].item() == pytest.approx(500.)


def test_reorganization_fee_is_the_only_mandatory_action_cost():
    result = hold(cash_merger_market(cash=100.), weights={"T": .5, "A": .5}, policy=replace(HOLD, initial_gross_leverage=.9),
                  corporate_actions=replace(POLICY, reorganization_fee=2.5)).require_complete()
    assert result.costs["component"].to_list() == ["reorganization_fee"]
    assert result.daily["equity"][-1] == pytest.approx(997.5)


def test_square_root_costs_bind_only_quoted_strategy_assets():
    market = spin_market()
    liquidity = rt.LiquidityResult(pl.DataFrame({"asset": ["P", "Q"], "daily_volatility": [.02, .02],
        "dollar_adv": [1e6, 1e6], "n_obs": [2, 2]}), dict(source="synthetic estimates", currency="USD",
        volatility_unit="daily_decimal", adv_unit="currency_per_trading_day", sample_start="2023-12-28",
        sample_end="2023-12-29", decision_session=D[0].isoformat(), decision_at=datetime.combine(D[0], time(), timezone.utc).isoformat()))
    costs = rt.SquareRootImpactCosts(liquidity=liquidity, commission_bps=1., commission_per_share=0., half_spread_bps=1.,
                                     impact_coefficient=.1)
    targets = schedule([(D[0], D[1], "P", .5, 1.), (D[0], D[1], "Q", .5, 1.), (D[3], D[4], "P", .5, 1.), (D[3], D[4], "Q", .5, 1.)])
    result = rt.scheduled_rebalance(market, targets=targets, initial_capital=1000., entry_session=D[1], end_session=D[-1],
                                    policy=REBALANCE, costs=costs, financing=NO_LOAN, corporate_actions=POLICY).require_complete()
    # The retained child stays outside the basket: P and Q are rebalanced to equal values.
    after = result.positions.filter(pl.col("session") == D[4])
    values = dict(after.select("asset", "market_value").iter_rows())
    assert values["P"] == pytest.approx(values["Q"]) and values["C"] > 0
    assert set(result.execution_costs["asset"]) <= {"P", "Q"}


def test_signed_schedule_with_spin_off_dividend_and_loan():
    market = make_market({"P": [100, 100, 100, 80, 80, 80, 80], "C": [None, None, None, 40, 40, 40, 40], "Q": [10]*7},
                         {"P": equity(), "C": equity(first_session=D[3]), "Q": equity()},
                         dividends=[("DQ", "Q", D[3], D[5], .1)],
                         actions=[action("SP", "distribution", "P", D[3], record=D[3])], legs=[stock_leg("SP", "C", .5, D[4])])
    rows = [(D[0], D[1], "P", 1.5), (D[0], D[1], "Q", -.5), (D[4], D[5], "P", 1.5), (D[4], D[5], "Q", -.5)]
    targets = pl.DataFrame(rows, schema={"decision_session": pl.Date, "session": pl.Date, "asset": pl.String,
                                         "equity_exposure": pl.Float64}, orient="row")
    policy = replace(REBALANCE, receivable_policy="require_target")
    financing = rt.Financing(cash_rate=0., borrowing_rate=.0365, day_count="ACT/365F", maintenance_equity_ratio=None,
                             on_breach="stop", cash_sweep="repay_debt")
    result = rt.scheduled_rebalance(market, targets=targets, initial_capital=1000., entry_session=D[1], end_session=D[-1],
                                    policy=policy, costs=ZERO, financing=financing,
                                    long_short=replace(SIGNED["long_short"], long_margin=.3, short_margin=.3),
                                    stock_borrow=borrow({"Q": .0365}),
                                    corporate_actions=replace(POLICY, pending_claim_margin=.5)).require_complete()
    # 7.5 C shares (300) are delivered and retained outside the signed basket.
    c = result.positions.filter((pl.col("asset") == "C") & (pl.col("session") == D[5]))["quantity"].item()
    assert c == pytest.approx(15*.5)
    liability = result.dividend_liabilities.filter(pl.col("action_id") == "DQ")
    assert liability["entitled_quantity"][0] == pytest.approx(-50.)
    exposures = dict(result.positions.filter(pl.col("session") == D[5]).select("asset", "market_value").iter_rows())
    equity_after = result.rebalances.filter(pl.col("session") == D[5])["equity_after"].item()
    assert exposures["P"] == pytest.approx(1.5*(equity_after-exposures["C"]))
    assert result.costs.filter(pl.col("component") == "borrowing_interest").height > 0


def test_security_returns_flag_suspension_gaps():
    halt = [dict(asset="S", first_session=D[2], last_session=D[3], reason="halt", source="t")]
    values = rt.security_returns(make_market({"S": [10, 11, None, None, 12, 12, 12]}, {"S": equity()}, suspensions=halt)).values
    assert values.filter(pl.col("session") == D[4]).select("period_start", "total_return", "status").row(0) == (D[1], None, "quote_gap")


def test_partial_report_and_plots_after_unvalued_stop():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    halt = [dict(asset="S", first_session=D[4], last_session=D[5], reason="halt", source="t")]
    market = make_market({"S": [10, 11, 12, 13, None, None, 13], "A": [5]*7}, {"S": equity(), "A": equity()}, suspensions=halt)
    targets = schedule([(D[0], D[1], "S", .5, 1.), (D[0], D[1], "A", .5, 1.), (D[2], D[3], "S", .2, 1.), (D[2], D[3], "A", .8, 1.)])
    result = rt.scheduled_rebalance(market, targets=targets, initial_capital=1000., entry_session=D[1], end_session=D[-1],
                                    policy=REBALANCE, costs=ZERO, financing=NO_LOAN)
    assert (result.status, result.stop_reason, result.stop_session) == ("stopped", "unvalued_position", D[3])
    report = rt.performance(result, periods_per_year=252, risk_free_annual_effective=0.,
                            minimum_acceptable_return_annual_effective=0., allow_partial=True)
    assert report.daily["session"].to_list() == [D[2], D[3]]
    rt.plots.allocation(report)
    rt.plots.exposures(report)
    marked = hold(warrant_market(), weights={"P": 1.})
    rt.plots.exposures(rt.performance(marked, periods_per_year=252, risk_free_annual_effective=0.,
                                      minimum_acceptable_return_annual_effective=0.))


def test_securities_only_lifecycle_matches_legacy_numbers(inputs, run):
    legacy_inputs = inputs(series={"A": [100., 110., 99.], "B": [50., 49., 51.]},
                           dividends=[("d1", "A", date(2024, 1, 3), date(2024, 1, 4), 1.)])
    legacy = rt.prepare_market_data(**legacy_inputs)
    lifecycle = rt.prepare_market_data(**legacy_inputs, **rt.corporate_action_inputs(securities=[
        dict(asset=a, security_type="equity", first_session=date(2024, 1, 2), unquoted_valuation="none", source="t")
        for a in ("A", "B")]))
    a, b = run(legacy), run(lifecycle)
    for name in ("daily", "positions", "trades", "attribution", "receivables", "valuations"):
        assert getattr(a, name).equals(getattr(b, name)), name
    assert a.corporate_actions.height == 0 and a.security_claims.height == 0
    assert a.daily.select(rt._actions.CLAIM_COLUMNS).to_numpy().sum() == 0
    assert legacy.securities is None and legacy.snapshot_id != lifecycle.snapshot_id


def test_corporate_action_example_runs_offline():
    import runpy
    from pathlib import Path
    example = runpy.run_path(str(Path(__file__).resolve().parents[1]/"examples"/"corporate_actions.py"))
    market, result, report, tables = example["run_example"]()
    assert result.status == "complete" and result.trades.height == 3
    pnl = dict(tables["audit"].filter(pl.col("stage") == "conversion_pnl").select("action_id", "pnl").iter_rows())
    assert pnl == {"SPIN": 0., "CASHDEAL": pytest.approx(-80.), "STOCKDEAL": pytest.approx(-460.)}
    assert report.attribution["pnl"].sum() == pytest.approx(result.daily["equity"][-1]-100_000.)


def test_review_regressions_rounding_entry_exercise_and_pending_claim_splits():
    # 1000/3 shares * 0.03 is 9.999...98 units: deliver 10, no negative cash in lieu.
    market = make_market({"T": [3, 3, 3, None, None, None, None], "B": [3]*7},
                         {"T": equity(last_session=D[2], terminated_date=D[3]), "B": equity()},
                         actions=[action("M", "stock_acquisition", "T", D[3])],
                         legs=[stock_leg("M", "B", .03, D[4], fraction="cash_in_lieu", price=3.)])
    result = hold(market, weights={"T": 1.}).require_complete()
    assert result.corporate_actions.filter(pl.col("stage") == "cash_in_lieu_claim").height == 0
    exercises = pl.DataFrame([("X1", D[2], D[3], "W", 1., D[4], 0.)], schema=rt._lifecycle.EXERCISE_SCHEMA, orient="row")
    with pytest.raises(ValueError, match="execute in"):
        hold(warrant_market(), weights={"P": 1.}, entry_session=D[3], warrant_exercises=exercises)
    with pytest.raises(ValueError, match="between effectiveness and delivery"):
        make_market({"T": [60, 60, 60, None, None, None, None], "B": [40, 40, 40, 40, 20, 20, 20]},
                    {"T": equity(last_session=D[2], terminated_date=D[3]), "B": equity()},
                    splits=[("SB", "B", D[4], 2.)], actions=[action("M2", "stock_acquisition", "T", D[3])],
                    legs=[stock_leg("M2", "B", 1.5, D[5])])
    with pytest.raises(ValueError, match="underlying splits"):
        make_market({"P": [100, 100, 96, 48, 48, 48, 48], "W": [None, None, None, 8, 8, 8, None]},
                    {"P": equity(), "W": dict(security_type="warrant", first_session=D[3], last_session=D[5],
                     terminated_date=D[6], unquoted_valuation="none", source="t")},
                    splits=[("SP", "P", D[3], 2.)],
                    warrants=[dict(asset="W", underlying_asset="P", units_per_warrant=1., exercise_price=90.,
                                   exercisable_from=D[3], expiry_date=D[5], exercise_style="american",
                                   settlement="physical", source="t")])
