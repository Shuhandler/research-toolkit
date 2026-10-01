from datetime import date, timedelta
import math

import polars as pl
import pytest

import research_toolkit as rt
from research_toolkit._rebalancing import TARGET_SCHEMA


@pytest.fixture
def scheduled(inputs):
    def simulate(series=None, baskets=None, *, costs=None, leverage=1., dividends=(), splits=(),
                 receivable_policy="reserve", maximum=1., borrowing_rate=0., cash_rate=0., threshold=.25):
        series = series or {"A": [100., 100., 120., 120., 132.], "B": [100., 100., 100., 100., 100.]}
        dates = [date(2024, 1, 2)+timedelta(days=i) for i in range(len(next(iter(series.values()))))]
        market = rt.prepare_market_data(**inputs(series=series, dates=dates, splits=splits, dividends=dividends))
        assets = sorted(series)
        if baskets is None:
            baskets = [(1, {a: 1/len(assets) for a in assets}, leverage),
                       (2, {a: 1/len(assets) for a in assets}, leverage)]
        targets = pl.DataFrame([(dates[i-1], dates[i], a, float(weights[a]), float(l))
            for i, weights, l in baskets for a in assets], schema=TARGET_SCHEMA, orient="row")
        kwargs = dict(targets=targets, initial_capital=100., entry_session=dates[1], end_session=dates[-1],
            policy=rt.RebalancePolicy(execution="scheduled_close", sizing="post_cost_equity", fractional_shares=True,
                terminal_action="mark_only", non_session="raise", receivable_policy=receivable_policy,
                max_asset_weight=maximum), costs=costs or rt.TradeCosts(commission_bps=0., half_spread_bps=0., impact_bps=0.),
            financing=rt.Financing(cash_rate=cash_rate, borrowing_rate=borrowing_rate, day_count="ACT/365F",
                maintenance_equity_ratio=threshold, on_breach="stop", cash_sweep="repay_debt"))
        return rt.scheduled_rebalance(market, **kwargs), market, kwargs
    return simulate


def reconcile(result):
    capital = result.metadata["initial_capital"]
    money_tolerance = min(.009, 1e-8 + 1e-12*capital)
    # Independently reconstruct balances and quantities from all posted events.
    for row in result.valuations.filter(pl.col("phase").is_in(["post_entry", "close"])).iter_rows(named=True):
        events = result.events.filter(pl.col("date") <= row["session"])
        assert row["cash"] == pytest.approx(capital+events["cash_delta"].sum(), abs=money_tolerance)
        assert row["debt"] == pytest.approx(events["debt_delta"].sum(), abs=money_tolerance)
        assert row["dividend_receivable"] == pytest.approx(events["receivable_delta"].sum(), abs=money_tolerance)
        positions = result.positions.filter(pl.col("session") == row["session"])
        for pos in positions.iter_rows(named=True):
            assert pos["quantity"] == pytest.approx(events.filter(pl.col("asset") == pos["asset"])["quantity_delta"].sum(), abs=1e-12)
        mv = math.fsum(q*p for q, p in positions.select("quantity", "raw_mark").iter_rows())
        assert row["equity"] == pytest.approx(mv+row["cash"]+row["dividend_receivable"]-row["debt"], abs=money_tolerance)
    for row in result.daily.iter_rows(named=True):
        assert row["pnl"] == pytest.approx(result.attribution.filter(pl.col("session") == row["session"])["pnl"].sum(), abs=money_tolerance)
    for trade in result.trades.iter_rows(named=True):
        cost = result.costs.filter(pl.col("trade_id") == trade["trade_id"])["amount"].sum()
        assert cost == pytest.approx(trade["trade_cost"])
    assert (result.diagnostics["residual"].abs() <= result.diagnostics["tolerance"]).all()
    assert result.daily["equity"][-1] == pytest.approx(capital*(1+result.daily["compounded_return"][-1]))


def test_drift_then_rebalance_hand_oracle(scheduled):
    result, _, _ = scheduled()
    trades = result.trades.filter(pl.col("execution") == "scheduled_close")
    assert trades["asset"].to_list() == ["A", "B"]  # Sales precede buys.
    assert trades["signed_notional"].to_list() == pytest.approx([-5, 5])
    assert result.daily["pnl"].to_list() == pytest.approx([10, 0, 5.5])
    assert result.daily["equity"].to_list() == pytest.approx([110, 110, 115.5])
    assert result.turnover["phase"].to_list() == ["entry", "rebalance"]
    assert result.turnover["turnover"].to_list() == pytest.approx([1, 10/110])
    reconcile(result)
    report = rt.performance(result, periods_per_year=252, risk_free_annual_effective=0., minimum_acceptable_return_annual_effective=0.)
    assert report.equity["phase"].to_list() == ["pre_entry", "post_entry", "pre_rebalance", "close", "close", "close"]
    assert report.allocation.group_by("session").agg(pl.col("weight").sum())["weight"].to_list() == pytest.approx([1.]*4)


def test_no_quantity_change_no_trade_or_fees(scheduled):
    costs = rt.TradeCosts(commission_bps=20., half_spread_bps=10., impact_bps=5.)
    result, _, _ = scheduled(series={"A": [100.]*5, "B": [80.]*5}, costs=costs)
    assert result.trades.height == 2
    assert result.rebalances["trade_cost"][0] == 0
    assert result.rebalances["gross_traded_notional"][0] == 0
    assert result.costs["date"].n_unique() == 1
    reconcile(result)


def test_full_switch_costs_and_exit(scheduled):
    result, _, _ = scheduled(series={"A": [100.]*5, "B": [50.]*5},
        baskets=[(1, {"A": 1., "B": 0.}, 1.), (2, {"A": 0., "B": 1.}, 1.)],
        costs=rt.TradeCosts(commission_bps=100., half_spread_bps=0., impact_bps=0.))
    # Entry buys 100/1.01. Sale yields 99% of that; purchase consumes 101%.
    after = 100*.99/(1.01**2)
    assert result.daily["equity"][0] == pytest.approx(after)
    assert result.positions.filter((pl.col("asset") == "A") & (pl.col("session") > date(2024, 1, 3)))["quantity"].to_list() == [0.]*3
    assert result.trades["signed_notional"].to_list() == pytest.approx([100/1.01, -100/1.01, after])
    reconcile(result)


def test_leverage_changes_reconcile_and_finance(scheduled):
    result, _, _ = scheduled(baskets=[(1, {"A": .5, "B": .5}, 1.),
        (2, {"A": .2, "B": .8}, 2.), (3, {"A": .8, "B": .2}, 0.)], borrowing_rate=.365,
        costs=rt.TradeCosts(commission_bps={"A": 10., "B": 20.}, half_spread_bps=5., impact_bps=0.))
    assert result.daily["debt"][0] > 0
    assert result.daily["debt"][1] == 0
    assert result.costs.filter(pl.col("component") == "borrowing_interest").height == 1
    assert result.daily["gross_exposure"][1] == 0
    reconcile(result)


def test_split_does_not_trigger_spurious_rebalance(scheduled):
    result, _, _ = scheduled(series={"A": [100., 100., 50., 50., 50.]},
        splits=[("split", "A", date(2024, 1, 4), 2.)])
    assert result.trades.height == 1
    assert result.rebalances["gross_traded_notional"][0] == 0
    assert result.daily["pnl"].to_list() == [0.]*3
    reconcile(result)


def test_entitlement_before_sale_and_payment_not_income_twice(scheduled):
    result, _, _ = scheduled(series={"A": [100., 100., 99., 99., 99.], "B": [100.]*5},
        baskets=[(1, {"A": 1., "B": 0.}, 1.), (2, {"A": 0., "B": 1.}, 1.)],
        dividends=[("div", "A", date(2024, 1, 4), date(2024, 1, 5), 1.)])
    assert result.daily["pnl"].to_list() == pytest.approx([0, 0, 0])
    assert result.rebalances["receivable_reserved"][0] == pytest.approx(1)
    assert result.rebalances["actual_gross_leverage"][0] == pytest.approx(.99)
    assert result.daily["cash"].to_list() == [0., 1., 1.]
    assert result.daily["dividend_receivable"].to_list() == [1., 0., 0.]
    reconcile(result)
    with pytest.raises(ValueError, match="receivables"):
        scheduled(series={"A": [100., 100., 99., 99., 99.], "B": [100.]*5},
            baskets=[(1, {"A": 1., "B": 0.}, 1.), (2, {"A": 0., "B": 1.}, 1.)],
            dividends=[("div", "A", date(2024, 1, 4), date(2024, 1, 5), 1.)], receivable_policy="require_target")


def test_pretrade_margin_cannot_be_rescued_by_schedule(scheduled):
    result, _, _ = scheduled(series={"A": [100., 100., 60., 100., 110.]},
        baskets=[(1, {"A": 1.}, 2.), (2, {"A": 1.}, 0.)])
    assert result.status == "stopped" and result.stop_session == date(2024, 1, 4)
    assert result.trades.height == 1 and result.rebalances.is_empty()
    with pytest.raises(ValueError, match="stopped"):
        rt.performance(result, periods_per_year=252, risk_free_annual_effective=0., minimum_acceptable_return_annual_effective=0.)
    reconcile(result)


def test_initial_only_matches_buy_hold(scheduled):
    result, market, kwargs = scheduled(baskets=[(1, {"A": .4, "B": .6}, 1.5)], borrowing_rate=.08,
        costs=rt.TradeCosts(commission_bps=5., half_spread_bps=2., impact_bps=1.))
    hold = rt.buy_and_hold(market, weights={"A": .4, "B": .6}, initial_capital=100.,
        entry_session=kwargs["entry_session"], end_session=kwargs["end_session"],
        policy=rt.BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity", fractional_shares=True,
            initial_gross_leverage=1.5, terminal_action="mark_only"), costs=kwargs["costs"], financing=kwargs["financing"])
    for name in ("daily", "positions", "trades", "costs", "events", "valuations", "attribution"):
        assert getattr(result, name).equals(getattr(hold, name))


@pytest.mark.parametrize("change", ["same_day", "non_session", "missing_asset", "negative", "concentration", "terminal"])
def test_invalid_targets(scheduled, change):
    _, market, kwargs = scheduled()
    targets = kwargs["targets"]
    if change == "same_day": targets = targets.with_columns(pl.col("session").alias("decision_session"))
    if change == "non_session": targets = targets.with_columns((pl.col("session")+pl.duration(days=100)).alias("session"))
    if change == "missing_asset": targets = targets.head(3)
    if change == "negative": targets = targets.with_columns(pl.lit(-.5).alias("weight"))
    if change == "concentration":
        from dataclasses import replace
        kwargs["policy"] = replace(kwargs["policy"], max_asset_weight=.4)
    if change == "terminal": targets = targets.with_columns(pl.when(pl.col("session") == date(2024, 1, 4)).then(pl.lit(kwargs["end_session"])).otherwise(pl.col("session")).alias("session"))
    with pytest.raises(ValueError): rt.scheduled_rebalance(market, **(kwargs | {"targets": targets}))


def test_costs_only_on_changed_notional_with_asymmetric_components(scheduled):
    costs = rt.TradeCosts(commission_bps={"A": 100., "B": 200.},
                         half_spread_bps={"A": 50., "B": 25.}, impact_bps=10.)
    result, _, _ = scheduled(costs=costs)
    for row in result.trades.iter_rows(named=True):
        rates = {"A": .016, "B": .0235}
        assert row["trade_cost"] == pytest.approx(abs(row["signed_notional"])*rates[row["asset"]])
    reconcile(result)


def test_unlevered_reserve_with_costs(scheduled):
    result, _, _ = scheduled(series={"A": [100., 100., 99., 99., 99.], "B": [100.]*5},
        baskets=[(1, {"A": .5, "B": .5}, 1.), (2, {"A": .2, "B": .8}, 1.)],
        dividends=[("div", "A", date(2024, 1, 4), date(2024, 1, 6), 1.)],
        costs=rt.TradeCosts(commission_bps=10., half_spread_bps=10., impact_bps=10.), threshold=None)
    assert result.daily["debt"].to_list() == [0.]*3
    assert result.rebalances["receivable_reserved"][0] == pytest.approx(.5/1.003)
    reconcile(result)


def test_concentration_drift_does_not_force_trades(scheduled):
    result, _, _ = scheduled(series={"A": [100., 100., 200., 300., 400.], "B": [100.]*5},
        baskets=[(1, {"A": .5, "B": .5}, 1.)], maximum=.5)
    assert result.trades.height == 2
    assert result.positions.filter(pl.col("asset") == "A")["weight"][-1] == pytest.approx(.8)


def test_future_prices_do_not_change_past_fills(scheduled):
    first, market, kwargs = scheduled()
    args = {name: getattr(market, name) for name in ("prices", "sessions", "splits", "dividends", "metadata")}
    args["prices"] = market.prices.with_columns(pl.when(pl.col("session") > date(2024, 1, 4))
        .then(pl.col("close")*1.5).otherwise(pl.col("close")).alias("close"))
    second = rt.scheduled_rebalance(rt.prepare_market_data(**args), **kwargs)
    assert first.trades.equals(second.trades)
    assert first.rebalances.equals(second.rebalances)
    assert first.daily.head(1).equals(second.daily.head(1))


def test_turnover_is_zero_for_cash_only_targets(scheduled):
    result, _, _ = scheduled(leverage=0.)
    assert result.trades.is_empty() and result.costs.is_empty()
    assert result.turnover["turnover"].to_list() == [0., 0.]
    reconcile(result)


def test_rolling_partial_requires_opt_in(scheduled):
    result, _, _ = scheduled(series={"A": [100., 100., 40., 100., 110.]}, leverage=2.)
    with pytest.raises(ValueError, match="stopped"):
        rt.rolling_risk(result, window=2, periods_per_year=252, risk_free_annual_effective=0.)
    partial = rt.rolling_risk(result, window=2, periods_per_year=252, risk_free_annual_effective=0., allow_partial=True)
    assert partial.metadata["stop_reason"] == "nonpositive_equity"
    assert partial.metadata["actual_end_session"] == result.stop_session.isoformat()


def test_one_year_style_example_reconciles_every_event():
    from pathlib import Path
    import runpy
    example = runpy.run_path(str(Path(__file__).resolve().parents[1]/"examples"/"scheduled_rebalancing.py"))
    outputs = example["run_example"]()
    result = outputs["result"]
    assert result.status == "complete"
    assert result.rebalances.height == 9
    assert result.trades.height > outputs["hold"].trades.height
    assert not result.positions.equals(outputs["hold"].positions)
    reconcile(result)
    for allocation in outputs["allocations"]:
        assert allocation.metadata["sample_end"] < allocation.metadata["decision_session"]


def test_reserve_liquidation_retains_only_receivable_secured_debt(scheduled):
    result, _, _ = scheduled(series={"A": [100., 100., 20., 20., 20.]},
        baskets=[(1, {"A": 1.}, 2.), (2, {"A": 1.}, 0.)],
        dividends=[("div", "A", date(2024, 1, 4), date(2024, 1, 5), 80.)])
    # Two shares earn $160 receivable. Liquidation raises $40 to repay the $100
    # loan. The remaining $60 is repaid when the receivable actually pays.
    assert result.daily["debt"].to_list() == pytest.approx([60., 0., 0.])
    assert result.daily["dividend_receivable"].to_list() == [160., 0., 0.]
    assert result.daily["cash"].to_list() == [0., 100., 100.]
    assert result.daily["equity"].to_list() == pytest.approx([100.]*3)
    reconcile(result)


def test_purchase_at_ex_close_does_not_earn_prior_entitlement(scheduled):
    result, _, _ = scheduled(series={"A": [100.]*5, "B": [100., 100., 99., 99., 99.]},
        baskets=[(1, {"A": 1., "B": 0.}, 1.), (2, {"A": 0., "B": 1.}, 1.)],
        dividends=[("div", "B", date(2024, 1, 4), date(2024, 1, 5), 1.)])
    assert result.receivables.is_empty()
    assert result.daily["pnl"].to_list() == pytest.approx([0.]*3)
    reconcile(result)
