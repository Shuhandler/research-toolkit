from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import math

import polars as pl
import pytest
import research_toolkit as rt
from test_allocation import panel, OPTIONS
from test_rebalancing import reconcile, scheduled  # synthetic shared accounting fixture


def model(assets=("A",), *, sigma=.1, adv=100., coefficient=1., commission=0., spread=0., per_share=0.):
    liquidity = rt.LiquidityResult(pl.DataFrame({"asset": list(assets), "daily_volatility": [sigma]*len(assets),
        "dollar_adv": [adv]*len(assets), "n_obs": [10]*len(assets)}),
        dict(source="synthetic declared estimates", currency="USD", volatility_unit="daily_decimal",
            adv_unit="currency_per_trading_day", sample_end="2023-12-29", decision_session="2024-01-01"))
    return rt.SquareRootImpactCosts(liquidity=liquidity, commission_bps=commission, half_spread_bps=spread,
        commission_per_share=per_share, impact_coefficient=coefficient)


def test_capped_water_filling_and_cutoff(inputs):
    r = panel(inputs)
    capped = rt.inverse_volatility_weights(r, max_asset_weight=.6, cap_policy="redistribute", **OPTIONS)
    assert capped.weights["weight"].to_list() == pytest.approx([.6, .4])
    assert capped.diagnostics["uncapped_weight"].to_list() == pytest.approx([2/3, 1/3])
    assert capped.diagnostics["at_cap"].to_list() == [True, False]
    assert capped.weights["weight"].sum() == pytest.approx(1)
    changed = replace(r, values=r.values.with_columns(pl.when(pl.col("session") >= OPTIONS["decision_session"])
        .then(.7).otherwise(pl.col("simple_return")).alias("simple_return")))
    assert capped.weights.equals(rt.inverse_volatility_weights(changed, max_asset_weight=.6, cap_policy="redistribute", **OPTIONS).weights)
    for cap in [.1, .4999999999999]:
        with pytest.raises(ValueError, match="infeasible"):
            rt.inverse_volatility_weights(r, max_asset_weight=cap, cap_policy="redistribute", **OPTIONS)
    equal = rt.inverse_volatility_weights(r, max_asset_weight=.5, cap_policy="redistribute", **OPTIONS)
    assert equal.weights["weight"].to_list() == pytest.approx([.5, .5])


def test_multiple_caps_and_permutation(inputs):
    r = panel(inputs, {"A": [100., 101., 99.99, 101., 103.], "B": [100., 102., 99.96, 105., 110.],
                       "C": [100., 120., 96., 101., 90.]})
    result = rt.inverse_volatility_weights(r, max_asset_weight=.4, cap_policy="redistribute", **OPTIONS)
    assert result.weights["weight"].to_list() == pytest.approx([.4, .4, .2])
    assert result.weights.equals(rt.inverse_volatility_weights(replace(r, values=r.values.reverse()),
        max_asset_weight=.4, cap_policy="redistribute", **OPTIONS).weights)


def test_liquidity_window_and_availability(inputs):
    r = panel(inputs)
    dates = [date(2024, 1, 3), date(2024, 1, 4)]
    volume = pl.DataFrame([(d, a, v, datetime(d.year, d.month, d.day, 21, tzinfo=timezone.utc))
        for d, v in zip(dates, [100., 300.]) for a in ["A", "B"]],
        schema={"session": pl.Date, "asset": pl.String, "dollar_volume": pl.Float64, "available_at": pl.Datetime("us", "UTC")}, orient="row")
    kw = dict(decision_session=OPTIONS["decision_session"], lookback=2, metadata=dict(source="synthetic volume", currency="USD"))
    result = rt.estimate_liquidity(r, volume, **kw)
    assert result.estimates["daily_volatility"].to_list() == pytest.approx([math.sqrt(.02), math.sqrt(.08)])
    assert result.estimates["dollar_adv"].to_list() == [200., 200.]
    rt.SquareRootImpactCosts(liquidity=result, commission_bps=0, commission_per_share=0, half_spread_bps=0, impact_coefficient=.5)
    with pytest.raises(ValueError, match="unavailable"):
        rt.estimate_liquidity(r, volume.with_columns(pl.col("available_at")+pl.duration(days=3)), **kw)
    with pytest.raises(ValueError, match="complete"):
        rt.estimate_liquidity(r, volume.head(3), **kw)


def test_cost_components_scaling_and_zero_orders():
    costs = model(sigma=.02, adv=10_000., coefficient=.5, commission=2., spread=3., per_share=.01)
    orders = pl.DataFrame({"asset": ["A"], "signed_notional": [100.], "reference_price": [10.]})
    row = rt.estimate_trade_costs(orders, costs=costs, execution_session=date(2024, 1, 2)).orders.row(0, named=True)
    assert row["order_adv_ratio"] == .01
    assert row["impact_bps"] == 10.
    assert row["commission"] == pytest.approx(.12)  # $0.02 + ten shares * $0.01
    assert row["half_spread"] == .03
    assert row["impact"] == .1
    assert row["total_cost"] == .25
    for notional, impact in [(400., .8), (-100., .1), (0., 0.)]:
        got = rt.estimate_trade_costs(orders.with_columns(pl.lit(notional).alias("signed_notional")), costs=costs,
            execution_session=date(2024, 1, 2)).orders.row(0, named=True)
        assert got["impact"] == pytest.approx(impact)
        if notional == 0:
            assert got["total_cost"] == got["total_cost_bps"] == 0


@pytest.mark.parametrize("capital,leverage,debt", [(110., 1., 0.), (60., 2., 50.), (210., .5, 0.)])
def test_entry_sizing_and_simulation_hand_oracle(inputs, run, policy, capital, leverage, debt):
    # A $100 purchase costs exactly $10. Post-cost equities 100, 50, 200.
    costs = model()
    estimate = rt.size_entry_orders(weights={"A": 1.}, initial_capital=capital, gross_leverage=leverage,
        prices=pl.DataFrame({"asset": ["A"], "reference_price": [100.]}), costs=costs, execution_session=date(2024, 1, 2))
    assert estimate.orders["signed_notional"][0] == pytest.approx(100.)
    assert estimate.metadata["total_cost"] == pytest.approx(10.)
    assert estimate.metadata["debt"] == pytest.approx(debt)
    result = run(rt.prepare_market_data(**inputs(series={"A": [100.]*3})), initial_capital=capital, costs=costs,
        policy=policy(leverage), cash_rate=None, cash_day_count=None,
        financing=rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F", maintenance_equity_ratio=.25,
            on_breach="stop", cash_sweep="repay_debt"))
    assert result.daily["equity"].to_list() == pytest.approx([capital-10]*2)
    assert result.daily["pnl"].to_list() == pytest.approx([-10, 0])
    assert result.execution_costs["total_cost"].sum() == pytest.approx(10)
    assert result.trades["trade_cost"].sum() == pytest.approx(10)
    reconcile(result)


def test_scheduled_costs_are_based_on_actual_changed_orders(scheduled):
    result, _, _ = scheduled(costs=model(("A", "B"), sigma=.01, adv=1000., coefficient=.5, per_share=.01))
    audit = result.execution_costs
    assert audit.height == result.trades.height == 4
    assert audit["total_cost"].sum() == pytest.approx(result.trades["trade_cost"].sum())
    assert audit.filter(pl.col("session") == date(2024, 1, 4))["impact_bps"].max() < audit.filter(pl.col("session") == date(2024, 1, 3))["impact_bps"].min()
    assert result.rebalances["actual_gross_leverage"][0] == pytest.approx(1.)
    reconcile(result)
    zero, _, _ = scheduled(series={"A": [100.]*5}, costs=model())
    assert zero.trades.height == zero.execution_costs.height == 1
    assert zero.rebalances["trade_cost"].to_list() == [0.]
    reconcile(zero)


def test_cost_model_drip_stays_free(inputs, run):
    dates = [date(2024, 1, 2)+timedelta(days=i) for i in range(4)]
    market = rt.prepare_market_data(**inputs(series={"A": [100., 90., 90., 90.]}, dates=dates,
        dividends=[("d", "A", dates[1], dates[1], 10.)]))
    result = run(market, costs=model(), dividend_reinvestment=rt.DividendReinvestment(
        execution="first_close_on_or_after_payment", funding="before_debt_repayment", scheduled_collision="rebalance_only", terminal_action="hold_cash"))
    assert result.trades.height == 2
    assert result.trades["trade_cost"][0] > 0
    assert result.trades["trade_cost"][1] == 0
    assert result.dividend_reinvestments["trade_cost"].sum() == 0
    assert result.execution_costs.height == 1  # Actual modeled ordinary orders only.
    reconcile(result)


def test_cost_units_cutoffs_and_mutation_rejected():
    costs = model()
    orders = pl.DataFrame({"asset": ["A"], "signed_notional": [100.], "reference_price": [100.]})
    with pytest.raises(ValueError, match="cutoff"):
        rt.estimate_trade_costs(orders, costs=costs, execution_session=date(2023, 12, 31))
    with pytest.raises(ValueError, match="daily volatility"):
        replace(costs, liquidity=replace(costs.liquidity, metadata=costs.liquidity.metadata | {"volatility_unit": "annualized"}))
    costs.liquidity.metadata["source"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        rt.estimate_trade_costs(orders, costs=costs, execution_session=date(2024, 1, 2))


@pytest.mark.parametrize("baskets", [[(1, {"A": 1., "B": 0.}, 1.), (2, {"A": 0., "B": 1.}, 1.)],
    [(1, {"A": .5, "B": .5}, 1.), (2, {"A": .2, "B": .8}, 2.), (3, {"A": .5, "B": .5}, 0.)]])
def test_nonlinear_switch_leverage_and_liquidation(scheduled, baskets):
    result, _, _ = scheduled(baskets=baskets, costs=model(("A", "B"), sigma=.02, adv=1000., coefficient=.5), borrowing_rate=.1)
    result.require_complete()
    reconcile(result)
    assert result.execution_costs["total_cost"].sum() == pytest.approx(result.trades["trade_cost"].sum())
    assert result.rebalances["actual_gross_leverage"].to_list() == pytest.approx([b[2] for b in baskets[1:]])


def test_nonlinear_receivable_reserve_and_split_no_trade(scheduled):
    result, _, _ = scheduled(series={"A": [100., 100., 90., 90., 90.], "B": [100.]*5},
        baskets=[(1, {"A": 1., "B": 0.}, 1.), (2, {"A": 0., "B": 1.}, 1.)],
        dividends=[("d", "A", date(2024, 1, 4), date(2024, 1, 5), 10.)],
        costs=model(("A", "B"), sigma=.01, adv=1000., coefficient=.5))
    assert result.rebalances["receivable_reserved"][0] > 0
    assert result.daily["debt"].to_list() == [0.]*3
    reconcile(result)
    split, _, _ = scheduled(series={"A": [100., 100., 50., 50., 50.]}, costs=model(),
        splits=[("s", "A", date(2024, 1, 4), 2.)])
    assert split.trades.height == 1
    assert split.rebalances["trade_cost"][0] == 0.
    reconcile(split)


def test_nonlinear_all_cash_and_stopped_retention(inputs, run, policy):
    market = rt.prepare_market_data(**inputs(series={"A": [100., 40., 50.]}))
    cash = run(market, costs=model(), policy=policy(0.))
    assert cash.trades.is_empty() and cash.execution_costs.is_empty()
    assert cash.daily["equity"].to_list() == [100.]*2
    stopped = run(market, costs=model(), policy=policy(2.), cash_rate=None, cash_day_count=None,
        financing=rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F", maintenance_equity_ratio=.25,
            on_breach="stop", cash_sweep="repay_debt"))
    assert stopped.status == "stopped" and stopped.execution_costs.height == 1
    reconcile(stopped)
