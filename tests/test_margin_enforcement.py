"""Opt-out maintenance-margin enforcement: hand-checked breaches that keep running."""
from datetime import date

import polars as pl
import pytest

import research_toolkit as rt
from test_corporate_actions import D, HOLD, POLICY, SIGNED, ZERO, borrow as ca_borrow, cash_merger_market
from test_long_short import borrow, loan, ls, scheduled as signed_scheduled, simulate as signed
from test_rebalancing import reconcile

ENFORCEMENT_KEYS = {"maintenance_margin_enforced", "maintenance_margin_breach_action"}


def leveraged(run, policy, market, ratio=.4, enforce=None, **financing):
    # 2x long: 2 shares at 100 on 100 capital and 100 debt.
    options = dict(policy=policy(2), cash_rate=None, cash_day_count=None,
                   financing=rt.Financing(**(dict(cash_rate=0., borrowing_rate=0., day_count="ACT/365F",
                       maintenance_equity_ratio=ratio, on_breach="stop", cash_sweep="repay_debt") | financing)))
    if enforce is not None:
        options["enforce_maintenance_margin"] = enforce
    return run(market, **options)


def test_default_enforces_and_explicit_true_is_identical(inputs, run, policy):
    market = rt.prepare_market_data(**inputs({"A": [100, 75, 200]}))
    default, explicit = leveraged(run, policy, market), leveraged(run, policy, market, enforce=True)
    assert default.status == "stopped" and default.stop_reason == "maintenance_margin_breach"
    assert default.metadata == explicit.metadata
    for name in ("daily", "positions", "trades", "events", "valuations", "attribution", "diagnostics"):
        assert getattr(default, name).equals(getattr(explicit, name))
    assert default.metadata["maintenance_margin_enforced"] is True
    assert default.metadata["maintenance_margin_breach_action"] == "stop"
    # Value 150, equity 50: requirement .4*150=60, shortfall 10.
    assert default.daily["margin_required"].to_list() == pytest.approx([60])
    assert default.daily["margin_shortfall"].to_list() == pytest.approx([10])
    with pytest.raises(ValueError, match="True or False"):
        leveraged(run, policy, market, enforce=0)


def test_disabled_long_run_continues_past_breach_with_debt_interest(inputs, run, policy):
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    market = rt.prepare_market_data(**inputs({"A": [100, 75, 70, 200]}, dates=dates))
    result = leveraged(run, policy, market, enforce=False, borrowing_rate=.365)
    assert result.require_complete() is result
    assert result.metadata["maintenance_margin_enforced"] is False
    assert result.metadata["maintenance_margin_breach_action"] == "report_only"
    # Debt capitalizes .1% per calendar day: 100 -> 100.1 -> 100.2001 -> 100.3003001.
    debt = [100.1, 100.2001, 100.3003001]
    assert result.daily["debt"].to_list() == pytest.approx(debt)
    assert result.daily["equity"].to_list() == pytest.approx([150-debt[0], 140-debt[1], 400-debt[2]])
    assert result.daily["margin_breached"].to_list() == [True, True, False]
    assert result.daily["margin_required"].to_list() == pytest.approx([60, 56, 160])
    assert result.daily["margin_shortfall"].to_list() == pytest.approx([60-(150-debt[0]), 56-(140-debt[1]), 0])
    assert result.trades.height == 1  # No forced liquidation.
    assert "maintenance_margin_breach" not in result.events["type"].to_list()
    reconcile(result)


def test_results_before_and_at_breach_match_the_enforced_run(inputs, run, policy):
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9)]
    market = rt.prepare_market_data(**inputs({"A": [100, 110, 90, 75, 120]}, dates=dates,
        dividends=[("D", "A", date(2024, 1, 5), date(2024, 1, 10), 1.)]))
    on = leveraged(run, policy, market, borrowing_rate=.05, cash_rate=.01)
    off = leveraged(run, policy, market, enforce=False, borrowing_rate=.05, cash_rate=.01)
    breach = date(2024, 1, 8)
    assert on.stop_session == breach and off.status == "complete"
    assert off.daily.filter(pl.col("session") > breach)["equity"].item() > 0
    for name in ("daily", "positions", "trades", "valuations", "attribution", "receivables", "costs", "diagnostics"):
        assert getattr(on, name).equals(getattr(off, name).filter(pl.col(
            "date" if name == "costs" else "session") <= breach)), name
    # The only extra event in the enforced run is its stop marker.
    assert on.events.filter(pl.col("type") != "maintenance_margin_breach").equals(
        off.events.filter(pl.col("date") <= breach))
    changed = {k for k in on.metadata if on.metadata[k] != off.metadata[k]}
    assert changed == ENFORCEMENT_KEYS | {"status", "stop_reason", "stop_session", "stop_time", "actual_end_session"}


def test_disabled_short_keeps_collateral_borrow_fees_rebate_and_cash_interest(inputs, policy):
    market = rt.prepare_market_data(**inputs({"A": [100, 160, 90]}))
    args = dict(long_short=ls(short_margin=.5, rebate_rate=.365), stock_borrow=borrow({"A": .365}),
                financing=loan(cash_rate=.365))
    on = signed(market, policy, **args)
    off = signed(market, policy, enforce_maintenance_margin=False, **args)
    assert on.stop_reason == "maintenance_margin_breach" and off.require_complete() is off
    # 2024-01-03: 100 + .1 interest - .1 fee + .1 rebate - 60 short loss; then +.0401 interest +70 gain.
    assert off.daily["equity"].to_list() == pytest.approx([40.1, 40.1+.0401+70])
    assert off.daily["margin_required"].to_list() == pytest.approx([80, 45])
    assert off.daily["margin_shortfall"].to_list() == pytest.approx([39.9, 0])
    assert off.daily["restricted_collateral"].to_list() == [160, 90]
    # Fees and rebates after the breach use the prior close's 160 short/collateral.
    assert off.stock_borrow_accruals["amount"].to_list() == pytest.approx([.1, .16])
    assert off.short_financing_accruals["rebate"].to_list() == pytest.approx([.1, .16])
    assert on.stock_borrow_accruals.equals(off.stock_borrow_accruals.head(1))
    assert off.events.filter(pl.col("type") == "cash_interest")["cash_delta"].to_list() == pytest.approx([.1, .0401])


@pytest.mark.parametrize("prices", [[100, 50, 120], [100, 45, 120]])
def test_disabled_long_still_stops_at_nonpositive_equity(inputs, run, policy, prices):
    market = rt.prepare_market_data(**inputs({"A": prices}))
    result = leveraged(run, policy, market, enforce=False)
    assert result.status == "stopped" and result.stop_reason == "nonpositive_equity"
    assert result.daily["equity"].to_list() == [2*prices[1]-100]
    with pytest.raises(ValueError, match="stopped"):
        result.require_complete()


@pytest.mark.parametrize("last,equity", [(200, 0), (220, -20)])
def test_disabled_short_still_stops_at_nonpositive_equity(inputs, policy, last, equity):
    market = rt.prepare_market_data(**inputs({"A": [100, last, 90]}))
    result = signed(market, policy, long_short=ls(short_margin=.5), enforce_maintenance_margin=False)
    assert result.stop_reason == "nonpositive_equity" and result.daily["equity"].item() == equity


def test_entry_checks_are_skipped_only_when_disabled(market, run, policy, inputs):
    # 3x long gives a 1/3 equity ratio, below the .4 requirement at entry.
    with pytest.raises(ValueError, match="initial leverage violates"):
        run(market, policy=policy(3), financing=rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F",
            maintenance_equity_ratio=.4, on_breach="stop", cash_sweep="repay_debt"), cash_rate=None, cash_day_count=None)
    result = run(market, policy=policy(3), financing=rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F",
        maintenance_equity_ratio=.4, on_breach="stop", cash_sweep="repay_debt"), cash_rate=None, cash_day_count=None,
        enforce_maintenance_margin=False)
    assert result.status == "complete" and result.daily["margin_breached"].all()
    # Signed 1.5 long / 1.5 short needs .5*300 = 150 against 100 equity.
    two = rt.prepare_market_data(**inputs({"A": [100, 100, 100], "B": [100, 100, 100]}))
    args = dict(quantities=None, equity_exposures={"A": 1.5, "B": -1.5}, long_short=ls(long_margin=.5, short_margin=.5),
                stock_borrow=borrow({"B": 0.}))
    with pytest.raises(ValueError, match="signed entry violates"):
        signed(two, policy, **args)
    result = signed(two, policy, enforce_maintenance_margin=False, **args)
    assert result.daily["margin_shortfall"].to_list() == pytest.approx([50, 50])
    # Borrowing still needs a declared ratio: it is a reporting input, not only a stop rule.
    with pytest.raises(ValueError, match="requires maintenance_equity_ratio"):
        run(market, policy=policy(2), financing=loan(), cash_rate=None, cash_day_count=None,
            enforce_maintenance_margin=False)


def test_scheduled_long_trades_through_a_breach(scheduled):
    series = {"A": [100., 100., 60., 100., 110.]}
    baskets = [(1, {"A": 1.}, 2.), (3, {"A": 1.}, 1.)]
    with pytest.raises(ValueError, match="target leverage violates"):
        scheduled(series=series, baskets=[(1, {"A": 1.}, 5.)])
    on, _, _ = scheduled(series=series, baskets=baskets)
    assert on.stop_reason == "maintenance_margin_breach" and on.rebalances.is_empty()
    result, market, kwargs = scheduled(series=series, baskets=baskets)
    off = rt.scheduled_rebalance(market, **kwargs, enforce_maintenance_margin=False)
    assert off.require_complete() is off and off.metadata["maintenance_margin_enforced"] is False
    # 2 shares at 60 against 100 debt: equity 20, requirement 30, shortfall 10. At 100,
    # sell one share to reach 1x and repay the debt; then hold one share to 110.
    assert off.daily["equity"].to_list() == pytest.approx([20, 100, 110])
    assert off.daily["margin_shortfall"].to_list() == pytest.approx([10, 0, 0])
    assert off.trades["signed_quantity"].to_list() == pytest.approx([2, -1])
    assert off.daily["debt"].to_list() == pytest.approx([100, 0, 0])
    reconcile(off)


def test_scheduled_signed_cover_executes_after_pre_trade_breach(inputs):
    dates = [date(2024, 1, i) for i in range(2, 7)]
    market = rt.prepare_market_data(**inputs({"A": [100, 100, 160, 90, 90]}, dates=dates))
    args = dict(long_short=ls(short_margin=.5), costs=ZERO)
    on = signed_scheduled(market, dates, {1: {"A": -1}, 2: {"A": 0}}, **args)
    off = signed_scheduled(market, dates, {1: {"A": -1}, 2: {"A": 0}}, enforce_maintenance_margin=False, **args)
    assert on.stop_reason == "maintenance_margin_breach" and on.rebalances.is_empty()
    assert off.require_complete() is off
    assert off.trades["signed_quantity"].to_list() == [-1, 1]
    assert off.daily["equity"].to_list() == pytest.approx([40, 40, 40])
    pre = off.valuations.filter(pl.col("phase") == "pre_rebalance")
    assert pre["equity"].item() == pytest.approx(40)  # The pre-trade breach remains auditable.


def test_corporate_action_obligation_requirement_reported_but_not_enforced():
    args = dict(quantities={"T": -5., "A": 20.}, initial_capital=1000., entry_session=D[0], end_session=D[-1],
                policy=HOLD, costs=ZERO, stock_borrow=ca_borrow({"T": .365}), corporate_actions=POLICY, **SIGNED)
    on = rt.buy_and_hold(cash_merger_market(), **args)
    off = rt.buy_and_hold(cash_merger_market(), enforce_maintenance_margin=False, **args)
    assert on.stop_reason == "maintenance_margin_breach" and on.stop_session == D[3]
    assert off.require_complete() is off
    # Long 1000 at .5 plus a 525 cash obligation at obligation_margin=1 until settlement on D[5].
    assert off.daily["margin_required"].to_list() == pytest.approx([750, 750, 1025, 1025, 500, 500])
    assert off.daily["margin_shortfall"].to_list() == pytest.approx([0, 0, 51, 51, 0, 0])
    assert off.daily["restricted_collateral"][-1] == 0.
    assert off.daily["equity"][-1] == pytest.approx(on.daily["equity"][-1])
