from dataclasses import replace
from datetime import date, timedelta

import polars as pl
import pytest
import research_toolkit as rt
from research_toolkit._signals import SIGNAL_SCHEMA
from test_rebalancing import reconcile


def plan(market, instructions, *, rebalance="each_signal"):
    dates = market.sessions["session"].to_list()
    closes = dict(market.sessions.iter_rows())
    signals = pl.DataFrame([(dates[i], a, float(w), float(leverage), closes[dates[i]], closes[dates[i]])
        for i, weights, leverage in instructions for a, w in weights.items()], schema=SIGNAL_SCHEMA, orient="row")
    return rt.signal_targets(signals, sessions=market.sessions, execution="next_session_close", rebalance=rebalance,
        metadata=market.metadata | {"signal_definition": "explicit synthetic allocation instructions"})


def simulate(market, signals, *, costs=None, maximum=1., financing=None, drip=None):
    return rt.scheduled_rebalance(market, targets=signals, initial_capital=100., entry_session=signals.targets["session"][0],
        end_session=market.sessions["session"][-1],
        policy=rt.RebalancePolicy(execution="scheduled_close", sizing="post_cost_equity", fractional_shares=True,
            terminal_action="mark_only", non_session="raise", receivable_policy="reserve", max_asset_weight=maximum),
        financing=financing or rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F", maintenance_equity_ratio=.25,
            on_breach="stop", cash_sweep="repay_debt"),
        costs=costs or rt.TradeCosts(commission_bps=0., half_spread_bps=0., impact_bps=0.), dividend_reinvestment=drip)


def test_close_signal_does_not_earn_execution_bar_or_weekend(inputs):
    dates = [date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9)]
    market = rt.prepare_market_data(**inputs(series={"A": [100., 200., 220.]}, dates=dates))
    signals = plan(market, [(0, {"A": 1.}, 1.)])
    assert signals.targets["session"][0] == dates[1]
    result = simulate(market, signals)
    assert result.trades["reference_price"][0] == 200
    assert result.trades["signed_quantity"][0] == .5
    assert result.daily["pnl"].to_list() == pytest.approx([10])
    assert result.daily["simple_return"].to_list() == pytest.approx([.1])
    assert result.signal_audit["decision_at"][0] < result.signal_audit["execution_at"][0]
    assert result.signal_audit["status"].to_list() == ["processed"]
    assert result.metadata["signals"]["execution"] == "next_session_close"
    reconcile(result)


def test_explicit_each_signal_vs_on_change_controls_rebalancing(inputs):
    dates = [date(2024, 1, 2)+timedelta(days=i) for i in range(5)]
    market = rt.prepare_market_data(**inputs(series={"A": [100., 100., 120., 132., 132.], "B": [100.]*5}, dates=dates))
    instructions = [(0, {"A": .5, "B": .5}, 1.), (1, {"A": .5, "B": .5}, 1.)]
    every = simulate(market, plan(market, instructions))
    changed = simulate(market, plan(market, instructions, rebalance="on_change"))
    assert every.trades["signed_notional"].to_list() == pytest.approx([50., 50., -5., 5.])
    assert every.daily["equity"][-1] == pytest.approx(115.5)
    assert changed.daily["equity"][-1] == pytest.approx(116.)
    assert changed.trades.height == 2 and changed.rebalances.is_empty()
    assert changed.signal_audit["status"].to_list() == ["processed"]*2+["unchanged_no_target"]*2
    reconcile(every)
    reconcile(changed)


def test_cash_signal_and_changed_signal_next_close(inputs):
    dates = [date(2024, 1, 2)+timedelta(days=i) for i in range(6)]
    market = rt.prepare_market_data(**inputs(series={"A": [100., 100., 100., 120., 60., 60.]}, dates=dates))
    # Decision day 1 buys at day 2, day 2 says sell at day 3; rise into day 3 belongs to old holding.
    signals = plan(market, [(0, {"A": 1.}, 0.), (1, {"A": 1.}, 1.), (2, {"A": 1.}, 0.)], rebalance="on_change")
    result = simulate(market, signals)
    assert result.trades["session"].to_list() == [dates[2], dates[3]]
    assert result.trades["signed_notional"].to_list() == [100., -120.]
    assert result.daily["pnl"].to_list() == [0., 20., 0., 0.]
    assert result.signal_audit["status"].to_list() == ["processed"]*3
    reconcile(result)


def test_future_observations_and_signals_leave_prior_trades_unchanged(inputs):
    dates = [date(2024, 1, 2)+timedelta(days=i) for i in range(6)]
    raw = inputs(series={"A": [100., 100., 110., 120., 130., 140.]}, dates=dates)
    first = rt.prepare_market_data(**raw)
    second = rt.prepare_market_data(**(raw | {"prices": raw["prices"].with_columns(
        pl.when(pl.col("session") >= dates[4]).then(pl.col("close")*2).otherwise(pl.col("close")).alias("close"))}))
    a = simulate(first, plan(first, [(0, {"A": 1.}, 1.), (2, {"A": 1.}, 0.), (3, {"A": 1.}, 1.)]))
    b = simulate(second, plan(second, [(0, {"A": 1.}, 1.), (2, {"A": 1.}, 0.), (3, {"A": 1.}, 2.)]))
    assert a.trades.filter(pl.col("session") < dates[4]).equals(b.trades.filter(pl.col("session") < dates[4]))
    assert a.daily.filter(pl.col("session") < dates[4]).equals(b.daily.filter(pl.col("session") < dates[4]))
    reconcile(a)
    reconcile(b)


@pytest.mark.parametrize("edit", ["late", "before_observation", "negative", "wrong_sum", "last_session", "non_session", "same_close", "missing_asset", "mutated"])
def test_signal_contract_rejects_invalid_inputs(inputs, edit):
    dates = [date(2024, 1, 2)+timedelta(days=i) for i in range(5)]
    market = rt.prepare_market_data(**inputs(series={"A": [100.]*5, "B": [100.]*5}, dates=dates))
    good = plan(market, [(0, {"A": .5, "B": .5}, 1.), (1, {"A": .5, "B": .5}, 1.)])
    signals = good.signals
    if edit == "late": signals = signals.with_columns(pl.col("available_at")+pl.duration(seconds=1))
    if edit == "before_observation": signals = signals.with_columns(pl.col("available_at")-pl.duration(seconds=1))
    if edit == "negative": signals = signals.with_columns(pl.when(pl.col("asset") == "A").then(-.5).otherwise(1.5).alias("weight"))
    if edit == "wrong_sum": signals = signals.with_columns(pl.lit(.4).alias("weight"))
    if edit == "last_session": signals = signals.head(2).with_columns(pl.lit(dates[-1]).alias("decision_session"))
    if edit == "non_session": signals = signals.with_columns(pl.col("decision_session")-pl.duration(days=30))
    if edit == "missing_asset": signals = signals.head(3)
    with pytest.raises(ValueError):
        if edit == "mutated":
            simulate(market, replace(good, targets=good.targets.with_columns(pl.lit(0.).alias("gross_leverage"))))
        else:
            rt.signal_targets(signals, sessions=market.sessions, execution="same_close" if edit == "same_close" else "next_session_close",
                rebalance="each_signal", metadata=good.metadata)


def test_calendar_mismatch_terminal_and_concentration_are_rejected(inputs):
    dates = [date(2024, 1, 2)+timedelta(days=i) for i in range(5)]
    market = rt.prepare_market_data(**inputs(series={"A": [100.]*5, "B": [100.]*5}, dates=dates))
    terminal = plan(market, [(0, {"A": .5, "B": .5}, 1.), (3, {"A": .5, "B": .5}, 1.)])
    with pytest.raises(ValueError, match="mark-only"): simulate(market, terminal)
    capped = plan(market, [(0, {"A": .8, "B": .2}, 1.)])
    with pytest.raises(ValueError, match="max_asset_weight"): simulate(market, capped, maximum=.6)
    mismatch = rt.signal_targets(capped.signals, sessions=market.sessions.with_columns(pl.col("close_at")+pl.duration(minutes=1)),
        execution="next_session_close", rebalance="each_signal", metadata=capped.metadata)
    with pytest.raises(ValueError, match="calendar"): simulate(market, mismatch)


def test_signal_result_and_plain_targets_use_identical_accounting(inputs):
    from test_execution_extensions import model
    dates = [date(2024, 1, 2)+timedelta(days=i) for i in range(6)]
    market = rt.prepare_market_data(**inputs(series={"A": [100., 100., 110., 120., 115., 118.]}, dates=dates))
    signals = plan(market, [(0, {"A": 1.}, 1.), (1, {"A": 1.}, 2.), (3, {"A": 1.}, 0.)])
    signal_run = simulate(market, signals, costs=model(sigma=.01))
    plain = rt.scheduled_rebalance(market, targets=signals.targets, initial_capital=100., entry_session=dates[1], end_session=dates[-1],
        policy=rt.RebalancePolicy(execution="scheduled_close", sizing="post_cost_equity", fractional_shares=True,
            terminal_action="mark_only", non_session="raise", receivable_policy="reserve", max_asset_weight=1.),
        financing=rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F", maintenance_equity_ratio=.25,
            on_breach="stop", cash_sweep="repay_debt"), costs=model(sigma=.01))
    for name in ("daily", "events", "positions", "trades", "costs", "execution_costs"):
        assert getattr(signal_run, name).equals(getattr(plain, name))
    assert plain.signal_audit.is_empty()
    reconcile(signal_run)


def test_signal_stop_status_and_free_drip_survive(inputs):
    dates = [date(2024, 1, 2)+timedelta(days=i) for i in range(6)]
    market = rt.prepare_market_data(**inputs(series={"A": [100., 100., 60., 70., 80., 90.]}, dates=dates))
    signals = plan(market, [(0, {"A": 1.}, 2.), (1, {"A": 1.}, 0.), (2, {"A": 1.}, 1.)])
    stopped = simulate(market, signals)
    assert stopped.status == "stopped"
    assert stopped.signal_audit["status"].to_list() == ["processed", "blocked_at_stop", "not_reached"]
    with pytest.raises(ValueError, match="stopped"):
        rt.performance(stopped, periods_per_year=252, risk_free_annual_effective=0., minimum_acceptable_return_annual_effective=0.)
    reconcile(stopped)
    dividend = rt.prepare_market_data(**inputs(series={"A": [100., 100., 90., 90., 90., 90.]}, dates=dates,
        dividends=[("d", "A", dates[2], dates[2], 10.)]))
    result = simulate(dividend, plan(dividend, [(0, {"A": 1.}, 1.)]),
        costs=rt.TradeCosts(commission_bps=100., half_spread_bps=0., impact_bps=0.),
        drip=rt.DividendReinvestment(execution="first_close_on_or_after_payment", funding="before_debt_repayment",
            scheduled_collision="rebalance_only", terminal_action="hold_cash"))
    assert result.trades["trade_cost"][0] > 0 and result.trades["trade_cost"][1] == 0
    reconcile(result)


def test_milestone3_synthetic_example_pipeline():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("chronological_research", Path(__file__).resolve().parents[1]/"examples/chronological_research.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    outputs = example.run_example(train_sessions=50, validation_sessions=40, test_sessions=50, initial_capital=250_000.)
    result = outputs["result"].require_complete()
    reconcile(result)
    assert outputs["evaluation"].metadata["status"] == "final_test"
    assert outputs["evaluation"].audit["event"].to_list() == ["validation_selection", "test_evaluation"]
    assert result.metadata["signals"]["selected_candidate"] == outputs["selection"].selected_candidate
    assert (result.signal_audit["execution_at"] > result.signal_audit["decision_at"]).all()
    assert outputs["report"].metadata["actual_end_session"] == result.metadata["end_session"]
    assert outputs["split"].transforms["n_obs"][0] == outputs["split"].train.height
