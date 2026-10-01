"""Payment-funded purchases, with independent cash/position accounting oracles."""
from datetime import date

import polars as pl
import pytest

import research_toolkit as rt
from research_toolkit._rebalancing import TARGET_SCHEMA
from test_rebalancing import reconcile as reconcile_ledger


def reconcile(result):
    reconcile_ledger(result)
    for row in result.dividend_reinvestments.iter_rows(named=True):
        payment = result.events.filter((pl.col("action_id") == row["action_id"]) &
                                       (pl.col("type") == "dividend_payment"))
        assert payment.height == 1
        assert payment["date"][0] == row["pay_date"]
        assert payment["cash_delta"][0] == pytest.approx(row["paid_amount"])
        assert row["paid_amount"] == pytest.approx(sum(row[key] for key in (
            "debt_repaid", "cash_released", "signed_notional", "trade_cost")))
        assert all(row[key] >= 0. for key in (
            "debt_repaid", "cash_released", "signed_notional", "trade_cost"))
        if row["trade_id"] is not None:
            trade = result.trades.filter(pl.col("trade_id") == row["trade_id"]).row(0, named=True)
            assert trade["asset"] == row["asset"] and trade["session"] >= row["pay_date"]
            assert trade["signed_notional"] == row["signed_notional"]
            assert trade["trade_cost"] == row["trade_cost"]


def drip(funding="after_debt_repayment", **kwargs):
    return rt.DividendReinvestment(**(dict(
        execution="first_close_on_or_after_payment", funding=funding,
        scheduled_collision="rebalance_only", terminal_action="hold_cash") | kwargs))


def finance(**kwargs):
    return rt.Financing(**(dict(cash_rate=0., borrowing_rate=0., day_count="ACT/365F",
        maintenance_equity_ratio=.25, on_breach="stop", cash_sweep="repay_debt") | kwargs))


def report(result, **kwargs):
    return rt.performance(result, periods_per_year=252, risk_free_annual_effective=0.,
        minimum_acceptable_return_annual_effective=0., **kwargs)


def reinvestment_trades(result):
    return result.trades.filter(pl.col("execution") == "dividend_reinvestment_close")


def test_pay_date_costs_and_future_pnl(inputs, run):
    dates = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    market = rt.prepare_market_data(**inputs(series={"A": [100., 100., 100., 110.]}, dates=dates,
        dividends=[("div", "A", dates[1], dates[2], 10.)]))
    result = run(market, initial_capital=101., dividend_reinvestment=drip(),
        costs=rt.TradeCosts(commission_bps=50., half_spread_bps=30., impact_bps=20.))
    # $101 buys one $100 share with a $1 fee. The $10 payment buys
    # 10/101 shares at $100, inclusive of a 1% fee. No ex-date purchase.
    assert result.positions["quantity"].to_list() == pytest.approx([1., 1., 111/101, 111/101])
    assert result.daily["pnl"].to_list() == pytest.approx([9., -10/101, 1110/101])
    assert result.daily["equity"][-1] == pytest.approx(12210/101)
    trade = reinvestment_trades(result).row(0, named=True)
    assert trade["session"] == dates[2]
    assert trade["signed_notional"] == pytest.approx(1000/101)
    assert trade["trade_cost"] == pytest.approx(10/101)
    audit = result.dividend_reinvestments.row(0, named=True)
    assert audit["trade_id"] == trade["trade_id"] and audit["paid_amount"] == 10.
    assert result.events.filter(pl.col("trade_id") == trade["trade_id"])["action_id"].unique().to_list() == ["div"]
    assert result.turnover["phase"].to_list() == ["entry", "dividend_reinvestment"]
    assert result.turnover["turnover"][-1] == pytest.approx((1000/101)/110)
    assert "pre_reinvestment" in report(result).equity["phase"].to_list()
    reconcile(result)


def test_weekend_payment_split_and_ex_close_entitlement(inputs, run):
    dates = [date(2024, 1, d) for d in (4, 5, 8, 9)]
    market = rt.prepare_market_data(**inputs(series={"A": [100., 100., 50., 55.]}, dates=dates,
        splits=[("split", "A", dates[2], 2.)],
        dividends=[("weekend", "A", dates[1], date(2024, 1, 6), 10.),
                   ("monday", "A", dates[2], dates[2], 1.)]))
    result = run(market, dividend_reinvestment=drip())
    assert reinvestment_trades(result)["session"].to_list() == [dates[2], dates[2]]
    assert result.dividend_reinvestments["paid_amount"].to_list() == [10., 2.]
    assert result.positions["quantity"].to_list() == pytest.approx([1., 1., 2.24, 2.24])
    assert result.daily["equity"][-1] == pytest.approx(123.2)
    assert result.events.filter(pl.col("type") == "dividend_payment")["date"].to_list() == [date(2024, 1, 6), dates[2]]
    assert result.daily.height == 3  # No fabricated weekend observations.
    reconcile(result)


@pytest.mark.parametrize("funding,debt,quantity,ending", [
    ("after_debt_repayment", 80., 2., 140.),
    ("before_debt_repayment", 100., 2.2, 142.),
])
def test_explicit_debt_priority(inputs, run, policy, funding, debt, quantity, ending):
    dates = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    market = rt.prepare_market_data(**inputs(series={"A": [100., 100., 100., 110.]}, dates=dates,
        dividends=[("div", "A", dates[1], dates[2], 10.)]))
    result = run(market, policy=policy(2.), financing=finance(), cash_rate=None, cash_day_count=None,
        dividend_reinvestment=drip(funding))
    assert result.daily["debt"][-1] == debt
    assert result.positions["quantity"][-1] == pytest.approx(quantity)
    assert result.daily["equity"][-1] == pytest.approx(ending)
    assert result.events.filter(pl.col("type") == "borrowing").height == 1
    assert result.dividend_reinvestments["status"][0] == (
        "debt_repaid" if funding == "after_debt_repayment" else "reinvested")
    report(result)
    reconcile(result)


def test_partial_debt_repayment_multiple_assets_pro_rata(inputs, run, policy):
    dates = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    market = rt.prepare_market_data(**inputs(series={"A": [100.]*4, "B": [50.]*4}, dates=dates,
        dividends=[("a", "A", dates[1], dates[2], 10.), ("b", "B", dates[1], dates[2], 10.)]))
    result = run(market, policy=policy(1.1), financing=finance(), cash_rate=None, cash_day_count=None,
        dividend_reinvestment=drip())
    # Entry holds .55 A and 1.1 B with $10 debt; payments 5.5 + 11
    # leave $6.5 after repayment, allocated 1/3 and 2/3 to the payers.
    audit = result.dividend_reinvestments.sort("asset")
    assert audit["debt_repaid"].to_list() == pytest.approx([10/3, 20/3])
    assert audit["signed_notional"].to_list() == pytest.approx([6.5/3, 13/3])
    assert result.daily["equity"][-1] == pytest.approx(116.5)
    assert result.daily["debt"][-1] == 0.
    reconcile(result)


def test_reserved_weekend_cash_interest_is_not_reinvested(inputs, run, policy):
    dates = [date(2024, 1, d) for d in (4, 5, 8, 9)]
    market = rt.prepare_market_data(**inputs(series={"A": [100.]*4}, dates=dates,
        dividends=[("div", "A", dates[1], date(2024, 1, 6), 10.)]))
    result = run(market, policy=policy(2.), financing=finance(cash_rate=.365),
        cash_rate=None, cash_day_count=None, dividend_reinvestment=drip("before_debt_repayment"))
    trade = reinvestment_trades(result).row(0, named=True)
    assert trade["signed_notional"] == 20.
    assert trade["session"] == dates[2]
    # Saturday payment is held through Sunday/Monday accrual. Interest sweeps
    # to debt each date; the $20 principal alone buys .2 shares on Monday.
    assert result.daily["debt"][-1] == pytest.approx(99.96)
    assert result.positions["quantity"][-1] == pytest.approx(2.2)
    reconcile(result)


def scheduled_run(market, dates, baskets, *, funding="after_debt_repayment"):
    targets = pl.DataFrame([(dates[i-1], dates[i], a, float(w), 1.)
        for i, weights in baskets for a, w in weights.items()], schema=TARGET_SCHEMA, orient="row")
    return rt.scheduled_rebalance(market, targets=targets, initial_capital=100.,
        entry_session=dates[1], end_session=dates[-1],
        policy=rt.RebalancePolicy(execution="scheduled_close", sizing="post_cost_equity", fractional_shares=True,
            terminal_action="mark_only", non_session="raise", receivable_policy="reserve", max_asset_weight=1.),
        costs=rt.TradeCosts(commission_bps=0., half_spread_bps=0., impact_bps=0.), financing=finance(),
        dividend_reinvestment=drip(funding))


@pytest.mark.parametrize("funding", ["after_debt_repayment", "before_debt_repayment"])
def test_rebalance_collision_uses_cash_without_drip_round_trip(inputs, funding):
    dates = [date(2024, 1, d) for d in (2, 3, 4, 5, 8)]
    market = rt.prepare_market_data(**inputs(series={"A": [100., 100., 90., 90., 90.], "B": [100.]*5}, dates=dates,
        dividends=[("div", "A", dates[2], dates[3], 10.)]))
    result = scheduled_run(market, dates, [(1, {"A": 1., "B": 0.}), (3, {"A": 0., "B": 1.})], funding=funding)
    assert reinvestment_trades(result).is_empty()
    assert result.trades["signed_notional"].to_list() == pytest.approx([100., -90., 100.])
    assert result.dividend_reinvestments["status"][0] == "scheduled_rebalance"
    assert result.dividend_reinvestments["cash_released"][0] == 10.
    assert result.daily["equity"][-1] == pytest.approx(100.)
    report(result)
    reconcile(result)


def test_standing_drip_can_reopen_previously_sold_asset(inputs):
    dates = [date(2024, 1, d) for d in (2, 3, 4, 5, 8)]
    market = rt.prepare_market_data(**inputs(series={"A": [100., 100., 90., 90., 90.], "B": [100.]*5}, dates=dates,
        dividends=[("div", "A", dates[2], dates[3], 10.)]))
    result = scheduled_run(market, dates, [(1, {"A": 1., "B": 0.}), (2, {"A": 0., "B": 1.})])
    assert reinvestment_trades(result)["asset"].to_list() == ["A"]
    assert result.positions.filter(pl.col("session") == dates[3])["quantity"].to_list() == pytest.approx([1/9, .9])
    report(result)
    reconcile(result)


@pytest.mark.parametrize("pay_day,status", [(4, "terminal_cash"), (5, None)])
def test_terminal_or_unpaid_dividend_does_not_trade(inputs, run, pay_day, status):
    market = rt.prepare_market_data(**inputs(series={"A": [100., 90., 90.]},
        dividends=[("div", "A", date(2024, 1, 3), date(2024, 1, pay_day), 10.)]))
    result = run(market, dividend_reinvestment=drip())
    assert reinvestment_trades(result).is_empty()
    assert result.daily["equity"][-1] == 100.
    if status:
        assert result.dividend_reinvestments["status"].to_list() == [status]
        assert result.daily["cash"][-1] == 10.
    else:
        assert result.dividend_reinvestments.is_empty()
        assert result.daily["dividend_receivable"][-1] == 10.
    reconcile(result)


def test_terminal_cash_still_repays_debt(inputs, run, policy):
    market = rt.prepare_market_data(**inputs(series={"A": [100., 100., 100.]},
        dividends=[("div", "A", date(2024, 1, 3), date(2024, 1, 4), 10.)]))
    result = run(market, policy=policy(2.), financing=finance(), cash_rate=None, cash_day_count=None,
        dividend_reinvestment=drip("before_debt_repayment"))
    assert result.daily["debt"][-1] == 80.
    assert result.dividend_reinvestments["cash_released"][0] == 20.
    assert reinvestment_trades(result).is_empty()
    reconcile(result)


def test_margin_stop_before_drip_preserves_paid_cash(inputs, run, policy):
    dates = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    market = rt.prepare_market_data(**inputs(series={"A": [100., 100., 55., 100.]}, dates=dates,
        dividends=[("div", "A", dates[1], dates[2], 1.)]))
    result = run(market, policy=policy(2.), financing=finance(), cash_rate=None, cash_day_count=None,
        dividend_reinvestment=drip("before_debt_repayment"))
    assert result.status == "stopped" and result.stop_session == dates[2]
    assert reinvestment_trades(result).is_empty()
    assert result.dividend_reinvestments["status"][0] == "stopped_before_trade"
    assert result.daily["cash"][-1] == 2.
    with pytest.raises(ValueError, match="stopped"):
        report(result)
    assert report(result, allow_partial=True).metadata["actual_end_session"] == dates[2].isoformat()
    reconcile(result)


def test_margin_checked_after_reinvestment(inputs, run, policy):
    dates = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    market = rt.prepare_market_data(**inputs(series={"A": [100., 100., 98., 100.]}, dates=dates,
        dividends=[("div", "A", dates[1], dates[2], 1.)]))
    # Before purchase: 98 / 196 = .5 (compliant); afterward 98 / 198 < .5.
    result = run(market, policy=policy(2.), financing=finance(maintenance_equity_ratio=.5),
        cash_rate=None, cash_day_count=None, dividend_reinvestment=drip("before_debt_repayment"))
    assert reinvestment_trades(result).height == 1
    assert result.status == "stopped" and result.stop_session == dates[2]
    report(result, allow_partial=True)
    reconcile(result)


def test_future_prices_cannot_change_prior_purchase(inputs, run):
    dates = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    def simulate(final_price):
        market = rt.prepare_market_data(**inputs(series={"A": [100., 90., 90., final_price]}, dates=dates,
            dividends=[("div", "A", dates[1], dates[2], 10.)]))
        return run(market, dividend_reinvestment=drip())
    a, b = simulate(100.), simulate(500.)
    assert a.trades.equals(b.trades)
    assert a.daily.head(2).equals(b.daily.head(2))


def test_no_dividends_or_all_cash_preserves_existing_results(market, run, policy):
    for exposure in (0., 1.):
        a, b = run(market, policy=policy(exposure)), run(market, policy=policy(exposure), dividend_reinvestment=drip())
        for name in ("daily", "positions", "trades", "costs", "events", "valuations", "turnover"):
            assert getattr(a, name).equals(getattr(b, name))
        assert a.dividend_reinvestments.is_empty() and b.dividend_reinvestments.is_empty()


def test_reinvested_shares_earn_only_later_dividends(inputs, run):
    dates = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    market = rt.prepare_market_data(**inputs(series={"A": [100.]*4}, dates=dates,
        dividends=[("first", "A", dates[1], dates[1], 10.), ("second", "A", dates[2], dates[2], 10.)]))
    result = run(market, dividend_reinvestment=drip())
    assert result.dividend_reinvestments["paid_amount"].to_list() == pytest.approx([10., 11.])
    assert result.positions["quantity"][-1] == pytest.approx(1.21)
    reconcile(result)


@pytest.mark.parametrize("field,value", [("execution", "ex_close"), ("funding", "borrow"),
    ("scheduled_collision", "both"), ("terminal_action", "buy")])
def test_invalid_policy_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        drip(**{field: value})


def test_bool_is_not_a_policy(market, run):
    with pytest.raises(ValueError, match="DividendReinvestment"):
        run(market, dividend_reinvestment=True)


def test_five_asset_saved_example_reconciles_and_plots():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "examples/dividend_reinvestment.py"
    spec = importlib.util.spec_from_file_location("dividend_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs, reports, summary = module.run_example()
    assert summary.height == 3
    # Five final-quarter dividends remain unpaid at the requested end.
    assert runs["reinvest_first"].dividend_reinvestments.height == 15
    assert runs["reinvest_first"].receivables.filter(
        pl.col("session") == date(2025, 1, 2)).filter(pl.col("outstanding") > 0).height == 5
    assert runs["reinvest_first"].dividend_reinvestments["status"].unique().to_list() == ["reinvested"]
    assert runs["cash_sweep"].daily.equals(runs["debt_first"].daily)
    for result in runs.values():
        reconcile(result)
        rt.rolling_risk(result, window=20, periods_per_year=252, risk_free_annual_effective=.02)
    # A second scenario verifies reinvestment without debt or cash reservations.
    unlevered, _, _ = module.run_example(gross_leverage=1.)
    assert unlevered["reinvest_first"].daily.equals(unlevered["debt_first"].daily)
    reconcile(unlevered["reinvest_first"])
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = rt.plots.turnover(runs["reinvest_first"])
    assert "Dividend reinvestment" in ax.get_legend_handles_labels()[1]
    fig.canvas.draw()
    plt.close(fig)
    for plot in (rt.plots.equity, rt.plots.drawdown, rt.plots.allocation, rt.plots.attribution):
        fig, _ = plot(reports["reinvest_first"])
        fig.canvas.draw()
        plt.close(fig)
