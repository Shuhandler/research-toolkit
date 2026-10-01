from datetime import date
from decimal import Decimal
import json
import math
import subprocess
import sys

import polars as pl
import pytest

import research_toolkit as rt


def test_dollar_pnl_returns_and_no_exit_trade(market, run):
    result = run(market)
    assert result.daily["pnl"].to_list() == pytest.approx([10, -11])
    assert result.daily["simple_return"].to_list() == pytest.approx([.1, -.1])
    assert result.daily["equity"].to_list() == pytest.approx([110, 99])
    assert result.daily["cumulative_simple_return"][-1] == pytest.approx(0)
    assert result.daily["compounded_return"][-1] == pytest.approx(-.01)
    assert result.trades.height == 1
    assert result.costs.is_empty()
    assert result.valuations["phase"].to_list() == ["pre_entry", "post_entry", "close", "close"]
    assert result.daily["includes_entry_costs"].to_list() == [True, False]
    assert result.daily["drawdown"].to_list() == pytest.approx([0, -.1])
    assert result.metadata["snapshot_id"] == market.snapshot_id
    assert result.status == "complete"
    json.dumps(result.metadata, allow_nan=False)


def test_entry_cost_denominator_and_component_records(inputs, run):
    market = rt.prepare_market_data(**inputs({"A": [100, 110, 110]}))
    result = run(market, costs=rt.TradeCosts(commission_bps=50., half_spread_bps=30., impact_bps=20.))
    notional = Decimal(100) / Decimal('1.01')
    expected_equity = float(notional * Decimal('1.1'))
    assert result.trades["signed_notional"][0] == pytest.approx(float(notional))
    assert result.costs["amount"].sum() == pytest.approx(float(notional * Decimal('.01')))
    assert result.daily["equity"][0] == pytest.approx(expected_equity)
    assert result.daily["pnl"][0] == pytest.approx(expected_equity - 100)
    assert result.daily["opening_equity"][0] == 100
    assert result.daily["simple_return"][0] == pytest.approx(expected_equity / 100 - 1)
    assert result.daily["pnl"][1] == 0
    assert set(result.costs["trade_id"]) == set(result.trades["trade_id"])
    assert result.events.filter(pl.col("type") == "trade")["cash_delta"][0] == pytest.approx(-float(notional))
    assert result.daily["cash"].to_list() == [0, 0]
    assert result.daily["debt"].to_list() == [0, 0]


def test_flat_prices_show_cost_loss_and_no_recurring_fee(inputs, run):
    result = run(rt.prepare_market_data(**inputs({"A": [100, 100, 100]})),
                 costs=rt.TradeCosts(commission_bps=100., half_spread_bps=0., impact_bps=0.))
    assert result.daily["simple_return"].to_list() == pytest.approx([-1/101, 0])
    assert result.daily["drawdown"].to_list() == pytest.approx([-1/101, -1/101])
    assert result.costs.height == 1


def test_split_and_dividend_combined_then_payment(inputs, run):
    ex, pay = date(2024, 1, 3), date(2024, 1, 4)
    market = rt.prepare_market_data(**inputs({"A": [100, 49, 49]},
        splits=[("S", "A", ex, 2.)], dividends=[("D", "A", ex, pay, 1.)]))
    result = run(market, initial_capital=1000.)
    assert result.positions["quantity"].to_list() == [10, 20, 20]
    assert result.daily["equity"].to_list() == [1000, 1000]
    assert result.daily["dividend_receivable"].to_list() == [20, 0]
    assert result.daily["cash"].to_list() == [0, 20]
    assert result.daily["pnl"].to_list() == [0, 0]
    assert result.receivables["entitled_quantity"].to_list() == [20, 20]
    assert result.events["type"].to_list() == ["trade", "split", "dividend_accrual", "dividend_payment"]
    assert result.trades.height == 1 and result.costs.is_empty()


def test_pure_split_neutrality(inputs, run):
    market = rt.prepare_market_data(**inputs({"A": [100, 50, 50]},
        splits=[("S", "A", date(2024, 1, 3), 2.)]))
    result = run(market)
    assert result.daily["pnl"].to_list() == [0, 0]
    assert result.positions["quantity"].to_list() == [1, 2, 2]


def test_ex_date_entry_and_pre_entry_actions_do_not_create_entitlements(inputs, run):
    market = rt.prepare_market_data(**inputs({"A": [100, 98, 98]}, dividends=[
        ("D", "A", date(2024, 1, 3), date(2024, 1, 4), 2.),
        ("OLD", "A", date(2024, 1, 2), date(2024, 1, 4), 2.)]))
    result = run(market, entry_session=date(2024, 1, 3))
    assert result.receivables.is_empty()
    assert result.daily["equity"].to_list() == [100]
    assert result.daily["cash"].to_list() == [0]


def test_unpaid_receivable_survives_later_split(inputs, run):
    market = rt.prepare_market_data(**inputs({"A": [100, 98, 49]},
        splits=[("S", "A", date(2024, 1, 4), 2.)],
        dividends=[("D", "A", date(2024, 1, 3), date(2024, 2, 1), 2.)]))
    result = run(market)
    assert result.daily["dividend_receivable"].to_list() == [2, 2]
    assert result.daily["equity"].to_list() == [100, 100]
    assert result.daily["cash"].to_list() == [0, 0]


def test_weekend_payment_and_calendar_interest(inputs, run):
    dates = [date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)]
    market = rt.prepare_market_data(**inputs({"A": [100, 98, 98]}, dates=dates,
        dividends=[("D", "A", dates[1], date(2024, 1, 6), 2.)]))
    result = run(market, cash_rate=.365)
    # Saturday payment earns interest on Sunday and Monday, not before payment.
    assert result.daily["cash"][-1] == pytest.approx(2.004002)
    assert result.daily["equity"][-1] == pytest.approx(100.004002)
    payment = result.events.filter(pl.col("type") == "dividend_payment")
    assert payment["date"][0] == date(2024, 1, 6)
    assert payment["time"][0] is None
    assert result.daily.height == 2  # No invented weekend market-return rows.


def test_all_cash_zero_trades_and_interest(inputs, run, policy):
    dates = [date(2024, 1, 5), date(2024, 1, 8)]
    market = rt.prepare_market_data(**inputs({"A": [100, 500]}, dates=dates))
    zero = run(market, policy=policy(0), cash_rate=0.)
    assert zero.daily["equity"].to_list() == [100]
    assert zero.trades.is_empty() and zero.costs.is_empty() and zero.events.is_empty()
    assert zero.daily["gross_leverage"][0] == 0
    interest = run(market, policy=policy(0), cash_rate=.365)
    assert interest.daily["equity"][0] == pytest.approx(100.3003001)
    assert interest.events.height == 3


def test_partial_exposure_funds_costs_and_retains_cash(inputs, run, policy):
    market = rt.prepare_market_data(**inputs({"A": [100, 100, 100]}))
    result = run(market, policy=policy(.5),
                 costs=rt.TradeCosts(commission_bps=100., half_spread_bps=0., impact_bps=0.))
    assert result.trades["signed_notional"][0] == pytest.approx(50/1.005)
    assert result.daily["cash"][0] == pytest.approx(50/1.005)
    assert result.daily["gross_leverage"][0] == pytest.approx(.5)
    assert result.daily["debt"].sum() == 0


def test_weight_drift_is_not_rebalancing(inputs, run):
    market = rt.prepare_market_data(**inputs({"A": [100, 200, 100], "B": [100, 100, 100]}))
    result = run(market)
    assert result.daily["equity"].to_list() == [150, 100]
    middle = result.positions.filter(pl.col("session") == date(2024, 1, 3))
    assert middle["weight"].to_list() == pytest.approx([2/3, 1/3])
    assert result.trades.height == 2
    different = run(market, weights={"A": .25, "B": .75})
    assert different.daily["equity"].to_list() == [125, 100]


def test_per_asset_costs_and_zero_weight_assets(inputs, run):
    market = rt.prepare_market_data(**inputs({"A": [100]*3, "B": [100]*3, "C": [100]*3}))
    costs = rt.TradeCosts(commission_bps={"A": 100., "B": 200., "C": 999.},
                         half_spread_bps=0., impact_bps=0.)
    result = run(market, weights={"A": .5, "B": .5, "C": 0.}, costs=costs)
    assert result.daily["equity"][0] == pytest.approx(100/1.015)
    assert result.trades["asset"].to_list() == ["A", "B"]
    assert result.positions.filter(pl.col("asset") == "C")["quantity"].sum() == 0


def test_independent_event_reconstruction_and_attribution(inputs, run):
    market = rt.prepare_market_data(**inputs({"A": [100, 98, 98], "B": [50, 55, 55]},
        dividends=[("D", "A", date(2024, 1, 3), date(2024, 1, 4), 2.)]))
    result = run(market, initial_capital=100_000_000., cash_rate=.05,
                 costs=rt.TradeCosts(commission_bps=5., half_spread_bps=2., impact_bps=1.))
    for row in result.daily.iter_rows(named=True):
        events = result.events.filter(pl.col("date") <= row["session"])
        reconstructed_cash = 100_000_000 + math.fsum(events["cash_delta"])
        reconstructed_receivable = math.fsum(events["receivable_delta"])
        positions = result.positions.filter(pl.col("session") == row["session"])
        total = math.fsum(positions["market_value"]) + reconstructed_cash + reconstructed_receivable
        assert abs(total - row["equity"]) < .01
        attributed = result.attribution.filter(pl.col("session") == row["session"])["pnl"].sum()
        assert attributed == pytest.approx(row["pnl"], abs=1e-7)
    assert (result.diagnostics["residual"].abs() <= result.diagnostics["tolerance"]).all()
    assert result.daily["cumulative_pnl"][-1] == pytest.approx(result.daily["equity"][-1] - 100_000_000)


def test_scaling_determinism_and_inputs_unchanged(inputs, run):
    data = inputs({f"S{i}": [100, 101+i, 99+i] for i in range(7)})
    market = rt.prepare_market_data(**data)
    before = market.prices.clone()
    first, second = run(market), run(market)
    scaled = run(market, initial_capital=100_000_000.)
    assert first.daily.equals(second.daily)
    assert first.events.equals(second.events)
    assert first.metadata == second.metadata
    assert market.prices.equals(before)
    assert scaled.daily["simple_return"].to_list() == pytest.approx(first.daily["simple_return"].to_list())
    assert scaled.daily["equity"].to_list() == pytest.approx([v*1_000_000 for v in first.daily["equity"]])


@pytest.mark.parametrize("weights", [{"A": -1.}, {"A": .5}, {"UNKNOWN": 1.}, {"A": float("nan")}, {}])
def test_invalid_weights(market, run, weights):
    with pytest.raises(ValueError):
        run(market, weights=weights)


@pytest.mark.parametrize("kw", [
    {"initial_capital": 0}, {"initial_capital": float("inf")}, {"cash_rate": -.01},
    {"cash_day_count": "ACT/360"}, {"entry_session": date(2024, 1, 4)},
    {"end_session": date(2024, 1, 6)}, {"entry_session": "2024-01-02"},
])
def test_invalid_run_config(market, run, kw):
    with pytest.raises(ValueError):
        run(market, **kw)


def test_reject_unimplemented_economics(inputs, run, policy):
    with pytest.raises(ValueError, match="explicit Financing"):
        run(rt.prepare_market_data(**inputs()), policy=policy(2))
    for basis in ["split_adjusted", "total_return_adjusted"]:
        with pytest.raises(ValueError, match="raw execution"):
            run(rt.prepare_market_data(**inputs(price_basis=basis)))
    with pytest.raises(ValueError, match="entry_close"):
        rt.BuyHoldPolicy(execution="next_open", sizing="post_cost_equity",
                          fractional_shares=True, initial_gross_leverage=1, terminal_action="mark_only")


def test_cost_inputs_must_be_explicit_and_valid(market, run):
    for commission in [-1., float("inf"), {"OTHER": 1.}]:
        with pytest.raises(ValueError):
            run(market, costs=rt.TradeCosts(commission_bps=commission, half_spread_bps=0., impact_bps=0.))


def test_import_has_no_optional_stack():
    subprocess.run([sys.executable, "-c", "import sys; import research_toolkit; "
                    "assert not any(x in sys.modules for x in "
                    "('torch', 'sklearn', 'matplotlib', 'altair', 'requests', 'pandas'))"], check=True)


def test_unrepresentable_positive_wealth_fails_instead_of_false_bankruptcy(inputs, run):
    market = rt.prepare_market_data(**inputs({"A": [100, 1e-300, 1e-300]}))
    with pytest.raises(ValueError, match="representable positive wealth"):
        run(market)


def test_input_row_order_does_not_change_ledger(inputs, run):
    data = inputs({"B": [50, 55, 52], "A": [100, 98, 98]},
        dividends=[("D", "A", date(2024, 1, 3), date(2024, 1, 4), 2.)])
    one = run(rt.prepare_market_data(**data))
    shuffled = {name: value.reverse() if isinstance(value, pl.DataFrame) else value
                for name, value in data.items()}
    two = run(rt.prepare_market_data(**shuffled))
    assert one.daily.equals(two.daily)
    assert one.events.equals(two.events)
    assert one.metadata == two.metadata
