from dataclasses import replace
from datetime import date
from decimal import Decimal
import json
import math

import polars as pl
import pytest

import research_toolkit as rt


@pytest.fixture
def financing():
    def make(**kwargs):
        return rt.Financing(**(dict(cash_rate=0., borrowing_rate=0., day_count="ACT/365F",
                                   maintenance_equity_ratio=.25, on_breach="stop",
                                   cash_sweep="repay_debt") | kwargs))
    return make


@pytest.fixture
def financed(run, policy, financing):
    def simulate(market, **kwargs):
        return run(market, **(dict(policy=policy(2), financing=financing(),
                                 cash_rate=None, cash_day_count=None) | kwargs))
    return simulate


def test_leverage_and_return_denominators_drift(inputs, financed):
    market = rt.prepare_market_data(**inputs({"A": [100, 125, 100]}))
    result = financed(market)
    assert result.daily["equity"].to_list() == [150, 100]
    assert result.daily["pnl"].to_list() == [50, -50]
    assert result.daily["simple_return"].to_list() == pytest.approx([.5, -1/3])
    assert result.daily["gross_leverage"].to_list() == pytest.approx([250/150, 2])
    assert result.positions["quantity"].to_list() == [2, 2, 2]
    assert result.daily["debt"].to_list() == [100, 100]
    assert result.trades.height == 1 and result.costs.is_empty()
    funding = result.events.row(0, named=True)
    assert funding["type"] == "borrowing"
    assert funding["cash_delta"] == funding["debt_delta"] == 100
    assert result.daily["compounded_return"][-1] == pytest.approx(0)
    assert result.require_complete() is result
    assert result.stop_reason is result.stop_session is result.stop_time is None


def test_cost_funded_leveraged_entry_against_decimal_oracle(inputs, financed):
    market = rt.prepare_market_data(**inputs({"A": [100, 100, 100]}))
    result = financed(market, costs=rt.TradeCosts(commission_bps=100., half_spread_bps=0., impact_bps=0.))
    gross = Decimal(200) / Decimal('1.02')
    cost = gross * Decimal('.01')
    equity = Decimal(100) - cost
    assert result.trades["signed_notional"][0] == pytest.approx(float(gross))
    assert result.costs["amount"].sum() == pytest.approx(float(cost))
    assert result.daily["equity"][0] == pytest.approx(float(equity))
    assert result.daily["debt"][0] == pytest.approx(float(equity))
    assert result.daily["simple_return"].to_list() == pytest.approx([-float(cost)/100, 0])
    assert result.daily["cash"].to_list() == [0, 0]
    assert result.daily["gross_leverage"].to_list() == pytest.approx([2, 2])
    # The cash ledger is funded before it pays for any shares or fees.
    running_cash = 100.
    for event in result.events.iter_rows(named=True):
        running_cash += event["cash_delta"]
        assert running_cash >= -1e-10


def test_weekend_interest_costs_are_not_trades(inputs, financed, financing):
    dates = [date(2024, 1, 5), date(2024, 1, 8)]
    result = financed(rt.prepare_market_data(**inputs({"A": [100, 100]}, dates=dates)),
                      financing=financing(borrowing_rate=.365))
    assert result.daily["debt"][0] == pytest.approx(100.3003001)
    assert result.daily["equity"][0] == pytest.approx(99.6996999)
    assert result.daily["pnl"][0] == pytest.approx(-.3003001)
    assert result.costs["amount"].to_list() == pytest.approx([.1, .1001, .1002001])
    assert result.costs["trade_id"].null_count() == 3
    assert result.costs["asset"].null_count() == 3
    assert result.costs["date"].to_list() == [date(2024, 1, 6), date(2024, 1, 7), dates[1]]
    assert result.costs["time"].null_count() == 3
    assert result.trades.height == result.daily.height == 1
    assert result.events.filter(pl.col("type") == "borrowing_interest")["cash_delta"].sum() == 0


def test_weekend_payment_repays_only_after_interest(inputs, financed, financing):
    dates = [date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)]
    market = rt.prepare_market_data(**inputs({"A": [100, 98, 98]}, dates=dates,
        dividends=[("D", "A", dates[1], date(2024, 1, 6), 2.)]))
    result = financed(market, financing=financing(borrowing_rate=.365))
    # Debt: Thu 100; Fri 100.1; Sat 100.2001 then $4 repayment;
    # Sun 96.2963001; Mon 96.3925964001. Receivables cannot repay on Friday.
    assert result.daily["debt"].to_list() == pytest.approx([100.1, 96.3925964001])
    assert result.daily["dividend_receivable"].to_list() == [4, 0]
    assert result.daily["cash"].to_list() == [0, 0]
    assert result.daily["equity"][-1] == pytest.approx(99.6074035999)
    saturday = result.events.filter(pl.col("date") == date(2024, 1, 6))
    assert saturday["type"].to_list() == ["borrowing_interest", "dividend_payment", "debt_repayment"]
    assert saturday["debt_delta"][-1] == saturday["cash_delta"][-1] == -4


def test_repayment_leaves_excess_cash_and_no_further_borrowing(inputs, financed, financing):
    market = rt.prepare_market_data(**inputs({"A": [100, 40, 40]},
        dividends=[("D", "A", date(2024, 1, 3), date(2024, 1, 3), 60.)]))
    result = financed(market, financing=financing(cash_rate=.365, borrowing_rate=.365))
    # Two shares earn $120; first pay $100.1 debt, leaving $19.9 cash.
    assert result.daily["debt"].to_list() == [0, 0]
    assert result.daily["cash"].to_list() == pytest.approx([19.9, 19.9199])
    assert result.daily["equity"].to_list() == pytest.approx([99.9, 99.9199])
    assert result.costs.height == 1
    assert result.trades.height == 1
    assert result.events.filter(pl.col("type") == "debt_repayment")["debt_delta"][0] == pytest.approx(-100.1)


def test_split_dividend_and_unpaid_receivable_with_debt(inputs, financed):
    market = rt.prepare_market_data(**inputs({"A": [100, 49, 49]},
        splits=[("S", "A", date(2024, 1, 3), 2.)],
        dividends=[("D", "A", date(2024, 1, 3), date(2024, 2, 1), 1.)]))
    result = financed(market)
    assert result.daily["equity"].to_list() == [100, 100]
    assert result.daily["dividend_receivable"].to_list() == [4, 4]
    assert result.daily["debt"].to_list() == [100, 100]
    assert result.positions["quantity"].to_list() == [2, 4, 4]
    assert result.events.filter(pl.col("type") == "debt_repayment").is_empty()


def test_margin_stop_preserves_failure_close_and_no_future_events(inputs, financed, financing):
    market = rt.prepare_market_data(**inputs({"A": [100, 75, 200]}))
    result = financed(market, financing=financing(maintenance_equity_ratio=.4))
    assert result.status == "stopped"
    assert result.stop_reason == "maintenance_margin_breach"
    assert result.stop_session == date(2024, 1, 3)
    assert result.stop_time == market.sessions["close_at"][1]
    assert result.daily["equity"].to_list() == [50]
    assert result.daily["equity_ratio"].to_list() == pytest.approx([1/3])
    assert result.daily["margin_breached"].to_list() == [True]
    assert result.events["date"].max() == result.stop_session
    assert result.events["type"][-1] == result.stop_reason
    assert result.positions["quantity"].to_list() == [2, 2]
    assert result.trades.height == 1  # No forced liquidation or exit fee.
    assert result.valuations["equity"][-1] == 50
    assert result.metadata["end_session"] == "2024-01-04"
    assert result.metadata["actual_end_session"] == "2024-01-03"
    with pytest.raises(ValueError, match="maintenance_margin_breach"):
        result.require_complete()
    json.dumps(result.metadata, allow_nan=False)


@pytest.mark.parametrize("last_price, expected_equity, expected_return", [(50, 0, -1), (40, -20, -1.2)])
def test_insolvency_records_unclipped_loss_and_undefined_ratios(inputs, financed, last_price, expected_equity, expected_return):
    result = financed(rt.prepare_market_data(**inputs({"A": [100, last_price, 200]})))
    assert result.stop_reason == "nonpositive_equity"
    assert result.daily["equity"].to_list() == [expected_equity]
    assert result.daily["simple_return"].to_list() == pytest.approx([expected_return])
    assert result.daily["compounded_return"].to_list() == pytest.approx([expected_return])
    assert result.daily["log_return"].to_list() == [None]
    assert result.daily["gross_leverage"].to_list() == [None]
    assert result.positions["weight"][-1] is None
    assert result.daily["debt"][0] == 100
    assert result.daily["pnl"][0] == expected_equity - 100
    with pytest.raises(ValueError, match="nonpositive_equity"):
        result.require_complete()


def test_interest_alone_can_trigger_breach_on_final_requested_session(inputs, financed, financing):
    dates = [date(2024, 1, 5), date(2024, 1, 8)]
    market = rt.prepare_market_data(**inputs({"A": [100, 100]}, dates=dates))
    result = financed(market, financing=financing(borrowing_rate=.365, maintenance_equity_ratio=.499))
    assert result.status == "stopped" and result.stop_session == dates[-1]
    assert result.costs.height == 3  # All accrued days through monitored close.
    with pytest.raises(ValueError):
        result.require_complete()


def test_threshold_equality_is_compliant_and_bad_initial_funding_rejected(inputs, financed, financing, policy):
    market = rt.prepare_market_data(**inputs({"A": [100, 100, 100]}))
    assert financed(market, financing=financing(maintenance_equity_ratio=.5)).status == "complete"
    with pytest.raises(ValueError, match="initial leverage violates"):
        financed(market, policy=policy(3), financing=financing(maintenance_equity_ratio=.4))
    with pytest.raises(ValueError, match="borrowing requires"):
        financed(market, financing=financing(maintenance_equity_ratio=None))


@pytest.mark.parametrize("exposure", [0, .5, 1])
def test_unlevered_equivalence_and_cash_only_ratio(inputs, financed, financing, run, policy, exposure):
    market = rt.prepare_market_data(**inputs({"A": [100, 98, 101]},
        dividends=[("D", "A", date(2024, 1, 3), date(2024, 1, 3), 2.)]))
    old = run(market, policy=policy(exposure), cash_rate=.05)
    new = financed(market, policy=policy(exposure), financing=financing(cash_rate=.05, maintenance_equity_ratio=None))
    for name in ("daily", "trades", "positions", "costs", "events", "valuations", "attribution", "receivables"):
        assert getattr(old, name).equals(getattr(new, name)), name
    if exposure == 0:
        assert new.daily["equity_ratio"].null_count() == new.daily.height
        assert new.trades.is_empty()


def test_financed_multi_asset_event_reconstruction_scaling_and_determinism(inputs, financed, financing):
    market = rt.prepare_market_data(**inputs({"A": [100, 49, 51], "B": [50, 52, 53]},
        splits=[("S", "A", date(2024, 1, 3), 2.)],
        dividends=[("D", "A", date(2024, 1, 3), date(2024, 1, 4), 1.)]))
    kwargs = dict(financing=financing(borrowing_rate=.08),
                  costs=rt.TradeCosts(commission_bps=5., half_spread_bps=2., impact_bps=1.))
    small = financed(market, **kwargs)
    capital = 100_000_000.
    large = financed(market, initial_capital=capital, **kwargs)
    repeated = financed(market, initial_capital=capital, **kwargs)
    assert large.daily.equals(repeated.daily) and large.events.equals(repeated.events)
    assert large.daily["simple_return"].to_list() == pytest.approx(small.daily["simple_return"].to_list())
    for row in large.daily.iter_rows(named=True):
        events = large.events.filter(pl.col("date") <= row["session"])
        cash = math.fsum([capital, *events["cash_delta"]])
        debt = math.fsum(events["debt_delta"])
        receivable = math.fsum(events["receivable_delta"])
        pos = large.positions.filter(pl.col("session") == row["session"])
        reconstructed = math.fsum([*pos["market_value"], cash, receivable, -debt])
        assert abs(reconstructed - row["equity"]) < .01
        assert abs(debt - row["debt"]) < .01
        attributed = large.attribution.filter(pl.col("session") == row["session"])["pnl"].sum()
        assert attributed == pytest.approx(row["pnl"], abs=1e-7)
    assert (large.diagnostics["residual"].abs() <= large.diagnostics["tolerance"]).all()
    charges = large.costs.filter(pl.col("component") == "borrowing_interest")["amount"].sum()
    interest_events = large.events.filter(pl.col("type") == "borrowing_interest")["debt_delta"].sum()
    assert charges == pytest.approx(interest_events)


@pytest.mark.parametrize("kwargs", [
    {"cash_rate": -.01}, {"borrowing_rate": float("inf")}, {"borrowing_rate": -.01},
    {"borrowing_rate": True}, {"day_count": "ACT/360"}, {"maintenance_equity_ratio": 0},
    {"maintenance_equity_ratio": 1.1}, {"maintenance_equity_ratio": float("nan")},
    {"on_breach": "liquidate"}, {"cash_sweep": "hold"},
])
def test_invalid_financing_rejected(financing, kwargs):
    with pytest.raises(ValueError):
        financing(**kwargs)


def test_ambiguous_financing_and_missing_explicit_rates_fail(market, run, financing):
    with pytest.raises(ValueError, match="do not mix"):
        run(market, financing=financing())
    with pytest.raises(ValueError, match="explicit Financing object"):
        run(market, financing={})
    with pytest.raises(ValueError, match="cash_rate"):
        run(market, cash_rate=None, cash_day_count=None)
    with pytest.raises(TypeError):
        rt.Financing(cash_rate=0., borrowing_rate=0.)
