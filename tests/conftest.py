from datetime import date, datetime, time, timezone
import socket

import polars as pl
import pytest

import research_toolkit as rt


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("unit tests must not access the network")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def inputs():
    def make(series=None, dates=None, splits=(), dividends=(), **metadata):
        series = {"A": [100.0, 110.0, 99.0]} if series is None else series
        dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)] if dates is None else dates
        prices = pl.DataFrame(
            [{"session": d, "asset": a, "close": float(p)}
             for a, prices in series.items() for d, p in zip(dates, prices)],
            schema={"session": pl.Date, "asset": pl.String, "close": pl.Float64},
        )
        sessions = pl.DataFrame({"session": dates, "close_at": [
            datetime.combine(d, time(21), tzinfo=timezone.utc) for d in dates]},
            schema={"session": pl.Date, "close_at": pl.Datetime("us", "UTC")})
        meta = {"source": "synthetic hand-calculated fixture", "retrieved_at": "2024-02-01T00:00:00Z",
                "currency": "USD", "asset_currencies": {a: "USD" for a in series},
                "calendar": "supplied_test_sessions", "calendar_version": "1",
                "timezone": "America/New_York", "price_basis": "raw", "frequency": "1d",
                "coverage_start": min(dates).isoformat(), "coverage_end": max(dates).isoformat(),
                "actions_complete": True, "dividend_basis": "post_split_share", **metadata}
        return dict(prices=prices, sessions=sessions,
                    splits=pl.DataFrame(splits, schema={"action_id": pl.String, "asset": pl.String,
                         "effective_session": pl.Date, "ratio": pl.Float64}, orient="row"),
                    dividends=pl.DataFrame(dividends, schema={"action_id": pl.String, "asset": pl.String,
                         "ex_session": pl.Date, "pay_date": pl.Date, "cash_per_share": pl.Float64}, orient="row"),
                    metadata=meta)
    return make


@pytest.fixture
def market(inputs):
    return rt.prepare_market_data(**inputs())


@pytest.fixture
def policy():
    def make(exposure=1.0):
        return rt.BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity",
                                fractional_shares=True, initial_gross_leverage=exposure,
                                terminal_action="mark_only")
    return make


@pytest.fixture
def run(policy):
    def simulate(market, **kwargs):
        defaults = dict(weights=rt.equal_weights(market.prices["asset"].unique()),
                        initial_capital=100.0, entry_session=market.sessions["session"][0],
                        end_session=market.sessions["session"][-1], policy=policy(),
                        costs=rt.TradeCosts(commission_bps=0.0, half_spread_bps=0.0, impact_bps=0.0),
                        cash_rate=0.0, cash_day_count="ACT/365F")
        return rt.buy_and_hold(market, **(defaults | kwargs))
    return simulate
