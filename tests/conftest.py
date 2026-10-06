from datetime import date, datetime, time, timedelta, timezone
import io
import json
import socket
import urllib.parse
import urllib.request

import polars as pl
import pytest

import research_toolkit as rt
from research_toolkit import adapters
from research_toolkit._rebalancing import TARGET_SCHEMA
from research_toolkit._research import LABEL_SCHEMA

# Shared fixtures used by several test modules live here, so modules never import fixtures.
YAHOO_RETRIEVED = datetime(2024, 6, 10, 12, tzinfo=timezone.utc)  # Mocked Yahoo retrieval time.


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


class Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *exc): return False


@pytest.fixture
def provider(monkeypatch):
    calls, sleeps = [], []
    script = {}

    def urlopen(request, timeout):
        url = request.full_url
        calls.append((url, timeout, request.get_header("User-agent")))
        symbol = urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[1])
        outcome = script[symbol].pop(0) if isinstance(script[symbol], list) else script[symbol]
        if isinstance(outcome, Exception):
            raise outcome
        return Response(json.dumps(outcome).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("time.sleep", sleeps.append)
    monkeypatch.setattr(adapters, "_now", lambda: YAHOO_RETRIEVED)
    return script, calls, sleeps


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


@pytest.fixture
def research_inputs():
    # Fifteen supplied weekday sessions, not an inferred exchange calendar.
    days = [date(2024, 1, 2)+timedelta(days=i) for i in range(21) if (date(2024, 1, 2)+timedelta(days=i)).weekday() < 5]
    closes = [datetime.combine(d, time(21), timezone.utc) for d in days]
    sessions = pl.DataFrame({"session": days, "close_at": closes}, schema={"session": pl.Date, "close_at": pl.Datetime("us", "UTC")})
    observations = pl.DataFrame([(d, a, t, float(i + offset)) for i, (d, t) in enumerate(zip(days, closes))
        for a, offset in [("A", 1), ("B", 3)]],
        schema={"session": pl.Date, "asset": pl.String, "available_at": pl.Datetime("us", "UTC"), "x": pl.Float64}, orient="row")
    labels = pl.DataFrame([(d, a, days[i+1], closes[i+1], float(i%2)) for i, d in enumerate(days[:13]) for a in ["A", "B"]], schema=LABEL_SCHEMA, orient="row")
    meta = dict(source="synthetic measurements", calendar="supplied weekday fixture", calendar_version="1", timezone="America/New_York",
                frequency="1d", feature_units={"x": "arbitrary_units"})
    return dict(days=days, sessions=sessions, observations=observations, labels=labels, metadata=meta)
