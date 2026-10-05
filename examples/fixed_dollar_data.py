"""Small offline synthetic inputs for fixed_dollar_rebalancing.ipynb.

The calendar and observations are supplied example assumptions, not market history.
"""
from datetime import date, datetime, time, timezone

import polars as pl
import research_toolkit as rt


def synthetic_inputs():
    days = [date(2024, 1, d) for d in (2, 3, 4, 5, 8, 9, 10, 11, 12, 16)]
    series = {
        "LONG_A": [100., 101., 99., 102., 104., 98., 103., 104., 102., 105.],
        "SHORT_B": [100., 98., 101., 99., 97., 105., 100., 102., 101., 100.],
        "HEDGE": [100., 100.5, 99., 101., 102., 98., 101., 102., 100., 103.],
    }
    sessions = pl.DataFrame({"session": days,
        "close_at": [datetime.combine(d, time(21), timezone.utc) for d in days]},
        schema={"session": pl.Date, "close_at": pl.Datetime("us", "UTC")})
    prices = pl.DataFrame([{"session": d, "asset": a, "close": p}
        for a, values in series.items() for d, p in zip(days, values)])
    market = rt.prepare_market_data(prices=prices, sessions=sessions,
        splits=pl.DataFrame(schema={"action_id":pl.String, "asset":pl.String, "effective_session":pl.Date, "ratio":pl.Float64}),
        dividends=pl.DataFrame(schema={"action_id":pl.String, "asset":pl.String, "ex_session":pl.Date,
            "pay_date":pl.Date, "cash_per_share":pl.Float64}),
        metadata={"source":"synthetic example", "retrieved_at":"2024-02-01T00:00:00Z", "currency":"USD",
            "asset_currencies":{a:"USD" for a in series}, "calendar":"supplied_synthetic_sessions", "calendar_version":"1",
            "timezone":"America/New_York", "price_basis":"raw", "frequency":"1d",
            "coverage_start":days[0].isoformat(), "coverage_end":days[-1].isoformat(),
            "actions_complete":True, "dividend_basis":"post_split_share"})
    dollar_volume = pl.DataFrame([{"session":d, "asset":a,
        "dollar_volume":float(30_000_000/(1+i*.2)*(1+j*.4)),
        "available_at":datetime.combine(d,time(21),timezone.utc)}
        for i,d in enumerate(days) for j,a in enumerate(series)],
        schema={"session":pl.Date,"asset":pl.String,"dollar_volume":pl.Float64,"available_at":pl.Datetime("us","UTC")})
    return market, dollar_volume
