"""Offline SOFR-interface example using clearly synthetic rates and stock prices.

Replace the two rate input tables with permitted saved historical observations
and their verified publication calendar. The simulator never downloads data.
"""
from datetime import date, datetime, timezone
from pathlib import Path
import json

import polars as pl
import research_toolkit as rt


ROOT = Path(__file__).resolve().parents[1]


def run_example():
    # 3.6%, 7.2%, 10.8% are synthetic annual rates chosen for hand calculations.
    # Actual source rates expressed as percentages must be divided by 100 first.
    rates = pl.DataFrame({"observation_date": [date(2024, 1, d) for d in (4, 5, 8)],
                          "sofr": [.036, .072, .108]})
    publications = pl.DataFrame({"observation_date": rates["observation_date"],
        "available_at": [datetime(2024, 1, d, 13, tzinfo=timezone.utc) for d in (5, 8, 9)]},
        schema={"observation_date": pl.Date, "available_at": pl.Datetime("us", "UTC")})
    financing = rt.SOFRFinancing(
        rates=rates, publication_calendar=publications,
        metadata={"source": "self-authored synthetic SOFR-interface example",
            "retrieved_at": "2024-02-01T00:00:00Z", "currency": "USD", "rate_units": "decimal",
            "timezone": "America/New_York", "vintage": "point_in_time", "calendar_complete": True,
            "calendar": "explicit synthetic publication dates", "calendar_version": "1",
            "coverage_start": "2024-01-05", "coverage_end": "2024-01-10"},
        borrowing_spread_bps=36., cash_rate=.02,
        day_count="ACT/360", cash_day_count="ACT/365F",
        rate_timing="known_at_accrual_start", max_rate_age_days=7,
        maintenance_equity_ratio=.25, on_breach="stop", cash_sweep="repay_debt",
    )
    dates = [date(2024, 1, d) for d in (5, 8, 9, 10)]
    market = rt.prepare_market_data(
        prices=pl.DataFrame({"session": dates, "asset": ["A"]*4, "close": [100.]*4}),
        sessions=pl.DataFrame({"session": dates, "close_at": [datetime(2024, 1, d.day, 21, tzinfo=timezone.utc) for d in dates]},
            schema={"session": pl.Date, "close_at": pl.Datetime("us", "UTC")}),
        splits=pl.DataFrame(schema={"action_id": pl.String, "asset": pl.String, "effective_session": pl.Date, "ratio": pl.Float64}),
        dividends=pl.DataFrame(schema={"action_id": pl.String, "asset": pl.String, "ex_session": pl.Date, "pay_date": pl.Date,
            "cash_per_share": pl.Float64}),
        metadata={"source": "self-authored flat-price fixture", "retrieved_at": "2024-02-01T00:00:00Z",
            "currency": "USD", "asset_currencies": {"A": "USD"}, "calendar": "explicit synthetic sessions",
            "calendar_version": "1", "timezone": "America/New_York", "price_basis": "raw", "frequency": "1d",
            "coverage_start": dates[0].isoformat(), "coverage_end": dates[-1].isoformat(),
            "actions_complete": True, "dividend_basis": "post_split_share"})
    return rt.buy_and_hold(market, weights={"A": 1.}, initial_capital=100., entry_session=dates[0], end_session=dates[-1],
        policy=rt.BuyHoldPolicy(execution="entry_close", sizing="post_cost_equity", fractional_shares=True,
            initial_gross_leverage=2., terminal_action="mark_only"),
        costs=rt.TradeCosts(commission_bps=0., half_spread_bps=0., impact_bps=0.), financing=financing).require_complete()


if __name__ == "__main__":
    result = run_example()
    print(result.financing_accruals.select("date", "observation_date", "sofr", "borrowing_rate", "opening_debt", "borrowing_interest"))
    print(result.daily.select("session", "debt", "equity", "pnl"))
    destination = ROOT / "artifacts/historical_sofr"
    destination.mkdir(parents=True, exist_ok=True)
    result.financing_accruals.write_csv(destination / "financing_accruals.csv")
    result.daily.write_csv(destination / "daily.csv")
    (destination / "run_metadata.json").write_text(json.dumps(result.metadata, indent=2, allow_nan=False)+"\n")
