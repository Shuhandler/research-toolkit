"""Synthetic walk-through of report diagnostics, factor regression and signal evaluation.

Run: python examples/diagnostics.py
Every number below is generated from a fixed seed for illustration; none is market
data or evidence about any strategy. No downloads, file writes or plots.
"""
from datetime import date, timedelta
import random

import polars as pl
import research_toolkit as rt


def run_example(seed=7, n_days=250):
    rng = random.Random(seed)
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days + 1)]
    intervals = {"period_start": days[:-1], "session": days[1:]}

    # Synthetic factors: a total-return market index and a long/short spread.
    mkt = [rng.gauss(.0004, .01) for _ in range(n_days)]
    smb = [rng.gauss(0., .005) for _ in range(n_days)]
    capital = 1_000_000.
    nav, pnl = capital, []
    for m, s in zip(mkt, smb):
        r = .0002 + .8*m + .3*s + rng.gauss(0., .004)
        pnl.append(nav*r)
        nav += pnl[-1]
    daily = pl.DataFrame({**intervals, "pnl": pnl})
    report = rt.series_performance(daily, initial_capital=capital, periods_per_year=252,
                                   risk_free_annual_effective=.03,
                                   metadata={"source": "synthetic example P&L", "currency": "USD"})

    calendar = rt.performance_diagnostics(report, annualization="calendar_time", days_per_year=365.25)
    trading = rt.performance_diagnostics(report, annualization="trading_periods", periods_per_year=252)

    factors = pl.DataFrame({**intervals, "MKT": mkt, "SMB": smb})
    factor_kw = dict(factor_metadata={"source": "synthetic example factors", "currency": "USD", "frequency": "1d"},
                     periods_per_year=252, model="excess_returns", risk_free="report")
    joint = rt.factor_regression(report, factors, factor_bases={"MKT": "total_return", "SMB": "long_short"},
                                 **factor_kw)
    market_only = rt.factor_regression(report, factors.select("period_start", "session", "MKT"),
                                       factor_bases={"MKT": "total_return"}, **factor_kw)

    # Cross-sectional signals on weekly dates; each outcome starts at the next day's close.
    assets = [f"SYN{i:02d}" for i in range(23)]
    signal_rows, outcome_rows = [], []
    for week in range(20):
        decided = date(2024, 1, 5) + timedelta(weeks=week)
        start, end = decided + timedelta(days=3), decided + timedelta(days=10)
        for asset in assets:
            score = rng.gauss(0., 1.)
            signal_rows.append((decided, asset, score))
            outcome_rows.append((decided, asset, start, end, .002*score + rng.gauss(0., .02)))
    signals = pl.DataFrame(signal_rows, schema={"signal_date": pl.Date, "asset": pl.String, "signal": pl.Float64},
                           orient="row")
    outcomes = pl.DataFrame(outcome_rows, schema={"signal_date": pl.Date, "asset": pl.String,
        "period_start": pl.Date, "period_end": pl.Date, "forward_return": pl.Float64}, orient="row")
    signal = rt.signal_diagnostics(signals, outcomes, timing="after_signal_session", quantiles=5,
        grouping="extremes", ties="asset_order", metadata={"signal_source": "synthetic scores",
        "return_source": "synthetic forward returns", "return_basis": "total_return"})
    return report, calendar, trading, joint, market_only, signal


if __name__ == "__main__":
    pl.Config.set_tbl_rows(20)
    report, calendar, trading, joint, market_only, signal = run_example()
    print("Calendar-time diagnostics (365.25-day years)\n", calendar.values)
    print("Trading-period CAGR (252 per year)\n", trading.values.filter(pl.col("metric") == "cagr"))
    print("Joint excess-return coefficients\n", joint.coefficients)
    print("Standalone single-factor fits\n", joint.standalone.select("factor", "beta", "correlation", "r_squared"))
    print("Joint fit summary\n", joint.summary)
    print("Market-only fit summary\n", market_only.summary)
    print("Per-date rank IC (first rows)\n", signal.ic.head())
    print("Signal summary\n", signal.summary)
    print("Equal-weight quantile means\n", signal.quantile_summary)
