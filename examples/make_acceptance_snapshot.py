"""Create self-authored synthetic fixtures; never fetch market data.

Run once with a NEW output directory: python examples/make_acceptance_snapshot.py PATH
The committed examples/snapshots/v1 is the canonical replay input. Regeneration
can differ in last-bit libm rounding across platforms; replay uses saved bytes.
"""
from datetime import date, datetime, time, timedelta, timezone
import math
from pathlib import Path
import sys

import polars as pl
import research_toolkit as rt


def make_market(*, benchmark=False):
    start, end = date(2024, 1, 2), date(2025, 1, 2)
    dates = [start+timedelta(days=i) for i in range((end-start).days+1)
             if (start+timedelta(days=i)).weekday() < 5]
    assets = ["SYNTH_INDEX"] if benchmark else [f"SYNTH_{x}" for x in "ABCDE"]
    splits = [] if benchmark else [("split-A", "SYNTH_A", dates[110], 2.0)]
    dividends = [] if benchmark else [(f"div-{a}-{j}", a, dates[j], dates[j]+timedelta(days=12), .4)
                                     for a in assets for j in (55, 120, 185, 258)]
    rows = []
    for k, asset in enumerate(assets):
        price = 100.+20*k
        for i, day in enumerate(dates):
            if i:
                common = .0003 + .006*math.sin(i*.37) + .003*math.cos(i*.13)
                idiosyncratic = 0 if benchmark else .003*math.sin(i*(.21+.07*k)+k)
                price *= 1+common+idiosyncratic
                if not benchmark:
                    if asset == "SYNTH_A" and i == 110: price /= 2
                    if i in (55, 120, 185, 258): price -= .4
            rows.append((day, asset, price))
    return rt.prepare_market_data(
        prices=pl.DataFrame(rows, schema={"session": pl.Date, "asset": pl.String, "close": pl.Float64}, orient="row"),
        sessions=pl.DataFrame({"session": dates, "close_at": [datetime.combine(d, time(21), tzinfo=timezone.utc) for d in dates]},
                             schema={"session": pl.Date, "close_at": pl.Datetime("us", "UTC")}),
        splits=pl.DataFrame(splits, schema={"action_id": pl.String, "asset": pl.String, "effective_session": pl.Date, "ratio": pl.Float64}, orient="row"),
        dividends=pl.DataFrame(dividends, schema={"action_id": pl.String, "asset": pl.String, "ex_session": pl.Date, "pay_date": pl.Date, "cash_per_share": pl.Float64}, orient="row"),
        metadata={"source": "self-authored synthetic acceptance fixture v1", "retrieved_at": "2025-01-03T00:00:00Z",
            "currency": "USD", "asset_currencies": {a: "USD" for a in assets},
            "calendar": "synthetic_weekdays_not_an_exchange_calendar", "calendar_version": "1",
            "timezone": "America/New_York", "price_basis": "total_return_adjusted" if benchmark else "raw",
            "frequency": "1d", "coverage_start": start.isoformat(), "coverage_end": end.isoformat(),
            "actions_complete": True, "dividend_basis": "post_split_share",
            "usage": "Self-authored fixture; no third-party market-data restrictions. Project license undecided.",
            "transformations": "Deterministic sinusoidal hypothetical returns; raw marks reduced for explicit actions.",
            "benchmark_basis": "synthetic total-return index; hypothetical reinvested distributions embedded, no trading costs" if benchmark else "not a benchmark",
            "calendar_note": "Every weekday, including real exchange holidays; 21:00 UTC close by assumption.",
        })


def main():
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=False)
    rt.save_snapshot(make_market(), target/"equities")
    rt.save_snapshot(make_market(benchmark=True), target/"benchmark")


if __name__ == "__main__":
    main()
