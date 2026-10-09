"""Offline provider handling for aliases, reclassified provider events and delistings."""
from datetime import date, datetime, timezone

import polars as pl
import pytest

import research_toolkit as rt
from research_toolkit.adapters import OVERRIDE_SCHEMA


def chart(days, closes, split_on=None):
    stamps = [int(datetime(d.year, d.month, d.day, 14, 30, tzinfo=timezone.utc).timestamp()) for d in days]
    events = {}
    if split_on is not None:
        t = stamps[days.index(split_on)]
        events = {"splits": {str(t): {"date": t, "numerator": 5, "denominator": 4}}}
    return {"chart": {"error": None, "result": [{"meta": {"symbol": "X", "currency": "USD",
        "exchangeTimezoneName": "America/New_York", "dataGranularity": "1d"}, "timestamp": stamps,
        "indicators": {"quote": [{"close": closes, "volume": [1.]*len(days)}]}, "events": events}]}}


def options(inputs, assets, factors):
    data = inputs(series={a: [1., 1., 1.] for a in assets})
    days = data["sessions"]["session"].to_list()
    table = pl.DataFrame([(d, a, f) for a, fs in factors.items() for d, f in zip(days, fs) if f is not None],
                         schema={"session": pl.Date, "asset": pl.String, "raw_price_factor": pl.Float64}, orient="row")
    return days, dict(sessions=data["sessions"], splits=data["splits"], dividends=data["dividends"],
                      metadata=data["metadata"] | {"price_basis": "raw"}, volume_basis="unknown",
                      availability="session_close_reconstructed", raw_adjustment_factors=table,
                      factor_metadata={"source": "synthetic", "basis": "cumulative_splits_after_session_through_retrieval",
                                       "verified_through": data["metadata"]["retrieved_at"]})


def test_provider_split_label_can_be_overridden_by_sourced_spin_off(inputs):
    days, kw = options(inputs, ["P"], {"P": [1.25, 1., 1.]})
    responses = {"P": chart(days, [80., 80., 81.], split_on=days[1])}
    with pytest.raises(ValueError, match="disagree"):
        rt.adapters.yahoo_chart(responses, **kw)
    override = pl.DataFrame([("P", days[1], "split", 1.25, "SPIN1", "issuer distribution notice")],
                            schema=OVERRIDE_SCHEMA, orient="row")
    converted = rt.adapters.yahoo_chart(responses, provider_split_overrides=override, **kw)
    # Raw closes undo the provider's split adjustment; no split enters the ledger tables.
    assert converted.market.prices["close"].to_list() == [100., 80., 81.]
    assert converted.market.splits.height == 0
    assert converted.metadata["reclassified_provider_events"][0]["action_id"] == "SPIN1"


def test_delisted_asset_keeps_pre_delisting_history_with_lifecycle(inputs):
    days, kw = options(inputs, ["A", "T"], {"A": [1., 1., 1.], "T": [1., 1., None]})
    kw["metadata"] = kw["metadata"] | {"asset_currencies": {"A": "USD", "T": "USD"}}
    lifecycle = rt.corporate_action_inputs(
        securities=[dict(asset="A", security_type="equity", first_session=days[0], unquoted_valuation="none", source="t"),
                    dict(asset="T", security_type="equity", first_session=days[0], last_session=days[1],
                         unquoted_valuation="none", source="t")])
    responses = {"A": chart(days, [10., 11., 12.]), "T": chart(days[:2], [5., 6.])}
    converted = rt.adapters.yahoo_chart(responses, lifecycle=lifecycle, **kw)
    assert converted.market.prices.filter(pl.col("asset") == "T")["close"].to_list() == [5., 6.]
    status = rt.security_status(converted.market)
    assert status.filter((pl.col("asset") == "T") & (pl.col("session") == days[2]))["status"].item() == "unquoted_after_last_session"
    with pytest.raises(ValueError, match="exactly match"):
        rt.adapters.yahoo_chart({"A": responses["A"], "T": chart(days, [5., 6., 7.])}, lifecycle=lifecycle, **kw)


def test_resolve_aliases_maps_dated_tickers_without_joining_securities():
    aliases = pl.DataFrame([("S1", "OLD", "XNYS", date(2024, 1, 1), date(2024, 1, 4), "t"),
                            ("S1", "NEW", "XNYS", date(2024, 1, 5), None, "t"),
                            ("S2", "OLD", "XNYS", date(2024, 2, 1), None, "t")],
                           schema={"asset": pl.String, "ticker": pl.String, "exchange": pl.String, "valid_from": pl.Date,
                                   "valid_to": pl.Date, "source": pl.String}, orient="row")
    rows = pl.DataFrame({"session": [date(2024, 1, 3), date(2024, 1, 8), date(2024, 2, 2)],
                         "ticker": ["OLD", "NEW", "OLD"], "close": [1., 2., 3.]})
    assert rt.adapters.resolve_aliases(rows, aliases)["asset"].to_list() == ["S1", "S1", "S2"]
    gap = pl.DataFrame({"session": [date(2024, 1, 20)], "ticker": ["OLD"], "close": [1.]})
    with pytest.raises(ValueError, match="matches 0 alias"):
        rt.adapters.resolve_aliases(gap, aliases)
