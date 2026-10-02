from copy import deepcopy
from datetime import date, datetime, timezone

import polars as pl
import pytest
import research_toolkit as rt


def fixture(inputs, *, split=False):
    data = inputs(series={"A": [100., 50., 55.]}, splits=[("s", "A", date(2024, 1, 3), 2.)] if split else [])
    timestamps = [int(datetime(d.year, d.month, d.day, 14, 30, tzinfo=timezone.utc).timestamp()) for d in data["sessions"]["session"]]
    payload = {"chart": {"error": None, "result": [{"meta": {"symbol": "A", "currency": "USD", "exchangeTimezoneName": "America/New_York", "dataGranularity": "1d"},
        "timestamp": timestamps, "indicators": {"quote": [{"close": [50., 50., 55.], "volume": [200., 300., 400.]}], "adjclose": [{"adjclose": [49., 49., 55.]}]},
        "events": {"splits": {str(timestamps[1]): {"date": timestamps[1], "numerator": 2, "denominator": 1}}} if split else {}}]}}
    args = {k: data[k] for k in ("sessions", "splits", "dividends", "metadata")}
    args["metadata"] = data["metadata"] | {"price_basis": "split_adjusted"}
    args.update(volume_basis="unknown", availability="session_close_reconstructed")
    return {"A": payload}, args


def raw_options(args, factors):
    return dict(metadata=args["metadata"] | {"price_basis": "raw"},
        raw_adjustment_factors=args["sessions"].select("session").with_columns(pl.lit("A").alias("asset"), pl.Series("raw_price_factor", factors, dtype=pl.Float64)),
        factor_metadata={"source": "synthetic verified split history", "basis": "cumulative_splits_after_session_through_retrieval",
                         "verified_through": args["metadata"]["retrieved_at"]})


def test_adjusted_conversion_preserves_provenance_without_inventing_volume(inputs, run):
    responses, args = fixture(inputs)
    converted = rt.adapters.yahoo_chart(responses, **args)
    assert converted.market.metadata["price_basis"] == "split_adjusted"
    assert converted.market.prices["close"].to_list() == [50., 50., 55.]
    assert converted.bars["dollar_volume"].null_count() == 3
    assert converted.bars["available_at"].equals(args["sessions"]["close_at"].rename("available_at"))
    assert "reconstructed_availability" in converted.diagnostics["code"]
    assert len(converted.metadata["response_identities"]["A"]["sha256"]) == 64
    with pytest.raises(ValueError, match="raw execution"):
        run(converted.market)
    total = rt.adapters.yahoo_chart(responses, **(args | {"metadata": args["metadata"] | {"price_basis": "total_return_adjusted"}}))
    assert total.market.prices["close"].to_list() == [49., 49., 55.]


def test_raw_split_reconstruction_volume_and_ledger(inputs, run):
    responses, args = fixture(inputs, split=True)
    converted = rt.adapters.yahoo_chart(responses, **(args | raw_options(args, [2., 1., 1.]) | {"volume_basis": "split_adjusted_shares"}))
    assert converted.market.prices["close"].to_list() == [100., 50., 55.]
    assert converted.bars["raw_share_volume"].to_list() == [100., 300., 400.]
    assert converted.bars["dollar_volume"].to_list() == [10000., 15000., 22000.]
    result = run(converted.market)
    assert result.daily["pnl"].to_list() == pytest.approx([0, 10])
    assert result.positions["quantity"].to_list() == [1., 2., 2.]
    # Later splits beyond selected coverage still affect historical Yahoo prices.
    outside = rt.adapters.yahoo_chart(responses, **(args | raw_options(args, [6., 3., 3.])))
    assert outside.market.prices["close"].to_list() == [300., 150., 165.]


def test_missing_raw_factors_and_inconsistent_split_rejected(inputs):
    responses, args = fixture(inputs, split=True)
    with pytest.raises(ValueError, match="raw_adjustment_factors"):
        rt.adapters.yahoo_chart(responses, **(args | {"metadata": args["metadata"] | {"price_basis": "raw"}}))
    with pytest.raises(ValueError, match="disagree"):
        rt.adapters.yahoo_chart(responses, **(args | raw_options(args, [1., 1., 1.])))
    kw = args | raw_options(args, [2., 1., 1.])
    kw["factor_metadata"]["verified_through"] = "2024-01-04T22:00:00Z"
    with pytest.raises(ValueError, match="verified through"):
        rt.adapters.yahoo_chart(responses, **kw)


@pytest.mark.parametrize("edit", ["missing", "duplicate", "null_close", "nan", "currency", "frequency", "capital_gains", "split", "dividend"])
def test_provider_errors_do_not_silently_drop_or_invent(inputs, edit):
    responses, args = fixture(inputs)
    r = responses["A"]["chart"]["result"][0]
    if edit == "missing": r["timestamp"].pop()
    if edit == "duplicate": r["timestamp"][1] = r["timestamp"][0]
    if edit == "null_close": r["indicators"]["quote"][0]["close"][0] = None
    if edit == "nan": r["indicators"]["quote"][0]["close"][0] = float("nan")
    if edit == "currency": r["meta"]["currency"] = "EUR"
    if edit == "frequency": r["meta"]["dataGranularity"] = "1h"
    if edit == "capital_gains": r["events"] = {"capitalGains": {}}
    if edit == "split": r["events"] = {"splits": {"s": {"date": r["timestamp"][1], "numerator": 2, "denominator": 1}}}
    if edit == "dividend": r["events"] = {"dividends": {"d": {"date": r["timestamp"][1], "amount": 1.}}}
    with pytest.raises(ValueError):
        rt.adapters.yahoo_chart(responses, **args)


def test_payment_dates_supplied_not_inferred_from_provider_events(inputs):
    responses, args = fixture(inputs)
    r = responses["A"]["chart"]["result"][0]
    r["events"] = {"dividends": {"d": {"date": r["timestamp"][1], "amount": 1.}}}
    args["dividends"] = pl.DataFrame([("d", "A", date(2024, 1, 3), date(2024, 2, 10), 2.)], schema=args["dividends"].schema, orient="row")
    converted = rt.adapters.yahoo_chart(responses, **args)
    assert converted.market.dividends["pay_date"][0] == date(2024, 2, 10)
    assert converted.market.dividends["cash_per_share"][0] == 2.  # Authoritative raw amount; provider may be split adjusted.
