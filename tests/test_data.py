from datetime import date

import polars as pl
import pytest

import research_toolkit as rt


def test_canonical_copy_and_snapshot_identity(inputs):
    data = inputs({"B": [20, 20, 20], "A": [100, 110, 99]})
    original = data["prices"].clone()
    one = rt.prepare_market_data(**data)
    two = rt.prepare_market_data(**(data | {"prices": data["prices"].reverse()}))
    assert data["prices"].equals(original)
    assert one.prices.equals(two.prices)
    assert one.snapshot_id == two.snapshot_id
    assert one.diagnostics["code"].to_list() == ["sorted_rows"]
    data["metadata"]["source"] = "changed"
    assert one.metadata["source"] != "changed"
    assert rt.prepare_market_data(**data).snapshot_id != one.snapshot_id


@pytest.mark.parametrize("fault, message", [
    ("duplicate", "duplicate"), ("missing", "alignment"), ("off_calendar", "alignment"),
    ("zero", "positive"), ("negative", "positive"), ("nan", "finite"), ("inf", "finite"),
    ("null", "null"), ("wrong_dtype", "schema"), ("empty", "empty"), ("extra_column", "schema"),
])
def test_price_validation(inputs, fault, message):
    data = inputs()
    prices = data["prices"]
    if fault == "duplicate": prices = pl.concat([prices, prices.head(1)])
    elif fault == "missing": prices = prices.tail(2)
    elif fault == "off_calendar": prices = prices.with_columns(pl.lit(date(2024, 1, 6)).alias("session")).head(1)
    elif fault == "empty": prices = prices.clear()
    elif fault == "wrong_dtype": prices = prices.with_columns(pl.col("close").cast(pl.Int64))
    elif fault == "extra_column": prices = prices.with_columns(pl.lit("USD").alias("currency"))
    else:
        value = {"zero": 0., "negative": -1., "nan": float("nan"), "inf": float("inf"), "null": None}[fault]
        prices = prices.with_columns(pl.lit(value, dtype=pl.Float64).alias("close"))
    with pytest.raises(ValueError, match=message):
        rt.prepare_market_data(**(data | {"prices": prices}))


@pytest.mark.parametrize("key, value, message", [
    ("actions_complete", False, "actions_complete"),
    ("frequency", "8h", "frequency"), ("price_basis", "unknown", "price_basis"),
    ("asset_currencies", {"A": "EUR"}, "single currency"),
    ("asset_currencies", {}, "exactly"), ("dividend_basis", "unknown", "dividend_basis"),
    ("retrieved_at", "2024-01-01", "UTC"), ("timezone", "Made/Up", "IANA"),
    ("timezone", "Asia/Tokyo", "session date"),
    ("coverage_start", "2024-01-01", "coverage_start"),
])
def test_metadata_validation(inputs, key, value, message):
    with pytest.raises(ValueError, match=message):
        rt.prepare_market_data(**inputs(**{key: value}))


def test_required_metadata_and_no_fill(inputs):
    data = inputs()
    del data["metadata"]["calendar_version"]
    with pytest.raises(ValueError, match="missing keys"):
        rt.prepare_market_data(**data)
    with pytest.raises(ValueError, match="missing='raise'"):
        rt.prepare_market_data(**inputs(), missing="forward_fill")


@pytest.mark.parametrize("fault", ["duplicate_id", "unknown_asset", "unknown_session", "bad_ratio", "early_pay", "null_pay", "untyped"])
def test_action_validation(inputs, fault):
    ex, pay = date(2024, 1, 3), date(2024, 1, 6)
    splits = [("S", "A", ex, 2.)]
    dividends = [("D", "A", ex, pay, 1.)]
    if fault == "duplicate_id": dividends = [("S", "A", ex, pay, 1.)]
    elif fault == "unknown_asset": splits = [("S", "B", ex, 2.)]
    elif fault == "unknown_session": splits = [("S", "A", pay, 2.)]
    elif fault == "bad_ratio": splits = [("S", "A", ex, 0.)]
    elif fault == "early_pay": dividends = [("D", "A", ex, date(2024, 1, 2), 1.)]
    elif fault == "null_pay": dividends = [("D", "A", ex, None, 1.)]
    data = inputs(splits=splits, dividends=dividends)
    if fault == "untyped": data["splits"] = pl.DataFrame()
    with pytest.raises(ValueError):
        rt.prepare_market_data(**data)


def test_changed_snapshot_cannot_bypass_validation(market, run):
    market.metadata["source"] = "edited after preparation"
    with pytest.raises(ValueError, match="changed after validation"):
        run(market)


def test_duplicate_session_and_ambiguous_split(inputs):
    data = inputs()
    with pytest.raises(ValueError, match="duplicate"):
        rt.prepare_market_data(**(data | {"sessions": pl.concat([data["sessions"], data["sessions"].head(1)])}))
    with pytest.raises(ValueError, match="multiple splits"):
        rt.prepare_market_data(**inputs(splits=[("S1", "A", date(2024, 1, 3), 2.), ("S2", "A", date(2024, 1, 3), 2.)]))
