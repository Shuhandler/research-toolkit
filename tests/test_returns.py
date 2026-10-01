from dataclasses import replace
from datetime import date
import math

import polars as pl
import pytest

import research_toolkit as rt


def test_simple_sum_compound_and_wealth(market):
    result = rt.returns(market, method="simple", basis="price")
    assert result.values["simple_return"][0] is None
    assert result.values["simple_return"].to_list()[1:] == pytest.approx([.1, -.1])
    assert result.values["period_start"].to_list() == [None, date(2024, 1, 2), date(2024, 1, 3)]
    for method, col, expected in [("sum", "cumulative_simple_return", 0),
                                   ("compound", "compounded_return", -.01),
                                   ("wealth", "wealth_multiple", .99)]:
        cumulative = rt.cumulative_returns(result, method=method)
        assert cumulative.values[col][0] is None
        assert cumulative.values[col][-1] == pytest.approx(expected)
    assert "wealth_multiple" not in result.values.columns


def test_log_conversion_and_multi_asset_isolation(inputs):
    market = rt.prepare_market_data(**inputs({"A": [100, 110, 99], "B": [20, 40, 80]}))
    logs = rt.returns(market, method="log", basis="price")
    compounded = rt.cumulative_returns(logs, method="compound").values
    a = compounded.filter(pl.col("asset") == "A")
    b = compounded.filter(pl.col("asset") == "B")
    assert a["compounded_return"][-1] == pytest.approx(-.01)
    assert b["compounded_return"][-1] == pytest.approx(3)
    summed = rt.cumulative_returns(logs, method="sum").values
    assert summed.filter(pl.col("asset") == "A")["cumulative_log_return"][-1] == pytest.approx(math.log(.99))


def test_one_observation_and_flat_prices(inputs):
    data = inputs({"A": [100]}, dates=[date(2024, 1, 2)])
    result = rt.returns(rt.prepare_market_data(**data), method="simple", basis="price")
    assert rt.cumulative_returns(result, method="wealth").values["wealth_multiple"].to_list() == [None]
    flat = rt.returns(rt.prepare_market_data(**inputs({"A": [100]*3})), method="log", basis="price")
    assert flat.values["log_return"].to_list() == [None, 0, 0]


def test_adjusted_basis_and_raw_split_diagnostic(inputs):
    adjusted = rt.prepare_market_data(**inputs(price_basis="total_return_adjusted"))
    with pytest.raises(ValueError, match="incompatible"):
        rt.returns(adjusted, method="simple", basis="price")
    assert rt.returns(adjusted, method="simple", basis="total_return").metadata["basis"] == "total_return"
    raw = rt.prepare_market_data(**inputs({"A": [100, 50, 50]}, splits=[("S", "A", date(2024, 1, 3), 2.)]))
    result = rt.returns(raw, method="simple", basis="price")
    assert result.values["simple_return"][1] == -.5
    assert "raw_price_split_discontinuity" in result.diagnostics["code"].to_list()
    with pytest.raises(ValueError, match="incompatible"):
        rt.returns(raw, method="simple", basis="total_return")


@pytest.mark.parametrize("fault", ["interior_null", "nan", "bad_start", "missing", "duplicate", "first_zero", "minus_one"])
def test_invalid_return_tables_rejected(market, fault):
    result = rt.returns(market, method="simple", basis="price")
    rows = result.values.to_dicts()
    if fault == "interior_null": rows[1]["simple_return"] = None
    elif fault == "nan": rows[1]["simple_return"] = float("nan")
    elif fault == "bad_start": rows[2]["period_start"] = date(2024, 1, 2)
    elif fault == "missing": rows.pop(1)
    elif fault == "duplicate": rows.append(rows[1])
    elif fault == "first_zero": rows[0]["simple_return"] = 0.
    elif fault == "minus_one": rows[1]["simple_return"] = -1.
    changed = replace(result, values=pl.DataFrame(rows, schema=result.values.schema))
    with pytest.raises(ValueError):
        rt.cumulative_returns(changed, method="compound")


def test_extreme_and_small_log_changes(inputs):
    market = rt.prepare_market_data(**inputs({"A": [1e-300, 1e300, 1e300]}))
    with pytest.raises(ValueError, match="representable"):
        rt.returns(market, method="simple", basis="price")
    logs = rt.returns(market, method="log", basis="price")
    assert math.isfinite(logs.values["log_return"][1])
    with pytest.raises(ValueError, match="overflow"):
        rt.cumulative_returns(logs, method="wealth")
    near = rt.prepare_market_data(**inputs({"A": [1e100, math.nextafter(1e100, math.inf), 1e100]}))
    assert rt.returns(near, method="log", basis="price").values["log_return"][1] > 0


def test_changed_frequency_and_null_keys_rejected(market):
    result = rt.returns(market, method="simple", basis="price")
    with pytest.raises(ValueError, match="frequency"):
        rt.cumulative_returns(replace(result, metadata=result.metadata | {"frequency": "8h"}), method="sum")
    with pytest.raises(ValueError, match="nonnull"):
        rt.cumulative_returns(replace(result, values=result.values.with_columns(
            pl.lit(None, dtype=pl.String).alias("asset"))), method="sum")
