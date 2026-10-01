from dataclasses import replace
import math

import polars as pl
import pytest

import research_toolkit as rt

KW = dict(periods_per_year=4, risk_free_annual_effective=0.0,
          minimum_acceptable_return_annual_effective=0.0)
BM = dict(source="hand oracle", basis="price_only", currency="USD", frequency="1d")


def metric(table, name):
    return table.filter(pl.col("metric") == name).row(0, named=True)


def benchmark(result, values):
    return result.daily.select("period_start", "session").with_columns(pl.Series("simple_return", values, dtype=pl.Float64))


def test_hand_oracle(market, run):
    result = run(market)
    report = rt.performance(result, benchmark=benchmark(result, [.05, -.05]), benchmark_metadata=BM, **KW)
    expected = {"ending_equity": 99, "cumulative_pnl": -1, "cumulative_simple_return": 0,
                "compounded_return": -.01, "annualized_arithmetic_mean": 0,
                "annualized_volatility": math.sqrt(.08), "sharpe": 0, "sortino": 0,
                "max_drawdown": -.1}
    for name, value in expected.items():
        row = metric(report.summary, name)
        assert row["value"] == pytest.approx(value, abs=1e-12)
        assert row["status"] == "ok"
        assert row["n_obs"] == 2
    assert metric(report.benchmark_comparison, "beta")["value"] == pytest.approx(2)
    assert metric(report.benchmark_comparison, "return_correlation")["value"] == pytest.approx(1)
    assert metric(report.benchmark_comparison, "compounded_return_difference")["value"] == pytest.approx(-.75)
    assert metric(report.benchmark_comparison, "relative_wealth_return")["value"] == pytest.approx(99/99.75-1)
    assert report.benchmark_series["equity"].to_list() == pytest.approx([100, 105, 99.75])
    assert report.allocation.group_by("session").agg(pl.col("weight").sum())["weight"].to_list() == pytest.approx([1, 1, 1])


def test_hurdle_conversion_and_all_observation_sortino(inputs, run):
    result = run(rt.prepare_market_data(**inputs(series={"A": [100., 110., 132.]})))
    report = rt.performance(result, periods_per_year=2, risk_free_annual_effective=.21,
        minimum_acceptable_return_annual_effective=.44)
    assert report.metadata["risk_free_periodic"] == pytest.approx(.1)
    assert report.metadata["minimum_acceptable_return_periodic"] == pytest.approx(.2)
    assert metric(report.summary, "sharpe")["value"] == pytest.approx(1)
    assert metric(report.summary, "sortino")["value"] == pytest.approx(-1)


def test_drawdown_keeps_entry_loss_even_if_first_close_recovers(inputs, run):
    result = run(rt.prepare_market_data(**inputs(series={"A": [100., 102., 104.]})),
                 costs=rt.TradeCosts(commission_bps=100., half_spread_bps=0., impact_bps=0.))
    report = rt.performance(result, **KW)
    assert metric(report.summary, "max_drawdown")["value"] == pytest.approx(-1/101)
    assert report.drawdowns["phase"].to_list() == ["pre_entry", "post_entry", "close", "close"]


@pytest.mark.parametrize("values", [[0., 0.], [.05, .05]])
def test_flat_benchmark_does_not_make_up_beta(market, run, values):
    result = run(market)
    report = rt.performance(result, benchmark=benchmark(result, values), benchmark_metadata=BM, **KW)
    assert metric(report.benchmark_comparison, "beta")["status"] == "zero_benchmark_variance"
    assert metric(report.benchmark_comparison, "return_correlation")["value"] is None


def test_flat_and_short(inputs, run):
    result = run(rt.prepare_market_data(**inputs(series={"A": [100., 100., 100.]})))
    report = rt.performance(result, **KW)
    assert metric(report.summary, "annualized_volatility")["value"] == 0
    assert metric(report.summary, "sharpe")["status"] == "zero_volatility"
    assert metric(report.summary, "sortino")["status"] == "zero_downside_risk"
    short = run(rt.prepare_market_data(**inputs()), end_session=result.daily["session"][0])
    report = rt.performance(short, **KW)
    for name in ("sharpe", "sortino", "annualized_volatility"):
        assert metric(report.summary, name)["status"] == "insufficient_samples"


@pytest.mark.parametrize("edit", ["missing", "start", "duplicate", "nan", "null", "loss"])
def test_invalid_benchmark(market, run, edit):
    result = run(market)
    b = benchmark(result, [.1, -.1])
    if edit == "missing": b = b.head(1)
    if edit == "duplicate": b = pl.concat([b, b.head(1)])
    if edit == "start": b = b.with_columns(pl.col("period_start")-pl.duration(days=1))
    if edit in {"nan", "null", "loss"}:
        b = b.with_columns(pl.lit({"nan": float("nan"), "null": None, "loss": -1.}[edit], dtype=pl.Float64).alias("simple_return"))
    with pytest.raises(ValueError): rt.performance(result, benchmark=b, benchmark_metadata=BM, **KW)


def test_benchmark_sorted_by_keys_not_row_position(market, run):
    result = run(market)
    b = benchmark(result, [.05, -.05])
    first = rt.performance(result, benchmark=b, benchmark_metadata=BM, **KW)
    second = rt.performance(result, benchmark=b.reverse(), benchmark_metadata=BM, **KW)
    assert first.benchmark_comparison.equals(second.benchmark_comparison)


@pytest.mark.parametrize("kwargs", [{"periods_per_year": 0}, {"periods_per_year": True},
    {"risk_free_annual_effective": -1}, {"minimum_acceptable_return_annual_effective": float("nan")},
    {"alignment": "inner"}])
def test_invalid_conventions(market, run, kwargs):
    with pytest.raises(ValueError): rt.performance(run(market), **(KW | kwargs))


def test_corrupted_result_rejected(market, run):
    result = run(market)
    bad = replace(result, daily=result.daily.with_columns((pl.col("pnl")+1).alias("pnl")))
    with pytest.raises(ValueError, match="reconcile"): rt.performance(bad, **KW)


def test_stopped_report_requires_opt_in(inputs, run, policy):
    result = run(rt.prepare_market_data(**inputs(series={"A": [100., 40., 45.]})),
        policy=policy(2), cash_rate=None, cash_day_count=None,
        financing=rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F",
            maintenance_equity_ratio=.25, on_breach="stop", cash_sweep="repay_debt"))
    with pytest.raises(ValueError, match="stopped"): rt.performance(result, **KW)
    report = rt.performance(result, allow_partial=True, **KW)
    assert report.metadata["status"] == "stopped"
    assert report.metadata["stop_reason"] == result.stop_reason
    assert report.metadata["actual_end_session"] == result.stop_session.isoformat()
    assert metric(report.summary, "compounded_return")["value"] == pytest.approx(-1.2)
    assert metric(report.summary, "max_drawdown")["value"] == pytest.approx(-1.2)


def test_asset_correlation_basis_and_nulls(inputs):
    market = rt.prepare_market_data(**inputs(series={"A": [100., 110., 99.], "B": [50., 52.5, 49.875], "FLAT": [1., 1., 1.]}))
    returns = rt.returns(market, method="simple", basis="price")
    result = rt.correlation(returns)
    pair = result.values.filter((pl.col("asset") == "A") & (pl.col("other_asset") == "B"))
    assert pair["correlation"][0] == pytest.approx(1)
    assert result.values.filter(pl.col("asset") == "FLAT")["correlation"].null_count() == 3
    assert result.metadata["basis"] == "price"
    with pytest.raises(ValueError, match="simple"):
        rt.correlation(rt.returns(market, method="log", basis="price"))
    corrupted = replace(returns, values=returns.values.with_columns(pl.when(pl.col("session") == market.sessions["session"][1]).then(None).otherwise(pl.col("simple_return")).alias("simple_return")))
    with pytest.raises(ValueError): rt.correlation(corrupted)


def test_zero_portfolio_beta_and_negative_correlation(inputs, run):
    flat = run(rt.prepare_market_data(**inputs(series={"A": [100., 100., 100.]})))
    report = rt.performance(flat, benchmark=benchmark(flat, [.1, -.1]), benchmark_metadata=BM, **KW)
    assert metric(report.benchmark_comparison, "beta")["value"] == 0
    assert metric(report.benchmark_comparison, "beta")["status"] == "ok"
    assert metric(report.benchmark_comparison, "return_correlation")["value"] is None
    moving = run(rt.prepare_market_data(**inputs()))
    report = rt.performance(moving, benchmark=benchmark(moving, [-.05, .05]), benchmark_metadata=BM, **KW)
    assert metric(report.benchmark_comparison, "return_correlation")["value"] == pytest.approx(-1)
    assert metric(report.benchmark_comparison, "beta")["value"] == pytest.approx(-2)


def test_annualization_is_explicit_and_does_not_change_total_return(market, run):
    result = run(market)
    annual4 = rt.performance(result, **KW)
    annual16 = rt.performance(result, **(KW | {"periods_per_year": 16}))
    assert metric(annual16.summary, "annualized_volatility")["value"] == pytest.approx(2*metric(annual4.summary, "annualized_volatility")["value"])
    assert metric(annual16.summary, "compounded_return")["value"] == metric(annual4.summary, "compounded_return")["value"]


def test_correlation_empty_and_one_interval(inputs):
    from datetime import date
    for prices, dates, status in [([100.], [date(2024, 1, 2)], "empty_sample"),
                                  ([100., 101.], [date(2024, 1, 2), date(2024, 1, 3)], "insufficient_samples")]:
        result = rt.correlation(rt.returns(rt.prepare_market_data(**inputs(series={"A": prices}, dates=dates)), method="simple", basis="price"))
        assert result.values["status"][0] == status
        assert result.values["correlation"][0] is None
        assert result.values["n_obs"][0] == len(prices)-1
