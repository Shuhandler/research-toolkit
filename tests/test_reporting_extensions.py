from datetime import date, datetime, timezone

import polars as pl
import pytest
import research_toolkit as rt
from test_metrics import KW, BM, metric, benchmark


@pytest.fixture
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yield plt
    plt.close("all")


def test_dated_risk_free_sharpe_denominators(inputs, run):
    # Returns 10%, 20%; RF 0%, 5%. Excess mean = 12.5%, std = sqrt(.00125).
    result = run(rt.prepare_market_data(**inputs(series={"A": [100., 110., 132.]})))
    rf = benchmark(result, [0., .05])
    kw = dict(periods_per_year=2, risk_free_returns=rf.reverse(), risk_free_metadata=BM,
        minimum_acceptable_return_annual_effective=0., benchmark=benchmark(result, [.05, .10]), benchmark_metadata=BM)
    standard = rt.performance(result, **kw)
    excess = rt.performance(result, sharpe_denominator="excess_returns", **kw)
    assert metric(standard.summary, "sharpe")["value"] == pytest.approx(2.5)
    assert metric(excess.summary, "sharpe")["value"] == pytest.approx(5.)
    assert metric(standard.benchmark_summary, "sharpe")["value"] == pytest.approx(2.)
    assert standard.risk_free_returns.equals(rf)
    rolling = rt.rolling_risk(result, window=2, periods_per_year=2, risk_free_returns=rf,
        risk_free_metadata=BM, sharpe_denominator="excess_returns")
    assert rolling.values["sharpe"][-1] == pytest.approx(5.)
    with pytest.raises(ValueError, match="exactly one"):
        rt.performance(result, risk_free_annual_effective=0., **kw)
    with pytest.raises(ValueError, match="intervals"):
        rt.performance(result, **(kw | {"risk_free_returns": rf.with_columns(pl.col("period_start")-pl.duration(days=1))}))
    report = rt.performance(result, sharpe_denominator="excess_returns", **(kw | {"risk_free_returns": result.daily.select("period_start", "session", "simple_return")}))
    assert metric(report.summary, "sharpe")["status"] == "zero_excess_volatility"


def test_calendar_risk_free_weekend_daycounts_timing():
    intervals = pl.DataFrame({"period_start": [date(2024, 1, 5)], "session": [date(2024, 1, 8)]})
    rates = pl.DataFrame({"date": [date(2024, 1, d) for d in (6, 7, 8)], "annual_rate": [.36, .72, .36],
        "available_at": [datetime(2024, 1, 5, tzinfo=timezone.utc)]*3})
    kw = dict(intervals=intervals, day_count="ACT/360", compounding="daily", rate_timing="known_at_accrual_start",
        timezone_name="America/New_York", metadata=BM)
    converted = rt.risk_free_returns(rates, **kw)
    assert converted.values["simple_return"][0] == pytest.approx(.004005002)
    assert rt.risk_free_returns(rates, **(kw | {"compounding": "simple"})).values["simple_return"][0] == pytest.approx(.004)
    assert rt.risk_free_returns(rates, **(kw | {"compounding": "simple", "day_count": "ACT/365F"})).values["simple_return"][0] == pytest.approx(1.44/365)
    with pytest.raises(ValueError, match="every accrual date"):
        rt.risk_free_returns(rates.tail(2), **kw)
    with pytest.raises(ValueError, match="unavailable"):
        rt.risk_free_returns(rates.with_columns(pl.col("available_at")+pl.duration(days=5)), **kw)
    with pytest.raises(ValueError, match="known_at_accrual_start"):
        rt.risk_free_returns(rates, **(kw | {"rate_timing": "same_day_quote"}))


def test_risk_free_result_consumed_and_negative_rate(market, run):
    result = run(market)
    rates = result.daily.select(pl.col("session").alias("date")).with_columns(pl.lit(-.036).alias("annual_rate"),
        pl.lit(datetime(2024, 1, 1, tzinfo=timezone.utc)).alias("available_at"))
    converted = rt.risk_free_returns(rates, intervals=result.daily.select("period_start", "session"), day_count="ACT/360",
        compounding="daily", rate_timing="known_at_accrual_start", timezone_name="America/New_York", metadata=BM)
    assert converted.values["simple_return"].to_list() == pytest.approx([-.0001]*2)
    kw = dict(periods_per_year=4, minimum_acceptable_return_annual_effective=0., risk_free_returns=converted)
    assert rt.performance(result, **kw).metadata["risk_free"]["day_count"] == "ACT/360"
    with pytest.raises(ValueError, match="already includes"):
        rt.performance(result, risk_free_metadata=BM, **kw)


def test_rolling_benchmark_oracles_and_undefined(market, run, plt):
    result = run(market)
    kw = dict(window=2, periods_per_year=4, risk_free_annual_effective=0., benchmark=benchmark(result, [.05, -.05]), benchmark_metadata=BM)
    rolling = rt.rolling_risk(result, **kw)
    assert rolling.values["beta"].to_list() == [None, pytest.approx(2.)]
    assert rolling.values["correlation"].to_list() == [None, pytest.approx(1.)]
    assert rolling.values["beta_status"][0] == "insufficient_samples"
    flat = rt.rolling_risk(result, **(kw | {"benchmark": benchmark(result, [.01, .01])}))
    assert flat.values["beta_status"][-1] == "zero_benchmark_variance"
    assert flat.values["beta"][-1] is None
    with pytest.raises(ValueError, match="intervals"):
        rt.rolling_risk(result, **(kw | {"benchmark": kw["benchmark"].tail(1)}))
    fig, ax = rt.plots.rolling_risk(rolling, metric="beta")
    assert ax.lines[0].get_ydata()[-1] == pytest.approx(2.)
    plt.close(fig)


@pytest.mark.parametrize("method,expected,label", [("sum", [0., .1, 0.], "sum"), ("compound", [0., .1, -.01], "compounded"), ("wealth", [1., 1.1, .99], "wealth")])
def test_cumulative_plots_prepared_data_only(market, run, monkeypatch, method, expected, label, plt):
    result = run(market)
    report = rt.performance(result, benchmark=benchmark(result, [.05, -.05]), benchmark_metadata=BM, **KW)
    def fail(*args, **kwargs):
        raise AssertionError("plot must only consume prepared tables")
    monkeypatch.setattr(rt, "performance", fail)
    monkeypatch.setattr(rt, "buy_and_hold", fail)
    monkeypatch.setattr(plt, "show", fail)
    fig, ax = rt.plots.cumulative_returns(report, method=method)
    assert ax.lines[0].get_ydata().tolist() == pytest.approx(expected)
    assert len(ax.lines) == 2 and label in ax.get_ylabel().lower()
    plt.close(fig)


def test_comparison_keeps_units_and_benchmark_once(market, run):
    result = run(market)
    report = rt.performance(result, benchmark=benchmark(result, [.05, -.05]), benchmark_metadata=BM, **KW)
    comparison = rt.compare_performance({"Base": report, "Copy": report}, benchmark_label="Reference")
    assert comparison.values["scenario"].unique().sort().to_list() == ["Base", "Copy", "Reference"]
    assert comparison.values.schema["period_start"] == pl.Date
    assert comparison.diagnostics.is_empty()
    sharpe = comparison.values.filter(pl.col("metric") == "sharpe")
    assert sharpe["unit"].to_list() == ["ratio"]*3
    assert comparison.values.filter((pl.col("scenario") == "Reference") & (pl.col("metric") == "ending_equity"))["value"][0] == pytest.approx(99.75)
    changed = rt.performance(result, benchmark=benchmark(result, [.05, -.05]), benchmark_metadata=BM,
        **(KW | {"periods_per_year": 252}))
    with pytest.raises(ValueError, match="incompatible"):
        rt.compare_performance({"Base": report, "Different": changed})
    explicit = rt.compare_performance({"Base": report, "Different": changed}, assumptions="separate", benchmark_label="Reference")
    assert "periods_per_year" in explicit.diagnostics["field"]
    assert explicit.values.filter(pl.col("kind") == "benchmark")["scenario"].n_unique() == 2


def test_comparison_unequal_coverage_and_stopped_status(inputs, run, policy, plt):
    market = rt.prepare_market_data(**inputs(series={"A": [100., 40., 45.]}))
    full = rt.performance(run(market), **KW)
    stopped = run(market, policy=policy(2.), cash_rate=None, cash_day_count=None,
        financing=rt.Financing(cash_rate=0., borrowing_rate=0., day_count="ACT/365F", maintenance_equity_ratio=.25,
            on_breach="stop", cash_sweep="repay_debt"))
    partial = rt.performance(stopped, allow_partial=True, **KW)
    reports = {"Full": full, "Stopped": partial}
    with pytest.raises(ValueError, match="allow_partial"):
        rt.compare_performance(reports)
    with pytest.raises(ValueError, match="covers different holding intervals"):
        rt.compare_performance(reports, allow_partial=True)
    # Same scalar risk-free rate, but its rows follow coverage: the message names coverage as the cause.
    with pytest.raises(ValueError, match="same risk_free convention, but its risk_free rows cover different intervals"):
        rt.compare_performance(reports, allow_partial=True, coverage="separate")
    table = rt.compare_performance(reports, allow_partial=True, coverage="separate", assumptions="separate")
    rows = table.values.filter(pl.col("scenario") == "Stopped")
    assert rows["run_status"].unique().to_list() == ["stopped"]
    assert rows["actual_end_session"].unique().to_list() == [date(2024, 1, 3)]
    fig, ax = rt.plots.cumulative_returns(partial, method="compound")
    assert "STOPPED" in ax.get_title()
    plt.close(fig)


def test_explicit_sofr_source_reuse_keeps_loan_spread_separate(inputs, run, policy):
    from test_sofr import simulate, sofr_config, check_audit
    from test_execution_extensions import model
    result = simulate(inputs, run, policy, costs=model(sigma=.01), finance=sofr_config(borrowing_spread_bps=100.))
    check_audit(result)
    rates = result.financing_accruals.select("date", pl.col("sofr").alias("annual_rate"), "available_at")
    rf = rt.risk_free_returns(rates, intervals=result.daily.select("period_start", "session"), day_count="ACT/360",
        compounding="daily", rate_timing="known_at_accrual_start", timezone_name="America/New_York", metadata=BM)
    assert rf.values["simple_return"].to_list() == pytest.approx([.000300030001, .0002, .0003])
    report = rt.performance(result, periods_per_year=252, risk_free_returns=rf,
        minimum_acceptable_return_annual_effective=0.)
    assert report.metadata["risk_free_annual_effective"] is None
    assert result.financing_accruals["borrowing_spread_bps"].to_list() == [100.]*5
    assert metric(report.summary, "entry_cost")["value"] > 0


def test_synthetic_workflow_integrates_previews_and_reports():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/"examples/research_workflow.py"
    spec = importlib.util.spec_from_file_location("research_workflow", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    outputs = example.run_example(initial_capital=500_000., leverages=(.5, 1.25))
    for label, result in outputs["runs"].items():
        preview = outputs["previews"][label]
        assert preview.orders["total_cost"].sum() == pytest.approx(result.execution_costs["total_cost"].sum())
        assert outputs["reports"][label].metadata["status"] == "complete"
        assert result.dividend_reinvestments["trade_cost"].sum() == 0
    assert outputs["comparison"].diagnostics.is_empty()
    assert outputs["comparison"].values.filter(pl.col("kind") == "benchmark")["scenario"].n_unique() == 1


def test_different_benchmarks_require_explicit_comparison_handling(market, run):
    result = run(market)
    one = rt.performance(result, benchmark=benchmark(result, [.01, -.01]), benchmark_metadata=BM, **KW)
    two = rt.performance(result, benchmark=benchmark(result, [.02, -.02]), benchmark_metadata=BM, **KW)
    with pytest.raises(ValueError, match="benchmark"):
        rt.compare_performance({"One": one, "Two": two}, benchmark_label="Benchmark")
    compared = rt.compare_performance({"One": one, "Two": two}, benchmark_label="Benchmark", assumptions="separate")
    assert compared.values.filter(pl.col("kind") == "benchmark")["scenario"].n_unique() == 2
