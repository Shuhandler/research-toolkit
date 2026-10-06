from dataclasses import replace
from datetime import date
import math

import polars as pl
import pytest

import research_toolkit as rt
from test_metrics import BM, metric

META = {"source": "hand-calculated P&L series", "currency": "USD"}
KW = dict(initial_capital=100.0, periods_per_year=4, risk_free_annual_effective=0.0, metadata=META)
# Tue, Wed, then a Friday-to-Monday interval: weekends are not gaps.
SESSIONS = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5), date(2024, 1, 8)]


def pnl_table(values, sessions=SESSIONS):
    return pl.DataFrame({"period_start": sessions[:-1], "session": sessions[1:], "pnl": values},
                        schema={"period_start": pl.Date, "session": pl.Date, "pnl": pl.Float64})


def test_nav_accounting_and_compounding_by_hand():
    report = rt.series_performance(pnl_table([10., -22., 5.5]), **KW)
    assert report.daily["opening_equity"].to_list() == pytest.approx([100, 110, 88])
    assert report.daily["equity"].to_list() == pytest.approx([110, 88, 93.5])
    # 10/100, -22/110, 5.5/88: each interval's P&L over the previous NAV.
    assert report.daily["simple_return"].to_list() == pytest.approx([.1, -.2, .0625])
    assert metric(report.summary, "compounded_return")["value"] == pytest.approx(-.065)
    assert metric(report.summary, "cumulative_simple_return")["value"] == pytest.approx(-.0375)
    assert metric(report.summary, "ending_equity")["value"] == pytest.approx(93.5)
    assert metric(report.summary, "cumulative_pnl")["value"] == pytest.approx(-6.5)
    assert metric(report.summary, "max_drawdown")["value"] == pytest.approx(88/110-1)
    assert report.cumulative.filter(pl.col("series") == "portfolio")["wealth"].to_list() == pytest.approx([1, 1.1, .88, .935])
    assert metric(report.summary, "sortino")["status"] == "mar_not_supplied"
    assert report.metadata["source_metadata"] == META and report.metadata["initial_capital"] == 100.


def test_initial_capital_is_first_drawdown_observation():
    report = rt.series_performance(pnl_table([-5., 2., 1.]), **KW)
    assert report.equity["equity"].to_list() == pytest.approx([100, 95, 97, 98])
    assert report.equity["phase"][0] == "initial_capital" and report.equity["session"][0] == SESSIONS[0]
    assert report.drawdowns["drawdown"].to_list() == pytest.approx([0, -.05, -.03, -.02])
    assert metric(report.summary, "max_drawdown")["value"] == pytest.approx(-.05)


def test_matches_ledger_report_for_same_pnl(market, run):
    result = run(market)
    ledger = rt.performance(result, periods_per_year=4, risk_free_annual_effective=.02,
                            minimum_acceptable_return_annual_effective=0.)
    series = rt.series_performance(result.daily.select("period_start", "session", "pnl"), initial_capital=100.,
        periods_per_year=4, risk_free_annual_effective=.02, minimum_acceptable_return_annual_effective=0.,
        metadata=META)
    for name in ("ending_equity", "compounded_return", "annualized_volatility", "sharpe", "sortino"):
        assert metric(series.summary, name)["value"] == pytest.approx(metric(ledger.summary, name)["value"])
    assert series.daily["simple_return"].to_list() == pytest.approx(ledger.daily["simple_return"].to_list())


def test_benchmark_must_align_exactly():
    daily = pnl_table([10., -22., 5.5])
    bench = daily.select("period_start", "session").with_columns(pl.Series("simple_return", [.05, -.1, .02]))
    report = rt.series_performance(daily, benchmark=bench, benchmark_metadata=BM, **KW)
    assert report.benchmark_series["equity"].to_list() == pytest.approx([100, 105, 94.5, 96.39])
    assert metric(report.benchmark_comparison, "benchmark_compounded_return")["value"] == pytest.approx(-.0361)
    assert metric(report.benchmark_comparison, "relative_wealth_return")["value"] == pytest.approx(93.5/96.39-1)
    with pytest.raises(ValueError, match="match both endpoints"):
        rt.series_performance(daily, benchmark=bench.head(2), benchmark_metadata=BM, **KW)
    shifted = bench.with_columns(pl.Series("period_start", [date(2024, 1, 1), SESSIONS[1], SESSIONS[2]]))
    with pytest.raises(ValueError, match="match both endpoints"):
        rt.series_performance(daily, benchmark=shifted, benchmark_metadata=BM, **KW)
    with pytest.raises(ValueError, match="benchmark_metadata requires"):
        rt.series_performance(daily, benchmark_metadata=BM, **KW)


def test_empty_or_reversed_intervals_raise():
    empty = pnl_table([1., 1., 1.]).with_columns(pl.Series("period_start", [SESSIONS[0], SESSIONS[1], SESSIONS[3]]))
    with pytest.raises(ValueError, match="start before"):
        rt.series_performance(empty, **KW)
    reversed_interval = [date(2024, 1, 3), date(2024, 1, 2), date(2024, 1, 5), date(2024, 1, 8)]
    with pytest.raises(ValueError, match="start before"):
        rt.series_performance(pnl_table([1., 1., 1.], reversed_interval), **KW)


def test_gaps_overlaps_order_and_duplicates_raise():
    gap = pnl_table([1., 1., 1.]).with_columns(pl.Series("period_start", [SESSIONS[0], date(2024, 1, 4), SESSIONS[2]]))
    with pytest.raises(ValueError, match="contiguous"):
        rt.series_performance(gap, **KW)
    overlap = pnl_table([1., 1., 1.]).with_columns(pl.Series("period_start", [SESSIONS[0], SESSIONS[0], SESSIONS[2]]))
    with pytest.raises(ValueError, match="overlaps"):
        rt.series_performance(overlap, **KW)
    with pytest.raises(ValueError, match="increasing session order"):
        rt.series_performance(pnl_table([1., 1., 1.]).reverse(), **KW)
    with pytest.raises(ValueError, match="duplicate"):
        rt.series_performance(pl.concat([pnl_table([1., 1., 1.]), pnl_table([1., 1., 1.]).tail(1)]), **KW)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            rt.series_performance(pnl_table([1., bad, 1.]), **KW)
    with pytest.raises(ValueError, match="null"):
        rt.series_performance(pnl_table([1., None, 1.]), **KW)
    with pytest.raises(ValueError, match="empty"):
        rt.series_performance(pnl_table([], SESSIONS[:1]), **KW)


@pytest.mark.parametrize("values", [[-100., 1., 1.], [10., -120., 1.]])
def test_nonpositive_nav_raises(values):
    with pytest.raises(ValueError, match="NAV must stay positive"):
        rt.series_performance(pnl_table(values), **KW)


@pytest.mark.parametrize("change", [dict(initial_capital=0.), dict(initial_capital=-1.), dict(initial_capital=float("nan")),
    dict(initial_capital=True), dict(periods_per_year=0), dict(periods_per_year=float("inf")),
    dict(risk_free_annual_effective=-1.), dict(risk_free_annual_effective=None),
    dict(minimum_acceptable_return_annual_effective=-1.), dict(metadata={"source": "no currency"}),
    dict(metadata=META | {"frequency": "1h"}), dict(metadata=META | {"status": "complete"}),
    dict(sharpe_denominator="other")])
def test_invalid_assumptions_raise(change):
    with pytest.raises(ValueError):
        rt.series_performance(pnl_table([1., 1., 1.]), **(KW | change))


def test_risk_free_conversion_and_inputs_untouched():
    daily = pnl_table([10., -22., 5.5])
    before = daily.clone()
    report = rt.series_performance(daily, **(KW | dict(risk_free_annual_effective=.21, minimum_acceptable_return_annual_effective=.21)))
    assert daily.equals(before)
    periodic = 1.21**.25-1
    assert report.risk_free_returns["simple_return"].to_list() == pytest.approx([periodic]*3)
    r = [.1, -.2, .0625]
    excess = [x-periodic for x in r]
    sd = math.sqrt(sum((x-sum(r)/3)**2 for x in r)/2)
    assert metric(report.summary, "sharpe")["value"] == pytest.approx(2*sum(excess)/3/sd)
    assert metric(report.summary, "sortino")["status"] == "ok"


def test_existing_comparison_accepts_series_reports():
    first = rt.series_performance(pnl_table([10., -22., 5.5]), **KW)
    second = rt.series_performance(pnl_table([1., 2., 3.]), **KW)
    table = rt.compare_performance({"a": first, "b": second}).values
    assert set(table["scenario"]) == {"a", "b"}


def test_existing_plots_accept_series_report(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    daily = pnl_table([10., -22., 5.5])
    bench = daily.select("period_start", "session").with_columns(pl.Series("simple_return", [.05, -.1, .02]))
    report = rt.series_performance(daily, benchmark=bench, benchmark_metadata=BM, **KW)
    try:
        _, ax = rt.plots.equity(report)
        assert ax.lines[0].get_ydata().tolist() == pytest.approx([100, 110, 88, 93.5])
        assert ax.lines[1].get_ydata().tolist() == pytest.approx([100, 105, 94.5, 96.39])
        _, ax = rt.plots.drawdown(report)
        assert ax.lines[0].get_ydata().tolist() == pytest.approx([0, 0, 88/110-1, 93.5/110-1])
        for name in ("pnl", "returns", "distribution"):
            getattr(rt.plots, name)(report)[0].savefig(tmp_path/f"{name}.png")
        for method in ("sum", "compound", "wealth"):
            rt.plots.cumulative_returns(report, method=method)[0].savefig(tmp_path/f"{method}.png")
    finally:
        plt.close("all")


def report_with(returns, pnl=None):
    base = rt.series_performance(pnl_table([1., 1., 1.]), **KW)
    pnl = returns if pnl is None else pnl
    return replace(base, daily=pl.DataFrame({"simple_return": returns, "pnl": pnl}, schema={"simple_return": pl.Float64, "pnl": pl.Float64}))


def test_tail_risk_reference_example():
    row = rt.tail_risk(report_with([5, 3, 2, 1, 0, -1, -2, -3, -5.5]), confidence=.75).row(0, named=True)
    # ceil(.75*9)=7th highest; ETL averages -2, -3 and -5.5. Signs stay negative.
    assert (row["var_return"], row["etl_return"], row["n_tail"], row["n_obs"], row["status"]) == (-2, -3.5, 3, 9, "ok")
    eight = rt.tail_risk(report_with([8., 7, 6, 5, 4, 3, 2, 1]), confidence=.75).row(0, named=True)
    assert (eight["var_return"], eight["etl_return"], eight["n_tail"]) == (3, 2, 3)


def test_tail_risk_includes_every_tie():
    row = rt.tail_risk(report_with([.03, -.02, .01, -.02, -.02, -.05]), confidence=.5).row(0, named=True)
    # Rank 3 is -0.02, which also occupies ranks 4 and 5: all ties plus the -0.05 enter the tail.
    assert row["var_return"] == -.02 and row["n_tail"] == 4
    assert row["etl_return"] == pytest.approx((-.02*3-.05)/4)


def test_rank_uses_decimal_confidence():
    values = [float(v) for v in range(10, 0, -1)]
    table = rt.tail_risk(report_with(values), confidence=[.7, .1, .95])
    assert table["var_return"].to_list() == [4., 10., 1.]  # Ranks 7, 1 and 10; no float drift.
    assert table["confidence"].to_list() == [.7, .1, .95]


def test_dollar_tail_is_independent_when_nav_changes():
    # NAV 100 -> 90 -> 120 -> 110: the worst return (-10/100) and worst P&L tie (-10, -10) differ.
    report = rt.series_performance(pnl_table([-10., 30., -10.]), **KW)
    row = rt.tail_risk(report, confidence=.9).row(0, named=True)
    assert row["var_return"] == pytest.approx(-.1) and row["n_tail"] == 1
    assert row["var_pnl"] == -10 and row["etl_pnl"] == -10 and row["n_tail_pnl"] == 2
    assert row["var_pnl"] != pytest.approx(row["var_return"]*report.daily["equity"][-1])
    shifted = rt.series_performance(pnl_table([-10., 100., -12.]), **KW)
    row = rt.tail_risk(shifted, confidence=.9).row(0, named=True)
    # Worst return is interval 1 (-10/100); worst dollar P&L is interval 3 (-12 on NAV 190).
    assert row["var_return"] == pytest.approx(-.1) and row["var_pnl"] == -12


@pytest.mark.parametrize("confidence", [0, 1, -.1, 1.5, True, "0.9", [], [.5, 1.]])
def test_tail_risk_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        rt.tail_risk(report_with([1., 2.]), confidence=confidence)


def test_tail_risk_empty_and_wrong_input():
    row = rt.tail_risk(report_with([]), confidence=.9).row(0, named=True)
    assert row["status"] == "empty_sample" and row["n_obs"] == 0 and row["var_return"] is None
    with pytest.raises(ValueError, match="PerformanceResult"):
        rt.tail_risk(pnl_table([1., 1., 1.]), confidence=.9)
