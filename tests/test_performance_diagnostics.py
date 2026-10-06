from dataclasses import replace
from datetime import date
import math

import polars as pl
import pytest

import research_toolkit as rt
from test_metrics import metric
from test_series_performance import META

# Two one-year intervals (365 days each) so calendar-time CAGR is hand-checkable.
YEARS = [date(2021, 1, 1), date(2022, 1, 1), date(2023, 1, 1)]
CAL = dict(annualization="calendar_time", days_per_year=365)


def report(pnl, sessions=YEARS, capital=100.):
    daily = pl.DataFrame({"period_start": sessions[:-1], "session": sessions[1:], "pnl": pnl},
                         schema={"period_start": pl.Date, "session": pl.Date, "pnl": pl.Float64})
    return rt.series_performance(daily, initial_capital=capital, periods_per_year=1,
                                 risk_free_annual_effective=0., metadata=META)


def value(diag, name):
    return metric(diag.values, name)["value"]


def test_calendar_and_trading_period_cagr_by_hand():
    rep = report([10., 11.])  # 100 -> 110 -> 121 over 730 calendar days.
    cal = rt.performance_diagnostics(rep, **CAL)
    assert value(cal, "cagr") == pytest.approx(.1)
    assert value(cal, "elapsed_calendar_days") == 730 and value(cal, "elapsed_years") == pytest.approx(2)
    assert cal.metadata["annualization"] == "calendar_time" and cal.metadata["days_per_year"] == 365
    # 365.25-day years give a slightly lower rate for the same wealth: (1.21)**(365.25/730)-1.
    julian = rt.performance_diagnostics(rep, annualization="calendar_time", days_per_year=365.25)
    assert value(julian, "cagr") == pytest.approx(1.21**(365.25/730) - 1)
    # Twelve holding intervals per year makes the same two intervals 1/6 of a year.
    trading = rt.performance_diagnostics(rep, annualization="trading_periods", periods_per_year=12)
    assert value(trading, "cagr") == pytest.approx(1.21**6 - 1)
    assert value(trading, "holding_intervals") == 2 and trading.metadata["periods_per_year"] == 12
    # CAGR is not the arithmetic annualized mean (10% per interval -> 120% at 12/year).
    assert value(trading, "cagr") != pytest.approx(12*.1)


def test_calendar_cagr_starts_at_initial_capital_not_first_valuation(market, run):
    result = run(market)
    rep = rt.performance(result, periods_per_year=252, risk_free_annual_effective=0.,
                         minimum_acceptable_return_annual_effective=0.)
    diag = rt.performance_diagnostics(rep, **CAL)
    days = (result.daily["session"][-1] - result.daily["period_start"][0]).days
    wealth = result.daily["equity"][-1]/result.metadata["initial_capital"]
    assert value(diag, "elapsed_calendar_days") == days
    assert value(diag, "cagr") == pytest.approx(wealth**(365/days) - 1)
    assert value(diag, "max_drawdown") == metric(rep.summary, "max_drawdown")["value"]


def test_initial_capital_drawdown_and_calmar():
    # 100 -> 90 -> 120: the only drawdown is the first-interval loss from initial capital.
    diag = rt.performance_diagnostics(report([-10., 30.]), **CAL)
    assert value(diag, "max_drawdown") == pytest.approx(-.1)
    assert value(diag, "cagr") == pytest.approx(math.sqrt(1.2) - 1)
    assert value(diag, "calmar") == pytest.approx((math.sqrt(1.2) - 1)/.1)


def test_negative_cagr_and_zero_drawdown():
    negative = rt.performance_diagnostics(report([-19., 0.]), **CAL)  # 100 -> 81 -> 81.
    assert value(negative, "cagr") == pytest.approx(-.1)
    assert value(negative, "calmar") == pytest.approx(-.1/.19)
    rising = rt.performance_diagnostics(report([10., 11.]), **CAL)
    assert value(rising, "max_drawdown") == 0
    row = metric(rising.values, "calmar")
    assert row["value"] is None and row["status"] == "zero_drawdown"


def test_bias_corrected_skewness_and_excess_kurtosis_by_hand():
    # Returns 0, 0, 0, 1: G1 = 2 and G2 = 4 (the spreadsheet SKEW/KURT values).
    sessions = [date(2024, 1, d) for d in (2, 3, 4, 5, 8)]
    diag = rt.performance_diagnostics(report([0., 0., 0., 100.], sessions), **CAL)
    assert value(diag, "skewness") == pytest.approx(2)
    assert value(diag, "excess_kurtosis") == pytest.approx(4)


def test_moments_match_pandas_reference():
    pd = pytest.importorskip("pandas")
    pnl = [3., -7., 12., 1., -4., 9., -15., 6., 2.]
    sessions = [date(2024, 1, 1 + i) for i in range(len(pnl) + 1)]
    rep = report(pnl, sessions, capital=1_000.)
    diag = rt.performance_diagnostics(rep, annualization="trading_periods", periods_per_year=252)
    returns = pd.Series(rep.daily["simple_return"].to_list())
    assert value(diag, "skewness") == pytest.approx(returns.skew(), rel=1e-12)
    assert value(diag, "excess_kurtosis") == pytest.approx(returns.kurt(), rel=1e-12)


def test_degenerate_samples_have_statuses():
    two = rt.performance_diagnostics(report([10., 11.]), **CAL)
    assert {metric(two.values, m)["status"] for m in ("skewness", "excess_kurtosis")} == {"insufficient_samples"}
    sessions = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    three = rt.performance_diagnostics(report([1., 2., 4.], sessions), **CAL)
    assert metric(three.values, "skewness")["status"] == "ok"
    assert metric(three.values, "excess_kurtosis")["status"] == "insufficient_samples"
    flat = rt.performance_diagnostics(report([0.]*4, sessions + [date(2024, 1, 8)]), **CAL)
    assert value(flat, "cagr") == 0 and metric(flat.values, "calmar")["status"] == "zero_drawdown"
    assert {metric(flat.values, m)["status"] for m in ("skewness", "excess_kurtosis")} == {"zero_variance"}
    # A one-interval sample is valid: values are reported with their tiny sample size.
    short = rt.performance_diagnostics(report([1.], [date(2024, 1, 2), date(2024, 1, 3)]), **CAL)
    assert value(short, "cagr") == pytest.approx(1.01**365 - 1) and metric(short.values, "cagr")["n_obs"] == 1


def test_nonpositive_ending_wealth_is_undefined_not_an_error():
    rep = report([10., 11.])
    capital = 100.
    daily = rep.daily.with_columns(pl.Series("simple_return", [.1, -1.5]), pl.Series("equity", [110., -55.]),
                                   pl.Series("compounded_return", [.1, -1.55]))
    drawdowns = rep.drawdowns.with_columns(pl.Series("equity", [capital, 110., -55.]),
                                           pl.Series("drawdown", [0., 0., -1.5]))
    summary = rep.summary.with_columns(pl.when(pl.col("metric") == "max_drawdown").then(-1.5)
                                       .otherwise(pl.col("value")).alias("value"))
    stopped = replace(rep, daily=daily, drawdowns=drawdowns, summary=summary,
                      metadata={**rep.metadata, "status": "stopped", "stop_reason": "margin"})
    with pytest.raises(ValueError, match="allow_partial"):
        rt.performance_diagnostics(stopped, **CAL)
    diag = rt.performance_diagnostics(stopped, allow_partial=True, **CAL)
    for name in ("cagr", "calmar"):
        row = metric(diag.values, name)
        assert row["value"] is None and row["status"] == "nonpositive_ending_wealth"
    assert diag.metadata["run_status"] == "stopped"


def test_malformed_inputs_raise():
    rep = report([10., 11.])
    with pytest.raises(ValueError, match="PerformanceResult"):
        rt.performance_diagnostics(rep.daily, **CAL)
    with pytest.raises(ValueError, match="annualization"):
        rt.performance_diagnostics(rep, annualization="arithmetic", periods_per_year=1)
    with pytest.raises(ValueError, match="days_per_year"):
        rt.performance_diagnostics(rep, annualization="calendar_time")
    with pytest.raises(ValueError, match="not periods_per_year"):
        rt.performance_diagnostics(rep, annualization="calendar_time", days_per_year=365, periods_per_year=252)
    with pytest.raises(ValueError, match="not days_per_year"):
        rt.performance_diagnostics(rep, annualization="trading_periods", periods_per_year=252, days_per_year=365)
    with pytest.raises(ValueError, match="periods_per_year"):
        rt.performance_diagnostics(rep, annualization="trading_periods", periods_per_year=0)
    tampered = replace(rep, daily=rep.daily.with_columns(pl.Series("equity", [110., 125.])))
    with pytest.raises(ValueError, match="reconcile"):
        rt.performance_diagnostics(tampered, **CAL)
    gap = replace(rep, daily=rep.daily.with_columns(pl.Series("period_start", [date(2021, 1, 1), date(2022, 1, 2)])))
    with pytest.raises(ValueError, match="contiguous"):
        rt.performance_diagnostics(gap, **CAL)
    with pytest.raises(ValueError, match="initial capital"):
        rt.performance_diagnostics(replace(rep, drawdowns=rep.drawdowns.slice(1)), **CAL)
