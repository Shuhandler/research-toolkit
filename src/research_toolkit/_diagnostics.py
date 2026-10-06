"""CAGR, Calmar and bias-corrected higher moments of an existing performance report."""

from copy import deepcopy
import math

import polars as pl

from ._data import _table
from ._metrics import METRIC_SCHEMA
from ._portfolio import _finite_above
from ._results import PerformanceDiagnostics, PerformanceResult
from ._risk_free import _intervals

DAILY_SCHEMA = {"period_start": pl.Date, "session": pl.Date, "simple_return": pl.Float64,
                "equity": pl.Float64, "compounded_return": pl.Float64}
ANNUALIZATIONS = {"calendar_time", "trading_periods"}


def _moments(r):
    """Adjusted Fisher-Pearson skewness G1 and bias-corrected excess kurtosis G2."""
    n = len(r)
    skew = kurt = (None, "insufficient_samples")
    if n < 3:
        return skew, kurt
    if len(set(r)) == 1:
        return (None, "zero_variance"), (None, "zero_variance") if n >= 4 else kurt
    mu = math.fsum(r)/n
    d = [x - mu for x in r]
    m2 = math.fsum(v*v for v in d)/n
    m3 = math.fsum(v**3 for v in d)/n
    m4 = math.fsum(v**4 for v in d)/n
    skew = (math.sqrt(n*(n-1))/(n-2) * m3/m2**1.5, "ok")
    if n >= 4:
        kurt = ((n-1)/((n-2)*(n-3)) * ((n+1)*(m4/(m2*m2) - 3) + 6), "ok")
    return skew, kurt


def _cagr(wealth, years):
    if wealth <= 0:
        return None, "nonpositive_ending_wealth"
    try:
        value = math.expm1(math.log(wealth)/years)
    except OverflowError:
        return None, "not_representable"
    return (value, "ok") if math.isfinite(value) else (None, "not_representable")


def performance_diagnostics(report, *, annualization, periods_per_year=None, days_per_year=None,
                            allow_partial=False) -> PerformanceDiagnostics:
    """CAGR, Calmar ratio, skewness and excess kurtosis from a prepared report.

    ``annualization='calendar_time'`` raises the ending wealth multiple to
    ``days_per_year / elapsed_calendar_days``, where the elapsed interval runs from
    the first ``period_start`` (initial capital) to the last ``session``.
    ``'trading_periods'`` uses ``periods_per_year / holding_intervals``. Only the
    selected convention's input may be supplied. Nothing is recalculated from prices
    and no backtest is rerun. Undefined statistics are null with a status.
    """
    if not isinstance(report, PerformanceResult):
        raise ValueError("performance_diagnostics consumes a PerformanceResult")
    if type(allow_partial) is not bool:
        raise ValueError("allow_partial must be boolean")
    if annualization not in ANNUALIZATIONS:
        raise ValueError("annualization must be 'calendar_time' or 'trading_periods'")
    if annualization == "calendar_time":
        if periods_per_year is not None:
            raise ValueError("calendar_time annualization uses days_per_year, not periods_per_year")
        basis = _finite_above(days_per_year, "days_per_year", lower=0)
    else:
        if days_per_year is not None:
            raise ValueError("trading_periods annualization uses periods_per_year, not days_per_year")
        basis = _finite_above(periods_per_year, "periods_per_year", lower=0)
    meta = report.metadata
    status = meta.get("status")
    if status not in {"complete", "stopped"}:
        raise ValueError("report has an unknown run status")
    if status != "complete" and not allow_partial:
        raise ValueError(f"report covers a stopped run ({meta.get('stop_reason')}); pass allow_partial=True")
    capital = _finite_above(meta.get("initial_capital"), "report initial_capital", lower=0)
    if not isinstance(report.daily, pl.DataFrame) or set(DAILY_SCHEMA) - set(report.daily.columns):
        raise ValueError(f"report.daily must contain {list(DAILY_SCHEMA)}")
    daily = _table(report.daily.select(list(DAILY_SCHEMA)), DAILY_SCHEMA, "report.daily", ["session"])

    common = dict(annualization=annualization,
                  periods_per_year=basis if annualization == "trading_periods" else None,
                  days_per_year=basis if annualization == "calendar_time" else None,
                  report_periods_per_year=meta.get("periods_per_year"), initial_capital=capital,
                  run_status=status, stop_reason=meta.get("stop_reason"), allow_partial=allow_partial,
                  frequency=meta.get("frequency"), return_basis=meta.get("return_basis"),
                  currency=meta.get("currency"),
                  return_series="periodic simple net portfolio returns (report.daily.simple_return)",
                  cagr_definition=("(ending_equity/initial_capital)**(days_per_year/elapsed_calendar_days)-1"
                                   if annualization == "calendar_time" else
                                   "(ending_equity/initial_capital)**(periods_per_year/holding_intervals)-1"),
                  calmar_definition="cagr/abs(max_drawdown); max_drawdown from report.drawdowns",
                  skewness_definition="adjusted Fisher-Pearson G1 = sqrt(n(n-1))/(n-2)*m3/m2**1.5; n>=3",
                  excess_kurtosis_definition="G2 = (n-1)/((n-2)(n-3))*((n+1)*(m4/m2**2-3)+6); n>=4; normal=0",
                  reliability="descriptive sample values; no minimum history imposed or significance implied",
                  drawdown_basis=meta.get("drawdown_basis"))
    if daily.is_empty():
        rows = [(name, None, unit, 0, "empty_sample") for name, unit in (
            ("holding_intervals", "intervals"), ("elapsed_calendar_days", "calendar_days"),
            ("elapsed_years", "years"), ("ending_wealth_multiple", "multiple"),
            ("compounded_return", "fraction"), ("cagr", "fraction/year"), ("max_drawdown", "fraction"),
            ("calmar", "ratio"), ("skewness", "dimensionless"), ("excess_kurtosis", "dimensionless"))]
        return PerformanceDiagnostics(pl.DataFrame(rows, schema=METRIC_SCHEMA, orient="row"),
                                      dict(common, n_obs=0, start=None, end=None))

    _intervals(daily.select("period_start", "session"))
    r = daily["simple_return"].to_list()
    n = len(r)
    wealth = 1.0
    for value in r:
        wealth *= 1 + value
    ending_multiple = daily["equity"][-1]/capital
    if not (math.isclose(wealth, ending_multiple, rel_tol=1e-9, abs_tol=1e-12)
            and math.isclose(daily["compounded_return"][-1], ending_multiple - 1, rel_tol=1e-9, abs_tol=1e-12)):
        raise ValueError("report returns, equity and compounded_return do not reconcile with initial capital")
    drawdowns = report.drawdowns
    if (not isinstance(drawdowns, pl.DataFrame) or {"equity", "drawdown"} - set(drawdowns.columns)
            or drawdowns.is_empty() or drawdowns.select("equity", "drawdown").null_count().sum_horizontal()[0]
            or not drawdowns["drawdown"].is_finite().all()):
        raise ValueError("report.drawdowns must contain finite equity and drawdown values")
    if not math.isclose(drawdowns["equity"][0], capital, rel_tol=1e-12):
        raise ValueError("report.drawdowns must start at initial capital")
    max_dd = drawdowns["drawdown"].min()
    reported = report.summary.filter(pl.col("metric") == "max_drawdown")["value"]
    if reported.len() != 1 or reported[0] is None or not math.isclose(reported[0], max_dd, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("report summary max_drawdown disagrees with report.drawdowns")

    start, end = daily["period_start"][0], daily["session"][-1]
    days = (end - start).days
    years = days/basis if annualization == "calendar_time" else n/basis
    cagr, cagr_status = _cagr(ending_multiple, years)
    if cagr is None:
        calmar = (None, cagr_status)
    elif max_dd == 0:
        calmar = (None, "zero_drawdown")
    else:
        calmar = (cagr/abs(max_dd), "ok")
    (skew, skew_status), (kurt, kurt_status) = _moments(r)
    rows = [("holding_intervals", float(n), "intervals", n, "ok"),
            ("elapsed_calendar_days", float(days), "calendar_days", n, "ok"),
            ("elapsed_years", years, "years", n, "ok"),
            ("ending_wealth_multiple", ending_multiple, "multiple", n, "ok"),
            ("compounded_return", ending_multiple - 1, "fraction", n, "ok"),
            ("cagr", cagr, "fraction/year", n, cagr_status),
            ("max_drawdown", max_dd, "fraction", n, "ok"),
            ("calmar", calmar[0], "ratio", n, calmar[1]),
            ("skewness", skew, "dimensionless", n, skew_status),
            ("excess_kurtosis", kurt, "dimensionless", n, kurt_status)]
    if any(v is not None and not math.isfinite(v) for _, v, *_ in rows):
        raise ValueError("diagnostic is not representable")
    return PerformanceDiagnostics(pl.DataFrame(rows, schema=METRIC_SCHEMA, orient="row"),
        dict(deepcopy(common), n_obs=n, start=start.isoformat(), end=end.isoformat(), elapsed_calendar_days=days,
             elapsed_years=years, cagr_exponent=1/years))
