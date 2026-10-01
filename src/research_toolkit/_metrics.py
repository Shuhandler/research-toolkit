"""Numerical reports with explicit sampling conventions and strict date joins."""

from copy import deepcopy
import math
from statistics import mean, stdev

import polars as pl

from ._data import _table
from ._results import BacktestResult, PerformanceResult, CorrelationResult
from ._returns import cumulative_returns

BENCHMARK_SCHEMA = {"period_start": pl.Date, "session": pl.Date, "simple_return": pl.Float64}
METRIC_SCHEMA = {"metric": pl.String, "value": pl.Float64, "unit": pl.String,
                 "n_obs": pl.Int64, "status": pl.String}


def _number(value, name, *, lower):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= lower:
        raise ValueError(f"{name} must be finite and greater than {lower}")
    return float(value)


def _pair(x, y):
    """Centered sums; correlation and regression slope share the same sample."""
    if len(x) < 2:
        return None, None, "insufficient_samples", "insufficient_samples"
    mx, my = mean(x), mean(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    xx, yy = math.fsum(v*v for v in dx), math.fsum(v*v for v in dy)
    xy = math.fsum(a*b for a, b in zip(dx, dy))
    if not all(math.isfinite(v) for v in (xx, yy, xy)):
        raise ValueError("correlation covariance is not representable")
    corr = max(-1.0, min(1.0, xy / math.sqrt(xx) / math.sqrt(yy))) if xx and yy else None
    return corr, xy / yy if yy else None, "ok" if xx and yy else "zero_volatility", "ok" if yy else "zero_benchmark_variance"


def _validated_run(result, allow_partial):
    if not isinstance(result, BacktestResult):
        raise ValueError("expected BacktestResult")
    if type(allow_partial) is not bool:
        raise ValueError("allow_partial must be boolean")
    if not allow_partial:
        result.require_complete()
    if result.status not in {"complete", "stopped"}:
        raise ValueError("unknown backtest status")
    meta = result.metadata
    daily = result.daily.sort("session")
    required = {"period_start": pl.Date, "session": pl.Date, **{k: pl.Float64 for k in (
        "opening_equity", "equity", "pnl", "simple_return", "cumulative_pnl",
        "cumulative_simple_return", "compounded_return", "cash", "debt", "dividend_receivable")}}
    _table(daily.select(list(required)), required, "daily", ["session"], nonempty=True)
    if meta.get("frequency") != "1d" or meta.get("return_basis") != "net_equity":
        raise ValueError("performance requires daily net-equity returns")
    capital = _number(meta.get("initial_capital"), "initial_capital", lower=0)
    start, end = meta["entry_session"], meta["actual_end_session"]
    if (daily["period_start"][0].isoformat() != start or daily["session"][-1].isoformat() != end
            or meta.get("status") != result.status
            or (result.status == "complete" and end != meta["end_session"])
            or (result.status == "stopped" and (not result.stop_reason or result.stop_session != daily["session"][-1]))):
        raise ValueError("run coverage/status metadata disagrees with daily records")
    previous, previous_date, accumulated, wealth = capital, None, 0.0, 1.0
    for row in daily.iter_rows(named=True):
        if row["period_start"] >= row["session"] or (previous_date and row["period_start"] != previous_date):
            raise ValueError("daily intervals must be contiguous and increasing")
        if previous <= 0:
            raise ValueError("return denominator must be positive")
        r = row["simple_return"]
        accumulated += r
        wealth *= 1 + r
        checks = [(row["opening_equity"], previous), (row["pnl"], row["equity"]-previous),
                  (r, row["pnl"]/previous), (row["cumulative_pnl"], row["equity"]-capital),
                  (row["cumulative_simple_return"], accumulated),
                  (row["compounded_return"], wealth-1), (row["equity"], capital*wealth)]
        if any(not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8) for a, b in checks):
            raise ValueError("daily P&L/return records do not reconcile")
        previous, previous_date = row["equity"], row["session"]
    vals = result.valuations.select("session", "phase", "equity")
    _table(vals, {"session": pl.Date, "phase": pl.String, "equity": pl.Float64},
           "valuations", ["session", "phase"], nonempty=True)
    expected_keys = [(daily["period_start"][0], "pre_entry"), (daily["period_start"][0], "post_entry")]
    expected_keys += [(d, "close") for d in daily["session"]]
    if vals.select("session", "phase").rows() != expected_keys:
        raise ValueError("valuations must include ordered pre/post-entry and every daily close")
    if not math.isclose(vals["equity"][0], capital) or vals["equity"][1] <= 0:
        raise ValueError("invalid entry valuations")
    if any(not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)
           for a, b in zip(vals["equity"].to_list()[2:], daily["equity"])):
        raise ValueError("valuations disagree with daily equity")
    return daily, vals, deepcopy(meta)


def performance(result, *, periods_per_year, risk_free_annual_effective,
                minimum_acceptable_return_annual_effective, benchmark=None,
                benchmark_metadata=None, alignment="strict", allow_partial=False) -> PerformanceResult:
    """Prepare net-return metrics and chart tables without I/O.

    Benchmark is a strictly matched Polars interval table; its metadata requires
    source, basis, currency and frequency. Stopped runs require allow_partial=True
    and a benchmark explicitly sliced to the actual interval endpoints.
    """
    daily, vals, meta = _validated_run(result, allow_partial)
    a = _number(periods_per_year, "periods_per_year", lower=0)
    rf = _number(risk_free_annual_effective, "risk_free_annual_effective", lower=-1)
    mar = _number(minimum_acceptable_return_annual_effective, "minimum_acceptable_return_annual_effective", lower=-1)
    if alignment != "strict":
        raise ValueError("only alignment='strict' is supported")
    try:
        rf_period, mar_period = math.expm1(math.log1p(rf)/a), math.expm1(math.log1p(mar)/a)
    except OverflowError as exc:
        raise ValueError("periodic hurdle is not representable") from exc
    r = daily["simple_return"].to_list()
    n, root = len(r), math.sqrt(a)
    rows = []

    def add(name, value, unit, status="ok", count=n):
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{name} is not representable")
        rows.append((name, value, unit, count, status))

    peak, dd = 0.0, []
    for row in vals.iter_rows(named=True):
        peak = max(peak, row["equity"])
        dd.append(row["equity"]/peak-1)
    drawdowns = vals.with_columns(pl.Series("drawdown", dd, dtype=pl.Float64))
    currency = meta["currency"]
    for name, value, unit in [
        ("ending_equity", daily["equity"][-1], currency),
        ("cumulative_pnl", daily["cumulative_pnl"][-1], currency),
        ("cumulative_simple_return", math.fsum(r), "fraction"),
        ("compounded_return", daily["compounded_return"][-1], "fraction"),
        ("annualized_arithmetic_mean", mean(r)*a, "fraction/year"),
        ("max_drawdown", min(dd), "fraction")]:
        add(name, value, unit)
    vol = stdev(r) if n >= 2 else None
    excess = [x-rf_period for x in r]
    downside = math.sqrt(mean(min(x-mar_period, 0)**2 for x in r))
    add("annualized_volatility", vol*root if vol is not None else None, "fraction/sqrt(year)",
        "ok" if n >= 2 else "insufficient_samples")
    add("sharpe", root*mean(excess)/vol if vol and n >= 2 else None, "ratio",
        "insufficient_samples" if n < 2 else "zero_volatility" if not vol else "ok")
    add("sortino", root*mean(x-mar_period for x in r)/downside if downside and n >= 2 else None, "ratio",
        "insufficient_samples" if n < 2 else "zero_downside_risk" if not downside else "ok")
    summary = pl.DataFrame(rows, schema=METRIC_SCHEMA, orient="row")
    rows = []
    benchmark_series = pl.DataFrame(schema={"session": pl.Date, "equity": pl.Float64})
    bm_meta = None
    if benchmark is None:
        if benchmark_metadata is not None:
            raise ValueError("benchmark_metadata requires a benchmark")
    else:
        b = _table(benchmark, BENCHMARK_SCHEMA, "benchmark", ["session"], nonempty=True).sort("session")
        if not b.select("period_start", "session").equals(daily.select("period_start", "session")):
            raise ValueError("benchmark intervals must match both endpoints exactly (strict alignment)")
        if (b["simple_return"] <= -1).any():
            raise ValueError("benchmark returns must preserve positive wealth")
        if not isinstance(benchmark_metadata, dict) or any(
            not isinstance(benchmark_metadata.get(k), str) or not benchmark_metadata[k].strip()
            for k in ("source", "basis", "currency", "frequency")):
            raise ValueError("benchmark_metadata requires source, basis, currency and frequency")
        if benchmark_metadata["currency"] != currency or benchmark_metadata["frequency"] != "1d":
            raise ValueError("benchmark currency/frequency must match the portfolio")
        bm_meta = deepcopy(benchmark_metadata)
        br = b["simple_return"].to_list()
        corr, beta, corr_status, beta_status = _pair(r, br)
        add("return_correlation", corr, "correlation", corr_status)
        add("beta", beta, "ratio", beta_status)
        wealth = [meta["initial_capital"]]
        for value in br:
            wealth.append(wealth[-1]*(1+value))
            if not math.isfinite(wealth[-1]) or wealth[-1] <= 0:
                raise ValueError("benchmark wealth is not representable")
        total = wealth[-1]/wealth[0]-1
        add("benchmark_compounded_return", total, "fraction")
        add("compounded_return_difference", 100*(daily["compounded_return"][-1]-total), "percentage_points")
        add("relative_wealth_return", daily["equity"][-1]/wealth[-1]-1, "fraction")
        benchmark_series = pl.DataFrame({"session": [daily["period_start"][0]]+daily["session"].to_list(), "equity": wealth})
    comparison = pl.DataFrame(rows, schema=METRIC_SCHEMA, orient="row")
    # Prepare chart tables once. Position weights include leverage; cash, debt and
    # receivables complete the balance sheet. No hidden normalization in plots.
    allocation = result.positions.select("session", (pl.lit("asset:")+pl.col("asset")).alias("component"), "weight")
    balances = result.valuations.filter(pl.col("phase") != "pre_entry")
    for col in ("cash", "debt", "dividend_receivable"):
        allocation = pl.concat([allocation, balances.select("session", pl.lit(f"account:{col}").alias("component"),
            pl.when(pl.col("equity") > 0).then(pl.col(col)/pl.col("equity")*(-1 if col == "debt" else 1))
            .otherwise(None).alias("weight"))])
    attribution = result.attribution.group_by("component").agg(pl.col("pnl").sum()).sort("component")
    if not math.isclose(attribution["pnl"].sum(), daily["pnl"].sum(), rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("dollar attribution does not reconcile")
    meta.update(periods_per_year=a, risk_free_annual_effective=rf, risk_free_periodic=rf_period,
                minimum_acceptable_return_annual_effective=mar, minimum_acceptable_return_periodic=mar_period,
                ddof=1, sortino_denominator="all_observations", alignment=alignment,
                annualization="sqrt_periods_per_year_no_serial_correlation_adjustment",
                n_obs=n, benchmark=bm_meta, allow_partial=allow_partial,
                drawdown_basis="pre_entry_post_entry_and_session_closes")
    return PerformanceResult(summary, comparison, benchmark_series, daily.clone(), vals.clone(),
                             drawdowns, allocation.sort("session", "component"), attribution, meta)


def correlation(result) -> CorrelationResult:
    """Pearson asset correlations of a complete, explicitly based simple-return panel.

    Only the structural leading null per asset is excluded. Raw price returns
    retain split jumps, so their basis and diagnostics remain attached.
    """
    validated = cumulative_returns(result, method="sum")
    if validated.metadata["method"] != "simple":
        raise ValueError("correlation requires simple returns")
    assets = validated.metadata["assets"]
    series = {asset: validated.values.filter(pl.col("asset") == asset)["simple_return"].to_list()[1:]
              for asset in assets}
    rows = []
    for x in assets:
        for y in assets:
            value, _, status, _ = _pair(series[x], series[y])
            rows.append((x, y, value, len(series[x]), "empty_sample" if not series[x] else status))
    values = pl.DataFrame(rows, schema={"asset": pl.String, "other_asset": pl.String,
        "correlation": pl.Float64, "n_obs": pl.Int64, "status": pl.String}, orient="row")
    return CorrelationResult(values, deepcopy(result.metadata), result.diagnostics.clone())
