"""Numerical reports with explicit sampling conventions and strict date joins."""

from copy import deepcopy
from fractions import Fraction
import math
from statistics import mean, stdev

import polars as pl

from ._data import _table
from ._results import BacktestResult, PerformanceResult, CorrelationResult
from ._returns import cumulative_returns
from ._risk_free import _risk_free, _aligned_returns

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
    scheduled = set(result.targets["session"].to_list()) if result.metadata.get("strategy") == "scheduled_rebalance" else set()
    reinvested = set(result.dividend_reinvestments.filter(
        pl.col("status").is_in(["reinvested", "stopped_before_trade", "payer_now_short"]))["session"].to_list())
    for d in daily["session"]:
        if d in scheduled:
            expected_keys.append((d, "pre_rebalance"))
        elif d in reinvested:
            expected_keys.append((d, "pre_reinvestment"))
        expected_keys.append((d, "close"))
    if vals.select("session", "phase").rows() != expected_keys:
        raise ValueError("valuations must include ordered pre/post-entry and every daily close")
    if not math.isclose(vals["equity"][0], capital) or vals["equity"][1] <= 0:
        raise ValueError("invalid entry valuations")
    if any(not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)
           for a, b in zip(vals.filter(pl.col("phase") == "close")["equity"], daily["equity"])):
        raise ValueError("valuations disagree with daily equity")
    return daily, vals, deepcopy(meta)


def _summary(r, rf, a, mar_period, denominator, currency, ending_equity, pnl, compounded, drawdown):
    n, root = len(r), math.sqrt(a)
    excess = [x-y for x, y in zip(r, rf)]
    vol = stdev(r) if n >= 2 else None
    sharpe_vol = stdev(excess) if n >= 2 and denominator == "excess_returns" else vol
    downside = math.sqrt(mean(min(x-mar_period, 0)**2 for x in r))
    rows = [("ending_equity", ending_equity, currency, n, "ok"), ("cumulative_pnl", pnl, currency, n, "ok"),
        ("cumulative_simple_return", math.fsum(r), "fraction", n, "ok"),
        ("compounded_return", compounded, "fraction", n, "ok"),
        ("annualized_arithmetic_mean", mean(r)*a, "fraction/year", n, "ok"),
        ("max_drawdown", drawdown, "fraction", n, "ok"),
        ("annualized_volatility", vol*root if vol is not None else None, "fraction/sqrt(year)", n,
         "ok" if n >= 2 else "insufficient_samples"),
        ("sharpe", root*mean(excess)/sharpe_vol if sharpe_vol else None, "ratio", n,
         "insufficient_samples" if n < 2 else ("zero_excess_volatility" if denominator == "excess_returns" else "zero_volatility")
         if not sharpe_vol else "ok"),
        ("sortino", root*mean(x-mar_period for x in r)/downside if downside and n >= 2 else None, "ratio", n,
         "insufficient_samples" if n < 2 else "zero_downside_risk" if not downside else "ok")]
    if any(v is not None and not math.isfinite(v) for _, v, _, _, _ in rows):
        raise ValueError("performance metric is not representable")
    return pl.DataFrame(rows, schema=METRIC_SCHEMA, orient="row")


def _cumulative_rows(start, dates, returns, series):
    rows, summed, wealth = [(start, series, 0., 0., 1.)], 0., 1.
    for day, value in zip(dates, returns):
        summed += value
        wealth *= 1+value
        if not math.isfinite(summed) or not math.isfinite(wealth):
            raise ValueError("cumulative returns are not representable")
        rows.append((day, series, summed, wealth-1, wealth))
    return rows


def _benchmark_report(daily, r, rf, a, mar_period, denominator, currency, capital, benchmark, metadata, frequency="1d"):
    """Strictly aligned benchmark comparison shared by ledger and P&L-series reports."""
    n, rows = len(r), []
    empty_series = pl.DataFrame(schema={"session": pl.Date, "equity": pl.Float64})
    if benchmark is None:
        if metadata is not None:
            raise ValueError("benchmark_metadata requires a benchmark")
        return (pl.DataFrame(schema=METRIC_SCHEMA), pl.DataFrame(schema=METRIC_SCHEMA), [], empty_series, None)

    def add(name, value, unit, status="ok"):
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{name} is not representable")
        rows.append((name, value, unit, n, status))

    b, bm_meta = _aligned_returns(benchmark, daily.select("period_start", "session"), metadata, currency,
                                  "benchmark", frequency)
    br = b["simple_return"].to_list()
    corr, beta, corr_status, beta_status = _pair(r, br)
    add("return_correlation", corr, "correlation", corr_status)
    add("beta", beta, "ratio", beta_status)
    active = [x-y for x, y in zip(r, br)]
    tracking = stdev(active) if n >= 2 else None
    add("annualized_tracking_error", tracking*math.sqrt(a) if tracking is not None else None,
        "fraction/sqrt(year)", "ok" if n >= 2 else "insufficient_samples")
    add("information_ratio", math.sqrt(a)*mean(active)/tracking if tracking else None, "ratio",
        "insufficient_samples" if n < 2 else "zero_tracking_error" if not tracking else "ok")
    wealth = [capital]
    for value in br:
        wealth.append(wealth[-1]*(1+value))
        if not math.isfinite(wealth[-1]) or wealth[-1] <= 0:
            raise ValueError("benchmark wealth is not representable")
    peak_b, drawdowns_b = wealth[0], []
    for value in wealth:
        peak_b = max(peak_b, value)
        drawdowns_b.append(value/peak_b-1)
    benchmark_summary = _summary(br, rf, a, mar_period, denominator, currency,
        wealth[-1], wealth[-1]-wealth[0], wealth[-1]/wealth[0]-1, min(drawdowns_b))
    self_corr, self_beta, self_corr_status, self_beta_status = _pair(br, br)
    benchmark_summary = pl.concat([benchmark_summary, pl.DataFrame([
        ("return_correlation", self_corr, "correlation", n, self_corr_status),
        ("beta", self_beta, "ratio", n, self_beta_status),
        ("annualized_tracking_error", 0. if n >= 2 else None, "fraction/sqrt(year)", n,
            "ok" if n >= 2 else "insufficient_samples"),
        ("information_ratio", None, "ratio", n,
            "zero_tracking_error" if n >= 2 else "insufficient_samples"),
        ("entry_cost", 0., currency, n, "not_modeled"),
        ("financing_cost", 0., currency, n, "not_modeled")], schema=METRIC_SCHEMA, orient="row")])
    cumulative = _cumulative_rows(daily["period_start"][0], daily["session"].to_list(), br, "benchmark")
    total = wealth[-1]/wealth[0]-1
    add("benchmark_compounded_return", total, "fraction")
    add("compounded_return_difference", 100*(daily["compounded_return"][-1]-total), "percentage_points")
    add("relative_wealth_return", daily["equity"][-1]/wealth[-1]-1, "fraction")
    series = pl.DataFrame({"session": [daily["period_start"][0]]+daily["session"].to_list(), "equity": wealth})
    return pl.DataFrame(rows, schema=METRIC_SCHEMA, orient="row"), benchmark_summary, cumulative, series, bm_meta


def performance(result, *, periods_per_year, risk_free_annual_effective=None,
                minimum_acceptable_return_annual_effective, benchmark=None,
                benchmark_metadata=None, alignment="strict", allow_partial=False,
                risk_free_returns=None, risk_free_metadata=None,
                sharpe_denominator="portfolio_returns") -> PerformanceResult:
    """Prepare net-return metrics and chart tables without I/O.

    Benchmark is a strictly matched Polars interval table; its metadata requires
    source, basis, currency and frequency. Stopped runs require allow_partial=True
    and a benchmark explicitly sliced to the actual interval endpoints. Supply
    exactly one scalar annual RF rate or dated interval RF returns. Sharpe uses
    portfolio volatility by default; excess_returns explicitly changes its denominator.
    """
    daily, vals, meta = _validated_run(result, allow_partial)
    a = _number(periods_per_year, "periods_per_year", lower=0)
    rf = risk_free_annual_effective
    rf_table, rf_meta, rf_period = _risk_free(daily, meta["currency"], rf, risk_free_returns, risk_free_metadata, a)
    if sharpe_denominator not in {"portfolio_returns", "excess_returns"}:
        raise ValueError("sharpe_denominator must be portfolio_returns or excess_returns")
    mar = _number(minimum_acceptable_return_annual_effective, "minimum_acceptable_return_annual_effective", lower=-1)
    if alignment != "strict":
        raise ValueError("only alignment='strict' is supported")
    try:
        mar_period = math.expm1(math.log1p(mar)/a)
    except OverflowError as exc:
        raise ValueError("periodic hurdle is not representable") from exc
    r = daily["simple_return"].to_list()
    n = len(r)
    peak, dd = 0.0, []
    for row in vals.iter_rows(named=True):
        peak = max(peak, row["equity"])
        dd.append(row["equity"]/peak-1)
    drawdowns = vals.with_columns(pl.Series("drawdown", dd, dtype=pl.Float64))
    currency = meta["currency"]
    summary = _summary(r, rf_table["simple_return"].to_list(), a, mar_period, sharpe_denominator,
        currency, daily["equity"][-1], daily["cumulative_pnl"][-1], daily["compounded_return"][-1], min(dd))
    extra = [("entry_cost", result.trades.filter(pl.col("execution") == "entry_close")["trade_cost"].sum(), currency, n, "ok"),
             ("financing_cost", result.costs.filter(pl.col("component") == "borrowing_interest")["amount"].sum(), currency, n, "ok"),
             ("ending_leverage", daily["gross_leverage"][-1], "ratio", n,
              "ok" if daily["gross_leverage"][-1] is not None else "nonpositive_equity")]
    summary = pl.concat([summary, pl.DataFrame(extra, schema=METRIC_SCHEMA, orient="row")])
    if meta.get("long_short") is not None:
        summary = pl.concat([summary, pl.DataFrame([
            ("stock_borrow_cost", result.costs.filter(pl.col("component") == "stock_borrow_fee")["amount"].sum(), currency, n, "ok"),
            ("short_dividend_expense", -result.attribution.filter(pl.col("component") == "short_dividend_expense")["pnl"].sum(), currency, n, "ok"),
            ("short_collateral_rebate", result.short_financing_accruals["rebate"].sum(), currency, n, "ok"),
        ], schema=METRIC_SCHEMA, orient="row")])
    cumulative_rows = _cumulative_rows(daily["period_start"][0], daily["session"].to_list(), r, "portfolio")
    comparison, benchmark_summary, benchmark_rows, benchmark_series, bm_meta = _benchmark_report(
        daily, r, rf_table["simple_return"].to_list(), a, mar_period, sharpe_denominator, currency,
        meta["initial_capital"], benchmark, benchmark_metadata)
    cumulative_rows += benchmark_rows
    # Prepare chart tables once. Position weights include leverage; cash, debt and
    # receivables complete the balance sheet. No hidden normalization in plots.
    allocation = result.positions.select("session", (pl.lit("asset:")+pl.col("asset")).alias("component"), "weight")
    balances = result.valuations.filter(pl.col("phase").is_in(["post_entry", "close"]))
    account_columns = ["cash", "debt", "dividend_receivable"]
    if meta.get("long_short") is not None:
        account_columns += ["restricted_collateral", "dividend_liability"]
    for col in account_columns:
        allocation = pl.concat([allocation, balances.select("session", pl.lit(f"account:{col}").alias("component"),
            pl.when(pl.col("equity") > 0).then(pl.col(col)/pl.col("equity")*(-1 if col in {"debt", "dividend_liability"} else 1))
            .otherwise(None).alias("weight"))])
    attribution = result.attribution.group_by("component").agg(pl.col("pnl").sum()).sort("component")
    if not math.isclose(attribution["pnl"].sum(), daily["pnl"].sum(), rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("dollar attribution does not reconcile")
    meta.update(periods_per_year=a, risk_free_annual_effective=rf, risk_free_periodic=rf_period,
                risk_free=rf_meta, sharpe_denominator=sharpe_denominator,
                minimum_acceptable_return_annual_effective=mar, minimum_acceptable_return_periodic=mar_period,
                ddof=1, sortino_denominator="all_observations", alignment=alignment,
                annualization="sqrt_periods_per_year_no_serial_correlation_adjustment",
                n_obs=n, benchmark=bm_meta, allow_partial=allow_partial,
                information_ratio_convention="sqrt_periods_per_year_mean_active_return_over_sample_std_active_return",
                drawdown_basis=("pre_entry_post_entry_pre_trade_and_session_closes"
                    if meta.get("dividend_reinvestment") is not None else
                    "pre_entry_post_entry_pre_rebalance_and_session_closes"
                    if meta.get("strategy") == "scheduled_rebalance" else
                    "pre_entry_post_entry_and_session_closes"))
    return PerformanceResult(summary, comparison, benchmark_series, daily.clone(), vals.clone(),
                             drawdowns, allocation.sort("session", "component"), attribution, meta,
        pl.DataFrame(cumulative_rows, schema={"session": pl.Date, "series": pl.String,
            "cumulative_simple_return": pl.Float64, "compounded_return": pl.Float64, "wealth": pl.Float64}, orient="row"),
        benchmark_summary, rf_table)


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


TAIL_SCHEMA = {"confidence": pl.Float64, "var_return": pl.Float64, "etl_return": pl.Float64,
               "var_pnl": pl.Float64, "etl_pnl": pl.Float64, "n_obs": pl.Int64, "n_tail": pl.Int64,
               "n_tail_pnl": pl.Int64, "status": pl.String}


def _historical_tail(values, confidence):
    """Signed threshold at descending one-based rank ceil(c*n), and the mean at or below it."""
    ordered = sorted(values, reverse=True)
    rank = math.ceil(Fraction(str(confidence)) * len(ordered))
    threshold = ordered[rank-1]
    tail = [v for v in ordered if v <= threshold]
    return threshold, math.fsum(tail)/len(tail), len(tail)


def tail_risk(report, *, confidence) -> pl.DataFrame:
    """Historical VaR and expected tail loss of a report's daily net returns and dollar P&L.

    Values keep their sign: a negative VaR/ETL is a loss. The return and dollar
    rules are applied independently, so their tails can contain different
    intervals when NAV changes; n_tail and n_tail_pnl report each count.
    """
    if not isinstance(report, PerformanceResult):
        raise ValueError("tail_risk consumes a PerformanceResult")
    levels = [confidence] if isinstance(confidence, (int, float)) else confidence
    try:
        levels = list(levels)
    except TypeError as exc:
        raise ValueError("confidence must be a number or a sequence of numbers") from exc
    if not levels or any(isinstance(c, bool) or not isinstance(c, (int, float)) or not 0 < c < 1 for c in levels):
        raise ValueError("confidence must satisfy 0 < confidence < 1")
    returns, pnl = report.daily["simple_return"].to_list(), report.daily["pnl"].to_list()
    if any(v is None or not math.isfinite(v) for v in returns + pnl):
        raise ValueError("report returns and P&L must be finite")
    rows = []
    for c in levels:
        if not returns:
            rows.append((float(c), None, None, None, None, 0, 0, 0, "empty_sample"))
            continue
        var_r, etl_r, n_r = _historical_tail(returns, c)
        var_p, etl_p, n_p = _historical_tail(pnl, c)
        rows.append((float(c), var_r, etl_r, var_p, etl_p, len(returns), n_r, n_p, "ok"))
    return pl.DataFrame(rows, schema=TAIL_SCHEMA, orient="row")
