"""Performance reports for an external daily net dollar P&L series, without a ledger."""

import json
import math

import polars as pl

from ._calendars import _reporting_sessions
from ._data import _table
from ._metrics import _benchmark_report, _cumulative_rows, _summary
from ._portfolio import _finite_above
from ._results import PerformanceResult
from ._risk_free import _risk_free

PNL_SCHEMA = {"period_start": pl.Date, "session": pl.Date, "pnl": pl.Float64}
RESERVED = {"status", "stop_reason", "entry_session", "actual_end_session", "end_session",
            "return_basis", "initial_capital"}


def _without_mar(table):
    return table.with_columns(
        pl.when(pl.col("metric") == "sortino").then(None).otherwise(pl.col("value")).alias("value"),
        pl.when(pl.col("metric") == "sortino").then(pl.lit("mar_not_supplied")).otherwise(pl.col("status")).alias("status"))


def series_performance(daily, *, initial_capital, periods_per_year, risk_free_annual_effective,
                       benchmark=None, benchmark_metadata=None, metadata,
                       minimum_acceptable_return_annual_effective=None,
                       sharpe_denominator="portfolio_returns", calendar=None,
                       frequency="1d") -> PerformanceResult:
    """Report a supplied net dollar P&L series against an explicit starting NAV.

    NAV is initial capital plus cumulative P&L, with no external flows. Each simple
    return is interval P&L divided by the previous NAV. Intervals must be ordered and
    contiguous (each period_start equals the previous session). frequency='1d' means
    one trading session per interval; 'multi_session' declares intervals that may span
    several sessions. Without ``calendar`` only continuity is checked, so a skipped
    trading session cannot be detected. With one reporting calendar ('XNYS' or a
    TradingCalendar), every endpoint must be a session and '1d' intervals must not
    skip sessions.
    """
    capital = _finite_above(initial_capital, "initial_capital", lower=0)
    a = _finite_above(periods_per_year, "periods_per_year", lower=0)
    if not isinstance(daily, pl.DataFrame) or set(PNL_SCHEMA) - set(daily.columns):
        raise ValueError(f"daily must be a Polars DataFrame with columns {list(PNL_SCHEMA)}")
    duplicated = daily.filter(pl.col("session").is_duplicated())["session"].unique().sort().to_list()
    if duplicated:
        raise ValueError(f"duplicate sessions: {[str(d) for d in duplicated]}")
    daily = _table(daily.select(list(PNL_SCHEMA)), PNL_SCHEMA, "daily", ["session"], nonempty=True)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("currency"), str) or not metadata["currency"].strip():
        raise ValueError("metadata requires a nonblank currency")
    if frequency not in {"1d", "multi_session"}:
        raise ValueError("frequency must be '1d' (one session per interval) or 'multi_session'")
    if metadata.get("frequency", frequency) != frequency:
        raise ValueError(f"metadata.frequency must match frequency={frequency!r}")
    if RESERVED & set(metadata):
        raise ValueError(f"metadata must not set report fields {sorted(RESERVED & set(metadata))}")
    try:
        source = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be finite JSON") from exc
    if sharpe_denominator not in {"portfolio_returns", "excess_returns"}:
        raise ValueError("sharpe_denominator must be portfolio_returns or excess_returns")

    checked = _check_sessions(daily, calendar, frequency)
    nav, accumulated, wealth = capital, 0.0, 1.0
    rows = []
    for start, session, pnl in daily.iter_rows():
        opening, nav = nav, nav + pnl
        if nav <= 0 or not math.isfinite(nav):
            raise ValueError(f"NAV must stay positive and finite; got {nav} at {session}")
        r = pnl / opening
        accumulated += r
        wealth *= 1 + r
        rows.append((start, session, opening, pnl, nav, r, nav - capital, accumulated, wealth - 1))
    daily = pl.DataFrame(rows, schema={"period_start": pl.Date, "session": pl.Date,
        **{k: pl.Float64 for k in ("opening_equity", "pnl", "equity", "simple_return", "cumulative_pnl",
                                   "cumulative_simple_return", "compounded_return")}}, orient="row")

    currency = metadata["currency"]
    rf_table, _, rf_period = _risk_free(daily, currency, risk_free_annual_effective, None, None, a, frequency)
    mar = None
    mar_period = 0.0
    if minimum_acceptable_return_annual_effective is not None:
        mar = _finite_above(minimum_acceptable_return_annual_effective, "minimum_acceptable_return_annual_effective", lower=-1)
        try:
            mar_period = math.expm1(math.log1p(mar)/a)
        except OverflowError as exc:
            raise ValueError("periodic hurdle is not representable") from exc

    # Initial capital is the first observation, so a first-interval loss is a drawdown.
    equity = pl.DataFrame({"session": [daily["period_start"][0]] + daily["session"].to_list(),
                           "phase": ["initial_capital"] + ["close"]*daily.height,
                           "equity": [capital] + daily["equity"].to_list()})
    peak, dd = 0.0, []
    for value in equity["equity"]:
        peak = max(peak, value)
        dd.append(value/peak - 1)
    drawdowns = equity.with_columns(pl.Series("drawdown", dd, dtype=pl.Float64))

    r, rf = daily["simple_return"].to_list(), rf_table["simple_return"].to_list()
    summary = _summary(r, rf, a, mar_period, sharpe_denominator, currency, daily["equity"][-1],
                       daily["cumulative_pnl"][-1], daily["compounded_return"][-1], min(dd))
    comparison, benchmark_summary, benchmark_rows, benchmark_series, bm_meta = _benchmark_report(
        daily, r, rf, a, mar_period, sharpe_denominator, currency, capital, benchmark, benchmark_metadata, frequency)
    if mar is None:
        summary, benchmark_summary = _without_mar(summary), _without_mar(benchmark_summary)
    cumulative = pl.DataFrame(
        _cumulative_rows(daily["period_start"][0], daily["session"].to_list(), r, "portfolio") + benchmark_rows,
        schema={"session": pl.Date, "series": pl.String, "cumulative_simple_return": pl.Float64,
                "compounded_return": pl.Float64, "wealth": pl.Float64}, orient="row")

    meta = dict(source_metadata=source, currency=currency, frequency=frequency, return_basis="net_equity",
        nav_construction="initial_capital_plus_cumulative_pnl_no_external_flows",
        return_definition="interval_pnl_over_previous_nav", initial_capital=capital,
        status="complete", stop_reason=None,
        entry_session=daily["period_start"][0].isoformat(), actual_end_session=daily["session"][-1].isoformat(),
        end_session=daily["session"][-1].isoformat(),
        interval_validation=("ordered_contiguous_exchange_calendar_sessions" if calendar is not None
                             else "ordered_contiguous_no_trading_calendar"),
        calendar_validation=checked,
        periods_per_year=a, risk_free_annual_effective=float(risk_free_annual_effective),
        risk_free_periodic=rf_period, risk_free=None, sharpe_denominator=sharpe_denominator,
        minimum_acceptable_return_annual_effective=mar,
        minimum_acceptable_return_periodic=mar_period if mar is not None else None,
        ddof=1, sortino_denominator="all_observations", alignment="strict",
        annualization="sqrt_periods_per_year_no_serial_correlation_adjustment",
        n_obs=daily.height, benchmark=bm_meta, allow_partial=False,
        information_ratio_convention="sqrt_periods_per_year_mean_active_return_over_sample_std_active_return",
        drawdown_basis="initial_capital_and_session_closes")
    return PerformanceResult(summary, comparison, benchmark_series, daily, equity, drawdowns,
        pl.DataFrame(schema={"session": pl.Date, "component": pl.String, "weight": pl.Float64}),
        pl.DataFrame(schema={"component": pl.String, "pnl": pl.Float64}), meta, cumulative,
        benchmark_summary, rf_table)


def _check_sessions(daily, calendar, frequency):
    """Validate order/continuity, then (optionally) every endpoint against one calendar."""
    previous = None
    for start, session in daily.select("period_start", "session").iter_rows():
        if start >= session:
            raise ValueError(f"interval ending {session} must start before it ends")
        if previous is not None and session <= previous:
            raise ValueError("daily intervals must be unique and in increasing session order")
        if previous is not None and start < previous:
            raise ValueError(f"interval {start} to {session} overlaps the previous interval ending {previous}")
        if previous is not None and start > previous:
            raise ValueError(f"interval ending {session} starts {start}, not at the previous session "
                             f"{previous}; intervals must be contiguous (no observation covers {previous} to {start})")
        previous = session
    first, last = daily["period_start"][0], daily["session"][-1]
    if calendar is None:
        return {"status": "not_validated", "calendar": None, "frequency": frequency,
                "limitation": "continuity only; without a trading calendar a skipped session cannot be detected"}
    sessions, meta = _reporting_sessions(calendar, first, last)
    index = {d: i for i, d in enumerate(sessions)}
    name = meta["calendar"]
    spans = []
    for start, session in daily.select("period_start", "session").iter_rows():
        for label, day in (("starts", start), ("ends", session)):
            if day not in index:
                raise ValueError(f"interval {start} to {session} {label} on {day}, which is not a {name} session")
        skipped = sessions[index[start]+1:index[session]]
        if frequency == "1d" and skipped:
            raise ValueError(f"interval {start} to {session} skips {name} session(s) "
                             f"{[d.isoformat() for d in skipped]}; supply those P&L observations or declare "
                             "frequency='multi_session'")
        spans.append(len(skipped)+1)
    return {"status": "validated", "calendar": name, "calendar_metadata": meta, "frequency": frequency,
            "library": meta["library"], "library_version": meta["library_version"],
            "validation_start": first.isoformat(), "validation_end": last.isoformat(),
            "expected_sessions": len(sessions), "observed_sessions": daily.height + 1,
            "sessions_per_interval": {"min": min(spans), "max": max(spans)}}
