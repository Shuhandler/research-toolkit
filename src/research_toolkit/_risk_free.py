"""Explicit dated benchmark-rate conversion; independent of cash and borrowing."""
from copy import deepcopy
from datetime import datetime, time, timedelta, timezone
import json
import math
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl

from ._data import _table
from ._results import RiskFreeResult

INTERVAL_SCHEMA = {"period_start": pl.Date, "session": pl.Date}
RETURN_SCHEMA = {**INTERVAL_SCHEMA, "simple_return": pl.Float64}
DAILY_RATE_SCHEMA = {"date": pl.Date, "annual_rate": pl.Float64, "available_at": pl.Datetime("us", "UTC")}


def _intervals(table):
    table = _table(table, INTERVAL_SCHEMA, "intervals", ["session"], nonempty=True).sort("session")
    previous = None
    for start, end in table.iter_rows():
        if start >= end or (previous is not None and start != previous):
            raise ValueError("holding intervals must be increasing and contiguous")
        previous = end
    return table


def _aligned_returns(table, intervals, metadata, currency, name):
    values = _table(table, RETURN_SCHEMA, name, ["session"], nonempty=True).sort("session")
    if not values.select("period_start", "session").equals(intervals):
        raise ValueError(f"{name} intervals must match both endpoints exactly (strict alignment)")
    if (values["simple_return"] <= -1).any():
        raise ValueError(f"{name} returns must preserve positive wealth")
    if not isinstance(metadata, dict) or any(not isinstance(metadata.get(k), str) or not metadata[k].strip()
        for k in ("source", "basis", "currency", "frequency")):
        raise ValueError(f"{name}_metadata requires source, basis, currency and frequency")
    if metadata["currency"] != currency or metadata["frequency"] != "1d":
        raise ValueError(f"{name} currency/frequency must match the portfolio")
    try:
        meta = json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} metadata must be finite JSON") from exc
    return values, meta


def risk_free_returns(daily_rates, *, intervals, day_count, compounding, rate_timing,
                      timezone_name, metadata) -> RiskFreeResult:
    """Convert explicitly assigned calendar-day annual nominal rates to interval returns.

    Missing dates, including weekends, raise. Availability must precede that date's
    local midnight. No acquisition, holiday carry or spread removal is implicit.
    """
    intervals = _intervals(intervals)
    rates = _table(daily_rates, DAILY_RATE_SCHEMA, "daily_rates", ["date"], nonempty=True).sort("date")
    if day_count not in {"ACT/360", "ACT/365F"} or compounding not in {"daily", "simple"}:
        raise ValueError("require explicit ACT/360 or ACT/365F and daily or simple compounding")
    if rate_timing != "known_at_accrual_start":
        raise ValueError("rate_timing must be known_at_accrual_start")
    try:
        zone = ZoneInfo(timezone_name)
    except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("timezone_name must be a valid IANA timezone") from exc
    expected = []
    for start, end in intervals.iter_rows():
        expected.extend(start+timedelta(days=i) for i in range(1, (end-start).days+1))
    if rates["date"].to_list() != expected:
        raise ValueError("daily_rates must cover every accrual date exactly, including weekends")
    divisor = 360 if day_count == "ACT/360" else 365
    daily = {}
    for day, rate, available in rates.iter_rows():
        cutoff = datetime.combine(day, time.min, tzinfo=zone).astimezone(timezone.utc)
        if available > cutoff:
            raise ValueError(f"rate for {day} was unavailable at accrual start")
        if 1+rate/divisor <= 0:
            raise ValueError("daily risk-free factor must preserve positive wealth")
        daily[day] = rate/divisor
    rows = []
    for start, end in intervals.iter_rows():
        values = [daily[start+timedelta(days=i)] for i in range(1, (end-start).days+1)]
        try:
            value = math.expm1(math.fsum(math.log1p(v) for v in values)) if compounding == "daily" else math.fsum(values)
        except OverflowError as exc:
            raise ValueError("risk-free return is not representable") from exc
        rows.append((start, end, value))
    table = pl.DataFrame(rows, schema=RETURN_SCHEMA, orient="row")
    currency = metadata.get("currency") if isinstance(metadata, dict) else None
    table, meta = _aligned_returns(table, intervals, metadata, currency, "risk_free")
    meta.update(day_count=day_count, compounding=compounding, rate_timing=rate_timing,
        timezone=timezone_name, annual_rate_kind="nominal_decimal", daily_rate_coverage="exact_no_carry")
    return RiskFreeResult(table, rates, meta)


def _risk_free(daily, currency, annual, supplied, metadata, periods_per_year):
    if (annual is None) == (supplied is None):
        raise ValueError("supply exactly one of risk_free_annual_effective or risk_free_returns")
    intervals = daily.select("period_start", "session")
    if supplied is None:
        if metadata is not None:
            raise ValueError("risk_free_metadata requires risk_free_returns")
        if isinstance(annual, bool) or not isinstance(annual, (int, float)) or not math.isfinite(annual) or annual <= -1:
            raise ValueError("risk_free_annual_effective must be finite and greater than -1")
        try:
            periodic = math.expm1(math.log1p(annual)/periods_per_year)
        except OverflowError as exc:
            raise ValueError("periodic risk-free return is not representable") from exc
        return intervals.with_columns(pl.lit(periodic).alias("simple_return")), None, periodic
    if isinstance(supplied, RiskFreeResult):
        if metadata is not None:
            raise ValueError("RiskFreeResult already includes risk_free_metadata")
        metadata, supplied = supplied.metadata, supplied.values
    table, meta = _aligned_returns(supplied, intervals, metadata, currency, "risk_free")
    return table, deepcopy(meta), None
