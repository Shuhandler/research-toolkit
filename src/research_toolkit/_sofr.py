"""Offline, publication-aware SOFR loan assumptions for the daily ledger."""

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timedelta
import json
from zoneinfo import ZoneInfo

import polars as pl

from ._data import _identity, _iso_date, _local_midnight_utc, _table
from ._financing import Financing
from ._portfolio import _number


RATE_SCHEMA = {"observation_date": pl.Date, "sofr": pl.Float64}
PUBLICATION_SCHEMA = {"observation_date": pl.Date, "available_at": pl.Datetime("us", "UTC")}
_NY = ZoneInfo("America/New_York")


@dataclass(frozen=True, kw_only=True)
class SOFRFinancing:
    """SOFR plus a spread on debt; separately configured fixed cash interest.

    Rates are annual decimal fractions. A supplied publication calendar prevents
    missing expected observations from silently becoming holiday carry. Rates are
    selected at New York midnight on each posted accrual date; only already
    available observations qualify. Interest capitalizes every calendar day.
    This is a declared loan model, not the New York Fed's SOFR Index calculation.
    """

    rates: pl.DataFrame
    publication_calendar: pl.DataFrame
    metadata: dict
    borrowing_spread_bps: float
    cash_rate: float
    day_count: str
    cash_day_count: str
    rate_timing: str
    max_rate_age_days: int
    maintenance_equity_ratio: float | None
    on_breach: str
    cash_sweep: str
    snapshot_id: str = field(init=False)

    def __post_init__(self):
        _number(self.borrowing_spread_bps, "borrowing_spread_bps")
        # Reuse the established cash/risk/sweep validation without changing it.
        Financing(cash_rate=self.cash_rate, borrowing_rate=0., day_count="ACT/365F",
            maintenance_equity_ratio=self.maintenance_equity_ratio,
            on_breach=self.on_breach, cash_sweep=self.cash_sweep)
        for name in ("day_count", "cash_day_count"):
            if getattr(self, name) not in {"ACT/360", "ACT/365F"}:
                raise ValueError(f"SOFR {name} must be 'ACT/360' or 'ACT/365F'")
        if self.rate_timing != "known_at_accrual_start":
            raise ValueError("SOFR rate_timing must be 'known_at_accrual_start'")
        if type(self.max_rate_age_days) is not int or self.max_rate_age_days < 1:
            raise ValueError("max_rate_age_days must be a positive integer")
        rates = _table(self.rates, RATE_SCHEMA, "SOFR rates", ["observation_date"], nonempty=True).sort("observation_date")
        calendar = _table(self.publication_calendar, PUBLICATION_SCHEMA,
            "SOFR publication_calendar", ["observation_date"], nonempty=True).sort("observation_date")
        if rates["observation_date"].to_list() != calendar["observation_date"].to_list():
            raise ValueError("SOFR rates must match every expected publication_calendar observation exactly")
        if (rates["sofr"] < 0).any():
            raise ValueError("negative SOFR is not supported; do not silently floor rates")
        for sofr in rates["sofr"]:
            _number(sofr + self.borrowing_spread_bps/10_000, "SOFR plus spread")
        if not isinstance(self.metadata, dict):
            raise ValueError("SOFR metadata must be a JSON dictionary")
        meta = deepcopy(self.metadata)
        try:
            json.dumps(meta, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("SOFR metadata must contain finite JSON values") from exc
        for key in ("source", "calendar", "calendar_version"):
            if not isinstance(meta.get(key), str) or not meta[key].strip():
                raise ValueError(f"SOFR metadata.{key} must be a nonblank string")
        required = {"currency": "USD", "rate_units": "decimal", "timezone": "America/New_York",
                    "vintage": "point_in_time", "calendar_complete": True}
        for key, value in required.items():
            if meta.get(key) != value or (key == "calendar_complete" and meta.get(key) is not True):
                raise ValueError(f"SOFR metadata.{key} must be {value!r}")
        start = _iso_date(meta.get("coverage_start"), "coverage_start")
        end = _iso_date(meta.get("coverage_end"), "coverage_end")
        if start > end:
            raise ValueError("SOFR coverage_start must not follow coverage_end")
        try:
            retrieved = datetime.fromisoformat(meta["retrieved_at"].replace("Z", "+00:00"))
            if retrieved.utcoffset() != timedelta(0):
                raise ValueError
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError("SOFR metadata.retrieved_at must be an ISO UTC timestamp") from exc
        previous = None
        for observed, available in calendar.iter_rows():
            if available.astimezone(_NY).date() <= observed:
                raise ValueError("SOFR available_at must follow its observation date in New York")
            if previous is not None and available <= previous:
                raise ValueError("SOFR publication times must increase with observation dates")
            if available > retrieved:
                raise ValueError("SOFR available_at cannot follow retrieved_at")
            previous = available
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "publication_calendar", calendar)
        object.__setattr__(self, "metadata", meta)
        object.__setattr__(self, "snapshot_id", _identity({"rates": rates, "publication_calendar": calendar}, meta))


def _sofr_plan(financing, entry, end, currency):
    # Dataclass freezing does not make Polars tables/dictionaries immutable.
    validated = replace(financing)
    if validated.snapshot_id != financing.snapshot_id:
        raise ValueError("SOFR inputs changed after construction; construct a new SOFRFinancing")
    if currency != "USD":
        raise ValueError("SOFRFinancing requires a USD portfolio")
    first = entry + timedelta(days=1)
    if first < _iso_date(validated.metadata["coverage_start"], "coverage_start") or end > _iso_date(
            validated.metadata["coverage_end"], "coverage_end"):
        raise ValueError("SOFR declared calendar coverage must include every requested accrual date")
    observations = validated.rates["observation_date"].to_list()
    available = validated.publication_calendar["available_at"].to_list()
    values = validated.rates["sofr"].to_list()
    plan = {}
    day = first
    while day <= end:
        cutoff = _local_midnight_utc(day, _NY)
        i = bisect_right(available, cutoff)-1
        if i < 0:
            raise ValueError(f"no SOFR rate available at accrual start on {day}; supply earlier observations")
        age = (day-observations[i]).days
        if age > validated.max_rate_age_days:
            raise ValueError(f"SOFR rate on {day} exceeds max_rate_age_days ({age})")
        plan[day] = dict(cutoff_at=cutoff, observation_date=observations[i], available_at=available[i],
            rate_age_days=age, sofr=values[i], borrowing_spread_bps=validated.borrowing_spread_bps,
            borrowing_rate=values[i]+validated.borrowing_spread_bps/10_000)
        if day == end:
            break
        day += timedelta(days=1)
    return plan, _sofr_metadata(validated)


def _sofr_metadata(financing):
    config = {f.name: deepcopy(getattr(financing, f.name)) for f in fields(financing)
              if f.name not in {"rates", "publication_calendar"}}
    # Embed exact source records for audit/replay without relying on a mutable URL.
    config["rates"] = [{"observation_date": d.isoformat(), "sofr": r}
                       for d, r in financing.rates.iter_rows()]
    config["publication_calendar"] = [{"observation_date": d.isoformat(), "available_at": t.isoformat()}
                                       for d, t in financing.publication_calendar.iter_rows()]
    config.update(model="historical_sofr_plus_spread", capitalization="daily_calendar_date",
                  cutoff_timezone="America/New_York", missing="raise", carry="latest_available_with_age_limit")
    return config
