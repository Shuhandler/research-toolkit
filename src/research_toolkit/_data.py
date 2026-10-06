"""Strict, provider-independent daily inputs and deterministic provenance."""

from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl

from ._results import MarketData


UTC = pl.Datetime("us", "UTC")
PRICE_SCHEMA = {"session": pl.Date, "asset": pl.String, "close": pl.Float64}
SESSION_SCHEMA = {"session": pl.Date, "close_at": UTC}
SPLIT_SCHEMA = {
    "action_id": pl.String, "asset": pl.String,
    "effective_session": pl.Date, "ratio": pl.Float64,
}
DIVIDEND_SCHEMA = {
    "action_id": pl.String, "asset": pl.String, "ex_session": pl.Date,
    "pay_date": pl.Date, "cash_per_share": pl.Float64,
}
DIAGNOSTIC_SCHEMA = {"code": pl.String, "table": pl.String, "count": pl.Int64}


def _table(frame, schema, name, keys, *, nonempty=False):
    if not isinstance(frame, pl.DataFrame):
        raise ValueError(f"{name} must be a Polars DataFrame")
    if frame.schema != schema:
        raise ValueError(f"{name} schema must be {schema}; got {dict(frame.schema)}")
    if nonempty and frame.is_empty():
        raise ValueError(f"{name} must not be empty")
    if any(frame.null_count().row(0)):
        raise ValueError(f"{name} contains null inputs")
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"{name} has duplicate keys: {keys}")
    for col, dtype in schema.items():
        if dtype == pl.Float64 and not frame[col].is_finite().all():
            raise ValueError(f"{name}.{col} must be finite")
        if dtype == pl.String and frame[col].str.strip_chars().eq("").any():
            raise ValueError(f"{name}.{col} must not be blank")
    return frame.select(list(schema)).clone()


def _local_midnight_utc(day, zone):
    """UTC instant of 00:00 local time on ``day`` in ``zone`` (a ZoneInfo or IANA name).

    The single information cutoff used for rates, borrow fees and liquidity inputs
    that must be known before a calendar date or decision session begins.
    """
    zone = ZoneInfo(zone) if isinstance(zone, str) else zone
    return datetime.combine(day, time.min, tzinfo=zone).astimezone(timezone.utc)


def _iso_date(value, name):
    if not isinstance(value, str):
        raise ValueError(f"metadata.{name} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"metadata.{name} must be an ISO date string") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"metadata.{name} must use YYYY-MM-DD")
    return parsed


def _identity(tables, metadata):
    payload = {
        "metadata": metadata,
        "tables": {name: {"schema": {k: str(v) for k, v in df.schema.items()},
                          "rows": df.to_dicts()} for name, df in tables.items()},
    }
    encoded = json.dumps(payload, default=lambda v: v.isoformat(),
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def prepare_market_data(*, prices, sessions, splits, dividends, metadata,
                        missing="raise") -> MarketData:
    """Validate and copy an entire daily panel; never fill or drop missing rows.

    Exact schemas and required JSON metadata are documented in ``docs/api.md``.
    Supply typed empty action tables and ``actions_complete=True`` when no actions
    occurred. The calendar is caller-supplied, not inferred from observed prices.
    Rows are sorted canonically with diagnostics; numerical dtypes are not coerced.
    """
    if missing != "raise":
        raise ValueError("only missing='raise' is supported")
    tables = {
        "prices": _table(prices, PRICE_SCHEMA, "prices", ["session", "asset"], nonempty=True),
        "sessions": _table(sessions, SESSION_SCHEMA, "sessions", ["session"], nonempty=True),
        "splits": _table(splits, SPLIT_SCHEMA, "splits", ["action_id"]),
        "dividends": _table(dividends, DIVIDEND_SCHEMA, "dividends", ["action_id"]),
    }
    keys = {"prices": ["session", "asset"], "sessions": ["session"],
            "splits": ["effective_session", "asset", "action_id"],
            "dividends": ["ex_session", "asset", "action_id"]}
    diagnostics = []
    for name, frame in tables.items():
        ordered = frame.sort(keys[name])
        if not ordered.equals(frame):
            diagnostics.append({"code": "sorted_rows", "table": name, "count": frame.height})
        tables[name] = ordered
    prices, sessions, splits, dividends = (tables[n] for n in tables)
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a JSON-compatible mapping")
    try:
        meta = json.loads(json.dumps(dict(metadata), allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ValueError("metadata must contain finite JSON-compatible values") from exc
    required = {
        "source", "retrieved_at", "currency", "asset_currencies", "calendar",
        "calendar_version", "timezone", "price_basis", "frequency", "coverage_start",
        "coverage_end", "actions_complete", "dividend_basis",
    }
    if required - meta.keys():
        raise ValueError(f"metadata missing keys: {sorted(required - meta.keys())}")
    for key in required - {"asset_currencies", "actions_complete"}:
        if not isinstance(meta[key], str) or not meta[key].strip():
            raise ValueError(f"metadata.{key} must be a nonblank string")
    if meta["frequency"] != "1d":
        raise ValueError("only frequency='1d' is supported")
    if meta["price_basis"] not in {"raw", "split_adjusted", "total_return_adjusted"}:
        raise ValueError("unknown price_basis")
    if meta["actions_complete"] is not True:
        raise ValueError("actions_complete=True is required, including for empty action tables")
    if meta["dividend_basis"] != "post_split_share":
        raise ValueError("dividend_basis must be 'post_split_share'")
    try:
        retrieved = datetime.fromisoformat(meta["retrieved_at"])
        if retrieved.utcoffset() != timedelta(0):
            raise ValueError("not UTC")
        zone = ZoneInfo(meta["timezone"])
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("retrieved_at must be UTC and timezone must be a valid IANA zone") from exc
    dates = sessions["session"].to_list()
    start = _iso_date(meta["coverage_start"], "coverage_start")
    end = _iso_date(meta["coverage_end"], "coverage_end")
    if start != dates[0] or end != dates[-1]:
        raise ValueError("coverage_start/end must match the first/last supplied session")
    closes = sessions["close_at"].to_list()
    if any(a >= b for a, b in zip(closes, closes[1:])):
        raise ValueError("session close_at must increase strictly")
    if any(t.astimezone(zone).date() != d for t, d in zip(closes, dates)):
        raise ValueError("session date disagrees with close_at in metadata timezone")
    assets = sorted(prices["asset"].unique().to_list())
    currencies = meta["asset_currencies"]
    if not isinstance(currencies, dict) or set(currencies) != set(assets):
        raise ValueError("asset_currencies must cover exactly the price assets")
    if any(c != meta["currency"] for c in currencies.values()):
        raise ValueError("only a single currency is supported")
    if (prices["close"] <= 0).any():
        raise ValueError("prices.close must be positive")
    expected = {(d, a) for d in dates for a in assets}
    actual = set(prices.select("session", "asset").iter_rows())
    if actual != expected:
        raise ValueError(f"prices/session alignment: missing={sorted(expected-actual)[:5]}, "
                         f"unexpected={sorted(actual-expected)[:5]}")
    ids = splits["action_id"].to_list() + dividends["action_id"].to_list()
    if len(ids) != len(set(ids)):
        raise ValueError("action_id must be unique across all action tables")
    if (splits["ratio"] <= 0).any() or (dividends["cash_per_share"] < 0).any():
        raise ValueError("split ratios must be positive and dividends nonnegative")
    if splits.select("asset", "effective_session").is_duplicated().any():
        raise ValueError("multiple splits for the same asset/session are ambiguous")
    for table, frame, date_col in (("splits", splits, "effective_session"),
                                   ("dividends", dividends, "ex_session")):
        for row in frame.iter_rows(named=True):
            if row["asset"] not in assets or row[date_col] not in dates:
                raise ValueError(f"{table} action {row['action_id']} has an unknown asset/session")
            if table == "dividends" and row["pay_date"] < row["ex_session"]:
                raise ValueError(f"dividend {row['action_id']} pays before its ex-session")
    # Extra columns/action types fail the exact schemas rather than being ignored.
    return MarketData(**tables, metadata=deepcopy(meta),
                      diagnostics=pl.DataFrame(diagnostics, schema=DIAGNOSTIC_SCHEMA),
                      snapshot_id=_identity(tables, meta))


def _validated_market(market):
    if not isinstance(market, MarketData):
        raise ValueError("expected MarketData from prepare_market_data")
    validated = prepare_market_data(prices=market.prices, sessions=market.sessions,
                                    splits=market.splits, dividends=market.dividends,
                                    metadata=market.metadata)
    if validated.snapshot_id != market.snapshot_id:
        raise ValueError("MarketData changed after validation; prepare a new snapshot")
    return validated
