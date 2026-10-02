"""Small offline adapters for explicitly supplied provider responses."""
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from zoneinfo import ZoneInfo

import polars as pl

from ._data import (_table, prepare_market_data, PRICE_SCHEMA, SESSION_SCHEMA,
                    SPLIT_SCHEMA, DIVIDEND_SCHEMA)
from ._results import ProviderDataResult

FACTOR_SCHEMA = {"session": pl.Date, "asset": pl.String, "raw_price_factor": pl.Float64}


def yahoo_chart(responses, *, sessions, splits, dividends, metadata, volume_basis,
                availability, raw_adjustment_factors=None, factor_metadata=None) -> ProviderDataResult:
    """Convert saved decoded Yahoo daily chart responses, without network access.

    metadata.price_basis selects split_adjusted Close, total_return_adjusted Adj
    Close, or explicitly reconstructed raw Close. Raw requires complete supplied
    cumulative split factors through retrieval, even when every factor is one.
    Authoritative calendars/actions/payment dates are always caller supplied.
    See docs/notebook-extensions.md for the restricted supported provider contract.
    """
    if not isinstance(responses, Mapping) or not responses or any(not isinstance(a, str) or not a.strip() for a in responses):
        raise ValueError("responses must map asset identifiers to decoded Yahoo chart JSON")
    if availability != "session_close_reconstructed":
        raise ValueError("explicit availability='session_close_reconstructed' is required")
    if volume_basis not in {"unknown", "raw_shares", "split_adjusted_shares"}:
        raise ValueError("volume_basis must be unknown, raw_shares or split_adjusted_shares")
    calendar = _table(sessions, SESSION_SCHEMA, "sessions", ["session"], nonempty=True).sort("session")
    split_table = _table(splits, SPLIT_SCHEMA, "splits", ["action_id"])
    dividend_table = _table(dividends, DIVIDEND_SCHEMA, "dividends", ["action_id"])
    meta = deepcopy(metadata)
    if not isinstance(meta, dict) or meta.get("price_basis") not in {"raw", "split_adjusted", "total_return_adjusted"}:
        raise ValueError("metadata.price_basis must explicitly select raw, split_adjusted or total_return_adjusted")
    basis = meta["price_basis"]
    try:
        zone = ZoneInfo(meta["timezone"])
        retrieved = datetime.fromisoformat(meta["retrieved_at"])
        if retrieved.utcoffset().total_seconds() != 0:
            raise ValueError("retrieval must be UTC")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError("metadata requires timezone and UTC retrieved_at") from exc
    dates = calendar["session"].to_list()
    closes = dict(calendar.iter_rows())
    if retrieved < max(closes.values()):
        raise ValueError("response retrieval precedes a requested session close")
    expected = {(d, a) for d in dates for a in responses}
    factors, factor_info = {}, None
    if basis == "raw":
        table = _table(raw_adjustment_factors, FACTOR_SCHEMA, "raw_adjustment_factors", ["session", "asset"], nonempty=True)
        if set(table.select("session", "asset").iter_rows()) != expected or (table["raw_price_factor"] <= 0).any():
            raise ValueError("raw factors must be positive and cover every session/asset exactly")
        if not isinstance(factor_metadata, dict) or not isinstance(factor_metadata.get("source"), str) or not factor_metadata["source"].strip() or factor_metadata.get("basis") != "cumulative_splits_after_session_through_retrieval":
            raise ValueError("factor_metadata requires source and cumulative_splits_after_session_through_retrieval basis")
        if factor_metadata.get("verified_through") != meta["retrieved_at"]:
            raise ValueError("split factors must be verified through the response retrieved_at, including later splits")
        factors = {(d, a): f for d, a, f in table.iter_rows()}
        ratios = {(r["effective_session"], r["asset"]): r["ratio"] for r in split_table.iter_rows(named=True)}
        for a in responses:
            for before, after in zip(dates, dates[1:]):
                if not math.isclose(factors[before, a]/factors[after, a], ratios.get((after, a), 1.), rel_tol=1e-10):
                    raise ValueError("raw factors disagree with authoritative in-window split actions")
        factor_info = deepcopy(factor_metadata)
    elif raw_adjustment_factors is not None or factor_metadata is not None:
        raise ValueError("raw adjustment factors apply only to raw reconstruction")
    prices, bars, identities = [], [], {}
    split_map = {(r["effective_session"], r["asset"]): r["ratio"] for r in split_table.iter_rows(named=True)}
    dividend_keys = set(dividend_table.select("ex_session", "asset").iter_rows())
    for asset, payload in responses.items():
        try:
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            chart = payload["chart"]
            if chart.get("error") is not None or len(chart["result"]) != 1:
                raise ValueError("expected one successful chart result")
            result = chart["result"][0]
            provider = result["meta"]
            if provider["currency"] != meta["currency"] or provider["exchangeTimezoneName"] != meta["timezone"] or provider["dataGranularity"] != "1d":
                raise ValueError("provider currency/timezone/frequency disagrees with metadata")
            timestamps = result["timestamp"]
            if not timestamps or any(type(t) is not int for t in timestamps):
                raise ValueError("daily timestamps must be nonempty integer Unix seconds")
            days = [datetime.fromtimestamp(t, timezone.utc).astimezone(zone).date() for t in timestamps]
            if len(set(days)) != len(days) or set(days) != set(dates):
                raise ValueError("provider bars must exactly match the authoritative sessions; no silent intersection")
            indicators = result["indicators"]
            if len(indicators["quote"]) != 1:
                raise ValueError("expected one quote block")
            quote = indicators["quote"][0]
            close, volume = quote["close"], quote["volume"]
            adjusted_blocks = indicators.get("adjclose", [])
            if len(adjusted_blocks) > 1:
                raise ValueError("ambiguous adjusted close blocks")
            adjusted = adjusted_blocks[0]["adjclose"] if adjusted_blocks else [None]*len(days)
            if any(len(v) != len(days) for v in (close, volume, adjusted)):
                raise ValueError("provider arrays have mismatched lengths")
            for d, timestamp, p, v, adj in zip(days, timestamps, close, volume, adjusted):
                def number(x, name, optional=False):
                    if x is None and optional:
                        return None
                    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 or (name != "volume" and x == 0):
                        raise ValueError(f"invalid provider {name} for {asset} on {d}")
                    return float(x)
                p, v = number(p, "close"), number(v, "volume", optional=volume_basis == "unknown")
                adj = number(adj, "adjusted close", optional=basis != "total_return_adjusted")
                f = factors.get((d, asset))
                raw_volume = v if volume_basis == "raw_shares" else v/f if f is not None and volume_basis == "split_adjusted_shares" else None
                selected = p*f if basis == "raw" else adj if basis == "total_return_adjusted" else p
                prices.append((d, asset, selected))
                bars.append((d, asset, timestamp, p, adj, v, f, raw_volume,
                    selected*raw_volume if basis == "raw" and raw_volume is not None else None, closes[d]))
            events = result.get("events", {})
            if set(events)-{"splits", "dividends"}:
                raise ValueError("unsupported provider corporate-action type")
            for kind, items in events.items():
                for event in items.values():
                    d = datetime.fromtimestamp(event["date"], timezone.utc).astimezone(zone).date()
                    if kind == "splits":
                        ratio = float(event["numerator"])/float(event["denominator"])
                        if (d, asset) not in split_map or not math.isclose(ratio, split_map[d, asset], rel_tol=1e-10):
                            raise ValueError("provider split missing from or inconsistent with authoritative actions")
                    elif (d, asset) not in dividend_keys:
                        raise ValueError("provider dividend requires an authoritative ex-date/payment-date record")
            identities[asset] = {"sha256": hashlib.sha256(encoded).hexdigest(), "provider_symbol": provider.get("symbol")}
        except (KeyError, TypeError, IndexError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError(f"malformed Yahoo daily chart response for {asset}") from exc
    adapter = dict(provider="Yahoo chart", response_identities=identities,
        provider_close_basis="split_adjusted", provider_adjclose_basis="total_return_adjusted",
        volume_basis=volume_basis, availability=availability,
        availability_limitation="session close proxy; not observed publication time or point-in-time revision history",
        action_source="authoritative caller tables; provider event amounts are not substituted",
        raw_factor_metadata=factor_info)
    meta["adapter"] = adapter
    market = prepare_market_data(prices=pl.DataFrame(prices, schema=PRICE_SCHEMA, orient="row"),
        sessions=calendar, splits=split_table, dividends=dividend_table, metadata=meta)
    bar_table = pl.DataFrame(bars, schema={"session": pl.Date, "asset": pl.String, "provider_timestamp": pl.Int64,
        "provider_close": pl.Float64, "provider_adjclose": pl.Float64, "provider_volume": pl.Float64,
        "raw_price_factor": pl.Float64, "raw_share_volume": pl.Float64, "dollar_volume": pl.Float64,
        "available_at": pl.Datetime("us", "UTC")}, orient="row").sort("session", "asset")
    for col, dtype in bar_table.schema.items():
        if dtype == pl.Float64 and not bar_table[col].drop_nulls().is_finite().all():
            raise ValueError("converted provider values are not representable")
    diagnostics = pl.DataFrame({"code": ["reconstructed_availability", "caller_supplied_action_completeness", "declared_volume_basis"],
        "detail": [adapter["availability_limitation"], adapter["action_source"], volume_basis]})
    return ProviderDataResult(market, bar_table, diagnostics, adapter)
