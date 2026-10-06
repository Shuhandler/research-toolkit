"""Small provider adapters. Only download_yahoo uses the network, and only when called."""
from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from zoneinfo import ZoneInfo

import polars as pl

from ._data import (_table, prepare_market_data as _prepare_market_data, PRICE_SCHEMA, SESSION_SCHEMA,
                    SPLIT_SCHEMA, DIVIDEND_SCHEMA)
from ._calendars import _exchange_sessions
from ._results import ProviderDataResult, ProviderDownloadResult

__all__ = ["yahoo_chart", "download_yahoo"]

FACTOR_SCHEMA = {"session": pl.Date, "asset": pl.String, "raw_price_factor": pl.Float64}


def yahoo_chart(responses, *, sessions, splits, dividends, metadata, volume_basis,
                availability, raw_adjustment_factors=None, factor_metadata=None) -> ProviderDataResult:
    """Convert saved decoded Yahoo daily chart responses, without network access.

    ``responses`` maps assets to decoded chart JSON, or is a ``ProviderDownloadResult``
    from ``download_yahoo``; its retained payloads are first limited to the download's
    inclusive requested sessions. metadata.price_basis selects split_adjusted Close,
    total_return_adjusted Adj Close, or explicitly reconstructed raw Close. Raw
    requires complete supplied cumulative split factors through retrieval, even when
    every factor is one. Authoritative calendars/actions/payment dates are always
    caller supplied. See docs/notebook-extensions.md for the provider contract.
    """
    source_download = None
    if isinstance(responses, ProviderDownloadResult):
        source_download, responses = responses, _download_payloads(responses)
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
            for d, timestamp, close_value, volume_value, adjusted_value in zip(days, timestamps, close, volume, adjusted):
                where = f"{asset} on {d}"
                p = _provider_number(close_value, "close", where)
                v = _provider_number(volume_value, "volume", where, optional=volume_basis == "unknown")
                adj = _provider_number(adjusted_value, "adjusted close", where, optional=basis != "total_return_adjusted")
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
    if source_download is not None:
        source = source_download.metadata
        adapter["source_download"] = {"requested_start": source["requested_start"], "requested_end": source["requested_end"],
            "retrieved_at": source["retrieved_at"], "bars": "limited_to_requested_inclusive_sessions",
            "response_sha256": {a: source["instruments"][a]["response_sha256"] for a in source["assets"]}}
    meta["adapter"] = adapter
    market = _prepare_market_data(prices=pl.DataFrame(prices, schema=PRICE_SCHEMA, orient="row"),
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


def _provider_number(x, name, where, optional=False):
    """A provider value as float: finite, nonnegative, and nonzero except volume; None only if optional."""
    if x is None and optional:
        return None
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 or (name != "volume" and x == 0):
        raise ValueError(f"invalid provider {name} for {where}")
    return float(x)


def _download_payloads(download):
    """Retained decoded payloads, limited to the download's inclusive requested sessions."""
    meta = download.metadata
    if not download.responses or set(download.responses) != set(meta.get("assets", ())):
        raise ValueError("this ProviderDownloadResult does not retain the decoded provider responses")
    start, end = date.fromisoformat(meta["requested_start"]), date.fromisoformat(meta["requested_end"])
    payloads = {}
    for asset, original in download.responses.items():
        payload = deepcopy(original)
        try:
            result = payload["chart"]["result"][0]
            zone = ZoneInfo(result["meta"]["exchangeTimezoneName"])
            stamps = result.get("timestamp") or []
            keep = [i for i, t in enumerate(stamps) if start <= _local_date(t, zone) <= end]
            result["timestamp"] = [stamps[i] for i in keep]
            for block in [*result["indicators"].get("quote", []), *result["indicators"].get("adjclose", [])]:
                for key, values in block.items():
                    if isinstance(values, list):
                        block[key] = [values[i] for i in keep]
            for kind, items in (result.get("events") or {}).items():
                result["events"][kind] = {k: v for k, v in items.items() if start <= _local_date(v["date"], zone) <= end}
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise ValueError(f"malformed retained Yahoo response for {asset}") from exc
        payloads[asset] = payload
    return payloads


def _local_date(stamp, zone):
    return datetime.fromtimestamp(stamp, timezone.utc).astimezone(zone).date()


YAHOO_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
DOWNLOAD_SCHEMA = {"session": pl.Date, "asset": pl.String, "close": pl.Float64,
                   "adj_close": pl.Float64, "volume": pl.Float64}
DOWNLOAD_DIAGNOSTIC_SCHEMA = {"asset": pl.String, "session": pl.Date, "code": pl.String, "detail": pl.String}


def _now():
    return datetime.now(timezone.utc)


def _yahoo_get(url, *, timeout, max_attempts, backoff_seconds):
    """GET with a per-request timeout and bounded retries of transient failures only."""
    import time as clock
    import urllib.error
    import urllib.request
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research-toolkit)",
                                                   "Accept": "application/json"})
    last = None
    for attempt in range(1, max_attempts+1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), attempt
        except urllib.error.HTTPError as exc:
            try:
                if exc.code != 429 and exc.code < 500:
                    detail = exc.read()[:500].decode("utf-8", "replace")
                    raise ValueError(f"Yahoo rejected the request (HTTP {exc.code}): {detail}") from exc
            finally:
                exc.close()
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < max_attempts:
            clock.sleep(backoff_seconds * 2**(attempt-1))
    raise ConnectionError(f"Yahoo request failed after {max_attempts} attempts: {last!r}") from last


def download_yahoo(assets, *, start, end, timeout=10.0, max_attempts=3, backoff_seconds=1.0,
                   calendars=None, as_of=None, coverage=None, exclusions=None,
                   incomplete="raise") -> ProviderDownloadResult:
    """Download daily Yahoo chart bars for ``start <= session <= end`` (both inclusive).

    Sessions are exchange-local dates from each instrument's reported exchange
    timezone. Yahoo's exclusive ``period2`` is translated by requesting a padded UTC
    window and keeping the inclusive local range. ``close`` is Yahoo Close (split
    adjusted, not dividend adjusted), ``adj_close`` is Yahoo Adj Close and ``volume``
    is reported volume; nothing is forward-filled, substituted or dropped. Failures
    and duplicate sessions raise; null values stay null and are listed in diagnostics.

    ``calendars`` maps EVERY asset to an exchange calendar ('XNYS', 'XTKS', ...).
    Each asset is then checked against its own sessions whose close (early closes
    and DST included) is at or before ``as_of`` (default: retrieval time). Missing,
    unexpected or not-yet-closed bars raise, or with incomplete='report' are listed
    and the result status is 'incomplete'. ``coverage`` declares known listing or
    delisting limits; ``exclusions`` explains suspensions/unscheduled closures.
    Without calendars, missing sessions cannot be fully detected. An asset with no
    bars in range raises unless calendar validation shows that no completed session
    was expected for it (declared coverage, exclusions or closes after ``as_of``).

    Output rows follow the requested asset order, then session. ``sessions`` holds
    each calendar's completed sessions for ``prepare_market_data``, and ``responses``
    keeps the decoded payloads for ``yahoo_chart`` raw-price reconstruction.
    """
    import urllib.parse
    if isinstance(assets, str) or not isinstance(assets, (list, tuple)) or not assets \
            or any(not isinstance(a, str) or not a.strip() for a in assets) or len(set(assets)) != len(assets):
        raise ValueError("assets must be a nonempty list of unique nonblank Yahoo symbols")
    if any(type(d) is not date for d in (start, end)) or start > end:
        raise ValueError("start and end must be datetime.date values with start <= end")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive seconds")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
        raise ValueError("max_attempts must be an integer from 1 to 10")
    if isinstance(backoff_seconds, bool) or not isinstance(backoff_seconds, (int, float)) or not 0 <= backoff_seconds <= 60:
        raise ValueError("backoff_seconds must be between 0 and 60")
    checks = _calendar_options(assets, calendars, as_of, coverage, exclusions, incomplete)
    # Local session dates can sit up to a day either side of their UTC date.
    period1 = int(datetime.combine(start - timedelta(days=1), time.min, timezone.utc).timestamp())
    period2 = int(datetime.combine(end + timedelta(days=2), time.min, timezone.utc).timestamp())
    params = {"period1": period1, "period2": period2, "interval": "1d", "events": "div,split",
              "includeAdjustedClose": "true"}
    rows: list[tuple] = []
    diagnostics: list[tuple] = []
    instruments, payloads = {}, {}
    cutoffs: dict[str, datetime] = {}
    for asset in assets:
        url = YAHOO_CHART_URL.format(symbol=urllib.parse.quote(asset, safe="")) + "?" + urllib.parse.urlencode(params)
        body, attempts = _yahoo_get(url, timeout=timeout, max_attempts=max_attempts, backoff_seconds=backoff_seconds)
        retrieved = _now()
        try:
            decoded = json.loads(body)
            chart = decoded["chart"]
            if chart.get("error") is not None or len(chart["result"]) != 1:
                raise ValueError(f"Yahoo returned an error for {asset}: {chart.get('error')}")
            result = chart["result"][0]
            provider = result["meta"]
            if provider["dataGranularity"] != "1d":
                raise ValueError(f"Yahoo returned {provider['dataGranularity']} bars for {asset}, not 1d")
            zone = ZoneInfo(provider["exchangeTimezoneName"])
            timestamps = result.get("timestamp") or []
            quote = result["indicators"]["quote"]
            adjusted = result["indicators"].get("adjclose")
            if timestamps and (len(quote) != 1 or not adjusted or len(adjusted) != 1):
                raise ValueError(f"expected one quote and one adjclose block for {asset}")
            columns = [quote[0]["close"], adjusted[0]["adjclose"], quote[0]["volume"]] if timestamps else [[], [], []]
            if any(type(t) is not int for t in timestamps) or any(len(c) != len(timestamps) for c in columns):
                raise ValueError(f"Yahoo arrays for {asset} are malformed or have mismatched lengths")
        except (KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed Yahoo chart response for {asset}") from exc
        open_period = (provider.get("currentTradingPeriod") or {}).get("regular") or {}
        open_end = open_period.get("end")
        asset_rows = {}
        for stamp, close, adj, volume in zip(timestamps, *columns):
            session = datetime.fromtimestamp(stamp, timezone.utc).astimezone(zone).date()
            if not start <= session <= end:
                continue
            if session in asset_rows:
                raise ValueError(f"Yahoo returned duplicate {asset} records for session {session}")
            values = []
            for name, value in (("close", close), ("adj_close", adj), ("volume", volume)):
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0 or (name != "volume" and value == 0)):
                    raise ValueError(f"invalid Yahoo {name} for {asset} on {session}: {value!r}")
                values.append(None if value is None else float(value))
            missing = [n for n, v in zip(("close", "adj_close", "volume"), values) if v is None]
            if missing:
                diagnostics.append((asset, session, "missing_value", ",".join(missing) + " null in provider response"))
            if calendars is None and isinstance(open_end, int) and retrieved.timestamp() < open_end \
                    and session == datetime.fromtimestamp(open_end, timezone.utc).astimezone(zone).date():
                diagnostics.append((asset, session, "session_open_at_retrieval",
                                    "retrieved before the regular session ended; values may be intraday"))
            asset_rows[session] = (session, asset, *values)
        validation = None
        if calendars is not None:
            validation, found = _validate_sessions(asset, set(asset_rows), provider, zone, start, end,
                retrieved, checks)
            diagnostics.extend(found)
            name, cutoff = checks["calendars"][asset], checks["as_of"] or retrieved
            cutoffs[name] = min(cutoffs.get(name, cutoff), cutoff)
        # Never return a requested asset with no bars unless the calendar explains every absence.
        if not asset_rows and (validation is None or validation["expected_sessions"] or validation["coverage_uncertain"]):
            raise ValueError(f"Yahoo returned no {asset} observations from {start} to {end}"
                             + ("" if validation is None else "; no declared coverage limit, exclusion or "
                                "not-yet-closed session explains the absence"))
        rows.extend(asset_rows[d] for d in sorted(asset_rows))
        payloads[asset] = decoded
        instruments[asset] = {"calendar_validation": validation,"provider_symbol": provider.get("symbol"), "currency": provider.get("currency"),
            "exchange": provider.get("exchangeName"), "exchange_timezone": provider["exchangeTimezoneName"],
            "instrument_type": provider.get("instrumentType"), "url": url, "attempts": attempts,
            "retrieved_at": retrieved.isoformat().replace("+00:00", "Z"),
            "response_sha256": hashlib.sha256(body).hexdigest()}
    values = pl.DataFrame(rows, schema=DOWNLOAD_SCHEMA, orient="row")
    # Without a calendar, only sessions another same-timezone asset reported can be flagged.
    by_zone = {}
    for asset, info in instruments.items() if calendars is None else ():
        by_zone.setdefault(info["exchange_timezone"], []).append(asset)
    sessions = {a: set(values.filter(pl.col("asset") == a)["session"].to_list()) for a in assets}
    for members in by_zone.values():
        union = set().union(*(sessions[a] for a in members))
        for a in members:
            diagnostics.extend((a, d, "session_not_reported", "another requested asset in this exchange timezone has this session")
                               for d in sorted(union - sessions[a]))
    if len({info["currency"] for info in instruments.values()}) > 1:
        diagnostics.append((None, None, "mixed_currencies", "assets report different currencies; see metadata.instruments"))
    metadata = dict(provider="Yahoo chart", source="Yahoo Finance chart API (v8)", endpoint=YAHOO_CHART_URL,
        request_parameters=params, frequency="1d", requested_start=start.isoformat(), requested_end=end.isoformat(),
        date_boundary="inclusive exchange-local start and end sessions; padded UTC request window filtered locally "
                      "because Yahoo period2 is exclusive",
        session_basis="exchange-local date of each bar timestamp in the instrument's exchangeTimezoneName",
        close_basis="split_adjusted", adj_close_basis="split_and_distribution_adjusted",
        volume_basis="provider_reported_adjustment_unverified",
        retrieved_at=max(i["retrieved_at"] for i in instruments.values()),
        timeout_seconds=float(timeout), max_attempts=max_attempts, backoff_seconds=float(backoff_seconds),
        instruments=instruments, assets=list(assets),
        calendar_validation=_validation_summary(instruments, checks),
        limitations=("calendar sessions come from the installed calendar version; unscheduled closures it lacks "
                     "appear as missing sessions until explained in exclusions" if calendars is not None else
                     "no trading calendar: absent sessions are flagged only relative to other requested assets")
                    + "; Yahoo history can be revised and is not a point-in-time vintage")
    if calendars is not None and incomplete == "raise" and metadata["calendar_validation"]["status"] == "incomplete":
        problems = [f"{a}: " + "; ".join(f"{k} {v['problems'][k]}" for k in v["problems"])
                    for a, v in ((a, i["calendar_validation"]) for a, i in instruments.items()) if v["status"] == "incomplete"]
        raise ValueError("Yahoo data is incomplete against the exchange calendars (pass incomplete='report' to "
                         "inspect, or explain suspensions/closures in exclusions): " + " | ".join(problems))
    calendar_sessions = {name: pl.DataFrame([(d, close) for d, close in _exchange_sessions(name, start, end)[0] if close <= cutoff],
                                            schema=SESSION_SCHEMA, orient="row") for name, cutoff in sorted(cutoffs.items())}
    return ProviderDownloadResult(values, pl.DataFrame(diagnostics, schema=DOWNLOAD_DIAGNOSTIC_SCHEMA, orient="row"),
                                  metadata, calendar_sessions, payloads)


EXCLUSION_SCHEMA = {"asset": pl.String, "session": pl.Date, "reason": pl.String}


def _calendar_options(assets, calendars, as_of, coverage, exclusions, incomplete):
    """Validate calendar options before any request; nothing applies without calendars."""
    if calendars is None:
        if any(v is not None for v in (as_of, coverage, exclusions)) or incomplete != "raise":
            raise ValueError("as_of, coverage, exclusions and incomplete require calendars")
        return None
    if not isinstance(calendars, Mapping) or set(calendars) != set(assets) \
            or any(not isinstance(v, str) or not v.strip() for v in calendars.values()):
        raise ValueError("calendars must map every requested asset to an exchange calendar identifier; "
                         "the adapter does not assume assets share an exchange")
    if incomplete not in {"raise", "report"}:
        raise ValueError("incomplete must be 'raise' or 'report'")
    if as_of is not None and (not isinstance(as_of, datetime) or as_of.utcoffset() is None):
        raise ValueError("as_of must be a timezone-aware datetime")
    limits = {}
    for asset, limit in (coverage or {}).items():
        if asset not in calendars or not isinstance(limit, Mapping) or set(limit) - {"start", "end", "source"} \
                or not isinstance(limit.get("source"), str) or not limit["source"].strip() \
                or any(limit.get(k) is not None and type(limit[k]) is not date for k in ("start", "end")) \
                or (limit.get("start") is None and limit.get("end") is None):
            raise ValueError("coverage maps requested assets to {'start': date|None, 'end': date|None, 'source': str}")
        limits[asset] = dict(limit)
    excluded = {}
    if exclusions is not None:
        table = _table(exclusions, EXCLUSION_SCHEMA, "exclusions", ["asset", "session"])
        for asset, session, reason in table.iter_rows():
            if asset not in calendars:
                raise ValueError(f"exclusion for unrequested asset {asset}")
            excluded.setdefault(asset, {})[session] = reason
    return {"calendars": dict(calendars), "as_of": as_of, "coverage": limits, "exclusions": excluded,
            "incomplete": incomplete}


def _validate_sessions(asset, observed, provider, zone, start, end, retrieved, checks):
    """Compare one asset's bars with ITS calendar's completed sessions; other assets are irrelevant."""
    sessions, cal = _exchange_sessions(checks["calendars"][asset], start, end)
    if cal["timezone"] != provider["exchangeTimezoneName"]:
        raise ValueError(f"{asset}: calendar {cal['calendar']} uses {cal['timezone']} but Yahoo reports "
                         f"exchange timezone {provider['exchangeTimezoneName']}; check the asset's calendar")
    as_of = checks["as_of"] or retrieved
    if as_of > retrieved:
        raise ValueError("as_of cannot be later than the retrieval time; later bars were not observed")
    limit = checks["coverage"].get(asset, {})
    coverage_start, coverage_source = limit.get("start"), limit.get("source")
    if coverage_start is None and isinstance(provider.get("firstTradeDate"), int):
        coverage_start = datetime.fromtimestamp(provider["firstTradeDate"], timezone.utc).astimezone(zone).date()
        coverage_source = "provider_first_trade_date" if limit.get("end") is None else f"{coverage_source}; provider_first_trade_date"
    coverage_end = limit.get("end")
    excluded = checks["exclusions"].get(asset, {})
    calendar_days = {d for d, _ in sessions}
    for day in excluded:
        if day not in calendar_days:
            raise ValueError(f"{asset}: excluded date {day} is not a {cal['calendar']} session in the requested range")
    expected, not_closed, outside = [], [], []
    for day, close_at in sessions:
        if close_at > as_of:
            not_closed.append(day)
        elif (coverage_start and day < coverage_start) or (coverage_end and day > coverage_end):
            outside.append(day)
        elif day not in excluded:
            expected.append(day)
    missing = [d for d in expected if d not in observed]
    uncertain = []
    if coverage_start is None:
        first_bar = min(observed, default=None)
        uncertain = [d for d in missing if first_bar is None or d < first_bar]
        missing = [d for d in missing if d not in uncertain]
    problems = {"missing_session": missing,
                "unexpected_session": sorted(observed - calendar_days),
                "session_open_at_validation": sorted(observed & set(not_closed)),
                "bar_outside_declared_coverage": sorted(observed & set(outside))}
    problems = {k: [d.isoformat() for d in v] for k, v in problems.items() if v}
    found = [(asset, date.fromisoformat(d), code, detail) for code, detail in (
                ("missing_session", "completed exchange session without a bar"),
                ("unexpected_session", f"bar on a date that is not a {cal['calendar']} session"),
                ("session_open_at_validation", "bar for a session whose close is after the validation time"),
                ("bar_outside_declared_coverage", "bar outside the declared listing/coverage limits"))
             for d in problems.get(code, [])]
    found += [(asset, d, "session_not_closed", "session close is after the validation time; not expected yet")
              for d in not_closed if d not in observed]
    found += [(asset, d, "coverage_uncertain", "before the first bar and listing date unknown; not assumed missing")
              for d in uncertain]
    found += [(asset, d, "excluded_session", excluded[d]) for d in sorted(excluded)]
    status = "incomplete" if problems else "coverage_uncertain" if uncertain else "complete"
    return {**cal, "validation_start": start.isoformat(), "validation_end": end.isoformat(),
            "validation_time": as_of.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "coverage_start": coverage_start.isoformat() if coverage_start else None,
            "coverage_end": coverage_end.isoformat() if coverage_end else None,
            "coverage_source": coverage_source, "expected_sessions": len(expected),
            "observed_sessions": len(observed), "not_yet_closed": [d.isoformat() for d in not_closed],
            "excluded": [{"session": d.isoformat(), "reason": r} for d, r in sorted(excluded.items())],
            "coverage_uncertain": [d.isoformat() for d in uncertain], "problems": problems, "status": status}, found


def _validation_summary(instruments, checks):
    if checks is None:
        return {"status": "not_validated", "limitation": "no trading calendar; missing sessions cannot be fully detected"}
    statuses = [i["calendar_validation"]["status"] for i in instruments.values()]
    first = next(iter(instruments.values()))["calendar_validation"]
    return {"status": "incomplete" if "incomplete" in statuses else
                      "coverage_uncertain" if "coverage_uncertain" in statuses else "complete",
            "calendars": checks["calendars"], "library": first["library"], "library_version": first["library_version"],
            "incomplete_policy": checks["incomplete"]}
