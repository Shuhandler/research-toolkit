from datetime import date, datetime, timezone
import io
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

import pytest

import research_toolkit as rt

RETRIEVED = datetime(2024, 6, 10, 12, tzinfo=timezone.utc)  # Same as conftest's provider fixture.


def stamp(*args):
    return int(datetime(*args, tzinfo=timezone.utc).timestamp())


def chart(symbol, zone, stamps, close, adj, volume, currency="USD", error=None):
    return {"chart": {"error": error, "result": None if error else [{
        "meta": {"symbol": symbol, "currency": currency, "exchangeName": "X", "exchangeTimezoneName": zone,
                 "dataGranularity": "1d", "instrumentType": "EQUITY"},
        "timestamp": stamps, "indicators": {"quote": [{"close": close, "volume": volume}],
                                            "adjclose": [{"adjclose": adj}]}}]}}


# New York bars stamped at 13:30 UTC; the last one at 01:00 UTC on 8 June is still 7 June locally.
NY = chart("AAA", "America/New_York",
           [stamp(2024, 5, 31, 13, 30), stamp(2024, 6, 3, 13, 30), stamp(2024, 6, 4, 13, 30), stamp(2024, 6, 8, 1),
            stamp(2024, 6, 10, 13, 30)],
           [9., 10., 11., 12., 99.], [8.5, 9.5, 10.5, 11.5, 98.], [100, 200, 300, 400, 500])
# Tokyo bars stamped 15:00 UTC are the NEXT local day (00:00 JST).
TOKYO = chart("BBB.T", "Asia/Tokyo", [stamp(2024, 6, 2, 15), stamp(2024, 6, 3, 15), stamp(2024, 6, 6, 15)],
              [3000., 3100., 3200.], [2900., 3000., 3100.], [10, 20, 30], currency="JPY")


def test_inclusive_exchange_local_sessions_and_columns(provider):
    script, calls, _ = provider
    script.update({"AAA": NY, "BBB.T": TOKYO})
    result = rt.adapters.download_yahoo(["AAA", "BBB.T"], start=date(2024, 6, 3), end=date(2024, 6, 7))
    v = result.values
    assert v.columns == ["session", "asset", "close", "adj_close", "volume"]
    assert v.filter(v["asset"] == "AAA")["session"].to_list() == [date(2024, 6, 3), date(2024, 6, 4), date(2024, 6, 7)]
    assert v.filter(v["asset"] == "BBB.T")["session"].to_list() == [date(2024, 6, 3), date(2024, 6, 4), date(2024, 6, 7)]
    aaa = v.filter(v["asset"] == "AAA")
    assert aaa["close"].to_list() == [10., 11., 12.] and aaa["adj_close"].to_list() == [9.5, 10.5, 11.5]
    assert aaa["volume"].to_list() == [200., 300., 400.]
    # Yahoo's period2 is exclusive: the padded UTC window covers both inclusive local endpoints.
    query = urllib.parse.parse_qs(urllib.parse.urlparse(calls[0][0]).query)
    assert int(query["period1"][0]) == stamp(2024, 6, 2) and int(query["period2"][0]) == stamp(2024, 6, 9)
    assert query["interval"] == ["1d"] and query["includeAdjustedClose"] == ["true"]
    m = result.metadata
    assert (m["requested_start"], m["requested_end"]) == ("2024-06-03", "2024-06-07")
    assert m["close_basis"] == "split_adjusted" and m["adj_close_basis"] == "split_and_distribution_adjusted"
    assert m["retrieved_at"] == "2024-06-10T12:00:00Z"
    assert m["instruments"]["BBB.T"]["exchange_timezone"] == "Asia/Tokyo"
    assert len(m["instruments"]["AAA"]["response_sha256"]) == 64 and m["instruments"]["AAA"]["attempts"] == 1
    assert result.diagnostics["code"].to_list() == ["mixed_currencies"]


def test_absent_sessions_and_null_values_are_reported_not_filled(provider):
    script, _, _ = provider
    other = chart("CCC", "America/New_York", [stamp(2024, 6, 3, 13, 30), stamp(2024, 6, 4, 13, 30)],
                  [20., None], [19., None], [5, None])
    script.update({"AAA": NY, "CCC": other})
    result = rt.adapters.download_yahoo(["AAA", "CCC"], start=date(2024, 6, 3), end=date(2024, 6, 7))
    ccc = result.values.filter(result.values["asset"] == "CCC")
    assert ccc.height == 2 and ccc["close"].to_list() == [20., None]
    d = result.diagnostics
    assert d.filter(d["code"] == "missing_value").select("asset", "session").rows() == [("CCC", date(2024, 6, 4))]
    assert d.filter(d["code"] == "session_not_reported").select("asset", "session").rows() == [("CCC", date(2024, 6, 7))]


def test_bounded_retries_with_timeout(provider):
    script, calls, sleeps = provider
    script["AAA"] = [urllib.error.URLError("reset"), urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO()), NY]
    result = rt.adapters.download_yahoo(["AAA"], start=date(2024, 6, 3), end=date(2024, 6, 7), timeout=2.5, backoff_seconds=.5)
    assert [c[1] for c in calls] == [2.5, 2.5, 2.5] and sleeps == [.5, 1.]
    assert calls[0][2].startswith("Mozilla")
    assert result.metadata["instruments"]["AAA"]["attempts"] == 3
    script["AAA"] = [TimeoutError("slow")]*3
    with pytest.raises(ConnectionError, match="after 3 attempts"):
        rt.adapters.download_yahoo(["AAA"], start=date(2024, 6, 3), end=date(2024, 6, 7))


def test_client_errors_fail_without_retry_or_dropping_assets(provider):
    script, calls, _ = provider
    script.update({"AAA": NY, "BAD": urllib.error.HTTPError("u", 404, "missing", {}, io.BytesIO(b'{"chart":{"error":"No data found"}}'))})
    with pytest.raises(ValueError, match="HTTP 404"):
        rt.adapters.download_yahoo(["AAA", "BAD"], start=date(2024, 6, 3), end=date(2024, 6, 7))
    assert len(calls) == 2
    script["BAD"] = chart("BAD", "", [], [], [], [], error={"code": "Not Found"})
    with pytest.raises(ValueError, match="error for BAD"):
        rt.adapters.download_yahoo(["BAD"], start=date(2024, 6, 3), end=date(2024, 6, 7))


@pytest.mark.parametrize("edit, match", [
    (lambda r: r["timestamp"].__setitem__(2, r["timestamp"][1] + 3600), "duplicate"),
    (lambda r: r["indicators"]["quote"][0]["close"].pop(), "mismatched"),
    (lambda r: r["indicators"].pop("adjclose"), "adjclose"),
    (lambda r: r["indicators"]["quote"][0]["close"].__setitem__(1, -1.), "invalid Yahoo close"),
    (lambda r: r["meta"].__setitem__("dataGranularity", "1wk"), "1wk"),
    (lambda r: r.__setitem__("timestamp", [stamp(2024, 5, 31, 13, 30)]) or r["indicators"]["quote"][0].update(close=[1.], volume=[1])
        or r["indicators"]["adjclose"][0].update(adjclose=[1.]), "no AAA observations"),
])
def test_malformed_duplicate_and_empty_responses_raise(provider, edit, match):
    script, _, _ = provider
    payload = json.loads(json.dumps(NY))
    edit(payload["chart"]["result"][0])
    script["AAA"] = payload
    with pytest.raises(ValueError, match=match):
        rt.adapters.download_yahoo(["AAA"], start=date(2024, 6, 3), end=date(2024, 6, 7))


@pytest.mark.parametrize("kwargs", [dict(assets="AAA"), dict(assets=[]), dict(assets=["A", "A"]),
    dict(start=datetime(2024, 6, 3)), dict(start=date(2024, 6, 8)), dict(timeout=0), dict(max_attempts=0)])
def test_invalid_requests_raise_before_network(provider, kwargs):
    _, calls, _ = provider
    args = dict(assets=["AAA"], start=date(2024, 6, 3), end=date(2024, 6, 7)) | kwargs
    with pytest.raises(ValueError):
        rt.adapters.download_yahoo(args.pop("assets"), **args)
    assert calls == []


def test_import_does_not_load_network_client():
    subprocess.run([sys.executable, "-c", "import research_toolkit, sys; "
                    "assert not {'urllib.request', 'http.client'} & set(sys.modules)"], check=True)


def test_rows_follow_request_order_and_payloads_are_retained(provider):
    script, _, _ = provider
    script.update({"AAA": NY, "BBB.T": TOKYO})
    result = rt.adapters.download_yahoo(["BBB.T", "AAA"], start=date(2024, 6, 3), end=date(2024, 6, 7))
    assert result.values["asset"].unique(maintain_order=True).to_list() == ["BBB.T", "AAA"]
    assert result.values.filter(result.values["asset"] == "AAA")["session"].is_sorted()
    # Decoded payloads are kept as reported (padding bars included) for yahoo_chart.
    assert result.responses == {"AAA": NY, "BBB.T": TOKYO}
    assert result.sessions == {}  # No calendar, so no validated session tables.


def test_bar_retrieved_before_its_session_ended_is_flagged(provider):
    script, _, _ = provider
    payload = json.loads(json.dumps(NY))
    payload["chart"]["result"][0]["meta"]["currentTradingPeriod"] = {"regular": {"end": stamp(2024, 6, 10, 20)}}
    script["AAA"] = payload  # Retrieval is 12:00 UTC on 10 June, before that day's 20:00 UTC close.
    result = rt.adapters.download_yahoo(["AAA"], start=date(2024, 6, 3), end=date(2024, 6, 10))
    flagged = result.diagnostics.filter(result.diagnostics["code"] == "session_open_at_retrieval")
    assert flagged.select("asset", "session").rows() == [("AAA", date(2024, 6, 10))]


def test_http_errors_are_closed_after_retry_or_failure(provider):
    script, _, _ = provider
    busy, missing = io.BytesIO(), io.BytesIO(b'{"chart":{"error":"No data found"}}')
    script["AAA"] = [urllib.error.HTTPError("u", 503, "busy", {}, busy), NY]
    rt.adapters.download_yahoo(["AAA"], start=date(2024, 6, 3), end=date(2024, 6, 7), backoff_seconds=0)
    script["BAD"] = urllib.error.HTTPError("u", 404, "missing", {}, missing)
    with pytest.raises(ValueError, match="HTTP 404"):
        rt.adapters.download_yahoo(["BAD"], start=date(2024, 6, 3), end=date(2024, 6, 7))
    assert busy.closed and missing.closed
