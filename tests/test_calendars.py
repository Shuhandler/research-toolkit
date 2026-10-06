"""Calendar-aware validation with the real exchange_calendars rules (offline, fixed dates)."""
from datetime import date, datetime, timezone

import polars as pl
import pytest

import research_toolkit as rt
from research_toolkit import adapters
from test_series_performance import KW, pnl_table
from test_yahoo_download import chart, provider  # noqa: F401  (provider is a fixture)

pytest.importorskip("exchange_calendars")

UTC = timezone.utc
HOUR = {"America/New_York": 14, "Europe/London": 8, "Asia/Tokyo": 0}  # Bar stamps on the local session date.


def series(*days, **kwargs):
    return rt.series_performance(pnl_table([1.]*(len(days)-1), list(days)), **(KW | kwargs))


def bars(symbol, zone, days, first_trade=None):
    stamps = [int(datetime(d.year, d.month, d.day, HOUR[zone], 30, tzinfo=UTC).timestamp()) for d in days]
    payload = chart(symbol, zone, stamps, [10.]*len(days), [9.]*len(days), [100]*len(days))
    if first_trade is not None:
        payload["chart"]["result"][0]["meta"]["firstTradeDate"] = int(
            datetime(first_trade.year, first_trade.month, first_trade.day, 14, 30, tzinfo=UTC).timestamp())
    return payload


def download(script, monkeypatch, responses, start, end, retrieved=datetime(2025, 1, 2, tzinfo=UTC), **kwargs):
    script.update(responses)
    monkeypatch.setattr(adapters, "_now", lambda: retrieved)
    kwargs.setdefault("calendars", {a: "XNYS" for a in responses})
    return rt.adapters.download_yahoo(list(responses), start=start, end=end, **kwargs)


# --- series_performance -------------------------------------------------------------

def test_no_calendar_keeps_continuity_only_behavior():
    report = series(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5))  # Skips Thursday unnoticed.
    assert report.metadata["calendar_validation"]["status"] == "not_validated"
    assert report.metadata["interval_validation"] == "ordered_contiguous_no_trading_calendar"


def test_skipped_ordinary_session_is_named():
    with pytest.raises(ValueError, match=r"2024-01-03 to 2024-01-05 skips XNYS session\(s\) \['2024-01-04'\]"):
        series(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5), calendar="XNYS")


def test_weekend_holiday_and_early_close_intervals_are_valid():
    # Fri->Mon weekend, Fri->Tue over Martin Luther King Day, Wed->Fri over Thanksgiving,
    # where Friday 29 November is a shortened (13:00) session.
    report = series(date(2024, 1, 11), date(2024, 1, 12), date(2024, 1, 16), calendar="XNYS")
    check = report.metadata["calendar_validation"]
    assert check["status"] == "validated" and check["calendar"] == "XNYS"
    assert (check["expected_sessions"], check["observed_sessions"]) == (3, 3)
    assert check["library"] == "exchange_calendars" and check["library_version"]
    series(date(2024, 11, 27), date(2024, 11, 29), date(2024, 12, 2), calendar="XNYS")


def test_first_interval_is_checked():
    with pytest.raises(ValueError, match=r"skips XNYS session\(s\) \['2024-01-03'\]"):
        series(date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 5), calendar="XNYS")
    with pytest.raises(ValueError, match="starts on 2024-01-01, which is not a XNYS session"):
        series(date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), calendar="XNYS")


def test_unexpected_dates_duplicates_and_overlaps():
    with pytest.raises(ValueError, match="ends on 2024-01-15, which is not a XNYS session"):
        series(date(2024, 1, 11), date(2024, 1, 12), date(2024, 1, 15), calendar="XNYS")
    duplicate = pl.concat([pnl_table([1., 1.], [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]),
                           pnl_table([1.], [date(2024, 1, 3), date(2024, 1, 4)])])
    with pytest.raises(ValueError, match=r"duplicate sessions: \['2024-01-04'\]"):
        rt.series_performance(duplicate, calendar="XNYS", **KW)
    overlap = pnl_table([1., 1.], [date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 5)]).with_columns(
        pl.Series("period_start", [date(2024, 1, 2), date(2024, 1, 3)]))
    with pytest.raises(ValueError, match="overlaps the previous interval ending 2024-01-04"):
        rt.series_performance(overlap, calendar="XNYS", **KW)


def test_multi_session_intervals_are_declared_not_missing():
    weekly = (date(2024, 1, 5), date(2024, 1, 12), date(2024, 1, 19))
    with pytest.raises(ValueError, match="frequency='multi_session'"):
        series(*weekly, calendar="XNYS")
    report = series(*weekly, calendar="XNYS", frequency="multi_session", periods_per_year=52)
    check = report.metadata["calendar_validation"]
    assert report.metadata["frequency"] == "multi_session"
    assert check["sessions_per_interval"] == {"min": 4, "max": 5}  # MLK Day removes one session.
    assert (check["expected_sessions"], check["observed_sessions"]) == (10, 3)
    with pytest.raises(ValueError, match="not a XNYS session"):
        series(date(2024, 1, 5), date(2024, 1, 15), calendar="XNYS", frequency="multi_session")
    bench = report.daily.select("period_start", "session").with_columns(pl.lit(.01).alias("simple_return"))
    meta = {"source": "s", "basis": "b", "currency": "USD"}
    with pytest.raises(ValueError, match="frequency must match"):
        rt.series_performance(pnl_table([1., 1.], list(weekly)), benchmark=bench,
            benchmark_metadata=meta | {"frequency": "1d"}, frequency="multi_session", **KW)
    rt.series_performance(pnl_table([1., 1.], list(weekly)), benchmark=bench,
        benchmark_metadata=meta | {"frequency": "multi_session"}, frequency="multi_session", **KW)


def test_multi_exchange_portfolio_requires_explicit_reporting_calendar():
    days = (date(2024, 5, 3), date(2024, 5, 6), date(2024, 5, 7))  # 6 May: London closed, New York open.
    for ambiguous in (["XNYS", "XLON"], {"A": "XNYS", "B": "XLON"}):
        with pytest.raises(ValueError, match="ONE explicit reporting calendar"):
            series(*days, calendar=ambiguous)
    with pytest.raises(ValueError, match="explicit combine"):
        rt.trading_calendar(["XNYS", "XLON"], start=date(2024, 5, 1), end=date(2024, 5, 31))
    union = rt.trading_calendar(["XNYS", "XLON"], start=date(2024, 5, 1), end=date(2024, 5, 31), combine="union")
    report = series(*days, calendar=union)
    assert report.metadata["calendar_validation"]["calendar"] == "union(XNYS,XLON)"
    inter = rt.trading_calendar(["XNYS", "XLON"], start=date(2024, 5, 1), end=date(2024, 5, 31), combine="intersection")
    with pytest.raises(ValueError, match=r"2024-05-06, which is not a intersection\(XNYS,XLON\) session"):
        series(*days, calendar=inter)
    series(date(2024, 5, 3), date(2024, 5, 7), calendar=inter)
    with pytest.raises(ValueError, match="not a XLON session"):
        series(*days, calendar="XLON")
    with pytest.raises(ValueError, match="covers 2024-05-01 to 2024-05-31"):
        series(date(2024, 4, 30), date(2024, 5, 1), calendar=union)


def test_trading_calendar_uses_utc_closes_with_dst_and_early_closes():
    cal = rt.trading_calendar("XNYS", start=date(2024, 3, 8), end=date(2024, 3, 11))
    assert cal.sessions["close_at"].to_list() == [datetime(2024, 3, 8, 21, tzinfo=UTC), datetime(2024, 3, 11, 20, tzinfo=UTC)]
    early = rt.trading_calendar("XNYS", start=date(2024, 11, 29), end=date(2024, 11, 29))
    assert early.sessions["close_at"].to_list() == [datetime(2024, 11, 29, 18, tzinfo=UTC)]
    with pytest.raises(ValueError, match="unknown exchange calendar"):
        rt.trading_calendar("NOT_AN_EXCHANGE", start=date(2024, 1, 2), end=date(2024, 1, 3))


# --- download_yahoo -------------------------------------------------------------------

JAN = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]  # Thursday 4 January missing.


def test_session_missing_from_every_asset_is_detected(provider, monkeypatch):
    script, _, _ = provider
    responses = {"AAA": bars("AAA", "America/New_York", JAN), "BBB": bars("BBB", "America/New_York", JAN)}
    # Without calendars, a date every asset lacks cannot be noticed.
    script.update(responses)
    plain = rt.adapters.download_yahoo(["AAA", "BBB"], start=date(2024, 1, 2), end=date(2024, 1, 5))
    assert plain.diagnostics.is_empty() and plain.metadata["calendar_validation"]["status"] == "not_validated"
    with pytest.raises(ValueError, match=r"AAA: missing_session \['2024-01-04'\] \| BBB: missing_session \['2024-01-04'\]"):
        download(script, monkeypatch, responses, date(2024, 1, 2), date(2024, 1, 5))
    report = download(script, monkeypatch, responses, date(2024, 1, 2), date(2024, 1, 5), incomplete="report")
    assert report.metadata["calendar_validation"]["status"] == "incomplete"
    missing = report.diagnostics.filter(pl.col("code") == "missing_session")
    assert missing.select("asset", "session").rows() == [("AAA", date(2024, 1, 4)), ("BBB", date(2024, 1, 4))]
    assert report.values.height == 6  # Nothing filled.
    check = report.metadata["instruments"]["AAA"]["calendar_validation"]
    assert (check["expected_sessions"], check["observed_sessions"], check["calendar"]) == (4, 3, "XNYS")
    assert check["library_version"] and check["validation_end"] == "2024-01-05"


def test_weekend_and_holiday_are_not_missing(provider, monkeypatch):
    script, _, _ = provider
    days = [date(2024, 1, 12), date(2024, 1, 16)]
    result = download(script, monkeypatch, {"AAA": bars("AAA", "America/New_York", days)}, date(2024, 1, 12), date(2024, 1, 16))
    assert result.metadata["calendar_validation"]["status"] == "complete"
    assert result.metadata["instruments"]["AAA"]["calendar_validation"]["expected_sessions"] == 2


def test_early_close_before_and_after_close(provider, monkeypatch):
    script, _, _ = provider
    done = [date(2024, 11, 26), date(2024, 11, 27), date(2024, 11, 29)]
    args = (date(2024, 11, 26), date(2024, 11, 29))
    after = download(script, monkeypatch, {"AAA": bars("AAA", "America/New_York", done)}, *args,
                     as_of=datetime(2024, 11, 29, 18, 30, tzinfo=UTC))  # Early close was 18:00 UTC.
    assert after.metadata["instruments"]["AAA"]["calendar_validation"]["expected_sessions"] == 3
    before = download(script, monkeypatch, {"AAA": bars("AAA", "America/New_York", done[:2])}, *args,
                      as_of=datetime(2024, 11, 29, 17, 30, tzinfo=UTC))
    check = before.metadata["instruments"]["AAA"]["calendar_validation"]
    assert check["status"] == "complete" and check["not_yet_closed"] == ["2024-11-29"]
    assert before.diagnostics.filter(pl.col("code") == "session_not_closed")["session"].to_list() == [date(2024, 11, 29)]
    with pytest.raises(ValueError, match=r"session_open_at_validation \['2024-11-29'\]"):
        download(script, monkeypatch, {"AAA": bars("AAA", "America/New_York", done)}, *args,
                 as_of=datetime(2024, 11, 29, 17, 30, tzinfo=UTC))


def test_daylight_saving_close_times(provider, monkeypatch):
    script, _, _ = provider
    # 8 March closes 21:00 UTC (EST); 11 March closes 20:00 UTC (EDT).
    friday = download(script, monkeypatch, {"AAA": bars("AAA", "America/New_York", [date(2024, 3, 7)])},
                      date(2024, 3, 7), date(2024, 3, 8), as_of=datetime(2024, 3, 8, 20, 30, tzinfo=UTC))
    assert friday.metadata["instruments"]["AAA"]["calendar_validation"]["not_yet_closed"] == ["2024-03-08"]
    with pytest.raises(ValueError, match=r"missing_session \['2024-03-11'\]"):
        download(script, monkeypatch, {"AAA": bars("AAA", "America/New_York", [date(2024, 3, 8)])},
                 date(2024, 3, 8), date(2024, 3, 11), as_of=datetime(2024, 3, 11, 20, 30, tzinfo=UTC))
    with pytest.raises(ValueError, match="as_of cannot be later"):
        download(script, monkeypatch, {"AAA": bars("AAA", "America/New_York", [date(2024, 3, 8)])},
                 date(2024, 3, 8), date(2024, 3, 8), retrieved=datetime(2024, 3, 8, 20, tzinfo=UTC),
                 as_of=datetime(2024, 3, 8, 22, tzinfo=UTC))


def test_each_asset_uses_its_own_exchange(provider, monkeypatch):
    script, _, _ = provider
    responses = {"AAA": bars("AAA", "America/New_York", [date(2024, 5, 3), date(2024, 5, 6), date(2024, 5, 7)]),
                 "LLL.L": bars("LLL.L", "Europe/London", [date(2024, 5, 3), date(2024, 5, 7)])}
    result = download(script, monkeypatch, responses, date(2024, 5, 3), date(2024, 5, 7),
                      calendars={"AAA": "XNYS", "LLL.L": "XLON"})
    assert result.metadata["calendar_validation"]["status"] == "complete"
    assert result.metadata["instruments"]["LLL.L"]["calendar_validation"]["expected_sessions"] == 2
    with pytest.raises(ValueError, match="uses America/New_York but Yahoo reports exchange timezone Europe/London"):
        download(script, monkeypatch, responses, date(2024, 5, 3), date(2024, 5, 7))
    with pytest.raises(ValueError, match="map every requested asset"):
        download(script, monkeypatch, responses, date(2024, 5, 3), date(2024, 5, 7), calendars={"AAA": "XNYS"})


def test_listing_dates_and_unknown_coverage(provider, monkeypatch):
    script, _, _ = provider
    listed = [date(2024, 1, 4), date(2024, 1, 5)]
    args = (date(2024, 1, 2), date(2024, 1, 5))
    result = download(script, monkeypatch, {"NEW": bars("NEW", "America/New_York", listed, first_trade=date(2024, 1, 4))}, *args)
    check = result.metadata["instruments"]["NEW"]["calendar_validation"]
    assert (check["status"], check["coverage_start"], check["coverage_source"]) == ("complete", "2024-01-04", "provider_first_trade_date")
    declared = download(script, monkeypatch, {"NEW": bars("NEW", "America/New_York", listed)}, *args,
                        coverage={"NEW": {"start": date(2024, 1, 4), "source": "exchange listing notice"}})
    assert declared.metadata["instruments"]["NEW"]["calendar_validation"]["coverage_source"] == "exchange listing notice"
    unknown = download(script, monkeypatch, {"NEW": bars("NEW", "America/New_York", listed)}, *args)
    check = unknown.metadata["instruments"]["NEW"]["calendar_validation"]
    assert check["status"] == "coverage_uncertain" and check["coverage_uncertain"] == ["2024-01-02", "2024-01-03"]
    assert unknown.metadata["calendar_validation"]["status"] == "coverage_uncertain"
    with pytest.raises(ValueError, match=r"bar_outside_declared_coverage \['2024-01-04'\]"):
        download(script, monkeypatch, {"NEW": bars("NEW", "America/New_York", listed)}, *args,
                 coverage={"NEW": {"start": date(2024, 1, 5), "source": "wrong listing date"}})
    # Unknown listing excuses only sessions before the first bar, never later gaps.
    with pytest.raises(ValueError, match=r"missing_session \['2024-01-04'\]"):
        download(script, monkeypatch, {"NEW": bars("NEW", "America/New_York", [date(2024, 1, 3), date(2024, 1, 5)])}, *args)


def test_suspensions_need_explanation_and_holiday_bars_are_unexpected(provider, monkeypatch):
    script, _, _ = provider
    responses = {"AAA": bars("AAA", "America/New_York", JAN)}
    reason = pl.DataFrame({"asset": ["AAA"], "session": [date(2024, 1, 4)], "reason": ["trading halt notice 123"]})
    result = download(script, monkeypatch, responses, date(2024, 1, 2), date(2024, 1, 5), exclusions=reason)
    check = result.metadata["instruments"]["AAA"]["calendar_validation"]
    assert check["status"] == "complete" and check["excluded"] == [{"session": "2024-01-04", "reason": "trading halt notice 123"}]
    with pytest.raises(ValueError, match="not a XNYS session"):
        download(script, monkeypatch, responses, date(2024, 1, 2), date(2024, 1, 5),
                 exclusions=reason.with_columns(pl.lit(date(2024, 1, 6)).alias("session")))
    holiday = {"AAA": bars("AAA", "America/New_York", [date(2024, 1, 12), date(2024, 1, 15), date(2024, 1, 16)])}
    with pytest.raises(ValueError, match=r"unexpected_session \['2024-01-15'\]"):
        download(script, monkeypatch, holiday, date(2024, 1, 12), date(2024, 1, 16))


def test_calendar_options_require_calendars(provider):
    for option in (dict(as_of=datetime(2024, 1, 5, tzinfo=UTC)), dict(incomplete="report"),
                   dict(coverage={"AAA": {"start": date(2024, 1, 2), "source": "s"}})):
        with pytest.raises(ValueError, match="require calendars"):
            rt.adapters.download_yahoo(["AAA"], start=date(2024, 1, 2), end=date(2024, 1, 5), **option)
