"""Optional exchange trading calendars. The calendar library loads only when requested."""

from collections.abc import Sequence
from copy import deepcopy
from datetime import date, timedelta
from functools import lru_cache

import polars as pl

from ._results import TradingCalendar

CALENDAR_SCHEMA = {"session": pl.Date, "close_at": pl.Datetime("us", "UTC")}


def _library():
    try:
        import exchange_calendars
    except ImportError as exc:
        raise ImportError("Calendar validation requires the optional extra: pip install -e '.[calendar]'") from exc
    return exchange_calendars


@lru_cache(maxsize=32)
def _exchange_sessions(name, start, end):
    """Sessions and UTC closes (early closes and DST included) for start <= session <= end."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("calendar identifiers must be nonblank exchange_calendars names such as 'XNYS'")
    xcals = _library()
    try:
        # Padding keeps both requested endpoints inside the library's generated bounds.
        cal = xcals.get_calendar(name, start=(start - timedelta(days=14)).isoformat(),
                                 end=(end + timedelta(days=14)).isoformat())
    except xcals.errors.InvalidCalendarName as exc:
        raise ValueError(f"unknown exchange calendar {name!r}") from exc
    except Exception as exc:
        raise ValueError(f"calendar {name!r} cannot cover {start} to {end}: {exc}") from exc
    rows = [(session.date(), close.to_pydatetime()) for session, close in cal.closes.items()
            if start <= session.date() <= end]
    zone = getattr(cal.tz, "key", None) or str(cal.tz)
    return tuple(rows), {"calendar": cal.name, "requested_identifier": name, "timezone": zone,
                         "library": "exchange_calendars", "library_version": xcals.__version__}


def _check_dates(start, end):
    if any(type(d) is not date for d in (start, end)) or start > end:
        raise ValueError("start and end must be datetime.date values with start <= end")


def trading_calendar(calendars, *, start, end, combine=None) -> TradingCalendar:
    """Explicit reporting sessions from one exchange calendar, or a declared union/intersection.

    One identifier (for example 'XNYS') needs no combine rule. Several identifiers
    require combine='union' or 'intersection'; the toolkit never chooses one.
    ``close_at`` is the latest close among the exchanges trading that session.
    """
    _check_dates(start, end)
    names = [calendars] if isinstance(calendars, str) else calendars
    if not isinstance(names, Sequence) or isinstance(names, str) or not names or len(set(names)) != len(names):
        raise ValueError("calendars must be one identifier or a list of unique identifiers")
    if len(names) == 1 and combine is not None:
        raise ValueError("combine applies only to several calendars")
    if len(names) > 1 and combine not in {"union", "intersection"}:
        raise ValueError("several calendars require an explicit combine='union' or 'intersection'")
    members, closes = [], []
    for name in names:
        rows, meta = _exchange_sessions(name, start, end)
        members.append(meta)
        closes.append(dict(rows))
    days = set(closes[0]).union(*closes[1:]) if combine != "intersection" else set(closes[0]).intersection(*closes[1:])
    rows = [(d, max(c[d] for c in closes if d in c)) for d in sorted(days)]
    sessions = pl.DataFrame(rows, schema=CALENDAR_SCHEMA, orient="row")
    if sessions.is_empty():
        raise ValueError(f"no trading sessions from {start} to {end}")
    return TradingCalendar(sessions, {
        "calendar": members[0]["calendar"] if len(members) == 1 else f"{combine}({','.join(m['calendar'] for m in members)})",
        "members": deepcopy(members), "combine": combine, "start": start.isoformat(), "end": end.isoformat(),
        "library": "exchange_calendars", "library_version": members[0]["library_version"],
        "n_sessions": sessions.height})


def _reporting_sessions(calendar, start, end):
    """Sorted session dates covering start..end and the calendar's metadata."""
    if isinstance(calendar, str):
        calendar = trading_calendar(calendar, start=start, end=end)
    if not isinstance(calendar, TradingCalendar):
        raise ValueError("a portfolio needs ONE explicit reporting calendar: an identifier such as 'XNYS' or "
                         "rt.trading_calendar([...], combine='union' or 'intersection'); the toolkit does not choose")
    meta = calendar.metadata
    if date.fromisoformat(meta["start"]) > start or date.fromisoformat(meta["end"]) < end:
        raise ValueError(f"reporting calendar covers {meta['start']} to {meta['end']}, not {start} to {end}")
    days = calendar.sessions["session"].to_list()
    return [d for d in days if start <= d <= end], deepcopy(meta)
