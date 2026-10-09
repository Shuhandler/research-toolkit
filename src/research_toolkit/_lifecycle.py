"""Security identities, dated lifecycles and provider-independent corporate-action inputs.

Validation only: no ledger postings, I/O or provider access. The shared ledger
consumes the compiled schedule in ``_actions``; analytical returns live here.
"""

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import math
from zoneinfo import ZoneInfo

import polars as pl

from ._portfolio import _number

S, D, F = pl.String, pl.Date, pl.Float64
UTC = pl.Datetime("us", "UTC")

SECURITY_SCHEMA = {"asset": S, "security_type": S, "first_session": D, "last_session": D,
                   "terminated_date": D, "unquoted_valuation": S, "source": S}
SUSPENSION_SCHEMA = {"asset": S, "first_session": D, "last_session": D, "reason": S, "source": S}
ALIAS_SCHEMA = {"asset": S, "ticker": S, "exchange": S, "valid_from": D, "valid_to": D, "source": S}
ACTION_SCHEMA = {"action_id": S, "action_type": S, "source_asset": S, "announced_at": UTC,
                 "effective_date": D, "record_date": D, "entitlement_basis": S,
                 "provider_label": S, "classification_source": S}
LEG_SCHEMA = {"action_id": S, "leg_id": S, "leg_type": S, "asset": S,
              "units_per_source_share": F, "cash_per_source_share": F, "accrual_per_day": F,
              "accrual_start": D, "accrual_end": D, "due_date": D, "fraction_policy": S,
              "cash_in_lieu_price": F}
WARRANT_SCHEMA = {"asset": S, "underlying_asset": S, "units_per_warrant": F, "exercise_price": F,
                  "exercisable_from": D, "expiry_date": D, "exercise_style": S, "settlement": S,
                  "source": S}
MARK_SCHEMA = {"session": D, "asset": S, "mark": F, "method": S, "source": S}
EXERCISE_SCHEMA = {"exercise_id": S, "decision_session": D, "session": D, "warrant": S,
                   "warrants": F, "delivery_date": D, "fee": F}

# Table name -> (schema, nullable columns, unique key, sort key).
LIFECYCLE_TABLES = {
    "securities": (SECURITY_SCHEMA, {"last_session", "terminated_date"}, ["asset"], ["asset"]),
    "suspensions": (SUSPENSION_SCHEMA, set(), ["asset", "first_session"], ["asset", "first_session"]),
    "aliases": (ALIAS_SCHEMA, {"valid_to"}, ["asset", "valid_from"], ["asset", "valid_from"]),
    "corporate_actions": (ACTION_SCHEMA, {"record_date", "provider_label"}, ["action_id"],
                          ["effective_date", "action_id"]),
    "action_legs": (LEG_SCHEMA, {"asset", "units_per_source_share", "cash_per_source_share",
                                 "accrual_per_day", "accrual_start", "accrual_end",
                                 "fraction_policy", "cash_in_lieu_price"},
                    ["action_id", "leg_id"], ["action_id", "leg_id"]),
    "warrants": (WARRANT_SCHEMA, set(), ["asset"], ["asset"]),
    "valuation_marks": (MARK_SCHEMA, set(), ["session", "asset"], ["session", "asset"]),
}
SECURITY_TYPES = {"equity", "warrant"}
ACTION_TYPES = {"cash_acquisition", "stock_acquisition", "distribution"}


def _typed(frame, schema, name, keys, nullable=frozenset()):
    """Exact schema, unique keys, no blank strings or nonfinite floats; nulls only where declared."""
    if not isinstance(frame, pl.DataFrame):
        raise ValueError(f"{name} must be a Polars DataFrame")
    if frame.schema != schema:
        raise ValueError(f"{name} schema must be {schema}; got {dict(frame.schema)}")
    for col in schema:
        if col not in nullable and frame[col].null_count():
            raise ValueError(f"{name}.{col} contains null inputs")
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"{name} has duplicate keys: {keys}")
    for col, dtype in schema.items():
        values = frame[col].drop_nulls()
        if dtype == pl.Float64 and not values.is_finite().all():
            raise ValueError(f"{name}.{col} must be finite")
        if dtype == pl.String and values.str.strip_chars().eq("").any():
            raise ValueError(f"{name}.{col} must not be blank")
    return frame.select(list(schema)).clone()


def corporate_action_inputs(*, securities, suspensions=(), aliases=(), actions=(), legs=(),
                            warrants=(), valuation_marks=()) -> dict:
    """Build typed lifecycle/action tables from lists of row dictionaries.

    Omitted nullable fields become null; every required field must be present.
    Returns keyword arguments for ``prepare_market_data``. No validation of
    economics happens here; ``prepare_market_data`` performs all checks.
    """
    supplied = {"securities": securities, "suspensions": suspensions, "aliases": aliases,
                "corporate_actions": actions, "action_legs": legs, "warrants": warrants,
                "valuation_marks": valuation_marks}
    out = {}
    for name, rows in supplied.items():
        schema, nullable, _, _ = LIFECYCLE_TABLES[name]
        if isinstance(rows, pl.DataFrame):
            out[name] = rows
            continue
        rows = list(rows)
        if any(not isinstance(r, Mapping) for r in rows):
            raise ValueError(f"{name} rows must be mappings")
        for r in rows:
            unknown, missing = set(r) - set(schema), set(schema) - nullable - set(r)
            if unknown or missing:
                raise ValueError(f"{name} row has unknown fields {sorted(unknown)} or missing required fields {sorted(missing)}")
        out[name] = pl.DataFrame([{c: r.get(c) for c in schema} for r in rows], schema=schema, orient="row")
    return out


def _quote_plan(securities, suspensions, dates):
    """Expected (session, asset) quotes and the lifecycle status of every other cell."""
    suspended = defaultdict(list)
    for r in suspensions.iter_rows(named=True):
        suspended[r["asset"]].append((r["first_session"], r["last_session"]))
    status = {}
    for r in securities.iter_rows(named=True):
        a = r["asset"]
        for d in dates:
            if r["terminated_date"] is not None and d >= r["terminated_date"]:
                s = "terminated"
            elif d < r["first_session"]:
                s = "not_listed"
            elif r["last_session"] is not None and d > r["last_session"]:
                s = "unquoted_after_last_session"
            elif any(lo <= d <= hi for lo, hi in suspended[a]):
                s = "suspended"
            else:
                s = "quoted"
            status[d, a] = s
    return status


def _local_midnight(day, zone):
    return datetime.combine(day, time.min, tzinfo=zone).astimezone(timezone.utc)


def _validate_lifecycle(prices, sessions, splits, dividends, raw, meta):
    """Validate lifecycle/action tables against prices; return tables and diagnostics."""
    tables = {}
    for name, (schema, nullable, keys, order) in LIFECYCLE_TABLES.items():
        frame = raw.get(name)
        if frame is None:
            if name == "securities":
                raise ValueError("lifecycle inputs require a securities table covering every asset")
            frame = pl.DataFrame(schema=schema)
        tables[name] = _typed(frame, schema, name, keys, nullable).sort(order)
    sec, susp, alias = tables["securities"], tables["suspensions"], tables["aliases"]
    actions, legs, warrants, marks = (tables[n] for n in ("corporate_actions", "action_legs", "warrants", "valuation_marks"))
    zone = ZoneInfo(meta["timezone"])
    dates = sessions["session"].to_list()
    closes = dict(sessions.select("session", "close_at").iter_rows())
    info = {r["asset"]: r for r in sec.iter_rows(named=True)}
    price_assets = set(prices["asset"])
    if not price_assets <= set(info):
        raise ValueError(f"securities must cover every priced asset; missing {sorted(price_assets - set(info))}")
    currencies = meta["asset_currencies"]
    if not isinstance(currencies, dict) or set(currencies) != set(info):
        raise ValueError("asset_currencies must cover exactly the securities table assets")
    for a, r in info.items():
        if r["security_type"] not in SECURITY_TYPES:
            raise ValueError(f"security {a} has unsupported security_type {r['security_type']!r}")
        if r["unquoted_valuation"] not in {"none", "supplied_mark"}:
            raise ValueError(f"security {a} unquoted_valuation must be 'none' or 'supplied_mark'")
        last, end = r["last_session"], r["terminated_date"]
        if last is not None and last < r["first_session"]:
            raise ValueError(f"security {a} last_session precedes first_session")
        if end is not None and (last is None or last >= end):
            raise ValueError(f"terminated security {a} needs a last_session strictly before terminated_date")
    for r in susp.iter_rows(named=True):
        lifecycle = info.get(r["asset"])
        if lifecycle is None or r["first_session"] > r["last_session"]:
            raise ValueError(f"suspension of {r['asset']} has an unknown asset or reversed dates")
        if r["first_session"] not in closes or r["last_session"] not in closes:
            raise ValueError("suspension dates must be supplied sessions")
    for a, rows in susp.group_by("asset"):
        spans = sorted(rows.select("first_session", "last_session").iter_rows())
        if any(b[0] <= prev[1] for prev, b in zip(spans, spans[1:])):
            raise ValueError(f"overlapping suspensions for {a[0]}")
    status = _quote_plan(sec, susp, dates)
    expected = {k for k, s in status.items() if s == "quoted"}
    actual = set(prices.select("session", "asset").iter_rows())
    if actual - expected:
        bad = sorted(actual - expected)[:5]
        raise ValueError("prices supplied where the lifecycle expects no quote: "
                         + ", ".join(f"{a} on {d} ({status[d, a]})" for d, a in bad))
    if expected - actual:
        raise ValueError(f"missing data: expected quotes absent for {sorted(expected - actual)[:5]}")
    # Aliases: dated, non-overlapping per security and per (ticker, exchange).
    for r in alias.iter_rows(named=True):
        if r["asset"] not in info or (r["valid_to"] is not None and r["valid_to"] < r["valid_from"]):
            raise ValueError(f"alias {r['ticker']} has an unknown asset or reversed validity")
    for key in (["asset"], ["ticker", "exchange"]):
        for group, rows in alias.group_by(key):
            spans = sorted((r["valid_from"], r["valid_to"] or date.max) for r in rows.iter_rows(named=True))
            if any(b[0] <= prev[1] for prev, b in zip(spans, spans[1:])):
                raise ValueError(f"overlapping alias validity for {group}; a ticker cannot join different securities")
    # Ordinary actions cannot refer to extinguished instruments.
    for table, col in ((splits, "effective_session"), (dividends, "ex_session")):
        for r in table.iter_rows(named=True):
            life = info[r["asset"]]
            if life["terminated_date"] is not None and r[col] >= life["terminated_date"]:
                raise ValueError(f"action {r['action_id']} applies to {r['asset']} after termination")
            if table is dividends and life["security_type"] != "equity":
                raise ValueError(f"dividend {r['action_id']} requires an equity security")
    split_keys = set(splits.select("asset", "effective_session").iter_rows())
    # Warrants.
    warrant_terms = {r["asset"]: r for r in warrants.iter_rows(named=True)}
    if set(warrant_terms) != {a for a, r in info.items() if r["security_type"] == "warrant"}:
        raise ValueError("warrant terms must cover exactly the securities with security_type='warrant'")
    for a, w in warrant_terms.items():
        if info.get(w["underlying_asset"], {}).get("security_type") != "equity":
            raise ValueError(f"warrant {a} underlying must be an equity security")
        if w["settlement"] != "physical":
            raise ValueError(f"warrant {a}: only physical cash-funded settlement is supported (no net-share or cash settlement)")
        if w["exercise_style"] not in {"american", "european"}:
            raise ValueError(f"warrant {a}: exercise_style must be american or european")
        _number(w["units_per_warrant"], f"{a} units_per_warrant", positive=True)
        _number(w["exercise_price"], f"{a} exercise_price")
        if w["exercisable_from"] > w["expiry_date"]:
            raise ValueError(f"warrant {a} becomes exercisable after expiry")
        if info[a]["terminated_date"] != w["expiry_date"] + timedelta(days=1):
            raise ValueError(f"warrant {a} terminated_date must be the day after expiry_date")
        if any(u == w["underlying_asset"] and d <= w["expiry_date"] for u, d in split_keys):
            raise ValueError(f"warrant {a}: underlying splits before expiry would require adjusted warrant terms, "
                             "which are unsupported")
    # Corporate actions and their legs.
    leg_rows = defaultdict(list)
    for r in legs.iter_rows(named=True):
        leg_rows[r["action_id"]].append(r)
    if set(leg_rows) - set(actions["action_id"]):
        raise ValueError("action_legs reference unknown action_id")
    ids = splits["action_id"].to_list() + dividends["action_id"].to_list() + actions["action_id"].to_list()
    if len(ids) != len(set(ids)):
        raise ValueError("action_id must be unique across splits, dividends and corporate actions")
    terminal = {}
    effective = defaultdict(list)
    for r in actions.iter_rows(named=True):
        aid, kind, src = r["action_id"], r["action_type"], r["source_asset"]
        if kind not in ACTION_TYPES:
            raise ValueError(f"action {aid}: unsupported action_type {kind!r}; supported: {sorted(ACTION_TYPES)}")
        if info.get(src, {}).get("security_type") != "equity":
            raise ValueError(f"action {aid}: source_asset must be a known equity security")
        basis, eff, rec = r["entitlement_basis"], r["effective_date"], r["record_date"]
        if basis == "record_date":
            raise ValueError(f"action {aid}: record-date entitlement cannot be represented safely; "
                             "supply the ex-date (regular way or due bill) that determines entitlement")
        if r["announced_at"] > _local_midnight(eff, zone):
            raise ValueError(f"action {aid}: announcement must be available before its effective date begins")
        rows = leg_rows.get(aid, [])
        if not rows:
            raise ValueError(f"action {aid} has no consideration or distribution legs")
        cash = [x for x in rows if x["leg_type"] == "cash"]
        securities_legs = [x for x in rows if x["leg_type"] == "security"]
        if len(cash) + len(securities_legs) != len(rows):
            raise ValueError(f"action {aid}: leg_type must be 'cash' or 'security'")
        if kind in {"cash_acquisition", "stock_acquisition"}:
            if basis != "effective_date":
                raise ValueError(f"action {aid}: acquisitions use entitlement_basis='effective_date'")
            if info[src]["terminated_date"] != eff:
                raise ValueError(f"action {aid}: the acquired security's terminated_date must equal effective_date")
            if src in terminal:
                raise ValueError(f"security {src} has more than one terminal action")
            terminal[src] = aid
            if kind == "cash_acquisition" and (securities_legs or not cash):
                raise ValueError(f"action {aid}: cash acquisitions require only cash legs")
            if kind == "stock_acquisition" and not securities_legs:
                raise ValueError(f"action {aid}: stock acquisitions require at least one security leg")
        else:
            if basis not in {"ex_date", "due_bill"}:
                raise ValueError(f"action {aid}: distributions use entitlement_basis 'ex_date' or 'due_bill'")
            if rec is None or (basis == "ex_date" and rec < eff) or (basis == "due_bill" and rec >= eff):
                raise ValueError(f"action {aid}: record_date is inconsistent with {basis} entitlement "
                                 "(regular way: record on/after ex-date; due bill: record before ex-date)")
            if cash or not securities_legs:
                raise ValueError(f"action {aid}: distributions deliver securities; record cash as a dividend")
            if info[src]["terminated_date"] is not None and info[src]["terminated_date"] <= eff:
                raise ValueError(f"action {aid}: distribution after the parent terminated")
        effective[src, eff].append(aid)
        if (src, eff) in split_keys:
            raise ValueError(f"action {aid} coincides with a split of {src}; ordering would be ambiguous")
        for leg in rows:
            where = f"action {aid} leg {leg['leg_id']}"
            if leg["due_date"] < eff:
                raise ValueError(f"{where}: due_date precedes effective_date")
            if leg["leg_type"] == "cash":
                if leg["asset"] is not None or leg["units_per_source_share"] is not None or leg["fraction_policy"] is not None:
                    raise ValueError(f"{where}: cash legs take cash_per_source_share and optional accrual only")
                _number(leg["cash_per_source_share"] if leg["cash_per_source_share"] is not None else -1.,
                        f"{where} cash_per_source_share")
                accrual = [leg[k] for k in ("accrual_per_day", "accrual_start", "accrual_end")]
                if any(v is None for v in accrual) and any(v is not None for v in accrual):
                    raise ValueError(f"{where}: date-dependent consideration needs accrual_per_day, accrual_start and accrual_end")
                if accrual[0] is not None:
                    _number(accrual[0], f"{where} accrual_per_day")
                    if accrual[1] > accrual[2] or accrual[2] > eff:
                        raise ValueError(f"{where}: accrual window must end on or before effective_date")
            else:
                child = leg["asset"]
                if child is None or child not in info or child == src:
                    raise ValueError(f"{where}: security leg needs a known asset other than the source")
                if leg["cash_per_source_share"] is not None or leg["accrual_per_day"] is not None:
                    raise ValueError(f"{where}: security legs cannot carry cash; add a separate cash leg")
                _number(leg["units_per_source_share"] if leg["units_per_source_share"] is not None else 0.,
                        f"{where} units_per_source_share", positive=True)
                if leg["fraction_policy"] not in {"fractional", "cash_in_lieu"}:
                    raise ValueError(f"{where}: fraction_policy must be 'fractional' or 'cash_in_lieu'")
                if (leg["fraction_policy"] == "cash_in_lieu") != (leg["cash_in_lieu_price"] is not None):
                    raise ValueError(f"{where}: cash_in_lieu requires (only) an explicit cash_in_lieu_price")
                if leg["cash_in_lieu_price"] is not None:
                    _number(leg["cash_in_lieu_price"], f"{where} cash_in_lieu_price", positive=True)
                life = info[child]
                if life["terminated_date"] is not None and life["terminated_date"] <= leg["due_date"]:
                    raise ValueError(f"{where}: delivered security terminates before delivery")
                if any(a == child and eff <= d <= leg["due_date"] for a, d in split_keys):
                    raise ValueError(f"{where}: a split of {child} between effectiveness and delivery is unsupported; "
                                     "pending claims are not re-denominated")
                if leg["due_date"] > eff and any(r["asset"] == child and eff <= r["ex_session"] <= leg["due_date"]
                                                 for r in dividends.iter_rows(named=True)):
                    raise ValueError(f"{where}: a dividend of {child} going ex before delivery is unsupported; "
                                     "pending claims do not receive dividends")
    for (src, eff), aids in effective.items():
        if len(aids) > 1:
            raise ValueError(f"conflicting actions {sorted(aids)} for {src} effective {eff}")
    for a, r in info.items():
        if r["security_type"] == "equity" and r["terminated_date"] is not None and a not in terminal:
            raise ValueError(f"security {a} terminates without a supported terminal action")
    for src, aid in terminal.items():
        eff = info[src]["terminated_date"]
        if any(e > eff for s, e in effective if s == src):
            raise ValueError(f"security {src} has actions after its acquisition {aid}")
    # Supplied valuation marks are explicit inputs, only where no market quote exists.
    for r in marks.iter_rows(named=True):
        a, d = r["asset"], r["session"]
        if a not in info or info[a]["unquoted_valuation"] != "supplied_mark":
            raise ValueError(f"valuation mark for {a} requires unquoted_valuation='supplied_mark'")
        if d not in closes or status[d, a] in {"quoted", "terminated"}:
            raise ValueError(f"valuation mark for {a} on {d} must be a supplied session without a market quote, before termination")
        if r["mark"] < 0 or (r["mark"] == 0 and info[a]["security_type"] != "warrant"):
            raise ValueError(f"valuation mark for {a} on {d} must be positive (zero only for warrants)")
    counts = defaultdict(int)
    for s in status.values():
        counts[s] += 1
    diagnostics = [{"code": f"lifecycle_{s}", "table": "prices", "count": n}
                   for s, n in sorted(counts.items()) if s != "quoted"]
    return tables, diagnostics


@dataclass(frozen=True, kw_only=True)
class CorporateActionPolicy:
    """Explicit run-level treatment of corporate-action consequences.

    distributed_securities: 'retain' holds delivered securities outside the target
    universe; 'liquidate_at_first_permitted_close' sells (or covers) them at the first
    quoted non-terminal close after delivery with ordinary trade costs.
    short_obligations: a short holder's distributed/successor securities are either
    settled in cash at the first quoted close on or after delivery
    ('cash_at_delivery_close') or become a borrowed short position
    ('borrowed_short_position', requiring explicit StockBorrow coverage).
    Margin fractions apply to signed portfolios: pending positive claims, pending
    delivery/cash obligations and long warrant holdings. 1.0 gives no margin credit.
    reorganization_fee is a cash fee per mandatory action applied to a nonzero position.
    """

    distributed_securities: str
    short_obligations: str
    pending_claim_margin: float
    obligation_margin: float
    warrant_margin: float
    reorganization_fee: float

    def __post_init__(self):
        if self.distributed_securities not in {"retain", "liquidate_at_first_permitted_close"}:
            raise ValueError("distributed_securities must be 'retain' or 'liquidate_at_first_permitted_close'")
        if self.short_obligations not in {"cash_at_delivery_close", "borrowed_short_position"}:
            raise ValueError("short_obligations must be 'cash_at_delivery_close' or 'borrowed_short_position'")
        if _number(self.pending_claim_margin, "pending_claim_margin") > 1:
            raise ValueError("pending_claim_margin must be in [0, 1]")
        for name in ("obligation_margin", "warrant_margin"):
            if not 0 < _number(getattr(self, name), name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        _number(self.reorganization_fee, "reorganization_fee")


def _lifecycle(market):
    return market.securities is not None


def _market_assets(market):
    """Every security known to the market (priced or lifecycle-declared)."""
    return set(market.securities["asset"]) if _lifecycle(market) else set(market.prices["asset"])


def security_status(market) -> pl.DataFrame:
    """Per-session lifecycle status, alias and announced-action knowledge as of each close.

    ``status`` is one of not_listed, quoted, suspended, unquoted_after_last_session
    or terminated; ``tradable`` is True only for quoted sessions. ``announced_actions``
    lists actions whose terms were available by that session's close and are not yet
    effective, so dated universes never use later announcements.
    """
    from ._data import _validated_market
    market = _validated_market(market)
    dates = market.sessions["session"].to_list()
    closes = dict(market.sessions.select("session", "close_at").iter_rows())
    if not _lifecycle(market):
        assets = sorted(market.prices["asset"].unique())
        status = {(d, a): "quoted" for d in dates for a in assets}
        aliases, actions = [], []
    else:
        assets = sorted(market.securities["asset"])
        status = _quote_plan(market.securities, market.suspensions, dates)
        aliases = market.aliases.to_dicts()
        actions = market.corporate_actions.to_dicts()
    by_asset, by_source = defaultdict(list), defaultdict(list)
    for x in aliases:
        by_asset[x["asset"]].append(x)
    for x in actions:
        by_source[x["source_asset"]].append(x)
    rows = []
    for d in dates:
        for a in assets:
            alias = [x for x in by_asset[a] if x["valid_from"] <= d and (x["valid_to"] is None or d <= x["valid_to"])]
            known = sorted(x["action_id"] for x in by_source[a]
                           if x["announced_at"] <= closes[d] and d < x["effective_date"])
            rows.append({"session": d, "asset": a, "status": status[d, a],
                         "tradable": status[d, a] == "quoted",
                         "ticker": alias[0]["ticker"] if alias else None,
                         "exchange": alias[0]["exchange"] if alias else None,
                         "announced_actions": ",".join(known) or None})
    return pl.DataFrame(rows, schema={"session": D, "asset": S, "status": S, "tradable": pl.Boolean,
                                      "ticker": S, "exchange": S, "announced_actions": S})


def security_returns(market):
    """Quoted-price and economic total returns across distributions and acquisitions.

    Requires raw prices (provider-adjusted series already embed an unknown action
    treatment). ``price_return`` is the quoted raw close change. ``total_return``
    reinvests ex-date dividends and distributed securities at the ex-date close
    (analytical, like ``reinvest_ex_close``) and values acquisition consideration at
    face cash plus successor closes on the first session on/after effectiveness.
    Cash in lieu is ignored (fractional entitlement). Missing inputs yield a null
    return with an explicit status; no value is manufactured.
    """
    from ._data import _validated_market
    from ._results import ReturnResult
    market = _validated_market(market)
    if market.metadata["price_basis"] != "raw":
        raise ValueError("security_returns requires raw prices; adjusted series would double count actions")
    dates = market.sessions["session"].to_list()
    quotes = {(d, a): p for d, a, p in market.prices.iter_rows()}
    marks = ({(r["session"], r["asset"]): r["mark"] for r in market.valuation_marks.iter_rows(named=True)}
             if _lifecycle(market) else {})
    assets = sorted(_market_assets(market))
    status = (_quote_plan(market.securities, market.suspensions, dates) if _lifecycle(market)
              else {(d, a): "quoted" for d in dates for a in assets})
    ratios = {(r["effective_session"], r["asset"]): r["ratio"] for r in market.splits.iter_rows(named=True)}
    cash = defaultdict(float)
    for r in market.dividends.iter_rows(named=True):
        cash[r["ex_session"], r["asset"]] += r["cash_per_share"]
    legs = defaultdict(list)
    distributions, acquisitions = defaultdict(list), {}
    if _lifecycle(market):
        for r in market.action_legs.iter_rows(named=True):
            legs[r["action_id"]].append(r)
        for r in market.corporate_actions.iter_rows(named=True):
            (distributions[r["source_asset"]].append((r["effective_date"], r["action_id"]))
             if r["action_type"] == "distribution" else
             acquisitions.__setitem__(r["source_asset"], r))

    def value(d, a):
        if (d, a) in quotes:
            return quotes[d, a], "market"
        if (d, a) in marks:
            return marks[d, a], "supplied_mark"
        return None, None

    rows = []
    for a in assets:
        previous, gap = None, False
        for d in dates:
            if status[d, a] != "quoted":
                if previous is not None and status[d, a] in {"suspended", "unquoted_after_last_session"}:
                    gap = True
                continue
            p = quotes[d, a]
            if previous is None or gap:
                rows.append((d, a, previous[0] if gap else None, None, None,
                             "quote_gap" if gap else "first_observation"))
                previous, gap = (d, p), False
                continue
            start, prior = previous
            factor = ratios.get((d, a), 1.)
            extra, note = cash[d, a], "ok"
            for eff, aid in distributions.get(a, []):
                if start < eff <= d:
                    for leg in legs[aid]:
                        v, how = value(d, leg["asset"])
                        if v is None:
                            note = "unvalued_distribution"
                            break
                        extra += leg["units_per_source_share"]*v/factor
                        note = "supplied_mark_valuation" if how == "supplied_mark" and note == "ok" else note
            total = None if note == "unvalued_distribution" else (p+extra)*factor/prior-1
            rows.append((d, a, start, p/prior-1, total, note))
            previous = (d, p)
        deal = acquisitions.get(a)
        if deal is not None and previous is not None:
            start, prior = previous
            settle = next((d for d in dates if d >= deal["effective_date"]), None)
            if settle is None:
                rows.append((deal["effective_date"], a, start, None, None, "consideration_after_coverage"))
            else:
                total, note = 0., "acquisition_consideration"
                for leg in legs[deal["action_id"]]:
                    if leg["leg_type"] == "cash":
                        total += _cash_amount(leg)
                    else:
                        v, how = value(settle, leg["asset"])
                        if v is None:
                            total, note = None, "unvalued_consideration"
                            break
                        total += leg["units_per_source_share"]*v
                rows.append((settle, a, start, None, None if total is None else total/prior-1, note))
    values = pl.DataFrame(rows, schema={"session": D, "asset": S, "period_start": D, "price_return": F,
                                        "total_return": F, "status": S}, orient="row").sort("asset", "session")
    meta = {"method": "simple", "basis": "price_and_economic_total_return", "frequency": "1d",
            "unit": "fraction", "snapshot_id": market.snapshot_id, "source": deepcopy(market.metadata),
            "distribution_policy": "reinvest_at_ex_date_close_analytical",
            "acquisition_policy": "face_cash_plus_successor_close_first_session_on_or_after_effective_date",
            "cash_in_lieu": "ignored_fractional_entitlement", "dividend_policy": "reinvest_ex_close",
            "not_executable": True}
    from ._data import DIAGNOSTIC_SCHEMA
    counts = values.group_by("status").len().sort("status")
    diagnostics = pl.DataFrame([{"code": s, "table": "security_returns", "count": n} for s, n in counts.iter_rows()],
                               schema=DIAGNOSTIC_SCHEMA)
    return ReturnResult(values, meta, diagnostics)


def _cash_amount(leg):
    """Final cash per source share, including an explicitly supplied date-dependent accrual."""
    amount = leg["cash_per_source_share"]
    if leg["accrual_per_day"] is not None:
        amount = math.fsum([amount, leg["accrual_per_day"]*(leg["accrual_end"]-leg["accrual_start"]).days])
    return amount
