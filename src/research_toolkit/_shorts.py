"""Explicit signed sizing, segregated collateral and stock-loan assumptions."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta, timezone
import math
import json
import hashlib
from zoneinfo import ZoneInfo

import polars as pl

from ._data import _table
from ._portfolio import _number


@dataclass(frozen=True, kw_only=True)
class LongShortPolicy:
    """Research margin, not a broker rule; restricted cash is marked each close.

    Collateral earns the separately supplied gross rebate; borrow fees are charged
    separately. Released collateral follows the financing cash sweep. No net rebate
    input is accepted. All rates are annual nominal decimals.
    """
    collateral_multiple: float
    long_margin: float
    short_margin: float
    rebate_rate: float
    rebate_day_count: str

    def __post_init__(self):
        if _number(self.collateral_multiple, "collateral_multiple") < 1:
            raise ValueError("collateral_multiple must be at least 1")
        for name in ("long_margin", "short_margin"):
            if not 0 < _number(getattr(self, name), name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        _number(self.rebate_rate, "rebate_rate")
        if self.rebate_day_count not in {"ACT/360", "ACT/365F"}:
            raise ValueError("rebate_day_count must be ACT/360 or ACT/365F")


@dataclass(frozen=True, kw_only=True)
class StockBorrow:
    """Explicit per-asset annual fees, fixed or assigned to every accrual date.

    Dated schema: date, asset, annual_rate, available_at (UTC microseconds).
    Supply weekends too; no fill or implied borrow availability. Metadata must
    identify source and basis ('modeled' or 'measured').
    """
    rates: Mapping[str, float] | pl.DataFrame
    day_count: str
    metadata: dict
    snapshot_id: str = field(init=False)

    def __post_init__(self):
        if self.day_count not in {"ACT/360", "ACT/365F"}:
            raise ValueError("borrow day_count must be ACT/360 or ACT/365F")
        if (not isinstance(self.metadata, dict) or not isinstance(self.metadata.get("source"), str)
                or not self.metadata["source"].strip() or self.metadata.get("basis") not in {"modeled", "measured"}):
            raise ValueError("borrow metadata requires source and basis modeled/measured")
        if isinstance(self.rates, Mapping):
            if any(not isinstance(a, str) or not a.strip() for a in self.rates):
                raise ValueError("borrow asset names must be nonblank strings")
            rates = {a: _number(v, f"borrow rate for {a}") for a, v in self.rates.items()}
        elif isinstance(self.rates, pl.DataFrame):
            rates = _table(self.rates, {"date": pl.Date, "asset": pl.String,
                "annual_rate": pl.Float64, "available_at": pl.Datetime("us", "UTC")},
                "borrow rates", ["date", "asset"], nonempty=False).sort("date", "asset").clone()
            for row in rates.iter_rows(named=True):
                _number(row["annual_rate"], "annual_rate")
                if not row["asset"].strip():
                    raise ValueError("borrow asset must be nonblank")
                if row["available_at"] > _cutoff(row["date"]):
                    raise ValueError("borrow rate must be available at New York midnight on its accrual date")
        else:
            raise ValueError("borrow rates require an asset mapping or dated Polars table")
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("borrow metadata must be finite JSON") from exc
        payload = {"rates": rates.to_dicts() if isinstance(rates, pl.DataFrame) else rates,
                   "metadata": self.metadata, "day_count": self.day_count}
        identity = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
            default=lambda value: value.isoformat()).encode()).hexdigest()
        object.__setattr__(self, "snapshot_id", identity)
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


def _cutoff(day):
    return datetime.combine(day, time(), ZoneInfo("America/New_York")).astimezone(timezone.utc)


def _borrow_plan(config, assets, start, end):
    if not isinstance(config, StockBorrow):
        raise ValueError("signed portfolios require explicit StockBorrow, including zero fees")
    validated = replace(config)
    if validated.snapshot_id != config.snapshot_id:
        raise ValueError("stock-borrow inputs changed after construction; construct a new StockBorrow")
    config = validated
    days = [start + timedelta(days=i) for i in range(1, (end-start).days+1)]
    if isinstance(config.rates, Mapping):
        if set(config.rates) != set(assets):
            raise ValueError("borrow rates must cover exactly all potentially short assets")
        plan = {(d, a): (config.rates[a], None) for d in days for a in assets}
        source = dict(config.rates)
    else:
        plan = {(r["date"], r["asset"]): (r["annual_rate"], r["available_at"])
                for r in config.rates.iter_rows(named=True)}
        if set(plan) != {(d, a) for d in days for a in assets}:
            raise ValueError("dated borrow rates must cover every requested calendar date and potentially short asset exactly")
        source = [{k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in r.items()}
                  for r in config.rates.to_dicts()]
    return plan, {"snapshot_id": config.snapshot_id, "rates": source, "day_count": config.day_count, "source": deepcopy(config.metadata),
        "rate_timing": "known_at_New_York_midnight", "valuation": "previous_supplied_close",
        "stock_loan_availability_verified": False, "rebate_is_net_of_borrow_fee": False}


def _signed(values, assets, column):
    if isinstance(values, pl.DataFrame):
        values = dict(_table(values, {"asset": pl.String, column: pl.Float64}, column,
                             ["asset"], nonempty=True).iter_rows())
    if not isinstance(values, Mapping) or not values or any(a not in assets for a in values):
        raise ValueError(f"{column} requires a nonempty asset mapping or Polars table of known assets")
    out = {}
    for a, v in values.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{column} must be finite signed numbers")
        out[a] = float(v)
    return dict(sorted(out.items()))


def _signed_targets(targets, market, entry, end):
    from datetime import date
    if type(entry) is not date or type(end) is not date or entry >= end:
        raise ValueError("entry/end must be dates with at least one holding interval")
    if not isinstance(targets, pl.DataFrame):
        raise ValueError("signed scheduled targets must be a Polars table")
    columns = [c for c in ("equity_exposure", "quantity", "target_notional") if c in targets.columns]
    if len(columns) != 1:
        raise ValueError("signed targets require exactly one of equity_exposure, quantity or target_notional")
    col = columns[0]
    table = _table(targets, {"decision_session": pl.Date, "session": pl.Date,
        "asset": pl.String, col: pl.Float64}, "signed targets", ["session", "asset"], nonempty=True).sort("session", "asset")
    sessions = set(market.sessions["session"])
    if entry not in sessions or end not in sessions:
        raise ValueError("entry/end must be supplied sessions")
    universe = set(table["asset"])
    if not universe <= set(market.prices["asset"]):
        raise ValueError("unknown target assets")
    plans = {}
    for day in table["session"].unique().sort():
        rows = table.filter(pl.col("session") == day)
        decision = rows["decision_session"][0]
        if (rows["decision_session"].n_unique() != 1 or set(rows["asset"]) != universe
                or day not in sessions or decision not in sessions or not decision < day
                or not entry <= day < end):
            raise ValueError("signed baskets require a complete universe, prior decision session, and execution sessions in [entry, end)")
        values = _signed(rows.select("asset", col), universe, col)
        plans[day] = (values, 1., decision)
    if min(plans) != entry:
        raise ValueError("first target must execute on entry_session")
    return table, plans, col


def _signed_size(values, equity, targets, kind, prices, components, marginal):
    """Solve equity after costs, or retain exact shares/dollars; never post a cost."""
    def evaluate(after):
        if kind == "quantity":
            desired = {a: targets[a]*prices[a] for a in targets}
        elif kind == "target_notional":
            desired = targets
        else:
            desired = {a: targets[a]*after for a in targets}
        changes = {a: 0. if math.isclose(desired[a], values[a], rel_tol=1e-13, abs_tol=0)
                   else desired[a]-values[a] for a in targets}
        return changes, math.fsum(math.fsum(components(a, n).values()) for a, n in changes.items())
    if kind in {"quantity", "target_notional"}:
        changes, fee = evaluate(equity)
        if fee >= equity:
            raise ValueError("insufficient equity for fixed targets and trade costs")
        return changes, fee
    bounds = {a: max(abs(values[a]), abs(targets[a]*equity-values[a])) for a in targets}
    if math.fsum(abs(targets[a])*marginal(a, bounds[a]) for a in targets) >= 1:
        raise ValueError("signed sizing cannot guarantee monotonic funding at these costs/exposures")
    if evaluate(0.)[1] >= equity:
        raise ValueError("insufficient equity for signed basket costs")
    changes, fee = evaluate(equity)
    if fee == 0:
        return changes, fee
    lo, hi = 0., equity
    for _ in range(160):
        mid = (lo+hi)/2
        if mid in (lo, hi):
            break
        if mid+evaluate(mid)[1] > equity:
            hi = mid
        else:
            lo = mid
    return evaluate((lo+hi)/2)
