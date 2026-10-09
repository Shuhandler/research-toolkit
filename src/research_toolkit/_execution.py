"""Explicit daily liquidity estimates and nonlinear per-order execution expenses."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime
import math
import json
from statistics import mean, stdev

import polars as pl

from ._data import _local_midnight_utc, _table, _identity
from ._portfolio import _number, _weights
from ._results import LiquidityResult, ExecutionResult

LIQUIDITY_SCHEMA = {"asset": pl.String, "daily_volatility": pl.Float64, "dollar_adv": pl.Float64, "n_obs": pl.Int64}
ORDER_SCHEMA = {"asset": pl.String, "signed_notional": pl.Float64, "reference_price": pl.Float64}
COST_SCHEMA = {**ORDER_SCHEMA, "signed_quantity": pl.Float64, "daily_volatility": pl.Float64,
    "dollar_adv": pl.Float64, "order_adv_ratio": pl.Float64, "impact_bps": pl.Float64,
    "commission": pl.Float64, "half_spread": pl.Float64, "impact": pl.Float64,
    "total_cost": pl.Float64, "total_cost_bps": pl.Float64}


def estimate_liquidity(returns, dollar_volume, *, decision_session, lookback, metadata) -> LiquidityResult:
    """Sample daily volatility and mean dollar volume strictly before a decision.

    Dollar volumes must be supplied explicitly with availability timestamps; this
    function does not infer share-volume adjustment or download provider inputs.
    """
    from ._allocation import _window
    series, meta = _window(returns, decision_session, lookback, 1.)
    # Liquidity uses daily units; the window helper's annualization argument does not apply.
    del meta["periods_per_year"]
    schema = {"session": pl.Date, "asset": pl.String, "dollar_volume": pl.Float64,
              "available_at": pl.Datetime("us", "UTC")}
    volumes = _table(dollar_volume, schema, "dollar_volume", ["session", "asset"], nonempty=True)
    if (volumes["dollar_volume"] < 0).any():
        raise ValueError("dollar volume must be nonnegative")
    source = meta["source"]
    if not isinstance(metadata, dict) or not isinstance(metadata.get("source"), str) or not metadata["source"].strip() or metadata.get("currency") != source["currency"]:
        raise ValueError("dollar-volume metadata requires source and matching currency")
    dates = returns.values.filter((pl.col("session") >= date.fromisoformat(meta["sample_start"])) &
        (pl.col("session") <= date.fromisoformat(meta["sample_end"])))['session'].unique().sort().to_list()
    selected = volumes.filter(pl.col("session").is_in(dates))
    expected = {(d, a) for d in dates for a in series}
    if set(selected.select("session", "asset").iter_rows()) != expected:
        raise ValueError("dollar volumes must match the complete return window and asset universe")
    cutoff = _local_midnight_utc(decision_session, source["timezone"])
    if any(t > cutoff for t in selected["available_at"]):
        raise ValueError("liquidity input was unavailable before the decision")
    rows = [(a, stdev(series[a]), mean(selected.filter(pl.col("asset") == a)["dollar_volume"]), lookback) for a in sorted(series)]
    table = pl.DataFrame(rows, schema=LIQUIDITY_SCHEMA, orient="row")
    if (table["dollar_adv"] <= 0).any():
        raise ValueError("average daily dollar volume must be positive")
    meta.update(source={"returns": deepcopy(source), "dollar_volume": deepcopy(metadata)}, currency=source["currency"],
        volatility_unit="daily_decimal", adv_unit="currency_per_trading_day", decision_at=cutoff.isoformat(),
        return_availability="prior_session_close_assumed", daily_volatility_ddof=1)
    return LiquidityResult(table, meta)


def _liquidity(value):
    if not isinstance(value, LiquidityResult):
        raise ValueError("expected LiquidityResult with explicit daily volatility and dollar ADV")
    table = _table(value.estimates, LIQUIDITY_SCHEMA, "liquidity", ["asset"], nonempty=True).sort("asset")
    if (table["daily_volatility"] < 0).any() or (table["dollar_adv"] <= 0).any() or (table["n_obs"] < 2).any():
        raise ValueError("liquidity requires nonnegative daily volatility, positive dollar ADV and n_obs >= 2")
    if not isinstance(value.metadata, dict):
        raise ValueError("liquidity metadata must be a finite JSON dictionary")
    meta = deepcopy(value.metadata)
    try:
        json.dumps(meta, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("liquidity metadata must be finite JSON") from exc
    if meta.get("volatility_unit") != "daily_decimal" or meta.get("adv_unit") != "currency_per_trading_day":
        raise ValueError("explicit daily volatility and dollar ADV units are required")
    if not meta.get("source") or not isinstance(meta.get("currency"), str) or not meta["currency"].strip():
        raise ValueError("liquidity source and currency are required")
    try:
        sample_end, decision = date.fromisoformat(meta["sample_end"]), date.fromisoformat(meta["decision_session"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("liquidity requires ISO sample_end and decision_session") from exc
    if sample_end >= decision:
        raise ValueError("liquidity sample must end strictly before decision_session")
    # Validate serializability and retain a deterministic identity for mutation checks.
    identity = _identity({"liquidity": table}, meta)
    return table, meta, identity


def _parameter(value, assets, name):
    if isinstance(value, Mapping):
        if set(value) != set(assets):
            raise ValueError(f"{name} mapping must cover exactly the allocation assets")
        return {a: _number(value[a], f"{name}[{a}]") for a in assets}
    return {a: _number(value, name) for a in assets}


@dataclass(frozen=True, kw_only=True)
class SquareRootImpactCosts:
    """Frozen liquidity snapshot; costs depend on actual absolute traded notional.

    impact fraction = coefficient * daily_volatility * sqrt(order / dollar_ADV).
    Order/ADV is not intraday participation. Costs are cash expenses at the supplied
    raw reference price, never a second adverse price adjustment. No fixed fees.
    """
    liquidity: LiquidityResult
    commission_bps: float | Mapping[str, float]
    commission_per_share: float | Mapping[str, float]
    half_spread_bps: float | Mapping[str, float]
    impact_coefficient: float | Mapping[str, float]
    snapshot_id: str = field(init=False)
    model_id: str = field(init=False)

    def __post_init__(self):
        table, meta, identity = _liquidity(self.liquidity)
        object.__setattr__(self, "liquidity", LiquidityResult(table, meta))
        object.__setattr__(self, "snapshot_id", identity)
        config = {}
        for name in ("commission_bps", "commission_per_share", "half_spread_bps", "impact_coefficient"):
            config[name] = _parameter(getattr(self, name), table["asset"].to_list(), name)
            object.__setattr__(self, name, deepcopy(getattr(self, name)))
        object.__setattr__(self, "model_id", _identity({"liquidity": table}, {"liquidity": meta, "parameters": config}))

    def _bind(self, assets, prices, execution_session, decision_session=None, currency=None):
        table, meta, identity = _liquidity(self.liquidity)
        if identity != self.snapshot_id:
            raise ValueError("liquidity changed after cost-model construction")
        if type(execution_session) is not date or (decision_session is not None and type(decision_session) is not date):
            raise ValueError("execution/decision sessions must be dates")
        cutoff = decision_session or execution_session
        if cutoff > execution_session or date.fromisoformat(meta["decision_session"]) > cutoff:
            raise ValueError("cost estimates must be available by the strategy decision/execution cutoff")
        if set(assets) != set(table["asset"]) or set(prices) != set(assets):
            raise ValueError("cost liquidity/prices must cover exactly the order assets")
        if currency is not None and meta["currency"] != currency:
            raise ValueError("cost liquidity currency must match the portfolio")
        config = {name: _parameter(getattr(self, name), assets, name) for name in
                  ("commission_bps", "commission_per_share", "half_spread_bps", "impact_coefficient")}
        if _identity({"liquidity": table}, {"liquidity": meta, "parameters": config}) != self.model_id:
            raise ValueError("cost-model parameters changed after construction")
        marks = {a: _number(prices[a], f"reference_price[{a}]", positive=True) for a in assets}
        return _ImpactBinding(table, meta, marks, config, execution_session, self.snapshot_id, self.model_id, decision_session)


class _ImpactBinding:
    def __init__(self, table, metadata, prices, config, session, snapshot_id, model_id, decision_session):
        self.liquidity = {r["asset"]: r for r in table.iter_rows(named=True)}
        self.prices, self.config = prices, config
        self.metadata = dict(model="square_root_impact", liquidity=metadata, estimates=table.to_dicts(),
            parameters=config, execution_session=session.isoformat(), snapshot_id=snapshot_id, model_id=model_id,
            decision_session=decision_session.isoformat() if decision_session else None,
            participation_definition="absolute_order_notional_over_daily_dollar_ADV_not_intraday_participation",
            impact_basis="modeled", fill_policy="reference_price_plus_cash_expenses")

    def row(self, asset, notional):
        if not math.isfinite(notional):
            raise ValueError("order notional must be finite")
        q, p, estimate, c = abs(notional), self.prices[asset], self.liquidity[asset], self.config
        ratio = q/estimate["dollar_adv"]
        impact_rate = c["impact_coefficient"][asset]*estimate["daily_volatility"]*math.sqrt(ratio)
        commission = q*c["commission_bps"][asset]/10_000 + q/p*c["commission_per_share"][asset]
        spread, impact = q*c["half_spread_bps"][asset]/10_000, q*impact_rate
        cost = math.fsum([commission, spread, impact])
        row = dict(asset=asset, signed_notional=notional, reference_price=p, signed_quantity=notional/p,
            daily_volatility=estimate["daily_volatility"], dollar_adv=estimate["dollar_adv"],
            order_adv_ratio=ratio, impact_bps=impact_rate*10_000, commission=commission,
            half_spread=spread, impact=impact, total_cost=cost, total_cost_bps=cost/q*10_000 if q else 0.)
        if any(not math.isfinite(v) for k, v in row.items() if k != "asset"):
            raise ValueError("execution cost is not representable")
        return row

    def audit_row(self, asset, notional):
        return dict(self.row(asset, notional), model_id=self.metadata["model_id"],
            snapshot_id=self.metadata["snapshot_id"], decision_session=(
                date.fromisoformat(self.metadata["decision_session"]) if self.metadata["decision_session"] else None))

    def components(self, asset, notional):
        row = self.row(asset, notional)
        return {c: row[c] for c in ("commission", "half_spread", "impact")}

    def marginal(self, asset, notional):
        r = self.row(asset, notional)
        c = self.config
        return c["commission_bps"][asset]/10_000 + c["commission_per_share"][asset]/self.prices[asset] + c["half_spread_bps"][asset]/10_000 + 1.5*r["impact_bps"]/10_000


def estimate_trade_costs(orders, *, costs, execution_session, decision_session=None) -> ExecutionResult:
    table = _table(orders, ORDER_SCHEMA, "orders", ["asset"], nonempty=True).sort("asset")
    if not isinstance(costs, SquareRootImpactCosts):
        raise ValueError("estimate_trade_costs requires SquareRootImpactCosts")
    binding = costs._bind(table["asset"].to_list(), dict(table.select("asset", "reference_price").iter_rows()),
                          execution_session, decision_session)
    rows = [binding.row(a, n) for a, n in table.select("asset", "signed_notional").iter_rows()]
    return ExecutionResult(pl.DataFrame(rows, schema=COST_SCHEMA), binding.metadata)


def _entry_size(capital, leverage, weights, binding):
    if leverage == 0:
        return 0.
    lo, hi = 0., capital*leverage
    if not math.isfinite(hi):
        raise ValueError("entry notional is not representable")
    for _ in range(160):
        mid = (lo+hi)/2
        fees = math.fsum(binding.row(a, mid*w)["total_cost"] for a, w in weights.items())
        if mid+leverage*fees > leverage*capital:
            hi = mid
        else:
            lo = mid
        if math.isclose(lo, hi, rel_tol=1e-15, abs_tol=0):
            break
    return (lo+hi)/2


def size_entry_orders(*, weights, initial_capital, gross_leverage, prices, costs,
                      execution_session, decision_session=None) -> ExecutionResult:
    marks = _table(prices, {"asset": pl.String, "reference_price": pl.Float64}, "prices", ["asset"], nonempty=True)
    w = _weights(weights, marks["asset"].to_list())
    w = {a: v/math.fsum(w.values()) for a, v in w.items()}
    c = _number(initial_capital, "initial_capital", positive=True)
    leverage = _number(gross_leverage, "gross_leverage")
    if not isinstance(costs, SquareRootImpactCosts):
        raise ValueError("size_entry_orders requires SquareRootImpactCosts")
    binding = costs._bind(list(w), dict(marks.iter_rows()), execution_session, decision_session)
    gross = _entry_size(c, leverage, w, binding)
    rows = [binding.row(a, gross*v) for a, v in w.items()]
    total = math.fsum(r["total_cost"] for r in rows)
    equity = c-total
    residual = gross-leverage*equity
    if equity <= 0 or abs(residual) > min(.009, 1e-8+1e-12*c):
        raise ArithmeticError("entry sizing does not reconcile to post-cost equity")
    net_cash = c-gross-total
    return ExecutionResult(pl.DataFrame(rows, schema=COST_SCHEMA), dict(binding.metadata,
        initial_capital=c, gross_leverage=leverage, gross_notional=gross, total_cost=total,
        post_cost_equity=equity, cash=max(net_cash, 0.), debt=max(-net_cash, 0.), sizing_residual=residual))


def _impact_basket(values, equity, receivable, weights, leverage, binding, receivable_policy):
    total = math.fsum(weights.values())
    w = {a: v/total for a, v in weights.items()}
    def evaluate(after):
        gross = leverage*after
        if leverage <= 1 and receivable_policy == "reserve":
            gross = max(0., min(gross, after-receivable))
        changes = {a: 0. if math.isclose(w[a]*gross, values[a], rel_tol=1e-13, abs_tol=0) else w[a]*gross-values[a] for a in w}
        return changes, math.fsum(binding.row(a, n)["total_cost"] for a, n in changes.items())
    unchanged, no_fees = evaluate(equity)
    if no_fees == 0:
        return unchanged, 0.
    bounds = {a: max(values[a], abs(w[a]*leverage*equity-values[a])) for a in w}
    marginal = {a: binding.marginal(a, bounds[a]) for a in w}
    if any(v >= 1 for v in marginal.values()) or leverage*math.fsum(w[a]*marginal[a] for a in w) >= 1:
        raise ValueError("nonlinear scheduled sizing cannot guarantee monotonic funding at these costs/leverage")
    lo, hi = 0., equity
    if evaluate(lo)[1] >= equity:
        raise ValueError("insufficient equity for scheduled impact costs")
    for _ in range(100):
        mid = (lo+hi)/2
        if mid in (lo, hi):
            break
        _, fees = evaluate(mid)
        if mid+fees > equity:
            hi = mid
        else:
            lo = mid
    return evaluate((lo+hi)/2)


def _bind_cost_schedule(costs, executions, assets, prices, closes, currency, session_assets=None):
    """Validate every requested basket and freeze one binding per execution.

    A dated mapping requires exact decision keys and explicit source-window and
    availability declarations. A static model retains its existing contract.
    """
    dated = isinstance(costs, Mapping)
    if dated:
        if any(type(d) is not date for d in costs):
            raise ValueError("cost-model keys must be decision dates")
        if set(costs) != set(executions.values()) or None in executions.values():
            raise ValueError("dated costs require exact coverage of all scheduled decision dates")
        if any(not isinstance(m, SquareRootImpactCosts) for m in costs.values()):
            raise ValueError("dated costs must map decision dates to SquareRootImpactCosts")
        models = dict(costs)
    else:
        models = None
    bindings = {}
    for session, decision in executions.items():
        model = models[decision] if dated else costs
        bound = assets if session_assets is None else session_assets[session]
        binding = model._bind(bound, {a: prices[session, a] for a in bound}, session, decision, currency)
        if dated:
            meta = binding.metadata["liquidity"]
            if meta["decision_session"] != decision.isoformat():
                raise ValueError("dated model decision_session must match its exact mapping key")
            try:
                start, end = date.fromisoformat(meta["sample_start"]), date.fromisoformat(meta["sample_end"])
                available = datetime.fromisoformat(meta["decision_at"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("dated liquidity requires ISO sample_start, sample_end and timezone-aware decision_at") from exc
            sample_closes = closes
            if "source_session_closes" in meta:
                try:
                    calendar = meta["source_session_closes"]
                    if not isinstance(calendar, dict) or not calendar:
                        raise ValueError("empty source calendar")
                    sample_closes = {date.fromisoformat(d): datetime.fromisoformat(t)
                                     for d, t in calendar.items()}
                    ordered = sorted(sample_closes.items())
                    if (any(t.tzinfo is None or t.utcoffset() is None for _, t in ordered)
                            or any(a[1] >= b[1] for a, b in zip(ordered, ordered[1:]))
                            or any(t != closes[d] for d, t in ordered if d in closes)):
                        raise ValueError("inconsistent source closes")
                    if set(calendar) != set(meta["sessions"]):
                        raise ValueError("source calendar does not match return sessions")
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("dated liquidity requires a valid source session calendar consistent with execution closes") from exc
            if start not in sample_closes or end not in sample_closes or not start <= end < decision:
                raise ValueError("dated liquidity sample must use supplied sessions strictly before its decision")
            if (available.tzinfo is None or available.utcoffset() is None
                    or not sample_closes[end] <= available <= closes[decision]):
                raise ValueError("dated liquidity availability must follow its sample and be no later than the decision close")
        bindings[session] = binding
    return bindings
