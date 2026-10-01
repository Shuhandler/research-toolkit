"""Explicit scheduled targets and cost-aware, self-financing basket sizing."""
from dataclasses import dataclass
import math

import polars as pl

from ._data import _table
from ._portfolio import _number, _weights

TARGET_SCHEMA = {"decision_session": pl.Date, "session": pl.Date, "asset": pl.String,
                 "weight": pl.Float64, "gross_leverage": pl.Float64}


@dataclass(frozen=True, kw_only=True)
class RebalancePolicy:
    """Required close execution, target concentration and receivable funding rules.

    reserve: unlevered purchases leave unpaid receivables out of spendable funding.
    require_target: reject a basket requiring a loan at target leverage <= 1.
    Limits apply to requested risky-asset proportions at trading decisions, not to
    subsequently drifting holdings. No automatic weight clipping occurs.
    """
    execution: str
    sizing: str
    fractional_shares: bool
    terminal_action: str
    non_session: str
    receivable_policy: str
    max_asset_weight: float

    def __post_init__(self):
        if self.execution != "scheduled_close" or self.sizing != "post_cost_equity":
            raise ValueError("require scheduled_close execution and post_cost_equity sizing")
        if self.fractional_shares is not True or self.terminal_action != "mark_only":
            raise ValueError("require fractional shares and mark_only terminal action")
        if self.non_session != "raise":
            raise ValueError("non_session must be 'raise'; supply actual calendar sessions")
        if self.receivable_policy not in {"reserve", "require_target"}:
            raise ValueError("receivable_policy must be reserve or require_target")
        if _number(self.max_asset_weight, "max_asset_weight", positive=True) > 1:
            raise ValueError("max_asset_weight must be in (0, 1]")


def _limit(weights, maximum):
    if _number(maximum, "max_asset_weight", positive=True) > 1:
        raise ValueError("max_asset_weight must be in (0, 1]")
    if any(w > maximum and not math.isclose(w, maximum, rel_tol=1e-12, abs_tol=0)
           for w in weights.values()):
        raise ValueError("target exceeds max_asset_weight; no clipping/renormalization is applied")


def _targets(targets, market, entry, end, policy):
    table = _table(targets, TARGET_SCHEMA, "targets", ["session", "asset"], nonempty=True).sort("session", "asset")
    calendar = market.sessions["session"].to_list()
    assets = sorted(table["asset"].unique().to_list())
    if not set(assets) <= set(market.prices["asset"]):
        raise ValueError("targets contain unknown assets")
    if table["session"][0] != entry:
        raise ValueError("first target must execute on entry_session")
    plans = {}
    for day in table["session"].unique().sort():
        rows = table.filter(pl.col("session") == day)
        if rows["decision_session"].n_unique() != 1 or rows["gross_leverage"].n_unique() != 1:
            raise ValueError("each basket must have one decision_session and gross_leverage")
        decision = rows["decision_session"][0]
        if day not in calendar or decision not in calendar or not entry <= day < end:
            raise ValueError("target dates must be supplied sessions; execution is in [entry, end)")
        if decision >= day:
            raise ValueError("decision_session must precede execution session")
        if rows["asset"].to_list() != assets:
            raise ValueError("every target must cover the same universe; use explicit zero weights")
        weights = _weights(rows.select("asset", "weight"), assets)
        _limit(weights, policy.max_asset_weight)
        leverage = _number(rows["gross_leverage"][0], "gross_leverage")
        plans[day] = (weights, leverage, decision)
    return table, plans


def _basket(values, equity, receivable, weights, leverage, rates, receivable_policy):
    """Solve E_after + trade_cost(E_after) = E_before by monotone bisection.

    Proportional component rates act on absolute changed notional, not the whole
    portfolio. Each total rate < 1 and L*weighted_rate < 1 guarantee monotonicity.
    The currency solver residual is checked by the shared accounting engine.
    """
    total_weight = math.fsum(weights.values())
    w = {a: weights[a]/total_weight for a in weights}
    k = {a: math.fsum(rates[c][a] for c in rates) for a in w}
    if any(rate >= 1 for rate in k.values()) or leverage*math.fsum(w[a]*k[a] for a in w) >= 1:
        raise ValueError("scheduled sizing requires each cost rate < 100% and leverage*weighted_cost < 1")

    def evaluate(after):
        gross = leverage*after
        if leverage <= 1 and receivable_policy == "reserve":
            gross = max(0., min(gross, after-receivable))
        targets = {a: w[a]*gross for a in w}
        # Only representational roundoff is a no-trade; never suppress material
        # trades with an arbitrary minimum notional or turnover threshold.
        changes = {a: 0. if math.isclose(targets[a], values[a], rel_tol=1e-13, abs_tol=0)
                   else targets[a]-values[a] for a in w}
        cost = math.fsum(abs(changes[a])*k[a] for a in w)
        return changes, cost

    lower, upper = 0., equity
    if evaluate(lower)[1] >= equity:
        raise ValueError("insufficient equity to fund scheduled trade costs")
    for _ in range(100):
        midpoint = (lower+upper)/2
        if midpoint in (lower, upper):
            break
        _, cost = evaluate(midpoint)
        if midpoint+cost > equity:
            upper = midpoint
        else:
            lower = midpoint
    changes, cost = evaluate((lower+upper)/2)
    if not math.isfinite(cost):
        raise ValueError("scheduled cost is not representable")
    return changes, cost
