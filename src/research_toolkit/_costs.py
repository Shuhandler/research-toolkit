"""Proportional cash-expense models applied only to actual trades."""

from collections.abc import Mapping
from dataclasses import dataclass

from ._portfolio import _number


@dataclass(frozen=True, kw_only=True)
class TradeCosts:
    """Modeled per-side bps of traded notional; scalar or exact asset mapping.

    All three components must be supplied, including zeros. Half-spread and impact
    are cash expenses; they are not also embedded in the reference fill price.
    """

    commission_bps: float | Mapping[str, float]
    half_spread_bps: float | Mapping[str, float]
    impact_bps: float | Mapping[str, float]

    def resolve(self, assets):
        rates = {}
        for component in ("commission", "half_spread", "impact"):
            value = getattr(self, f"{component}_bps")
            if isinstance(value, Mapping):
                if set(value) != set(assets):
                    raise ValueError(f"{component}_bps mapping must cover exactly the allocation assets")
                rates[component] = {a: _number(value[a], f"{component}_bps[{a}]") / 10000
                                    for a in assets}
            else:
                rate = _number(value, f"{component}_bps") / 10000
                rates[component] = {a: rate for a in assets}
        return rates
