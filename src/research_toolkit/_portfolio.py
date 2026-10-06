"""Small explicit allocation and execution policies."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

import polars as pl


def _number(value, name, *, positive=False):
    """Finite real (bool excluded) that is nonnegative, or strictly positive."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric input")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _finite_above(value, name, *, lower):
    """Finite real (bool excluded) strictly greater than ``lower``; same accepted types as ``_number``."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= lower:
        raise ValueError(f"{name} must be finite and greater than {lower}")
    return float(value)


@dataclass(frozen=True, kw_only=True)
class BuyHoldPolicy:
    """Required explicit policy; exposure above 1 requires Financing at simulation.

    Static weights must have been chosen before entry; fills are idealized at the
    declared closing price. There are no implicit rebalances or terminal sales.
    """

    execution: str
    sizing: str
    fractional_shares: bool
    initial_gross_leverage: float
    terminal_action: str

    def __post_init__(self):
        if self.execution != "entry_close" or self.sizing != "post_cost_equity":
            raise ValueError("only entry_close execution and post_cost_equity sizing are supported")
        if self.fractional_shares is not True or self.terminal_action != "mark_only":
            raise ValueError("fractional_shares=True and terminal_action='mark_only' are required")
        _number(self.initial_gross_leverage, "initial_gross_leverage")


def equal_weights(assets) -> pl.DataFrame:
    """Return sorted risky-asset proportions; cash exposure is a separate policy."""
    if isinstance(assets, str):
        raise ValueError("assets must be a collection of unique asset names")
    try:
        assets = list(assets)
    except TypeError as exc:
        raise ValueError("assets must be a collection of asset names") from exc
    if not assets or any(not isinstance(a, str) or not a.strip() for a in assets):
        raise ValueError("assets must contain nonblank names")
    if len(set(assets)) != len(assets):
        raise ValueError("duplicate allocation assets")
    return pl.DataFrame({"asset": sorted(assets), "weight": [1.0 / len(assets)] * len(assets)})


def _weights(weights, assets):
    if isinstance(weights, pl.DataFrame):
        if weights.schema != {"asset": pl.String, "weight": pl.Float64}:
            raise ValueError("weights schema must be asset: String, weight: Float64")
        if weights["asset"].is_duplicated().any():
            raise ValueError("duplicate weight assets")
        weights = dict(weights.iter_rows())
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("weights must be a nonempty asset mapping or Polars weight table")
    if any(a not in assets for a in weights):
        raise ValueError("weights contain unknown assets")
    checked = {a: _number(w, f"weight[{a}]") for a, w in weights.items()}
    total = math.fsum(checked.values())
    if not math.isclose(total, 1.0, abs_tol=1e-12, rel_tol=0):
        raise ValueError(f"risky weights must sum to 1; got {total}")
    # Preserve the user's values, including representational roundoff; sizing
    # divides by their actual sum and records both supplied and resolved weights.
    return dict(sorted(checked.items()))
