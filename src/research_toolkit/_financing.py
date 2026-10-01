"""Explicit cash-loan assumptions for daily research accounts."""

from dataclasses import dataclass
import math

from ._portfolio import _number


@dataclass(frozen=True, kw_only=True)
class Financing:
    """Fixed nominal annual rates with calendar-day capitalization.

    All fields are explicit. ``maintenance_equity_ratio`` is equity/gross risky
    exposure, not a broker margin rule. It may be None only without initial
    borrowing. The only supported sweep repays debt; the only breach action stops
    the simulation without a sale. Rates use decimal fractions, not percentages.
    """

    cash_rate: float
    borrowing_rate: float
    day_count: str
    maintenance_equity_ratio: float | None
    on_breach: str
    cash_sweep: str

    def __post_init__(self):
        _number(self.cash_rate, "cash_rate")
        _number(self.borrowing_rate, "borrowing_rate")
        if self.day_count != "ACT/365F":
            raise ValueError("only financing day_count='ACT/365F' is supported")
        if self.maintenance_equity_ratio is not None:
            if _number(self.maintenance_equity_ratio, "maintenance_equity_ratio", positive=True) > 1:
                raise ValueError("maintenance_equity_ratio must be in (0, 1]")
        if self.on_breach != "stop" or self.cash_sweep != "repay_debt":
            raise ValueError("financing requires on_breach='stop' and cash_sweep='repay_debt'")


def _below_margin(equity_ratio, threshold):
    """Treat only relative floating-point roundoff at the threshold as equality."""
    return threshold is not None and equity_ratio is not None and (
        equity_ratio < threshold and not math.isclose(equity_ratio, threshold, rel_tol=1e-12, abs_tol=0)
    )
