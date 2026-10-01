"""Explicit payment-funded dividend reinvestment assumptions."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class DividendReinvestment:
    """Opt-in same-asset fractional purchases, never funded by a new loan.

    Payment dates have no intraday timestamp: execution assumes cash is available
    before the first supplied close on or after payment. This models a standing
    instruction, not a broker's actual DRIP fill. All choices are explicit.
    """

    execution: str
    funding: str
    scheduled_collision: str
    terminal_action: str

    def __post_init__(self):
        if self.execution != "first_close_on_or_after_payment":
            raise ValueError("reinvestment execution must be 'first_close_on_or_after_payment'")
        if self.funding not in {"after_debt_repayment", "before_debt_repayment"}:
            raise ValueError("reinvestment funding must be 'after_debt_repayment' or 'before_debt_repayment'")
        if self.scheduled_collision != "rebalance_only":
            raise ValueError("reinvestment scheduled_collision must be 'rebalance_only'")
        if self.terminal_action != "hold_cash":
            raise ValueError("reinvestment terminal_action must be 'hold_cash'")
