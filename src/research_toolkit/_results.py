"""Concrete numerical containers. No calculation, plotting, or I/O side effects."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import polars as pl


@dataclass(frozen=True)
class MarketData:
    """Validated daily inputs; construct with :func:`prepare_market_data`.

    Tables are owned copies but Polars tables/dictionaries remain mutable. Consumers
    revalidate the snapshot identity to detect edits; prepare a new snapshot to edit.
    """

    prices: pl.DataFrame
    sessions: pl.DataFrame
    splits: pl.DataFrame
    dividends: pl.DataFrame
    metadata: dict[str, Any]
    diagnostics: pl.DataFrame
    snapshot_id: str


@dataclass(frozen=True)
class ReturnResult:
    """Keyed numerical returns plus their basis, frequency, and input identity."""

    values: pl.DataFrame
    metadata: dict[str, Any]
    diagnostics: pl.DataFrame


@dataclass(frozen=True)
class BacktestResult:
    """Daily portfolio records, including the failure close for a stopped run.

    All monetary fields use metadata currency. Call ``require_complete()`` before
    presenting results as covering the entire requested period.
    """

    daily: pl.DataFrame
    positions: pl.DataFrame
    trades: pl.DataFrame
    costs: pl.DataFrame
    events: pl.DataFrame
    valuations: pl.DataFrame
    attribution: pl.DataFrame
    receivables: pl.DataFrame
    diagnostics: pl.DataFrame
    targets: pl.DataFrame
    rebalances: pl.DataFrame
    turnover: pl.DataFrame
    dividend_reinvestments: pl.DataFrame
    financing_accruals: pl.DataFrame
    execution_costs: pl.DataFrame
    signal_audit: pl.DataFrame
    stock_borrow_accruals: pl.DataFrame
    short_financing_accruals: pl.DataFrame
    dividend_liabilities: pl.DataFrame
    metadata: dict[str, Any]
    status: str = "complete"
    stop_reason: str | None = None
    stop_session: date | None = None
    stop_time: datetime | None = None

    def require_complete(self) -> "BacktestResult":
        """Return this result or reject a stopped run as a full-period result."""
        if self.status != "complete":
            raise ValueError(f"backtest {self.status} on {self.stop_session}: {self.stop_reason}")
        return self


@dataclass(frozen=True)
class PerformanceResult:
    """Numerical metrics and prepared plot tables, retaining actual run coverage."""

    summary: pl.DataFrame
    benchmark_comparison: pl.DataFrame
    benchmark_series: pl.DataFrame
    daily: pl.DataFrame
    equity: pl.DataFrame
    drawdowns: pl.DataFrame
    allocation: pl.DataFrame
    attribution: pl.DataFrame
    metadata: dict[str, Any]
    cumulative: pl.DataFrame
    benchmark_summary: pl.DataFrame
    risk_free_returns: pl.DataFrame


@dataclass(frozen=True)
class CorrelationResult:
    """Common-sample asset return correlations with basis and null reasons."""

    values: pl.DataFrame
    metadata: dict[str, Any]
    diagnostics: pl.DataFrame


@dataclass(frozen=True)
class AllocationResult:
    """Requested risky-asset weights and the trailing estimates that produced them."""
    weights: pl.DataFrame
    estimates: pl.DataFrame
    metadata: dict[str, Any]
    diagnostics: pl.DataFrame


@dataclass(frozen=True)
class RiskResult:
    """Estimated covariance risk contributions, separate from realized attribution."""
    values: pl.DataFrame
    covariance: pl.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RollingRiskResult:
    """Prepared trailing net-return risk diagnostics, retaining run status/coverage."""
    values: pl.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RiskFreeResult:
    """Explicit holding-period risk-free returns and their daily rate inputs."""
    values: pl.DataFrame
    daily_rates: pl.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LiquidityResult:
    """Daily volatility and dollar ADV estimated strictly before a decision."""
    estimates: pl.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExecutionResult:
    """Per-order cost estimates, with sizing/accounting assumptions when applicable."""
    orders: pl.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ComparisonResult:
    """Long-form scenario metrics and explicit coverage/assumption differences."""
    values: pl.DataFrame
    diagnostics: pl.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ProviderDataResult:
    """Converted market inputs and provider bars with declared adjustment semantics."""
    market: MarketData
    bars: pl.DataFrame
    diagnostics: pl.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FeatureResult:
    """Lagged daily features, availability lineage and explicit unusable rows."""
    values: pl.DataFrame
    availability: pl.DataFrame
    diagnostics: pl.DataFrame
    sessions: pl.DataFrame
    metadata: dict[str, Any]
    snapshot_id: str


@dataclass(frozen=True)
class ResearchSplit:
    """Chronological samples and excluded rows; labels never fit feature transforms."""
    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame
    excluded: pl.DataFrame
    transforms: pl.DataFrame
    metadata: dict[str, Any]
    snapshot_id: str


@dataclass(frozen=True)
class ResearchSelection:
    """Validation losses and the chosen candidate's separately recorded parameters."""
    scores: pl.DataFrame
    selected_candidate: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ResearchEvaluation:
    """Final-test numerical loss with independent/exploratory use and copied audit."""
    scores: pl.DataFrame
    audit: pl.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SignalResult:
    """Dated long-only allocation instructions mapped to later supplied closes."""
    signals: pl.DataFrame
    targets: pl.DataFrame
    audit: pl.DataFrame
    sessions: pl.DataFrame
    metadata: dict[str, Any]
    snapshot_id: str
