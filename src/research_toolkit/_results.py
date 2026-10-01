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
    """Buy-and-hold records, including the failure close for a stopped run.

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


@dataclass(frozen=True)
class CorrelationResult:
    """Common-sample asset return correlations with basis and null reasons."""

    values: pl.DataFrame
    metadata: dict[str, Any]
    diagnostics: pl.DataFrame
