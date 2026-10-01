"""Concrete numerical containers. No calculation, plotting, or I/O side effects."""

from dataclasses import dataclass
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
    """Unlevered buy-and-hold records; all monetary fields use metadata currency."""

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
