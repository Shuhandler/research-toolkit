"""Auditable daily research calculations and financed buy-and-hold accounting."""

from ._backtest import buy_and_hold
from ._costs import TradeCosts
from ._data import prepare_market_data
from ._financing import Financing
from ._portfolio import BuyHoldPolicy, equal_weights
from ._results import BacktestResult, MarketData, ReturnResult, PerformanceResult, CorrelationResult
from ._metrics import performance, correlation
from ._snapshots import save_snapshot, load_snapshot
from . import plots
from ._returns import cumulative_returns, returns

__all__ = [
    "prepare_market_data", "returns", "cumulative_returns", "equal_weights", "buy_and_hold",
    "performance", "correlation", "save_snapshot", "load_snapshot", "plots",
    "PerformanceResult", "CorrelationResult", "MarketData", "ReturnResult", "BacktestResult", "BuyHoldPolicy", "TradeCosts", "Financing",
]
