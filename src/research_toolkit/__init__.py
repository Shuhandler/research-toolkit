"""Auditable daily research calculations and unlevered buy-and-hold accounting."""

from ._backtest import buy_and_hold
from ._costs import TradeCosts
from ._data import prepare_market_data
from ._portfolio import BuyHoldPolicy, equal_weights
from ._results import BacktestResult, MarketData, ReturnResult
from ._returns import cumulative_returns, returns

__all__ = [
    "prepare_market_data", "returns", "cumulative_returns", "equal_weights", "buy_and_hold",
    "MarketData", "ReturnResult", "BacktestResult", "BuyHoldPolicy", "TradeCosts",
]
