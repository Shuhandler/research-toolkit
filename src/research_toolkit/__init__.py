"""Auditable daily research, allocation and financed portfolio accounting."""

from ._backtest import buy_and_hold, scheduled_rebalance
from ._rebalancing import RebalancePolicy
from ._allocation import inverse_volatility_weights, risk_contributions, rolling_risk
from ._results import AllocationResult, RiskResult, RollingRiskResult
from ._costs import TradeCosts
from ._data import prepare_market_data
from ._financing import Financing
from ._dividends import DividendReinvestment
from ._portfolio import BuyHoldPolicy, equal_weights
from ._results import BacktestResult, MarketData, ReturnResult, PerformanceResult, CorrelationResult
from ._metrics import performance, correlation
from ._snapshots import save_snapshot, load_snapshot
from . import plots
from ._returns import cumulative_returns, returns

__all__ = [
    "DividendReinvestment",
    "inverse_volatility_weights", "risk_contributions", "rolling_risk",
    "AllocationResult", "RiskResult", "RollingRiskResult", "scheduled_rebalance", "RebalancePolicy", "prepare_market_data", "returns", "cumulative_returns", "equal_weights", "buy_and_hold",
    "performance", "correlation", "save_snapshot", "load_snapshot", "plots",
    "PerformanceResult", "CorrelationResult", "MarketData", "ReturnResult", "BacktestResult", "BuyHoldPolicy", "TradeCosts", "Financing",
]
