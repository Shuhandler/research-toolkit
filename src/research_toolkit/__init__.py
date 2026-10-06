"""Auditable daily research, allocation and financed portfolio accounting."""

from ._version import __version__ as __version__
from ._shorts import LongShortPolicy, StockBorrow
from ._research import lagged_features, chronological_split, standardize
from ._research_eval import ResearchStudy
from ._signals import signal_targets
from ._results import FeatureResult, ResearchSplit, ResearchSelection, ResearchEvaluation, SignalResult
from ._backtest import buy_and_hold, scheduled_rebalance
from ._rebalancing import RebalancePolicy
from ._allocation import inverse_volatility_weights, risk_contributions, rolling_risk
from ._results import AllocationResult, RiskResult, RollingRiskResult
from ._costs import TradeCosts
from ._execution import SquareRootImpactCosts, estimate_liquidity, estimate_trade_costs, size_entry_orders
from ._data import prepare_market_data
from ._financing import Financing
from ._sofr import SOFRFinancing
from ._dividends import DividendReinvestment
from ._portfolio import BuyHoldPolicy, equal_weights
from ._results import BacktestResult, MarketData, ReturnResult, PerformanceResult, CorrelationResult
from ._metrics import performance, correlation, tail_risk
from ._series import series_performance
from ._calendars import trading_calendar
from ._risk_free import risk_free_returns
from ._comparison import compare_performance
from ._results import RiskFreeResult, ComparisonResult, LiquidityResult, ExecutionResult, ProviderDataResult, ProviderDownloadResult, TradingCalendar
from ._snapshots import save_snapshot, load_snapshot
from . import plots, adapters
from ._returns import cumulative_returns, returns

__all__ = [
    "LongShortPolicy", "StockBorrow",
    "lagged_features", "chronological_split", "standardize", "ResearchStudy", "signal_targets",
    "FeatureResult", "ResearchSplit", "ResearchSelection", "ResearchEvaluation", "SignalResult",
    "SquareRootImpactCosts", "estimate_liquidity", "estimate_trade_costs", "size_entry_orders",
    "risk_free_returns", "compare_performance", "RiskFreeResult", "ComparisonResult",
    "LiquidityResult", "ExecutionResult", "ProviderDataResult", "ProviderDownloadResult",
    "series_performance", "tail_risk", "trading_calendar", "TradingCalendar",
    "SOFRFinancing",
    "DividendReinvestment",
    "inverse_volatility_weights", "risk_contributions", "rolling_risk",
    "AllocationResult", "RiskResult", "RollingRiskResult", "scheduled_rebalance", "RebalancePolicy", "prepare_market_data", "returns", "cumulative_returns", "equal_weights", "buy_and_hold",
    "performance", "correlation", "save_snapshot", "load_snapshot", "plots", "adapters",
    "PerformanceResult", "CorrelationResult", "MarketData", "ReturnResult", "BacktestResult", "BuyHoldPolicy", "TradeCosts", "Financing",
]
