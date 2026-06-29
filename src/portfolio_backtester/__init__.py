"""Reusable portfolio backtesting framework."""

from .config import BacktestConfig
from .data import MarketData, YahooFinanceDataLoader
from .engine import BacktestEngine, BacktestResult
from .metrics import compute_performance_metrics
from .strategies import (
    BaseStrategy,
    DualMomentumStrategy,
    EqualWeightStrategy,
    VolatilityTargetMomentumStrategy,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "BaseStrategy",
    "DualMomentumStrategy",
    "EqualWeightStrategy",
    "MarketData",
    "VolatilityTargetMomentumStrategy",
    "YahooFinanceDataLoader",
    "compute_performance_metrics",
]
