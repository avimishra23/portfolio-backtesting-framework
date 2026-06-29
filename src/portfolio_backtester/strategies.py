"""Reusable strategy definitions that emit target portfolio weights."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .features import build_feature_panel, momentum, realized_volatility, simple_moving_average


def rebalance_mask(index: pd.Index, frequency: str) -> pd.Series:
    """Return True on the last available observation of each rebalance period."""

    if len(index) == 0:
        return pd.Series(dtype=bool)
    periods = pd.Series(pd.DatetimeIndex(index).to_period(frequency), index=index)
    mask = periods.ne(periods.shift(-1)).fillna(True)
    mask.iloc[0] = True
    return mask


def normalize_long_only(scores: pd.DataFrame, max_gross_leverage: float = 1.0) -> pd.DataFrame:
    """Convert non-negative scores into long-only weights with cash allowed."""

    clipped = scores.clip(lower=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    row_sums = clipped.sum(axis=1)
    weights = clipped.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    return weights * min(float(max_gross_leverage), 1.0)


class BaseStrategy(ABC):
    """Abstract strategy contract.

    Strategies should use information available at the close and emit target
    weights. The engine shifts weights by one session before execution to avoid
    look-ahead bias.
    """

    name = "base_strategy"

    @abstractmethod
    def generate_weights(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        config: BacktestConfig,
        features: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Return a daily DataFrame of target weights indexed like prices."""


@dataclass
class EqualWeightStrategy(BaseStrategy):
    """Periodic equal-weight portfolio with cash for unavailable assets."""

    name: str = "equal_weight"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        config: BacktestConfig,
        features: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        universe = list(config.tickers)
        available = prices[universe].notna().astype(float)
        weights = available.div(available.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        mask = rebalance_mask(prices.index, config.rebalance_frequency)
        return weights.where(mask, np.nan).ffill().fillna(0.0)


@dataclass
class DualMomentumStrategy(BaseStrategy):
    """Long the strongest assets with positive absolute momentum."""

    lookback: int = 126
    top_n: int = 2
    name: str = "dual_momentum"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        config: BacktestConfig,
        features: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        universe = list(config.tickers)
        scores = momentum(prices[universe], self.lookback)
        positive_scores = scores.where(scores > 0.0, 0.0)
        ranks = positive_scores.rank(axis=1, ascending=False, method="first")
        selected = positive_scores.where(ranks <= self.top_n, 0.0)
        equal_selected = selected.gt(0.0).astype(float)
        weights = normalize_long_only(equal_selected, config.max_gross_leverage)
        mask = rebalance_mask(prices.index, config.rebalance_frequency)
        return weights.where(mask, np.nan).ffill().fillna(0.0)


@dataclass
class VolatilityTargetMomentumStrategy(BaseStrategy):
    """Trend-following cross-asset momentum with inverse-volatility sizing."""

    momentum_lookback: int = 126
    volatility_lookback: int = 63
    trend_window: int = 200
    name: str = "vol_target_momentum"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        config: BacktestConfig,
        features: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        universe = list(config.tickers)
        local_features = features or build_feature_panel(prices, returns)
        mom = local_features.get("momentum_126", momentum(prices[universe], self.momentum_lookback))[universe]
        vol = local_features.get("volatility_63", realized_volatility(returns[universe], self.volatility_lookback))[universe]
        trend = prices[universe] > simple_moving_average(prices[universe], self.trend_window)

        risk_adjusted_score = mom.clip(lower=0.0) / vol.replace(0, np.nan)
        risk_adjusted_score = risk_adjusted_score.where(trend, 0.0)
        weights = normalize_long_only(risk_adjusted_score, config.max_gross_leverage)
        mask = rebalance_mask(prices.index, config.rebalance_frequency)
        return weights.where(mask, np.nan).ffill().fillna(0.0)
