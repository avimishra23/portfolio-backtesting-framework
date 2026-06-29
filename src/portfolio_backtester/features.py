"""Feature engineering for cross-asset portfolio strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_moving_average(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling simple moving averages."""

    return prices.rolling(window=window, min_periods=max(5, window // 3)).mean()


def momentum(prices: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """Compute lookback total return momentum."""

    return prices.pct_change(periods=lookback, fill_method=None)


def realized_volatility(returns: pd.DataFrame, lookback: int = 63, periods_per_year: int = 252) -> pd.DataFrame:
    """Compute annualized rolling realized volatility."""

    return returns.rolling(window=lookback, min_periods=max(10, lookback // 3)).std() * np.sqrt(periods_per_year)


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute a vectorized Relative Strength Index."""

    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def zscore(values: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """Compute rolling z-scores for cross-asset ranking or normalization."""

    rolling_mean = values.rolling(window=lookback, min_periods=max(20, lookback // 3)).mean()
    rolling_std = values.rolling(window=lookback, min_periods=max(20, lookback // 3)).std()
    return (values - rolling_mean) / rolling_std.replace(0, np.nan)


def build_feature_panel(prices: pd.DataFrame, returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the default feature dictionary used by example strategies."""

    return {
        "sma_50": simple_moving_average(prices, 50),
        "sma_200": simple_moving_average(prices, 200),
        "momentum_63": momentum(prices, 63),
        "momentum_126": momentum(prices, 126),
        "volatility_63": realized_volatility(returns, 63),
        "rsi_14": rsi(prices, 14),
    }
