"""Portfolio execution simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .metrics import compute_performance_metrics, drawdown_series
from .strategies import BaseStrategy


@dataclass(frozen=True)
class BacktestResult:
    """Container returned by the portfolio engine."""

    strategy_name: str
    config: BacktestConfig
    weights: pd.DataFrame
    target_weights: pd.DataFrame
    portfolio_returns: pd.Series
    equity: pd.Series
    drawdown: pd.Series
    turnover: pd.Series
    costs: pd.Series
    metrics: pd.Series
    benchmark_returns: pd.Series | None = None
    benchmark_equity: pd.Series | None = None


class BacktestEngine:
    """Long-only, cash-aware, transaction-cost-aware portfolio backtester."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        strategy: BaseStrategy,
        features: dict[str, pd.DataFrame] | None = None,
    ) -> BacktestResult:
        """Generate strategy weights, simulate execution, and calculate metrics."""

        missing = sorted(set(self.config.tickers).difference(prices.columns))
        if missing:
            raise ValueError(f"Price data is missing configured tickers: {missing}")

        prices = prices.sort_index()
        returns = returns.reindex(prices.index).fillna(0.0)
        target_weights = strategy.generate_weights(prices, returns, self.config, features)
        target_weights = self._sanitize_weights(target_weights.reindex(prices.index).fillna(0.0))

        execution_weights = target_weights.shift(1).fillna(0.0)
        portfolio = self._simulate(returns[list(self.config.tickers)], execution_weights[list(self.config.tickers)])

        benchmark_returns = None
        benchmark_equity = None
        if self.config.benchmark in returns.columns:
            benchmark_returns = returns[self.config.benchmark].reindex(portfolio["returns"].index).fillna(0.0)
            benchmark_equity = self.config.initial_cash * (1.0 + benchmark_returns).cumprod()

        metrics = compute_performance_metrics(
            portfolio["returns"],
            benchmark_returns=benchmark_returns,
            risk_free_rate=self.config.risk_free_rate,
        )

        return BacktestResult(
            strategy_name=strategy.name,
            config=self.config,
            weights=portfolio["weights"],
            target_weights=target_weights,
            portfolio_returns=portfolio["returns"],
            equity=portfolio["equity"],
            drawdown=drawdown_series(portfolio["equity"]),
            turnover=portfolio["turnover"],
            costs=portfolio["costs"],
            metrics=metrics,
            benchmark_returns=benchmark_returns,
            benchmark_equity=benchmark_equity,
        )

    def _simulate(self, returns: pd.DataFrame, execution_weights: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
        """Simulate close-to-close returns with rebalance-only turnover."""

        index = returns.index
        columns = list(returns.columns)
        returns_array = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        target_array = execution_weights.reindex(index).fillna(0.0).to_numpy(dtype=float)
        target_array = np.nan_to_num(target_array, nan=0.0, posinf=0.0, neginf=0.0)

        n_rows, n_assets = returns_array.shape
        equity = np.empty(n_rows, dtype=float)
        portfolio_returns = np.zeros(n_rows, dtype=float)
        turnover = np.zeros(n_rows, dtype=float)
        costs = np.zeros(n_rows, dtype=float)
        realized_weights = np.zeros((n_rows, n_assets), dtype=float)

        equity[0] = self.config.initial_cash
        current_weights = np.zeros(n_assets, dtype=float)
        previous_target = np.zeros(n_assets, dtype=float)

        for row in range(1, n_rows):
            asset_returns = returns_array[row]
            gross_return = float(np.dot(current_weights, asset_returns))
            pre_cost_equity = equity[row - 1] * (1.0 + gross_return)

            if pre_cost_equity <= 0 or not np.isfinite(pre_cost_equity):
                raise RuntimeError(f"Portfolio equity became invalid on {index[row]}.")

            drifted_value = current_weights * equity[row - 1] * (1.0 + asset_returns)
            drifted_weights = drifted_value / pre_cost_equity
            desired_target = target_array[row]
            should_rebalance = not np.allclose(desired_target, previous_target, atol=1e-10, rtol=1e-8)

            if should_rebalance:
                turnover[row] = float(np.abs(desired_target - drifted_weights).sum())
                costs[row] = pre_cost_equity * turnover[row] * self.config.total_cost_rate
                equity[row] = pre_cost_equity - costs[row]
                current_weights = desired_target.copy()
                previous_target = desired_target.copy()
            else:
                equity[row] = pre_cost_equity
                current_weights = drifted_weights

            portfolio_returns[row] = equity[row] / equity[row - 1] - 1.0
            realized_weights[row] = current_weights

        return {
            "equity": pd.Series(equity, index=index, name="Equity"),
            "returns": pd.Series(portfolio_returns, index=index, name="Portfolio Return"),
            "turnover": pd.Series(turnover, index=index, name="Turnover"),
            "costs": pd.Series(costs, index=index, name="Trading Costs"),
            "weights": pd.DataFrame(realized_weights, index=index, columns=columns),
        }

    def _sanitize_weights(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Enforce portfolio constraints before orders reach the simulator."""

        clean = weights.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if not self.config.allow_short:
            clean = clean.clip(lower=0.0)

        gross = clean.abs().sum(axis=1)
        too_large = gross > self.config.max_gross_leverage
        if too_large.any():
            clean.loc[too_large] = clean.loc[too_large].div(gross.loc[too_large], axis=0) * self.config.max_gross_leverage
        return clean
