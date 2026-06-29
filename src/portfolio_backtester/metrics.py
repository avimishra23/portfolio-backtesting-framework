"""Portfolio performance analytics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def infer_periods_per_year(index: pd.Index) -> int:
    """Infer an annualization factor from a DatetimeIndex."""

    if len(index) < 3:
        return 252
    dates = pd.DatetimeIndex(index)
    median_days = np.median(np.diff(dates.values).astype("timedelta64[D]").astype(float))
    if median_days <= 2:
        return 252
    if median_days <= 8:
        return 52
    if median_days <= 35:
        return 12
    return 1


def equity_curve(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Convert periodic returns into an equity curve."""

    clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return initial_value * (1.0 + clean).cumprod()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute drawdown as percentage decline from running peak."""

    running_peak = equity.cummax()
    return equity / running_peak - 1.0


def compute_performance_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Compute institutional portfolio statistics."""

    clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    clean = clean.astype(float)
    periods_per_year = infer_periods_per_year(clean.index)
    rf_periodic = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = clean - rf_periodic
    curve = equity_curve(clean)
    drawdowns = drawdown_series(curve)
    n_periods = max(len(clean), 1)

    ending_value = float(curve.iloc[-1])
    cagr = ending_value ** (periods_per_year / n_periods) - 1.0 if ending_value > 0 else np.nan
    annual_volatility = clean.std(ddof=0) * math.sqrt(periods_per_year)
    downside = clean.where(clean < rf_periodic, 0.0)
    downside_volatility = downside.std(ddof=0) * math.sqrt(periods_per_year)
    max_drawdown = float(drawdowns.min())

    sharpe = _safe_divide(excess.mean() * periods_per_year, excess.std(ddof=0) * math.sqrt(periods_per_year))
    sortino = _safe_divide(excess.mean() * periods_per_year, downside_volatility)
    calmar = _safe_divide(cagr, abs(max_drawdown))

    metrics = {
        "Total Return": ending_value - 1.0,
        "CAGR": cagr,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_drawdown,
        "Best Period": clean.max(),
        "Worst Period": clean.min(),
        "Hit Rate": (clean > 0).mean(),
        "Skew": clean.skew(),
        "Kurtosis": clean.kurtosis(),
        "VaR 95": clean.quantile(0.05),
        "CVaR 95": clean[clean <= clean.quantile(0.05)].mean(),
    }

    if benchmark_returns is not None:
        benchmark = benchmark_returns.reindex(clean.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        active = clean - benchmark
        covariance = np.cov(clean, benchmark, ddof=0)[0, 1] if benchmark.std(ddof=0) > 0 else np.nan
        beta = covariance / np.var(benchmark, ddof=0) if benchmark.var(ddof=0) > 0 else np.nan
        alpha = (clean.mean() - beta * benchmark.mean()) * periods_per_year if pd.notna(beta) else np.nan
        tracking_error = active.std(ddof=0) * math.sqrt(periods_per_year)
        metrics.update(
            {
                "Benchmark CAGR": equity_curve(benchmark).iloc[-1] ** (periods_per_year / n_periods) - 1.0,
                "Alpha": alpha,
                "Beta": beta,
                "Tracking Error": tracking_error,
                "Information Ratio": _safe_divide(active.mean() * periods_per_year, tracking_error),
                "Active Return": clean.sum() - benchmark.sum(),
            }
        )

    return pd.Series(metrics, dtype=float)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return np.nan
    return float(numerator / denominator)
