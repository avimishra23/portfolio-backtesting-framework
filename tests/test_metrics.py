import numpy as np
import pandas as pd

from portfolio_backtester.metrics import compute_performance_metrics, drawdown_series, equity_curve


def test_metrics_are_finite_for_variable_positive_returns():
    index = pd.bdate_range("2022-01-03", periods=252)
    returns = pd.Series(np.tile([0.0005, 0.0015], 126), index=index)

    metrics = compute_performance_metrics(returns, risk_free_rate=0.0)

    assert metrics["Total Return"] > 0
    assert metrics["CAGR"] > 0
    assert np.isfinite(metrics["Sharpe Ratio"])
    assert metrics["Max Drawdown"] == 0


def test_drawdown_series_starts_at_zero():
    index = pd.bdate_range("2022-01-03", periods=4)
    curve = pd.Series([100, 110, 99, 120], index=index)

    drawdown = drawdown_series(curve)

    assert drawdown.iloc[0] == 0
    assert drawdown.min() < 0


def test_equity_curve_compounds_returns():
    index = pd.bdate_range("2022-01-03", periods=2)
    returns = pd.Series([0.10, -0.10], index=index)

    curve = equity_curve(returns, initial_value=100)

    assert round(curve.iloc[-1], 2) == 99.0
