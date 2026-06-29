import pandas as pd
import numpy as np

from portfolio_backtester import BacktestConfig, BacktestEngine, EqualWeightStrategy


def test_engine_runs_on_synthetic_prices():
    index = pd.bdate_range("2022-01-03", periods=90)
    steps = np.arange(len(index), dtype=float)
    prices = pd.DataFrame(
        {
            "AAA": 100 + steps,
            "BBB": 80 + pd.Series(steps).pow(0.8).to_numpy(),
            "SPY": 100 + steps * 0.5,
        },
        index=index,
    )
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    config = BacktestConfig(
        tickers=("AAA", "BBB"),
        benchmark="SPY",
        start="2022-01-03",
        min_observations=20,
        rebalance_frequency="M",
    )

    result = BacktestEngine(config).run(prices, returns, EqualWeightStrategy())

    assert result.equity.iloc[-1] > 0
    assert result.weights[["AAA", "BBB"]].sum(axis=1).max() <= 1.0 + 1e-9
    assert "Sharpe Ratio" in result.metrics.index
