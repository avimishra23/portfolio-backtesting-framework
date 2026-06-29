"""Optional integrations with vectorbt and backtrader."""

from __future__ import annotations

import pandas as pd

from .config import BacktestConfig


def run_vectorbt_sma_strategy(prices: pd.DataFrame, config: BacktestConfig, ticker: str | None = None):
    """Run a simple vectorbt moving-average strategy for validation and comparison."""

    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise RuntimeError("Install vectorbt to use this adapter: `pip install vectorbt`.") from exc

    symbol = (ticker or config.tickers[0]).upper()
    if symbol not in prices.columns:
        raise ValueError(f"{symbol} is not available in prices.")

    close = prices[symbol].dropna()
    fast_ma = vbt.MA.run(close, window=50)
    slow_ma = vbt.MA.run(close, window=200)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    return vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=config.initial_cash,
        fees=config.commission_bps / 10_000.0,
        slippage=config.slippage_bps / 10_000.0,
        freq="1D",
    )


def run_backtrader_sma_strategy(prices: pd.DataFrame, config: BacktestConfig, ticker: str | None = None) -> float:
    """Run a minimal Backtrader SMA crossover and return final portfolio value."""

    try:
        import backtrader as bt
    except ImportError as exc:
        raise RuntimeError("Install backtrader to use this adapter: `pip install backtrader`.") from exc

    symbol = (ticker or config.tickers[0]).upper()
    if symbol not in prices.columns:
        raise ValueError(f"{symbol} is not available in prices.")

    class SmaCross(bt.Strategy):
        params = dict(fast=50, slow=200)

        def __init__(self):
            fast = bt.ind.SMA(period=self.p.fast)
            slow = bt.ind.SMA(period=self.p.slow)
            self.crossover = bt.ind.CrossOver(fast, slow)

        def next(self):
            if not self.position and self.crossover > 0:
                self.buy()
            elif self.position and self.crossover < 0:
                self.close()

    data = pd.DataFrame(
        {
            "open": prices[symbol],
            "high": prices[symbol],
            "low": prices[symbol],
            "close": prices[symbol],
            "volume": 0,
            "openinterest": 0,
        }
    ).dropna()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(config.initial_cash)
    cerebro.broker.setcommission(commission=config.commission_bps / 10_000.0)
    cerebro.addstrategy(SmaCross)
    cerebro.adddata(bt.feeds.PandasData(dataname=data))
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
    cerebro.run()
    return float(cerebro.broker.getvalue())
