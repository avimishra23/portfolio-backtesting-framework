# Portfolio Backtesting Framework

[![Open UI In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/portfolio-backtesting-framework/blob/main/notebooks/Portfolio_Backtesting_UI.ipynb)

Advanced quantitative finance project for building, testing, and presenting systematic investment strategies with a reusable Python engine.

## Project Overview

This project builds a professional portfolio backtesting framework for testing multi-asset investment strategies using Yahoo Finance market data. It includes a configurable data pipeline, feature engineering layer, reusable strategy interface, transaction-cost-aware execution simulator, performance analytics, optional `vectorbt` and `backtrader` integrations, and institutional-style Plotly tear sheets.

The default example compares a cross-asset momentum strategy against `SPY` using ETFs such as `SPY`, `QQQ`, `TLT`, and `GLD`, but the framework is designed to support any liquid Yahoo Finance ticker universe.

## Simplest User Interface

Use `notebooks/Portfolio_Backtesting_UI.ipynb` when you want a simple app-like Colab experience. Users run the setup cell once, then interact with a dashboard containing ticker inputs, strategy selection, dates, cash amount, transaction costs, and a `Run Backtest` button.

Use `notebooks/Portfolio_Backtesting_Framework.ipynb` when you want the full professional code walkthrough and engine implementation.

## Real-World Finance Use Case

A quant research analyst or portfolio manager wants to test whether a rules-based allocation model improves risk-adjusted returns versus a passive benchmark. The workflow needs to answer:

- Does the strategy beat the benchmark after transaction costs?
- What is the CAGR, Sharpe ratio, Sortino ratio, Calmar ratio, and max drawdown?
- How stable are returns across regimes?
- How often does the strategy rebalance, and how expensive is turnover?
- What assets drive exposure over time?
- Is the strategy suitable for deeper research, paper trading, or production migration?

## System Architecture

```text
Yahoo Finance
    |
    v
Data Loader
    - download adjusted OHLCV data
    - retry transient failures
    - normalize single and multi-asset output
    |
    v
Cleaning Layer
    - align symbols to one calendar
    - forward-fill existing histories
    - remove invalid values
    - calculate returns
    |
    v
Feature Layer
    - moving averages
    - momentum
    - realized volatility
    - RSI
    - z-scores
    |
    v
Strategy Layer
    - equal weight
    - dual momentum
    - volatility-targeted momentum
    - custom strategy interface
    |
    v
Backtest Engine
    - next-session execution
    - cash-aware portfolio returns
    - rebalance-only turnover
    - commission and slippage costs
    - leverage and long-only constraints
    |
    v
Analytics and Dashboard
    - equity curve
    - drawdown chart
    - rolling Sharpe
    - weights chart
    - monthly return heatmap
    - metrics table
    - HTML tear sheet
```

## Required APIs and Data Sources

- Yahoo Finance market data through `yfinance.download`
- Daily adjusted OHLCV prices
- ETF, equity, index proxy, crypto, or futures proxy tickers available through Yahoo Finance
- Optional benchmark ticker such as `SPY`, `QQQ`, `IWM`, `ACWI`, or `^GSPC`

Default universe:

```python
["SPY", "QQQ", "TLT", "GLD"]
```

Default benchmark:

```python
"SPY"
```

## Required Python Libraries

Core libraries:

- `pandas`
- `numpy`
- `yfinance`
- `plotly`
- `scipy`
- `nbformat`

Optional professional backtesting integrations:

- `vectorbt`
- `backtrader`

Notebook support:

- `ipywidgets`

## Folder/File Structure

```text
portfolio-backtesting-framework/
    README.md
    requirements.txt
    configs/
        default_config.json
    notebooks/
        Portfolio_Backtesting_Framework.ipynb
        Portfolio_Backtesting_UI.ipynb
    reports/
        generated tear sheets go here
    src/
        portfolio_backtester/
            __init__.py
            adapters.py
            config.py
            dashboard.py
            data.py
            engine.py
            features.py
            metrics.py
            strategies.py
            visualization.py
```

The project is Colab-friendly, but it is structured like a normal Python repository so it can graduate from notebook research into production code.

## Step-by-Step Building Guide

1. Configure the research universe in `BacktestConfig`.
2. Download adjusted daily prices from Yahoo Finance.
3. Clean data and align all symbols to one calendar.
4. Build features such as momentum, moving averages, volatility, and RSI.
5. Select a strategy class or implement a custom strategy using `BaseStrategy`.
6. Generate target weights from the strategy.
7. Shift target weights by one session to avoid look-ahead bias.
8. Run the execution simulator with commission and slippage assumptions.
9. Compute CAGR, Sharpe, Sortino, Calmar, drawdown, beta, alpha, and tracking error.
10. Generate an interactive Plotly tear sheet and export results.

## Data Collection Pipeline

The `YahooFinanceDataLoader`:

- accepts a config object with tickers, benchmark, date range, and minimum observations
- downloads all symbols in one request where possible
- retries transient failures
- extracts adjusted close prices and volume
- handles Yahoo Finance single-index and MultiIndex response formats
- validates that all configured portfolio tickers have usable prices
- returns a `MarketData` object containing prices, returns, volume, and raw data

## Data Cleaning and Feature Engineering

Cleaning logic:

- converts index to timezone-naive `DatetimeIndex`
- sorts observations chronologically
- removes infinite values
- drops symbols with no price history
- forward-fills existing histories without backfilling pre-inception data
- computes close-to-close returns

Feature layer:

- 50-day and 200-day moving averages
- 63-day and 126-day momentum
- 63-day realized volatility
- 14-day RSI
- rolling z-scores

## Core Models/Algorithms

Included strategies:

- `EqualWeightStrategy`: periodically rebalances each available asset to equal weight.
- `DualMomentumStrategy`: selects the strongest assets with positive absolute momentum.
- `VolatilityTargetMomentumStrategy`: combines trend filtering, momentum ranking, and inverse-volatility sizing.

Execution model:

- strategy signals are generated using information available at the close
- target weights are shifted by one trading session before execution
- turnover is charged only when the target allocation changes
- cash is allowed when strategy weights sum below 100 percent
- commission and slippage are applied in basis points
- portfolio constraints enforce long-only and maximum gross leverage rules

## Visualizations and Dashboard Components

The notebook and package generate:

- equity curve versus benchmark
- drawdown chart
- rolling Sharpe ratio
- realized weights through time
- monthly returns heatmap
- performance metrics table
- interactive Plotly tear sheet
- exportable HTML report

## Performance Metrics

The framework computes:

- Total return
- CAGR
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Maximum drawdown
- Best and worst period
- Hit rate
- Skew and kurtosis
- 95 percent VaR and CVaR
- Benchmark CAGR
- Alpha
- Beta
- Tracking error
- Information ratio
- Active return

## Final Deliverables

- GitHub-ready README
- Colab notebook with section headers and complete code
- Reusable Python package under `src/`
- Config file for repeatable experiments
- Requirements file
- Interactive report export workflow
- Optional `vectorbt` and `backtrader` adapters

## Potential Upgrades

- Add walk-forward optimization and train/test splits.
- Add parameter sweeps with parallel execution.
- Add risk parity, minimum variance, and Black-Litterman allocators.
- Add factor models using Fama-French or custom risk factors.
- Add survivorship-bias-aware equity universes.
- Add dividend, split, and corporate-action diagnostics.
- Add tax-aware rebalancing.
- Add order book assumptions and intraday bars.
- Add database storage with DuckDB or PostgreSQL.
- Add CI tests and benchmark snapshots.
- Add FastAPI endpoints for strategy runs.
- Add Streamlit or Dash front end for non-technical users.

## References

- [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [vectorbt portfolio API](https://vectorbt.dev/api/portfolio/base/)
- [Backtrader quickstart](https://www.backtrader.com/docu/quickstart/quickstart/)
