"""Plotly visualizations for portfolio tear sheets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .engine import BacktestResult
from .metrics import drawdown_series, equity_curve


CHART_TEMPLATE = "plotly_white"


def plot_equity_curve(result: BacktestResult) -> go.Figure:
    """Plot strategy equity versus benchmark equity."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.equity.index,
            y=result.equity,
            mode="lines",
            name=result.strategy_name,
            line=dict(width=2.5, color="#0F6B5B"),
        )
    )
    if result.benchmark_equity is not None:
        fig.add_trace(
            go.Scatter(
                x=result.benchmark_equity.index,
                y=result.benchmark_equity,
                mode="lines",
                name=result.config.benchmark,
                line=dict(width=1.8, color="#4C566A", dash="dot"),
            )
        )
    fig.update_layout(
        title="Equity Curve",
        template=CHART_TEMPLATE,
        yaxis_title="Portfolio Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_drawdown(result: BacktestResult) -> go.Figure:
    """Plot strategy and benchmark drawdowns."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.drawdown.index,
            y=result.drawdown,
            mode="lines",
            fill="tozeroy",
            name=f"{result.strategy_name} drawdown",
            line=dict(width=1.8, color="#B23A48"),
        )
    )
    if result.benchmark_equity is not None:
        benchmark_drawdown = drawdown_series(result.benchmark_equity)
        fig.add_trace(
            go.Scatter(
                x=benchmark_drawdown.index,
                y=benchmark_drawdown,
                mode="lines",
                name=f"{result.config.benchmark} drawdown",
                line=dict(width=1.4, color="#4C566A", dash="dot"),
            )
        )
    fig.update_layout(
        title="Drawdown",
        template=CHART_TEMPLATE,
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )
    return fig


def plot_rolling_sharpe(result: BacktestResult, window: int = 126) -> go.Figure:
    """Plot annualized rolling Sharpe ratio."""

    periods = 252
    returns = result.portfolio_returns
    rf_periodic = (1.0 + result.config.risk_free_rate) ** (1.0 / periods) - 1.0
    rolling = (returns - rf_periodic).rolling(window).mean() / returns.rolling(window).std()
    rolling = rolling * np.sqrt(periods)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling, mode="lines", name="Rolling Sharpe"))
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#4C566A")
    fig.update_layout(
        title=f"Rolling Sharpe Ratio ({window} sessions)",
        template=CHART_TEMPLATE,
        yaxis_title="Sharpe",
        hovermode="x unified",
    )
    return fig


def plot_weights(result: BacktestResult) -> go.Figure:
    """Plot realized portfolio weights through time."""

    fig = go.Figure()
    for column in result.weights.columns:
        fig.add_trace(
            go.Scatter(
                x=result.weights.index,
                y=result.weights[column],
                mode="lines",
                stackgroup="one",
                name=column,
            )
        )
    fig.update_layout(
        title="Realized Portfolio Weights",
        template=CHART_TEMPLATE,
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        hovermode="x unified",
    )
    return fig


def plot_monthly_returns(result: BacktestResult) -> go.Figure:
    """Plot monthly returns as a year by month heatmap."""

    monthly = (1.0 + result.portfolio_returns).resample("ME").prod() - 1.0
    heatmap = monthly.to_frame("return")
    heatmap["year"] = heatmap.index.year
    heatmap["month"] = heatmap.index.strftime("%b")
    pivot = heatmap.pivot(index="year", columns="month", values="return")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=month_order)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="Return"),
            hovertemplate="Year=%{y}<br>Month=%{x}<br>Return=%{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(title="Monthly Return Heatmap", template=CHART_TEMPLATE)
    return fig


def plot_metrics_table(result: BacktestResult) -> go.Figure:
    """Render a clean table of performance statistics."""

    formatted = result.metrics.copy()
    percent_rows = [
        "Total Return",
        "CAGR",
        "Annual Volatility",
        "Max Drawdown",
        "Best Period",
        "Worst Period",
        "Hit Rate",
        "VaR 95",
        "CVaR 95",
        "Benchmark CAGR",
        "Tracking Error",
        "Active Return",
    ]
    display_values = []
    for metric, value in formatted.items():
        if pd.isna(value):
            display_values.append("n/a")
        elif metric in percent_rows:
            display_values.append(f"{value:.2%}")
        else:
            display_values.append(f"{value:.2f}")

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Metric", "Value"], fill_color="#0F6B5B", font=dict(color="white"), align="left"),
                cells=dict(values=[formatted.index, display_values], fill_color="#F7F9FB", align="left"),
            )
        ]
    )
    fig.update_layout(title="Performance Metrics", template=CHART_TEMPLATE, height=560)
    return fig


def build_tearsheet(result: BacktestResult) -> go.Figure:
    """Create a compact multi-panel interactive Plotly tear sheet."""

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{}, {}],
        ],
        subplot_titles=("Equity Curve", "Drawdown", "Rolling 126D Sharpe", "Turnover"),
        vertical_spacing=0.09,
    )
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity, name=result.strategy_name), row=1, col=1)
    if result.benchmark_equity is not None:
        fig.add_trace(go.Scatter(x=result.benchmark_equity.index, y=result.benchmark_equity, name=result.config.benchmark), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.drawdown.index, y=result.drawdown, name="Drawdown", fill="tozeroy"), row=2, col=1)

    rolling_sharpe = (
        result.portfolio_returns.rolling(126).mean()
        / result.portfolio_returns.rolling(126).std()
        * np.sqrt(252)
    )
    fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name="Rolling Sharpe"), row=3, col=1)
    fig.add_trace(go.Bar(x=result.turnover.index, y=result.turnover, name="Turnover"), row=3, col=2)
    fig.update_layout(template=CHART_TEMPLATE, title="Portfolio Backtest Tear Sheet", height=900, hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    return fig
