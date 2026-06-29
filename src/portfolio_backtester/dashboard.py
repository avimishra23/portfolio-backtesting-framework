"""Export helpers for reports and tear sheets."""

from __future__ import annotations

from pathlib import Path

from .engine import BacktestResult
from .visualization import (
    build_tearsheet,
    plot_drawdown,
    plot_equity_curve,
    plot_metrics_table,
    plot_monthly_returns,
    plot_rolling_sharpe,
    plot_weights,
)


def export_html_report(result: BacktestResult, output_dir: str | Path = "reports") -> Path:
    """Write an interactive HTML tear sheet and return its path."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / f"{result.strategy_name}_tear_sheet.html"
    fig = build_tearsheet(result)
    fig.write_html(report_file, include_plotlyjs="cdn", full_html=True)
    return report_file


def build_dashboard_figures(result: BacktestResult):
    """Return all standard figures for notebook display."""

    return {
        "equity": plot_equity_curve(result),
        "drawdown": plot_drawdown(result),
        "rolling_sharpe": plot_rolling_sharpe(result),
        "weights": plot_weights(result),
        "monthly_returns": plot_monthly_returns(result),
        "metrics": plot_metrics_table(result),
        "tearsheet": build_tearsheet(result),
    }
