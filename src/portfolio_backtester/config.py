"""Configuration objects for the portfolio backtesting framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


def _as_tuple(values: Iterable[str]) -> tuple[str, ...]:
    cleaned = tuple(str(value).strip().upper() for value in values if str(value).strip())
    if not cleaned:
        raise ValueError("At least one ticker is required.")
    return cleaned


@dataclass(frozen=True)
class BacktestConfig:
    """Runtime configuration shared by data, strategy, and execution layers."""

    tickers: tuple[str, ...] = field(default_factory=lambda: ("SPY", "QQQ", "TLT", "GLD"))
    benchmark: str = "SPY"
    start: str = "2015-01-01"
    end: str | None = None
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    rebalance_frequency: str = "M"
    risk_free_rate: float = 0.02
    max_gross_leverage: float = 1.0
    allow_short: bool = False
    min_observations: int = 252

    def __post_init__(self) -> None:
        object.__setattr__(self, "tickers", _as_tuple(self.tickers))
        object.__setattr__(self, "benchmark", str(self.benchmark).strip().upper())

        if not self.benchmark:
            raise ValueError("benchmark cannot be empty.")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Transaction cost assumptions cannot be negative.")
        if self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive.")
        if self.min_observations < 20:
            raise ValueError("min_observations should be at least 20 trading observations.")

    @property
    def all_symbols(self) -> tuple[str, ...]:
        """Return portfolio tickers plus benchmark, preserving order."""

        symbols = list(self.tickers)
        if self.benchmark not in symbols:
            symbols.append(self.benchmark)
        return tuple(symbols)

    @property
    def total_cost_rate(self) -> float:
        """Round-trip trading cost rate used by the execution simulator."""

        return (self.commission_bps + self.slippage_bps) / 10_000.0
