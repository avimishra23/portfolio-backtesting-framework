"""Market data loading and cleaning utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .config import BacktestConfig


@dataclass(frozen=True)
class MarketData:
    """Cleaned market data aligned to a single trading calendar."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    volumes: pd.DataFrame
    raw: pd.DataFrame


class YahooFinanceDataLoader:
    """Download and normalize daily OHLCV data from Yahoo Finance via yfinance."""

    def __init__(self, config: BacktestConfig, retries: int = 3, pause_seconds: float = 1.5) -> None:
        self.config = config
        self.retries = max(1, int(retries))
        self.pause_seconds = max(0.0, float(pause_seconds))

    def download(self) -> MarketData:
        """Download data and return adjusted close prices, returns, and volume."""

        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "yfinance is required for Yahoo Finance downloads. "
                "Install it with `pip install yfinance`."
            ) from exc

        symbols = self.config.all_symbols
        raw: pd.DataFrame | None = None
        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                raw = yf.download(
                    list(symbols),
                    start=self.config.start,
                    end=self.config.end,
                    auto_adjust=True,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                    repair=True,
                )
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:  # yfinance can raise transport and parsing errors.
                last_error = exc
            time.sleep(self.pause_seconds * attempt)

        if raw is None or raw.empty:
            raise RuntimeError(f"Yahoo Finance returned no data. Last error: {last_error}")

        prices = self._extract_field(raw, "Close", symbols)
        volumes = self._extract_field(raw, "Volume", symbols)
        prices = self._clean_prices(prices)
        volumes = self._clean_volumes(volumes, prices.index)

        missing = sorted(set(self.config.tickers).difference(prices.columns))
        if missing:
            raise RuntimeError(f"No usable adjusted close prices were found for: {missing}")

        if len(prices) < self.config.min_observations:
            raise RuntimeError(
                f"Only {len(prices)} rows were downloaded. "
                f"Need at least {self.config.min_observations} observations."
            )

        returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return MarketData(prices=prices, returns=returns, volumes=volumes, raw=raw)

    @staticmethod
    def _extract_field(raw: pd.DataFrame, field: str, symbols: Sequence[str]) -> pd.DataFrame:
        """Extract one OHLCV field from either single-index or MultiIndex yfinance output."""

        if isinstance(raw.columns, pd.MultiIndex):
            level_0 = raw.columns.get_level_values(0)
            level_1 = raw.columns.get_level_values(1)
            if field in level_0:
                frame = raw.xs(field, axis=1, level=0, drop_level=True)
            elif field in level_1:
                frame = raw.xs(field, axis=1, level=1, drop_level=True)
            else:
                raise RuntimeError(f"Field `{field}` was not present in downloaded data.")
        else:
            if field not in raw.columns:
                raise RuntimeError(f"Field `{field}` was not present in downloaded data.")
            if len(symbols) != 1:
                raise RuntimeError("Expected MultiIndex columns for multi-asset data.")
            frame = raw[[field]].rename(columns={field: symbols[0]})

        frame = frame.copy()
        frame.columns = [str(col).upper() for col in frame.columns]
        ordered = [symbol for symbol in symbols if symbol in frame.columns]
        return frame.loc[:, ordered]

    @staticmethod
    def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
        """Clean adjusted close prices without fabricating a trading history."""

        cleaned = prices.copy()
        cleaned.index = pd.to_datetime(cleaned.index).tz_localize(None)
        cleaned = cleaned.sort_index()
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.dropna(axis=1, how="all")
        cleaned = cleaned.ffill()
        cleaned = cleaned.dropna(axis=0, how="all")
        return cleaned

    @staticmethod
    def _clean_volumes(volumes: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
        """Clean volume data and align it to the price calendar."""

        cleaned = volumes.copy()
        cleaned.index = pd.to_datetime(cleaned.index).tz_localize(None)
        cleaned = cleaned.sort_index().reindex(index)
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        return cleaned.fillna(0.0)
