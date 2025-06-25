import numpy as np
import pandas as pd
from typing import Optional, List
from abc import ABC, abstractmethod

class DataProvider(ABC):
    @abstractmethod
    def download(self, symbol: str, **kwargs) -> pd.DataFrame:
        pass

class DataLoader:
    """
    Generic financial data loaders that uses a pluggable data provider (e.g., YFinanceProvider, FinnhubProvider)
    to retrieve time series data for an asset and its benchmark. It computes returns and volume changes
    across various timeframes, merges benchmark features, and cleans the final dataset.

    Attributes:
        provider (DataProvider): The data source to retrieve time series data from.
        asset (str): The primary asset symbol (e.g., "AAPL", "BTC-USD").
        benchmark_asset (str): The benchmark asset for comparison (e.g., "SPY").
        start_date (str, optional): Start date for data retrieval (format "YYYY-MM-DD").
        end_date (str, optional): End date for data retrieval (format "YYYY-MM-DD").
        freq (str): Data frequency (e.g., "1d", "1wk").
        timeframes (List[int]): List of time windows to calculate returns and volume changes.
    """

    def __init__(
        self,
        provider: DataProvider,
        asset: str,
        benchmark_asset: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        freq: str = "1d",
        timeframes: Optional[List[int]] = None,
    ):
        self.provider = provider
        self.asset = asset
        self.benchmark_asset = benchmark_asset or asset
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.timeframes = timeframes or [1, 2, 5, 10, 20, 40]

    def _download(self, symbol: str) -> pd.DataFrame:
        """
        Internal helper to fetch raw time series data from the provider.
        """
        kwargs = {"interval": self.freq}
        if self.start_date:
            kwargs["start"] = self.start_date
        if self.end_date:
            kwargs["end"] = self.end_date
        if not self.start_date and not self.end_date:
            kwargs["period"] = "max"
        return self.provider.download(symbol, **kwargs)

    def load(self) -> pd.DataFrame:
        """
        Loads and processes data for the asset and its benchmark. Computes:
            - Forward return
            - Rolling returns and volume changes
            - Benchmark returns and volume changes
            - Linear interpolation to handle missing data
            - Removes rows with NaNs or infs

        Returns:
            pd.DataFrame: Cleaned and feature-rich DataFrame indexed by date.
        """
        df = self._download(self.asset)
        df_benchmark = self._download(self.benchmark_asset)

        # Forward-looking return (next period)
        df["forward_return"] = df["Close"].pct_change().shift(-1)

        for i in self.timeframes:
            # Asset features
            df[f"return_{i}"] = df["Close"].pct_change(i)
            df[f"volume_{i}"] = df["Volume"].pct_change(i)

            # Benchmark features
            df_benchmark[f"benchmark_return_{i}"] = df_benchmark["Close"].pct_change(i)
            df_benchmark[f"benchmark_volume_{i}"] = df_benchmark["Volume"].pct_change(i)

        # Merge benchmark-derived columns into the main dataframe
        df = df.merge(
            df_benchmark[[col for col in df_benchmark.columns if col.startswith("benchmark_")]],
            how="left",
            left_index=True,
            right_index=True
        )

        # Handle missing or infinite values
        df.interpolate(method="linear", limit_direction="both", inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        return df
