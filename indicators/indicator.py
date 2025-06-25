import numpy as np
import pandas as pd
from typing import List
from .base import BaseIndicator

class VolatilityIndicator(BaseIndicator):
    """
    Adds rolling volatility (standard deviation of log returns) for specified timeframes.
    """

    def __init__(self, timeframes: List[int] = [5, 10, 20, 40], in_place: bool = False):
        """
        Parameters:
        - timeframes: List of rolling window sizes to compute volatility.
        - in_place: Whether to modify the original DataFrame.
        """
        super().__init__(in_place)
        self.timeframes = timeframes

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds volatility columns for each specified timeframe.
        Assumes the DataFrame contains a 'return_1' column.
        """
        df_ = self._prepare_df(df)
        for i in self.timeframes:
            df_[f"volatility_{i}"] = np.log1p(df_["return_1"]).rolling(i).std()
        return df_


class MACDIndicator(BaseIndicator):
    """
    Computes the MACD (Moving Average Convergence Divergence) indicator.
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds MACD line, signal line, and histogram to the DataFrame.
        Assumes the DataFrame contains a 'Close' column.
        """
        df_ = self._prepare_df(df)
        close = df_["Close"]
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df_["macd_line"] = macd_line
        df_["macd_signal_line"] = signal_line
        df_["macd_histogram"] = macd_line - signal_line
        return df_


class RSIIndicator(BaseIndicator):
    """
    Computes the RSI (Relative Strength Index) using exponential weighted averages.
    """

    def __init__(self, window: int = 5, in_place: bool = False):
        """
        Parameters:
        - window: Number of periods for calculating average gains/losses.
        - in_place: Whether to modify the original DataFrame.
        """
        super().__init__(in_place)
        self.window = window

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the 'rsi' column to the DataFrame based on 'return_1'.
        """
        df_ = self._prepare_df(df)
        pos_gain = df_["return_1"].where(df_["return_1"] > 0, 0).ewm(self.window).mean()
        neg_gain = df_["return_1"].where(df_["return_1"] < 0, 0).ewm(self.window).mean()
        rs = np.abs(pos_gain / neg_gain)
        df_["rsi"] = 100 * rs / (1 + rs)
        return df_


class BollingerBandsIndicator(BaseIndicator):
    """
    Computes Bollinger Bands (mid, upper, and lower) using exponential moving average and rolling std.
    """

    def __init__(self, window: int = 10, in_place: bool = False):
        """
        Parameters:
        - window: Rolling window for standard deviation and EMA calculation.
        - in_place: Whether to modify the original DataFrame.
        """
        super().__init__(in_place)
        self.window = window

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'bollinger_mid', 'bollinger_upper', and 'bollinger_lower' to the DataFrame.
        Assumes the DataFrame contains a 'Close' column.
        """
        df_ = self._prepare_df(df)
        close = df_["Close"]
        mid = close.ewm(span=self.window).mean()
        std = close.rolling(self.window).std()
        df_["bollinger_mid"] = mid
        df_["bollinger_lower"] = mid - 2 * std
        df_["bollinger_upper"] = mid + 2 * std
        return df_