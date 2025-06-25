import yfinance as yf
import pandas as pd
from .base import DataProvider

class YFinanceProvider(DataProvider):
    def download(self, symbol: str, **kwargs) -> pd.DataFrame:
        df = yf.download(symbol, **kwargs)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
