import pandas as pd
from base import DataProvider

class FinnhubProvider(DataProvider):
    def download(self, symbol: str, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("FinnhubProvider is not yet implemented.")
