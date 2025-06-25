import pandas as pd

class BaseIndicator:
    """
    Base class for all technical indicators.
    Provides shared logic for in-place DataFrame modification and interface enforcement.
    """

    def __init__(self, in_place: bool = False):
        """
        Parameters:
        - in_place: If True, modify the input DataFrame directly. If False, operate on a copy.
        """
        self.in_place = in_place

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the indicator calculation to the provided DataFrame.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Each indicator must implement the apply method.")

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of the DataFrame unless in_place is True.
        """
        return df if self.in_place else df.copy()
