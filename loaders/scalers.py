import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Optional


class ProcessedAssetData:
    """
    A wrapper around a preprocessed financial DataFrame that:
    - Extracts forward return as the target
    - Normalizes feature columns
    - Prepares it for use in a trading environment
    """

    def __init__(self, df: pd.DataFrame, scaler: Optional[StandardScaler] = None):
        """
        Args:
            df (pd.DataFrame): DataFrame with 'forward_return' and feature columns.
            scaler (Optional[StandardScaler]): Pre-fit or new scaler.
        """
        assert "forward_return" in df.columns, "'forward_return' column missing in DataFrame"
        self.frame = df.copy()
        self.scaler = scaler or StandardScaler()

        self.features = self.frame.drop(columns=["forward_return"])
        self.targets = self.frame["forward_return"].values

        # Fit or apply scaler
        self.features_scaled = self.scaler.fit_transform(self.features)

        # Store (target, features) tuples for __getitem__
        self._data: List[np.ndarray] = [
            np.concatenate(([self.targets[i]], self.features_scaled[i]))
            for i in range(len(self.frame))
        ]

    def __getitem__(self, idx: int) -> np.ndarray:
        """Returns a single row: [forward_return, feature_1, ..., feature_n]"""
        return self._data[idx]

    def __len__(self) -> int:
        """Returns number of samples (timesteps)"""
        return len(self._data)
