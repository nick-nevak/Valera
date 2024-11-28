from scipy.stats import mode
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors


class DataFrameConverter(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)


class KNNModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.knn.fit(X[~np.isnan(X).any(axis=1)])
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        col_data = X[:, 0].reshape(-1, 1)
        nan_mask = np.isnan(col_data)

        # Ensure there are NaNs to be imputed
        if not nan_mask.any():
            return X

        # Find nearest neighbors for rows without NaN values
        distances, neighbors = self.knn.kneighbors(
            col_data[~nan_mask].reshape(-1, 1))

        for i, idx in enumerate(np.where(nan_mask)[0]):
            neighbor_vals = col_data[neighbors[i]].flatten()
            mode_result = mode(neighbor_vals, nan_policy='omit')
            count_value = mode_result.count if np.isscalar(
                mode_result.count) else mode_result.count[0]
            imputed_value = mode_result.mode if np.isscalar(
                mode_result.mode) else mode_result.mode[0]
            col_data[idx, 0] = imputed_value

        # Ensure column assignment back to X
        X[:, 0] = col_data.flatten()

        return X
