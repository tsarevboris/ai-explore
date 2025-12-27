"""Custom preprocessing transformers."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import rbf_kernel


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """
    Transformer that computes similarity to cluster centers using RBF kernel.

    Useful for adding spatial/geographic features based on clustering.
    """

    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        X = validate_data(self, X, ensure_2d=True)

        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)

        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return np.array([f"Cluster {i} similarity" for i in range(self.n_clusters)])


class KNeighborsMedian(BaseEstimator, TransformerMixin):
    """
    Transformer that predicts target value based on k-nearest neighbors.

    Uses KNeighborsRegressor to predict the average target value of the
    k nearest neighbors (not actually median, despite the name).
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        X = validate_data(self, X, ensure_2d=True)

        self.regressor_ = KNeighborsRegressor(n_neighbors=self.k)
        self.regressor_.fit(X, y)

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)

        return self.regressor_.predict(X).reshape(-1, 1)

    def get_feature_names_out(self, names=None):
        return np.array(["KNeighborsMedian"])
