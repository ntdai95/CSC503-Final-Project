import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_random_state


class SampleWeightedMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, mlp_kwargs=None, random_state=None):
        self.mlp_kwargs = mlp_kwargs
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        y = np.asarray(y)
        if sample_weight is None:
            X_resampled, y_resampled = X, y
        else:
            probabilities = sample_weight / np.sum(sample_weight)
            rng = check_random_state(self.random_state)
            sampled_indices = rng.choice(len(X), size=len(X), replace=True, p=probabilities)
            X_resampled, y_resampled = X[sampled_indices], y[sampled_indices]

        self.model_ = MLPClassifier(random_state=self.random_state, **(self.mlp_kwargs or {}))
        self.model_.fit(X_resampled, y_resampled)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
