import pandas as pd
from fairlearn.reductions import ClassificationMoment, ErrorRate, FalsePositiveRateParity, TruePositiveRateParity


METRIC_MOMENT_CLASSES = {"fpr": FalsePositiveRateParity, "fnr": TruePositiveRateParity}


class MultiFeatureParity(ClassificationMoment):
    def __init__(self, feature_config):
        super().__init__()
        self.feature_config = feature_config
        self.sub_moments = {}

    def default_objective(self):
        return ErrorRate()

    def _load_sub_moment(self, X, y, feature_column, metric, bound):
        moment = METRIC_MOMENT_CLASSES[metric](difference_bound=bound)
        moment.load_data(X, y, sensitive_features=feature_column)
        return moment

    def load_data(self, X, y, sensitive_features):
        self.X = X
        self._y = y
        self.sub_moments = {}
        for feature, metrics in self.feature_config.items():
            for metric, bound in metrics.items():
                self.sub_moments[(feature, metric)] = self._load_sub_moment(
                    X, y, feature_column=sensitive_features[feature], metric=metric, bound=bound)
        self._index = self.bound().index

    @property
    def index(self):
        return self._index

    def _combine(self, parts):
        return pd.concat(dict(zip(self.sub_moments.keys(), parts)), names=["feature", "metric"])

    def _split(self, vec):
        return {key: vec.xs(key, level=("feature", "metric")) for key in self.sub_moments}

    def gamma(self, predictor):
        return self._combine([m.gamma(predictor) for m in self.sub_moments.values()])

    def bound(self):
        return self._combine([m.bound() for m in self.sub_moments.values()])

    def project_lambda(self, lambda_vec):
        split = self._split(lambda_vec)
        return self._combine([m.project_lambda(split[key]) for key, m in self.sub_moments.items()])

    def signed_weights(self, lambda_vec):
        split = self._split(lambda_vec)
        return sum(m.signed_weights(split[key]) for key, m in self.sub_moments.items())
