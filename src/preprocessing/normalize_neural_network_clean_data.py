from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import DATA_COLUMN_CONFIG, NORMALIZE_NEURAL_NETWORK_CLEAN_DATA_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "pre-processed"


class NeuralNetworkCleanDataNormalizer:
    def __init__(self, train_df, test_df, config=None, data_config=None):
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        self.config = {**NORMALIZE_NEURAL_NETWORK_CLEAN_DATA_CONFIG, **(config or {})}
        self.data_config = {**DATA_COLUMN_CONFIG, **(data_config or {})}

        self.target_column = self.data_config["target_column"]
        self.education_column = self.data_config["education_column"]
        self.education_min = self.data_config["education_min"]
        self.education_max = self.data_config["education_max"]

        self.normalized_train_df = None
        self.normalized_test_df = None
        self.categorical_columns = None
        self.continuous_columns = None
        self.feature_names = None
        self.continuous_scaler = StandardScaler()
        self.onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit_transformers(self):
        X_train = self.train_df.drop(columns=[self.target_column])
        self.categorical_columns = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        self.continuous_columns = X_train.select_dtypes(include="number").columns.tolist()
        self.continuous_columns.remove(self.education_column)

        self.continuous_scaler.fit(X_train[self.continuous_columns])
        self.onehot_encoder.fit(X_train[self.categorical_columns])

        onehot_columns = self.onehot_encoder.get_feature_names_out(self.categorical_columns).tolist()
        self.feature_names = self.continuous_columns + [self.education_column] + onehot_columns

    def transform_features(self, X):
        continuous_values = self.continuous_scaler.transform(X[self.continuous_columns])
        education_values = X[[self.education_column]].round().clip(self.education_min, self.education_max).astype(int).to_numpy()
        categorical_values = self.onehot_encoder.transform(X[self.categorical_columns])

        transformed_df = pd.DataFrame(np.hstack([continuous_values, education_values, categorical_values]),
                                      columns=self.feature_names)
        transformed_df[self.education_column] = transformed_df[self.education_column].astype(int)
        return transformed_df

    def normalize(self):
        if set(self.train_df.columns) != set(self.test_df.columns):
            raise ValueError("The training and test files do not contain the same columns.")

        self.test_df = self.test_df[self.train_df.columns]
        self.fit_transformers()

        train_features = self.transform_features(self.train_df.drop(columns=[self.target_column]))
        test_features = self.transform_features(self.test_df.drop(columns=[self.target_column]))
        self.normalized_train_df = pd.concat([train_features, self.train_df[self.target_column]], axis=1)
        self.normalized_test_df = pd.concat([test_features, self.test_df[self.target_column]], axis=1)

    def save(self):
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        self.normalized_train_df.to_csv(OUTPUT_DIRECTORY / self.config["train_output_filename"], index=False)
        self.normalized_test_df.to_csv(OUTPUT_DIRECTORY / self.config["test_output_filename"], index=False)

    def run(self):
        self.normalize()
        self.save()
        return self.normalized_train_df, self.normalized_test_df


if __name__ == "__main__":
    train_path = OUTPUT_DIRECTORY / NORMALIZE_NEURAL_NETWORK_CLEAN_DATA_CONFIG["train_input_filename"]
    test_path = OUTPUT_DIRECTORY / NORMALIZE_NEURAL_NETWORK_CLEAN_DATA_CONFIG["test_input_filename"]
    normalizer = NeuralNetworkCleanDataNormalizer(pd.read_csv(train_path), pd.read_csv(test_path))
    train_df, test_df = normalizer.run()
