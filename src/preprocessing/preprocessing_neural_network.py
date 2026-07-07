from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DATA_PATH = PROJECT_ROOT / "data" / "pre-processed" / "loan_data_preprocessed.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "pre-processed"


class NeuralNetworkPreprocessing:
    def __init__(self, df):
        self.df = df
        self.target_column = "loan_status"
        self.education_column = "person_education"
        self.test_size = 0.2
        self.random_state = 1

        self.train_df = None
        self.test_df = None
        self.train_processed_df = None
        self.test_processed_df = None
        self.categorical_columns = None
        self.continuous_columns = None
        self.feature_names = None

        self.continuous_scaler = StandardScaler()
        self.category_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def split_train_test(self):
        train_df, test_df = train_test_split(self.df, test_size=self.test_size, stratify=self.df[self.target_column],
                                             random_state=self.random_state)
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

    def fit_transformers(self):
        X_train = self.train_df.drop(columns=[self.target_column])
        self.categorical_columns = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        self.continuous_columns = X_train.select_dtypes(include="number").columns.tolist()
        self.continuous_columns.remove(self.education_column)

        self.continuous_scaler.fit(X_train[self.continuous_columns])
        self.category_encoder.fit(X_train[self.categorical_columns])
        self.onehot_encoder.fit(X_train[self.categorical_columns])

        onehot_columns = self.onehot_encoder.get_feature_names_out(self.categorical_columns).tolist()
        self.feature_names = self.continuous_columns + [self.education_column] + onehot_columns

    def transform_features(self, X):
        continuous_values = self.continuous_scaler.transform(X[self.continuous_columns])
        education_values = X[[self.education_column]].round().clip(1, 5).astype(int).to_numpy()
        categorical_values = self.onehot_encoder.transform(X[self.categorical_columns])

        transformed_df = pd.DataFrame(np.hstack([continuous_values, education_values, categorical_values]),
                                      columns=self.feature_names)
        transformed_df[self.education_column] = transformed_df[self.education_column].astype(int)
        return transformed_df

    def preprocess(self):
        self.split_train_test()
        self.fit_transformers()

        X_train = self.train_df.drop(columns=[self.target_column])
        X_test = self.test_df.drop(columns=[self.target_column])
        train_features = self.transform_features(X_train)
        test_features = self.transform_features(X_test)

        self.train_processed_df = pd.concat([train_features, self.train_df[self.target_column]], axis=1)
        self.test_processed_df = pd.concat([test_features, self.test_df[self.target_column]], axis=1)

    def save(self):
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        self.train_processed_df.to_csv(OUTPUT_DIRECTORY / "loan_data_nn_normal_train.csv", index=False)
        self.test_processed_df.to_csv(OUTPUT_DIRECTORY / "loan_data_nn_normal_test.csv", index=False)

    def run(self):
        self.preprocess()
        self.save()
        return self.train_processed_df, self.test_processed_df


if __name__ == "__main__":
    neural_network_preprocessing = NeuralNetworkPreprocessing(pd.read_csv(INPUT_DATA_PATH))
    train_df, test_df = neural_network_preprocessing.run()
