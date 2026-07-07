from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from .preprocessing_neural_network import NeuralNetworkPreprocessing


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DATA_PATH = PROJECT_ROOT / "data" / "pre-processed" / "loan_data_preprocessed.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "pre-processed"


class NeuralNetworkSmotePreprocessing(NeuralNetworkPreprocessing):
    def __init__(self, df):
        super().__init__(df)
        self.k_neighbors = 5
        self.smote_train_df = None

    def apply_smote(self):
        X_train = self.train_df.drop(columns=[self.target_column])
        y_train = self.train_df[self.target_column]

        continuous_values = self.continuous_scaler.transform(X_train[self.continuous_columns])
        education_values = X_train[[self.education_column]].round().clip(1, 5).astype(int).to_numpy()
        categorical_values = self.category_encoder.transform(X_train[self.categorical_columns])
        smote_columns = self.continuous_columns + [self.education_column] + self.categorical_columns
        X_smote = np.hstack([continuous_values, education_values, categorical_values])

        first_categorical_index = len(self.continuous_columns)
        categorical_indices = list(range(first_categorical_index, len(smote_columns)))
        smote = SMOTENC(categorical_features=categorical_indices, sampling_strategy="auto",
                        k_neighbors=self.k_neighbors, random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_smote, y_train)

        resampled_df = pd.DataFrame(X_resampled, columns=smote_columns)
        resampled_df[self.education_column] = resampled_df[self.education_column].round().clip(1, 5).astype(int)
        resampled_df[self.categorical_columns] = resampled_df[self.categorical_columns].round()

        category_labels = self.category_encoder.inverse_transform(resampled_df[self.categorical_columns])
        category_labels_df = pd.DataFrame(category_labels, columns=self.categorical_columns)
        onehot_values = self.onehot_encoder.transform(category_labels_df)
        education_values = resampled_df[[self.education_column]].to_numpy()

        features = np.hstack([resampled_df[self.continuous_columns].to_numpy(), education_values, onehot_values])
        features_df = pd.DataFrame(features, columns=self.feature_names)
        features_df[self.education_column] = features_df[self.education_column].astype(int)
        self.smote_train_df = pd.concat([features_df, pd.Series(y_resampled, name=self.target_column)], axis=1)

        print("Class distribution before SMOTENC:")
        print(y_train.value_counts())
        print("\nClass distribution after SMOTENC:")
        print(self.smote_train_df[self.target_column].value_counts())

    def save(self):
        super().save()
        self.smote_train_df.to_csv(OUTPUT_DIRECTORY / "loan_data_nn_normal_train_smote.csv", index=False)

    def run(self):
        self.preprocess()
        self.apply_smote()
        self.save()
        return self.smote_train_df, self.train_processed_df, self.test_processed_df


if __name__ == "__main__":
    neural_network_smote_preprocessing = NeuralNetworkSmotePreprocessing(pd.read_csv(INPUT_DATA_PATH))
    smote_train_df, train_df, test_df = neural_network_smote_preprocessing.run()
