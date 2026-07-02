import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


INPUT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "loan_data_preprocessed.csv"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "pre-processed"

class LoanDataPreprocessing:
    def __init__(self):
        self.df = pd.read_csv(INPUT_DATA_PATH)
        self.target_column = "loan_status"
        self.test_size = 0.2
        self.random_state = 42
        self.n_estimators = 100
        self.contamination = 0.05
        self.categorical_columns = ["person_gender", "person_home_ownership", "loan_intent"]
        self.rounded_column = "person_education"
        self.train_df = None
        self.test_df = None
        self.outlier_train_df = None
        self.clean_train_df = None
        self.smote_train_df = None
    
    def split_train_test(self):
        train_df, test_df = train_test_split(self.df, test_size=self.test_size, stratify=self.df[self.target_column],
                                             random_state=self.random_state)
        
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        self.test_df.to_csv(OUTPUT_DIRECTORY / "loan_data_raw_test_1.csv", index=False)

    def remove_outliers(self):
        numerical_features = []
        for feature_column in self.train_df.select_dtypes(include="number").columns:
            if feature_column != "loan_status":
                numerical_features.append(feature_column)

        X = self.train_df[numerical_features].values

        isolation_forest = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination, 
                                           random_state=self.random_state)
        predictions = isolation_forest.fit_predict(X)

        self.outlier_train_df = self.train_df[predictions == -1].reset_index(drop=True)
        self.clean_train_df = self.train_df[predictions == 1].reset_index(drop=True)

        self.outlier_train_df.to_csv(OUTPUT_DIRECTORY / "loan_data_outlier_train_2.csv", index=False)
        self.clean_train_df.to_csv(OUTPUT_DIRECTORY / "loan_data_clean_train_3.csv", index=False)

    def apply_smote_nc(self):
        X = self.clean_train_df.drop(columns=self.target_column)
        y = self.clean_train_df[self.target_column]

        categorical_features = []
        for column in self.categorical_columns:
            categorical_features.append(X.columns.get_loc(column))

        smote_nc = SMOTENC(sampling_strategy="auto", categorical_features=categorical_features,
                           random_state=self.random_state)
        smote_X, smote_y = smote_nc.fit_resample(X, y)

        smote_df = smote_X.reset_index(drop=True)
        smote_df[self.rounded_column] = smote_df[self.rounded_column].round().astype(int)
        smote_df[self.target_column] = pd.Series(smote_y).reset_index(drop=True)

        self.smote_train_df = smote_df
        self.smote_train_df.to_csv(OUTPUT_DIRECTORY / "loan_data_smoted_raw_train_4.csv", index=False)

    def run(self):
        self.split_train_test()
        self.remove_outliers()
        self.apply_smote_nc()
        return self.smote_train_df, self.test_df


if __name__ == "__main__":
    # How to call:
    loan_data_preprocessing = LoanDataPreprocessing()
    train_df, test_df = loan_data_preprocessing.run()