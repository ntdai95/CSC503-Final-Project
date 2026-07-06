import pandas as pd
from pathlib import Path


INPUT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "loan_data_hard_filtered.csv"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "pre-processed"

class DataPreprocessing:
    def __init__(self):
        self.df = pd.read_csv(INPUT_DATA_PATH)
        self.bias_column = "previous_loan_defaults_on_file"
        self.rounded_column = "person_education"
        self.education_mapping_rules = {
            "High School": 1,
            "Associate": 2,
            "Bachelor": 3,
            "Master": 4,
            "Doctorate": 5,
        }
        
        self.preprocessed_df = None

    def preprocess(self):
        df = self.df.drop(columns=[self.bias_column])
        df[self.rounded_column] = df[self.rounded_column].map(self.education_mapping_rules)
        self.preprocessed_df = df

    def save(self):
        self.preprocessed_df.to_csv(OUTPUT_DIRECTORY / "loan_data_preprocessed.csv", index=False)

    def run(self):
        self.preprocess()
        self.save()
        return self.preprocessed_df


if __name__ == "__main__":
    data_preprocessing = DataPreprocessing()
    preprocessed_df = data_preprocessing.run()
