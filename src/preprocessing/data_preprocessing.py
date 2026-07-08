from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DATA_PATH = PROJECT_ROOT / "data" / "pre-processed" / "loan_data_hard_filtered.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "pre-processed"


class DataPreprocessing:
    def __init__(self, df):
        self.df = df
        self.output_filename = "loan_data_preprocessed.csv"
        self.bias_column = "previous_loan_defaults_on_file"
        self.education_mapping_rules = {
            "High School": 1,
            "Associate": 2,
            "Bachelor": 3,
            "Master": 4,
            "Doctorate": 5,
        }
        self.rounded_column = "person_education"
        self.preprocessed_df = None

    def preprocess(self):
        df = self.df.drop(columns=[self.bias_column])
        df[self.rounded_column] = df[self.rounded_column].map(self.education_mapping_rules)
        self.preprocessed_df = df

    def save(self):
        self.preprocessed_df.to_csv(OUTPUT_DIRECTORY / self.output_filename, index=False)

    def run(self):
        self.preprocess()
        self.save()
        return self.preprocessed_df


if __name__ == "__main__":
    data_preprocessing = DataPreprocessing(pd.read_csv(INPUT_DATA_PATH))
    preprocessed_df = data_preprocessing.run()
