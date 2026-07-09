from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_data.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "pre-processed"


class HardOutlierFilters:
    def __init__(self, df):
        self.raw_df = df
        self.output_filename = "loan_data_hard_filtered.csv"
        self.bad_mask = None
        self.clean_df = None

    def apply_hard_filters(self):
        raw = self.raw_df
        filters = {
            "age < 18 (below legal loan age)": raw["person_age"] < 18,
            "age > 120 (physiologically impossible)": raw["person_age"] > 120,
            "emp_exp < 0 (negative experience)": raw["person_emp_exp"] < 0,
            "emp_exp > 80 (impossible career length)": raw["person_emp_exp"] > 80,
            "emp_exp > (age-16) (worked before age 16)": raw["person_emp_exp"] > (raw["person_age"] - 16),
            "credit_score < 300 (below FICO floor)": raw["credit_score"] < 300,
            "credit_score > 850 (above FICO ceiling)": raw["credit_score"] > 850,
            "income <= 0 (non-positive income)": raw["person_income"] <= 0,
            "loan_amnt <= 0 (non-positive loan)": raw["loan_amnt"] <= 0,
            "interest_rate < 0 (negative rate)": raw["loan_int_rate"] < 0,
            "interest_rate > 100 (illegal interest rate / error)": raw["loan_int_rate"] > 100,
            "loan_percent_income < 0 (negative ratio)": raw["loan_percent_income"] < 0,
            "cred_hist > age (history longer than life)": raw["cb_person_cred_hist_length"] > raw["person_age"],
        }

        self.bad_mask = pd.concat(filters.values(), axis=1).any(axis=1)
        self.clean_df = raw[~self.bad_mask].copy()

    def save(self):
        self.clean_df.to_csv(OUTPUT_DIRECTORY / self.output_filename, index=False)

    def run(self):
        self.apply_hard_filters()
        self.save()
        return self.clean_df


if __name__ == "__main__":
    hard_outlier_filters = HardOutlierFilters(pd.read_csv(INPUT_DATA_PATH))
    clean_df = hard_outlier_filters.run()
