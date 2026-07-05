from pathlib import Path
import pandas as pd
from scipy.stats import uniform
from basemodel_xgboost import BaseModelXGBoost


INPUT_TRAINING_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "loan_data_clean_train_3.csv"
INPUT_TESTING_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "loan_data_raw_test_1.csv"

class WeightedXGBoost(BaseModelXGBoost):
    def __init__(self, train_df, test_df):
        super().__init__(train_df, test_df)
        self.base_scale_pos_weight = (self.train_df[self.target_column] == 0).sum() / (self.train_df[self.target_column] == 1).sum()
        self.param_distributions["model__scale_pos_weight"] = uniform(self.base_scale_pos_weight * 0.5, self.base_scale_pos_weight)


if __name__ == "__main__":
    weighted_xgboost = WeightedXGBoost(pd.read_csv(INPUT_TRAINING_DATA_PATH), pd.read_csv(INPUT_TESTING_DATA_PATH))
    weighted_xgboost.run()
