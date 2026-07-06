from pathlib import Path
import pandas as pd
from basemodel_xgboost_06 import BaseModelXGBoost


INPUT_TRAINING_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "loan_data_smoted_raw_train_4.csv"
INPUT_TESTING_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "loan_data_raw_test_1.csv"

class SmoteXGBoost(BaseModelXGBoost):
    def __init__(self, train_df, test_df):
        super().__init__(train_df, test_df)


if __name__ == "__main__":
    smote_xgboost = SmoteXGBoost(pd.read_csv(INPUT_TRAINING_DATA_PATH), pd.read_csv(INPUT_TESTING_DATA_PATH))
    smote_xgboost.run()
