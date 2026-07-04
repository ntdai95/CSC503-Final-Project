from pathlib import Path
import pandas as pd
from basemodel_xgboost import BaseModelXGBoost


INPUT_TRAINING_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "loan_data_smoted_raw_train_4.csv"
INPUT_TESTING_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "loan_data_raw_test_1.csv"

class SmoteXGBoost(BaseModelXGBoost):
    def __init__(self):
        self.train_df = pd.read_csv(INPUT_TRAINING_DATA_PATH)
        self.test_df = pd.read_csv(INPUT_TESTING_DATA_PATH)
        super().__init__()


if __name__ == "__main__":
    smote_xgboost = SmoteXGBoost()
    smote_xgboost.run()
