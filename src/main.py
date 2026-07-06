import pandas as pd
from pathlib import Path
from hard_outlier_filters_02 import HardOutlierFilters
from data_preprocessing_04 import DataPreprocessing
from loan_data_preprocessing_05 import LoanDataPreprocessing
from baseline_xgboost_07 import BaselineXGBoost
from smote_xgboost_07 import SmoteXGBoost
from weighted_xgboost_07 import WeightedXGBoost


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pre-processed"
raw_test = pd.read_csv(DATA_DIR / "loan_data_raw_test_1.csv")
clean_train = pd.read_csv(DATA_DIR / "loan_data_clean_train_3.csv")
smoted_train = pd.read_csv(DATA_DIR / "loan_data_smoted_raw_train_4.csv")


HardOutlierFilters().run()
DataPreprocessing().run()
LoanDataPreprocessing().run()
baseline_xgboost = BaselineXGBoost(clean_train, raw_test)
baseline_xgboost.run()
smote_xgboost = SmoteXGBoost(smoted_train, raw_test)
smote_xgboost.run()
weighted_xgboost = WeightedXGBoost(clean_train, raw_test)
weighted_xgboost.run()
