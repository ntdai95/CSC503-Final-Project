import pandas as pd
from pathlib import Path
from hard_outlier_filters_02 import HardOutlierFilters
from data_preprocessing_04 import DataPreprocessing
from loan_data_preprocessing_05 import LoanDataPreprocessing
from baseline_xgboost_07 import BaselineXGBoost
from smote_xgboost_07 import SmoteXGBoost
from weighted_xgboost_07 import WeightedXGBoost


RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pre-processed"

raw_df = pd.read_csv(RAW_DATA_DIR / "loan_data.csv")
HardOutlierFilters(raw_df).run()

hard_filtered_df = pd.read_csv(DATA_DIR / "loan_data_hard_filtered.csv")
DataPreprocessing(hard_filtered_df).run()

preprocessed_df = pd.read_csv(DATA_DIR / "loan_data_preprocessed.csv")
LoanDataPreprocessing(preprocessed_df).run()

test_df = pd.read_csv(DATA_DIR / "loan_data_raw_test_1.csv")
clean_train_df = pd.read_csv(DATA_DIR / "loan_data_clean_train_3.csv")
smoted_train_df = pd.read_csv(DATA_DIR / "loan_data_smoted_raw_train_4.csv")
baseline_xgboost = BaselineXGBoost(clean_train_df, test_df)
baseline_xgboost.run()
smote_xgboost = SmoteXGBoost(smoted_train_df, test_df)
smote_xgboost.run()
weighted_xgboost = WeightedXGBoost(clean_train_df, test_df)
weighted_xgboost.run()
