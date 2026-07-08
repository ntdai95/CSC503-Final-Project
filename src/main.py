import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from preprocessing import HardOutlierFilters, DataPreprocessing, XGBoostPreprocessing, NeuralNetworkPreprocessing, NeuralNetworkSmotePreprocessing
from model.extreme_gradient_boosting import BaselineXGBoost, SmoteXGBoost, WeightedXGBoost
from model.neural_network import NeuralNetwork, SmoteNeuralNetwork
from model.quantum import QELMIsing
from fairness import FairnessEnhancedModel


RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pre-processed"


################################## Preprocessing ##################################
raw_df = pd.read_csv(RAW_DATA_DIR / "loan_data.csv")
HardOutlierFilters(raw_df).run()

hard_filtered_df = pd.read_csv(DATA_DIR / "loan_data_hard_filtered.csv")
DataPreprocessing(hard_filtered_df).run()

preprocessed_df = pd.read_csv(DATA_DIR / "loan_data_preprocessed.csv")


###################################### XGBoost ####################################
XGBoostPreprocessing(preprocessed_df).run()

test_df = pd.read_csv(DATA_DIR / "loan_data_raw_test_1.csv")
clean_train_df = pd.read_csv(DATA_DIR / "loan_data_clean_train_3.csv")
smoted_train_df = pd.read_csv(DATA_DIR / "loan_data_smoted_raw_train_4.csv")

baseline_xgboost = BaselineXGBoost(clean_train_df, test_df)
baseline_xgboost.run()
smote_xgboost = SmoteXGBoost(smoted_train_df, test_df)
smote_xgboost.run()
weighted_xgboost = WeightedXGBoost(clean_train_df, test_df)
weighted_xgboost.run()


################################## Neural Network ##################################
NeuralNetworkPreprocessing(preprocessed_df).run()
NeuralNetworkSmotePreprocessing(preprocessed_df).run()

nn_train_df = pd.read_csv(DATA_DIR / "loan_data_nn_normal_train.csv")
nn_test_df = pd.read_csv(DATA_DIR / "loan_data_nn_normal_test.csv")
nn_smote_train_df = pd.read_csv(DATA_DIR / "loan_data_nn_normal_train_smote.csv")

neural_network = NeuralNetwork(nn_train_df, nn_test_df)
neural_network.run()
smote_neural_network = SmoteNeuralNetwork(nn_smote_train_df, nn_test_df)
smote_neural_network.run()


#################################### Quantum #######################################
qelm_train_df, qelm_test_df = QELMIsing().run()

qelm_smote_xgboost = SmoteXGBoost(qelm_train_df, qelm_test_df)
qelm_smote_xgboost.run()
qelm_weighted_xgboost = WeightedXGBoost(qelm_train_df, qelm_test_df)
qelm_weighted_xgboost.run()


################################### Fairness #######################################
fairness_train_df = clean_train_df.copy()
fairness_test_df = test_df.copy()
categorical_cols = fairness_train_df.drop(columns=["loan_status"]).select_dtypes(include=["object", "category", "string"]).columns
for col in categorical_cols:
    fairness_train_df[col] = fairness_train_df[col].astype("category")
    fairness_test_df[col] = fairness_test_df[col].astype("category")

fairness_estimator = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                   random_state=42, n_jobs=-1, enable_categorical=True)
fairness_enhanced_model = FairnessEnhancedModel(fairness_train_df, fairness_test_df, fairness_estimator)
fairness_enhanced_model.run()
