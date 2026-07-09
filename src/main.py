import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from preprocessing import HardOutlierFilters, DataPreprocessing, XGBoostPreprocessing, NeuralNetworkPreprocessing, NeuralNetworkSmotePreprocessing
from model.extreme_gradient_boosting import BaseModelXGBoost, WeightedXGBoost
from model.neural_network import NeuralNetwork, SmoteNeuralNetwork
from model.quantum import QELMIsing
from fairness import FairnessEnhancedModel, SampleWeightedMLPClassifier


RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pre-processed"


################################## Preprocessing ##################################
raw_df = pd.read_csv(RAW_DATA_DIR / "loan_data.csv")
HardOutlierFilters(raw_df).run()

hard_filtered_df = pd.read_csv(DATA_DIR / "loan_data_hard_filtered.csv")
DataPreprocessing(hard_filtered_df).run()

preprocessed_df = pd.read_csv(DATA_DIR / "loan_data_preprocessed.csv")


###################################### XGBoost ####################################
print("\n###################################### XGBoost ####################################\n")
XGBoostPreprocessing(preprocessed_df).run()

test_df = pd.read_csv(DATA_DIR / "loan_data_raw_test_1.csv")
clean_train_df = pd.read_csv(DATA_DIR / "loan_data_clean_train_3.csv")
smoted_train_df = pd.read_csv(DATA_DIR / "loan_data_smoted_raw_train_4.csv")

print("\nBaseline XGBOOST:\n")
baseline_xgboost = BaseModelXGBoost(clean_train_df, test_df)
baseline_xgboost.run()
print("\nSmote XGBOOST:\n")
smote_xgboost = BaseModelXGBoost(smoted_train_df, test_df)
smote_xgboost.run()
print("\nWeighted XGBOOST:\n")
weighted_xgboost = WeightedXGBoost(clean_train_df, test_df)
weighted_xgboost.run()


################################## Neural Network ##################################
print("\n################################## Neural Network ##################################\n")
NeuralNetworkPreprocessing(preprocessed_df).run()
NeuralNetworkSmotePreprocessing(preprocessed_df).run()

nn_train_df = pd.read_csv(DATA_DIR / "loan_data_nn_normal_train.csv")
nn_test_df = pd.read_csv(DATA_DIR / "loan_data_nn_normal_test.csv")
nn_smote_train_df = pd.read_csv(DATA_DIR / "loan_data_nn_normal_train_smote.csv")

print("\nNeural Network:\n")
neural_network = NeuralNetwork(nn_train_df, nn_test_df)
neural_network.run()
print("\nSmote Neural Network:\n")
smote_neural_network = SmoteNeuralNetwork(nn_smote_train_df, nn_test_df)
smote_neural_network.run()


#################################### Quantum #######################################
print("\n################################## Quantum ##################################\n")
qelm_train_df, qelm_test_df = QELMIsing().run()

print("\nQuantum Smote XGBoost:\n")
qelm_smote_xgboost = BaseModelXGBoost(qelm_train_df, qelm_test_df)
qelm_smote_xgboost.run()
print("\nQuantum Weighted XGBoost:\n")
qelm_weighted_xgboost = WeightedXGBoost(qelm_train_df, qelm_test_df)
qelm_weighted_xgboost.run()
print("\nQuantum Smote Neural Network:\n")
qelm_smote_neural_network = SmoteNeuralNetwork(qelm_train_df, qelm_test_df, config={"model_filename": "neural_network_qelm_smote.joblib"})
qelm_smote_neural_network.run()


################################### Fairness #######################################
print("\n################################## Fairness ##################################\n")
fairness_train_df = clean_train_df.copy()
fairness_test_df = test_df.copy()
categorical_cols = fairness_train_df.drop(columns=["loan_status"]).select_dtypes(include=["object", "category", "string"]).columns
for col in categorical_cols:
    fairness_train_df[col] = fairness_train_df[col].astype("category")
    fairness_test_df[col] = fairness_test_df[col].astype("category")

print("\nFairness Weighted XGBOOST:\n")
fairness_xgboost_estimator = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                           random_state=42, n_jobs=-1, enable_categorical=True)
fairness_xgboost_model = FairnessEnhancedModel(fairness_train_df, fairness_test_df, fairness_xgboost_estimator,
                                               needs_encoding=False, model_name="xgboost", max_iter=20)
fairness_xgboost_model.run()

print("\nFairness Logistic Regression:\n")
fairness_lr_estimator = LogisticRegression(max_iter=1000, random_state=42)
fairness_lr_model = FairnessEnhancedModel(fairness_train_df, fairness_test_df, fairness_lr_estimator,
                                          needs_encoding=True, model_name="logistic_regression", max_iter=20)
fairness_lr_model.run()

print("\nFairness Neural Network:\n")
fairness_nn_estimator = SampleWeightedMLPClassifier(mlp_kwargs={"hidden_layer_sizes": (128, 64), "max_iter": 600}, random_state=42)
fairness_nn_model = FairnessEnhancedModel(fairness_train_df, fairness_test_df, fairness_nn_estimator,
                                          needs_encoding=True, model_name="neural_network", max_iter=20)
fairness_nn_model.run()

print("\nFairness Quantum XGBOOST:\n")
fairness_qelm_estimator = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                        random_state=42, n_jobs=-1)
fairness_qelm_model = FairnessEnhancedModel(qelm_train_df, qelm_test_df, fairness_qelm_estimator,
                                            sensitive_source_train_df=smoted_train_df,
                                            sensitive_source_test_df=test_df,
                                            needs_encoding=False, model_name="qelm_xgboost", max_iter=20)
fairness_qelm_model.run()
