import pandas as pd
from pathlib import Path
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
print("\n###################################### XGBoost ####################################")
XGBoostPreprocessing(preprocessed_df).run()

test_df = pd.read_csv(DATA_DIR / "loan_data_raw_test_1.csv")
clean_train_df = pd.read_csv(DATA_DIR / "loan_data_clean_train_3.csv")
smoted_train_df = pd.read_csv(DATA_DIR / "loan_data_smoted_raw_train_4.csv")

print("\n\nBaseline XGBOOST:\n")
baseline_xgboost = BaseModelXGBoost(clean_train_df, test_df)
baseline_xgboost.run()
print("\n\nSmote XGBOOST:\n")
smote_xgboost = BaseModelXGBoost(smoted_train_df, test_df)
smote_xgboost.run()
print("\n\nWeighted XGBOOST:\n")
weighted_xgboost = WeightedXGBoost(clean_train_df, test_df)
weighted_xgboost.run()


################################## Neural Network ##################################
print("\n################################## Neural Network ##################################")
NeuralNetworkPreprocessing(preprocessed_df).run()
NeuralNetworkSmotePreprocessing(preprocessed_df).run()

nn_train_df = pd.read_csv(DATA_DIR / "loan_data_nn_normal_train.csv")
nn_test_df = pd.read_csv(DATA_DIR / "loan_data_nn_normal_test.csv")
nn_smote_train_df = pd.read_csv(DATA_DIR / "loan_data_nn_normal_train_smote.csv")

print("\n\nNeural Network:\n")
neural_network = NeuralNetwork(nn_train_df, nn_test_df)
neural_network.run()
print("\n\nSmote Neural Network:\n")
smote_neural_network = SmoteNeuralNetwork(nn_smote_train_df, nn_test_df)
smote_neural_network.run()


#################################### Quantum #######################################
print("\n################################## Quantum ##################################")
qelm_train_df, qelm_test_df = QELMIsing().run()

print("\n\nQuantum Smote XGBoost:\n")
qelm_smote_xgboost = BaseModelXGBoost(qelm_train_df, qelm_test_df)
qelm_smote_xgboost.run()
print("\n\nQuantum Weighted XGBoost:\n")
qelm_weighted_xgboost = WeightedXGBoost(qelm_train_df, qelm_test_df)
qelm_weighted_xgboost.run()
print("\n\nQuantum Smote Neural Network:\n")
qelm_smote_neural_network = SmoteNeuralNetwork(qelm_train_df, qelm_test_df, config={"model_filename": "neural_network_qelm_smote.joblib"})
qelm_smote_neural_network.run()


################################### Fairness #######################################
print("\n################################## Fairness ##################################")
fairness_train_df = clean_train_df.copy()
fairness_test_df = test_df.copy()
categorical_cols = fairness_train_df.drop(columns=["loan_status"]).select_dtypes(include=["object", "category", "string"]).columns
for col in categorical_cols:
    fairness_train_df[col] = fairness_train_df[col].astype("category")
    fairness_test_df[col] = fairness_test_df[col].astype("category")

print("\n\nFairness Baseline XGBOOST:\n")
def tuned_xgboost_estimator(base_model):
    estimator = XGBClassifier()
    for param_name, param_value in base_model.get_best_model_params().items():
        setattr(estimator, param_name, param_value)

    estimator.enable_categorical = True
    return estimator

fairness_xgboost_estimator = tuned_xgboost_estimator(baseline_xgboost)
fairness_xgboost_model = FairnessEnhancedModel(fairness_train_df, fairness_test_df, fairness_xgboost_estimator,
                                               needs_encoding=False, model_name="xgboost", max_iter=20)
fairness_xgboost_model.run()

print("\n\nFairness QELM + Smote XGBOOST:\n")
def tuned_nn_estimator(base_model):
    params = dict(base_model.get_best_model_params())
    random_state = params.pop("random_state", None)
    return SampleWeightedMLPClassifier(mlp_kwargs=params, random_state=random_state)

fairness_qelm_xgboost_estimator = tuned_xgboost_estimator(qelm_smote_xgboost)
fairness_qelm_xgboost_model = FairnessEnhancedModel(qelm_train_df, qelm_test_df, fairness_qelm_xgboost_estimator,
                                                    sensitive_source_train_df=smoted_train_df,
                                                    sensitive_source_test_df=test_df,
                                                    needs_encoding=False, model_name="qelm_xgboost", max_iter=20)
fairness_qelm_xgboost_model.run()

print("\n\nFairness Neural Network:\n")
fairness_nn_estimator = tuned_nn_estimator(neural_network)
fairness_nn_model = FairnessEnhancedModel(fairness_train_df, fairness_test_df, fairness_nn_estimator,
                                          needs_encoding=True, model_name="neural_network", max_iter=20)
fairness_nn_model.run()

print("\n\nFairness QELM + Neural Network:\n")
fairness_qelm_nn_estimator = tuned_nn_estimator(qelm_smote_neural_network)
fairness_qelm_nn_model = FairnessEnhancedModel(qelm_train_df, qelm_test_df, fairness_qelm_nn_estimator,
                                               sensitive_source_train_df=smoted_train_df,
                                               sensitive_source_test_df=test_df,
                                               needs_encoding=False, model_name="qelm_neural_network", max_iter=20)
fairness_qelm_nn_model.run()
