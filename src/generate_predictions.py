import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from preprocessing import HardOutlierFilters, DataPreprocessing, XGBoostPreprocessing, NeuralNetworkCleanDataNormalizer, NeuralNetworkSmoteDataNormalizer
from model.extreme_gradient_boosting import BaseModelXGBoost, WeightedXGBoost
from model.neural_network import NeuralNetwork, SmoteNeuralNetwork
from model.quantum import QELMIsing
from fairness import FairnessEnhancedModel, SampleWeightedMLPClassifier
from config import NORMALIZE_NEURAL_NETWORK_CLEAN_DATA_CONFIG, NORMALIZE_NEURAL_NETWORK_SMOTE_DATA_CONFIG
from config import MODEL_NEURAL_NETWORK_CLEAN_DATA_CONFIG, MODEL_NEURAL_NETWORK_SMOTE_DATA_CONFIG


RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pre-processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "visualizations"


def tuned_xgboost_estimator(base_model):
    estimator = XGBClassifier()
    for param_name, param_value in base_model.get_best_model_params().items():
        setattr(estimator, param_name, param_value)

    estimator.enable_categorical = True
    return estimator


def tuned_nn_estimator(base_model):
    params = dict(base_model.get_best_model_params())
    random_state = params.pop("random_state", None)
    return SampleWeightedMLPClassifier(mlp_kwargs=params, random_state=random_state)


def write_predictions(predictions):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (y_test, y_pred) in predictions.items():
        output_df = pd.DataFrame({"y_test": pd.Series(y_test).reset_index(drop=True), 
                                  "y_pred": pd.Series(y_pred).reset_index(drop=True)})
        output_path = OUTPUT_DIR / f"{name}_y_test_y_pred.csv"
        output_df.to_csv(output_path, index=False)
        print("Saved predictions to:", output_path)


def main():
    predictions = {}

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

    print("\n\nXGBoost without SMOTE:\n")
    baseline_xgboost = BaseModelXGBoost(clean_train_df, test_df)
    baseline_xgboost.run()
    predictions["XGB_No_SMOTE"] = (baseline_xgboost.y_test, baseline_xgboost.y_pred)

    print("\n\nXGBoost with SMOTE:\n")
    smote_xgboost = BaseModelXGBoost(smoted_train_df, test_df)
    smote_xgboost.run()
    predictions["XGB_SMOTE"] = (smote_xgboost.y_test, smote_xgboost.y_pred)

    print("\n\nWeighted XGBoost:\n")
    weighted_xgboost = WeightedXGBoost(clean_train_df, test_df)
    weighted_xgboost.run()
    predictions["XGB_Weighted"] = (weighted_xgboost.y_test, weighted_xgboost.y_pred)

    ################################## Neural Network ##################################
    print("\n################################## Neural Network ##################################")
    NeuralNetworkCleanDataNormalizer(clean_train_df, test_df).run()
    NeuralNetworkSmoteDataNormalizer(smoted_train_df, test_df).run()
    nn_train_df = pd.read_csv(DATA_DIR / NORMALIZE_NEURAL_NETWORK_CLEAN_DATA_CONFIG["train_output_filename"])
    nn_test_df = pd.read_csv(DATA_DIR / NORMALIZE_NEURAL_NETWORK_CLEAN_DATA_CONFIG["test_output_filename"])
    nn_smote_train_df = pd.read_csv(DATA_DIR / NORMALIZE_NEURAL_NETWORK_SMOTE_DATA_CONFIG["train_output_filename"])
    nn_smote_test_df = pd.read_csv(DATA_DIR / NORMALIZE_NEURAL_NETWORK_SMOTE_DATA_CONFIG["test_output_filename"])

    print("\n\nNeural Network without SMOTE:\n")
    baseline_neural_network = NeuralNetwork(nn_train_df, nn_test_df, config=MODEL_NEURAL_NETWORK_CLEAN_DATA_CONFIG)
    baseline_neural_network.run()
    predictions["NN_No_SMOTE"] = (baseline_neural_network.y_test, baseline_neural_network.y_pred)

    print("\n\nNeural Network with SMOTE:\n")
    smote_neural_network = SmoteNeuralNetwork(nn_smote_train_df, nn_smote_test_df, config=MODEL_NEURAL_NETWORK_SMOTE_DATA_CONFIG)
    smote_neural_network.run()
    predictions["NN_SMOTE"] = (smote_neural_network.y_test, smote_neural_network.y_pred)

    #################################### Quantum #######################################
    print("\n################################## Quantum ##################################")
    qelm_train_df, qelm_test_df = QELMIsing().run()
    qelm_feature_columns = [column for column in qelm_train_df.columns if column != "loan_status"]

    qelm_scaler = StandardScaler()
    qelm_nn_train_df = qelm_train_df.copy()
    qelm_nn_test_df = qelm_test_df.copy()
    qelm_nn_train_df[qelm_feature_columns] = qelm_scaler.fit_transform(qelm_train_df[qelm_feature_columns])
    qelm_nn_test_df[qelm_feature_columns] = qelm_scaler.transform(qelm_test_df[qelm_feature_columns])

    print("\n\nQuantum Smote XGBoost:\n")
    qelm_smote_xgboost = BaseModelXGBoost(qelm_train_df, qelm_test_df)
    qelm_smote_xgboost.run()
    predictions["XGB_SMOTE_QELM"] = (qelm_smote_xgboost.y_test, qelm_smote_xgboost.y_pred)

    print("\n\nQuantum Smote Neural Network:\n")
    qelm_smote_neural_network = SmoteNeuralNetwork(qelm_nn_train_df, qelm_nn_test_df, config={"model_filename": "neural_network_qelm_smote.joblib"})
    qelm_smote_neural_network.run()
    predictions["NN_SMOTE_QELM"] = (qelm_smote_neural_network.y_test, qelm_smote_neural_network.y_pred)

    ################################### Fairness #######################################
    print("\n################################## Fairness ##################################")
    fairness_train_df = clean_train_df.copy()
    fairness_test_df = test_df.copy()
    categorical_cols = fairness_train_df.drop(columns=["loan_status"]).select_dtypes(include=["object", "category", "string"]).columns
    for col in categorical_cols:
        fairness_train_df[col] = fairness_train_df[col].astype("category")
        fairness_test_df[col] = fairness_test_df[col].astype("category")

    fairness_smote_train_df = smoted_train_df.copy()
    fairness_smote_test_df = test_df.copy()
    for col in categorical_cols:
        fairness_smote_train_df[col] = fairness_smote_train_df[col].astype("category")
        fairness_smote_test_df[col] = fairness_smote_test_df[col].astype("category")

    print("\n\nFairness Baseline XGBOOST:\n")
    fairness_xgboost_estimator = tuned_xgboost_estimator(baseline_xgboost)
    fairness_xgboost_model = FairnessEnhancedModel(fairness_train_df, fairness_test_df, fairness_xgboost_estimator,
                                                   needs_encoding=False, model_name="xgboost", max_iter=20)
    fairness_xgboost_model.run()
    predictions["XGB_Fairlearn"] = (fairness_xgboost_model.y_test, fairness_xgboost_model.y_pred)

    print("\n\nFairness SMOTE XGBOOST:\n")
    fairness_smote_xgboost_estimator = tuned_xgboost_estimator(smote_xgboost)
    fairness_smote_xgboost_model = FairnessEnhancedModel(fairness_smote_train_df, fairness_smote_test_df, fairness_smote_xgboost_estimator,
                                                         needs_encoding=False, model_name="xgboost_smote", max_iter=20)
    fairness_smote_xgboost_model.run()
    predictions["XGB_SMOTE_Fairlearn"] = (fairness_smote_xgboost_model.y_test, fairness_smote_xgboost_model.y_pred)

    print("\n\nFairness QELM + Smote XGBOOST:\n")
    fairness_qelm_xgboost_estimator = tuned_xgboost_estimator(qelm_smote_xgboost)
    fairness_qelm_xgboost_model = FairnessEnhancedModel(qelm_train_df, qelm_test_df, fairness_qelm_xgboost_estimator,
                                                        sensitive_source_train_df=smoted_train_df,
                                                        sensitive_source_test_df=test_df,
                                                        needs_encoding=False, model_name="qelm_xgboost", max_iter=20)
    fairness_qelm_xgboost_model.run()
    predictions["XGB_SMOTE_QELM_Fairlearn"] = (fairness_qelm_xgboost_model.y_test, fairness_qelm_xgboost_model.y_pred)

    print("\n\nFairness Baseline Neural Network:\n")
    fairness_nn_estimator = tuned_nn_estimator(baseline_neural_network)
    fairness_nn_model = FairnessEnhancedModel(nn_train_df, nn_test_df, fairness_nn_estimator,
                                              sensitive_source_train_df=clean_train_df,
                                              sensitive_source_test_df=test_df,
                                              needs_encoding=False, model_name="neural_network", max_iter=20)
    fairness_nn_model.run()
    predictions["NN_Fairlearn"] = (fairness_nn_model.y_test, fairness_nn_model.y_pred)

    print("\n\nFairness SMOTE Neural Network:\n")
    fairness_smote_nn_estimator = tuned_nn_estimator(smote_neural_network)
    fairness_smote_nn_model = FairnessEnhancedModel(nn_smote_train_df, nn_smote_test_df, fairness_smote_nn_estimator,
                                                    sensitive_source_train_df=smoted_train_df,
                                                    sensitive_source_test_df=test_df,
                                                    needs_encoding=False, model_name="neural_network_smote", max_iter=20)
    fairness_smote_nn_model.run()
    predictions["NN_SMOTE_Fairlearn"] = (fairness_smote_nn_model.y_test, fairness_smote_nn_model.y_pred)

    print("\n\nFairness QELM + Smote Neural Network:\n")
    fairness_qelm_nn_estimator = tuned_nn_estimator(qelm_smote_neural_network)
    fairness_qelm_nn_model = FairnessEnhancedModel(qelm_nn_train_df, qelm_nn_test_df, fairness_qelm_nn_estimator,
                                                   sensitive_source_train_df=smoted_train_df,
                                                   sensitive_source_test_df=test_df,
                                                   needs_encoding=False, model_name="qelm_neural_network", max_iter=20)
    fairness_qelm_nn_model.run()
    predictions["NN_SMOTE_QELM_Fairlearn"] = (fairness_qelm_nn_model.y_test, fairness_qelm_nn_model.y_pred)

    ################################### Spreadsheet #####################################
    write_predictions(predictions)


if __name__ == "__main__":
    main()
