import pandas as pd
from pathlib import Path

from .preprocessing import HardOutlierFilters, DataPreprocessing, XGBoostPreprocessing
from .run_all_models import AllModels
from .ensemble_evaluation import EnsembleEvaluator, PooledFairnessEvaluator

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pre-processed"


def main():
    raw_df = pd.read_csv(RAW_DATA_DIR / "loan_data.csv")
    HardOutlierFilters(raw_df).run()

    hard_filtered_df = pd.read_csv(DATA_DIR / "loan_data_hard_filtered.csv")
    DataPreprocessing(hard_filtered_df).run()

    preprocessed_df = pd.read_csv(DATA_DIR / "loan_data_preprocessed.csv")
    XGBoostPreprocessing(preprocessed_df).run()

    normal_train_df = pd.read_csv(DATA_DIR / "loan_data_clean_train_3.csv")
    normal_test_df = pd.read_csv(DATA_DIR / "loan_data_clean_test_5.csv")
    outlier_train_df = pd.read_csv(DATA_DIR / "loan_data_outlier_train_2.csv")
    outlier_test_df = pd.read_csv(DATA_DIR / "loan_data_outlier_test_6.csv")

    print("################# Training Models on Normal Data #################")
    normal_models = AllModels(normal_train_df, normal_test_df)
    normal_predictions = normal_models.run_all()

    print("################# Training Models on Outliers #################")
    outlier_models = AllModels(outlier_train_df, outlier_test_df)
    outlier_predictions = outlier_models.run_all()

    print("\n################# Ensemble Evaluation (Normal + Outlier) #################")
    evaluator = EnsembleEvaluator(normal_predictions, outlier_predictions)
    evaluator.evaluate()

    fairness_pairs = {
        "xgboost": (normal_models.fairness_xgboost, outlier_models.fairness_xgboost),
        "xgboost_smote": (normal_models.fairness_smote_xgboost, outlier_models.fairness_xgboost),
        "qelm_xgboost": (normal_models.fairness_qelm_xgboost, outlier_models.fairness_qelm_xgboost),
        "neural_network": (normal_models.fairness_neural_network, outlier_models.fairness_neural_network),
        "neural_network_smote": (normal_models.fairness_smote_neural_network, outlier_models.fairness_neural_network),
        "qelm_neural_network": (normal_models.fairness_qelm_neural_network, outlier_models.fairness_qelm_neural_network),
    }

    for fairness_name, (fairness_normal_model, fairness_outlier_model) in fairness_pairs.items():
        if fairness_normal_model is not None and fairness_outlier_model is not None:
            PooledFairnessEvaluator(fairness_name, fairness_normal_model, fairness_outlier_model).run()


if __name__ == "__main__":
    main()
