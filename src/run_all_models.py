from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from model.extreme_gradient_boosting import BaseModelXGBoost, WeightedXGBoost
from model.neural_network import NeuralNetwork, SmoteNeuralNetwork
from model.quantum import QELMIsing
from fairness import FairnessEnhancedModel, SampleWeightedMLPClassifier
from config import NORMALIZE_NEURAL_NETWORK_CLEAN_DATA_CONFIG, NORMALIZE_NEURAL_NETWORK_SMOTE_DATA_CONFIG
from config import MODEL_NEURAL_NETWORK_CLEAN_DATA_CONFIG, MODEL_NEURAL_NETWORK_SMOTE_DATA_CONFIG
from preprocessing import NeuralNetworkCleanDataNormalizer, NeuralNetworkSmoteDataNormalizer
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pre-processed"

class AllModels():
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.smoted_train_df = pd.read_csv(DATA_DIR / "loan_data_smoted_raw_train_4.csv")
        if len(self.train_df) > 3000:
            self._is_normal = True
        else:
            self._is_normal = False

        self.baseline_xgboost = None
        self.smote_xgboost = None
        self.weighted_xgboost = None
        self.qelm_xgboost = None
        self.qelm_neural_network = None
        self.qelm_train_df = pd.DataFrame()
        self.qelm_test_df = pd.DataFrame()
        self.qelm_nn_train_df = pd.DataFrame()
        self.qelm_nn_test_df = pd.DataFrame()

        self.baseline_neural_network = None
        self.nn_train_df = pd.DataFrame()
        self.nn_test_df = pd.DataFrame()
        self.smote_neural_network = None
        self.nn_smote_train_df = pd.DataFrame()
        self.nn_smote_test_df = pd.DataFrame()

        self.fairness_xgboost = None
        self.fairness_smote_xgboost = None
        self.fairness_qelm_xgboost = None
        self.fairness_neural_network = None
        self.fairness_smote_neural_network = None
        self.fairness_qelm_neural_network = None


    def run_XGB(self):
        print("\n###################################### XGBoost ####################################")
        print("\n\nBaseline XGBOOST:\n")
        self.baseline_xgboost = BaseModelXGBoost(self.train_df, self.test_df)
        self.baseline_xgboost.run()

        if self._is_normal == True:
            print("\n\nSmote XGBOOST:\n")
            self.smote_xgboost = BaseModelXGBoost(self.smoted_train_df, self.test_df)
            self.smote_xgboost.run()

        print("\n\nWeighted XGBOOST:\n")
        self.weighted_xgboost = WeightedXGBoost(self.train_df, self.test_df)
        self.weighted_xgboost.run()

    def run_NN(self):
        print("\n################################## Neural Network ##################################")
        NeuralNetworkCleanDataNormalizer(self.train_df, self.test_df).run()

        self.nn_train_df = pd.read_csv(DATA_DIR / NORMALIZE_NEURAL_NETWORK_CLEAN_DATA_CONFIG["train_output_filename"])
        self.nn_test_df = pd.read_csv(DATA_DIR / NORMALIZE_NEURAL_NETWORK_CLEAN_DATA_CONFIG["test_output_filename"])
        
        print("\n\nBaseline Neural Network:\n")
        self.baseline_neural_network = NeuralNetwork(self.nn_train_df, self.nn_test_df, config=MODEL_NEURAL_NETWORK_CLEAN_DATA_CONFIG)
        self.baseline_neural_network.run()

        if self._is_normal == True:
            NeuralNetworkSmoteDataNormalizer(self.smoted_train_df, self.test_df).run()
            self.nn_smote_train_df = pd.read_csv(DATA_DIR / NORMALIZE_NEURAL_NETWORK_SMOTE_DATA_CONFIG["train_output_filename"])
            self.nn_smote_test_df = pd.read_csv(DATA_DIR / NORMALIZE_NEURAL_NETWORK_SMOTE_DATA_CONFIG["test_output_filename"])
            print("\n\nSmote Neural Network:\n")
            self.smote_neural_network = SmoteNeuralNetwork(self.nn_smote_train_df, self.nn_smote_test_df, config=MODEL_NEURAL_NETWORK_SMOTE_DATA_CONFIG)
            self.smote_neural_network.run()

    def run_qelm(self):
        print("\n################################## Quantum ##################################")
        self.qelm_train_df, self.qelm_test_df = QELMIsing(self.train_df, self.test_df, self._is_normal).run()
        qelm_feature_columns = []
        for column in self.qelm_train_df.columns:
            if column != "loan_status":
                qelm_feature_columns.append(column)

        qelm_scaler = StandardScaler()
        self.qelm_nn_train_df = self.qelm_train_df.copy()
        self.qelm_nn_test_df = self.qelm_test_df.copy()
        self.qelm_nn_train_df[qelm_feature_columns] = qelm_scaler.fit_transform(self.qelm_train_df[qelm_feature_columns])
        self.qelm_nn_test_df[qelm_feature_columns] = qelm_scaler.transform(self.qelm_test_df[qelm_feature_columns])


        print("\n\nQuantum XGBoost:\n")
        self.qelm_xgboost = BaseModelXGBoost(self.qelm_train_df, self.qelm_test_df)
        self.qelm_xgboost.run()


        print("\n\nQuantum Neural Network:\n")
        self.qelm_neural_network = NeuralNetwork(self.qelm_nn_train_df, self.qelm_nn_test_df, config={"model_filename": "neural_network_qelm.joblib"})
        self.qelm_neural_network.run()

    def run_fairness(self):
        print("\n################################## Fairness ##################################")
        fairness_train_df = self.train_df.copy()
        fairness_test_df = self.test_df.copy()
        categorical_cols = fairness_train_df.drop(columns=["loan_status"]).select_dtypes(include=["object", "category", "string"]).columns
        for col in categorical_cols:
            fairness_train_df[col] = fairness_train_df[col].astype("category")
            fairness_test_df[col] = fairness_test_df[col].astype("category")


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

        print("\n\nFairness Baseline XGBOOST:\n")
        fairness_xgboost_estimator = tuned_xgboost_estimator(self.baseline_xgboost)
        self.fairness_xgboost = FairnessEnhancedModel(fairness_train_df, fairness_test_df, fairness_xgboost_estimator,
                                                    needs_encoding=False, model_name="xgboost", max_iter=20, _is_normal=self._is_normal)
        self.fairness_xgboost.run()

        if self._is_normal == True:
            fairness_smote_train_df = self.smoted_train_df.copy()
            fairness_smote_test_df = self.test_df.copy()
            for col in categorical_cols:
                fairness_smote_train_df[col] = fairness_smote_train_df[col].astype("category")
                fairness_smote_test_df[col] = fairness_smote_test_df[col].astype("category")
            print("\n\nFairness SMOTE XGBOOST:\n")
            fairness_smote_xgboost_estimator = tuned_xgboost_estimator(self.smote_xgboost)
            self.fairness_smote_xgboost = FairnessEnhancedModel(fairness_smote_train_df, fairness_smote_test_df, fairness_smote_xgboost_estimator,
                                                                needs_encoding=False, model_name="xgboost_smote", max_iter=20, _is_normal=self._is_normal)
            self.fairness_smote_xgboost.run()


        print("\n\nFairness QELM + XGBOOST:\n")
        fairness_qelm_xgboost_estimator = tuned_xgboost_estimator(self.qelm_xgboost)
        self.fairness_qelm_xgboost = FairnessEnhancedModel(self.qelm_train_df, self.qelm_test_df, fairness_qelm_xgboost_estimator,
                                                            sensitive_source_train_df=self.train_df,
                                                            sensitive_source_test_df=self.test_df,
                                                            needs_encoding=False, model_name="qelm_xgboost", max_iter=20, _is_normal=self._is_normal)
        self.fairness_qelm_xgboost.run()


        print("\n\nFairness Baseline Neural Network:\n")
        fairness_nn_estimator = tuned_nn_estimator(self.baseline_neural_network)
        self.fairness_neural_network = FairnessEnhancedModel(self.nn_train_df, self.nn_test_df, fairness_nn_estimator,
                                                sensitive_source_train_df=self.train_df,
                                                sensitive_source_test_df=self.test_df,
                                                needs_encoding=False, model_name="neural_network", max_iter=20,
                                                _is_normal=self._is_normal)
        self.fairness_neural_network.run()


        print("\n\nFairness SMOTE Neural Network:\n")
        if self._is_normal == True:
            fairness_smote_nn_estimator = tuned_nn_estimator(self.smote_neural_network)
            self.fairness_smote_neural_network = FairnessEnhancedModel(self.nn_smote_train_df, self.nn_smote_test_df, fairness_smote_nn_estimator,
                                                            sensitive_source_train_df=self.smoted_train_df,
                                                            sensitive_source_test_df=self.test_df,
                                                            needs_encoding=False, model_name="neural_network_smote", max_iter=20,
                                                            _is_normal=self._is_normal)
            self.fairness_smote_neural_network.run()


        print("\n\nFairness QELM + Neural Network:\n")
        fairness_qelm_nn_estimator = tuned_nn_estimator(self.qelm_neural_network)
        self.fairness_qelm_neural_network = FairnessEnhancedModel(self.qelm_nn_train_df, self.qelm_nn_test_df, fairness_qelm_nn_estimator,
                                                    sensitive_source_train_df=self.train_df,
                                                    sensitive_source_test_df=self.test_df,
                                                    needs_encoding=False, model_name="qelm_neural_network", max_iter=20,
                                                    _is_normal=self._is_normal)
        self.fairness_qelm_neural_network.run()

    def get_predictions(self):
        model_map = {
            "XGB_No_SMOTE": self.baseline_xgboost,
            "XGB_SMOTE": self.smote_xgboost,
            "XGB_Weighted": self.weighted_xgboost,
            "XGB_QELM": self.qelm_xgboost,
            "XGB_Fairlearn": self.fairness_xgboost,
            "XGB_SMOTE_Fairlearn": self.fairness_smote_xgboost,
            "XGB_QELM_Fairlearn": self.fairness_qelm_xgboost,
            "NN_No_SMOTE": self.baseline_neural_network,
            "NN_SMOTE": self.smote_neural_network,
            "NN_QELM": self.qelm_neural_network,
            "NN_Fairlearn": self.fairness_neural_network,
            "NN_SMOTE_Fairlearn": self.fairness_smote_neural_network,
            "NN_QELM_Fairlearn": self.fairness_qelm_neural_network,
        }
        predictions = {}
        for name, model in model_map.items():
            if model is not None and model.y_pred is not None:
                predictions[name] = (model.y_test, model.y_pred)

        return predictions

    def run_all(self):
        self.run_XGB()
        self.run_NN()
        self.run_qelm()
        self.run_fairness()
        return self.get_predictions()
