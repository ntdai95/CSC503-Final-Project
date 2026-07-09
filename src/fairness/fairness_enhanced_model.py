from pathlib import Path
from fairlearn.reductions import ErrorRate, ExponentiatedGradient
from fairlearn.metrics import MetricFrame, false_negative_rate, false_positive_rate, true_negative_rate, true_positive_rate
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .multi_feature_parity import MultiFeatureParity
except ImportError:
    from multi_feature_parity import MultiFeatureParity


OUTPUT_DIRECTORY = Path(__file__).resolve().parents[2] / "outputs"


class FairnessEnhancedModel:
    def __init__(self, train_df, test_df, estimator, sensitive_dict=None,
                 sensitive_source_train_df=None, sensitive_source_test_df=None,
                 needs_encoding=False, model_name="model", max_iter=50, show_plot=False):

        self.train_df = train_df
        self.test_df = test_df
        self.estimator = estimator
        self.needs_encoding = needs_encoding
        self.model_name = model_name
        self.max_iter = max_iter
        self.show_plot = show_plot
        self.sensitive_dict = sensitive_dict or {'person_age': {'bins': 5, 'fpr': 0.05, 'fnr': 0.05},
                                                 'person_income': {'bins': 6, 'fpr': 0.05, 'fnr': 0.05}}
        self.target = 'loan_status'
        self.y_train = train_df[self.target]
        self.X_train = train_df.drop(columns=[self.target])
        self.y_test = test_df[self.target]
        self.X_test = test_df.drop(columns=[self.target])
        if sensitive_source_train_df is not None:
            self.sensitive_source_train_df = self._strip_target(sensitive_source_train_df)
            self.sensitive_source_test_df = self._strip_target(sensitive_source_test_df)
        else:
            self.sensitive_source_train_df = self.X_train
            self.sensitive_source_test_df = self.X_test

        self.eval_bins = {
            'person_age':                 5,
            'person_income':              6,
            'person_emp_exp':             5,
            'cb_person_cred_hist_length': 4,
            'credit_score':               5,
            'loan_amnt':                  4,
            'loan_int_rate':              4,
            'loan_percent_income':        4,
        }

        self.objective = ErrorRate(costs={"fp": 0.7, "fn": 0.3})
        self.metrics_dict = {
            'accuracy': accuracy_score,
            'f1': partial(f1_score, average='binary', zero_division=0),
            'precision': partial(precision_score, zero_division=0),
            'recall': partial(recall_score, zero_division=0),
            'tpr': true_positive_rate,
            'tnr': true_negative_rate,
            'fpr': false_positive_rate,
            'fnr': false_negative_rate,
        }

        self.sensitive_train_df = pd.DataFrame(index=self.sensitive_source_train_df.index)
        self.eval_num_test_df = pd.DataFrame(index=self.sensitive_source_test_df.index)
        self.final_summary_df = pd.DataFrame()
        self.encoder = None
        self.X_train_model = None
        self.X_test_model = None
        self.mitigator = None
        self.y_pred = None

    def _strip_target(self, df):
        return df.drop(columns=[self.target]) if self.target in df.columns else df

    def _create_quantile_bins(self, source_df, column, q):
        _, edges = pd.qcut(source_df[column], q=q, duplicates='drop', retbins=True)
        num_bins = len(edges) - 1
        labels = [f'Q{i+1}' for i in range(num_bins)]

        edges = edges.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf
        binned = pd.cut(source_df[column], bins=edges, labels=labels, include_lowest=True)
        return binned, edges, labels

    def _bin_features(self, features_dict, train_source, test_source):
        binned_train_df = pd.DataFrame(index=train_source.index)
        binned_test_df = pd.DataFrame(index=test_source.index)
        for col, q in features_dict.items():
            binned, edges, labels = self._create_quantile_bins(train_source, col, q)
            binned_train_df[col] = binned
            binned_test_df[col] = pd.cut(test_source[col], bins=edges, labels=labels, include_lowest=True)

        return binned_train_df, binned_test_df

    def _encode_features(self):
        categorical_columns = self.X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        numeric_columns = [col for col in self.X_train.columns if col not in categorical_columns]
        self.encoder = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
                ("num", StandardScaler(), numeric_columns),
            ],
        )

        X_train_model = self.encoder.fit_transform(self.X_train)
        X_test_model = self.encoder.transform(self.X_test)
        return X_train_model, X_test_model

    def _build_feature_config(self):
        feature_config = {}
        for feature, spec in self.sensitive_dict.items():
            feature_config[feature] = {}
            for metric in ("fpr", "fnr"):
                if metric in spec:
                    feature_config[feature][metric] = spec[metric]

        return feature_config

    def fit(self):
        if self.needs_encoding:
            self.X_train_model, self.X_test_model = self._encode_features()
        else:
            self.X_train_model, self.X_test_model = self.X_train, self.X_test

        constraints = MultiFeatureParity(self._build_feature_config())
        self.mitigator = ExponentiatedGradient(
            estimator=self.estimator,
            constraints=constraints,
            objective=self.objective,
            max_iter=self.max_iter,
        )
        self.mitigator.fit(self.X_train_model, self.y_train, sensitive_features=self.sensitive_train_df)
        self.y_pred = self.mitigator.predict(self.X_test_model)

    def evaluate(self):
        print("Summary metrics (Test):")
        print(f"Accuracy: {self.metrics_dict['accuracy'](self.y_test, self.y_pred):.4f}")
        print(f"F1-score: {self.metrics_dict['f1'](self.y_test, self.y_pred):.4f}")
        print(f"Precision: {self.metrics_dict['precision'](self.y_test, self.y_pred):.4f}")
        print(f"Recall: {self.metrics_dict['recall'](self.y_test, self.y_pred):.4f}")
        print(f"TPR: {self.metrics_dict['tpr'](self.y_test, self.y_pred):.4f}")
        print(f"TNR: {self.metrics_dict['tnr'](self.y_test, self.y_pred):.4f}")
        print(f"FPR: {self.metrics_dict['fpr'](self.y_test, self.y_pred):.4f}")
        print(f"FNR: {self.metrics_dict['fnr'](self.y_test, self.y_pred):.4f}")
        num_features = list(self.eval_bins.keys())
        eval_df = pd.concat([self.sensitive_source_test_df.drop(columns=num_features), self.eval_num_test_df], axis=1)
        summary_list = []
        for col in eval_df.columns:
            mf = MetricFrame(metrics=self.metrics_dict,
                            y_true=self.y_test,
                            y_pred=self.y_pred,
                            sensitive_features=eval_df[col])
            feature_range = mf.by_group.apply(lambda x: x.max() - x.min())
            feature_range.name = col
            summary_list.append(feature_range)

        self.final_summary_df = pd.DataFrame(summary_list)

    def _generate_heatmap(self):
        plt.figure(figsize=(8, 4))
        sns.heatmap(self.final_summary_df, annot=True, cmap="Blues", fmt=".3f")
        plt.title("Metric Disparity per Feature (Max - Min Metric)")
        plt.tight_layout()
        if self.show_plot:
            plt.show()
        else:
            OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIRECTORY / f"fairness_metric_disparity_heatmap_{self.model_name}.png"
            plt.savefig(output_path)
            print("Saved fairness heatmap to:", output_path)

        plt.close()

    def run(self):
        bin_spec = {feature: spec['bins'] for feature, spec in self.sensitive_dict.items()}
        self.sensitive_train_df, _ = self._bin_features(bin_spec, self.sensitive_source_train_df,
                                                         self.sensitive_source_test_df)
        self.fit()
        _, self.eval_num_test_df = self._bin_features(self.eval_bins, self.sensitive_source_train_df, self.sensitive_source_test_df)
        self.evaluate()
        self._generate_heatmap()
