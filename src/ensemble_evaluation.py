from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fairlearn.metrics import (MetricFrame, false_negative_rate, false_positive_rate,
                               true_negative_rate, true_positive_rate)
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
VISUALIZATION_DIRECTORY = PROJECT_ROOT / "data" / "visualizations"

# Model types that only exist on the normal partition -> outlier-partition stand-in.
SMOTE_FALLBACKS = {
    "XGB_SMOTE": "XGB_No_SMOTE",
    "NN_SMOTE": "NN_No_SMOTE",
    "XGB_SMOTE_Fairlearn": "XGB_Fairlearn",
    "NN_SMOTE_Fairlearn": "NN_Fairlearn",
}

METRICS = {
    "accuracy": accuracy_score,
    "f1": partial(f1_score, average="binary", zero_division=0),
    "precision": partial(precision_score, zero_division=0),
    "recall": partial(recall_score, zero_division=0),
    "tpr": true_positive_rate,
    "tnr": true_negative_rate,
    "fpr": false_positive_rate,
    "fnr": false_negative_rate,
}


def _as_series(values):
    return pd.Series(np.asarray(values).ravel()).reset_index(drop=True)


def pool_predictions(normal_pair, outlier_pair):
    y_true = pd.concat([_as_series(normal_pair[0]), _as_series(outlier_pair[0])], ignore_index=True)
    y_pred = pd.concat([_as_series(normal_pair[1]), _as_series(outlier_pair[1])], ignore_index=True)
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    return {name: float(metric(y_true, y_pred)) for name, metric in METRICS.items()}


class EnsembleEvaluator:
    def __init__(self, normal_predictions, outlier_predictions, save_outputs=True):
        self.normal_predictions = normal_predictions
        self.outlier_predictions = outlier_predictions
        self.save_outputs = save_outputs
        self.pooled_predictions = {}
        self.summary_df = pd.DataFrame()

    def _outlier_pair_for(self, name):
        if name in self.outlier_predictions:
            return self.outlier_predictions[name], ""

        fallback = SMOTE_FALLBACKS.get(name)
        if fallback is not None and fallback in self.outlier_predictions:
            return self.outlier_predictions[fallback], f"outliers scored by {fallback}"

        return None, "no outlier counterpart - normal partition only"

    def evaluate(self):
        rows = []
        for name, normal_pair in self.normal_predictions.items():
            outlier_pair, note = self._outlier_pair_for(name)
            if outlier_pair is None:
                y_true, y_pred = pool_predictions(normal_pair, (pd.Series(dtype=float), pd.Series(dtype=float)))
            else:
                y_true, y_pred = pool_predictions(normal_pair, outlier_pair)

            self.pooled_predictions[name] = (y_true, y_pred)

            print(f"\n############### Ensemble (all data): {name} ###############")
            if note:
                print(f"[note] {note}")
            print(f"n = {len(y_true)} (normal: {len(_as_series(normal_pair[0]))}, "
                  f"outlier: {len(y_true) - len(_as_series(normal_pair[0]))})")
            print("\nClassification report:")
            print(classification_report(y_true, y_pred, zero_division=0))
            print("Confusion matrix: Row: Actual class, Column: Predicted class")
            print(confusion_matrix(y_true, y_pred))

            metrics = compute_metrics(y_true, y_pred)
            print("\nSummary metrics (pooled test):")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

            rows.append({"model": name, "note": note, "n_samples": len(y_true), **metrics})

        self.summary_df = pd.DataFrame(rows).set_index("model")
        print("\n############### Ensemble summary (all models) ###############")
        print(self.summary_df.round(4).to_string())

        if self.save_outputs:
            self._save()

        return self.summary_df

    def _save(self):
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        summary_path = OUTPUT_DIRECTORY / "ensemble_metrics_summary.csv"
        self.summary_df.to_csv(summary_path)
        print("\nSaved ensemble summary to:", summary_path)

        VISUALIZATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
        for name, (y_true, y_pred) in self.pooled_predictions.items():
            path = VISUALIZATION_DIRECTORY / f"{name}_ensemble_3070_y_test_y_pred.csv"
            pd.DataFrame({"y_test": y_true, "y_pred": y_pred}).to_csv(path, index=False)
            print("Saved pooled predictions to:", path)


class PooledFairnessEvaluator:
    def __init__(self, model_name, normal_model, outlier_model, show_plot=False):
        self.model_name = model_name
        self.normal_model = normal_model
        self.outlier_model = outlier_model
        self.show_plot = show_plot
        self.eval_bins = dict(normal_model.eval_bins)
        self.metrics_dict = dict(normal_model.metrics_dict)
        self.final_summary_df = pd.DataFrame()

    @staticmethod
    def _quantile_bin(train_values, test_values, q):
        _, edges = pd.qcut(train_values, q=q, duplicates="drop", retbins=True)
        labels = [f"Q{i + 1}" for i in range(len(edges) - 1)]
        edges = edges.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf
        return pd.cut(test_values, bins=edges, labels=labels, include_lowest=True)

    def run(self):
        y_true = pd.concat([_as_series(self.normal_model.y_test),
                            _as_series(self.outlier_model.y_test)], ignore_index=True)
        y_pred = pd.concat([_as_series(self.normal_model.y_pred),
                            _as_series(self.outlier_model.y_pred)], ignore_index=True)

        pooled_train = pd.concat([self.normal_model.sensitive_source_train_df,
                                  self.outlier_model.sensitive_source_train_df], ignore_index=True)
        pooled_test = pd.concat([self.normal_model.sensitive_source_test_df,
                                 self.outlier_model.sensitive_source_test_df], ignore_index=True)

        eval_df = pooled_test.drop(columns=list(self.eval_bins.keys()))
        for col, q in self.eval_bins.items():
            eval_df[col] = self._quantile_bin(pooled_train[col], pooled_test[col], q)

        print(f"\n############### Pooled fairness (all data): {self.model_name} ###############")
        print("Summary metrics (pooled test):")
        for metric_name, metric in self.metrics_dict.items():
            print(f"{metric_name}: {metric(y_true, y_pred):.4f}")

        summary_list = []
        for col in eval_df.columns:
            mf = MetricFrame(metrics=self.metrics_dict, y_true=y_true, y_pred=y_pred,
                             sensitive_features=eval_df[col])
            feature_range = mf.by_group.apply(lambda x: x.max() - x.min())
            feature_range.name = col
            summary_list.append(feature_range)

        self.final_summary_df = pd.DataFrame(summary_list)
        self._generate_heatmap()
        return self.final_summary_df

    def _generate_heatmap(self):
        plt.figure(figsize=(8, 4))
        sns.heatmap(self.final_summary_df, annot=True, cmap="Blues", fmt=".3f")
        plt.title(f"Metric Disparity per Feature (Max - Min Metric) - Ensemble: {self.model_name}")
        plt.tight_layout()
        if self.show_plot:
            plt.show()
        else:
            OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIRECTORY / f"fairness_metric_disparity_heatmap_{self.model_name}_ensemble.png"
            plt.savefig(output_path)
            print("Saved pooled fairness heatmap to:", output_path)

        plt.close()
