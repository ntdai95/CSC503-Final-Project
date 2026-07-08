from fairlearn.reductions import ErrorRate, EqualizedOdds, ExponentiatedGradient
from fairlearn.metrics import MetricFrame, false_negative_rate, false_positive_rate, true_negative_rate, true_positive_rate
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

class FairnessEnhancedModel:
    def __init__(self, train_df, test_df, estimator, 
                 sensitive_dict={}):
        
        self.train_df = train_df
        self.test_df = test_df
        self.estimator = estimator
        self.sensitive_dict = sensitive_dict or {
            'num__person_age': 5,
            'num__person_income': 6
        }
        
        self.target = 'loan_status'
        self.y_train = train_df[self.target]
        self.X_train = train_df.drop(columns=[self.target])
        self.y_test = test_df[self.target]
        self.X_test = test_df.drop(columns=[self.target])

        self.eval_bins = {
            'num__person_age':                 5,
            'num__person_income':              6,
            'num__person_emp_exp':             5,
            'num__cb_person_cred_hist_length': 4,
            'num__credit_score':               5,
            'num__loan_amnt':                  4,
            'num__loan_int_rate':              4,
            'num__loan_percent_income':        4,
        }

        self.difference_bound = 0.05
        self.constraints = EqualizedOdds(difference_bound=self.difference_bound)
        self.objective = ErrorRate(costs={"fp":0.7, "fn":0.3})

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

        self.binned_col_names = []
        self.bin_edges = {}
        self.sensitive_train_df = pd.DataFrame(index=self.X_train.index)
        self.sensitive_test_df = pd.DataFrame(index=self.X_test.index)
        self.eval_num_test_df = pd.DataFrame(index=self.X_test.index)
        self.final_summary_df = pd.DataFrame()


    def _create_quantile_bins(self, column, q):
        _, edges = pd.qcut(self.X_train[column], q=q, duplicates='drop', retbins=True)
        num_bins = len(edges) - 1
        labels = [f'Q{i+1}' for i in range(num_bins)]

        edges = edges.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf

        binned = pd.cut(self.X_train[column], bins=edges, labels=labels, include_lowest=True)
        return binned, edges, labels

    def _bin_features(self, features_dict):
        binned_train_df = pd.DataFrame(index=self.X_train.index)
        binned_test_df = pd.DataFrame(index=self.X_test.index)
        for col, q in features_dict.items():
            col_name = f"{col}_bin"
            binned, edges, labels = self._create_quantile_bins(col, q)
            binned_train_df[col_name] = binned
            binned_test_df[col_name] = pd.cut(self.X_test[col], bins=edges, 
                                              labels=labels, include_lowest=True)
            self.bin_edges[col_name] = (edges, labels)
        return binned_train_df, binned_test_df

    def fit(self):
        mitigator = ExponentiatedGradient(
            estimator=self.estimator,
            constraints=self.constraints,
            objective=self.objective
        )

        mitigator.fit(self.X_train, self.y_train, sensitive_features=self.sensitive_train_df)

        self.y_pred = mitigator.predict(self.X_test)

    def evaluate(self):
        num_features = list(self.eval_bins.keys())
        eval_df = pd.concat([self.X_test, self.eval_num_test_df], axis=1)
        eval_df.drop(columns=num_features)

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
        plt.tight_layout()
        plt.title("Metric Disparity per Feature (Max - Min Metric)")
        plt.show()

    def run(self):
        self.sensitive_train_df, self.sensitive_test_df = self._bin_features(self.sensitive_dict)
        self.fit()
        _, self.eval_num_test_df = self._bin_features(self.eval_bins)
        self.evaluate()
        self._generate_heatmap()
