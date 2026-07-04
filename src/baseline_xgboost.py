from pathlib import Path
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


INPUT_TRAINING_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "loan_data_clean_train_3.csv"
INPUT_TESTING_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "loan_data_raw_test_1.csv"

class BaselineXGBoost:
    def __init__(self):
        self.train_df = pd.read_csv(INPUT_TRAINING_DATA_PATH)
        self.test_df = pd.read_csv(INPUT_TESTING_DATA_PATH)
        self.target_column = "loan_status"
        self.random_state = 42
        self.n_iter = 40
        self.n_splits = 5
        self.scoring = "f1"
        self.param_distributions = {
            "model__n_estimators": randint(300, 550),
            "model__max_depth": randint(6, 10),
            "model__learning_rate": uniform(0.07, 0.09),
            "model__subsample": uniform(0.78, 0.17),
            "model__colsample_bytree": uniform(0.75, 0.20),
            "model__min_child_weight": randint(3, 8),
            "model__gamma": uniform(0.05, 0.35),
            "model__reg_lambda": uniform(1.5, 2.0),
            "model__reg_alpha": uniform(0, 0.5),
        }

        self.X_train = self.train_df.drop(columns=[self.target_column])
        self.y_train = self.train_df[self.target_column]
        self.X_test = self.test_df.drop(columns=[self.target_column])
        self.y_test = self.test_df[self.target_column]
        self.categorical_cols = self.X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        self.numeric_cols = [col for col in self.X_train.columns if col not in self.categorical_cols]
        self.preprocessor = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols),
                                                            ("num", StandardScaler(), self.numeric_cols)])
        self.xgb_search = None
        self.best_xgb_model = None
        self.y_pred = None
        self.y_proba = None

    def tune(self):
        xgb_model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                  random_state=self.random_state, n_jobs=-1)
        xgb_pipeline = Pipeline(steps=[("preprocess", self.preprocessor), ("model", xgb_model)])
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.xgb_search = RandomizedSearchCV(estimator=xgb_pipeline, param_distributions=self.param_distributions,
                                             n_iter=self.n_iter, scoring=self.scoring, cv=cv, random_state=self.random_state,
                                             n_jobs=-1, verbose=1)
        self.xgb_search.fit(self.X_train, self.y_train)
        self.best_xgb_model = self.xgb_search.best_estimator_

    def evaluate(self):
        self.y_proba = self.best_xgb_model.predict_proba(self.X_test)[:, 1]
        self.y_pred = self.best_xgb_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, self.y_pred)
        by_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        print("Best CV F1 score:", self.xgb_search.best_score_)
        print("Best parameters:")
        print(self.xgb_search.best_params_)
        print("\nConfusion matrix: Row: Actual class, Column: Predicted class")
        print(cm)
        print("\nBy-class accuracy:")
        for class_label, accuracy in zip(self.best_xgb_model.classes_, by_class_accuracy):
            print(f"Class {class_label}: {accuracy:.4f}")

        print("\nClassification report:")
        print(classification_report(self.y_test, self.y_pred))

    def run(self):
        self.tune()
        self.evaluate()
        return self.best_xgb_model


if __name__ == "__main__":
    baseline_xgboost = BaselineXGBoost()
    baseline_xgboost.run()
