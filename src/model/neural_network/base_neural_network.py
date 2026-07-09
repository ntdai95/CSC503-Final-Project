from copy import deepcopy
from pathlib import Path
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from fairlearn.metrics import true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate

try:
    from src.config import DATA_COLUMN_CONFIG
except ModuleNotFoundError:
    from config import DATA_COLUMN_CONFIG


MODEL_DIRECTORY = Path(__file__).resolve().parents[3] / "models"


class BaseNeuralNetwork:
    def __init__(self, train_df, test_df, config, data_config=None):
        self.train_df = train_df
        self.test_df = test_df
        self.config = deepcopy(config)
        self.data_config = {**DATA_COLUMN_CONFIG, **(data_config or {})}

        self.target_column = self.data_config["target_column"]
        self.model_path = MODEL_DIRECTORY / self.config["model_filename"]
        self.param_distributions = self.config["search_param_distributions"]

        self.X_train = self.train_df.drop(columns=[self.target_column])
        self.y_train = self.train_df[self.target_column]
        self.X_test = self.test_df.drop(columns=[self.target_column])
        self.y_test = self.test_df[self.target_column]
        self.nn_search = None
        self.best_nn_model = None
        self.y_pred = None

    def tune(self):
        nn_model = MLPClassifier(solver=self.config["model_solver"], max_iter=self.config["model_max_iter"],
                                 early_stopping=self.config["model_early_stopping"],
                                 validation_fraction=self.config["model_validation_fraction"],
                                 n_iter_no_change=self.config["model_n_iter_no_change"],
                                 random_state=self.config["model_random_state"])
        nn_pipeline = Pipeline(steps=[("model", nn_model)])
        cv = StratifiedKFold(n_splits=self.config["cv_n_splits"], shuffle=self.config["cv_shuffle"],
                             random_state=self.config["cv_random_state"])
        self.nn_search = RandomizedSearchCV(estimator=nn_pipeline, param_distributions=self.param_distributions,
                                            n_iter=self.config["search_n_iter"], scoring=self.config["search_scoring"],
                                            cv=cv, random_state=self.config["search_random_state"],
                                            n_jobs=self.config["search_n_jobs"], verbose=self.config["search_verbose"])
        self.nn_search.fit(self.X_train, self.y_train)
        self.best_nn_model = self.nn_search.best_estimator_

    def evaluate(self):
        self.y_pred = self.best_nn_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, self.y_pred)
        by_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        print("Best CV F1 score:", self.nn_search.best_score_)
        print("Best parameters:")
        print(self.nn_search.best_params_)
        print("\nClassification report:")
        print(classification_report(self.y_test, self.y_pred))
        print("\nConfusion matrix: Row: Actual class, Column: Predicted class")
        print(cm)
        print("\nBy-class accuracy:")
        for class_label, accuracy in zip(self.best_nn_model.classes_, by_class_accuracy):
            print(f"Class {class_label}: {accuracy:.4f}")

        print("\nSummary metrics (Test):")
        print(f"Accuracy: {accuracy_score(self.y_test, self.y_pred):.4f}")
        print(f"F1-score: {f1_score(self.y_test, self.y_pred, average='binary', zero_division=0):.4f}")
        print(f"Precision: {precision_score(self.y_test, self.y_pred, zero_division=0):.4f}")
        print(f"Recall: {recall_score(self.y_test, self.y_pred, zero_division=0):.4f}")
        print(f"TPR: {true_positive_rate(self.y_test, self.y_pred):.4f}")
        print(f"TNR: {true_negative_rate(self.y_test, self.y_pred):.4f}")
        print(f"FPR: {false_positive_rate(self.y_test, self.y_pred):.4f}")
        print(f"FNR: {false_negative_rate(self.y_test, self.y_pred):.4f}")

    def get_best_model_params(self):
        return self.best_nn_model.named_steps["model"].get_params()

    def save(self):
        MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
        dump(self.best_nn_model, self.model_path)
        print("Saved model to:", self.model_path)

    def run(self):
        self.tune()
        self.evaluate()
        self.save()
        return self.best_nn_model
