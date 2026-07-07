from pathlib import Path
from joblib import dump
from scipy.stats import loguniform
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


MODEL_DIRECTORY = Path(__file__).resolve().parents[3] / "models"


class BaseNeuralNetwork:
    def __init__(self, train_df, test_df, model_filename):
        self.train_df = train_df
        self.test_df = test_df
        self.model_path = MODEL_DIRECTORY / model_filename
        self.target_column = "loan_status"
        self.random_state = 1
        self.n_iter = 20
        self.n_splits = 5
        self.scoring = "f1"
        self.n_jobs = -1
        self.verbose = 1
        self.max_iter = 200
        self.n_iter_no_change = 12
        self.param_distributions = {
            "model__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": loguniform(1e-5, 1e-2),
            "model__learning_rate_init": loguniform(1e-4, 5e-3),
            "model__batch_size": [64, 128, 256],
        }

        self.X_train = self.train_df.drop(columns=[self.target_column])
        self.y_train = self.train_df[self.target_column]
        self.X_test = self.test_df.drop(columns=[self.target_column])
        self.y_test = self.test_df[self.target_column]
        self.nn_search = None
        self.best_nn_model = None
        self.y_pred = None

    def tune(self):
        nn_model = MLPClassifier(solver="adam", max_iter=self.max_iter, early_stopping=True, validation_fraction=0.1,
                                 n_iter_no_change=self.n_iter_no_change, random_state=self.random_state)
        nn_pipeline = Pipeline(steps=[("model", nn_model)])
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.nn_search = RandomizedSearchCV(estimator=nn_pipeline, param_distributions=self.param_distributions,
                                            n_iter=self.n_iter, scoring=self.scoring, cv=cv,
                                            random_state=self.random_state, n_jobs=self.n_jobs, verbose=self.verbose)
        self.nn_search.fit(self.X_train, self.y_train)
        self.best_nn_model = self.nn_search.best_estimator_

    def evaluate(self):
        self.y_pred = self.best_nn_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, self.y_pred)
        by_class_accuracy = cm.diagonal() / cm.sum(axis=1)

        print("Best CV F1 score:", self.nn_search.best_score_)
        print("Best parameters:")
        print(self.nn_search.best_params_)
        print("\nConfusion matrix: Row: Actual class, Column: Predicted class")
        print(cm)
        print("\nBy-class accuracy:")
        for class_label, accuracy in zip(self.best_nn_model.classes_, by_class_accuracy):
            print(f"Class {class_label}: {accuracy:.4f}")

        print("\nClassification report:")
        print(classification_report(self.y_test, self.y_pred))

    def save(self):
        MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
        dump(self.best_nn_model, self.model_path)
        print("Saved model to:", self.model_path)

    def run(self):
        self.tune()
        self.evaluate()
        self.save()
        return self.best_nn_model
