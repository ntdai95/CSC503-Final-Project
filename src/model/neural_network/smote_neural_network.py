from pathlib import Path
import pandas as pd
from .base_neural_network import BaseNeuralNetwork


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "pre-processed" / "loan_data_nn_normal_train_smote.csv"
INPUT_TESTING_DATA_PATH = PROJECT_ROOT / "data" / "pre-processed" / "loan_data_nn_normal_test.csv"


class SmoteNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, train_df, test_df):
        super().__init__(train_df, test_df, "neural_network_smote.joblib")


if __name__ == "__main__":
    smote_neural_network = SmoteNeuralNetwork(pd.read_csv(INPUT_TRAINING_DATA_PATH), pd.read_csv(INPUT_TESTING_DATA_PATH))
    smote_neural_network.run()
