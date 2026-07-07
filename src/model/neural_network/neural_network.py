from pathlib import Path
import pandas as pd
from .base_neural_network import BaseNeuralNetwork


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "pre-processed" / "loan_data_nn_normal_train.csv"
INPUT_TESTING_DATA_PATH = PROJECT_ROOT / "data" / "pre-processed" / "loan_data_nn_normal_test.csv"


class NeuralNetwork(BaseNeuralNetwork):
    def __init__(self, train_df, test_df):
        super().__init__(train_df, test_df, "neural_network.joblib")


if __name__ == "__main__":
    neural_network = NeuralNetwork(pd.read_csv(INPUT_TRAINING_DATA_PATH), pd.read_csv(INPUT_TESTING_DATA_PATH))
    neural_network.run()
