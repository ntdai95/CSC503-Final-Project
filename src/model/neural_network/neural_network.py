from pathlib import Path
import pandas as pd
from .base_neural_network import BaseNeuralNetwork

from ...config import MODEL_NEURAL_NETWORK_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "pre-processed" / MODEL_NEURAL_NETWORK_CONFIG["train_filename"]
INPUT_TESTING_DATA_PATH = PROJECT_ROOT / "data" / "pre-processed" / MODEL_NEURAL_NETWORK_CONFIG["test_filename"]


class NeuralNetwork(BaseNeuralNetwork):
    def __init__(self, train_df, test_df, config=None, data_config=None):
        model_config = {**MODEL_NEURAL_NETWORK_CONFIG, **(config or {})}
        super().__init__(train_df, test_df, model_config, data_config)


if __name__ == "__main__":
    neural_network = NeuralNetwork(pd.read_csv(INPUT_TRAINING_DATA_PATH), pd.read_csv(INPUT_TESTING_DATA_PATH))
    neural_network.run()
