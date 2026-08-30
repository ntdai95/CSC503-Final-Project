from pathlib import Path
import pandas as pd
from .base_neural_network import BaseNeuralNetwork

from ...config import MODEL_NEURAL_NETWORK_SMOTE_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "pre-processed" / MODEL_NEURAL_NETWORK_SMOTE_CONFIG["train_filename"]
INPUT_TESTING_DATA_PATH = PROJECT_ROOT / "data" / "pre-processed" / MODEL_NEURAL_NETWORK_SMOTE_CONFIG["test_filename"]


class SmoteNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, train_df, test_df, config=None, data_config=None):
        model_config = {**MODEL_NEURAL_NETWORK_SMOTE_CONFIG, **(config or {})}
        super().__init__(train_df, test_df, model_config, data_config)


if __name__ == "__main__":
    smote_neural_network = SmoteNeuralNetwork(pd.read_csv(INPUT_TRAINING_DATA_PATH), pd.read_csv(INPUT_TESTING_DATA_PATH))
    smote_neural_network.run()
