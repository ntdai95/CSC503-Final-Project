from pathlib import Path
import pandas as pd
from .neural_network import NeuralNetwork

from ...config import MODEL_NEURAL_NETWORK_CLEAN_DATA_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "pre-processed"


if __name__ == "__main__":
    config = MODEL_NEURAL_NETWORK_CLEAN_DATA_CONFIG
    train_df = pd.read_csv(DATA_DIRECTORY / config["train_filename"])
    test_df = pd.read_csv(DATA_DIRECTORY / config["test_filename"])
    neural_network = NeuralNetwork(train_df, test_df, config=config)
    neural_network.run()
