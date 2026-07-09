from pathlib import Path
import pandas as pd
from .normalize_neural_network_clean_data import NeuralNetworkCleanDataNormalizer

try:
    from src.config import NORMALIZE_NEURAL_NETWORK_SMOTE_DATA_CONFIG
except ModuleNotFoundError:
    from config import NORMALIZE_NEURAL_NETWORK_SMOTE_DATA_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "pre-processed"


class NeuralNetworkSmoteDataNormalizer(NeuralNetworkCleanDataNormalizer):
    def __init__(self, train_df, test_df, config=None, data_config=None):
        normalization_config = {**NORMALIZE_NEURAL_NETWORK_SMOTE_DATA_CONFIG, **(config or {})}
        super().__init__(train_df, test_df, normalization_config, data_config)


if __name__ == "__main__":
    train_path = OUTPUT_DIRECTORY / NORMALIZE_NEURAL_NETWORK_SMOTE_DATA_CONFIG["train_input_filename"]
    test_path = OUTPUT_DIRECTORY / NORMALIZE_NEURAL_NETWORK_SMOTE_DATA_CONFIG["test_input_filename"]
    normalizer = NeuralNetworkSmoteDataNormalizer(pd.read_csv(train_path), pd.read_csv(test_path))
    train_df, test_df = normalizer.run()
