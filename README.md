# CSC503-Final-Project

## 📂 Dataset
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/sumit12100012/loan-approval-classification/data)

### Folder Structure:
```
├── data
│   ├── raw
│   │   ├── loan_data.csv
│   │   └── ...
│   └── pre-processed
│       ├── loan_data_preprocessed.csv
│       ├── loan_data_nn_normal_train.csv
│       ├── loan_data_nn_normal_train_smote.csv
│       └── loan_data_nn_normal_test.csv
├── notebooks
│   ├── data_preprocessing.ipynb
│   ├── data_statistics.ipynb
│   ├── 07_smote_NN.ipynb
│   ├── 08_neural_network.ipynb
│   └── ...
└── src
    ├── config.py
    ├── model
    │   ├── __init__.py
    │   └── neural_network
    │       ├── __init__.py
    │       ├── base_neural_network.py
    │       ├── neural_network.py
    │       └── smote_neural_network.py
    └── preprocessing
        ├── __init__.py
        ├── preprocessing_neural_network.py
        └── preprocessing_neural_network_smote.py
```

### Install
Python version: Python 3.13.7
```bash
pip3 install -r requirements.txt

# xgboost Python package on macOS requires the OpenMP library (libomp) to run
brew install libomp
```

### Neural Network
Run the following commands from the project root:
```bash
# Preprocessing - train/test split, then normalize, then SMOTE
python -m src.preprocessing.preprocessing_neural_network
python -m src.preprocessing.preprocessing_neural_network_smote

# Train Model - train/test split, then normalize, then SMOTE
python -m src.model.neural_network.neural_network
python -m src.model.neural_network.smote_neural_network

# Preprocessing - train/test split, then SMOTE, then normalize
# dataset these scripts ues already split and SMOTE, so these scripts only normalize and encodes
python -m src.preprocessing.normalize_neural_network_clean_data
python -m src.preprocessing.normalize_neural_network_smote_data

# Train Model - train/test split, then SMOTE, then normalize
python -m src.model.neural_network.clean_data_neural_network
python -m src.model.neural_network.smote_data_neural_network
```

The train model commands save:
- `models/neural_network.joblib`
- `models/neural_network_smote.joblib`

### Parameter Configuration
Parameters are stored in:
```text
src/config.py
```
