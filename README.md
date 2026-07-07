# CSC503-Final-Project

## 📂 Dataset
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/sumit12100012/loan-approval-classification/data)

### Folder Structure:
```
├── data
│   ├── raw
│   │   └── loan_data.csv
│   │   └── ...
│   ├── pre_processed
│   │   └── loan_data_preprocessed_nn.csv
│   │   └── loan_data_preprocessed_xgb.csv
├── notebooks
│   ├── data_preprocessing.ipynb
│   ├── data_statistics.ipynb
├── src
│   ├── model/
│   │   ├── __init__.py
│   │   ├── neural_network/
│   │       ├── __init__.py
│   │       ├── base_neural_network.py
│   │       ├── neural_network.py
│   │       ├── smote_neural_network.py
│   ├── preprocessing/
│       ├── __init__.py
│       ├── preprocessing_neural_network.py
│       ├── preprocessing_neural_network_smote.py
```

### Install
Python version: Python 3.13.7

```bash
pip3 install -r requirements.txt
```

### Neural Network

run the following command in the root directory
```bash
### Preprocessing
python -m src.preprocessing.preprocessing_neural_network
python -m src.preprocessing.preprocessing_neural_network_smote # generate SMOTE dataset
### Model
python -m src.model.neural_network.neural_network
python -m src.model.neural_network.smote_neural_network # train model with SMOTE dataset
```

The model commands save:
- `models/neural_network.joblib`
- `models/neural_network_smote.joblib`
