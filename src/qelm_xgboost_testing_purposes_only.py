from pathlib import Path
import types
import pandas as pd
from model.extreme_gradient_boosting import BaseModelXGBoost, WeightedXGBoost


###################################################### QUANTUM PART ###############################################################

INPUT_TRAINING_QUANTUM_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "df_quantum_train.csv"
INPUT_TESTING_QUANTUM_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "df_quantum_test.csv"

print("WITH QUANTUM DF:\n")
print("BASELINE XGBOOST:")
baseline_xgboost = BaseModelXGBoost(pd.read_csv(INPUT_TRAINING_QUANTUM_DATA_PATH), pd.read_csv(INPUT_TESTING_QUANTUM_DATA_PATH))

###### EXAMPLES ON HOW TO MODIFY MODEL SETTINGS FOR BASELINE XGBOOST (ALL SETTINGS ARE IN THE basemodel_xgboost.py file) #####

# UNCOMMENT THE CORRESPONDING PART BELOW (1, 2, or 3)
# For other models, add the uncommented part between instance creation such as 
# smote_xgboost = BaseModelXGBoost(pd.read_csv(INPUT_TRAINING_QUANTUM_DATA_PATH), pd.read_csv(INPUT_TESTING_QUANTUM_DATA_PATH)) and
# run() method calling such as smote_xgboost.run()

# (1) If any of the basemodel parameters (found in the __init__() method of the BaseModelXGBoost class) need to be changed for 
# example number of iterations from original 40 to 20:
# baseline_xgboost.n_iter = 20

# (2) If basemodel tune method need to be changed:
# def new_tune(self):
#     xgb_model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
#                               random_state=self.random_state, n_jobs=-1)
#     pipeline = Pipeline(steps=[("preprocess", self.preprocessor), ("model", xgb_model)])
#     pipeline.fit(self.X_train, self.y_train)

# baseline_xgboost.tune = types.MethodType(new_tune, baseline_xgboost)

# (3) If basemodel evaluate method need to be changed:
# def new_evaluate(self):
#     print("This is a new evaluate method that just simply print this statement.")

# baseline_xgboost.evaluate = types.MethodType(new_evaluate, baseline_xgboost)

baseline_xgboost.run()
print("SMOTE XGBOOST:")
smote_xgboost = BaseModelXGBoost(pd.read_csv(INPUT_TRAINING_QUANTUM_DATA_PATH), pd.read_csv(INPUT_TESTING_QUANTUM_DATA_PATH))
# ADD THE MODIFICATIONS HERE
smote_xgboost.run()
print("WEIGHTED XGBOOST:")
weighted_xgboost = WeightedXGBoost(pd.read_csv(INPUT_TRAINING_QUANTUM_DATA_PATH), pd.read_csv(INPUT_TESTING_QUANTUM_DATA_PATH))
# ADD THE MODIFICATIONS HERE
weighted_xgboost.run()


###################################################### PCA PART ###############################################################

INPUT_TRAINING_PCA_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "df_pca_train.csv"
INPUT_TESTING_PCA_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pre-processed" / "df_pca_test.csv"

# There is currently no label (target column) in the pca dfs
print("\nWITH PCA DF:\n")
print("BASELINE XGBOOST:")
baseline_xgboost = BaseModelXGBoost(pd.read_csv(INPUT_TRAINING_PCA_DATA_PATH), pd.read_csv(INPUT_TESTING_PCA_DATA_PATH))
# ADD THE MODIFICATIONS HERE AFTER FIXING MISSING LABEL
baseline_xgboost.run()
print("SMOTE XGBOOST:")
smote_xgboost = BaseModelXGBoost(pd.read_csv(INPUT_TRAINING_PCA_DATA_PATH), pd.read_csv(INPUT_TESTING_PCA_DATA_PATH))
# ADD THE MODIFICATIONS HERE AFTER FIXING MISSING LABEL
smote_xgboost.run()
print("WEIGHTED XGBOOST:")
weighted_xgboost = WeightedXGBoost(pd.read_csv(INPUT_TRAINING_PCA_DATA_PATH), pd.read_csv(INPUT_TESTING_PCA_DATA_PATH))
# ADD THE MODIFICATIONS HERE AFTER FIXING MISSING LABEL
weighted_xgboost.run()