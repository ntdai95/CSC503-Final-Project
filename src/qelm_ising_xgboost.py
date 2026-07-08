import quantum_extreme_learning as qelm
from smote_xgboost_07 import SmoteXGBoost
from weighted_xgboost_07 import WeightedXGBoost

def main():
    train_df, test_df = qelm.QELMIsing().run()

    smote_xgboost = SmoteXGBoost(train_df, test_df)
    smote_xgboost.run()

    weighted_xgboost = WeightedXGBoost(train_df, test_df)
    weighted_xgboost.run()

if __name__ == "__main__":
    main()
