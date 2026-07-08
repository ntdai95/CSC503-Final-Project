from .quantum_extreme_learning import QELMIsing

try:
    from src.model.extreme_gradient_boosting.smote_xgboost import SmoteXGBoost
    from src.model.extreme_gradient_boosting.weighted_xgboost import WeightedXGBoost
except ModuleNotFoundError:
    from model.extreme_gradient_boosting.smote_xgboost import SmoteXGBoost
    from model.extreme_gradient_boosting.weighted_xgboost import WeightedXGBoost


def main():
    train_df, test_df = QELMIsing().run()

    smote_xgboost = SmoteXGBoost(train_df, test_df)
    smote_xgboost.run()

    weighted_xgboost = WeightedXGBoost(train_df, test_df)
    weighted_xgboost.run()

if __name__ == "__main__":
    main()
