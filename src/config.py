from scipy.stats import loguniform


DATA_COLUMN_CONFIG = {
    "target_column": "loan_status",
    "education_column": "person_education",
    "education_min": 1,
    "education_max": 5,
}

PREPROCESSING_NEURAL_NETWORK_CONFIG = {
    "input_filename": "loan_data_preprocessed.csv",
    "train_output_filename": "loan_data_nn_normal_train.csv",
    "test_output_filename": "loan_data_nn_normal_test.csv",
    "split_test_size": 0.2,
    "split_random_state": 1,
}

PREPROCESSING_NEURAL_NETWORK_SMOTE_CONFIG = {
    "smote_output_filename": "loan_data_nn_normal_train_smote.csv",
    "smote_sampling_strategy": "auto",
    "smote_k_neighbors": 5,
    "smote_random_state": 1,
}

MODEL_NEURAL_NETWORK_SEARCH_SPACE = {
    "model__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
    "model__activation": ["relu", "tanh"],
    "model__alpha": loguniform(1e-5, 1e-2),
    "model__learning_rate_init": loguniform(1e-4, 5e-3),
    "model__batch_size": [64, 128, 256],
}

MODEL_NEURAL_NETWORK_CONFIG = {
    "train_filename": "loan_data_nn_normal_train.csv",
    "test_filename": "loan_data_nn_normal_test.csv",
    "model_filename": "neural_network.joblib",
    "model_solver": "adam",
    "model_max_iter": 200,
    "model_early_stopping": True,
    "model_validation_fraction": 0.1,
    "model_n_iter_no_change": 12,
    "model_random_state": 1,
    "cv_n_splits": 5,
    "cv_shuffle": True,
    "cv_random_state": 1,
    "search_n_iter": 20,
    "search_scoring": "f1",
    "search_random_state": 1,
    "search_n_jobs": -1,
    "search_verbose": 1,
    "search_param_distributions": MODEL_NEURAL_NETWORK_SEARCH_SPACE,
}

# MODEL_NEURAL_NETWORK_SMOTE_SEARCH_SPACE = {
#     "model__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
#     "model__activation": ["relu", "tanh"],
#     "model__alpha": loguniform(1e-5, 1e-2),
#     "model__learning_rate_init": loguniform(1e-4, 5e-3),
#     "model__batch_size": [64, 128, 256],
# }

MODEL_NEURAL_NETWORK_SMOTE_CONFIG = {
    "train_filename": "loan_data_nn_normal_train_smote.csv",
    "test_filename": "loan_data_nn_normal_test.csv",
    "model_filename": "neural_network_smote.joblib",
    "model_solver": "adam",
    "model_max_iter": 200,
    "model_early_stopping": True,
    "model_validation_fraction": 0.1,
    "model_n_iter_no_change": 12,
    "model_random_state": 1,
    "cv_n_splits": 5,
    "cv_shuffle": True,
    "cv_random_state": 1,
    "search_n_iter": 20,
    "search_scoring": "f1",
    "search_random_state": 1,
    "search_n_jobs": -1,
    "search_verbose": 1,
    "search_param_distributions": MODEL_NEURAL_NETWORK_SEARCH_SPACE,
    # "search_param_distributions": MODEL_NEURAL_NETWORK_SMOTE_SEARCH_SPACE,
}


# These files are already split. This script only normalizes and encodes them.
NORMALIZE_NEURAL_NETWORK_CLEAN_DATA_CONFIG = {
    "train_input_filename": "loan_data_clean_train_3.csv",
    "test_input_filename": "loan_data_raw_test_1.csv",
    "train_output_filename": "loan_data_nn_clean_normalized_train.csv",
    "test_output_filename": "loan_data_nn_clean_normalized_test.csv",
}

# SMOTENC has already been applied to the training file. This script only normalizes and encodes it.
NORMALIZE_NEURAL_NETWORK_SMOTE_DATA_CONFIG = {
    "train_input_filename": "loan_data_smoted_raw_train_4.csv",
    "test_input_filename": "loan_data_raw_test_1.csv",
    "train_output_filename": "loan_data_nn_smote_normalized_train.csv",
    "test_output_filename": "loan_data_nn_smote_normalized_test.csv",
}

MODEL_NEURAL_NETWORK_CLEAN_DATA_SEARCH_SPACE = {
    "model__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
    "model__activation": ["relu", "tanh"],
    "model__alpha": loguniform(1e-5, 1e-2),
    "model__learning_rate_init": loguniform(1e-4, 5e-3),
    "model__batch_size": [64, 128, 256],
}

MODEL_NEURAL_NETWORK_SMOTE_DATA_SEARCH_SPACE = {
    "model__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
    "model__activation": ["relu", "tanh"],
    "model__alpha": loguniform(1e-5, 1e-2),
    "model__learning_rate_init": loguniform(1e-4, 5e-3),
    "model__batch_size": [64, 128, 256],
}

MODEL_NEURAL_NETWORK_CLEAN_DATA_CONFIG = {
    "train_filename": "loan_data_nn_clean_normalized_train.csv",
    "test_filename": "loan_data_nn_clean_normalized_test.csv",
    "model_filename": "neural_network_clean_data.joblib",
    "model_solver": "adam",
    "model_max_iter": 200,
    "model_early_stopping": True,
    "model_validation_fraction": 0.1,
    "model_n_iter_no_change": 12,
    "model_random_state": 1,
    "cv_n_splits": 5,
    "cv_shuffle": True,
    "cv_random_state": 1,
    "search_n_iter": 20,
    "search_scoring": "f1",
    "search_random_state": 1,
    "search_n_jobs": -1,
    "search_verbose": 1,
    "search_param_distributions": MODEL_NEURAL_NETWORK_CLEAN_DATA_SEARCH_SPACE,
}

MODEL_NEURAL_NETWORK_SMOTE_DATA_CONFIG = {
    "train_filename": "loan_data_nn_smote_normalized_train.csv",
    "test_filename": "loan_data_nn_smote_normalized_test.csv",
    "model_filename": "neural_network_smote_data.joblib",
    "model_solver": "adam",
    "model_max_iter": 200,
    "model_early_stopping": True,
    "model_validation_fraction": 0.1,
    "model_n_iter_no_change": 12,
    "model_random_state": 1,
    "cv_n_splits": 5,
    "cv_shuffle": True,
    "cv_random_state": 1,
    "search_n_iter": 20,
    "search_scoring": "f1",
    "search_random_state": 1,
    "search_n_jobs": -1,
    "search_verbose": 1,
    "search_param_distributions": MODEL_NEURAL_NETWORK_SMOTE_DATA_SEARCH_SPACE,
}
