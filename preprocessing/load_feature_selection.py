import os
import pandas as pd
from preprocessing.forward_feature_selection import forward_feature_selection

def get_or_run_feature_selection(X_train_unprocessed, y_train, X_test_unprocessed, feature_names,
                                  n_features=12, save_dir="results"):
    """
    Load precomputed feature-selected datasets if available, else run feature selection and save them.

    Parameters:
    - X_train_unprocessed, y_train, X_test_unprocessed: Original train/test sets
    - feature_names: List of original feature names (after dropping ID/label)
    - n_features: Number of features to select
    - save_dir: Directory where to save/load CSV files

    Returns:
    - X_train_features, X_test_features (numpy arrays)
    """
    features_train_path = os.path.join(save_dir, "X_train_features.csv")
    features_test_path = os.path.join(save_dir, "X_test_features.csv")

    if os.path.exists(features_train_path) and os.path.exists(features_test_path):
        print("Loaded precomputed feature selection.")
        X_train_features = pd.read_csv(features_train_path).values
        X_test_features = pd.read_csv(features_test_path).values
    else:
        print("Running feature selection...")
        X_train_features, X_test_features = forward_feature_selection(
            X_train_unprocessed, y_train, X_test_unprocessed, feature_names, n_features=n_features
        )
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame(X_train_features).to_csv(features_train_path, index=False)
        pd.DataFrame(X_test_features).to_csv(features_test_path, index=False)
        print("Feature-selected data saved.")

    return X_train_features, X_test_features
