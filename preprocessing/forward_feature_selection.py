'''This module contains the function select_top_features which is used to select the top n informative
    features using sklearn's SequentialFeatureSelector. It takes the training data and the number of
    features to select as input and returns the reduced dataset with the label.'''

import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector

def forward_feature_selection(X_train, y_train, X_test, y_test, feature_names):
    '''Use sklearn's SequentialFeatureSelector to select the top n informative features.'''

    hardcoded_features = [
        'PREDICT_original_sf_prax_avg_2.5D', 'PREDICT_original_sf_area_avg_2.5D',
        'PREDICT_original_tf_LBP_energy_R8_P24', 'PREDICT_original_tf_GLCM_homogeneityd1.0A1.57',
        'PREDICT_original_tf_Gabor_peak_position_F0.05_A0.0', 'PREDICT_original_tf_Gabor_min_F0.05_A0.79',
        'PREDICT_original_tf_Gabor_max_F0.05_A1.57', 'PREDICT_original_tf_Gabor_max_F0.2_A0.0',
        'PREDICT_original_tf_Gabor_kurtosis_F0.2_A0.0', 'PREDICT_original_tf_Gabor_mean_F0.5_A0.79',
        'PREDICT_original_tf_Gabor_max_F0.5_A2.36', 'PREDICT_original_phasef_phasesym_quartile_range_WL3_N5'
    ]

    if True:  # Toggle this to `False` to re-enable automatic selection
        indices = [feature_names.index(f) for f in hardcoded_features]
        X_train_features = X_train[:, indices]
        X_test_features = X_test[:, indices]
        print(f"[DEBUG] Using hardcoded features: {hardcoded_features}")
        return X_train_features, X_test_features

    # Calculate the number of features to select
    n_samples = len(y_train) + len(y_test)
    n_classes = len(np.unique(y_train))
    n_features = math.floor(n_samples / (10 * n_classes))

    # Create model and feature selector
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction='forward',
        scoring='roc_auc',
        cv=3,
        n_jobs=-1
    )

    X_train_features = selector.fit_transform(X_train, y_train)
    X_test_features = selector.transform(X_test)

    # Get selected feature names
    selected_mask = selector.get_support()  # Boolean mask of selected features
    selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
    print(f"Selected features: {selected_features}")

    return X_train_features, X_test_features
