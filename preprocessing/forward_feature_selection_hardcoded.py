'''This module contains the function select_top_features which is used to select the top n informative
    features using sklearn's SequentialFeatureSelector. It takes the training data and the number of
    features to select as input and returns the reduced dataset with the label.'''

import numpy as np

def forward_feature_selection_hardcoded(X_train, X_test, feature_names):
    '''Use sklearn's SequentialFeatureSelector to select the top n informative features.'''

    selected_features = ['PREDICT_original_sf_prax_avg_2.5D', 'PREDICT_original_sf_area_avg_2.5D',
        'PREDICT_original_tf_LBP_energy_R8_P24', 'PREDICT_original_tf_GLCM_homogeneityd1.0A1.57',
        'PREDICT_original_tf_Gabor_peak_position_F0.05_A0.0',
        'PREDICT_original_tf_Gabor_min_F0.05_A0.79', 'PREDICT_original_tf_Gabor_max_F0.05_A1.57',
        'PREDICT_original_tf_Gabor_max_F0.2_A0.0', 'PREDICT_original_tf_Gabor_kurtosis_F0.2_A0.0',
        'PREDICT_original_tf_Gabor_mean_F0.5_A0.79', 'PREDICT_original_tf_Gabor_max_F0.5_A2.36',
        'PREDICT_original_phasef_phasesym_quartile_range_WL3_N5']

    selected_indices = [np.where(feature_names == name)[0][0] for name in selected_features]

    X_train_features = X_train[:, selected_indices]
    X_test_features = X_test[:, selected_indices]

    return X_train_features, X_test_features
