'''This module contains the function scale_data which is used to scale the data using different
    scaling methods. It also removes features with zero variance.'''

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold

def scale_data(X_train, X_test, scaling_method='standard'):
    '''This function scales the data using different scaling methods. It also removes features
        with zero variance.'''
    # Remove near-zero variance features
    var_thresh = 1e-15
    selector = VarianceThreshold(threshold=var_thresh)
    X_train_filtered = selector.fit_transform(X_train)
    X_test_filtered = selector.transform(X_test)

    # Choose scaler
    scaler_options = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }

    if scaling_method not in scaler_options:
        raise ValueError("Invalid method! Choose from 'standard', 'minmax', or 'robust'.")

    scaler = scaler_options[scaling_method]
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_test_scaled = scaler.transform(X_test_filtered)

    return X_train_scaled, X_test_scaled
