'''This module contains the function scale_data which is used to scale the data using different
    scaling methods. It also removes features with zero variance.'''

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_data(data, scaling_method='standard'):
    '''This function scales the data using different scaling methods. It also removes features
        with zero variance.'''
    # Separate features and label
    features = data.drop(columns=['label'])
    label = data['label']

    # Remove near-zero variance features
    var_thresh = 1e-15
    var_features = data.var(axis=0)
    features_filtered = features.loc[:, var_features > var_thresh]

    # Choose scaler
    scaler_options = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }

    if scaling_method not in scaler_options:
        raise ValueError("Invalid method! Choose from 'standard', 'minmax', or 'robust'.")

    scaler = scaler_options[scaling_method]
    features_scaled = scaler.fit_transform(features_filtered)

    # Convert back to DataFrame with original (filtered) column names
    data_scaled = pd.DataFrame(features_scaled, columns=features_filtered.columns, index=data.index)
    data_scaled.insert(0, 'label', label)  # Insert label column back into the DataFrame

    return data_scaled
