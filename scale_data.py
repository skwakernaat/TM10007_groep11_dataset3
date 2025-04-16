'''This module contains the function scale_data which is used to scale the data using different
    scaling methods. It also removes features with zero variance.'''

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_data(X_train, X_test, scaling_method='standard'):
    '''This function scales the data using different scaling methods. It also removes features
        with zero variance.'''

    # Choose scaler
    scaler_options = {
        'standard': StandardScaler(), #standardization
        'minmax': MinMaxScaler(), #normalization
    }

    scaler = scaler_options[scaling_method]
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled
