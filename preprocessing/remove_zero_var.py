''''This module contains the function remove_zero_var which is used to remove features
    with near-zero variance.'''

from sklearn.feature_selection import VarianceThreshold

def remove_zero_var(X_train, X_test):
    '''This function removes features with near-zero variance.'''

    # Remove near-zero variance features
    var_thresh = 1e-15
    selector = VarianceThreshold(threshold=var_thresh)
    X_train_filtered = selector.fit_transform(X_train)
    X_test_filtered = selector.transform(X_test)

    return X_train_filtered, X_test_filtered
