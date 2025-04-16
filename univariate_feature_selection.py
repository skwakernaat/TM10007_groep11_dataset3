'''This module contains the function univariate_feature_selection which is used to select
    the best features using the ANOVA F-test. It also plots the scores of the features.'''

from sklearn.feature_selection import SelectKBest, f_classif

def univariate_feature_selection(X_train, y_train, X_test, n_features=30):
    '''This function calculates the ANOVA per feature and returns the top n features and
        their scores.'''

    # Select all features initially
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_train_features = selector.fit_transform(X_train, y_train)
    X_test_features = selector.transform(X_test)

    return X_train_features, X_test_features
