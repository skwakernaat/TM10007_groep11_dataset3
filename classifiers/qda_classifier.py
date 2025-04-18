"""This module contains the function qda_with_grid_search, which trains and tests a
    Quadratic Discriminant Analysis (QDA) classifier using grid search
    for hyperparameter tuning."""

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from classifiers.grid_search import grid_search

def qda_with_grid_search(X, y):
    """Train and test QDA classifier with grid search for hyperparameter tuning."""

    # Make QDA model
    clf = QuadraticDiscriminantAnalysis()

    # Define hyperparameters for grid search
    # reg_param is to handle situation where the covariance matrix is singular
    param_grid = {"reg_param": [0.01, 0.2575, 0.505, 0.7525, 1.0],}

    # Perform grid search with cross-validation and get the top 3 models
    top_3_models = grid_search(X, y, clf, param_grid)[1]

    return top_3_models
