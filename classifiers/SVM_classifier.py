'''This module contains the function svm_classifier_with_grid_search which is used to
    perform a grid search for the SVM classifier.'''

import numpy as np
from sklearn.svm import SVC
from classifiers.grid_search import grid_search

def svm_classifier_with_grid_search(X, y):
    '''This function performs a grid search for the SVM classifier. It takes the training data
        and labels as input and returns the top 3 models based on accuracy.'''

    # Create SVM model
    clf = SVC(random_state=42)
    # Define parameter grid with different kernels and their respective parameters
    param_grid = [
    {
        'kernel': ['linear'],
        'C': np.linspace(0.01, 0.5, 5)  # 5 values from 0.01 to 1
    },
    {
        'kernel': ['rbf'],
        'C': np.linspace(0.01, 0.5, 5),
        'gamma': list(np.logspace(-3, -1, 4)),
    },
    {
        'kernel': ['poly'],
        'C': list(np.linspace(0.01, 0.5, 5)),
        'gamma': list(np.logspace(-3, -1, 4)),
        'degree': list(range(1, 4, 1)),
        'coef0': list(np.linspace(0.01, 1, 5))
    }
]
    # Perform grid search with cross-validation and get the top 3 models
    top_3_df, top_3_models = grid_search(X, y, clf, param_grid)

    return top_3_models
