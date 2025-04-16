'''This module contains the function svm_classifier_with_grid_search which is used to
    perform a grid search for the SVM classifier.'''

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def svm_classifier_with_grid_search(X, y):
    '''This function performs a grid search for the SVM classifier. It takes the training data
        and labels as input and returns the top 3 models based on accuracy.'''
    # Define parameter grid with different kernels and their respective parameters
    param_grid = [
    {
        'kernel': ['linear'],
        'C': np.linspace(0.01, 1, 5)  # 5 values from 0.01 to 1
    },
    {
        'kernel': ['rbf'],
        'C': np.linspace(0.01, 1, 5),
        'gamma': list(np.logspace(-3, -1, 4)),
    },
    {
        'kernel': ['poly'],
        'C': list(np.linspace(0.01, 1, 5)),
        'gamma': list(np.logspace(-3, -1, 4)),
        'degree': list(range(1, 6, 1)),
        'coef0': list(np.linspace(0.01, 1, 5))
    }
]

    svm = SVC(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X, y)

    results_df = pd.DataFrame(grid_search.cv_results_)
    top_models = results_df.sort_values(by='mean_test_score', ascending=False).head(3)

    return top_models
