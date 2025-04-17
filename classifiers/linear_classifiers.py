'''This module contains the function linear_classifier which is used to train a linear classifier
    on the data and evaluate its performance.'''

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from classifiers.grid_search import grid_search

def linear_classifier_with_grid_search(X, y, n_folds=5):
    '''This function performs a grid search for linear classifiers. It takes the training data
        and labels as input and returns the top 3 models based on accuracy.'''
    classifiers_and_grids = {
        "Logistic Regression": (LogisticRegression(max_iter=1000),
            {
                "C": np.logspace(-3, 3, 5),
                "penalty": ['l2'],
                "solver": ['lbfgs', 'liblinear'],
            }
        ),
        "LDA": (LinearDiscriminantAnalysis(),
            {
                "solver": ['svd', 'lsqr'],
                "shrinkage": [None]
            }
        ),
        "SGD Classifier": (SGDClassifier(loss='log_loss', max_iter=1000),
            {
                "alpha": np.logspace(-4, -1, 4),
                "penalty": ['l2', 'l1'],
                "learning_rate": ['constant', 'optimal', 'invscaling'],
                "eta0": [0.001, 0.01, 0.1]
            }
        )
    }

    # Perform grid search for each classifiers
    best_results = {}
    for name, (clf, param_grid) in classifiers_and_grids.items():
        top_3_df, top_3_models = grid_search(X, y, clf, param_grid)

        best_results[name] = top_3_models
    return best_results



