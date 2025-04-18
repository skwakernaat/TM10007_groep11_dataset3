'''This module contains the function linear_classifier which is used to train a linear classifier
    on the data and evaluate its performance.'''

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from classifiers.grid_search import grid_search

def linear_classifier_with_grid_search(X, y):
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
            [
                {"solver": ["svd"]},  # no shrinkage allowed
                {"solver": ["lsqr"], "shrinkage": [None, "auto"]},
            ]
        ),
        "SGD Classifier": (
            CalibratedClassifierCV(estimator=make_pipeline(StandardScaler(),
                                            SGDClassifier(loss='log_loss', max_iter=1000)), cv=5),
            {
                "estimator__sgdclassifier__alpha": np.logspace(-4, -1, 4),
                "estimator__sgdclassifier__penalty": ['l2', 'l1'],
                "estimator__sgdclassifier__learning_rate": ['constant', 'optimal', 'invscaling'],
                "estimator__sgdclassifier__eta0": [0.001, 0.01, 0.1],
            }
        )
    }

    # Perform grid search for each classifiers
    best_results = {}
    for name, (clf, param_grid) in classifiers_and_grids.items():
        top_3_models = grid_search(X, y, clf, param_grid)[1]

        best_results[name] = top_3_models

    return best_results
