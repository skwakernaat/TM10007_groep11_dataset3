'''This module contains the function linear_classifier which is used to train a linear classifier
    on the data and evaluate its performance.'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    best_results = {}

    # Perform grid search for each classifier
    for name, (clf, param_grid) in classifiers_and_grids.items():
        grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X, y)

        # Get grid results in df
        results_df = pd.DataFrame(grid_search.cv_results_)
        # Sort by mean test score and take top 3 models
        top_models = results_df.sort_values(by='mean_test_score', ascending=False).head(3)

        # Store the best results for each classifier
        best_results[name] = top_models

    return best_results


