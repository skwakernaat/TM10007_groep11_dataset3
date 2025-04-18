'''This module contains a function to perform grid search for hyperparameter tuning of classifiers.
    It uses StratifiedKFold for cross-validation and returns the top 3 models based on accuracy.'''

import copy
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def grid_search(X, y, clf, param_grid, n_top=3):
    '''Perform grid search with Stratified K-Fold and return top models based on scoring metric.'''

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search_classifier = GridSearchCV(clf, param_grid, cv=cv, scoring='roc_auc', refit=True)
    grid_search_classifier.fit(X, y)

    results_df = pd.DataFrame(grid_search_classifier.cv_results_)
    top_n_df = results_df.sort_values(by='mean_test_score', ascending=False).head(n_top)

    top_n_models = []
    for _, row in top_n_df.iterrows():
        # Get the best parameters
        best_params = row['params']

        # Deepcopy the original model and set the parameters
        model = copy.deepcopy(clf)
        model.set_params(**best_params)
        model.fit(X, y)

        score = row['mean_test_score']
        top_n_models.append((model, score))

    return top_n_df, top_n_models
