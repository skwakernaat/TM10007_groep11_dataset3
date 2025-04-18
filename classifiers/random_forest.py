'''This module contains the function random_forest_classifier_grid_search which is used to
    perform a grid search for the Random Forest classifier.'''

from sklearn.ensemble import RandomForestClassifier
from classifiers.grid_search import grid_search

def random_forest_classifier_grid_search(X, y):
    '''This function performs a grid search for the Random Forest classifier. It takes the training
        data and labels as input and returns the top 3 models based on accuracy.'''
    # Make random forest model
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Define hyperparameters for grid search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7],
        'min_samples_split': [4, 6],
        'min_samples_leaf': [4, 6],
        "min_impurity_decrease": [0.01],
        "bootstrap": [True, False],
    }

    # Perform grid search with cross-validation and get the top 3 models
    top_3_df, top_3_models = grid_search(X, y, clf, param_grid)

    return top_3_models
