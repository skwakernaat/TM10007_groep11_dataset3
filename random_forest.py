'''This module contains the function random_forest_classifier_grid_search which is used to
    perform a grid search for the Random Forest classifier.'''

from sklearn.ensemble import RandomForestClassifier

from grid_search import grid_search

def random_forest_classifier_grid_search(X, y):
    '''This function performs a grid search for the Random Forest classifier. It takes the training
        data and labels as input and returns the top 3 models based on accuracy.'''
    # Make random forest model
    clf = RandomForestClassifier(random_state=42)

    # Define hyperparameters for grid search
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None] + list(range(1, 31, 10)),
        'min_samples_split': list(range(2, 11, 2))
    }

    # Perform grid search with cross-validation and get the top 3 models
    top_3_df, top_3_models = grid_search(X, y, clf, param_grid)

    return top_3_models
