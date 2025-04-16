'''This module contains the function random_forest_classifier_grid_search which is used to
    perform a grid search for the Random Forest classifier.'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd

def random_forest_classifier_grid_search(X, y):
    '''This function performs a grid search for the Random Forest classifier. It takes the training
        data and labels as input and returns the top 3 models based on accuracy.'''

    # Define hyperparameters for grid search
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None] + list(range(1, 31, 10)),
        'min_samples_split': list(range(2, 11, 2))
    }

    # Make random forest model
    clf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X, y)

    results_df = pd.DataFrame(grid_search.cv_results_)
    top_models = results_df.sort_values(by='mean_test_score', ascending=False).head(3)

    return top_models
