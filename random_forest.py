from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
from evaluate_model_with_kfold import evaluate_model_with_kfold
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd

def random_forest_classifier_grid_search(X, y):
    '''Deze functie traint een Random Forest model en evalueert de prestaties met k-fold cross-validation via evaluate_model_with_kfold.
    A grid search is used to search for the best hyperparameters.'''

    #define hyperparameters for grid search
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None + list(range(1, 30, 5))],
        'min_samples_split': list(range(2, 11, 1))
    }

    # Definieer het Random Forest model
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X, y)


    results_df = pd.DataFrame(grid_search.cv_results_)
    top_models = results_df.sort_values(by='mean_test_score', ascending=False).head(3)
    return top_models


