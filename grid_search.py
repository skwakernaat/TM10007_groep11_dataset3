from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import numpy as np


def grid_search(X, y, clf, param_grid):
    '''This function performs a grid search for the given classifier. It takes the training data,
        labels, classifier, and hyperparameters as input and returns the top 3 models based on accuracy.'''


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search_classifier = GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy')
    grid_search_classifier.fit(X, y)
    results_df = pd.DataFrame(grid_search_classifier.cv_results_)
    top_3_df = results_df.sort_values(by='mean_test_score', ascending=False).head(3)

    top_3_models = []
    for _, row in top_3_df.iterrows():
        params = row['params']
        score = row['mean_test_score']

        # Create and train model with these parameters
        model = clf.__class__(**params)
        model.fit(X, y)
        top_3_models.append((model, score))

    return top_3_df, top_3_models



