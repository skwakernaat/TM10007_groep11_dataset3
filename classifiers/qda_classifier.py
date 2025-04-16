"""In this module the QDA classifier is trained and tested"""

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd

def qda_with_grid_search(X, y, n_splits=5):
    """Train and test QDA classifier with grid search."""
    # # initialize and train model
    # qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
    # qda.fit(data_train_n_features, labels_train)

    # # predict on the test set
    # predictions = qda.predict(data_test_n_features)

    # reg_param is to handle situation where the covariance matrix is singular
    param_grid = {"reg_param": [0.01, 0.2575, 0.505, 0.7525, 1.0],}

    qda = QuadraticDiscriminantAnalysis()

    # Cross-validation for hyperparameter tuning, gird search
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid_search = GridSearchCV(qda, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X, y)

    results_df = pd.DataFrame(grid_search.cv_results_)
    top_models = results_df.sort_values(by='mean_test_score', ascending=False).head(3)

    return top_models
