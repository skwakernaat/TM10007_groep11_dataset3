import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from evaluate_model_with_kfold import evaluate_model_with_kfold

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
import numpy as np

def svm_classifier_with_grid_search(X, y):
    # Define parameter grid with different kernels and their respective parameters
    param_grid = param_grid = [
    {
        'kernel': ['linear'],
        'C': np.linspace(0.01, 1, 5)  # 5 values from 0.01 to 1
    },
    {
        'kernel': ['rbf'],
        'C': np.linspace(0.01, 1, 5),
        'gamma': list(np.logspace(-3, -1, 4)),
    },
    {
        'kernel': ['poly'],
        'C': list(np.linspace(0.01, 1, 5)),
        'gamma': list(np.logspace(-3, -1, 4)),
        'degree': list(range(1, 6, 1)),
        'coef0': list(np.linspace(0.01, 1, 5))
    }
]

    svm = SVC()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X, y)


    results_df = pd.DataFrame(grid_search.cv_results_)
    top_models = results_df.sort_values(by='mean_test_score', ascending=False).head(3)
    return top_models


