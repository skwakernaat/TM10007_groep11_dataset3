import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold


def plot_learning_curve(clf, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), random_state=42):
    """
    Plot a learning curve of a given classifier.
    """
    # Create a StratifiedKFold object for cross-validation
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Generate the learning curve data
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator=clf,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        shuffle=True,
        random_state=random_state,
    )

    # Calculate the mean and standard deviation of training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Plotting the learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', label='Training score')
    plt.plot(train_sizes_abs, val_scores_mean, 'o-', label='Validation score')
    plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1)
    plt.fill_between(train_sizes_abs, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1)

    plt.title('Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel(scoring.capitalize())
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
