'''This module contains the function plot_learning_curve which is used to plot the learning curve
    of a given classifier.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from IPython.display import display

def plot_learning_curve(clfs, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10),
                        random_state=42):
    """Plot a learning curve of a given classifier."""

    # Set up 3 subplots vertically
    _, axes = plt.subplots(nrows=len(clfs), ncols=1, figsize=(8, 12), sharex=True)

    # Create a StratifiedKFold object for cross-validation
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for idx, (clf, _) in enumerate(clfs):
        ax = axes[idx]

        # Get the classifier name
        clf_name = clf.__class__.__name__

        # Get the parameters of the classifier
        params = clf.get_params()
        params_str = ', '.join([f'{k}={v}' for k, v in params.items()])

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
        # Plot on this subplot
        ax.plot(train_sizes_abs, train_scores_mean, 'o-', label='Training score')
        ax.plot(train_sizes_abs, val_scores_mean, 'o-', label='Validation score')
        ax.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1)
        ax.fill_between(train_sizes_abs, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.1)

        # Titles and labels
        ax.set_title(f'Parameters: {params_str}', fontsize=6)
        ax.set_ylabel(scoring.capitalize())
        ax.grid(True)
        ax.legend(loc='best', fontsize=8)

    # Common X label
    axes[-1].set_xlabel("Training Set Size")

    # Adjust spacing
    plt.suptitle(f"Learning Curves for {clf_name} Variants", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the suptitle
    plt.show()
    display(plt.gcf())  #ensures rendering in Google Colab
