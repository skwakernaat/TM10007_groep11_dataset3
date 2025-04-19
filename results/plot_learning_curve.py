'''This module contains the function plot_learning_curve which is used to plot the learning curve
    of a given classifier.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold

def plot_learning_curve(clfs, X, y, cv=5, scoring='accuracy',
                        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42):
    '''Plot learning curves from the selected classifiers.'''

    # Support for both dict.values() and simple lists
    if isinstance(clfs, dict):
        clfs = list(clfs.values())

    # Flatten lists of tuples if needed
    flat_clfs = []
    for item in clfs:
        if isinstance(item, tuple):
            flat_clfs.append(item)
        elif isinstance(item, list) and all(isinstance(x, tuple) for x in item):
            flat_clfs.extend(item)
        else:
            raise ValueError("Expected a list of (model, score) tuples or dict of such lists.")

    # Setup plots
    fig, axes = plt.subplots(nrows=len(flat_clfs), ncols=1, figsize=(8, 4 * len(flat_clfs)), sharex=True)
    if len(flat_clfs) == 1:
        axes = [axes]

    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for idx, (clf, _) in enumerate(flat_clfs):
        ax = axes[idx]
        name = clf.__class__.__name__

        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator=clf, X=X, y=y,
            train_sizes=train_sizes, cv=cv,
            scoring=scoring, n_jobs=-1
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        ax.plot(train_sizes_abs, train_scores_mean, 'o-', label='Training score')
        ax.plot(train_sizes_abs, val_scores_mean, 'o-', label='Validation score')
        ax.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1)
        ax.fill_between(train_sizes_abs, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.1)

        ax.set_title(name)
        ax.set_ylabel(scoring.capitalize())
        ax.legend(loc='best')
        ax.grid(True)

    axes[-1].set_xlabel("Training Set Size")
    plt.tight_layout()
    plt.show()
