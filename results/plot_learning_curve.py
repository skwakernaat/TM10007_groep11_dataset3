'''This module contains the function plot_learning_curve which is used to plot the learning curve
    of a given classifier.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold

def plot_learning_curve(clfs, X, y, cv=5, scoring='accuracy',
                        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42):
    '''Generate and save learning curve plots for the given classifiers.'''

    if isinstance(clfs, dict):
        clfs = list(clfs.values())

    flat_clfs = [x for item in clfs for x in (item if isinstance(item, list) else [item])]

    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for clf, _ in flat_clfs:
        fig, ax = plt.subplots(figsize=(8, 4))
        name = clf.__class__.__name__

        train_sizes_abs, train_scores, val_scores = learning_curve(
            clf, X, y, train_sizes=train_sizes, cv=cv,
            scoring=scoring, n_jobs=-1
        )

        ax.plot(train_sizes_abs, np.mean(train_scores, axis=1), 'o-', label='Training score')
        ax.plot(train_sizes_abs, np.mean(val_scores, axis=1), 'o-', label='Validation score')
        ax.fill_between(train_sizes_abs,
                        np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                        alpha=0.1)
        ax.fill_between(train_sizes_abs,
                        np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                        alpha=0.1)

        ax.set_title(name)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(scoring.capitalize())
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        filename = f"learning_curve_{name.lower().replace(' ', '_')}.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)
