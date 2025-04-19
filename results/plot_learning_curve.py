'''This module contains the function plot_learning_curve, which generates and saves a
    learning curve plot for multiple classifiers.'''

import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold

def plot_learning_curve(clfs, X, y, cv=5, scoring='accuracy',
                        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42):
    '''Generate and save a learning curve plot with subplots, grouped by classifier type.'''

    n_clfs = len(clfs)
    fig, axes = plt.subplots(nrows=n_clfs, ncols=1, figsize=(8, 4 * n_clfs), sharex=True)
    if n_clfs == 1:
        axes = [axes]

    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Automatically detect group name based on the most common classifier class
    class_names = [clf.__class__.__name__ for clf, _ in clfs]
    group_name = Counter(class_names).most_common(1)[0][0]

    for idx, (clf, _) in enumerate(clfs):
        ax = axes[idx]

        # Get the classifier parameters, limit to 9 for display
        params = clf.get_params()
        params_str = ', '.join([f'{k}={v}' for k, v in list(params.items())[:16]])

        # Inline wrapping logic
        wrapped_lines = []
        current_line = ""
        for param in params_str.split(', '):
            if len(current_line) + len(param) + 2 > 160:
                wrapped_lines.append(current_line.rstrip(', '))
                current_line = param + ', '
            else:
                current_line += param + ', '

        if current_line:  # Add any leftover line
            wrapped_lines.append(current_line.rstrip(', '))

        wrapped_params = '\n'.join(wrapped_lines)

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

        ax.set_title(f"Parameters:\n{wrapped_params}", fontsize=7, wrap=True)
        ax.set_ylabel(scoring.capitalize())
        ax.legend(loc='best', fontsize=8)
        ax.grid(True)

    axes[-1].set_xlabel("Training Set Size")

    fig.suptitle(f"Learning Curves - {group_name}", fontsize=16)
    fig.subplots_adjust(top=0.88, hspace=0.6)

    filename = f"learning_curves_{group_name.lower().replace(' ', '_')}.png"
    filepath = os.path.abspath(filename)
    plt.savefig(filepath, dpi=300)
    print(f"Saved learning curve plot of {group_name} to: {filepath}")
    plt.close(fig)
