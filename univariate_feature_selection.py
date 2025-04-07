'''This module contains the function univariate_feature_selection which is used to select
    the best features using the ANOVA F-test. It also plots the scores of the features.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

def univariate_feature_selection(data_train, labels_train, n_features=30):
    '''This function calculates the ANOVA per feature and returns the top n features and their scores.'''

    # Scale the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(data_train)

    # Select all features initially
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(x_train, labels_train)

    # Get scores and their indices
    all_scores = selector.scores_
    all_indices = np.arange(len(all_scores))
    feature_scores = list(zip(all_indices, all_scores))

<<<<<<< HEAD
    # Koppel de scores aan de indices
    feature_scores = list(zip(selected_indices, selected_scores))

    # Sorteer op basis van de scores (aflopend)
=======

    # Sort scores descending
>>>>>>> main
    sorted_feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)

    # Pick top n_features
    top_indices = [x[0] for x in sorted_feature_scores[:n_features]]
    #top_scores = [x[1] for x in sorted_feature_scores[:n_features]]


    # Sort all scores descending
    sorted_all_scores = sorted(all_scores, reverse=True)

    # Get top features from original DataFrame
    data_train_n_features = data_train[:, top_indices]

    # Plot: x = feature rank (1 = best), y = F-score
    plt.figure(figsize=(12, 4))
    plt.title("ANOVA F-test Scores (Sorted)")
    plt.xlabel("Feature Rank (High to Low Score)")
    plt.ylabel("F-score")
    plt.plot(range(1, len(sorted_all_scores) + 1), sorted_all_scores, marker='o', linestyle='-', markersize=3)
    plt.grid(True)
    plt.xticks(ticks=range(0, len(sorted_all_scores) + 1, 10))
    plt.tight_layout()
    plt.show()

<<<<<<< HEAD
    return sorted_indices
=======
    return data_train_n_features, sorted_all_scores
>>>>>>> main
