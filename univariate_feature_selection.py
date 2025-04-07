'''This module contains the function univariate_feature_selection which is used to select
    the best features using the ANOVA F-test. It also plots the scores of the features.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

def univariate_feature_selection(data_train_scaled, labels_train):
    '''This function calculates the ANOVA per feature and picks the top k amount of features'''

    # Schaal de data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(data_train_scaled)

    # Bepaal het aantal features dat je wilt selecteren (bijvoorbeeld 100)
    k_best = np.shape(data_train_scaled)[1]

    # Pas univariate feature selectie toe met de ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k=k_best)
    x_train_selected = selector.fit_transform(x_train_scaled, labels_train)

    # De geselecteerde features (scores en indices)
    selected_indices = selector.get_support(indices=True)
    selected_scores = selector.scores_

    # Koppel de scores aan de indices
    feature_scores = list(zip(selected_indices, selected_scores))

    # Sorteer op basis van de scores (aflopend)
    sorted_feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)

    # Verkrijg de gesorteerde indices en scores
    sorted_indices = [x[0] for x in sorted_feature_scores]
    sorted_scores = [x[1] for x in sorted_feature_scores]

    # Laat de gesorteerde indices en scores zien
    #print("Geselecteerde feature indices (gesorteerd):", sorted_indices)
    #print("Feature scores (gesorteerd):", sorted_scores)
    plt.figure()
    plt.title("Top Feature Scores from ANOVA F-test")
    plt.xlabel("Feature Rank")
    plt.ylabel("F-score")
    plt.plot(range(1, 100), sorted_scores[:99])
    plt.tight_layout()
    plt.show()

    return sorted_indices
