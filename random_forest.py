from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from tqdm import tqdm

def random_forest_classifier(data_train, labels_train, data_test, labels_test, n_estimators=100, random_state=42, max_features_to_test=50, plot=True):
    '''This function trains a Random Forest classifier and evaluates its performance.'''

    # Train a model with all features to get feature importances
    feature_importances_all_runs = np.zeros(data_train.shape[1])

    # Perform 50 runs and accumulate feature importances
    for run in range(50):
        print(f"\n--- Run {run+1} ---")

        # Use a different random_state for each run (e.g., 42+run)
        current_state = 42 + run

        # Train a model with all features to obtain feature importances
        model_full = RandomForestClassifier(n_estimators=n_estimators, random_state=current_state)
        model_full.fit(data_train, labels_train)
        importances = model_full.feature_importances_

        # Add the importances of this run to the cumulative importances
        feature_importances_all_runs += importances

    # Calculate the average feature importances over the 50 runs
    average_feature_importances = feature_importances_all_runs / 50

    # Sort the features based on average importance (from high to low)
    average_importances_sorted_indices = np.argsort(average_feature_importances)[::-1]

    # Plot the average feature importances (with feature indices on the x-axis)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(average_feature_importances)), average_feature_importances[average_importances_sorted_indices])

    # Set the x-axis based on the sorted feature indices
    plt.xticks(range(len(average_feature_importances)), [f"F{idx}" for idx in average_importances_sorted_indices], rotation=90)

    plt.xlabel('Feature Index')
    plt.ylabel('Average Importance')
    plt.title('Average Feature Importances (over 50 Runs)')
    plt.tight_layout()
    plt.show()

    # Test the model with different numbers of selected features (1 to max_features_to_test)
    accuracies = []
    auc_scores = []

    # Create a KFold split with shuffle and a fixed random_state for consistent evaluation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Use tqdm for a progress bar
    for n in tqdm(range(1, max_features_to_test + 1), desc="Evaluating features", unit="features"):
        # Select the top n features based on average importance
        top_n_indices_subset = average_importances_sorted_indices[:n]

        # Select the subset of data with the top n features
        X_train_subset = data_train[:, top_n_indices_subset]
        X_test_subset = data_test[:, top_n_indices_subset]

        # Train the model with the selected features
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        # Calculate both the accuracy and the AUC (roc_auc)
        accuracy_scores = cross_val_score(model, X_train_subset, labels_train, cv=kf, scoring='accuracy')
        auc_scores_run = cross_val_score(model, X_train_subset, labels_train, cv=kf, scoring='roc_auc')

        accuracies.append(accuracy_scores.mean())
        auc_scores.append(auc_scores_run.mean())

    # Plot the accuracy and AUC for different numbers of selected features
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_features_to_test + 1), accuracies, marker='o', label='Accuracy', color='b')
    plt.plot(range(1, max_features_to_test + 1), auc_scores, marker='x', label='AUC', color='r')

    plt.xlabel('Number of Features')
    plt.ylabel('Score')
    plt.title(f'Accuracy and AUC vs Number of Features (max {max_features_to_test} features)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Combine the scores (e.g., as a weighted average)
    alpha = 0.5  # Set how important you think accuracy is compared to AUC (0.5 means both are equally important)
    combined_scores = [alpha * acc + (1 - alpha) * auc for acc, auc in zip(accuracies, auc_scores)]

    # Plot the combined score
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_features_to_test + 1), combined_scores, marker='o', label='Combined Score', color='g')
    plt.xlabel('Number of Features')
    plt.ylabel('Combined Score')
    plt.title(f'Combined Score (Accuracy and AUC) vs Number of Features')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Find the optimal number of features based on the combined score
    best_n_features_combined = np.argmax(combined_scores) + 1  # +1 because indexing starts at 0
    print(f"Optimal number of features (combined score): {best_n_features_combined} with a combined score of {max(combined_scores):.4f}")

    return best_n_features_combined, model_full
