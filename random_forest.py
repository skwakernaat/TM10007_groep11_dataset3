from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold
from evaluate_model_with_kfold import evaluate_model_with_kfold  # Importeer de functie

def random_forest_classifier(data_train, labels_train, data_test, labels_test, n_estimators=100, random_state=42, max_features_to_test=50, plot=True):
    '''This function trains a Random Forest classifier and evaluates its performance.'''

    # Train a model with all features to get feature importances
    feature_importances_all_runs = np.zeros(data_train.shape[1])

    # Perform 1 run and accumulate feature importances
    for run in range(1):
        current_state = 42 + run

        # Train a model with all features to obtain feature importances
        model_full = RandomForestClassifier(n_estimators=n_estimators, random_state=current_state)
        model_full.fit(data_train, labels_train)
        importances = model_full.feature_importances_

        # Add the importances of this run to the cumulative importances
        feature_importances_all_runs += importances

    # Calculate the average feature importances over the 1 run
    average_feature_importances = feature_importances_all_runs / 1

    # Sort the features based on average importance (from high to low)
    average_importances_sorted_indices = np.argsort(average_feature_importances)[::-1]

    # Test the model with different numbers of selected features (1 to max_features_to_test)
    accuracies = []
    sensitivities = []

    # Perform StratifiedKFold cross-validation using the `evaluate_model_with_kfold` function
    for n in range(1, max_features_to_test + 1):
        # Select the top n features based on average importance
        top_n_indices_subset = average_importances_sorted_indices[:n]

        # Select the subset of data with the top n features
        X_train_subset = data_train[:, top_n_indices_subset]
        X_test_subset = data_test[:, top_n_indices_subset]

        # Call the evaluate_model_with_kfold function with the subset of features
        results = evaluate_model_with_kfold(X_train_subset, labels_train, model=RandomForestClassifier(n_estimators=n_estimators, random_state=random_state), n_splits=5)

        # Append the results for accuracy and sensitivity
        accuracies.append(results['mean_accuracy'])
        sensitivities.append(results['mean_sensitivity'])

    # Get the accuracy and sensitivity for the best number of features
    best_n_features_accuracy = accuracies[np.argmax(accuracies)]
    best_n_features_sensitivity = sensitivities[np.argmax(sensitivities)]

    # Print the results
    print(f"Best Accuracy: {best_n_features_accuracy:.4f} with {np.argmax(accuracies) + 1} features")
    print(f"Best Sensitivity (Recall): {best_n_features_sensitivity:.4f} with {np.argmax(sensitivities) + 1} features")

    return best_n_features_accuracy, best_n_features_sensitivity, model_full
