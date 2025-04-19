'''This module contains the main function which is used to run the entire pipeline of the project.
    It loads the data, cleans it, splits it into training and test sets, balances the data, scales it,
    performs feature selection, and trains different classifiers.'''
#%%
from preprocessing.load_data import load_data
from preprocessing.clean_data import clean_data
from preprocessing.split_data import split_data
from preprocessing.check_data_balance import check_balance
from preprocessing.load_feature_selection import get_or_run_feature_selection
from preprocessing.scale_data import scale_data
from classifiers.qda_classifier import qda_with_grid_search
from classifiers.random_forest import random_forest_classifier_grid_search
from classifiers.SVM_classifier import svm_classifier_with_grid_search
from classifiers.linear_classifiers import linear_classifier_with_grid_search
from results.plot_learning_curve import plot_learning_curve
from results.evaluate_model import evaluate_model

#%% Preprocessing steps
data = load_data()

# Checks for NaN and Null values, and replaces GIST and non-GIST with 1 and 0 respectively
data_cleaned = clean_data(data)

# Splits the data into training (80%) and test (20%) sets
X_train_unprocessed, X_test_unprocessed, y_train, y_test, feature_names = split_data(data_cleaned)

# Checks for the balance between GIST and non-GIST in the training set
check_balance(y_train)

# Forward greedy feature selection on the train and test data based on the training data
X_train_features, X_test_features = get_or_run_feature_selection(
                            X_train_unprocessed, y_train, X_test_unprocessed, feature_names,
                            n_features=12, save_dir="results")

# Scales the training and test data based on the training data
X_train_scaled, X_test_scaled = scale_data(X_train_features, X_test_features)

# Make duplicates
X_train = X_train_scaled.copy()
X_test = X_test_scaled.copy()

#%% Making the classifiers
# Makes the top 3 models for each classifier
results_svm = svm_classifier_with_grid_search(X_train, y_train)

results_qda = qda_with_grid_search(X_train, y_train)

results_rf = random_forest_classifier_grid_search(X_train, y_train)

results_linear = linear_classifier_with_grid_search(X_train, y_train)
#%% Plot the learning curves for each classifier
# Plot the learning curves for each classifier
# Plot the learning curves for each classifier
for models in results_linear.values():
    plot_learning_curve(models, X_train, y_train)

plot_learning_curve(results_qda, X_train, y_train)

plot_learning_curve(results_svm, X_train, y_train)

plot_learning_curve(results_rf, X_train, y_train)

#%% Compute the final results on the test set
best_models = [ #enter the best models here, based on the learning curves
    results_rf[1][0],
    results_qda[2][0],
    results_svm[1][0],
    results_linear['LDA'][0][0],
    results_linear['Logistic Regression'][0][0],
    results_linear['SGD Classifier'][2][0]
    ]

results = evaluate_model(X_test, y_test, best_models)
