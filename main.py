'''This module contains the main function which is used to run the entire pipeline of the project.
    It loads the data, cleans it, splits it into training and test sets, balances the data, scales it,
    performs feature selection, and trains different classifiers.'''

#%%
from preprocessing.load_data import load_data
from preprocessing.clean_data import clean_data
from preprocessing.split_data import split_data
from preprocessing.balance_data import balance_data
from preprocessing.remove_zero_var import remove_zero_var
from preprocessing.univariate_feature_selection import univariate_feature_selection
from preprocessing.scale_data import scale_data
from classifiers.qda_classifier import qda_with_grid_search
from classifiers.random_forest import random_forest_classifier_grid_search
from classifiers.SVM_classifier import svm_classifier_with_grid_search
from classifiers.linear_classifiers import linear_classifier_with_grid_search
from results.display_results import display_results
from results.plot_learning_curve import plot_learning_curve

#%%
data = load_data()

# Checks for NaN and Null values, and replaces GIST and non-GIST with 1 and 0 respectively
data_cleaned = clean_data(data)

# Splits the data into training (80%) and test (20%) sets
X_train_unprocessed, X_test_unprocessed, y_train, y_test = split_data(data_cleaned)

#%%
# Checks for the balance between GIST and non-GIST in the training set
X_train_balanced, X_test_balanced = balance_data(X_train_unprocessed, y_train, X_test_unprocessed)

# Removes features with near-zero variance from the training and test data
X_train_filtered, X_test_filtered = remove_zero_var(X_train_balanced, X_test_balanced)

# Univariate feature selection on the train and test data based on the training data
X_train_features, X_test_features = univariate_feature_selection(X_train_filtered, y_train,
                                                                 X_test_filtered, n_features=30)

# Scales the training and test data based on the training data
X_train_scaled, X_test_scaled = scale_data(X_train_features, X_test_features)

X_train = X_train_scaled
X_test = X_test_scaled

#%%
# functions for different classifiers
results_svm = svm_classifier_with_grid_search(X_train, y_train)

results_qda = qda_with_grid_search(X_train, y_train)

#results_rf = random_forest_classifier_grid_search(X_train, y_train)

results_linear = linear_classifier_with_grid_search(X_train, y_train)
