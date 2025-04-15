'''This module contains the main function which is used to run the entire pipeline of the project.
    It loads the data, cleans it, splits it into training and test sets, balances the data, scales it,
    performs feature selection, and trains different classifiers.'''

#%%
from SVM_classifier import svm_classifier
from SVM_classifier import svm_poly_kernel
from rfecv_feature_selection import rfecv_features
from load_data import load_data
from clean_data import clean_data
from balance_data import balance_data
from split_data import split_data
from scale_data import scale_data
from univariate_feature_selection import univariate_feature_selection
from linear_classifiers import linear_classifier
from qda_classifier import qda_func
from random_forest import random_forest_classifier
from display_results import display_results
from plot_learning_curve import plot_learning_curve
from remove_zero_var import remove_zero_var

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

#rfecv_features(X_train_filtered, y_train)

#%%
# functions for different classifiers
results_svm = svm_classifier(X_train, y_train)

results_svm_polykernel = svm_poly_kernel(X_train, y_train)

results_linear = linear_classifier(X_train, y_train)

results_qda = qda_func(X_train, y_train)

results_rf = random_forest_classifier(X_train, y_train, n_estimators=100, random_state=42)

df_results = display_results(results_svm, results_svm_polykernel,
                             results_linear, results_qda, results_rf)

print(df_results)

plot_learning_curve(results_svm[-1]['model'], X_train, y_train)
