
#%%
import numpy as np
import seaborn
import pandas as pd
import torch
import itertools
import matplotlib.pyplot as plt
import statistics
import math
from load_data import load_data
from sklearn import model_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler



from clean_data import clean_data
from balance_data import balance_data
from split_data import split_data
from scale_data import scale_data
from univariate_feature_selection import univariate_feature_selection
from qda_classifier import qda_with_grid_search
from random_forest import random_forest_classifier_grid_search
from display_results import display_results
from SVM_classiefier import svm_classifier_with_grid_search
from linear_classifiers import linear_classifier_with_grid_search

#%%
data = load_data()

data_cleaned = clean_data(data)

data_balanced = balance_data(data_cleaned)

data_scaled = scale_data(data_balanced)

data_train, data_test, labels_train, labels_test = split_data(data_scaled)

#%%
# Feature selection
data_train_n_features, data_test_n_features = univariate_feature_selection(
                                                            data_train, labels_train, data_test, n_features=30)
# Train and evaluate classifiers
grid_svm = svm_classifier_with_grid_search(data_train_n_features, labels_train)

grid_linear = linear_classifier_with_grid_search(data_train_n_features, labels_train)

grid_qda = qda_with_grid_search(data_train_n_features, labels_train)
print(grid_qda)

grid_rf = random_forest_classifier_grid_search(data_train_n_features, labels_train)

#df_results = display_results(results_svm, results_svm_polykernel, results_linear, results_qda, results_rf)

#print(df_results)
# %%
