
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
from SVM_classiefier import pca_selection
from SVM_classiefier import svm_classifier
from SVM_classiefier import svm_poly_kernel
from rfecv_feature_selection import rfecv_features


from clean_data import clean_data
from balance_data import balance_data
from split_data import split_data
from scale_data import scale_data
from univariate_feature_selection import univariate_feature_selection
from linear_classifiers import linear_classifier
from qda_classifier import qda_func
from random_forest import random_forest_classifier

#%%
data = load_data()

data_cleaned = clean_data(data)

data_balanced = balance_data(data_cleaned)

data_scaled = scale_data(data_balanced)

data_train, data_test, labels_train, labels_test = split_data(data_scaled)

#%%
# Feature selection
data_train_n_features, sorted_scores = univariate_feature_selection(
                                                            data_train, labels_train, n_features=30)
rfecv_features(data_train_n_features, labels_train)

#%% multiple svm classifiers with different kernels
svm_classifier(data_train_n_features, labels_train)

svm_poly_kernel(data_train_n_features, labels_train)

linear_classifier(data_train_n_features, data_train_n_features, labels_train, labels_train)

qda_predictions = qda_func(data_train_n_features, labels_train, data_train_n_features, labels_train)

a, b = random_forest_classifier(data_train, labels_train, data_test, labels_test, n_estimators=100, random_state=42, max_features_to_test=50, plot=True)
