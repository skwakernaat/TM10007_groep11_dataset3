import numpy as np
import seaborn
import pandas as pd
import torch
import itertools
import matplotlib.pyplot as plt
import statistics
import math
from worcgist.load_data import load_data
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
from linear_classifiers import linear_classifier

data = load_data()

data_cleaned = clean_data(data)

data_balanced = balance_data(data_cleaned)

data_scaled = scale_data(data_balanced)

data_train, data_test, data_val, labels_train, labels_test, labels_val = split_data(data_scaled)

sorted_indices = univariate_feature_selection(data_train, labels_train)

top_indices = sorted_indices[:2] # Select top amount of features features
data_train = data_train[:, top_indices]
data_val = data_val[:, top_indices]
data_test = data_test[:, top_indices]

linear_classifier(data_train, data_val, labels_train, labels_val)
