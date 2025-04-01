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

data = load_data()

data_cleaned = clean_data(data)

data_balanced = balance_data(data_cleaned)

data_train, data_test, labels_train, labels_test, column_headers = split_data(data_balanced)

data_train_scaled, column_header_nozerovar = scale_data(data_train, column_headers)

sorted_indices, sorted_scores = univariate_feature_selection(data_train_scaled, labels_train)
