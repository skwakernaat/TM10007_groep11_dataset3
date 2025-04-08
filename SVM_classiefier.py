import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel, sigmoid_kernel
from sklearn.model_selection import cross_val_score
from evaluate_model_with_kfold import evaluate_model_with_kfold



def pca_selection(data_selected_uni, n_features=2):
    '''
    Load the sklearn breast data set, but reduce the number of features with PCA.
    '''
    x = data_selected_uni

    pca = PCA(n_components=n_features)
    x_pca = pca.fit_transform(x)
    return x_pca

def svm_classifier(data_selected_uni, labels_train):

    # Construct classifiers
    svmlin = SVC(kernel='linear', gamma='scale')
    svmrbf = SVC(kernel='rbf', gamma='scale')
    svmpoly = SVC(kernel='poly', degree=3, gamma='scale')

    clsfs = [KNeighborsClassifier(), RandomForestClassifier(), svmlin, svmpoly, svmrbf]
    clsfnames = ['KNN', 'Random Forest', 'SVM Linear', 'SVM Poly', 'SVM RBF']

    for clsf, clsfn in zip(clsfs, clsfnames):
        results = []
        results = []

        print(f"Training {clsfn}...")

        # Fit the classifier to the training data
        # Note: Ensure that data_selected_uni and labels_train are numpy arrays or similar structures
        clsf.fit(data_selected_uni, labels_train)

        result = evaluate_model_with_kfold(data_selected_uni, labels_train, model=clsf, n_splits=5)
        results.append(result)
    return result




def svm_poly_kernel(data_selected_uni, labels_train):
    '''
    SVM with polynomial kernel
    '''

    degrees = [1, 3, 5]
    coef0s = [0.01, 0.5, 1]
    slacks = [0.01, 0.5, 1]

    clsfs = list()
    for degree in degrees:
        for coef0 in coef0s:
            for slack in slacks:
                clsfs.append(SVC(kernel='poly', degree=degree, coef0=coef0, C=slack, gamma='scale'))

    for clsf in clsfs:
        print(f"Training {clsf}...")
        result = []
        results = []

        # Fit the classifier to the training data
        clsf.fit(data_selected_uni, labels_train)

        result = evaluate_model_with_kfold(data_selected_uni, labels_train, model=clsf, n_splits=5)
        results.append(result)
    return result
