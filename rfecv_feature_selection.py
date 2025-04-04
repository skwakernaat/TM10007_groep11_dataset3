from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import feature_selection
from sklearn import model_selection
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.feature_selection import RFECV



def rfecv_features(data_selected_uni, labels_train):
    '''
    Recursive feature elimination with cross-validation.
    '''

    # Create the RFE object and compute a cross-validated score.
    #SVC will be used to evaluate feature importance based on classification performance.
    svc = svm.SVC(kernel="linear")

    # classifications
    rfecv = feature_selection.RFECV(
        estimator=svc, step=1, #how many features to remove at each iteration
        cv=model_selection.StratifiedKFold(4), #cross-validation strategy.
        scoring='roc_auc') #metric to evaluate model performance
    rfecv.fit(data_selected_uni, labels_train)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])
    plt.show()

    optimal_features = rfecv.n_features_
    print(f"Optimal number of features: {optimal_features}")