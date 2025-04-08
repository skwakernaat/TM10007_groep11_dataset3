'''This module contains the function linear_classifier which is used to train a linear classifier
    on the data and evaluate its performance.'''

from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from evaluate_model_with_kfold import evaluate_model_with_kfold

def linear_classifier(data_train, labels_train, n_folds=5):
    '''This function trains a linear classifier on the data and evaluates its performance.'''
    # Define classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "LDA": LinearDiscriminantAnalysis(),
        "Linear SVM": SVC(kernel='linear'),
        "SGD Classifier": SGDClassifier(loss='log_loss', max_iter=1000)
    }

    results_list = []

    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        result = evaluate_model_with_kfold(data_train, labels_train, clf, n_folds)
        results_list.append(result)

    return results_list
