'''This module contains the function linear_classifier which is used to train a linear classifier
    on the data and evaluate its performance.'''

from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

def linear_classifier(data_train, data_val, labels_train, labels_val):
    '''This function trains a linear classifier on the data and evaluates its performance.'''
    # Define classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "LDA": LinearDiscriminantAnalysis(),
        "Linear SVM": SVC(kernel='linear'),
        "SGD Classifier": SGDClassifier(loss='log_loss', max_iter=1000)
}

    for name, clf in classifiers.items():
        clf.fit(data_train, labels_train)
        y_pred = clf.predict(data_val)
        acc = accuracy_score(labels_val, y_pred)
        print(f"\n{name}")
        print("Accuracy:", round(acc, 3))
        print(classification_report(labels_val, y_pred))
