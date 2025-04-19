'''This module contains the function final_test which is used to evaluate the performance
    of different classifiers on the test set. It calculates accuracy, sensitivity,
    specificity, and ROC AUC score for each classifier.'''

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_model(X_test, y_test, clfs):
    '''This function evaluates the performance of different classifiers on the test set.
        It calculates accuracy, sensitivity, specificity, and ROC AUC score for each classifier.'''

    results = {}

    for clf in clfs:
        # Get the classifier name
        clf_name = clf.__class__.__name__

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Calculate scores
        acc = accuracy_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

        results[clf_name] = {
        'accuracy': acc,
        'roc_auc': auc,
        'sensitivity': sens,
        'specificity': spec
        }

    for clf_name, scores in results.items():
        print(f"Results for {clf_name}:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")

    return results