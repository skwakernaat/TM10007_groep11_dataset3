'''This module contains the function final_test which is used to evaluate the performance
    of different classifiers on the test set. It calculates accuracy, sensitivity,
    specificity, and ROC AUC score for each classifier.'''
import pandas as pd

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

    return results

def save_evaluation_results(results: dict, output_path="results/model_performance.csv"):
    """
    Saves classifier evaluation results including hyperparameters to a CSV file.
    """
    import pandas as pd
    import os

    rows = []
    for clf_name, metrics in results.items():
        flat_row = {
            'classifier': clf_name,
            'accuracy': metrics['accuracy'],
            'roc_auc': metrics['roc_auc'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
        }
        # Flatten hyperparameters into one column
        flat_row['hyperparameters'] = str(metrics['hyperparameters'])
        rows.append(flat_row)

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
