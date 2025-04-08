import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression  # You can swap in your own model

def evaluate_model_with_kfold(X, y, model=None, n_splits=5):
    if model is None:
        model = LogisticRegression(max_iter=1000)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_list, sens_list, spec_list = [], [], []
    cm_total = np.zeros((2, 2), dtype=float)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        acc_list.append(acc)
        sens_list.append(sens)
        spec_list.append(spec)
        cm_total += cm

    # Average confusion matrix
    mean_cm = cm_total / n_splits

    return {
        'model': model,
        'mean_accuracy': np.mean(acc_list),
        'mean_sensitivity': np.mean(sens_list),
        'mean_specificity': np.mean(spec_list),
        'mean_confusion_matrix': mean_cm
    }
