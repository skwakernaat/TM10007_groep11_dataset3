import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression  # You can swap in your own model

def evaluate_model_with_kfold(X, y, model=None, n_splits=5):
    if model is None:
        model = LogisticRegression(max_iter=1000)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies = []
    sensitivities = []
    specificities = []
    all_conf_matrices = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Confusion matrix structure: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (Recall)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity

        accuracies.append(acc)
        sensitivities.append(sens)
        specificities.append(spec)
        all_conf_matrices.append(cm)

    # Aggregate results
    print(f"Model:    {model}")
    print(f"\nStratified {n_splits}-Fold Cross-Validation Results:")
    print(f"Mean Accuracy:    {np.mean(accuracies):.3f}")
    print(f"Mean Sensitivity: {np.mean(sensitivities):.3f}")
    print(f"Mean Specificity: {np.mean(specificities):.3f}")

    return {
        'model': model,
        'accuracies': accuracies,
        'sensitivities': sensitivities,
        'specificities': specificities,
        'confusion_matrices': all_conf_matrices
    }