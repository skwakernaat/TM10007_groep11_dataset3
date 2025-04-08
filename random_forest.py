from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
from evaluate_model_with_kfold import evaluate_model_with_kfold  # Importeer de functie

def random_forest_classifier(data_train, labels_train, data_test, labels_test, n_estimators=100, random_state=42, plot=True):
    '''Deze functie traint een Random Forest model en evalueert de prestaties met k-fold cross-validation via evaluate_model_with_kfold.'''

    # Definieer het Random Forest model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # Voer k-fold cross-validation uit om het model te evalueren
    kfold_results = evaluate_model_with_kfold(data_train, labels_train, model=model, n_splits=5)

    # Print de gemiddelde prestaties
    print(f"Cross-validation Results:")
    print(f"Mean Accuracy: {kfold_results['mean_accuracy']:.4f}")
    print(f"Mean Sensitivity (Recall): {kfold_results['mean_sensitivity']:.4f}")
    print(f"Mean Specificity: {kfold_results['mean_specificity']:.4f}")
    print(f"Average Confusion Matrix:\n{kfold_results['mean_confusion_matrix']}")

    # Train het model met de volledige trainingsdata
    model.fit(data_train, labels_train)

    # Voer de voorspellingen uit op de testset
    predictions = model.predict(data_test)

    # Bereken de prestaties op de testset
    accuracy = accuracy_score(labels_test, predictions)
    sensitivity = np.sum((predictions == 1) & (labels_test == 1)) / np.sum(labels_test == 1)
    specificity = np.sum((predictions == 0) & (labels_test == 0)) / np.sum(labels_test == 0)

    print(f"Testset Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    return accuracy, sensitivity, specificity, model
