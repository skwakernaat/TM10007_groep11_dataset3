from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
from evaluate_model_with_kfold import evaluate_model_with_kfold  # Importeer de functie

def random_forest_classifier(data_train, labels_train, data_test, labels_test, n_estimators=100, random_state=42, plot=True):
    '''Deze functie traint een Random Forest model en evalueert de prestaties met k-fold cross-validation via evaluate_model_with_kfold.'''

    # Definieer het Random Forest model
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # Voer k-fold cross-validation uit om het model te evalueren
    kfold_results = evaluate_model_with_kfold(data_train, labels_train, model=clf, n_splits=5)

    return kfold_results
