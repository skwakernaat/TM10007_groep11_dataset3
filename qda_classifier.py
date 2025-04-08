"""In this module the QDA classifier is trained and tested"""
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def qda_func(data_train_n_features, labels_train, data_test_n_features, labels_test):
    """Train and test QDA classifier

    Args:
        data_train_n_features (numpy.ndarray): training data with selected features
        labels_train (numpy.ndarray): training labels
        top_indices (list): indices of the selected features
        data_test (numpy.ndarray): test data
        labels_test (numpy.ndarray): test labels
    Returns:
        
    """
    # initialize and train model
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(data_train_n_features, labels_train)

    # predict on the test set
    predictions = qda.predict(data_test_n_features)

    # evalutate performance
    print("Accuracy:", accuracy_score(labels_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(labels_test, predictions))
    print("Classification Report:\n", classification_report(labels_test, predictions))

    return predictions
