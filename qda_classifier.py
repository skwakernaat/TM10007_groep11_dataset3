"""In this module the QDA classifier is trained and tested"""
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from evaluate_model_with_kfold import evaluate_model_with_kfold

def qda_func(X, y, n_splits=5):
    """Train and test QDA classifier

    Args:
        data_train_n_features (numpy.ndarray): training data with selected features
        labels_train (numpy.ndarray): training labels
        top_indices (list): indices of the selected features
        data_test (numpy.ndarray): test data
        labels_test (numpy.ndarray): test labels
    Returns:

    """
    # # initialize and train model
    # qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
    # qda.fit(data_train_n_features, labels_train)

    # # predict on the test set
    # predictions = qda.predict(data_test_n_features)

    qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
    results = evaluate_model_with_kfold(X, y, qda, n_splits)

    return results
