'''This module contains the function select_top_features which is used to select the top n informative
    features using sklearn's SequentialFeatureSelector. It takes the training data and the number of
    features to select as input and returns the reduced dataset with the label.'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector

def forward_feature_selection(X_train, y_train, X_test, feature_names, n_features):
    '''Use sklearn's SequentialFeatureSelector to select the top n informative features.'''

    # Create model and feature selector
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction='forward',
        scoring='roc_auc',
        cv=3,
        n_jobs=-1
    )

    X_train_features = selector.fit_transform(X_train, y_train)
    X_test_features = selector.transform(X_test)

    # Get selected feature names
    selected_mask = selector.get_support()  # Boolean mask of selected features
    selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
    print(f"Selected features: {selected_features}")

    return X_train_features, X_test_features
