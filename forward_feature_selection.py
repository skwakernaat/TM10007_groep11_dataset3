from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import LabelEncoder

def select_top_features(train_df, n_features=12):
    """
    Use sklearn's SequentialFeatureSelector to select the top n informative features.

    Parameters:
    - train_df: DataFrame including features, 'label', and 'ID'.
    - n_features: Number of features to select.

    Returns:
    - DataFrame with selected features + 'label' column.
    """
    # Prepare features and label
    X = train_df.drop(columns=["ID", "label"])
    y = LabelEncoder().fit_transform(train_df["label"])

    # Create model and feature selector
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction='forward',
        scoring='accuracy',
        cv=5,
        n_jobs=-1
    )

    selector.fit(X, y)

    # Get selected feature names
    selected_columns = X.columns[selector.get_support()].tolist()

    # Return the reduced dataset with label
    reduced_df = train_df[selected_columns + ['label']].copy()
    return reduced_df
