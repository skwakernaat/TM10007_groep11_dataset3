'''This module contains the function split_data which is used to split the data into a training and test set.'''

from sklearn.model_selection import train_test_split

def split_and_save_gist_dataset(df):
    """
    Splits the GIST dataset into train and test sets with stratification and returns the train and test DataFrames.

    Parameters:
    - df: DataFrame of the GIST dataset.

    Returns:
    - (train_df, test_df): Tuple of train and test DataFrames.
    """

    # Stratified train-test split on the "label" column
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )
    return train_df, test_df
