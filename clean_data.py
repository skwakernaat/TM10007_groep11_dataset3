'''This module contains the function clean_data which is used to clean the data by removing unnecessary columns and replacing GIST and non-GIST with 1 and 0 respectively.'''

import pandas as pd

def clean_data(data):
    '''This function cleans the data by removing unnecessary columns and replacing GIST and non-GIST with 1 and 0 respectively.'''
    # Make a copy of the dataframe
    data_cleaned = data.copy()

    # Check for the occurence of NaN and null values
    count_nan = data_cleaned.isna().sum().sum()
    count_null = data_cleaned.isnull().sum().sum()

    print(f'\nThere are {count_nan} NaN values and {count_null} null values in the dataframe.\n')

    # Replace GIST and non-GIST with the values 1 and 0 respectively
    pd.set_option('future.no_silent_downcasting', True) # to remove the FutureWarning
    data_cleaned['label'] = data_cleaned['label'].replace('GIST', 1)
    data_cleaned['label'] = data_cleaned['label'].replace('non-GIST', 0)

    return data_cleaned