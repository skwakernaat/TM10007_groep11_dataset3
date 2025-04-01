'''x'''

def clean_data(data):
    # Make a copy of the dataframe
    data_cleaned = data.copy()

    # Check for the occurence of NaN and null values
    count_nan = data_cleaned.isna().sum().sum()
    count_null = data_cleaned.isnull().sum().sum()

    print(f'There are {count_nan} NaN values and {count_null} null values in the dataframe.')

    # Replace GIST and non-GIST with the values 1 and 0 respectively
    data_cleaned['label'] = data_cleaned['label'].replace('GIST', 1)
    data_cleaned['label'] = data_cleaned['label'].replace('non-GIST', 0)

    return data_cleaned