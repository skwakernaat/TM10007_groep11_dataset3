'''x'''

import numpy as np
from sklearn import model_selection

def split_data(data):
    # Extract label column from data_cleaned
    labels = data['label']
    data_cleaned_nolabel = data.drop(columns=['label'])

    # Extract column headers
    column_headers = np.array(list(data.columns.values))
    column_headers = column_headers[1:]

    # Convert to array
    labels_array = labels.to_numpy()
    data_array = data_cleaned_nolabel.to_numpy()

    # Splitting data into train and test data sets. 80% is used for training set. Ratio of GIST and non-GIST is kept the same (50/50)
    data_train, data_test, labels_train, labels_test = model_selection.train_test_split(data_array, labels_array, test_size=0.20, stratify=labels_array)

    return data_train, data_test, labels_train, labels_test, column_headers
