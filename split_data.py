'''This module contains the function split_data which is used to split the data into a training, validation and test set.'''

from sklearn import model_selection

def split_data(data):
    '''This function splits the data into a training, validation and test set.'''
    # Extract label column from data
    labels = data['label']
    data_nolabels = data.drop(columns=['label'])

    # Convert to array
    labels_array = labels.to_numpy()
    data_array = data_nolabels.to_numpy()

    # Convert to integer
    labels_array = labels_array.astype(int)

    # Splitting data into train and test data sets. 80% is used for training set. Ratio of GIST and non-GIST is kept the same (50/50)
    data_train_val, data_test, labels_train_val, labels_test = model_selection.train_test_split(data_array, labels_array, test_size=0.20, stratify=labels_array)

    # Splitting train data into train and validation data sets. 75% is used for training set. Ratio of GIST and non-GIST is kept the same (50/50)
    data_train, data_val, labels_train, labels_val = model_selection.train_test_split(data_train_val, labels_train_val, test_size=0.25, stratify=labels_train_val)

    return data_train, data_test, data_val, labels_train, labels_test, labels_val
