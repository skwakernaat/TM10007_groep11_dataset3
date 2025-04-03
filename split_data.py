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

    # Splitting data into train and test data sets. 80% is used for training set. Ratio of GIST and non-GIST is kept the same (50/50)
    data_train, data_test, labels_train, labels_test = model_selection.train_test_split(data_array, labels_array, test_size=0.20, stratify=labels_array)

    return data_train, data_test, labels_train, labels_test
