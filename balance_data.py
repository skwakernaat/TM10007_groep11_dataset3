'''This module contains the function balance_data which is used to balance the data by checking if
the amount of 1's and 0's are (approximately) equal (check for data imbalance).'''

import numpy as np

def balance_data(X_train, y_train, X_test,):
    '''This function checks if the amount of 1's and 0's are (approximately) equal
        (check for data imbalance).'''

    X_train_balanced = X_train.copy()
    X_test_balanced = X_test.copy()

    # Check if the amount of 1's and 0's are (approximately) equal
    # (check for data imbalance)
    counts = np.bincount(y_train)
    count_gist = counts[1]
    count_non_gist = counts[0]

    percentage_gist = count_gist / (count_gist+count_non_gist) * 100
    percentage_non_gist = count_non_gist / (count_gist+count_non_gist) * 100

    print(f'\nThere are {count_gist} occurences of GIST in the training dataset.')
    print(f'There are {count_non_gist} occurences of non-GIST in the training dataset.')
    print(f'The classes GIST and non-GIST are respectively {percentage_gist:.3f}% and'
          f'{percentage_non_gist:.3f}% of the training dataset.\n')

    return X_train_balanced, X_test_balanced