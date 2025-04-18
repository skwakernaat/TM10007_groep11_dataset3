'''This module contains the function balance_data which is used to balance the data by
    checking if the amount of 1's and 0's are (approximately) equal (check for data imbalance).'''

import numpy as np

def check_balance(y_train):
    '''This function checks if the amount of 1's and 0's are (approximately) equal
        (check for data imbalance).'''

    y_train_balanced = y_train.copy()

    counts = np.bincount(y_train_balanced)
    count_non_gist = counts[0]
    count_gist = counts[1]

    percentage_gist = count_gist / (count_gist+count_non_gist) * 100
    percentage_non_gist = count_non_gist / (count_gist+count_non_gist) * 100

    print(f'\nThere are {count_gist} occurences of GIST in the dataset.')
    print(f'There are {count_non_gist} occurences of non-GIST in the dataset.')
    print(f'The classes GIST and non-GIST are respectively {percentage_gist:.3f}% '
          f'and {percentage_non_gist:.3f}% of the dataset.\n')
