'''x'''

def balance_data(data):
    data_balanced = data.copy()

    # Check if the amount of 1's and 0's are (approximately) equal
    # (check for data imbalance)
    count_gist = data_balanced['label'].value_counts()[1]
    count_non_gist = data_balanced['label'].value_counts()[0]

    percentage_gist = count_gist / (count_gist+count_non_gist) * 100
    percentage_non_gist = count_non_gist / (count_gist+count_non_gist) * 100

    print(f'There are {count_gist} occurences of GIST in the dataset.')
    print(f'There are {count_non_gist} occurences of non-GIST in the dataset.')
    print(f'The classes GIST and non-GIST are respectively {percentage_gist:.3f}% and {percentage_non_gist:.3f}% of the dataset.')

    return data_balanced