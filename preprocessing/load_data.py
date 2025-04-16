'''This module contains the function load_data which is used to load the GIST radiomic
    features dataset from a CSV file.'''

import os
import pandas as pd

def load_data():
    '''This function loads the GIST radiomic features dataset from a CSV file.'''

    # Get the current directory of this file
    this_directory = os.path.dirname(os.path.abspath(__file__))

    # Load the dataset
    data = pd.read_csv(os.path.join(this_directory, 'GIST_radiomicFeatures.csv'), index_col=0)

    return data