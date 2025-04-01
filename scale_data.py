'''x'''

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_data(data, column_headers):
    # Drop columns (features) that only contain the same value (variance = 0)
    var_features = np.var(data, axis=0) # calculate variance per column/feature
    var_thresh = 1e-25 # value is chosen because the ANOVA test rounds this to zero and cannot work otherwise

    for idx, var in enumerate(var_features):
        if var <= var_thresh:
            var_features[idx] = 0
        else:
            var_features[idx] = 1

    # Make a new matrix without zero variance features
    data_train_nozerovar = data[:, var_features == 1]

    column_header_nozerovar = column_headers[var_features == 1]

    #----------------------Scaling-----------------------

    # Kies de schaalmethode hier: 'standard', 'minmax', of 'robust'
    scaling_method = 'standard'  # Verander naar 'minmax' als je MinMaxScaler wilt gebruiken

    # Schalen van de volledige dataset (alle features)
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Ongeldige keuze! Kies 'standard', 'minmax' of 'robust'.")

    # Pas de scaler toe op de volledige dataset
    x_train_scaled = scaler.fit_transform(data_train_nozerovar)  # Alle features worden hier geschaald

    return x_train_scaled, column_header_nozerovar
