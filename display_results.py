'''This module provides a function to display results from model evaluations in a
    tabular format. It combines results from different models, flattens them into a
    DataFrame, and sorts them based on sensitivity. The function can handle both
    single dictionaries and lists of dictionaries as input.'''

import pandas as pd

def display_results(*results_lists):
    '''Combines and flattens dictionaries of results (single or in lists) into a
        sorted DataFrame.'''
    all_results = []

    # Flatten and normalize input
    for item in results_lists:
        if isinstance(item, dict):
            all_results.append(item)
        elif isinstance(item, list):
            for sub_item in item:
                if isinstance(sub_item, dict):
                    all_results.append(sub_item)
                else:
                    raise TypeError(f"Expected dict inside list, got {type(sub_item)}")
        else:
            raise TypeError(f"Expected dict or list of dicts, got {type(item)}")

    # Convert each result to a flat dict with extracted info
    flat_results = []
    for entry in all_results:
        model = entry['model']
        model_name = type(model).__name__

        try:
            params = model.get_params()
        except:
            params = {}

        cm = entry['mean_confusion_matrix']
        tn, fp, fn, tp = cm.ravel()

        flat_results.append({
            'model': model_name,
            'mean_accuracy': entry['mean_accuracy'],
            'mean_sensitivity': entry['mean_sensitivity'],
            'mean_specificity': entry['mean_specificity'],
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'C': params.get('C', None),
            'kernel': params.get('kernel', None),
            'degree': params.get('degree', None),
            'coef0': params.get('coef0', None)
        })

    df = pd.DataFrame(flat_results)
    df_sorted = df.sort_values(by='mean_sensitivity', ascending=False)

    return df_sorted
