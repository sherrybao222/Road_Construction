import pandas as pd
import numpy as np

def df_choice_add_features(data_choice_level):
    data_choice_level["checkEnd"] = pd.to_numeric(data_choice_level["checkEnd"]) # Convert checkEnd to integer
    data_choice_level['currNumCities'] = data_choice_level.currNumCities - 1 # starting from 0
    data_choice_level['allMAS'] = data_choice_level.allMAS - 1
    data_choice_level['currMas'] = data_choice_level.currMas - 1
    data_choice_level['N_more'] = data_choice_level["currMas"] - data_choice_level["currNumCities"]
    data_choice_level['scaled_N_more'] = data_choice_level.N_more/data_choice_level.currMas

    bins = [np.nextafter(0, -1), np.nextafter(0, 1)] + list(np.linspace(1/9, 8/9, 8)) + [np.nextafter(1, 0), np.nextafter(1, 1)]
    labels = ['0'] + [f'0.{i}' for i in range(1, 10)] + ['1']
    # Bin the data
    data_choice_level['scaled_N_more_bin'], cutoff_nmore = pd.cut(data_choice_level['scaled_N_more'], bins=bins, labels=labels, include_lowest=True, retbins=True)
    data_choice_level['efficiency'] = (300-data_choice_level['leftover'])/data_choice_level['currNumCities']
    data_choice_level['budget_change'] = abs(data_choice_level['budget_change'].values)
    data_choice_level['within_reach_change'] = abs(data_choice_level['within_reach_change'].values)

    return data_choice_level