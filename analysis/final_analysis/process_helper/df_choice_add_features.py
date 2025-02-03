import pandas as pd
import numpy as np

def df_add_first_move(data_choice_level):

    new_data_choice_level = pd.DataFrame()
    for sub in range(100):
        dat_sbj  = data_choice_level[data_choice_level['subjects']==sub]
        
        for pzi in np.unique(data_choice_level['puzzleID']):
            
            dat_sbj_pzi_basic = dat_sbj[(dat_sbj['puzzleID'] == pzi) & (dat_sbj['condition'] == 0)]   
            dat_sbj_pzi_undo = dat_sbj[(dat_sbj['puzzleID'] == pzi) & (dat_sbj['condition'] == 1)]   

            dat_sbj_pzi_basic.loc[:, 'first_move'] = (dat_sbj_pzi_basic.index == dat_sbj_pzi_basic.index[1])
            dat_sbj_pzi_undo.loc[:, 'first_move'] = (dat_sbj_pzi_undo.index == dat_sbj_pzi_undo.index[1])

            new_data_choice_level = pd.concat([new_data_choice_level, dat_sbj_pzi_basic])
            new_data_choice_level = pd.concat([new_data_choice_level, dat_sbj_pzi_undo])

    return new_data_choice_level

def df_choice_add_features(data_choice_level):
    data_choice_level = df_add_first_move(data_choice_level)
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