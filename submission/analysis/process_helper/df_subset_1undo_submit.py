import numpy as np
def df_subset_1undo_submit(sc_data_choice_level):
    """
    select the data before the very first undo or submit.
    """
    data_subset_before1undo_index = []

    for sub in np.unique(sc_data_choice_level['subjects']):
        dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub]
        
        for pzi in np.unique(sc_data_choice_level['puzzleID']):
            
            dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi]     

            for i in range(len(dat_sbj_pzi)):
                if (dat_sbj_pzi.iloc[i]['firstUndo'] != 1)&(dat_sbj_pzi.iloc[i]['submit'] != 1):
                    data_subset_before1undo_index.append(dat_sbj_pzi.index[i])
                elif (dat_sbj_pzi.iloc[i]['firstUndo'] == 1)|(dat_sbj_pzi.iloc[i]['submit'] == 1):
                    data_subset_before1undo_index.append(dat_sbj_pzi.index[i])
                    break

    data_subset_before1undo = sc_data_choice_level.loc[data_subset_before1undo_index,:]
    return data_subset_before1undo