import pandas as pd
import numpy as np

def df_do_redo_pair(sc_data_choice_level, condition):
    final_df = pd.DataFrame()

    for sub in range(100):
        dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index_copy"]) # here the index is original index in data_choice_level    
        
        for pzi in np.unique(sc_data_choice_level['puzzleID']):  
            dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        
            
            error_do_seq_list = []
            error_redo_seq_list = []
            error_do_move_list = []
            error_redo_move_list = [] 
            n_connect_do_seq_list = []
            n_connect_redo_seq_list = []
            category_list = []

            # last undo index when it is not the start city
            lastUndo_idx = dat_sbj_pzi.query(condition).index 
            # exclude the case when there is no error from the beginning
            for x in lastUndo_idx:
                # start with last undo index, find the first undo index
                j = 0
                while dat_sbj_pzi.loc[x-j,"firstUndo"] == 0:
                    j = j + 1
                # start with last undo index, find the next undo index or checkEnd
                k = 1
                while (dat_sbj_pzi.loc[x+k,"firstUndo"] == 0) and (dat_sbj_pzi.loc[x+k,"submit"] == 0):
                    k = k + 1
                # get cumulative error at undo beginning
                n_connect_do_seq = dat_sbj_pzi.loc[x-j-1,"currNumCities"]
                n_connect_do_seq_list.append(n_connect_do_seq)
                n_connect_redo_seq = dat_sbj_pzi.loc[x+k-1,"currNumCities"]
                n_connect_redo_seq_list.append(n_connect_redo_seq)
                min_n_connect = min(n_connect_do_seq, n_connect_redo_seq)

                error_do_seq = dat_sbj_pzi.loc[x-(j-n_connect_do_seq+min_n_connect)-1,"allMAS"] - dat_sbj_pzi.loc[x-(j-n_connect_do_seq+min_n_connect)-1,"currMas"]
                error_do_seq_list.append(error_do_seq)
                error_redo_seq = dat_sbj_pzi.loc[x+(k-n_connect_redo_seq+min_n_connect)-1,"allMAS"] - dat_sbj_pzi.loc[x+(k-n_connect_redo_seq+min_n_connect)-1,"currMas"]
                error_redo_seq_list.append(error_redo_seq)
                
                error_do_move = dat_sbj_pzi.loc[x-1,"allMAS"] - dat_sbj_pzi.loc[x-1,"currMas"]
                error_do_move_list.append(error_do_move)
                error_redo_move = dat_sbj_pzi.loc[x+1,"allMAS"] - dat_sbj_pzi.loc[x+1,"currMas"]
                error_redo_move_list.append(error_redo_move)
                choice_do = dat_sbj_pzi.loc[x-1,"choice"]
                choice_redo = dat_sbj_pzi.loc[x+1,"choice"]

                if choice_do == choice_redo: # redo and do same
                    category_list.append(0) # identical
                elif error_do_move == error_redo_move: # redo and do better
                    category_list.append(1) # same
                elif error_do_move > error_redo_move: # redo and do worse
                    category_list.append(2) # redo better
                else:
                    category_list.append(3) # do better


            
            # if category is not empty, add it to an empty dataframe
            final_df = pd.concat([final_df,pd.DataFrame({'subjects':sub,'puzzleID':pzi,
                                                             'error_do_seq':error_do_seq_list, 'error_do_move':error_do_move_list,
                                                             'error_redo_seq':error_redo_seq_list, 'error_redo_move':error_redo_move_list,
                                                             'n_connect_do_seq':n_connect_do_seq_list, 'n_connect_redo_seq':n_connect_redo_seq_list,
                                                             'category':category_list})])
    return final_df