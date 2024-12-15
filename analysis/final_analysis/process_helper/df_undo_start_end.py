import pandas as pd
import numpy as np

def df_undo_start_end(sc_data_choice_level, condition):
    final_df = pd.DataFrame()

    for sub in range(100):
        dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index_copy"]) # here the index is original index in data_choice_level    
        
        for pzi in np.unique(sc_data_choice_level['puzzleID']):  
            dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        
            
            n_connect_beginning_list = []
            cum_error_beginning_list = []
            error_beginning_list = []
            # error_rate_beginning_list = []
            terminal_beginning_list = []
            undo_length_list = []

            # last undo index when it is not the start city
            lastUndo_idx = dat_sbj_pzi.query(condition).index 
            # exclude the case when there is no error from the beginning
            for x in lastUndo_idx:
                j = 0
                # start with last undo index, find the first undo index
                while dat_sbj_pzi.loc[x-j,"firstUndo"] == 0:
                    j = j + 1
                # get cumulative error at undo beginning
                n_connect_beginning = dat_sbj_pzi.loc[x-j-1,"currNumCities"]
                n_connect_beginning_list.append(n_connect_beginning)
                cum_error_beginning = dat_sbj_pzi.loc[x-j-1,"allMAS"] - dat_sbj_pzi.loc[x-j-1,"currMas"]
                cum_error_beginning_list.append(cum_error_beginning)
                error_beginning = dat_sbj_pzi.loc[x-j-1,"severityOfErrors"]
                error_beginning_list.append(error_beginning)
                # error_rate_beginning = dat_sbj_pzi.loc[x-j-1,"error_rate"]
                # error_rate_beginning_list.append(error_rate_beginning)
                terminal_beginning = dat_sbj_pzi.loc[x-j-1,"checkEnd"]
                terminal_beginning_list.append(terminal_beginning)
                undo_length = j+1
                undo_length_list.append(undo_length)

            # cumulative error at undo terminal
            df_undoTarget = dat_sbj_pzi.loc[lastUndo_idx,:]
            cum_error_end_list = list(df_undoTarget['allMAS'] - df_undoTarget['currMas'])
            n_connect_end_list = list(df_undoTarget['currNumCities'])
            # error_rate_end = list(df_undoTarget['error_rate'])

            # cumulative error at the state before undo terminal
            df_undoTarget_before = dat_sbj_pzi.loc[lastUndo_idx-1,:]
            accu_severity_error_before = list(df_undoTarget_before['allMAS'] - df_undoTarget_before['currMas'])
            
            # categorize undo too much, on-target, or too little
            category = [np.nan]*len(cum_error_end_list) # the number of undo terminal
            for i in range(len(cum_error_end_list)): 
                if accu_severity_error_before[i]==0: # undo too much
                    category[i] = 0
                elif (cum_error_end_list[i]==0)&(accu_severity_error_before[i]>0): # undo exactly the right amount
                    category[i] = 1
                elif cum_error_end_list[i] > 0: # undo too little
                    category[i] = 2

            sequential_single = (dat_sbj_pzi.loc[lastUndo_idx,"firstUndo"] == 1)&(dat_sbj_pzi.loc[lastUndo_idx,"lastUndo"] == 1)
            
            # get mas gain
            mas_gain = []
            for i in range(len(cum_error_beginning_list)):
                mas_gain.append(cum_error_beginning_list[i] - cum_error_end_list[i])
            
            # if category is not empty, add it to an empty dataframe
            if len(category) > 0:
                final_df = pd.concat([final_df,pd.DataFrame({'subjects':sub,'puzzleID':pzi,
                                                             'cum_error_beginning':cum_error_beginning_list, 'cum_error_end':cum_error_end_list,
                                                             'n_connect_beginning':n_connect_beginning_list, 'n_connect_end':n_connect_end_list,
                                                             'undo_length': undo_length_list,
                                                             'mas_gain':mas_gain,
                                                             'error_beginning':error_beginning_list,
                                                            #  'error_rate_beginning':error_rate_beginning_list,
                                                             'sequential_single':sequential_single,
                                                             'terminal_beginning':terminal_beginning_list,
                                                             'category':category})])
    return final_df