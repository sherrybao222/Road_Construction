import pandas as pd
import numpy as np

def df_undo_start_end(sc_data_choice_level, condition):
    final_df = pd.DataFrame()

    for sub in range(100):
        dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index_copy"]) # here the index is original index in data_choice_level    
        
        for pzi in np.unique(sc_data_choice_level['puzzleID']):  
            dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        
            
            cum_error_beginning_list = []
            error_beginning_list = []
            error_rate_beginning_list = []
            terminal_beginning_list = []
        
            # last undo index when it is not the start city
            lastUndo_idx = dat_sbj_pzi.query(condition).index 
            # exclude the case when there is no error from the beginning
            for x in lastUndo_idx:
                j = 0
                # start with last undo index, find the first undo index
                while dat_sbj_pzi.loc[x-j,"firstUndo"] == 0:
                    j = j + 1
                # get cumulative error at undo beginning
                cum_error_beginning = dat_sbj_pzi.loc[x-j-1,"allMAS"] - dat_sbj_pzi.loc[x-j-1,"currMas"]
                cum_error_beginning_list.append(cum_error_beginning)
                error_beginning = dat_sbj_pzi.loc[x-j-1,"severityOfErrors"]
                error_beginning_list.append(error_beginning)
                error_rate_beginning = dat_sbj_pzi.loc[x-j-1,"error_rate"]
                error_rate_beginning_list.append(error_rate_beginning)
                terminal_beginning = dat_sbj_pzi.loc[x-j-1,"checkEnd"]
                terminal_beginning_list.append(terminal_beginning)

            # cumulative error at undo terminal
            df_undoTarget = dat_sbj_pzi.loc[lastUndo_idx,:]
            accu_severity_error = list(df_undoTarget['allMAS'] - df_undoTarget['currMas'])
            # error_rate_end = list(df_undoTarget['error_rate'])

            # cumulative error at the state before undo terminal
            df_undoTarget_before = dat_sbj_pzi.loc[lastUndo_idx-1,:]
            accu_severity_error_before = list(df_undoTarget_before['allMAS'] - df_undoTarget_before['currMas'])
            category = [np.nan]*len(accu_severity_error) # the number of undo terminal
            for i in range(len(accu_severity_error)): 
                if accu_severity_error_before[i]==0: # undo too much
                    category[i] = 0
                elif (accu_severity_error[i]==0)&(accu_severity_error_before[i]>0): # undo exactly the right amount
                    category[i] = 1
                elif accu_severity_error[i] > 0: # undo too little
                    category[i] = 2

            sequential_single = (dat_sbj_pzi.loc[lastUndo_idx,"firstUndo"] == 1)&(dat_sbj_pzi.loc[lastUndo_idx,"lastUndo"] == 1)
            
            # use each value in cum_error_beginning_list to subtract accu_severity_error
            mas_gain = []
            cum_error_end_list = []
            # error_rate_end_list = []
            for i in range(len(cum_error_beginning_list)):
                mas_gain.append(cum_error_beginning_list[i] - accu_severity_error[i])
                cum_error_end_list.append(accu_severity_error[i])
                # error_rate_end_list.append(error_rate_end[i])
            
            # if category is not empty, add it to an empty dataframe
            if len(category) > 0:
                final_df = pd.concat([final_df,pd.DataFrame({'subjects':sub,'puzzleID':pzi,
                                                             'cum_error_beginning':cum_error_beginning_list, 'cum_error_end':cum_error_end_list,
                                                             'mas_gain':mas_gain,
                                                             'error_beginning':error_beginning_list,
                                                            #  'error_rate_beginning':error_rate_beginning_list,
                                                             'sequential_single':sequential_single,
                                                             'terminal_beginning':terminal_beginning_list,
                                                             'category':category})])
    return final_df