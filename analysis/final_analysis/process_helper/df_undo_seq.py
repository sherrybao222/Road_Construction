import pandas as pd
import numpy as np
import ast

def df_undo_seq(sc_data_choice_level, condition):
    final_df = pd.DataFrame()
    sc_data_choice_level['path'] = sc_data_choice_level['path'].apply(ast.literal_eval)

    for sub in range(100):
        dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index_copy"]) # here the index is original index in data_choice_level    
        
        for pzi in np.unique(sc_data_choice_level['puzzleID']):  
            dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        

            n_connect_beginning_list = []
            path_list = []  
            currMAS_list = []

            firstUndo_idx = dat_sbj_pzi.query(condition).index 
            for x in firstUndo_idx:
                n_connect_beginning = dat_sbj_pzi.loc[x-1,"currNumCities"]
                n_connect_beginning_list.append(n_connect_beginning)

                path = dat_sbj_pzi.loc[x-1,"path"]
                path_list.append(path)

                currMAS_sublist = []
                for i in range(len(path)): # go back to each possible loc in undo sequence
                    choice = path[-i-1]
                    if i < len(path)-1:
                        choice_before = path[-i-2]
                        ind = dat_sbj_pzi[dat_sbj_pzi.choice == choice].index
                        ind_before = dat_sbj_pzi[dat_sbj_pzi.choice == choice_before].index 

                        for j in range(len(ind)):
                            for k in range(len(ind_before)):
                                if ind[j]-1 == ind_before[k]:
                                    currMAS = dat_sbj_pzi.loc[ind[j],"currMas"]
                                    
                    else:
                        currMAS = dat_sbj_pzi.loc[0,"currMas"]

                    currMAS_sublist.insert(0, currMAS) 

                currMAS_list.append(currMAS_sublist)
                
                
                                
            # if category is not empty, add it to an empty dataframe
            final_df = pd.concat([final_df,pd.DataFrame({'subjects':sub,'puzzleID':pzi,
                                                         'n_connect_beginning': n_connect_beginning_list,
                                                         'path_list': path_list,
                                                         'currMAS_list': currMAS_list})])
    return final_df