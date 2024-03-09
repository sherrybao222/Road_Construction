import numpy as np
import os
from glob import glob
from numpy import genfromtxt
import pandas as pd

# import data
data_all = []

# directories
# home_dir = 'G:\My Drive\\researches\\nyu\\road-construction-local-dk\data_online_2022/'
# map_dir = 'active_map/'
# data_dir  = 'data/preprocessed'
home_dir = '/Users/dbao/My_Drive'+'/road_construction/data/2022_online/'
map_dir = 'active_map/'
data_dir  = 'data/preprocessed'
R_out_dir = home_dir+'R_analysis_data/choice_level/'


flist = glob(home_dir + data_dir + '/preprocess4_sub_*.csv')
# import experiment data
for fname in flist:
    with open(fname, 'r') as f:
        all_data = pd.read_csv(f)
        data_all.append(all_data)


## ==========================================================================
# DATA SAVE FOR THE FURTHER ANALYSIS USING R
subjects = []            # subject id
puzzleID = []            # puzzle id
trialID = []             # trial ID
allMAS = []

undo_c = []              # undo condition binary
undo = []                # undo binary
firstUndo = []           # first undo in a sequence of undo
lastUndo  = []            # binary: new path start, where undo goes back to, (excluding the first starting city), last undo

submit = []
checkEnd = []

severityOfErrors = []    # severity of errors
error = []               # error binary
budget_change = []       # budget change
within_reach_change = [] # ncities within reach change
missed_reward = []       # missed reward
error_rate = []          # error rate

path = [] 
choice = []
currNumCities = []       # current number of connected cities
reward = []              # reward
currMas = []             # current MAS
currNos = []             # current number of optimal solutions
leftover = []            # budget left
within_reach = []        # ncities within reach

RT = []                  # reaction time of moves (including undo)
undoRT = []              # can get undoRT using undo binary

tortuosity = []       # tortuosity

for i in range(len(data_all)):
    data_all[i] = data_all[i].replace('undo',1)
    data_all[i] = data_all[i].replace('basic',0)

    # empty list to save per subject
    temp_puzzleID = data_all[i].map_id
    temp_trialID = data_all[i].trial_id
    temp_allMAS = []

    temp_undo = data_all[i].undoIndicator
    temp_undo_c = data_all[i].condition
    temp_firstUndo = []               # first undo in a sequence of undo
    temp_lastUndo = []            # binary: new path start, where undo goes back to, (excluding the first starting city)
    temp_submit = data_all[i].submit
    temp_checkEnd = data_all[i].checkEnd
        
    temp_path = data_all[i].chosen_all
    temp_choice = data_all[i].currentChoice
    temp_currNumCities = data_all[i].n_city_all
    temp_reward = [2*pow(i-1, 2) for i in temp_currNumCities]
    temp_currMas = data_all[i].mas_all
    temp_currNos = data_all[i].n_opt_paths_all
    temp_leftover = data_all[i].currentBudget
    temp_within_reach = data_all[i].n_within_reach

    temp_RT = data_all[i].rt_all # can get undoRT using undo binary

    temp_tortuosity = data_all[i].tortuosity_all

    mas_all_trial = np.array(data_all[i].mas_all)
    errors_trial = np.array([0, *(mas_all_trial[1:] - mas_all_trial[:-1]).tolist()])
    budget_change_trial = np.array([0, *(np.array(data_all[i].currentBudget)[1:] - np.array(data_all[i].currentBudget)[:-1]).tolist()])
    within_reach_change_trial = np.array([0, *(np.array(data_all[i].n_within_reach)[1:] - np.array(data_all[i].n_within_reach)[:-1]).tolist()])

    severe_error_trial = np.zeros(np.array(errors_trial).shape)
    severe_error_trial[errors_trial<0] = errors_trial[errors_trial<0]
    severe_error_trial = np.abs(severe_error_trial).astype(np.int16)
    
    errors_trial = np.zeros(np.array(errors_trial).shape).astype(np.int16)
    errors_trial[severe_error_trial!=0] = 1
    
    this_allMAS = np.nan
    
    for ti in range(data_all[i].shape[0]):
        subjects.append(i)
        if (data_all[i].n_city_all[ti]==1):
            severityOfErrors.append(0)
            error.append(0)
            budget_change.append(0)
            within_reach_change.append(0)
        else:
            severityOfErrors.append(severe_error_trial[ti])
            error.append(errors_trial[ti])
            budget_change.append(budget_change_trial[ti])
            within_reach_change.append(within_reach_change_trial[ti])
        
        temp_firstUndo.append((np.array(data_all[i].undoIndicator)[ti] == 1) & (np.array(data_all[i].rt_all)[ti] != -1) & (np.array(data_all[i].undoIndicator)[ti-1] == 0))         # first undo in a sequence of undo
            
        if (ti != data_all[i].shape[0]-1):
            temp_lastUndo.append((np.array(data_all[i].undoIndicator)[ti] == 1) & (np.array(data_all[i].rt_all)[ti] != -1) & (np.array(data_all[i].undoIndicator)[ti+1] == 0) & (np.array(data_all[i].submit)[ti] == 0))         # binary: new path start, where undo goes back to, (excluding the first starting city)
        else:
            temp_lastUndo.append(0)
            
        if np.array(data_all[i].undoIndicator )[ti]==1: # & (data_all[i].n_city_all != 1)
            undoRT.append(np.array(data_all[i].rt_all)[ti])
        else:
            undoRT.append(np.nan) # if there is no undo -1
            
        if (data_all[i].rt_all[ti] == -1): # which means if the trial has changed / only save once per trial
            this_allMAS = data_all[i].mas_all[ti]
        temp_allMAS.append(this_allMAS)
                
    # subjects.extend(temp_subjects)
    puzzleID.extend(temp_puzzleID)
    trialID.extend(temp_trialID)
    allMAS.extend(temp_allMAS)

    undo_c.extend(temp_undo_c)
    undo.extend(temp_undo)
    firstUndo.extend(temp_firstUndo)        # first undo in a sequence of undo
    lastUndo.extend(temp_lastUndo)           # binary: new path start, where undo goes back to, (excluding the first starting city), last undo
    submit.extend(temp_submit)
    checkEnd.extend(temp_checkEnd)
    
    path.extend(temp_path)
    choice.extend(temp_choice)
    currNumCities.extend(temp_currNumCities)
    reward.extend(temp_reward)
    currMas.extend(temp_currMas)
    currNos.extend(temp_currNos)
    leftover.extend(temp_leftover)
    within_reach.extend(temp_within_reach)

    RT.extend(temp_RT)

    tortuosity.extend(temp_tortuosity)

cumulative_error = np.array(allMAS) - np.array(currMas)
for index, element in enumerate(allMAS):
    missed_reward.append(2*pow(element-1, 2) - 2*pow(currMas[index]-1, 2))
    error_rate.append(cumulative_error[index]/currNumCities[index])

## ==========================================================================
######## all choice-level data in one file
headerList = ['subjects', 'puzzleID','trialID','allMAS',
              'path','choice','currNumCities','currMas','reward',
              'currNos', 'leftover','within_reach',
              'condition','undo','firstUndo','lastUndo',
              'submit','checkEnd',
              'severityOfErrors', 'error', 'budget_change', 'within_reach_change',
              'cumulative_error','missed_reward', 'error_rate', 
              'RT','undoRT','tortuosity']

dataList = [np.array(puzzleID).astype(np.int16), 
            np.array(trialID).astype(np.int16),
            np.array(allMAS).astype(np.int16),
            np.array(path),
            np.array(choice),
            np.array(currNumCities).astype(np.int16), 
            np.array(currMas).astype(np.int16),
            np.array(reward).astype(np.int16),
            
            np.array(currNos).astype(np.int16),
            np.array(leftover),
            np.array(within_reach).astype(np.int16),

            np.array(undo_c).astype(np.int16),
            np.array(undo),
            np.array(firstUndo).astype(np.int16),
            np.array(lastUndo).astype(np.int16),

            np.array(submit).astype(np.int16),
            np.array(checkEnd).astype(np.int16),

            np.array(severityOfErrors).astype(np.int16),
            np.array(error).astype(np.int16), 
            np.array(budget_change).astype(np.int16),
            np.array(within_reach_change).astype(np.int16),

            np.array(cumulative_error),
            np.array(missed_reward), 
            np.array(error_rate),
            
            np.array(RT),np.array(undoRT),np.array(tortuosity)]
data = [subjects]
for data_ in dataList:
    data.append(data_.reshape((-1)).tolist())
    
data = np.array(data).transpose()

df = pd.DataFrame(data)
df.columns = headerList
df.to_csv(R_out_dir + 'choicelevel_data_without_tree.csv', index=False)


