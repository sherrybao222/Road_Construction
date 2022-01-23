import numpy as np
import os
from glob import glob
from numpy import genfromtxt
import pandas as pd

# import data
data_all = []

# directories
home_dir = '/Users/dbao/google_drive_db'+'/road_construction/data/2022_online/'
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

undo_c = []              # undo condition binary
undo = []                # undo binary
firstUndo = []           # first undo in a sequence of undo
lastUndo  = []            # binary: new path start, where undo goes back to, (excluding the first starting city), last undo

submit = []
checkEnd = []

severityOfErrors = []    # severity of errors
error = []               # error binary

currNumCities = []       # current number of connected cities
currMas = []             # current MAS
currNos = []             # current number of optimal solutions
leftover = []            # budget left
within_reach = []        # ncities within reach

RT = []                  # reaction time of moves (including undo)
undoRT = []              # can get undoRT using undo binary

for i in range(len(data_all)):
    # prev_mapid = -1  # arbitrary number
    # prev_mapname = -1
    # prev_trial = -1
    data_all[i] = data_all[i].replace('undo',1)
    data_all[i] = data_all[i].replace('basic',0)

    # empty list to save per subject
    temp_subjects = []
    temp_puzzleID = data_all[i].map_id
    temp_trialID = data_all[i].trial_id

    temp_undo = data_all[i].undoIndicator
    temp_undo_c = data_all[i].condition
    temp_firstUndo = []               # first undo in a sequence of undo
    temp_lastUndo = []            # binary: new path start, where undo goes back to, (excluding the first starting city)
    temp_submit = data_all[i].submit
    temp_checkEnd = data_all[i].checkEnd
    
    temp_severityOfErrors = []
    temp_error = []
    
    temp_currNumCities = data_all[i].n_city_all
    temp_currMas = data_all[i].mas_all
    temp_currNos = data_all[i].n_opt_paths_all
    temp_leftover = data_all[i].currentBudget
    temp_within_reach = data_all[i].n_within_reach

    temp_RT = data_all[i].rt_all # can get undoRT using undo binary
    temp_undoRT = []

    mas_all_trial = np.array(data_all[i].mas_all)
    errors_trial = np.array([0, *(mas_all_trial[1:] - mas_all_trial[:-1]).tolist()])
    severe_error_trial = np.zeros(np.array(errors_trial).shape)
    severe_error_trial[errors_trial<0] = errors_trial[errors_trial<0]
    severe_error_trial = np.abs(severe_error_trial).astype(np.int16)
    
    errors_trial = np.zeros(np.array(errors_trial).shape).astype(np.int16)
    errors_trial[severe_error_trial!=0] = 1
    
    for ti in range(data_all[i].shape[0]):
        subjects.append(i)
        # puzzleID.append(np.array(data_all[i].map_id)[ti])
        # trialID.append(np.array(data_all[i].trial_id)[ti])
        
        # undo_c.append(np.array(data_all[i].condition)[ti])
        # undo.append(np.array(data_all[i].undoIndicator & (data_all[i].n_city_all != 1))[ti])
                
        severityOfErrors.append(severe_error_trial[ti])
        error.append(errors_trial[ti])
        

        temp_firstUndo.append((np.array(data_all[i].undoIndicator)[ti] == 1) & (np.array(data_all[i].rt_all)[ti] != -1) & (np.array(data_all[i].undoIndicator)[ti-1] == 0))         # first undo in a sequence of undo
            
        if (ti != data_all[i].shape[0]-1):
            temp_lastUndo.append((np.array(data_all[i].undoIndicator)[ti] == 1) & (np.array(data_all[i].rt_all)[ti] != -1) & (np.array(data_all[i].undoIndicator)[ti+1] == 0) & (np.array(data_all[i].submit)[ti] == 0))         # binary: new path start, where undo goes back to, (excluding the first starting city)
        else:
            temp_lastUndo.append(0)
            
        # currNumCities.append(np.array(data_all[i].n_city_all)[ti])
        # currMas.append(np.array(data_all[i].mas_all)[ti])
        # currNos.append(np.array(data_all[i].n_opt_paths_all)[ti])
        # leftover.append(np.array(data_all[i].currentBudget)[ti])
        # within_reach.append(np.array(data_all[i].n_within_reach)[ti])

        # RT.append(np.array(data_all[i].rt_all)[ti]) # can get undoRT using undo binary
        if np.array(data_all[i].undoIndicator & (data_all[i].n_city_all != 1))[ti]==1:
            undoRT.append(np.array(data_all[i].rt_all)[ti])
        else:
            undoRT.append(-1) # if there is no undo
            
    subjects.extend(temp_subjects)
    puzzleID.extend(temp_puzzleID)
    trialID.extend(temp_trialID)

    undo_c.extend(temp_undo_c)
    undo.extend(temp_undo)
    firstUndo.extend(temp_firstUndo)        # first undo in a sequence of undo
    lastUndo.extend(temp_lastUndo)           # binary: new path start, where undo goes back to, (excluding the first starting city), last undo
    submit.extend(temp_submit)
    checkEnd.extend(temp_checkEnd)
    
    severityOfErrors.extend(temp_severityOfErrors)
    error.extend(temp_error)
    
    currNumCities.extend(temp_currNumCities)
    currMas.extend(temp_currMas)
    currNos.extend(temp_currNos)
    leftover.extend(temp_leftover)
    within_reach.extend(temp_within_reach)

    RT.extend(temp_RT)
    undoRT.extend(temp_undoRT)
    
np.savetxt(R_out_dir + 'choicelevel_subjects.csv', np.array(subjects).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_puzzleID.csv', np.array(puzzleID).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_trialID.csv', np.array(trialID).astype(np.int16),fmt='%d',delimiter=',',encoding=None)

np.savetxt(R_out_dir + 'choicelevel_undo_c.csv', np.array(undo_c).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_undo.csv', np.array(undo).astype(np.int16),fmt='%d',delimiter=',',encoding=None)

np.savetxt(R_out_dir + 'choicelevel_severityOfErrors.csv', np.array(severityOfErrors).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_error.csv', np.array(error).astype(np.int16),fmt='%d',delimiter=',',encoding=None)

np.savetxt(R_out_dir + 'choicelevel_currNumCities.csv', np.array(currNumCities).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_currMas.csv', np.array(currMas).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_currNos.csv', np.array(currNos).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_leftover.csv', np.array(leftover),fmt='%f',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_within_reach.csv', np.array(within_reach).astype(np.int16),fmt='%d',delimiter=',',encoding=None)

np.savetxt(R_out_dir + 'choicelevel_RT.csv', np.array(RT),fmt='%f',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_undoRT.csv', np.array(undoRT),fmt='%f',delimiter=',',encoding=None)


## ==========================================================================
######## all choice-level data in one file
headerList = ['subjects', 'puzzleID','trialID',
              'currNumCities','currMas','currNos', 'leftover','within_reach',
              'condition','undo','firstUndo','lastUndo',
              'submit','checkEnd',
              'severityOfErrors', 'error',
              'RT','undoRT']

dataList = [np.array(puzzleID).astype(np.int16), np.array(trialID).astype(np.int16),
            np.array(currNumCities).astype(np.int16), np.array(currMas).astype(np.int16), np.array(currNos).astype(np.int16),np.array(leftover).astype(np.int16),np.array(within_reach).astype(np.int16),
            np.array(undo_c).astype(np.int16),np.array(undo),np.array(firstUndo).astype(np.int16),np.array(lastUndo).astype(np.int16),
            np.array(submit).astype(np.int16),np.array(checkEnd).astype(np.int16),
            np.array(severityOfErrors).astype(np.int16),np.array(error).astype(np.int16),
            np.array(RT).astype(np.int16),np.array(undoRT).astype(np.int16)]
data = [subjects]
for data_ in dataList:
    data.append(data_.reshape((-1)).tolist())

data = np.array(data).transpose()

np.savetxt(R_out_dir + 'choicelevel_data.csv',data, delimiter=',',fmt='%d,%d,%d, %d,%d,%d,%f,%d, %d,%d,%d,%d,%d,%d, %d,%d, %f,%f', header=",".join(headerList),comments='')
headerList_ = [" ", *headerList]
np.savetxt(R_out_dir + 'choicelevel_data.txt',data, delimiter=' ',fmt='%d,%d,%d, %d,%d,%d,%f,%d, %d,%d,%d,%d,%d,%d, %d,%d, %f,%f', header=" ".join(headerList_),comments='')


# undo data saving
ind_data = np.where(np.array(undoRT) != -1)
data = [np.array(subjects)[ind_data]]
for data_ in dataList:
    data.append(data_[ind_data])
data = np.array(data).transpose()
np.savetxt(R_out_dir + 'choicelevel_undo_data.csv',data, delimiter=',',fmt='%d,%d,%d, %d,%d,%d,%f,%d, %d,%d,%d,%d,%d,%d, %d,%d, %f,%f',header=",".join(headerList),comments='')
headerList_ = [" ", *headerList]
np.savetxt(R_out_dir + 'choicelevel_undo_data.txt',data, delimiter=' ',fmt='%d,%d,%d, %d,%d,%d,%f,%d, %d,%d,%d,%d,%d,%d, %d,%d, %f,%f',header=" ".join(headerList_),comments='')



