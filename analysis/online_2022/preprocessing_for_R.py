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

flist = glob(home_dir + data_dir + '/preprocess4_sub_*.csv')
# import experiment data
for fname in flist:
    with open(fname, 'r') as f:
        all_data = pd.read_csv(f)
        data_all.append(all_data)


## ==========================================================================
# DATA SAVE FOR THE FURTHER ANALYSIS USING R
R_out_dir = home_dir+'R_analysis_data/'

############### Puzzle level
# reward on each trial
reward = []
# number of cities connected
numCities = []
# MAS
mas = []
# number of optimal solutions
nos = []
# undo condition or not : 1 for with undo condition and 0 for without undo condition
undo_c = []
# budget left after the maximum number of cities have been connected.
leftover = []
# number of errors in a puzzle
numError = []
# sum of severity of errors
sumSeverityErrors = []
# number of undos
numUNDO = []
# time taken for a trial
TT = []
# puzzle id
puzzleID = []
# trial id
trialID = []

for i in range(len(data_all)):
    ti = 0
    prev_mapid = -1  # arbitrary number
    prev_mapname = -1
    prev_trial = -1
    data_all[i].map_name[data_all[i].map_name == 'undo']  = 1
    data_all[i].map_name[data_all[i].map_name == 'basic'] = 0

    # empty list to save per subject
    temp_reward            = []        
    temp_numCities         = []
    temp_mas               = []
    temp_nos               = []
    temp_undo_c            = []
    temp_leftover          = []
    temp_numError          = []
    temp_sumSeverityErrors = []
    temp_numUNDO = []
    temp_TT = []
    temp_puzzleID = []
    temp_trialID = []

    # for ti in range(undo_trials.shape[0]):
    while ti < data_all[i].shape[0]:
        if (prev_trial != np.array(data_all[i].trial_id)[ti]):
        # if (prev_mapid != np.array(data_all[i].map_id)[ti]) or (prev_mapname != data_all[i].map_name[ti]):  # which means if the trial has changed
        #     single_trial = data_all[i][np.array(data_all[i].map_id) == np.array(data_all[i].map_id)[ti]]
            single_trial = data_all[i][np.array(data_all[i].trial_id) == np.array(data_all[i].trial_id)[ti]]
            
            temp_reward.append(pow(np.array(single_trial.n_city_all)[-1],2))
            temp_numCities.append(np.array(single_trial.n_city_all)[-1])
            temp_mas.append(np.array(single_trial.mas_all)[0])
            temp_nos.append(np.array(single_trial.n_opt_paths_all)[0])
            temp_undo_c.append(np.double(np.array(single_trial.map_name)[0]).astype(np.int16))
            temp_leftover.append(np.array(single_trial.budget_all)[-1])
            mas_all_trial = np.array(single_trial.mas_all)
            errors_trial = (mas_all_trial[1:] - mas_all_trial[:-1])
            temp_numError.append(np.sum(errors_trial<0)) # how many errors?
            temp_sumSeverityErrors.append(np.sum(np.abs(errors_trial[errors_trial<0])))
            temp_numUNDO.append(np.sum(np.array(single_trial.undo_all)))
            temp_TT.append(np.array(single_trial.time_all)[-1]/1000)
            temp_puzzleID.append(np.array(single_trial.map_id)[0])
            temp_trialID.append(np.array(single_trial.trial_id)[0])

            prev_mapid = np.array(data_all[i].map_id)[ti]
            prev_mapname = data_all[i].map_name[ti]
            prev_trial = np.array(data_all[i].trial_id)[ti]
        ti += 1
    
    reward.append(temp_reward)
    numCities.append(temp_numCities)
    mas.append(temp_mas)
    nos.append(temp_nos)
    undo_c.append(temp_undo_c)
    leftover.append(temp_leftover)
    numError.append(temp_numError)
    sumSeverityErrors.append(temp_sumSeverityErrors)
    numUNDO.append(temp_numUNDO)
    TT.append(temp_TT)
    puzzleID.append(temp_puzzleID)
    trialID.append(temp_trialID)

    print('*'*10)
    print(i)
    print(np.unique(temp_mas))


np.savetxt(R_out_dir + 'reward.csv', np.array(reward).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'numCities.csv', np.array(numCities).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'mas.csv', np.array(mas).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'nos.csv', np.array(nos).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'undo_c.csv', np.array(undo_c).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'leftover.csv', np.array(leftover).transpose(),fmt='%f',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'numError.csv', np.array(numError).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'sumSeverityErrors.csv', np.array(sumSeverityErrors).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'numUNDO.csv', np.array(numUNDO).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'TT.csv', np.array(TT).transpose(),fmt='%f',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'puzzleID.csv', np.array(puzzleID).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)

## ==========================================================================
######## all puzzle-level data in one file
subjects = []
for i in range(len(data_all)):
    subjects.extend(((np.ones(len(np.unique(np.array(data_all[i].trial_id))))*(i+1)).astype(np.int16).tolist()))
headerList = ['subjects', 'puzzleID', 'numCities', 'mas', 'nos', 'undo_c', 'leftover', 'numError', 'sumSeverityErrors', 'numUNDO', 'TT']
dataList = [np.array(puzzleID).astype(np.int16), np.array(numCities).astype(np.int16), np.array(mas).astype(np.int16),  np.array(nos).astype(np.int16),  np.array(undo_c).astype(np.int16),
            np.array(leftover),  np.array(numError).astype(np.int16),  np.array(sumSeverityErrors).astype(np.int16),   np.array(numUNDO).astype(np.int16),
            np.array(TT)]
data = [subjects]
for data_ in dataList:
    data.append(data_.reshape((-1)).tolist())

data = np.array(data).transpose()

np.savetxt(R_out_dir + 'data.csv',data,delimiter=',',fmt='%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%f',header=",".join(headerList),comments='')

headerList_ = [" ", *headerList]
np.savetxt(R_out_dir + 'data.txt',data,delimiter=' ',fmt='%d %d %d %d %d %d %f %d %d %d %f',header=" ".join(headerList_),comments='')

#######################################################################################################################################
##################### choice-level

# import data
data_all = []

flist = glob(home_dir + data_dir + '/preprocess4_sub_*.csv')
# import experiment data
for fname in flist:
    with open(fname, 'r') as f:
        all_data = pd.read_csv(f)
        data_all.append(all_data)

## ==========================================================================
# DATA SAVE FOR THE FURTHER ANALYSIS USING R
R_out_dir = 'R_analysis/'

# undoc binary
undo_c = []
# undo binary
undo = []
# severity of errors
severityOfErrors = []
# error binary
error = []
# current number of connected cities
currNumCities = []
# current MAS
currMas = []
# current number of optimal solutions
currNos = []
# reaction time of moves (including undo)
RT = [] # can get undoRT using undo binary
undoRT = []
# subject id
subjects = []
# puzzle id
puzzleID = []
# budget left
leftover = []
# trial ID
trialID = []
# ncities within reach
within_reach = []


for i in range(len(data_all)):
    ti = 0
    prev_mapid = -1  # arbitrary number
    prev_mapname = -1
    prev_trial = -1
    data_all[i].map_name[data_all[i].map_name == 'undo']  = 1
    data_all[i].map_name[data_all[i].map_name == 'basic'] = 0

    # empty list to save per subject
    temp_undo = []
    temp_undo_c = []
    temp_severityOfErrors = []
    temp_error = []
    temp_currNumCities = []
    temp_currMas = []
    temp_currNos = []
    temp_RT = [] # can get undoRT using undo binary
    temp_undoRT = []
    temp_subjects = []
    temp_puzzleID = []
    temp_leftover = []
    temp_trialID = []
    temp_within_reach = []

    mas_all_trial = np.array(data_all[i].mas_all)
    errors_trial = np.array([0, *(mas_all_trial[1:] - mas_all_trial[:-1]).tolist()])
    severe_error_trial = np.zeros(np.array(errors_trial).shape)
    severe_error_trial[errors_trial<0] = errors_trial[errors_trial<0]
    severe_error_trial = np.abs(severe_error_trial).astype(np.int16)
    errors_trial = np.zeros(np.array(errors_trial).shape).astype(np.int16)
    errors_trial[severe_error_trial!=0] = 1

    # for ti in range(undo_trials.shape[0]):
    while ti < data_all[i].shape[0]:
        temp_undo_c.append(np.array(data_all[i].map_name)[ti])
        temp_undo.append(np.array(data_all[i].undo_all)[ti])
        temp_severityOfErrors.append(severe_error_trial[ti])
        temp_error.append(errors_trial[ti])
        temp_currNumCities.append(np.array(data_all[i].n_city_all)[ti])
        temp_currMas.append(np.array(data_all[i].mas_all)[ti])
        temp_currNos.append(np.array(data_all[i].n_opt_paths_all)[ti])
        temp_RT.append(np.array(data_all[i].rt_all)[ti]) # can get undoRT using undo binary
        if np.array(data_all[i].undo_all)[ti]==1:
            temp_undoRT.append(np.array(data_all[i].rt_all)[ti])
        else:
            temp_undoRT.append(-1) # if there is no undo
        temp_subjects.append(i)
        temp_puzzleID.append(np.array(data_all[i].map_id)[ti])
        temp_leftover.append(np.array(data_all[i].budget_all)[ti])
        temp_trialID.append(np.array(data_all[i].trial_id)[ti])
        temp_within_reach.append(np.array(data_all[i].n_within_reach)[ti])
        # if (prev_trial != np.array(data_all[i].trial_id)[ti]):
        # # if (prev_mapid != np.array(data_all[i].map_id)[ti]) or (prev_mapname != data_all[i].map_name[ti]):  # which means if the trial has changed
        # #     single_trial = data_all[i][np.array(data_all[i].map_id) == np.array(data_all[i].map_id)[ti]]
        #     single_trial = data_all[i][np.array(data_all[i].trial_id) == np.array(data_all[i].trial_id)[ti]]
        #     temp_numCities.append(np.array(single_trial.n_city_all)[-1])
        #     temp_mas.append(np.array(single_trial.mas_all)[0])
        #     temp_nos.append(np.array(single_trial.n_opt_paths_all)[0])
        #     temp_undo_c.append(np.double(np.array(single_trial.map_name)[0]).astype(np.int16))
        #     temp_leftover.append(np.array(single_trial.budget_all)[-1])
        #     mas_all_trial = np.array(single_trial.mas_all)
        #     errors_trial = (mas_all_trial[1:] - mas_all_trial[:-1])
        #     temp_numError.append(np.sum(errors_trial<0)) # how many errors?
        #     temp_sumSeverityErrors.append(np.sum(np.abs(errors_trial[errors_trial<0])))
        #     temp_numUNDO.append(np.sum(np.array(single_trial.undo_all)))
        #     temp_TT.append(np.array(single_trial.time_all)[-1]/1000)
        #     temp_puzzleID.append(np.array(single_trial.map_id)[0])
        #     temp_trialID.append(np.array(single_trial.trial_id)[0])
        #
        #     prev_mapid = np.array(data_all[i].map_id)[ti]
        #     prev_mapname = data_all[i].map_name[ti]
        #     prev_trial = np.array(data_all[i].trial_id)[ti]
        ti += 1
    undo_c.extend(temp_undo_c)
    undo.extend(temp_undo)
    severityOfErrors.extend(temp_severityOfErrors)
    error.extend(temp_error)
    currNumCities.extend(temp_currNumCities)
    currMas.extend(temp_currMas)
    currNos.extend(temp_currNos)
    RT.extend(temp_RT)
    undoRT.extend(temp_undoRT)
    subjects.extend(temp_subjects)
    puzzleID.extend(temp_puzzleID)
    leftover.extend(temp_leftover)
    trialID.extend(temp_trialID)
    within_reach.extend(temp_within_reach)

    # undo.append(temp_undo)
    # severityOfErrors.append(temp_severityOfErrors)
    # error.append(temp_error)
    # currNumCities.append(temp_currNumCities)
    # currMas.append(temp_currMas)
    # currNos.append(temp_currNos)
    # RT.append(temp_RT)
    # undoRT.append(temp_undoRT)
    # subjects.append(temp_subjects)
    # puzzleID.append(temp_puzzleID)
    # leftover.append(temp_leftover)
    # trialID.append(temp_trialID)

# undo.append(temp_undo)
# severityOfErrors.append(temp_severityOfErrors)
# error.append(temp_error)
# currNumCities.append(temp_currNumCities)
# currMas.append(temp_currMas)
# currNos.append(temp_currNos)
# RT.append(temp_RT)
# undoRT.append(temp_undoRT)
# subjects.append(temp_subjects)
# puzzleID.append(temp_puzzleID)
# leftover.append(temp_leftover)
# trialID.append(temp_trialID)

np.savetxt(R_out_dir + 'choicelevel_undo_c.csv', np.array(undo_c).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_undo.csv', np.array(undo).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_severityOfErrors.csv', np.array(severityOfErrors).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_error.csv', np.array(error).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_currNumCities.csv', np.array(currNumCities).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_currMas.csv', np.array(currMas).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_currNos.csv', np.array(currNos).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_RT.csv', np.array(RT),fmt='%f',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_undoRT.csv', np.array(undoRT),fmt='%f',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_subjects.csv', np.array(subjects).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_puzzleID.csv', np.array(puzzleID).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_trialID.csv', np.array(trialID).astype(np.int16),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_leftover.csv', np.array(leftover),fmt='%f',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'choicelevel_within_reach.csv', np.array(within_reach).astype(np.int16),fmt='%d',delimiter=',',encoding=None)


## ==========================================================================
######## all puzzle-level data in one file
headerList = ['subjects', 'puzzleID','trialID','currNumCities','currMas','currNos',
              'undo','severityOfErrors', 'error','RT','undoRT','leftover','within_reach']
dataList = [np.array(puzzleID).astype(np.int16), np.array(trialID).astype(np.int16),
            np.array(currNumCities).astype(np.int16), np.array(currMas).astype(np.int16), np.array(currNos).astype(np.int16),
            np.array(undo),np.array(severityOfErrors),np.array(error),np.array(RT),np.array(undoRT),np.array(leftover),np.array(within_reach).astype(np.int16)]
data = [subjects]
for data_ in dataList:
    data.append(data_.reshape((-1)).tolist())

data = np.array(data).transpose()

np.savetxt(R_out_dir + 'choicelevel_data.csv',data,delimiter=',',fmt='%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%d',header=",".join(headerList),comments='')
headerList_ = [" ", *headerList]
np.savetxt(R_out_dir + 'choicelevel_data.txt',data,delimiter=' ',fmt='%d %d %d %d %d %d %d %d %d %d %d %f %d',header=" ".join(headerList_),comments='')


# undo data saving
ind_data = np.where(np.array(undoRT) != -1)
data = [np.array(subjects)[ind_data]]
for data_ in dataList:
    data.append(data_[ind_data])
data = np.array(data).transpose()
np.savetxt(R_out_dir + 'choicelevel_undo_data.csv',data,delimiter=',',fmt='%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%d',header=",".join(headerList),comments='')
headerList_ = [" ", *headerList]
np.savetxt(R_out_dir + 'choicelevel_undo_data.txt',data,delimiter=' ',fmt='%d %d %d %d %d %d %d %d %d %d %d %f %d',header=" ".join(headerList_),comments='')


