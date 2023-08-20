import numpy as np
import os
from glob import glob
from numpy import genfromtxt
import pandas as pd

# import data
data_all = []

# directories
home_dir = '/Users/dbao/My_Drive'+'/road_construction/data/2022_online/'
map_dir = 'active_map/'
data_dir  = 'data/preprocessed'
# home_dir = 'G:\My Drive\\researches\\nyu\\road-construction-local-dk\data_online_2022/'
# map_dir = 'active_map/'
# data_dir  = 'data/preprocessed'
R_out_dir = home_dir+'R_analysis_data/'


flist = glob(home_dir + data_dir + '/preprocess4_sub_*.csv')
# import experiment data
for fname in flist:
    with open(fname, 'r') as f:
        all_data = pd.read_csv(f)
        data_all.append(all_data)

## ==========================================================================
############### Puzzle level
puzzleID = []         # puzzle id
trialID = []          # trial id

reward = []           # reward on each trial
numCities = []        # number of cities connected
mas = []              # MAS

nos = []              # number of optimal solutions
leftover = []         # budget left after the maximum number of cities have been connected.
numError = []         # number of errors in a puzzle
sumSeverityErrors = []# sum of severity of errors

undo_c = []           # undo condition or not : 1 for with undo condition and 0 for without undo condition
numUNDO = []          # number of undos
numFullUndo = []      # number of "single undo", combine sequential undo into one
numEnd  = []          # how many time reaches the end

TT = []               # total time taken for a trial
RT1= []               # first step RT
RTlater = []          # later step RT
RTsubmit=[]           # submit RT

tortuosity = []       # tortuosity


for i in range(len(data_all)): # iterate over subjects
    data_all[i] = data_all[i].replace('undo',1)
    data_all[i] = data_all[i].replace('basic',0)

    # prev_mapid = -1  # arbitrary number
    # prev_mapname = -1
    prev_trial = -1

    # empty list to save per subject
    temp_puzzleID          = []
    temp_trialID           = []

    temp_reward            = []        
    temp_numCities         = []
    temp_mas               = []
    
    temp_nos               = []
    temp_leftover          = []
    
    temp_numError          = []
    temp_sumSeverityErrors = []

    temp_undo_c            = []
    temp_numUNDO           = []
    temp_numFullUndo       = []
    temp_numEnd            = []
    
    temp_TT                = []
    temp_RT1               = []
    temp_RTlater           = []
    temp_RTsubmit          = []

    temp_tortuosity = []

    ti = 0
    while ti < data_all[i].shape[0]: # iterate over moves
                          
        if (prev_trial != np.array(data_all[i].trial_id)[ti]): # which means if the trial has changed / only save once per trial
            
            single_trial = data_all[i][np.array(data_all[i].trial_id) == np.array(data_all[i].trial_id)[ti]]
            
            temp_puzzleID.append(np.array(single_trial.map_id)[0])
            temp_trialID.append(np.array(single_trial.trial_id)[0])
            
            temp_reward.append(pow(np.array(single_trial.n_city_all)[-1],2))
            temp_numCities.append(np.array(single_trial.n_city_all)[-1])
            temp_mas.append(np.array(single_trial.mas_all)[0])
            
            temp_nos.append(np.array(single_trial.n_opt_paths_all)[0])
            temp_leftover.append(np.array(single_trial.currentBudget)[-1])
            
            mas_all_trial = np.array(single_trial.mas_all)
            errors_trial = (mas_all_trial[1:] - mas_all_trial[:-1])
            temp_numError.append(np.sum(errors_trial<0)) # how many errors?
            temp_sumSeverityErrors.append(np.sum(np.abs(errors_trial[errors_trial<0])))
            
            temp_undo_c.append(np.double(np.array(single_trial.condition)[0]).astype(np.int16))
            temp_numUNDO.append(np.sum(np.array(single_trial.undoIndicator )))
            temp_numEnd.append(np.sum(np.array(single_trial.checkEnd))-1) # remove redundent count from submit
            
            n_path = 0
            if (np.array(single_trial.condition)[0]==1):
                for ai in range(1,single_trial.shape[0]):
                    if (np.array(single_trial.undoIndicator)[ai] == 1) and (np.array(single_trial.undoIndicator)[ai-1] == 0): 
                        n_path = n_path + 1
            temp_numFullUndo.append(n_path)
            
            temp_TT.append(np.array(single_trial.time_all)[-1]/1000 - np.array(single_trial.time_all)[0]/1000)
            temp_RT1.append(np.array(single_trial.rt_all)[1]/1000) 
            temp_RTsubmit.append(np.array(single_trial.rt_all)[-1]/1000)         
            index_later = single_trial.index[(single_trial['rt_all'] != -1) & (single_trial['undoIndicator'] != 1)& (single_trial['submit'] != 1)]
            RT_later = single_trial.loc[index_later,'rt_all']
            temp_RTlater.append(np.mean(RT_later)/1000)         

            temp_tortuosity.append(np.array(single_trial.tortuosity_all)[-1])

            # prev_mapid = np.array(data_all[i].map_id)[ti]
            # prev_mapname = data_all[i].condition[ti]
            prev_trial = np.array(data_all[i].trial_id)[ti]
        
        ti += 1

    puzzleID.append(temp_puzzleID)
    trialID.append(temp_trialID)

    reward.append(temp_reward)
    numCities.append(temp_numCities)
    mas.append(temp_mas)
    
    nos.append(temp_nos)
    leftover.append(temp_leftover)
    
    numError.append(temp_numError)
    sumSeverityErrors.append(temp_sumSeverityErrors)
    numEnd.append(temp_numEnd)
    numFullUndo.append(temp_numFullUndo)
    
    undo_c.append(temp_undo_c)
    numUNDO.append(temp_numUNDO)
    
    TT.append(temp_TT)
    RT1.append(temp_RT1)  
    RTlater.append(temp_RTlater)             
    RTsubmit.append(temp_RTsubmit)   

    tortuosity.append(temp_tortuosity)

    print('*'*10)
    print(i)
    # print(np.unique(temp_mas))

np.savetxt(R_out_dir + 'puzzleID.csv', np.array(puzzleID).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)

np.savetxt(R_out_dir + 'reward.csv', np.array(reward).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'numCities.csv', np.array(numCities).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'mas.csv', np.array(mas).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'nos.csv', np.array(nos).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'leftover.csv', np.array(leftover).transpose(),fmt='%f',delimiter=',',encoding=None)

np.savetxt(R_out_dir + 'numError.csv', np.array(numError).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'sumSeverityErrors.csv', np.array(sumSeverityErrors).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)

np.savetxt(R_out_dir + 'undo_c.csv', np.array(undo_c).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'numUNDO.csv', np.array(numUNDO).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'TT.csv', np.array(TT).transpose(),fmt='%f',delimiter=',',encoding=None)

np.savetxt(R_out_dir + 'tortuosity.csv', np.array(tortuosity).transpose(),fmt='%f',delimiter=',',encoding=None)

## ==========================================================================
######## all puzzle-level data in one file
headerList = ['subjects', 'puzzleID', 
              'reward', 'numCities', 'mas', 
              'nos', 'leftover', 
              'numError', 'sumSeverityErrors', 
              'condition','numUNDO', 'numFullUndo', 'numEnd', 
              'TT','RT1','RTlater','RTsubmit',
              'tortuosity']
subjects = []
for i in range(len(data_all)):
    subjects.extend(((np.ones(len(np.unique(np.array(data_all[i].trial_id))))*(i)).astype(np.int16).tolist())) #i+1
data = [subjects]
dataList = [np.array(puzzleID).astype(np.int16), 
            np.array(reward).astype(np.int16), np.array(numCities).astype(np.int16), np.array(mas).astype(np.int16), 
            np.array(nos).astype(np.int16), np.array(leftover),
            np.array(numError).astype(np.int16), np.array(sumSeverityErrors).astype(np.int16), 
            np.array(undo_c).astype(np.int16), np.array(numUNDO).astype(np.int16), np.array(numFullUndo).astype(np.int16), np.array(numEnd).astype(np.int16),
            np.array(TT),np.array(RT1),np.array(RTlater),np.array(RTsubmit),
            np.array(tortuosity)]
for data_ in dataList:
    data.append(data_.reshape((-1)).tolist())

data = np.array(data).transpose()

np.savetxt(R_out_dir + 'data.csv',data,delimiter=',', 
           fmt='%d,%d, %d,%d,%d, %d,%f, %d,%d, %d,%d,%d,%d, %f,%f,%f,%f, %f', 
           header=",".join(headerList),comments='')

# headerList_ = [" ", *headerList]
# np.savetxt(R_out_dir + 'data.txt',data,delimiter=' ', fmt='%d,%d, %d,%d,%d, %d,%f, %d,%d, %d,%d,%d,%d, %f,%f,%f,%f, %f', header=" ".join(headerList_),comments='')

