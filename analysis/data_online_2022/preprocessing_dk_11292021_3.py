# Preprocessing code re-written by Dongjae Kim

import json
import pandas as pd
import numpy as np

from tqdm import tqdm

import fnmatch  # match file name
import os

from glob import glob
from util.load_and_fit import *

# directories
home_dir = './'
map_dir = 'active_map/'
data_dir  = 'data/'

# load maps
# load basic map
with open(home_dir + map_dir + 'basicMap.json', 'r') as file:
    basic_map = json.load(file)
# load undo map
with open(home_dir + map_dir + 'undoMap.json', 'r') as file:
    undo_map = json.load(file)

# parameters
subCount = 0
orderBlk = [[2, 3, 3, 2], [3, 2, 2, 3]]  # condition orders

all_city = list(range(1, 30))  # all city names (excluding start)
basic_index = []  # basic version trial index
for i, cond in enumerate(
        orderBlk[1]
):  # for now we only have this order; will need another script for another order
    if cond == 2:
        basic_index.extend(range(i * 23, (i + 1) * 23))

# find the list of subject files in the data directory
target_expfile = 'FIN_*_RC-Phaser_*.csv'
# target_expfile = 'PARTICIPANT_RC-Phaser_*.csv'
flist = glob(home_dir + data_dir + target_expfile)

while subCount < len(flist):
    map_id = []
    choice_all = []  # all single choice
    chosen_all = []  # all chosen cities at the single choice
    remain_all = []  # remaining cities at the single choice
    budget_all = []  # budget at the choice
    n_city_all = []  # number of connected cities at the choice
    n_u_all = []  # n of cities within reach
    u_trial_all = []  # cities within reach

    map_name = [] # whether it is coming from basic or undo.
    time_all = [] # raw time in ms. used to calculate rt_all.
    rt_all = [] # reaction time for every clicks
    undo_all = [] # indicating whether the click is undo or not (1 or 0)
    n_opt_paths_all = [] # n of optimal paths at the moment
    n_subopt_paths_all = [] # n of suboptimal (which has #ConnectableCities of opt - 1) paths at the moment.
    mas_all = [] # maximum achievable score
    trial_id = []
    n_within_reach = [] # number of cities within reach
    # error_all =[] # this can be inferred from mas_all.





    thefile = flist[subCount]
    # with open(thefile, 'r') as f:
    #     data_exp = pd.read_csv(f)

    # load a file
    # read int not str type of data
    from ast import literal_eval
    with open(thefile, 'r') as f:
        data_exp = pd.read_csv(f, converters={"choiceDyn": literal_eval})

    # find files with name...
    # target_expfile = 'PARTICIPANT_RC-Phaser_*.csv'
    # for thefile in os.listdir(home_dir + data_dir):
    #     if fnmatch.fnmatch(thefile, target_expfile):
    #         with open(home_dir + data_dir + thefile, 'r') as file:
    #             data_exp = pd.read_csv(file)


    # Drop check trials beforehand for easy code
    data_exp = data_exp.drop(
        data_exp[data_exp.checkTrial == 1].index)

    # find the order of blocks and trials
    condition_blocks = []
    # trials_index_blocks =[]
    for bi in np.unique(np.array(data_exp.blockID)): # for every unique block ID (0,1,2,3)
        condition_blocks.append(np.unique(np.array(data_exp.condition[data_exp.blockID == bi])).squeeze().tolist()) # save condition of each block
        basic_index = []
        if condition_blocks[-1] == 2:
            basic_index.extend(range(bi * 23, (bi + 1) * 23))


    subName = thefile.split('RC-Phaser_')[1].split('.csv')[0]
    subNum = data_exp["subID"][0]

    if not os.path.exists(home_dir + data_dir + '/preprocess4_sub_' + subName +
              '.csv'):
    # if True:
        # updated
        # block wise itrative saving
        for t_i in tqdm(range(len(condition_blocks))):
            bi = condition_blocks[t_i]
            # for t_i, bi in enumerate(condition_blocks):
            basic_index = range(t_i * 23, (t_i + 1) * 23)
            for indTrial_ind in tqdm(range(len(basic_index))):
                indTrial=basic_index[indTrial_ind]
                single_trial = data_exp[data_exp.trialID.astype(np.int64) == indTrial]
                single_trial = single_trial.reset_index()
                if not (len(single_trial) is 0):
                    currenMapID = int(single_trial['mapID'][0])
                    choiceLen = []
                    choiceHis = []
                    budgetHis = []
                    time = []
                    undo = []
                    mapName = []
                    trialId = []

                    optPaths = []
                    suboptPaths = []
                    withinReach = []
                    MAS = []
                    '''
                    n_opt_paths_all = [] # n of optimal paths at the moment
                    n_subopt_paths_all = [] # n of suboptimal (which has #ConnectableCities of opt - 1) paths at the moment.
                    mas_all = [] # maximum achievable score
                    '''

                    try:
                        for node_ in TS.nodes_bucket:
                            node_.parent=None
                            for node__ in node_.leaves:
                                node__.parent = None
                        for node_ in TS.nodes_bucket:
                            node_.children = ()
                        del TS
                    except:
                        print('')
                    # make tree structure for further analysis.
                    if bi == 2:
                        mmap = data_map(basic_map[currenMapID])
                    elif bi ==3:
                        mmap = data_map(undo_map[currenMapID])

                    TS = TreeStructure(mmap, map_id = currenMapID)
                    # TS.get_Tree()
                    TS.get_Tree(Node(0,budget = 300))
                    TS.gen_path_flatten()

                    # TSs.append(TreeStructure(mmap))
                    # TSs[-1].get_Tree(Node(0,budget = 300))
                    # TSs[-1].gen_path_flatten()

                    # rendered_tree = TS.render_out()
                    # currNode = TS.root_
                    currPath = []
                    # for i in range(0, len(single_trial)):
                    for i in range(len(single_trial)):
                        if (i == 0) or (single_trial['cityNr'][i - 1] !=
                                        single_trial['cityNr'][i]):

                            trialId.append(single_trial['trialID'][i])
                            choiceHis.append(single_trial['choiceHis'][i])
                            choiceLen.append(len(single_trial['choiceDyn'][i]))
                            budgetHis.append(single_trial['budgetHis'][i])
                            time.append(int(single_trial['time'][i]))
                            undo.append(int(single_trial['undo'][i]))
                            if bi is 2:
                                mapName.append('basic')
                            elif bi is 3:
                                mapName.append('undo')

                            # get all unconnected cities for each step
                            chosen_trial = single_trial['choiceDyn'][i]
                            chosen_all.append(chosen_trial)
                            remain_trial = [x for x in all_city if x not in chosen_trial]
                            remain_all.append(remain_trial)

                            # get unconnected cities within reach for each step
                            n_u = 0
                            u_trial = []
                            for c in remain_trial:
                                if basic_map[currenMapID]['distance'][int(
                                        single_trial['choiceHis'][i])][int(c)] <= float(
                                            single_trial['budgetHis'][i]):
                                    n_u = n_u + 1
                                    u_trial.append(c)
                            n_u_all.append(n_u)
                            u_trial_all.append(u_trial)



                            # path that up-to-date for path
                            if int(single_trial['undo'][i]) is 0:
                                currPath.append(int(single_trial['choiceHis'][i]))
                            else:
                                currPath.pop(-1)

                            ind_currpath = np.where([currPath == TS.paths[j] for j in range(len(TS.paths))])[0].squeeze().tolist()


                            # # reachable leaf-nodes
                            nodes_filtered = TS.nodes_bucket[ind_currpath].leaves
                            MAS_eachPath = [len(j.path) for j in nodes_filtered]


                            withinReach.append(len(TS.nodes_bucket[ind_currpath].children))

                            # reachable
                            # nodes_filtered = TSs[-1].nodes_bucket[ind_currpath].leaves
                            # MAS_eachPath = [len(j.path) for j in nodes_filtered]

                            max_MAS = max(MAS_eachPath)
                            if i == 0:
                                print(len(nodes_filtered))

                            if (currenMapID == 39) and (i==0):
                                print('*'*20)
                                print(indTrial)
                                print(max_MAS)
                                print('')

                            MAS.append(max_MAS)
                            optPaths.append(np.sum(np.array(MAS_eachPath) == max_MAS))
                            suboptPaths.append(np.sum(np.array(MAS_eachPath) == (max_MAS-1)))



                    # add to the dataframe of all maps
                    choice_all.extend(choiceHis)
                    budget_all.extend(budgetHis)
                    n_city_all.extend(choiceLen)
                    map_id_single = [currenMapID] * len(choiceHis)
                    map_id.extend(map_id_single)

                    map_name.extend(mapName)
                    time_all.extend(time)
                    rt_all.extend([0, *[a-time[j-1] for j, a in enumerate(time) if j > 0]])
                    undo_all.extend(undo)
                    n_opt_paths_all.extend(optPaths)
                    n_subopt_paths_all.extend(suboptPaths)
                    mas_all.extend(MAS)
                    trial_id.extend(trialId)
                    n_within_reach.extend(withinReach)

        data = {
            'map_name': map_name,
            'trial_id': trial_id,
            'map_id': map_id,
            'choice_all': choice_all,
            'budget_all': budget_all,
            'n_city_all': n_city_all,
            'n_u_all': n_u_all,
            'u_trial_all': u_trial_all,
            'chosen_all': chosen_all,
            'undo_all': undo_all,
            'time_all': time_all,
            'rt_all': rt_all,
            'n_opt_paths_all': n_opt_paths_all,
            'n_subopt_paths_all': n_subopt_paths_all,
            'mas_all': mas_all,
            'n_within_reach':n_within_reach
        }
        # Create DataFrame
        df = pd.DataFrame(data)
        # df.to_csv(home_dir + data_dir + '/preprocess_sub_' + str(subNum) +
        #           '.csv',
        #           index=False)

        df.to_csv(home_dir + data_dir + '/preprocess4_sub_' + subName +
                  '.csv', index=False)

    subCount = subCount + 1

import numpy as np
import os
from glob import glob
from numpy import genfromtxt
import pandas as pd

# import data
data_all = []

# directories
home_dir = './'
map_dir = 'active_map/'
data_dir  = 'data/'

flist = glob(home_dir + data_dir + '/preprocess4_sub_*.csv')
# import experiment data
for fname in flist:
    with open(fname, 'r') as f:
        all_data = pd.read_csv(f)
        data_all.append(all_data)


## ==========================================================================
# DATA SAVE FOR THE FURTHER ANALYSIS USING R
R_out_dir = 'R_analysis/'

# Puzzle level
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
## glm data
# choice-level



# import data
data_all = []

# directories
home_dir = './'
map_dir = 'active_map/'
data_dir  = 'data/in-lab-pre-pilot/'

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
