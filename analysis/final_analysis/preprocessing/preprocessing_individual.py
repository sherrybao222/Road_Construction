# Preprocessing code re-written by Dongjae Kim

import json
import pandas as pd
import numpy as np

from tqdm import tqdm
import os
from glob import glob
# from util.load_and_fit import *

from ast import literal_eval

# ============================================================================
# directories
home_dir = '/Users/dbao/My_Drive'+'/road_construction/data/2022_online/'
# home_dir = 'G:\My Drive\\researches\\nyu\\road-construction-local-dk\data_online_2022/'
map_dir = 'active_map/'
data_dir  = 'data/'


# load basic map
with open(home_dir + map_dir + 'basicMap_action_gap.json', 'r') as file: #
    basic_map = json.load(file)
# load undo map
# with open(home_dir + map_dir + 'undoMap.json', 'r') as file:
#     undo_map = json.load(file)
with open(home_dir + map_dir + 'tree/map_tree', 'r') as file:
    map_tree = json.load(file)
# parameters
subCount = 0
all_city = list(range(1, 30))  # all city names (excluding start)

# find the list of subject files in the data directory
target_expfile = 'FIN_*.csv' # target_expfile = 'PARTICIPANT_RC-Phaser_*.csv'
flist = glob(home_dir + data_dir + target_expfile)

# ============================================================================
while subCount < len(flist):
    
    map_name = []        # whether it is coming from basic or undo.
    trial_id = []
    map_id = []
    action_gap = []  # difficulty of the puzzle
    
    undo_all = []        # indicating whether this move is undo or not (1 or 0)
    submit =[]           # indicating whether this is a submit
    checkEnd =[]         # indicate whether people reach the end (not necessarily submit)
    
    currentChoice = []   # current choice
    chosen_all = []      # all chosen cities at the current choice
    n_city_all = []      # number of connected cities at the current choice
 
    n_within_reach = []  # number of cities within reach
    cities_reach = []     # cities within reach
    budget_all = []      # budget at the current choice

    time_all = []        # raw time in ms. used to calculate rt_all.
    rt_all = []          # reaction time for every clicks
    
    n_opt_paths_all = [] # n of optimal paths at the moment
    n_subopt_paths_all=[]# n of suboptimal (which has #ConnectableCities of opt - 1) paths at the moment.
    mas_all = []         # maximum achievable score

    # tortuosity_all = []  # tortuosity of the path

    ### load a file
    thefile = flist[subCount]
    # read int not str type of data
    with open(thefile, 'r') as f:
        data_exp = pd.read_csv(f, converters={"choiceDyn": literal_eval}, low_memory=False)

    # Drop check trials beforehand for easy code
    data_exp = data_exp.drop(
        data_exp[data_exp.checkTrial == 1].index)

    # find the order of blocks and trials
    condition_blocks = []
    for bi in np.unique(np.array(data_exp.blockID)): # for every unique block ID (0,1,2,3)
        condition_blocks.append(np.unique(np.array(data_exp.condition[data_exp.blockID == bi])).squeeze().tolist()) # save condition of each block

    subName = thefile.split('FIN_')[1].split('.csv')[0]
    subNum = data_exp["subID"][0]
    if not os.path.exists(home_dir + data_dir + 'preprocessed/preprocess4_sub_' 
                          + subName + '.csv'):
        
        # block-wise interation ==============================================
        for t_i in tqdm(range(len(condition_blocks))):
            bi = condition_blocks[t_i]
            basic_index = range(t_i * 23, (t_i + 1) * 23)
            
            # trial-wise interation ==========================================
            for indTrial_ind in tqdm(range(len(basic_index))):
                
                indTrial = basic_index[indTrial_ind]
                single_trial = data_exp[data_exp.trialID.astype(np.int64) == indTrial]
                single_trial = single_trial.reset_index()
                
                if len(single_trial) != 0:
                    
                    currenMapID = int(single_trial['mapID'][0])
                    time = []

                    tortuosity = []
                    # try:
                    #     for node_ in TS.nodes_bucket:
                    #         node_.parent=None
                    #         for node__ in node_.leaves:
                    #             node__.parent = None
                    #     for node_ in TS.nodes_bucket:
                    #         node_.children = ()
                    #     del TS
                    # except:
                    #     print('')
                        
                    # # make tree structure for further analysis.
                    # if bi == 2:
                    #     mmap = data_map(basic_map[currenMapID])
                    # elif bi ==3:
                    #     mmap = data_map(undo_map[currenMapID])

                    # TS = TreeStructure(mmap, map_id = currenMapID)
                    # TS.get_Tree(Node(0,budget = 300))
                    # TS.gen_path_flatten()
                    # # rendered_tree = TS.render_out()
                    # # currNode = TS.root_
                    
                    TS = map_tree[currenMapID]
                    currPath = []
                    
                    # move-wise interation ==========================================
                    for i in range(len(single_trial)): 
                        if (i == 0) or (single_trial['submit'][i] == '1') or (single_trial['cityNr'][i-1] != single_trial['cityNr'][i]):
                            
                            if bi == 2:
                                map_name.append('basic')
                            elif bi == 3:
                                map_name.append('undo')
                            trial_id.append(single_trial['trialID'][i])
                            map_id.append(currenMapID)
                            action_gap.append(basic_map[currenMapID]['action_gap'])
                            
                            currentChoice.append(single_trial['choiceHis'][i])
                            n_city_all.append(len(single_trial['choiceDyn'][i]))
                            chosen_trial = single_trial['choiceDyn'][i]
                            chosen_all.append(chosen_trial)
                            
                            undo_all.append(int(single_trial['undo'][i]))
                            submit.append(int(single_trial['submit'][i]))
                            
                            budget_all.append(single_trial['budgetHis'][i])
                            time.append(int(single_trial['time'][i]))

                            # tortuosity ------------ comment out 21/05/2024
                            # mapid_temp = int(single_trial.mapID[0])
                            # xys = np.array(basic_map[mapid_temp]['xy'])
                            # if len(single_trial['choiceDyn'][i]) > 1: # for those connected at least 2
                            #     if len(single_trial['choiceDyn'][i]) == 2: # it should be one when two cities are connected
                            #         tortuosity.append(1)
                            #     else:
                            #         # xy_0 = xys[int(single_trial['choiceHis'][0]),:]
                            #         # xy_n = xys[int(single_trial['choiceHis'][i]),:]
                            #         tortuos_len = float(single_trial['budgetHis'][0]) - float(single_trial['budgetHis'][i])
                            #         distance_ = np.sqrt(np.sum((xys[int(single_trial['choiceHis'][i]),:] - xys[int(single_trial['choiceHis'][0]),:])**2))
                            #         tortuosity.append(tortuos_len/distance_)
                            # else:
                            #     tortuosity.append(0)
                            
                            # get unconnected cities within reach for each step
                            remain_trial = [x for x in all_city if x not in chosen_trial]
                            u_trial = []
                            for c in remain_trial:
                                if basic_map[currenMapID]['distance'][int(
                                        single_trial['choiceHis'][i])][int(c)] <= float(
                                            single_trial['budgetHis'][i]):
                                    u_trial.append(c)
                            cities_reach.append(u_trial)
                            n_within_reach.append(len(u_trial))
                            
                            if (len(u_trial)==0):
                                checkEnd.append(1)
                            else:
                                checkEnd.append(0)
                            
                            # if (int(single_trial['submit'][i]) != 1):
                            #     # path that up-to-date for path
                            #     if int(single_trial['undo'][i]) == 0:
                            #         currPath.append(int(single_trial['choiceHis'][i]))
                            #     elif len(currPath) >= 2:
                            #         currPath.pop(-1)
                            currPath = single_trial['choiceDyn'][i]
                            ind_currpath = np.where([currPath == TS['paths'][j] for j in range(len(TS['paths']))])[0].squeeze().tolist()

                            # # # reachable leaf-nodes
                            # nodes_filtered = TS.nodes_bucket[ind_currpath].leaves
                            # MAS_eachPath = [len(j.path) for j in nodes_filtered]

                            # max_MAS = max(MAS_eachPath)
                            # if i == 0:
                            #     print(len(nodes_filtered))
                            # MAS_eachPath = TS.mas[ind_currpath]
                            # max_MAS = max(MAS_eachPath)
                            
                            # MAS.append(max_MAS)
                            # optPaths.append(np.sum(np.array(MAS_eachPath) == max_MAS))
                            # suboptPaths.append(np.sum(np.array(MAS_eachPath) == (max_MAS-1)))
                            
                            mas_all.append(TS['mas'][ind_currpath])
                            n_opt_paths_all.append(TS['optpath'][ind_currpath])
                            n_subopt_paths_all.append(TS['suboptpath'][ind_currpath])
                            
                            if (currenMapID == 39) and (i==0):
                                print('*'*20)
                                print(indTrial)
                                print('')

                    time_all.extend(time)
                    rt_all.extend([-1, *[a - time[j-1] for j, a in enumerate(time) if j > 0]])

                    # tortuosity_all.extend(tortuosity)
                    
        data = {
            'condition': map_name,
            'trial_id': trial_id,
            'map_id': map_id,
            'action_gap': action_gap,
            
            'undoIndicator': undo_all,
            'submit': submit,
            'checkEnd': checkEnd,
            
            'currentChoice': currentChoice,
            'chosen_all': chosen_all,
            'n_city_all': n_city_all,
            
            'n_within_reach': n_within_reach,
            'cities_reach': cities_reach,
            'currentBudget': budget_all,            
            
            'time_all': time_all,
            'rt_all': rt_all,
            
            'n_opt_paths_all': n_opt_paths_all,
            'n_subopt_paths_all': n_subopt_paths_all,
            'mas_all': mas_all

            # 'tortuosity_all':tortuosity_all
            }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.to_csv(home_dir + data_dir + 'preprocessed/preprocess4_sub_' + subName + '.csv', index=False)

    subCount = subCount + 1