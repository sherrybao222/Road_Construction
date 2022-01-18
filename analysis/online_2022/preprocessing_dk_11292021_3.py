# Preprocessing code re-written by Dongjae Kim

import json
import pandas as pd
import numpy as np

from tqdm import tqdm
import os
from glob import glob
from util.load_and_fit import *

from ast import literal_eval

# ============================================================================
# directories
home_dir = '/Users/dbao/google_drive_db'+'/road_construction/data/2022_online/'
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
all_city = list(range(1, 30))  # all city names (excluding start)

# find the list of subject files in the data directory
target_expfile = 'FIN_*_RC-Phaser_*.csv' # target_expfile = 'PARTICIPANT_RC-Phaser_*.csv'
flist = glob(home_dir + data_dir + target_expfile)

# ============================================================================
while subCount < len(flist):
    
    map_name = [] # whether it is coming from basic or undo.
    trial_id = []
    map_id = []
    
    choice_all = []  # all single choice
    chosen_all = []  # all chosen cities at the single choice
    n_city_all = []  # number of connected cities at the choice

    n_within_reach = [] # number of cities within reach
    u_trial_all = []  # cities within reach
    budget_all = []  # budget at the choice
    undo_all = [] # indicating whether the click is undo or not (1 or 0)

    time_all = [] # raw time in ms. used to calculate rt_all.
    rt_all = [] # reaction time for every clicks
    
    n_opt_paths_all = [] # n of optimal paths at the moment
    n_subopt_paths_all = [] # n of suboptimal (which has #ConnectableCities of opt - 1) paths at the moment.
    mas_all = [] # maximum achievable score
    # error_all =[] # this can be inferred from mas_all.

    # load a file
    thefile = flist[subCount]
    # read int not str type of data
    with open(thefile, 'r') as f:
        data_exp = pd.read_csv(f, converters={"choiceDyn": literal_eval})

    # Drop check trials beforehand for easy code
    data_exp = data_exp.drop(
        data_exp[data_exp.checkTrial == 1].index)

    # find the order of blocks and trials
    condition_blocks = []
    # trials_index_blocks =[]
    for bi in np.unique(np.array(data_exp.blockID)): # for every unique block ID (0,1,2,3)
        condition_blocks.append(np.unique(np.array(data_exp.condition[data_exp.blockID == bi])).squeeze().tolist()) # save condition of each block

    subName = thefile.split('RC-Phaser_')[1].split('.csv')[0]
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
                    TS.get_Tree(Node(0,budget = 300))
                    TS.gen_path_flatten()
                    # rendered_tree = TS.render_out()
                    # currNode = TS.root_
                    
                    currPath = []
                    for i in range(len(single_trial)):
                        if (i == 0) or (single_trial['cityNr'][i-1] !=
                                        single_trial['cityNr'][i]):

                            trialId.append(single_trial['trialID'][i])
                            choiceHis.append(single_trial['choiceHis'][i])
                            choiceLen.append(len(single_trial['choiceDyn'][i]))
                            budgetHis.append(single_trial['budgetHis'][i])
                            time.append(int(single_trial['time'][i]))
                            undo.append(int(single_trial['undo'][i]))
                            
                            if bi == 2:
                                mapName.append('basic')
                            elif bi == 3:
                                mapName.append('undo')

                            # get all unconnected cities for each step
                            chosen_trial = single_trial['choiceDyn'][i]
                            chosen_all.append(chosen_trial)
                            remain_trial = [x for x in all_city if x not in chosen_trial]
                            # get unconnected cities within reach for each step
                            u_trial = []
                            for c in remain_trial:
                                if basic_map[currenMapID]['distance'][int(
                                        single_trial['choiceHis'][i])][int(c)] <= float(
                                            single_trial['budgetHis'][i]):
                                    u_trial.append(c)
                            u_trial_all.append(u_trial)

                            # path that up-to-date for path
                            if int(single_trial['undo'][i]) == 0:
                                currPath.append(int(single_trial['choiceHis'][i]))
                            else:
                                currPath.pop(-1)

                            ind_currpath = np.where([currPath == TS.paths[j] for j in range(len(TS.paths))])[0].squeeze().tolist()

                            # # reachable leaf-nodes
                            nodes_filtered = TS.nodes_bucket[ind_currpath].leaves
                            MAS_eachPath = [len(j.path) for j in nodes_filtered]
                            withinReach.append(len(TS.nodes_bucket[ind_currpath].children))

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
                    map_name.extend(mapName)
                    trial_id.extend(trialId)
                    map_id_single = [currenMapID] * len(choiceHis)
                    map_id.extend(map_id_single)
                    
                    choice_all.extend(choiceHis)
                    n_city_all.extend(choiceLen)
                    budget_all.extend(budgetHis)
                    undo_all.extend(undo)
                    
                    time_all.extend(time)
                    rt_all.extend([0, *[a - time[j-1] for j, a in enumerate(time) if j > 0]])
                    
                    n_opt_paths_all.extend(optPaths)
                    n_subopt_paths_all.extend(suboptPaths)
                    mas_all.extend(MAS)
                    n_within_reach.extend(withinReach)

        data = {
            'map_name': map_name,
            'trial_id': trial_id,
            'map_id': map_id,
            
            'choice_all': choice_all,
            'chosen_all': chosen_all,
            'n_city_all': n_city_all,
            
            'n_within_reach': n_within_reach,
            'u_trial_all': u_trial_all,
            'budget_all': budget_all,            
            'undo_all': undo_all,
            
            'time_all': time_all,
            'rt_all': rt_all,
            
            'n_opt_paths_all': n_opt_paths_all,
            'n_subopt_paths_all': n_subopt_paths_all,
            'mas_all': mas_all}
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.to_csv(home_dir + data_dir + '/preprocess4_sub_' + subName +
                  '.csv', index=False)

    subCount = subCount + 1