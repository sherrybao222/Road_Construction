'''
preprocess raw data for basic condition
generate a sheet for each board position (choice) and corresponding feature
'''

import json
import pandas as pd

import fnmatch # match file name
import os

# directories
home_dir = '/Users/dbao/google_drive_db/road_construction/'
map_dir = 'data/pilot_03 2020/experiment/map/active_map/'
data_dir  = 'data/test_2021/'    

# load map
with open(home_dir + map_dir + 'basic_map_48_all4','r') as file:
    basic_map = json.load(file) 

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

while subCount < 1:
    
    map_id = [] 
    choice_all = []  # all single choice
    chosen_all = []  # all chosen cities at the single choice
    remain_all = []  # remaining cities at the single choice
    budget_all = []  # budget at the choice
    n_city_all = []  # number of connected cities at the choice
    n_u_all = []  # n of cities within reach
    u_trial_all = []  # cities within reach

    # find files with name...
    target_expfile = 'PARTICIPANT_RC-Phaser_*.csv' 
    for thefile in os.listdir(home_dir+data_dir):
        if fnmatch.fnmatch(thefile, target_expfile):
            with open( home_dir+data_dir + thefile,'r') as file: 
                data_exp = pd.read_csv(file) 
    
    subNum = data_exp["subID"][0]
    for indTrial in basic_index:        
        single_trial = data_exp[data_exp["trialID"] == indTrial]
        single_trial = single_trial.drop(single_trial[single_trial.checkTrial == 1].index)
        single_trial = single_trial.reset_index()
        
        currenMapID = int(single_trial['mapID'][0])
        choiceHis = []
        budgetHis = []
        time = []
        undo = []
        
        for i in range(0,len(single_trial)):
            if (i==0)or(single_trial['cityNr'][i-1] != single_trial['cityNr'][i]):
                choiceHis.append(single_trial['choiceHis'][i])
                budgetHis.append(single_trial['budgetHis'][i])
                time.append(single_trial['time'][i])
                undo.append(single_trial['undo'][i])
                
                # get all unconnected cities for each step    
                chosen_trial = json.loads(single_trial['choiceDyn'][i])
                chosen_all.append(chosen_trial)
                remain_trial = [x for x in all_city if x not in chosen_trial]
                remain_all.append(remain_trial)
                
                # get unconnected cities within reach for each step    
                n_u = 0
                u_trial = []
                for c in remain_trial:
                    if basic_map[0][currenMapID]['distance'][int(
                        single_trial['choiceHis'][i])][int(c)] <= float(
                            single_trial['budgetHis'][i]):
                        n_u = n_u + 1
                        u_trial.append(c)
                n_u_all.append(n_u)
                u_trial_all.append(u_trial)
        
        # add to the dataframe of all maps
        choice_all.extend(choiceHis)
        budget_all.extend(budgetHis)
        n_city_all.extend(range(1,len(choiceHis)+1))
        map_id_single = [currenMapID] * len(choiceHis)
        map_id.extend(map_id_single)
    
    data = {'map_id': map_id, 
            'choice_all': choice_all, 
            'budget_all': budget_all, 
            'n_city_all': n_city_all,
            'n_u_all': n_u_all,
            'u_trial_all': u_trial_all,
            'chosen_all': chosen_all, 
            'remain_all': remain_all}
    # Create DataFrame 
    df = pd.DataFrame(data) 
    df.to_csv(home_dir + data_dir + 'organized/preprocess_sub_'+str(subNum) + '.csv', index=False)
    
    subCount = subCount + 1