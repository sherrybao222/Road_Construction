import json
import pandas as pd
import numpy as np

## directories
# home_dir = '/Users/dbao/google_drive/'
# input_dir = 'road_construction/experiments/pilot_0320/data/data_pilot_preprocessed/'
# map_dir = 'road_construction/experiments/pilot_0320/map/active_map/'

# hpc directories
home_dir = '/home/db4058/road_construction/data/'
input_dir = 'data_pilot_preprocessed/'
map_dir = 'active_map/'

with open(home_dir + map_dir + 'basic_map_48_all4','r') as file:
    basic_map = json.load(file) 
    
subs = [1]#,2,4] # subject index 

for sub in subs:
    sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_'+str(sub) + '.csv')
    LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
    # # for trial-dependant repeats
    # with open(home_dir + input_dir + 'n_repeat_' + str(sub),'r') as file:
    #     repeats = json.load(file) 

    # general repeats
    repeats = 20
