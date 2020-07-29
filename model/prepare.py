import json
import pandas as pd

# directories
home_dir = '/Users/Toby/Downloads/bao/'
input_dir = 'road_construction/rc_all_data/data/data_pilot_preprocessed/'
map_dir = 'road_construction/map/active_map/'

with open(home_dir + map_dir + 'basic_map_48_all4','r') as file:
    basic_map = json.load(file) 
    
subs = [1]#,2,4] # subject index 

for sub in subs:
    sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_'+str(sub) + '.csv')
 
    with open(home_dir + input_dir + 'n_repeat_' + str(sub),'r') as file:
        repeats = json.load(file) 


