'''
    choose maps meeting a certain criteria
    which has a difference of 4 cities for optimal and greedy path
'''

import scipy.io as sio

## load maps from mat file
#basic_1 = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic_map.mat',  struct_as_record=False)['map_list'][0]
#basic_2 = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic_map_500.mat',  struct_as_record=False)['map_list'][0]
#
#diff_list_1 = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic_summary.mat',  struct_as_record=False)['diff_list'][0]
#diff_list_2 = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic_summary_500.mat',  struct_as_record=False)['diff_list'][0]

# load map from json
map_path = '/Users/dbao/google_drive_db/road_construction/data/test_2021/experiment/map/map_pool/'
save_path = '/Users/dbao/google_drive_db/road_construction/data/test_2021/experiment/map/active_map/'

import json

# every file is 500 maps
with open(map_path+'basic_map_2000','r') as file: 
    basic_all = json.load(file)
with open(map_path+'basic_summary_2000','r') as file: 
    summary = json.load(file)
    diff_list = summary[0] 
    optimal_list = summary[1] 
    optimal_number = summary[3] 
    
with open(map_path+'basic_map_2500','r') as file: 
    basic_all.extend(json.load(file))
with open(map_path+'basic_summary_2500','r') as file: 
    summary = json.load(file)
    diff_list.extend(summary[0]) 
    optimal_list.extend(summary[1]) 
    optimal_number.extend(summary[3]) 
    

basic_map = []
optimal = []
optimal_n = []
ind = 0

def filter_map(basic_map, basic_all, diff_list, optimal_list, optimal_number,ind):
    if diff_list[ind] == 4:
        basic_map.append(basic_all[ind])
        optimal.append(optimal_list[ind])
        optimal_n.append(optimal_number[ind])
#    elif diff_list[ind] == 2 and num[1] < 6:
#        basic_map.append(basic_all[ind])
#        num[1] = num[1] + 1
#    elif diff_list[ind] == 3 and num[2] < 6:
#        basic_map.append(basic_all[ind])
#        num[2] = num[2] + 1
#    elif diff_list[ind] == 4 and num[3] < 6:
#        basic_map.append(basic_all[ind])
#        num[3] = num[3] + 1

while len(basic_map) < 20 and ind <2000:
    filter_map(basic_map, basic_all, diff_list, optimal_list, optimal_number,ind)
    ind = ind + 1
#
# saving json
with open(save_path+'basic_map_training_20','w') as file: 
    json.dump((basic_map,optimal,optimal_n),file)            
