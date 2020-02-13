import scipy.io as sio

## load maps from mat file
#basic_1 = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic_map.mat',  struct_as_record=False)['map_list'][0]
#basic_2 = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic_map_500.mat',  struct_as_record=False)['map_list'][0]
#
#diff_list_1 = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic_summary.mat',  struct_as_record=False)['diff_list'][0]
#diff_list_2 = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic_summary_500.mat',  struct_as_record=False)['diff_list'][0]

# load map from json
import json
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_map','r') as file: 
    basic_1 = json.load(file)
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_summary','r') as file: 
    diff_list_1 = json.load(file)[0] 
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_map_500','r') as file: 
    basic_2 = json.load(file)
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_summary_500','r') as file: 
    diff_list_2 = json.load(file)[0] 

basic_map = []

num_1 = 0
num_2 = 0
num_3 = 0
num_4 = 0

for ind in range(500):
    if diff_list_1[ind] == 1 and num_1 < 6:
        basic_map.append(basic_1[ind])
        num_1 = num_1 + 1
    elif diff_list_1[ind] == 2 and num_2 < 6:
        basic_map.append(basic_1[ind])
        num_2 = num_2 + 1
    elif diff_list_1[ind] == 3 and num_3 < 6:
        basic_map.append(basic_1[ind])
        num_3 = num_3 + 1
    elif diff_list_1[ind] == 4 and num_4 < 6:
        basic_map.append(basic_1[ind])
        num_4 = num_4 + 1

if len(basic_map) < 24:
    for ind in range(500):
        if diff_list_2[ind] == 1 and num_1 < 6:
            basic_map.append(basic_2[ind])
            num_1 = num_1 + 1
        elif diff_list_2[ind] == 2 and num_2 < 6:
            basic_map.append(basic_2[ind])
            num_2 = num_2 + 1
        elif diff_list_2[ind] == 3 and num_3 < 6:
            basic_map.append(basic_2[ind])
            num_3 = num_3 + 1
        elif diff_list_2[ind] == 4 and num_4 < 6:
            basic_map.append(basic_2[ind])
            num_4 = num_4 + 1
#
# pickle
with open('basic_map_24','w') as file: 
    json.dump(basic_map,file)            
# saving mat
sio.savemat('basic_map_24.mat', {'map_list':basic_map})
