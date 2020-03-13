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
    basic_all = json.load(file)
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_summary','r') as file: 
    diff_list = json.load(file)[0] 
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_map_500','r') as file: 
    basic_all.extend(json.load(file))
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_summary_500','r') as file: 
    diff_list.extend(json.load(file)[0]) 
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_map_1000','r') as file: 
    basic_all.extend(json.load(file))
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_summary_1000','r') as file: 
    diff_list.extend(json.load(file)[0]) 
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_map_1500','r') as file: 
    basic_all.extend(json.load(file))
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_summary_1500','r') as file: 
    diff_list.extend(json.load(file)[0]) 


basic_map = []
num = [0,0,0,0]
ind = 0

def filter_map(basic_map, basic_all, diff_list, num, ind):
    if diff_list[ind] == 4 and num[0] < 48:
        basic_map.append(basic_all[ind])
        num[0] = num[0] + 1
#    elif diff_list[ind] == 2 and num[1] < 6:
#        basic_map.append(basic_all[ind])
#        num[1] = num[1] + 1
#    elif diff_list[ind] == 3 and num[2] < 6:
#        basic_map.append(basic_all[ind])
#        num[2] = num[2] + 1
#    elif diff_list[ind] == 4 and num[3] < 6:
#        basic_map.append(basic_all[ind])
#        num[3] = num[3] + 1


while len(basic_map) < 48 and ind < 2000:
    filter_map(basic_map, basic_all, diff_list, num, ind)
    ind = ind + 1
#
# json
with open('basic_map_48_all4','w') as file: 
    json.dump(basic_map,file)            
# saving mat
sio.savemat('basic_map_48_all4.mat', {'map_list':basic_map})
