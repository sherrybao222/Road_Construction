import json
import numpy as np

with open('/Users/sherrybao/Downloads/Research/Road_Construction/experiment/data_002/test_all','r') as file: 
    all_data = json.load(file)
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/num_48','r') as file: 
    num_map = json.load(file) 

ind_map = list(range(0,24))
ind_map.extend(list(range(24*5,24*6)))

ans_list = []
correct_list = num_map[2]
budget_list = []
map_id = list(range(48))

for ind in ind_map:
    ans_list.append(int(all_data[ind]['num_est'][-1]))
    budget_list.append(all_data[ind]['total'])
    
mx = np.column_stack((map_id, budget_list, ans_list, correct_list))

temp = mx.view(np.ndarray)
sorted_mx = temp[np.lexsort((temp[:, 1], ))]

import matplotlib.pyplot as plt
plt.scatter(map_id,sorted_mx[:,2])
plt.scatter(map_id,sorted_mx[:,3])

