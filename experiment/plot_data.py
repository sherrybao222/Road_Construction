import json
import numpy as np

# import experiment data
with open('/Users/sherrybao/Downloads/Research/Road_Construction/experiment/data_002/test_all','r') as file: 
    all_data = json.load(file)
# import num_est map
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/num_48','r') as file: 
    num_map = json.load(file) 
# map index
ind_map = list(range(0,24))
ind_map.extend(list(range(24*5,24*6)))

ans_list = []
correct_list = num_map[2] # correct answer
budget_list = []
map_id = list(range(48))

for ind in ind_map:
    ans_list.append(int(all_data[ind]['num_est'][-1]))
    budget_list.append(all_data[ind]['total'])
    
mx = np.column_stack((map_id, budget_list, ans_list, correct_list)) # organize data into a matrix

temp = mx.view(np.ndarray)
sorted_mx = temp[np.lexsort((temp[:, 1], ))] # sort by budget

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x = sorted_mx[:,3]
y = sorted_mx[:,2]
ax.scatter(x,y)

ax.set_xlim((0,11))
ax.set_ylim((0,11))
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))

ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3") # diagnal
ax.grid(b=True, which='major', color='k', linestyle='--')

plt.xticks(np.arange(x0,x1, 1.0))
plt.yticks(np.arange(y0,y1, 1.0))
plt.xlabel("correct answer")
plt.ylabel("reported answer")
#ax.set_aspect('equal')
#fig.savefig('test.png', dpi=600)
#plt.close(fig)
