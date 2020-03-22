import json
import numpy as np

data_all = []

# import experiment data
for num in [1,2,4]:
    with open('/Users/sherrybao/Downloads/Research/Road_Construction/data_copy/data_pilot/sub_'+str(num)+'/test_all_'+str(num),'r') as file: 
        all_data = json.load(file)
        data_all.append(all_data)
        
# import map
with open('/Users/sherrybao/Downloads/Research/Road_Construction/pilot_031220/map/num_48','r') as file: 
    num_map = json.load(file) 
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map_48_all4','r') as file: 
    rc_map = json.load(file) 

# map index
num_ind = []
rc_ind = []
undo_ind = []
# participant answer
num = []
rc = []
undo = []
# correct answer
num_list = num_map[2]*3
rc_list = rc_map[2]*3
budget_list = []

for data in data_all:
    for i in range(len(data[0])):
        if data[0][i]['cond'][-1] == 1:
            num.append(int(data[0][i]['num_est'][-1]))
            budget_list.append(data[0][i]['total'])
            num_ind.append(i)
        if data[0][i]['cond'][-1] == 2:
            rc.append(data[0][i]['n_city'][-1])
            rc_ind.append(i)
        if data[0][i]['cond'][-1] == 3:
            undo.append(data[0][i]['n_city'][-1])
            undo_ind.append(i)

# organize data into a matrix    
mx_num = np.column_stack((num_ind, budget_list, num, num_list)) 
mx_rc = np.column_stack((rc_ind, rc, rc_list)) 
mx_undo = np.column_stack((undo_ind, undo, rc_list)) 

#temp = mx_num.view(np.ndarray)
#sorted_mx = temp[np.lexsort((temp[:, 1], ))] # sort by budget
#temp = mx_rc.view(np.ndarray)
#sorted_mx_rc = temp[np.lexsort((temp[:, 2], ))] # sort by optimal
#temp = mx_undo.view(np.ndarray)
#sorted_mx_undo = temp[np.lexsort((temp[:, 2], ))] # sort by budget


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x = mx_num[:,3]
y = mx_num[:,2]
ax.scatter(x,y,alpha=0.2)

ax.set_xlim((0,11))
ax.set_ylim((0,11))
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))

ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3") # diagnal
ax.grid(b=True, which='major', color='k', linestyle='--')

plt.xticks(np.arange(x0,x1, 1.0))
plt.yticks(np.arange(y0,y1, 1.0))
plt.xlabel("correct answer in number estimation")
plt.ylabel("reported answer in number estimation")
#ax.set_aspect('equal')
fig.savefig('num_est.png',dpi=600)
plt.close(fig)

# ---------------------------------------------------
fig, ax = plt.subplots()
x = mx_rc[:,2]
y = mx_rc[:,1]
ax.scatter(x,y,alpha=0.2)

ax.set_xlim((4,12))
ax.set_ylim((4,12))
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))

ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3") # diagnal
ax.grid(b=True, which='major', color='k', linestyle='--')

plt.xticks(np.arange(x0,x1, 1.0))
plt.yticks(np.arange(y0,y1, 1.0))
plt.xlabel("optimal answer in road construction w/o undo")
plt.ylabel("reported answer in road construction w/o undo")
#ax.set_aspect('equal')
fig.savefig('rc.png',dpi=600)
plt.close(fig)

# ---------------------------------------------------
fig, ax = plt.subplots()
x = mx_undo[:,1]
y = mx_rc[:,1]
ax.scatter(x,y,alpha=0.2)

ax.set_xlim((4,12))
ax.set_ylim((4,12))
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))

ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3") # diagnal
ax.grid(b=True, which='major', color='k', linestyle='--')

plt.xticks(np.arange(x0,x1, 1.0))
plt.yticks(np.arange(y0,y1, 1.0))
plt.xlabel("reported answer in road construction w/ undo")
plt.ylabel("reported answer in road construction w/o undo")
#ax.set_aspect('equal')
fig.savefig('rc_undo.png',dpi=600)
plt.close(fig)
