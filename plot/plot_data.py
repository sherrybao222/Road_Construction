import json
import numpy as np
from statistics import mean,stdev
from operator import eq
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import math

data_all = []

# import experiment data
for num in [1,2,4]:
    with open('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/data_copy/data_pilot/sub_'+str(num)+'/test_all_'+str(num),'r') as file: 
        all_data = json.load(file)
        data_all.append(all_data)
        
# import map
with open('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/pilot_031220/map/num_48','r') as file: 
    num_map = json.load(file) 
with open('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/map/active_map/basic_map_48_all4','r') as file: 
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
# time for a trial
t_rc = []
t_undo = []
f_t_rc = []
f_t_undo = []
undo_click = []
# statistics list
mean_rc = []
mean_undo = []
ac_num = [] # accuracy for num_est
pc_undo = [] # Percentage of using undos among all undo trials
mean_t_rc = []
mean_t_undo = []# Average time spent in RC VS RCU across maps
mean_f_t_rc = []
mean_f_t_undo = []# Average time spent in RC VS RCU across maps


for data in data_all:
    for i in range(len(data[0])):
        if data[0][i]['cond'][-1] == 1:
            num.append(int(data[0][i]['num_est'][-1]))
            budget_list.append(data[0][i]['total'])
            num_ind.append(i)
        if data[0][i]['cond'][-1] == 2:
            rc.append(data[0][i]['n_city'][-1])
            t_rc.append(data[0][i]['time'][-1]-data[0][i]['time'][0])
            ind_choice = next(x for x, val in enumerate(data[0][i]['choice_his']) 
                                  if val != 0)
            f_t_rc.append(data[0][i]['time'][ind_choice]-data[0][i]['time'][0])
            rc_ind.append(i)
        if data[0][i]['cond'][-1] == 3:
            undo.append(data[0][i]['n_city'][-1])
            t_undo.append(data[0][i]['time'][-1]-data[0][i]['time'][0])
            ind_choice = next(x for x, val in enumerate(data[0][i]['choice_his']) 
                                  if val != 0)
            f_t_undo.append(data[0][i]['time'][ind_choice]-data[0][i]['time'][0])
            if 1 in set(data[0][i]['undo_press']):
                undo_click.append(1)
            else:
                undo_click.append(0)
            undo_ind.append(i)

bool_numest = map(eq, num, num_list)
int_numest = list(np.array(list(bool_numest)).astype(float))

mx_num = [0]*3
sorted_mx = [0]*3

for i in range(0,3):
    mean_rc.append(mean(rc[48*i:48*(i+1)]))
    mean_undo.append(mean(undo[48*i:48*(i+1)]))
    mean_t_rc.append(mean(t_rc[48*i:48*(i+1)]))
    mean_t_undo.append(mean(t_undo[48*i:48*(i+1)]))
    mean_f_t_rc.append(mean(f_t_rc[48*i:48*(i+1)]))
    mean_f_t_undo.append(mean(f_t_undo[48*i:48*(i+1)]))
    pc_undo.append(mean(undo_click[48*i:48*(i+1)]))
    ac_num.append(mean(int_numest[48*i:48*(i+1)]))

    mx_num[i] = np.column_stack((num_ind[48*i:48*(i+1)], budget_list[48*i:48*(i+1)], num[48*i:48*(i+1)], num_list[48*i:48*(i+1)],int_numest[48*i:48*(i+1)])) 
    temp = mx_num[i].view(np.ndarray)
    sorted_mx[i] = temp[np.lexsort((temp[:, 1], ))] # sort by budget

budget_numest = []
for i in range(0,3):
    sub = []
    for j in range(0,3):
        sub.append(mean(sorted_mx[i][16*j:16*(j+1),4]))
    budget_numest.append(sub)
    
rc_undo = np.zeros((12,12))
for ind,val in enumerate(rc):
        rc_undo[val,undo[ind]] = rc_undo[val,undo[ind]]+1/len(rc)
num_mx = np.zeros((11,11))
for ind,val in enumerate(num):
        num_mx[val,num_list[ind]] = num_mx[val,num_list[ind]]+1/len(num)
rc_mx = np.zeros((12,12))
for ind,val in enumerate(rc):
        rc_mx[val,rc_list[ind]] = rc_mx[val,rc_list[ind]]+1/len(rc)

# =============================================================================
# plot
fig, ax = plt.subplots()
#ax.scatter(x,y,alpha=0.2)
ax = sns.heatmap(num_mx,cmap="YlGnBu",linewidths=.3,linecolor = 'k')

ax.set_xlim((0,11))
ax.set_ylim((0,11))
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))

ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3") # diagnal
#ax.grid(b=True, which='major', color='k', linestyle='--')

#plt.xticks(np.arange(x0,x1, 1.0))
#plt.yticks(np.arange(y0,y1, 1.0))
plt.xlabel("correct answer in number estimation")
plt.ylabel("reported answer in number estimation")
#ax.set_aspect('equal')
fig.savefig('num_est.png',dpi=600)
plt.close(fig)

# ---------------------------------------------------
fig, ax = plt.subplots()
#ax.scatter(x,y,alpha=0.2)
ax = sns.heatmap(rc_mx,cmap="YlGnBu",linewidths=.3,linecolor = 'k')

ax.set_xlim((4,12))
ax.set_ylim((4,12))
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))

ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3") # diagnal

#plt.xticks(np.arange(x0,x1, 1.0))
#plt.yticks(np.arange(y0,y1, 1.0))
plt.xlabel("optimal answer in road construction w/o undo")
plt.ylabel("reported answer in road construction w/o undo")
ax.grid(b=True, which='both', color='b', linestyle='--')

#ax.set_aspect('equal')
fig.savefig('rc.png',dpi=600)
plt.close(fig)

# ---------------------------------------------------
fig, ax = plt.subplots()
ax = sns.heatmap(rc_undo,cmap="YlGnBu",linewidths=.3,linecolor = 'k')

ax.set_xlim((4,12))
ax.set_ylim((4,12))
x0,x1 = ax.get_xlim()
y0,y1 = ax.get_ylim()
ax.set_aspect(abs(x1-x0)/abs(y1-y0))

ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3") # diagnal

plt.xlabel("reported answer in road construction w/ undo")
plt.ylabel("reported answer in road construction w/o undo")
#ax.set_aspect('equal')
fig.savefig('rc_undo.png',dpi=600)
plt.close(fig)

# ---------------------------------------------------
ind = [0.5,0.8]
err_rc = stdev(mean_rc)/math.sqrt(len(mean_rc))
err_undo = stdev(mean_undo)/math.sqrt(len(mean_undo))
fig, ax = plt.subplots()
ax.bar(ind,[mean(mean_rc),mean(mean_undo)],width = 0.1,
       color = '#66cccc',edgecolor='k')
plotline1, caplines1, barlinecols1 = ax.errorbar(ind, [mean(mean_rc),mean(mean_undo)], yerr=[err_rc,err_undo], lolims=True, capsize = 0, ls='None', color='k')
caplines1[0].set_marker('_')
caplines1[0].set_markersize(7)
ax.set_ylim((7,9))
ax.set_xticks([0.3,0.5,0.8,1])
ax.set_xticklabels(['','w/o undo','w/ undo',' '])
ax.grid(b=True, which='major', axis = 'y',color='k', linestyle='--')
ax.set_facecolor('white')
ax.spines['bottom'].set_color('k')
ax.spines['left'].set_color('k')
ax.tick_params(axis='y', colors='k')
plt.ylabel('connected number of cities')

fig.savefig('av_rc.png',dpi=600)
plt.close(fig)

# ---------------------------------------------------
ind = [0.5,0.8]
err_rc = stdev(mean_t_rc)/math.sqrt(len(mean_rc))
err_undo = stdev(mean_t_undo)/math.sqrt(len(mean_undo))
fig, ax = plt.subplots()
ax.bar(ind,[mean(mean_t_rc),mean(mean_t_undo)],width = 0.1,
       color = '#eed06f',edgecolor='k')
plotline1, caplines1, barlinecols1 = ax.errorbar(ind, [mean(mean_t_rc),mean(mean_t_undo)], yerr=[err_rc,err_undo], lolims=True, capsize = 0, ls='None', color='k')
caplines1[0].set_marker('_')
caplines1[0].set_markersize(7)
ax.set_ylim((10,45))
ax.set_xticks([0.3,0.5,0.8,1])
ax.set_xticklabels(['','w/o undo','w/ undo',' '])
ax.grid(b=True, which='major', axis = 'y',color='k', linestyle='--')
ax.set_facecolor('white')
ax.spines['bottom'].set_color('k')
ax.spines['left'].set_color('k')
ax.tick_params(axis='y', colors='k')
plt.ylabel('trial duration (s)')

fig.savefig('t_rc.png',dpi=600)
plt.close(fig)

# ---------------------------------------------------
ind = [0.5,0.8]
err_rc = stdev(mean_f_t_rc)/math.sqrt(len(mean_rc))
err_undo = stdev(mean_f_t_undo)/math.sqrt(len(mean_undo))
fig, ax = plt.subplots()
ax.bar(ind,[mean(mean_f_t_rc),mean(mean_f_t_undo)],width = 0.1,
       color = '#eed06f',edgecolor='k')
plotline1, caplines1, barlinecols1 = ax.errorbar(ind, [mean(mean_f_t_rc),mean(mean_f_t_undo)], yerr=[err_rc,err_undo], lolims=True, capsize = 0, ls='None', color='k')
caplines1[0].set_marker('_')
caplines1[0].set_markersize(7)
ax.set_ylim((0,20))
ax.set_xticks([0.3,0.5,0.8,1])
ax.set_xticklabels(['','w/o undo','w/ undo',' '])
ax.grid(b=True, which='major', axis = 'y',color='k', linestyle='--')
ax.set_facecolor('white')
ax.spines['bottom'].set_color('k')
ax.spines['left'].set_color('k')
ax.tick_params(axis='y', colors='k')
plt.ylabel('time spent until first move (s)')

fig.savefig('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/plot/fig/f_t_rc.png',dpi=600)
plt.close(fig)

# ---------------------------------------------------
ind = [0.5]
err = stdev(pc_undo)/math.sqrt(len(pc_undo))
fig, ax = plt.subplots()
ax.bar(ind,mean(pc_undo),width = 0.1,
       color = '#eed06f',edgecolor='k')
plotline1, caplines1, barlinecols1 = ax.errorbar(ind, mean(pc_undo), yerr=err, lolims=True, capsize = 0, ls='None', color='k')
caplines1[0].set_marker('_')
caplines1[0].set_markersize(7)
ax.set_ylim((0,1))
ax.set_xticks([0.3,0.5,0.7])
ax.set_xticklabels(['','w/ undo',''])
ax.grid(b=True, which='major', axis = 'y',color='k', linestyle='--')
ax.set_facecolor('white')
ax.spines['bottom'].set_color('k')
ax.spines['left'].set_color('k')
ax.tick_params(axis='y', colors='k')
plt.ylabel('undo percentage')

fig.savefig('pc_undo.png',dpi=600)
plt.close(fig)

# ----------------------------------------------------
ind = [0.5,0.8,1.1]
err = np.std(budget_numest,axis=0)/math.sqrt(len(mean_rc))
fig, ax = plt.subplots()
ax.bar(ind,np.mean(budget_numest,axis = 0),width = 0.1,
       color = '#99cccc',edgecolor='k')
plotline1, caplines1, barlinecols1 = ax.errorbar(ind, np.mean(budget_numest,axis = 0), yerr=err, lolims=True, capsize = 0, ls='None', color='k')
caplines1[0].set_marker('_')
caplines1[0].set_markersize(7)
ax.set_ylim((0,1))
ax.set_xticks([0.3,0.5,0.8,1.1,1.3])
ax.set_xticklabels(['',200,350,500,''])
ax.grid(b=True, which='major', axis = 'y',color='k', linestyle='--')
ax.set_facecolor('white')
ax.spines['bottom'].set_color('k')
ax.spines['left'].set_color('k')
ax.tick_params(axis='y', colors='k')
plt.ylabel('accuracy in number estimation')
plt.xlabel('budget length')

fig.savefig('av_num.png',dpi=600)
plt.close(fig)

# =============================================================================
# statistical tests
from scipy.stats import wilcoxon
stat, p = wilcoxon(rc, undo)
print('stat=%.3f, p=%.10f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')    
# =============================================================================
