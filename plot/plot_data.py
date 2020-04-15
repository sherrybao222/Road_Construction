import json
import numpy as np
from statistics import mean,stdev,median
from operator import eq
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import math
import matplotlib.lines as mlines
from scipy.stats import wilcoxon
import pandas as pd 

data_all = [] # prepare for all data
subs = [1,2,4] # subject index
budget = [200,350,400]

# import experiment data
for num in subs:
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
num_list = num_map[2]*3 # duplicate for all subjects
rc_list = rc_map[2]*3

undo_click = [] # if commit undo in a trial
n_undo = [] #number of undo

budget_list = [] #budget for number est

# time for a trial
t_rc = [] # whole trial
t_undo = [] # whole trial
f_t_rc = [] # first move
f_t_undo = [] # first move

# get time index for further calculatation of time of a single move
rc_t_all = [] #choice index in basic trl
rcu_t_all = [] # choice and undo index in undo trl
u_t_all = [] # undo index in undo trl
rc_u_t_all = [] # choice index in undo trl

# time of a single move
t_everyact_rc = [] # time for choice (exclude first choice) in basic
m_t_everyact_rc = []
t_everyact_undo = [] # time for every action (except first choice and submit) in undo
t_everyundo = [] # time for undo
t_everyc_undo = [] # time for every choice (exclude first choice) in undo
m_t_everyc_undo = []

t_s_rc = [] # time to submit in basic
t_s_undo = [] # time to submit in undo


# across subject statistics summary
mean_rc = [] # mean of number of connect city in rc
mean_undo = [] # mean of number of connect city in undo trl
ac_num = [] # accuracy for num_est
budget_numest = [0]*3 # accuracy/mbe/mae for num_est for every budget
budget_numest_err = [0]*3
pc_undo = [] # Percentage of using undos among all undo trials
mean_t_rc = [] # mean of whole trial time for rc trl
mean_t_undo = [] # mean of whole trial time for undo trl
mean_f_t_rc = [] # mean of first move time for rc trl
mean_f_t_undo = []# mean of first move time for undo trl



for data in data_all:
    for i in range(len(data[0])): # index for every trial
        if data[0][i]['cond'][-1] == 1:
            num.append(int(data[0][i]['num_est'][-1]))
            budget_list.append(data[0][i]['total'])
            
            num_ind.append(i)
            
        if data[0][i]['cond'][-1] == 2:
            rc.append(data[0][i]['n_city'][-1])
            t_rc.append(data[0][i]['time'][-1]-data[0][i]['time'][0])
            
           
            ind_choice = next(x for x, val in enumerate(data[0][i]['choice_his']) 
            if val != 0)  # index for first move
            f_t_rc.append(data[0][i]['time'][ind_choice]-data[0][i]['time'][0])
            
            v = np.array(data[0][i]['choice_his'])
            temp = np.where(v[:-1] != v[1:])[0]
            rc_t_all.append(list(map(lambda x : x + 1, temp)))
            
            temp = [data[0][i]['time'][rc_t_all[-1][p+1]]-data[0][i]['time'][rc_t_all[-1][p]] for p,x in enumerate(rc_t_all[-1]) if p<len(rc_t_all[-1])-1]
            t_everyact_rc.append(temp)
            m_t_everyact_rc.append(median(t_everyact_rc[-1]))
            
            t_s_rc.append(data[0][i]['time'][-1]-data[0][i]['time'][rc_t_all[-1][-1]])           
            
            rc_ind.append(i)
            
        if data[0][i]['cond'][-1] == 3:
            undo.append(data[0][i]['n_city'][-1])
            t_undo.append(data[0][i]['time'][-1]-data[0][i]['time'][0])
            
            ind_choice = next(x for x, val in enumerate(data[0][i]['choice_his']) 
                                  if val != 0)# index for first move
            f_t_undo.append(data[0][i]['time'][ind_choice]-data[0][i]['time'][0])
            
            v = np.array(data[0][i]['choice_his'])
            temp = np.where(v[:-1] != v[1:])[0]
            rc_t_all.append(list(map(lambda x : x + 1, temp)))

            v = np.array(data[0][i]['choice_his'])
            temp = np.where(v[:-1] != v[1:])[0]
            rcu_t_all.append(list(map(lambda x : x + 1, temp)))
            
            temp = [data[0][i]['time'][rcu_t_all[-1][p+1]]-data[0][i]['time'][rcu_t_all[-1][p]] for p,x in enumerate(rcu_t_all[-1]) if p<len(rcu_t_all[-1])-1]
            t_everyact_undo.append(temp)

            u_t_all.append([i for i,x in enumerate(data[0][i]['undo_press']) 
                                        if x == 1])
            
            np_rcu_t_all = np.array(rcu_t_all[-1])# change to numpy array for next opertation
            
            t_everyundo.append([t_everyact_undo[-1][np.where(np_rcu_t_all == x)[0][0]-1]for x in u_t_all[-1]])            
            rc_u_t_all.append([e for e in rcu_t_all[-1] if e not in u_t_all[-1]])
            
            t_everyc_undo.append([e for e in t_everyact_undo[-1] if e not in t_everyundo[-1]])
            m_t_everyc_undo.append(median(t_everyc_undo[-1]))

            t_s_undo.append(data[0][i]['time'][-1]-data[0][i]['time'][rcu_t_all[-1][-1]])

            n_undo.append(sum(data[0][i]['undo_press']))
            
            if 1 in set(data[0][i]['undo_press']):
                undo_click.append(1)
            else:
                undo_click.append(0)
                
            undo_ind.append(i)

bool_numest = map(eq, num, num_list) # correct or not in num_est 
int_numest = list(np.array(list(bool_numest)).astype(float))# indicator num_est is correct

for i in range(0,3):
    mean_rc.append(mean(rc[48*i:48*(i+1)]))
    mean_undo.append(mean(undo[48*i:48*(i+1)]))
    
    mean_t_rc.append(mean(t_rc[48*i:48*(i+1)]))
    mean_t_undo.append(mean(t_undo[48*i:48*(i+1)]))
    mean_f_t_rc.append(mean(f_t_rc[48*i:48*(i+1)]))
    mean_f_t_undo.append(mean(f_t_undo[48*i:48*(i+1)]))
    
    pc_undo.append(mean(undo_click[48*i:48*(i+1)]))
    ac_num.append(mean(int_numest[48*i:48*(i+1)]))


# calculated for heatmap, ignore for now    
rc_undo = np.zeros((12,12))
for ind,val in enumerate(rc):
        rc_undo[val,undo[ind]] = rc_undo[val,undo[ind]]+1/len(rc)

# difference
diff = [] # number connected (actual - max) in rc
zip_obj = zip(rc,rc_list)
for x,y in zip_obj:
    diff.append(x-y)
    
diff_undo = [] # number connected (actual - max) in undo
zip_obj = zip(undo,rc_list)
for x,y in zip_obj:
    diff_undo.append(x-y)
    
diff_n = [] # number estimation (reported - correct) 
diff_a_n = [] # number estimation abosolute error
zip_obj = zip(num,num_list)
for x,y in zip_obj:
    diff_n.append(x-y)
    diff_a_n.append(abs(x-y))
    
# initialize
mx_num = [0]*3 # number est summary matrix 
sorted_mx = [0]*3 # number est summary matrix (sorted by budget)
  
for i in range(0,3):
    mx_num[i] = np.column_stack((num_ind[48*i:48*(i+1)], budget_list[48*i:48*(i+1)], 
          num[48*i:48*(i+1)], num_list[48*i:48*(i+1)],int_numest[48*i:48*(i+1)],
          diff_n[48*i:48*(i+1)], diff_a_n[48*i:48*(i+1)]))
    temp = mx_num[i].view(np.ndarray)
    sorted_mx[i] = temp[np.lexsort((temp[:, 1], ))] # sort by budget
    
for i in range(0,3):
    sub = []
    sub1 = []
    sub2 = []
    for j in range(0,3):
        sub.append(mean(sorted_mx[i][16*j:16*(j+1),4]))
        sub1.append(mean(sorted_mx[i][16*j:16*(j+1),5]))
        sub2.append(mean(sorted_mx[i][16*j:16*(j+1),6]))
    budget_numest[i] = [sub,sub1,sub2]

for i in range(0,3):
    sub = []
    sub1 = []
    sub2 = []
    for j in range(0,3):
        sub.append(stdev(sorted_mx[i][16*j:16*(j+1),4])/math.sqrt(16))
        sub1.append(stdev(sorted_mx[i][16*j:16*(j+1),5])/math.sqrt(16))
        sub2.append(stdev(sorted_mx[i][16*j:16*(j+1),6])/math.sqrt(16))
    budget_numest_err[i] = [sub,sub1,sub2]


# organize some data into dataframe  
# intialise data of lists.

sub = [x for s in subs for x in [s]*48] 
data = {'sub':sub, 'num_ind':num_ind, 'rc_ind':rc_ind, 'undo_ind':undo_ind,
        'num_list':num_list,  'budget_list':budget_list, 'rc_list':rc_list,
        'undo_click':undo_click, 'n_undo':n_undo, 
        'diff_n':diff_n, 'diff':diff,'diff_undo':diff_undo, 
        't_rc':t_rc, 't_undo':t_undo, 
        'f_t_rc':f_t_rc,'f_t_undo':f_t_undo,
        't_everyact_rc':t_everyact_rc,'m_t_everyact_rc':m_t_everyact_rc,
        't_everyc_undo':t_everyc_undo,'m_t_everyc_undo':m_t_everyc_undo,
        't_s_rc':t_s_rc, 't_s_undo':t_s_undo,
        't_everyundo':t_everyundo} 
  
# Create DataFrame 
df = pd.DataFrame(data) 
df.to_csv('filtered.csv', index=True)
# =============================================================================
# plotting per subject    
# =============================================================================
# number est scatter all - final
fig, axs = plt.subplots(1, 3, sharey=True)
for i in range(0,3):
    u, c = np.unique(np.c_[num_list[48*i:48*(i+1)],num[48*i:48*(i+1)]], return_counts=True, axis=0)
    axs[i].scatter(u[:,0],u[:,1],s =(c*1.5)**2,c= '#0776d8')
    #ax = sns.heatmap(num_mx,cmap="YlGnBu",linewidths=.3,linecolor = 'k')
    
    axs[i].set_xlim((0,11))
    axs[i].set_ylim((0,11))
    x0,x1 = axs[i].get_xlim()
    y0,y1 = axs[i].get_ylim()
    axs[i].set_aspect(abs(x1-x0)/abs(y1-y0))
    
    axs[i].plot(axs[i].get_xlim(), axs[i].get_ylim(), ls="--", c=".3") # diagnal
    axs[i].grid(b=True, which='major', color='k', linestyle='--',alpha=0.2)
    axs[i].set_facecolor('white')
    
    axs[i].set_xticks(np.arange(x0,x1, 1.0))
    axs[i].set_yticks(np.arange(y0,y1, 1.0))
    axs[i].title.set_text('S'+str(i+1))

axs[1].set_xlabel("Number estimation (correct)")
axs[0].set_ylabel("Number estimation (reported)")
fig.set_figwidth(12)

#ax.set_aspect('equal')
fig.savefig('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/plot/fig/num_est_scatter_all.png',dpi=600)
plt.close(fig)

# ----------------------------------------------------
fig, axs = plt.subplots(1, 3, sharey=True)
for i in range(0,3):

    ind = [0.5,0.8,1.1]
    axs[i].bar(ind,budget_numest[i][2],width = 0.1,
           color = '#99cccc',edgecolor='k')
           
    plotline1, caplines1, barlinecols1 = axs[i].errorbar(ind, budget_numest[i][2], yerr=budget_numest_err[i][2], lolims=True, capsize = 0, ls='None', color='k')
    caplines1[0].set_marker('_')
    caplines1[0].set_markersize(5)

    axs[i].set_ylim((0,3))
    axs[i].set_xticks([0.3,0.5,0.8,1.1,1.3])
    axs[i].set_xticklabels(['',200,350,500,''])
    axs[i].set_facecolor('white')
    axs[i].spines['bottom'].set_color('k')
    axs[i].spines['left'].set_color('k')
    axs[i].tick_params(axis='y', colors='k')
    axs[i].title.set_text('S'+str(i+1))
    
axs[0].set_ylabel('Mean absolute error in number estimation')
axs[1].set_xlabel('Budget length (px)')

plt.show()
fig.savefig('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/plot/fig/num_mae.png',dpi=600)
plt.close(fig)

# ----------------------------------------------------
fig, axs = plt.subplots(1, 3, sharey=True)
for i in range(0,3):

    ind = [0.5,0.8,1.1]
    axs[i].bar(ind,budget_numest[i][1],width = 0.1,
           color = '#99cccc',edgecolor='k')
    
    new_err =[]
    for j,x in enumerate(budget_numest_err[i][1]):
        if budget_numest[i][1][j]< 0:
            new_err.append(-x)
        else:
            new_err.append(x)
    plotline1, caplines1, barlinecols1 = axs[i].errorbar(ind, budget_numest[i][1], yerr=new_err, lolims=True, capsize = 0, ls='None', color='k')
    caplines1[0].set_marker('_')
    caplines1[0].set_markersize(5)

    axs[i].set_ylim((-3,1))
    axs[i].set_xticks([0.3,0.5,0.8,1.1,1.3])
    axs[i].set_xticklabels(['',200,350,500,''])
    axs[i].set_facecolor('white')
    axs[i].spines['bottom'].set_color('k')
#    axs[i].spines['bottom'].set_position('zero')
    axs[i].spines['left'].set_color('k')
    axs[i].tick_params(axis='y', colors='k')
    axs[i].title.set_text('S'+str(i+1))
    
axs[0].set_ylabel('Mean error in number estimation')
axs[1].set_xlabel('Budget length (px)')

plt.show()
fig.savefig('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/plot/fig/num_mbe.png',dpi=600)
plt.close(fig)

# =============================================================================
# num_hist for every budget
fig, axs = plt.subplots(3, 3, sharey=True)

for i in range(0,3): # subject
    for j in range(0,3): # budget level
        axs[i,j].hist(sorted_mx[i][16*j:16*(j+1),5], range(-4,4), facecolor='#727bda', density=1,
              align = 'left',edgecolor='k')
    
        axs[i,j].set_ylim((0,1))
        axs[i,j].set_xlim((-5,3))
    
        axs[i,j].set_xticks(range(-4,3))
        axs[i,j].set_yticks(np.arange(0,1.1, 0.1))
        axs[i,j].set_yticklabels([0,'',0.2,'',0.4,'',0.6,'',0.8,'',1.0])
    
        axs[i,j].set_facecolor('white')
        axs[i,j].spines['bottom'].set_color('k')
        axs[i,j].spines['left'].set_color('k')
        axs[i,j].tick_params(axis='y', colors='k',direction='in',left = True)
        axs[i,1].title.set_text('S'+str(i+1))
        axs[0,j].text(-2, 0.8, 'budget = '+ '{:.0f}'.format(budget[j])+'px',fontsize=10)

fig.subplots_adjust(top=1)
axs[2,1].set_xlabel('Number estimation (reported - correct)')
axs[1,0].set_ylabel('Frequency')
fig.set_figheight(11)
fig.set_figwidth(10)

plt.show()
fig.savefig('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/plot/fig/num_hist_budget.png',dpi=600,bbox_inches='tight')
plt.close(fig)
# =============================================================================
# rc scatter all - final
fig, axs = plt.subplots(2, 3, sharey=True)
for j in range(0,2):
    for i in range(0,3):
        u, c = np.unique(np.c_[rc_list[48*i:48*(i+1)],rc[48*i:48*(i+1)]], return_counts=True, axis=0)
        u1, c1 = np.unique(np.c_[rc_list[48*i:48*(i+1)],undo[48*i:48*(i+1)]], return_counts=True, axis=0)
        if j == 0:
            axs[j,i].scatter(u[:,0],u[:,1],s =c*15,facecolors='none',
                           edgecolors = '#727bda')
        else:
            axs[j,i].scatter(u1[:,0],u1[:,1],s =c*15,facecolors='none',
                           edgecolors = '#e13f42')
        #ax = sns.heatmap(num_mx,cmap="YlGnBu",linewidths=.3,linecolor = 'k')
        
        axs[j,i].set_xlim((4,12))
        axs[j,i].set_ylim((4,12))
        x0,x1 = axs[j,i].get_xlim()
        y0,y1 = axs[j,i].get_ylim()
        axs[j,i].set_aspect(abs(x1-x0)/abs(y1-y0))
        
        axs[j,i].plot(axs[j,i].get_xlim(), axs[j,i].get_ylim(), ls="--", c=".3") # diagnal
        axs[j,i].grid(b=True, which='major', color='k', linestyle='--',alpha=0.2)
        axs[j,i].set_facecolor('white')
        
        axs[j,i].set_xticks(np.arange(x0,x1, 1.0))
        axs[j,i].set_yticks(np.arange(y0,y1, 1.0))
        axs[0,i].title.set_text('S'+str(i+1))

        axs[1,i].set_xlabel("Number connectable (maximal)")
        axs[j,0].set_ylabel("Number connected (actual)")

title_1 = mlines.Line2D([], [], color='white', label='without undo')
title_2 = mlines.Line2D([], [], color='white', label='with undo')
rc_led_1 = mlines.Line2D([], [], color='white', marker='o',markeredgecolor='#727bda',
                          markersize=math.sqrt(1*15), label='1')
rc_led_2 = mlines.Line2D([], [], color='white', marker='o',markeredgecolor='#727bda',
                          markersize=math.sqrt(5*15), label='5')
rc_led_3 = mlines.Line2D([], [], color='white', marker='o',markeredgecolor='#727bda',
                          markersize=math.sqrt(10*15), label='10')
rc_led_4 = mlines.Line2D([], [], color='white', marker='o',markeredgecolor='#727bda',
                          markersize=math.sqrt(15*15), label='15')
undo_led_1 = mlines.Line2D([], [], color='white', marker='o',markeredgecolor='#e13f42',
                          markersize=math.sqrt(1*15), label='1')
undo_led_2 = mlines.Line2D([], [], color='white', marker='o',markeredgecolor='#e13f42',
                          markersize=math.sqrt(5*15), label='5')
undo_led_3 = mlines.Line2D([], [], color='white', marker='o',markeredgecolor='#e13f42',
                          markersize=math.sqrt(10*15), label='10')
undo_led_4 = mlines.Line2D([], [], color='white', marker='o',markeredgecolor='#e13f42',
                          markersize=math.sqrt(15*15), label='15')

lgd = axs[0,2].legend(bbox_to_anchor=(2.04, 0.1),prop={'size': 12},title="Number of trials",handletextpad=0.01,handlelength=3,
           handles=[title_1,rc_led_1,rc_led_2,rc_led_3,rc_led_4,
                    title_2,undo_led_1,undo_led_2,undo_led_3,undo_led_4],facecolor = 'white',ncol=2)

for vpack in lgd._legend_handle_box.get_children():
        vpack.get_children()[0].get_children()[0].set_width(0)
       
#ax.set_aspect('equal')
fig.set_figwidth(12)
fig.set_figheight(9)

plt.show()
fig.savefig('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/plot/fig/rc_scatter_all.png',dpi=600,bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# rc_undo_hist - final
fig, axs = plt.subplots(1, 3, sharey=True)
for i in range(0,3):
    mean1 = mean(diff[48*i:48*(i+1)])
    mean2 = mean(diff_undo[48*i:48*(i+1)])

    temp = [diff[48*i:48*(i+1)],diff_undo[48*i:48*(i+1)]]
    temp = np.array(temp)
    new = temp.transpose()
    axs[i].hist(new, range(-4,2), color=['#0776d8','#e13f42'], density=1,
                 align = 'left',edgecolor='k')

    axs[i].set_ylim((0,1))
    axs[i].set_xlim((-4,1))
    axs[i].set_xticks(range(-3,1))
    axs[i].set_yticks(np.arange(0,1.1, 0.1))
    axs[i].set_yticklabels([0,'',0.2,'',0.4,'',0.6,'',0.8,'',1.0])


    axs[i].set_facecolor('white')
    axs[i].spines['bottom'].set_color('k')
    axs[i].spines['left'].set_color('k')
    axs[i].tick_params(axis='y', colors='k',direction='in',left = True)
    axs[i].title.set_text('S'+str(i+1))
    axs[i].text(-3.5, 0.7, 'without undo mean:'+ '{:.2f}'.format(mean1),fontsize=6)
    axs[i].text(-3.5, 0.6, 'with undo mean:'+'{:.2f}'.format(mean2),fontsize=6)
axs[1].set_xlabel('Number (actual connected - maximum connectable)')
axs[0].set_ylabel('Proportion of trials')

import matplotlib.patches as mpatches
rc_led = mpatches.Patch(color='#0776d8', label='without undo')
undo_led = mpatches.Patch(color='#e13f42', label='with undo')
lgd = plt.legend(handles=[rc_led,undo_led],facecolor = 'white')

fig.set_figwidth(10)

plt.show()
fig.savefig('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/plot/fig/rc_undo_hist_all.png',dpi=600,bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# number undos boxplot - final
fig, axs = plt.subplots(1, 3, sharey=True)

for i in range(0,3):
    axs[i].boxplot(n_undo[48*i:48*(i+1)],widths = 0.6)  
    axs[i].plot([1]*48,n_undo[48*i:48*(i+1)], 'o',
       markerfacecolor = '#727bda',markeredgecolor = 'none',alpha = 0.2)     

    axs[i].set_ylim((-4,40))
    axs[i].set_xlim((0,2))

    axs[i].set_xticklabels([])

    axs[i].set_facecolor('white')
    axs[i].spines['bottom'].set_color('k')
    axs[i].spines['left'].set_color('k')
    axs[i].title.set_text('S'+str(i+1))
axs[1].set_xlabel('Number of undos per trial')
axs[0].set_ylabel('Number of undos per trial')
plt.show()
fig.savefig('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/plot/fig/n_undo_hist_all.png',dpi=600)
plt.close(fig)
# =============================================================================
# n_undo - n_max - final
fig, axs = plt.subplots(1, 3, sharey=True)
for i in range(0,3):
    n_undo_mx = []
    for j in list(set(rc_list[48*i:48*(i+1)])):
        n_undo_mx.append(df.loc[(df['sub'] == subs[i]) & (df['rc_list'] == j),'n_undo'])
    
    err_1 = [np.quantile(x,q=0.25) for x in n_undo_mx]
    err_2 = [np.quantile(x,q=0.75) for x in n_undo_mx]

    axs[i].plot(list(set(rc_list[48*i:48*(i+1)])),[np.median(x) for x in n_undo_mx],marker = 'o',
       markerfacecolor = '#727bda',markeredgecolor = 'none')     
    axs[i].errorbar(list(set(rc_list[48*i:48*(i+1)])), [np.median(x) for x in n_undo_mx], 
       yerr=[err_1,err_2], capsize = 3, ls='None', color='k')

    axs[i].set_ylim((-4,50))
    axs[i].set_xlim((4,12))

    
    axs[i].set_xticks(range(6,12))

    axs[i].set_facecolor('white')
    axs[i].spines['bottom'].set_color('k')
    axs[i].spines['left'].set_color('k')
    axs[i].title.set_text('S'+str(i+1))
axs[1].set_xlabel('Maximum connectable number')
axs[0].set_ylabel('Number of undos per trial')

plt.show()
fig.savefig('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/plot/fig/n_undo_max.png',dpi=600)
plt.close(fig)
# =============================================================================
# whole time boxplot - final
fig, axs = plt.subplots(1, 3, sharey=True)

for i in range(0,3):
    axs[i].boxplot([t_rc[48*i:48*(i+1)], t_undo[48*i:48*(i+1)]],widths = 0.6)  
    axs[i].plot([1,2],[t_rc[48*i:48*(i+1)], t_undo[48*i:48*(i+1)]], 'o',
       markerfacecolor = '#727bda',markeredgecolor = 'none',alpha = 0.2)     
    #plotline1, caplines1, barlinecols1 = ax.errorbar(ind, [mean(mean_t_rc),mean(mean_t_undo)], yerr=[err_rc,err_undo], lolims=True, capsize = 0, ls='None', color='k')
    #caplines1[0].set_marker('_')
    #caplines1[0].set_markersize(7)
    axs[i].set_ylim((0,110))
    
    axs[i].set_xticklabels(['w/o undo','w/ undo'])
    #ax.grid(b=True, which='major', axis = 'y',color='k', linestyle='--')
    axs[i].set_facecolor('white')
    axs[i].spines['bottom'].set_color('k')
    axs[i].spines['left'].set_color('k')
    axs[i].tick_params(axis='y', colors='k')
    axs[i].title.set_text('S'+str(i+1))
axs[0].set_ylabel('Trial duration (s)')
plt.show()
fig.savefig('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/plot/fig/t_rc_hist_all.png',dpi=600)
plt.close(fig)

# =============================================================================
# action time
def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))

def text(p):
    if p < 0.001:
        axs[i].text((x1+x2)*.5, y+h, r"$p = {0:s}$".format(as_si(p,1)), ha='center', va='bottom', color=col)
    elif p > 0.1:
        axs[i].text((x1+x2)*.5, y+h, r"$p = {:.2f}$".format(p), ha='center', va='bottom', color=col)

    elif 0.01 < p < 0.1:
        axs[i].text((x1+x2)*.5, y+h, r"$p = {:.3f}$".format(p), ha='center', va='bottom', color=col)
    else:
        axs[i].text((x1+x2)*.5, y+h, r"$p = {:.4f}$".format(p), ha='center', va='bottom', color=col)


fig, axs = plt.subplots(1, 3, sharey=True)

for i in range(0,3):
    df_part = df.loc[df['sub'] == subs[i],['f_t_rc','f_t_undo',
                     'm_t_everyact_rc','m_t_everyc_undo',
                     't_s_rc','t_s_undo']]

    undobox = []
    for x in t_everyundo[48*i:48*(i+1)]:
        try: 
            undobox.append(median(x))
        except:  pass

    axs[i].plot([1,2],[f_t_rc[48*i:48*(i+1)], f_t_undo[48*i:48*(i+1)]],c = '#6a6763',linewidth=0.3) 
    axs[i].plot([2.5,3.5],[[median(x) for x in t_everyact_rc[48*i:48*(i+1)]], [median(x) for x in t_everyc_undo[48*i:48*(i+1)]]],c = '#6a6763',linewidth=0.3) 
    axs[i].plot([4,5],[t_s_rc[48*i:48*(i+1)],t_s_undo[48*i:48*(i+1)]],c = '#6a6763',linewidth=0.3) 

       
    bx = axs[i].boxplot([f_t_rc[48*i:48*(i+1)],f_t_undo[48*i:48*(i+1)],
       [median(x) for x in t_everyact_rc[48*i:48*(i+1)]],
       [median(x) for x in t_everyc_undo[48*i:48*(i+1)]],
       t_s_rc[48*i:48*(i+1)],t_s_undo[48*i:48*(i+1)],
       undobox],positions =[1,2,2.5,3.5,4,5,5.5],widths = 0.3,showfliers=False,whis = 1.5,
       medianprops = dict(color = 'k')) 
    
    
    # statistical annotation
    stat1, p1 = wilcoxon(df_part['f_t_rc'], df_part['f_t_undo'])
    x1, x2 = 1,2  
    if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
        y, h, col = bx['caps'][1]._y[0] + 2, 2, 'k'
    else:
        y, h, col = bx['caps'][3]._y[0] + 2, 2, 'k'
    
    axs[i].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    text(p1)
    
    #--------------------------------------
    stat2, p2 = wilcoxon(df_part['m_t_everyact_rc'], df_part['m_t_everyc_undo'])

    x1, x2 = 2.5,3.5  
    y, h, col = bx['caps'][5]._y[0] + 2, 2, 'k'
    axs[i].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    text(p2)
    #--------------------------------------
    stat3, p3 = wilcoxon(df_part['t_s_rc'], df_part['t_s_undo'])

    x1, x2 = 4,5  
    y, h, col = bx['caps'][11]._y[0] + 2, 2, 'k'
    axs[i].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    text(p3)
    #--------------------------------------
    stat4, p4 = wilcoxon(df_part['f_t_rc'], df_part['m_t_everyact_rc'])

    x1, x2 = 1,2.5  
    if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
        y, h, col = bx['caps'][1]._y[0] + 6, 2, 'k'
    else:
        y, h, col = bx['caps'][3]._y[0] + 6, 2, 'k'
        
    axs[i].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    text(p4)
    #--------------------------------------
    stat5, p5 = wilcoxon(df_part['f_t_undo'], df_part['m_t_everyc_undo'])

    x1, x2 = 2,3.5  
    if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
        y, h, col = bx['caps'][1]._y[0] + 12, 2, 'k'
    else:
        y, h, col = bx['caps'][3]._y[0] + 12, 2, 'k'
        
    axs[i].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    text(p5)


    #--------------------------------------

    axs[i].set_xticks([1.5,3,4.5,5.5])
#    axs[i].set_xticklabels(labels)
    #ax.grid(b=True, which='major', axis = 'y',color='k', linestyle='--')
    axs[i].set_xticklabels(labels = ['first choice\nwithout undo with undo','later choices\n\nwithout undo with undo', 'submit\nwithout undo with undo','undo'],fontsize=16)

    axs[i].set_facecolor('white')
    axs[i].spines['bottom'].set_color('k')
    axs[i].spines['left'].set_color('k')
    axs[i].tick_params(axis='y', colors='k', direction='in',left = True)   
    axs[i].tick_params(axis='x', colors='k')
    axs[i].title.set_text('S'+str(i+1))
axs[0].set_ylabel('Response time (s)',fontsize=16)

fig.set_figwidth(26)
fig.set_figheight(12)

plt.show()
fig.savefig('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/plot/fig/action_t.png',dpi=600,bbox_inches='tight')
plt.close(fig)

# =============================================================================
# all subjects
# =============================================================================
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
#err_rc = stdev(mean_t_rc)/math.sqrt(len(mean_rc))
#err_undo = stdev(mean_t_undo)/math.sqrt(len(mean_undo))
fig, ax = plt.subplots()
#ax.bar(ind,[mean(mean_t_rc),mean(mean_t_undo)],width = 0.1,
#       color = '#eed06f',edgecolor='k')
ax.boxplot([mean_t_rc, mean_t_undo],widths = 0.6)  
ax.plot([1,2],[mean_t_rc, mean_t_undo], 'o')     
#plotline1, caplines1, barlinecols1 = ax.errorbar(ind, [mean(mean_t_rc),mean(mean_t_undo)], yerr=[err_rc,err_undo], lolims=True, capsize = 0, ls='None', color='k')
#caplines1[0].set_marker('_')
#caplines1[0].set_markersize(7)
ax.set_ylim((10,45))

ax.set_xticklabels(['w/o undo','w/ undo'])
#ax.grid(b=True, which='major', axis = 'y',color='k', linestyle='--')
ax.set_facecolor('white')
ax.spines['bottom'].set_color('k')
ax.spines['left'].set_color('k')
ax.tick_params(axis='y', colors='k')
plt.ylabel('trial duration (s)')

fig.savefig('/Users/sherrybao/Downloads/Research/Road_Construction/rc_all_data/plot/fig/t_rc.png',dpi=600)
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
# Wilcoxon Signed-Rank Test
from scipy.stats import wilcoxon
stat, p = wilcoxon(rc, undo)
print('stat=%.3f, p=%.10f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')  
#----------------------------------------------------------------
# Student's t-test
from scipy.stats import ttest_ind
stat, p = ttest_ind(t_rc, t_undo)
print('stat=%.3f, p=%.6f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
# Student's t-test
from scipy.stats import ttest_ind
stat, p = ttest_ind(f_t_rc, f_t_undo)
print('stat=%.3f, p=%.6f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
# =============================================================================
