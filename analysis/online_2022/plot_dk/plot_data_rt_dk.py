# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# -

from scipy import stats
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import ttest_rel,ttest_ind

# +
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
# import R's "base" package
utils = importr('utils')
utils.chooseCRANmirror(ind=1) # select the first mirror in the list
from rpy2.robjects.vectors import StrVector

packnames = ['lme4', 'optimx', 'pbkrtest', 'lmerTest',
             'ggplot2', 'dplyr', 'sjPlot', 'car']

names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# +
# %load_ext rpy2.ipython

from rpy2.robjects.packages import importr



# import R's "base" package
lme4 = importr('lme4')
optimx = importr('optimx')
pbkrtest = importr('pbkrtest')
lmerTest = importr('lmerTest')
ggplot = importr('ggplot2')
dplyr = importr('dplyr')
sjplot = importr('sjPlot')
car = importr('car')

# +
# home_dir = '/Users/dbao/google_drive_db'+'/road_construction/data/2022_online/'
home_dir = 'G:\My Drive\\researches\\nyu\\road-construction-local-dk\data_online_2022\\'

map_dir = 'active_map/'
data_dir  = 'data/preprocessed'
out_dir = home_dir + 'figures/figures_all/'
R_out_dir = home_dir + 'R_analysis_data/'

# +
data_puzzle_level = pd.read_csv(R_out_dir +  'data.csv')
puzzleID_order_data = data_puzzle_level.sort_values(["subjects","puzzleID","condition"])
data_choice_level = pd.read_csv(R_out_dir +  'choice_level/choicelevel_data.csv')

single_condition_data = puzzleID_order_data[puzzleID_order_data['condition']==1].copy()
single_condition_data = single_condition_data.reset_index()


# +
# helper functions
def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))

# add p-value to figure
def text(p):
    if p == 0:
        axs.text((x1+x2)*.5, y+h,  r"$p = {:.1f}$".format(p), ha='center', va='bottom', color=col, fontsize = 8)
    elif p < 0.001:
        axs.text((x1+x2)*.5, y+h, r"$p = {0:s}$".format(as_si(p,1)), ha='center', va='bottom', color=col, fontsize = 8)
    elif p > 0.1:
        axs.text((x1+x2)*.5, y+h, r"$p = {:.2f}$".format(p), ha='center', va='bottom', color=col, fontsize = 8)

    elif 0.01 < p < 0.1:
        axs.text((x1+x2)*.5, y+h, r"$p = {:.3f}$".format(p), ha='center', va='bottom', color=col, fontsize = 8)
    else:
        axs.text((x1+x2)*.5, y+h, r"$p = {:.4f}$".format(p), ha='center', va='bottom', color=col, fontsize = 8)
# -

# ## action RT

# +
# choice-level all action RT
# index_start = data_choice_level.index[data_choice_level['RT'] == -1]
# RT_first_move = data_choice_level.loc[index_start+1,:]
# index_later = data_choice_level.index[(data_choice_level['RT'] != -1) & (data_choice_level['submit'] != 1)& (data_choice_level['undo'] != 1)]
# RT_later_move = data_choice_level.loc[index_later,:]
# index_submit = data_choice_level.index[data_choice_level['submit'] == 1]
# RT_submit = data_choice_level.loc[index_submit,:]


# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

# df_part = df.loc[df['sub'] == subs[i],['f_t_rc','f_t_undo',
#                  'm_t_everyact_rc','m_t_everyc_undo',
#                  't_s_rc','t_s_undo']]

# undobox = []
# for x in t_everyundo[48*i:48*(i+1)]:
#     try: 
#         undobox.append(median(x))
#     except:  pass

# axs[i].plot([1,2],[f_t_rc[48*i:48*(i+1)], f_t_undo[48*i:48*(i+1)]],c = '#6a6763',linewidth=0.3) 
# axs[i].plot([2.5,3.5],[[median(x) for x in t_everyact_rc[48*i:48*(i+1)]], [median(x) for x in t_everyc_undo[48*i:48*(i+1)]]],c = '#6a6763',linewidth=0.3) 
# axs[i].plot([4,5],[t_s_rc[48*i:48*(i+1)],t_s_undo[48*i:48*(i+1)]],c = '#6a6763',linewidth=0.3) 

# plot with puzzle-level RT
bx = axs.boxplot([puzzleID_order_data[puzzleID_order_data['condition']==0]['RT1'],puzzleID_order_data[puzzleID_order_data['condition']==1]['RT1'],
    puzzleID_order_data[puzzleID_order_data['condition']==0]['RTlater'],puzzleID_order_data[puzzleID_order_data['condition']==1]['RTlater'],
    puzzleID_order_data[puzzleID_order_data['condition']==0]['RTsubmit'],puzzleID_order_data[puzzleID_order_data['condition']==1]['RTsubmit']],
   positions =[1,2,3.5,4.5,6,7],widths = 0.3,showfliers=False,whis = 1.5,
   medianprops = dict(color = 'k'))  #
    
# normality test
# Shapiro-Wilk Test
    
# stats = [np.nan]*6
# ps = [np.nan]*6
# for s in range(len(stats)):
#     stats[s],ps[s]  = shapiro([math.log2(x) for x in df_part.iloc[:,s]])
#     print('Statistics=%.3f, p=%.3f' % (stats[s], ps[s]))
#     # interpret
#     alpha = 0.05
#     if ps[s] > alpha:
#         print('Sample looks Gaussian (fail to reject H0)')
#     else:
#         print('Sample does not look Gaussian (reject H0)')   
    
#     from scipy.stats import anderson
#     result = [np.nan]*6
#     for s in range(len(result)):
#         result[s]= anderson([math.log2(x) for x in df_part.iloc[:,s]])
#         print('Statistic: %.3f' % result[s].statistic)
#         p = 0
#         for i in range(len(result[s].critical_values)):
#         	sl, cv = result[s].significance_level[i], result[s].critical_values[i]
#         	if result[s].statistic < result[s].critical_values[i]:
#         		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
#         	else:
#         		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
    
# run paired-sample t test
stat1, p1 = ttest_rel(puzzleID_order_data[puzzleID_order_data['condition']==0]['RT1'],puzzleID_order_data[puzzleID_order_data['condition']==1]['RT1'])
x1, x2 = 1,2  
if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
    y, h, col = bx['caps'][1]._y[0] + 2, 0.5, 'k'
else:
    y, h, col = bx['caps'][3]._y[0] + 2, 0.5, 'k'

axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

#--------------------------------------
stat2, p2 = ttest_rel(puzzleID_order_data[puzzleID_order_data['condition']==0]['RTlater'],puzzleID_order_data[puzzleID_order_data['condition']==1]['RTlater'])

x1, x2 = 3.5,4.5  
y, h, col = bx['caps'][5]._y[0] + 2, 0.5, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p2)

#--------------------------------------
stat3, p3 = ttest_rel(puzzleID_order_data[puzzleID_order_data['condition']==0]['RTsubmit'],puzzleID_order_data[puzzleID_order_data['condition']==1]['RTsubmit'])

x1, x2 = 6,7 
y, h, col = bx['caps'][11]._y[0] + 2, 0.5, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p3)

#--------------------------------------
stat4, p4 = ttest_rel(puzzleID_order_data[puzzleID_order_data['condition']==0]['RT1'], puzzleID_order_data[puzzleID_order_data['condition']==0]['RTlater'])

x1, x2 = 1,3.5  
if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
    y, h, col = bx['caps'][1]._y[0] + 3.5, 0.5, 'k'
else:
    y, h, col = bx['caps'][3]._y[0] + 3.5, 0.5, 'k'

axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p4)
#--------------------------------------
stat5, p5 = ttest_rel(puzzleID_order_data[puzzleID_order_data['condition']==1]['RT1'], puzzleID_order_data[puzzleID_order_data['condition']==1]['RTlater'])

x1, x2 = 2,4.5  
if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
    y, h, col = bx['caps'][1]._y[0] + 5, 0.5, 'k'
else:
    y, h, col = bx['caps'][3]._y[0] + 5, 0.5, 'k'

axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p5)

#--------------------------------------
axs.set_xticks([1,1.5,2, 3.5,4,4.5, 6,6.5,7])
axs.set_xticklabels(labels = ['\nwithout \nundo','first choice','\nwith \nundo','\nwithout \nundo','later choices','\nwith \nundo', '\nwithout \nundo','submit','\nwith \nundo'])#,fontsize=18

axs.set_facecolor('white')
axs.spines['bottom'].set_color('k')
axs.spines['left'].set_color('k')
axs.tick_params(axis='y', colors='k', direction='in',left = True) #, labelsize = 16
axs.tick_params(axis='x', colors='k')
# axs.set_title('S'+str(i+1), fontsize = 16)
axs.set_ylabel('Response time (s)') #,fontsize=18

# fig.set_figwidth(26)
# fig.set_figheight(12)

plt.show()
fig.savefig(out_dir + 'action_RT.png', dpi=600, bbox_inches='tight')
# -

# ## different types of undoing RT

# +
index_singleUndo = data_choice_level.index[(data_choice_level['firstUndo'] == 1)&(data_choice_level['lastUndo'] == 1)]
RT_singleUndo = data_choice_level.loc[index_singleUndo,:]

index_firstUndo = data_choice_level.index[(data_choice_level['firstUndo'] == 1) &(data_choice_level['lastUndo'] != 1)]
RT_firstUndo = data_choice_level.loc[index_firstUndo,:]
index_laterUndo = data_choice_level.index[(data_choice_level['firstUndo'] != 1) & (data_choice_level['undo'] == 1)]
RT_laterUndo = data_choice_level.loc[index_laterUndo,:]

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

bx = axs.boxplot([RT_firstUndo['undoRT']/1000,RT_laterUndo['undoRT']/1000,RT_singleUndo['undoRT']/1000],
   positions =[1,2,3],widths = 0.3,showfliers=False,whis = 1.5,
   medianprops = dict(color = 'k'))  #

# run 2-independent-sample t test
stat1, p1 = ttest_ind(RT_firstUndo['undoRT']/1000,RT_laterUndo['undoRT']/1000,equal_var=False)
x1, x2 = 1,2  
if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
    y, h, col = bx['caps'][1]._y[0] + 2, 0.5, 'k'
else:
    y, h, col = bx['caps'][3]._y[0] + 2, 0.5, 'k'

axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

#--------------------------------------
stat2, p2 = ttest_ind(RT_laterUndo['undoRT']/1000,RT_singleUndo['undoRT']/1000,equal_var=False)

x1, x2 = 2,3 
y, h, col = bx['caps'][5]._y[0] + 2, 0.5, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p2)

#--------------------------------------
stat3, p3 = ttest_ind(RT_firstUndo['undoRT']/1000,RT_singleUndo['undoRT']/1000,equal_var=False)

x1, x2 = 1,3
if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
    y, h, col = bx['caps'][1]._y[0] + 3.5, 0.5, 'k'
else:
    y, h, col = bx['caps'][3]._y[0] + 3.5, 0.5, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p3)

#--------------------------------------
axs.set_xticks([1,1.5,2,3])
axs.set_xticklabels(labels = ['\nfirst undo','sequential','\nlater undo','single undo'])#,fontsize=18
axs.set_facecolor('white')
axs.spines['bottom'].set_color('k')
axs.spines['left'].set_color('k')
axs.tick_params(axis='y', colors='k', direction='in',left = True) #, labelsize = 16
axs.tick_params(axis='x', colors='k')
# axs.set_title('S'+str(i+1), fontsize = 16)
axs.set_ylabel('Response time (s)') #,fontsize=18

# fig.set_figwidth(26)
# fig.set_figheight(12)

plt.show()
fig.savefig(out_dir + 'undo_RT.png', dpi=600, bbox_inches='tight')

# +
print(stat1, p1)
print(stat2, p2)
print(stat3, p3)


import scipy
print(scipy.stats.shapiro(RT_firstUndo['undoRT']/1000))
print(scipy.stats.shapiro(RT_laterUndo['undoRT']/1000))
import statsmodels.api as sm
import pylab as py
sm.qqplot(RT_firstUndo['undoRT']/1000, line ='45')
py.show()

# +
# flip order
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

# bx = axs.boxplot([RT_firstUndo['undoRT']/1000,RT_laterUndo['undoRT']/1000,RT_singleUndo['undoRT']/1000],
#    positions =[1,2,3],widths = 0.3,showfliers=False,whis = 1.5,
#    medianprops = dict(color = 'k'))  #
bx = axs.boxplot([RT_singleUndo['undoRT']/1000, RT_firstUndo['undoRT']/1000, RT_laterUndo['undoRT']/1000],
   positions =[1,2,3],widths = 0.3,showfliers=False,whis = 1.5,
   medianprops = dict(color = 'k'))  #

# run 2-independent-sample t test
stat1, p1 = ttest_ind(RT_laterUndo['undoRT']/1000,RT_singleUndo['undoRT']/1000,equal_var=False)

x1, x2 = 1,2  
if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
    y, h, col = bx['caps'][1]._y[0] + 2, 0.5, 'k'
else:
    y, h, col = bx['caps'][3]._y[0] + 2, 0.5, 'k'

# axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# text(p1)

#--------------------------------------
stat3, p3 = ttest_ind(RT_firstUndo['undoRT']/1000,RT_singleUndo['undoRT']/1000,equal_var=False)

x1, x2 = 1,3
if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
    y, h, col = bx['caps'][1]._y[0] + 3.5, 0.5, 'k'
else:
    y, h, col = bx['caps'][3]._y[0] + 3.5, 0.5, 'k'
# axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# text(p3)


#--------------------------------------
stat2, p2 = ttest_ind(RT_firstUndo['undoRT']/1000,RT_laterUndo['undoRT']/1000,equal_var=False)

x1, x2 = 2,3 
y, h, col = bx['caps'][5]._y[0] + 7, 0.5, 'k'
# axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# text(p2)


#--------------------------------------
axs.set_yticks([0,2,4,6,8,10])
axs.set_xticks([1,2,2.5,3])
axs.set_xticklabels(labels = ['Single undo','\nfirst ','Sequential undo','\nlater undo'])#,fontsize=18
axs.set_facecolor('white')
axs.spines['bottom'].set_color('k')
axs.spines['left'].set_color('k')
axs.tick_params(axis='y', colors='k', direction='in',left = True) #, labelsize = 16
axs.tick_params(axis='x', colors='k')
# axs.set_title('S'+str(i+1), fontsize = 16)
axs.set_ylabel('Response time (s)') #,fontsize=18

# fig.set_figwidth(26)
# fig.set_figheight(12)

plt.show()
fig.savefig(out_dir + 'undo_RT.png', dpi=600, bbox_inches='tight')
fig.savefig(out_dir + 'undo_RT.pdf', dpi=600, bbox_inches='tight')
# -

RT_singleUndo['undoRT']

# ## budget before submit/undo at the end of trial

# +
# only undo condition
index_first_undo =  data_choice_level.index[data_choice_level['firstUndo'] == 1]
df_beforeUndo = data_choice_level.loc[index_first_undo-1,:]
index_end_undo = df_beforeUndo.index[df_beforeUndo['checkEnd'] == 1]
leftover_undo = df_beforeUndo.loc[index_end_undo,'leftover']

index_notundo = data_choice_level.index[(data_choice_level['undo'] == 0)&(data_choice_level['RT'] != -1)]
df_notbeforeUndo = data_choice_level.loc[index_notundo-1,:]
index_end_notundo = df_notbeforeUndo.index[(df_notbeforeUndo['checkEnd'] == 1)&(df_notbeforeUndo['condition'] == 1)]
leftover_notundo = df_notbeforeUndo.loc[index_end_notundo,'leftover']


# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

bx = axs.boxplot([leftover_undo,leftover_notundo],
   positions =[1,2],widths = 0.3,showfliers=False,whis = 1.5,
   medianprops = dict(color = 'k'))  #

# run 2-independent-sample t test
stat1, p1 = ttest_ind(leftover_undo,leftover_notundo,equal_var=False)
x1, x2 = 1,2  
if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
    y, h, col = bx['caps'][1]._y[0] + 2, 0.5, 'k'
else:
    y, h, col = bx['caps'][3]._y[0] + 2, 0.5, 'k'

axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

#--------------------------------------
axs.set_xticks([1,2])
axs.set_xticklabels(labels = ['budget before undo','budget before submit'])#,fontsize=18

axs.set_facecolor('white')
axs.spines['bottom'].set_color('k')
axs.spines['left'].set_color('k')
axs.tick_params(axis='y', colors='k', direction='in',left = True) #, labelsize = 16
axs.tick_params(axis='x', colors='k')
# axs.set_title('S'+str(i+1), fontsize = 16)
axs.set_ylabel('budget') #,fontsize=18

# fig.set_figwidth(26)
# fig.set_figheight(12)

plt.show()
fig.savefig(out_dir + 'budget_before_submit_undo.png', dpi=600, bbox_inches='tight')
# -

# ### counts of errors before undo (by accumulated severity)

# +
index_first_undo =  data_choice_level.index[data_choice_level['firstUndo'] == 1]
df_beforeUndo = data_choice_level.loc[index_first_undo-1,:]

MAS_trial = df_beforeUndo['allMAS']
accu_severity_error = MAS_trial - df_beforeUndo['currMas']
groupby_error = accu_severity_error.value_counts()
print(groupby_error/sum(groupby_error))

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
axs.bar(groupby_error.index,groupby_error/sum(groupby_error))
axs.set_ylabel('proportion of first undo')
axs.set_xlabel('accumulated error before first undo action')
plt.show()
fig.savefig(out_dir + 'undo_accumulated_error.pdf', dpi=600, bbox_inches='tight')


# +
index_first_undo =  data_choice_level.index[data_choice_level['firstUndo'] == 1]
df_beforeUndo = data_choice_level.loc[index_first_undo-1,:]

instant_severity_error = df_beforeUndo['severityOfErrors']
groupby_error_instant = instant_severity_error.value_counts()
print(groupby_error_instant)

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
axs.bar(groupby_error_instant.index,groupby_error_instant/sum(groupby_error_instant))
axs.set_ylabel('proportion of first undo')
axs.set_xlabel('instant error before first undo action')
plt.show()
fig.savefig(out_dir + 'undo_instant_error.pdf', dpi=600, bbox_inches='tight')


# -
# ### conditional probability version of two figures above,

index_error = puzzle_error.index[puzzle_error == 0]
print(index_error)

# +
# FROM EACH SUBJECT
dat_subjects = []
for i in np.unique(np.array(data_choice_level['subjects'])):
    temp_data = []
    index_subjects =  data_choice_level.index[data_choice_level['subjects'] == i]
    
    puzzle_error = data_choice_level['allMAS'] - data_choice_level['currMas']
    
    # no error
    index_error = puzzle_error.index[puzzle_error == 0]
    index_error = np.array(index_error)
    index_error = np.intersect1d(index_error, index_subjects)
    index_error += 1
    if np.any(index_error>(data_choice_level.shape[0]-1)):
        index_error = np.delete(index_error, np.where(index_error>(data_choice_level.shape[0]-1)))
    temp_data.append(np.mean(data_choice_level['undo'][index_error]))
#     temp_data.append(np.mean(data_choice_level['firstUndo'][index_error]))


    # YES error
    index_error = puzzle_error.index[puzzle_error != 0]
    index_error = np.array(index_error)
    index_error = np.intersect1d(index_error, index_subjects)
    index_error += 1
    if np.any(index_error>(data_choice_level.shape[0]-1)):
        index_error = np.delete(index_error, np.where(index_error>(data_choice_level.shape[0]-1)))
    temp_data.append(np.mean(data_choice_level['undo'][index_error]))
#     temp_data.append(np.mean(data_choice_level['firstUndo'][index_error]))
    
    dat_subjects.append(temp_data)

dat_subjects = np.array(dat_subjects)
print(np.mean(dat_subjects,axis=0))

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
axs.bar([1,2],np.mean(dat_subjects,axis = 0),color=[.7,.7,.7], edgecolor = 'k', yerr=np.std(dat_subjects,axis = 0)/np.sqrt(dat_subjects.shape[0]))
axs.set_ylabel('P (undo)')
axs.set_xticks([1,2])
axs.set_yticks(np.linspace(0,0.16,5))
axs.set_xticklabels(labels = ['No error', 'Error'])#,fontsize=18
fig.set_figheight(4)
fig.set_figwidth(3)
axs.set_xlabel('puzzle-level')
plt.show()
fig.savefig(out_dir + 'conditional_undo_masError.pdf', dpi=600, bbox_inches='tight')

from scipy.stats import ttest_ind
stat1, p1 = ttest_ind(dat_subjects[:,0], dat_subjects[:,1])
print(stat1)
print(p1)
axs.set_title('p=' + str(p1))

# -

import scipy
scipy.stats.shapiro(dat_subjects[:,1])


# +
# FROM EACH SUBJECT
dat_subjects = []
for i in np.unique(np.array(data_choice_level['subjects'])):
    temp_data = []
    index_subjects =  data_choice_level.index[data_choice_level['subjects'] == i]
    
    # no error
    index_error = data_choice_level['severityOfErrors'].index[data_choice_level['severityOfErrors'] == 0]
    index_error = np.array(index_error)
    index_error = np.intersect1d(index_error, index_subjects)
    index_error += 1
    if np.any(index_error>(data_choice_level.shape[0]-1)):
        index_error = np.delete(index_error, np.where(index_error>(data_choice_level.shape[0]-1)))
#     temp_data.append(np.mean(data_choice_level['undo'][index_error]))
    temp_data.append(np.mean(data_choice_level['undo'][index_error]))


    # YES error
    index_error = data_choice_level['severityOfErrors'].index[data_choice_level['severityOfErrors'] != 0]
    index_error = np.array(index_error)
    index_error = np.intersect1d(index_error, index_subjects)
    index_error += 1
    if np.any(index_error>(data_choice_level.shape[0]-1)):
        index_error = np.delete(index_error, np.where(index_error>(data_choice_level.shape[0]-1)))
#     temp_data.append(np.mean(data_choice_level['undo'][index_error]))
    temp_data.append(np.mean(data_choice_level['undo'][index_error]))
    
    dat_subjects.append(temp_data)

dat_subjects = np.array(dat_subjects)

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
axs.bar([1,2],np.mean(dat_subjects,axis = 0),color=[.7,.7,.7], edgecolor = 'k', yerr=np.std(dat_subjects,axis = 0)/np.sqrt(dat_subjects.shape[0]))
axs.set_ylabel('P (undo)')
axs.set_xticks([1,2])
axs.set_xticklabels(labels = ['No error', 'Error'])#,fontsize=18
fig.set_figheight(4)
fig.set_figwidth(3)
axs.set_xlabel('move-level')
plt.show()
fig.savefig(out_dir + 'conditional_pundo_givenError.pdf', dpi=600, bbox_inches='tight')
# -

ttest_ind
stat1, p1 = ttest_ind(dat_subjects[:,0], dat_subjects[:,1])
print(stat1)
print(p1)
import statsmodels.api as sm
import pylab as py
sm.qqplot(dat_subjects[:,0], line ='45')
py.show()

import scipy
scipy.stats.shapiro(dat_subjects[:,1])

# +
# FROM ALL SUBJECTS

data_choice_level['severityOfErrors']
dat = []

# no error
index_error = data_choice_level['severityOfErrors'].index[data_choice_level['severityOfErrors'] == 0]
index_error += 1
index_error = np.array(index_error)
if np.any(index_error>(data_choice_level.shape[0]-1)):
    index_error = np.delete(index_error, np.where(index_error>(data_choice_level.shape[0]-1)))
# dat.append(np.mean(data_choice_level['undo'][index_error]))
dat.append(np.mean(data_choice_level['firstUndo'][index_error]))


# YES error
index_error = data_choice_level['severityOfErrors'].index[data_choice_level['severityOfErrors'] != 0]
index_error += 1
index_error = np.array(index_error)
if np.any(index_error>(data_choice_level.shape[0]-1)):
    index_error = np.delete(index_error, np.where(index_error>(data_choice_level.shape[0]-1)))
# dat.append(np.mean(data_choice_level['undo'][index_error]))
dat.append(np.mean(data_choice_level['firstUndo'][index_error]))
# -

dat

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
axs.bar([1,2],dat,color=[.7,.7,.7], edgecolor = 'k')
axs.set_ylabel('P (undo)')
axs.set_xticks([1,2])
axs.set_xticklabels(labels = ['No error', 'Error'])#,fontsize=18
fig.set_figheight(4)
fig.set_figwidth(3)
plt.show()
fig.savefig(out_dir + 'conditional_pundo_givenError.pdf', dpi=600, bbox_inches='tight')


# -

# ## benefit of undo - number of full undoing

# +
basic_score = puzzleID_order_data[puzzleID_order_data['condition']==0]['numCities'].reset_index(drop=True)
basic_score_z = basic_score/puzzleID_order_data[puzzleID_order_data['condition']==0]['mas'].reset_index(drop=True)
single_condition_data['numCities_z'] = single_condition_data['numCities']/single_condition_data['mas']

single_condition_data['undo_benefit'] = single_condition_data['numCities'] - basic_score
single_condition_data['undo_benefit_z'] = single_condition_data['numCities_z'] - basic_score_z

undo_benefit_sub = single_condition_data.groupby(['subjects'])['undo_benefit'].mean()
undo_count_sub = single_condition_data.groupby(['subjects'])['numFullUndo'].mean()

# +
benefit_undo = (np.array(puzzleID_order_data[puzzleID_order_data['condition']==1]['numCities']) 
        - np.array(puzzleID_order_data[puzzleID_order_data['condition']==0]['numCities']))

undo_count = np.array(puzzleID_order_data[puzzleID_order_data['condition']==1]['numFullUndo'])

yerr = stats.binned_statistic(undo_count, benefit_undo, statistic=lambda y: np.std(y)/np.sqrt(len(y)), bins=[0,1,2,3,4,100])
bins = stats.binned_statistic(undo_count, benefit_undo, 'mean', bins=[0,1,2,3,4,100])


# +
fig, axs = plt.subplots()         
axs.plot(bins[1][:-1], bins[0], color = '#81b29a', linewidth=3)
plotline1, caplines1, barlinecols1 = axs.errorbar(bins[1][:-1], bins[0], yerr[0], capsize = 0, ls='None', color='k')

# non-parametric version of anova (because number of observations is different: https://www.reneshbedre.com/blog/anova.html)
# Kruskal-Wallis Test
stat1, p1 = stats.kruskal(benefit_undo[undo_count==1], benefit_undo[undo_count==2], benefit_undo[undo_count==3],benefit_undo[undo_count>=4])
x1, x2 = 1,4  
y, h, col = bins[0][1] + 0.1, 0, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

stat2, p2 = stats.kruskal(benefit_undo[undo_count==0], benefit_undo[undo_count==1], benefit_undo[undo_count==2], benefit_undo[undo_count==3],benefit_undo[undo_count>=4])
x1, x2 = 0,4  
y, h, col = bins[0][1] + 0.05, 0, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p2)

axs.set_xticks([0,1,2,3,4])
axs.set_xticklabels([0,1,2,3,'4+'])
axs.set_xlabel('number of full undoing')
axs.set_ylabel('benefit of undo (n_undo - n_basic)')
fig.savefig(out_dir + 'undobenefit_undonum.png', dpi=600, bbox_inches='tight')
# -


scatter_data = single_condition_data.groupby(['undo_benefit','numFullUndo'])['index'].size().to_frame(name = 'count').reset_index()

scatter_data['count']

# +
# # %matplotlib notebook
# fig1, ax1 = plt.subplots()
# sns.scatterplot(scatter_data['numFullUndo'], scatter_data['undo_benefit'], size = scatter_data['count'], sizes = (3,100), data=scatter_data) 
# ax1.set_xlabel("number of undo")
# ax1.set_ylabel("benefit of undo")
# -

undo_puzzle = single_condition_data[single_condition_data['numUNDO']>0].groupby(['subjects']).size()
count = [len(single_condition_data.groupby(['subjects']).size())]
for i in range(1,47):
    count.append(sum(undo_puzzle>=i))

# +
fig, axs = plt.subplots()

plt.bar(list(range(0,47)),count)
axs.set_xlabel("undo in >= number of puzzles")
axs.set_ylabel("number of subjects")
# axs.plot(bins[1][:-1], bins[0], color = '#81b29a', linewidth=3)

# +
order = single_condition_data.groupby(['puzzleID'])['numFullUndo'].mean().to_frame()
sort_order = order.sort_values('numFullUndo')

# %matplotlib notebook
fig, axs = plt.subplots(1, 1)

bx = sns.barplot(x='puzzleID', y='numFullUndo', data = single_condition_data, color = '#ccd5ae',order=sort_order.index) 


# +
#TODO: with a caption stating that each point is a subject, the Spearman rho, and the p-value
fig1, ax1 = plt.subplots()
# ax1.plot(undo_count_sub,undo_benefit_sub,'o',s=2,c='k')
ax1.scatter(undo_count_sub,undo_benefit_sub,10,c='k')
ax1.set_xlabel("Average number of undos")
ax1.set_ylabel("Benefit of undo")
fig1.savefig(out_dir + 'benefit_undo.pdf', dpi=600, bbox_inches='tight')



# -

out_dir

# +
# one rho per person

import scipy

rhos = []

for i in range(101):
    singlesub_porder = puzzleID_order_data[puzzleID_order_data['subjects'] == i].copy()
    
    
    wundo_nct = np.array(singlesub_porder[singlesub_porder['condition']==1]['numCities'])
    woundo_nct = np.array(singlesub_porder[singlesub_porder['condition']==0]['numCities'])
    
    undo_benefit = wundo_nct - woundo_nct
    num_undo = np.array(singlesub_porder[singlesub_porder['condition']==1]['numFullUndo'])
#     num_undo = np.array(singlesub_porder[singlesub_porder['condition']==1]['numUNDO'])
    
    
    coeff, p = scipy.stats.spearmanr(undo_benefit, num_undo)
    RR = np.corrcoef(undo_benefit, num_undo)    
    
    print('-'*20)
    print(coeff)
    print(p)
#     print(RR[0,1])
    if not np.isnan(coeff):
        rhos.append(coeff)
#         rhos.append(RR[0,1])
        

# +
# one rho per puzzle

import scipy

rhos = []

for i in range(45):
    singlesub_porder = puzzleID_order_data[puzzleID_order_data['puzzleID'] == i].copy()
    
    
    wundo_nct = np.array(singlesub_porder[singlesub_porder['condition']==1]['numCities'])
    woundo_nct = np.array(singlesub_porder[singlesub_porder['condition']==0]['numCities'])
    
    undo_benefit = wundo_nct - woundo_nct
    num_undo = np.array(singlesub_porder[singlesub_porder['condition']==1]['numFullUndo'])
#     num_undo = np.array(singlesub_porder[singlesub_porder['condition']==1]['numUNDO'])
    
    
    coeff, p = scipy.stats.spearmanr(undo_benefit, num_undo)
    RR = np.corrcoef(undo_benefit, num_undo)    
    
    print('-'*20)
    print(coeff)
    print(p)
#     print(RR[0,1])
    if not np.isnan(coeff):
        rhos.append(coeff)
#         rhos.append(RR[0,1])
        

# +
wundo_nct = np.array(puzzleID_order_data[puzzleID_order_data['condition']==1]['numCities'])
woundo_nct = np.array(puzzleID_order_data[puzzleID_order_data['condition']==0]['numCities'])

undo_benefit = wundo_nct - woundo_nct


data_ = {'puzzleID':puzzleID_order_data['puzzleID'][puzzleID_order_data['condition']==1].tolist(), 
         'subjects':puzzleID_order_data['subjects'][puzzleID_order_data['condition']==1].tolist(), 
         'benefitUndo':undo_benefit.tolist(),
         'numUNDO':puzzleID_order_data['numUNDO'][puzzleID_order_data['condition']==1].tolist(), 
         'numFullUndo':puzzleID_order_data['numFullUndo'][puzzleID_order_data['condition']==1].tolist()  }


import pandas as pd

# Calling DataFrame constructor on list
df = pd.DataFrame(data_)
print(df)



# +
# bar plot for showing benefit of undo for each puzzle
order = single_condition_data.groupby(['puzzleID'])['numFullUndo'].mean().to_frame()
sort_order = order.sort_values('numFullUndo')

order2 = df.groupby(['puzzleID'])['benefitUndo'].mean().to_frame()
sort_order2 = order2.sort_values('benefitUndo')

# %matplotlib notebook
fig, axs = plt.subplots(1, 1)

# bx = sns.barplot(ax = axs[0],x='puzzleID', y='numFullUndo', data = single_condition_data, color = '#ccd5ae',order=sort_order.index) 
bx = sns.barplot(x='puzzleID', y='benefitUndo', data = df, color = '#ccd5ae',order=sort_order2.index) 



# +
# bar plot for showing benefit of undo for each puzzle
order = df.groupby(['puzzleID'])['numFullUndo'].mean().to_frame()
sort_order = order.sort_values('numFullUndo')

order2 = df.groupby(['puzzleID'])['benefitUndo'].mean().to_frame()
sort_order2 = order2.sort_values('benefitUndo')

# %matplotlib notebook
fig, axs = plt.subplots(2, 1)

bx = sns.barplot(ax = axs[0],x='puzzleID', y='numFullUndo', data = df, color = '#ccd5ae',order=sort_order2.index) 
bx = sns.barplot(ax = axs[1], x='puzzleID', y='benefitUndo', data = df, color = '#ccd5ae',order=sort_order2.index) 



# +
iis = []

for i in range(46):
    a = np.array(df['benefitUndo'][df['puzzleID']==i])[sort_order2.index]
    b = np.zeros(np.array(df['benefitUndo'][df['puzzleID']==0]).shape)
    stat,p = ttest_ind(a,b)
    if p < 0.05:
        iis.append(i)
        print(i)

# +
puzzleID = []
subjects = []
benefitUndo=[]
numUNDO = []
numFullUndo = []

wundo_nct = np.array(puzzleID_order_data[puzzleID_order_data['condition']==1]['numCities'])
woundo_nct = np.array(puzzleID_order_data[puzzleID_order_data['condition']==0]['numCities'])
undo_benefit = wundo_nct - woundo_nct

for i in iis:
    puzzleData = puzzleID_order_data[puzzleID_order_data['puzzleID']==i].copy()

    wundo_nct = np.array(puzzleData[puzzleData['condition']==1]['numCities'])
    woundo_nct = np.array(puzzleData[puzzleData['condition']==0]['numCities'])  
    
    undo_benefit = wundo_nct - woundo_nct
    puzzleID.extend(puzzleData[puzzleID_order_data['condition']==1]['puzzleID'].tolist())
    benefitUndo.extend(undo_benefit)
    subjects.extend(puzzleData[puzzleID_order_data['condition']==1]['subjects'].tolist())
    numUNDO.extend(puzzleData[puzzleID_order_data['condition']==1]['numUNDO'].tolist())
    numFullUndo.extend(puzzleData[puzzleID_order_data['condition']==1]['numFullUndo'].tolist())
    
data_ = {'puzzleID':puzzleID, 
         'subjects':subjects, 
         'benefitUndo':benefitUndo,
         'numUNDO':numUNDO, 
         'numFullUndo':numFullUndo }


import pandas as pd

# Calling DataFrame constructor on list
df = pd.DataFrame(data_)
print(df)


# +
# bar plot for showing benefit of undo for each puzzle
order = df.groupby(['puzzleID'])['numFullUndo'].mean().to_frame()
sort_order = order.sort_values('numFullUndo')

order2 = df.groupby(['puzzleID'])['benefitUndo'].mean().to_frame()
sort_order2 = order2.sort_values('benefitUndo')

# %matplotlib notebook
fig, axs = plt.subplots(2, 1)

# bx = sns.barplot(ax = axs[0],x='puzzleID', y='numFullUndo', data = df, color = '#ccd5ae',order=sort_order2.index) 
bx = sns.barplot(ax = axs[0],x='puzzleID', y='numUNDO', data = df, color = '#ccd5ae',order=sort_order2.index) 
bx = sns.barplot(ax = axs[1], x='puzzleID', y='benefitUndo', data = df, color = '#ccd5ae',order=sort_order2.index) 
# -

rhos = []
for i in range(101):
    puzzleData = puzzleID_order_data[puzzleID_order_data['subjects']==i].copy()

    wundo_nct = np.array(puzzleData[puzzleData['condition']==1]['numCities'])
    woundo_nct = np.array(puzzleData[puzzleData['condition']==0]['numCities'])  
    
    undo_benefit = wundo_nct - woundo_nct

    coeff, p = scipy.stats.spearmanr(undo_benefit, puzzleData[puzzleData['condition']==1]['numUNDO'])
#     RR = np.corrcoef(undo_benefit, num_undo)    
    
    print('-'*20)
    print(coeff)
    print(p)
#     print(RR[0,1])
    if not np.isnan(coeff):
        rhos.append(coeff)

p

h,p = ttest_ind(np.array(rhos),np.zeros(np.array(rhos).shape))

# +
wundo_nct = np.array(puzzleID_order_data[puzzleID_order_data['condition']==1]['numCities'])
woundo_nct = np.array(puzzleID_order_data[puzzleID_order_data['condition']==0]['numCities'])

undo_benefit = wundo_nct - woundo_nct
num_undo = np.array(puzzleID_order_data[puzzleID_order_data['condition']==1]['numUNDO'])



fig1, ax1 = plt.subplots()
# ax1.plot(undo_count_sub,undo_benefit_sub,'o',s=2,c='k')
ax1.scatter(num_undo,undo_benefit,10,c='k')
ax1.set_xlabel("Average number of undos")
ax1.set_ylabel("Benefit of undo")
fig.savefig(out_dir + 'benefit_undo_tot.pdf', dpi=600, bbox_inches='tight')


# +
# one rho per person
undo_benefit_sub = single_condition_data.groupby(['subjects'])['undo_benefit'].mean()
undo_count_sub = single_condition_data.groupby(['subjects'])['numFullUndo'].mean()



# spearman rho for individuals
import scipy
fig1, ax1 = plt.subplots()


# -

undo_benefit_puzzle = single_condition_data.groupby(['puzzleID'])['undo_benefit'].mean()
undo_count_puzzle = single_condition_data.groupby(['puzzleID'])['numFullUndo'].mean()
fig1, ax1 = plt.subplots()
ax1.plot(undo_count_puzzle,undo_benefit_puzzle,'o')
ax1.set_xlabel("average number of undo")
ax1.set_ylabel("benefit of undo")

# + magic_args="-i single_condition_data" language="R"
#
# single_condition_data$subjects <- factor(single_condition_data$subjects)
# single_condition_data$puzzleID <- factor(single_condition_data$puzzleID)
# # single_condition_data$numFullUndo[single_condition_data$numFullUndo >4] <- 4
# # single_condition_data$numFullUndo <- factor(single_condition_data$numFullUndo)
#
# str(single_condition_data)

# + language="R"
#
# model = lmer(undo_benefit_z ~ numFullUndo + (numFullUndo|subjects) + (numFullUndo|puzzleID),
#                                   data=single_condition_data , control=lmerControl(optimizer="optimx",
#                                                                    optCtrl=list(method="nlminb")))
#
# # get the coefficients for the best fitting model
# summary(model)

# + language="R"
# anova(model)
# plot(model)
#
# ranef(model)
# ## QQ-plots:
# # par(mfrow = c(1, 2))
# # qqnorm(ranef(model)$subjects[, 1], main = "Random effects of subjects")
# # qqnorm(resid(model), main = "Residuals")
#
# qqPlot(resid(model), distribution = "norm")
# -

# ## count of error - number of optimal solutions

# +
error_basic = np.array(puzzleID_order_data[puzzleID_order_data['condition']==0]['numError']) 
# error_undo = np.array(puzzleID_order_data[puzzleID_order_data['condition']==1]['numError']) 

n_optimal = np.array(puzzleID_order_data[puzzleID_order_data['condition']==1]['nos'])

bins1 = stats.binned_statistic(n_optimal, error_basic, 'mean', bins=[1,3,6,9,100])
# bins2 = stats.binned_statistic(n_optimal, error_undo, 'mean', bins=[1,3,6,9,15])

fig, axs = plt.subplots()         
axs.plot(bins1[1][:-1], bins1[0], color = '#81b29a', linewidth=3,label='basic')
# axs.plot(bins2[1][:-1], bins2[0],linewidth=3,label='undo')

# non-parametric version of anova (because number of observations is different: https://www.reneshbedre.com/blog/anova.html)
# Kruskal-Wallis Test
stat1, p1 = stats.kruskal(error_basic[(n_optimal<3) & (n_optimal>=1)], error_basic[(n_optimal<6) & (n_optimal>=3)], error_basic[(n_optimal<9) & (n_optimal>=6)],error_basic[n_optimal>=9])
x1, x2 = 1,9
y, h, col = bins1[0][0] + 0.05, 0, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

axs.set_xlabel('number of optimal solutions')
axs.set_ylabel('count of error')
axs.set_xticks([1,3,6,9])
axs.set_xticklabels([1,3,6,'9+'])
# axs.legend()
fig.savefig(out_dir + 'error_optimal.png', dpi=600, bbox_inches='tight')

# -


