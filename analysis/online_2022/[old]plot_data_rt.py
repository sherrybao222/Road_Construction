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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#action-RT" data-toc-modified-id="action-RT-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>action RT</a></span><ul class="toc-item"><li><span><a href="#average-within-subject" data-toc-modified-id="average-within-subject-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>average within subject</a></span><ul class="toc-item"><li><span><a href="#bar-plot" data-toc-modified-id="bar-plot-1.1.1"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>bar plot</a></span></li><li><span><a href="#box-and-whisker-plot" data-toc-modified-id="box-and-whisker-plot-1.1.2"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>box and whisker plot</a></span></li></ul></li><li><span><a href="#[discarded]-average-all-data-points" data-toc-modified-id="[discarded]-average-all-data-points-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>[discarded] average all data points</a></span></li></ul></li><li><span><a href="#different-types-of-undoing-RT" data-toc-modified-id="different-types-of-undoing-RT-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>different types of undoing RT</a></span><ul class="toc-item"><li><span><a href="#averaged-within-subject" data-toc-modified-id="averaged-within-subject-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>averaged within subject</a></span><ul class="toc-item"><li><span><a href="#bar-plot" data-toc-modified-id="bar-plot-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>bar plot</a></span></li><li><span><a href="#box-plot" data-toc-modified-id="box-plot-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>box plot</a></span></li></ul></li><li><span><a href="#[discarded]-average-all-data-points" data-toc-modified-id="[discarded]-average-all-data-points-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>[discarded] average all data points</a></span></li></ul></li><li><span><a href="#branching-node-RT" data-toc-modified-id="branching-node-RT-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>branching node RT</a></span><ul class="toc-item"><li><span><a href="#GLMM" data-toc-modified-id="GLMM-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>GLMM</a></span></li><li><span><a href="#average-within-puzzle-and-subject" data-toc-modified-id="average-within-puzzle-and-subject-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>average within puzzle and subject</a></span></li><li><span><a href="#average-within-subject" data-toc-modified-id="average-within-subject-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>average within subject</a></span><ul class="toc-item"><li><span><a href="#box-plot" data-toc-modified-id="box-plot-3.3.1"><span class="toc-item-num">3.3.1&nbsp;&nbsp;</span>box plot</a></span></li><li><span><a href="#bar-plot" data-toc-modified-id="bar-plot-3.3.2"><span class="toc-item-num">3.3.2&nbsp;&nbsp;</span>bar plot</a></span></li></ul></li><li><span><a href="#average-within-puzzle" data-toc-modified-id="average-within-puzzle-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>average within puzzle</a></span><ul class="toc-item"><li><span><a href="#box-plot" data-toc-modified-id="box-plot-3.4.1"><span class="toc-item-num">3.4.1&nbsp;&nbsp;</span>box plot</a></span></li><li><span><a href="#bar-plot" data-toc-modified-id="bar-plot-3.4.2"><span class="toc-item-num">3.4.2&nbsp;&nbsp;</span>bar plot</a></span></li></ul></li></ul></li><li><span><a href="#first-RT-and-numUndo" data-toc-modified-id="first-RT-and-numUndo-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>first RT and numUndo</a></span></li></ul></div>

# +
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# -

from scipy import stats
from scipy.stats import sem
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import ttest_rel,ttest_ind

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

# + language="R"
# sessionInfo()
# -

home_dir = '/Users/dbao/google_drive_db'+'/road_construction/data/2022_online/'
map_dir = 'active_map/'
data_dir  = 'data/preprocessed'
out_dir = home_dir + 'figures/cogsci_2022/'
R_out_dir = home_dir + 'R_analysis_data/'

# +
data_puzzle_level = pd.read_csv(R_out_dir +  'data.csv')
puzzleID_order_data = data_puzzle_level.sort_values(["subjects","puzzleID","condition"])
data_choice_level = pd.read_csv(R_out_dir +  'choice_level/choicelevel_data.csv')

basic_condition_data = puzzleID_order_data[puzzleID_order_data['condition']==0].copy()
basic_condition_data = basic_condition_data.reset_index()
undo_condition_data = puzzleID_order_data[puzzleID_order_data['condition']==1].copy()
undo_condition_data = undo_condition_data.reset_index()


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

# # action RT

# ## average within subject

# +
def get_RT_cond(cond,RTtype):
    RT = puzzleID_order_data[puzzleID_order_data['condition']==cond]
    RT_sub = RT.groupby(['subjects'])[RTtype].mean()
    RT_sub_sem = sem(RT_sub)
    return [RT_sub,RT_sub_sem]

RT1_basic = get_RT_cond(0,'RT1')
RT1_undo = get_RT_cond(1,'RT1')

RTlater_basic = get_RT_cond(0,'RTlater')
RTlater_undo = get_RT_cond(1,'RTlater')

RTsubmit_basic = get_RT_cond(0,'RTsubmit')
RTsubmit_undo = get_RT_cond(1,'RTsubmit')
# -


# ### bar plot

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

bx = axs.bar([1,2, 3.5,4.5, 6,7],
             
             [np.mean(RT1_basic[0]),np.mean(RT1_undo[0]),
              np.mean(RTlater_basic[0]),np.mean(RTlater_undo[0]),
              np.mean(RTsubmit_basic[0]),np.mean(RTsubmit_undo[0])],
             
             color = (.7,.7,.7), 
             
             edgecolor = 'k',
             
             yerr=[RT1_basic[1],RT1_undo[1],
              RTlater_basic[1],RTlater_undo[1],
              RTsubmit_basic[1],RTsubmit_undo[1]])

axs.set_xticks([1,1.5,2, 3.5,4,4.5, 6,6.5,7])
axs.set_xticklabels(labels = ['\nwithout \nundo','first choice','\nwith \nundo',
                              '\nwithout \nundo','later choices','\nwith \nundo', 
                              '\nwithout \nundo','submit','\nwith \nundo'])#,fontsize=18
axs.set_ylabel('Response time (s)')

# run 2-independent-sample t test
stat1, p1 = ttest_ind(RT1_basic[0], RT1_undo[0], equal_var=False)
x1, x2 = 1,2  
if bx[0].get_height() > bx[1].get_height():
    y, h, col = bx[0].get_height() + 0.5, 0.2, 'k'
else:
    y, h, col = bx[1].get_height() + 0.5, 0.2, 'k'

axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

#--------------------------------------
stat2, p2 = ttest_ind(RTlater_basic[0],RTlater_undo[0],equal_var=False)

x1, x2 = 3.5,4.5
y, h, col = bx[2].get_height() + 1, 0.2, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p2)

#--------------------------------------
stat3, p3 = ttest_ind(RTsubmit_basic[0],RTsubmit_undo[0],equal_var=False)

x1, x2 = 6,7
y, h, col = bx[5].get_height() + 1, 0.2, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p3)

#--------------------------------------
stat4, p4 = ttest_ind(RT1_basic[0],RTlater_basic[0],equal_var=False)

x1, x2 = 1,3.5
y, h, col = bx[0].get_height() + 1, 0.2, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p4)

#--------------------------------------
stat5, p5 = ttest_ind(RT1_undo[0],RTlater_undo[0],equal_var=False)

x1, x2 = 2,4.5
y, h, col = bx[1].get_height() + 0.5, 0.2, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p5)

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
fig.savefig(out_dir + 'conditional_undo_masError.pdf', dpi=600, bbox_inches='tight')
# -


# ### box and whisker plot

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

# bx = axs.bar([1,2, 3.5,4.5, 6,7],
             
#              [np.mean(RT1_basic[0]),np.mean(RT1_undo[0]),
#               np.mean(RTlater_basic[0]),np.mean(RTlater_undo[0]),
#               np.mean(RTsubmit_basic[0]),np.mean(RTsubmit_undo[0])],
             
#              color = (.7,.7,.7), 
             
#              edgecolor = 'k',
             
#              yerr=[RT1_basic[1],RT1_undo[1],
#               RTlater_basic[1],RTlater_undo[1],
#               RTsubmit_basic[1],RTsubmit_undo[1]])

# plot with puzzle-level RT
bx = axs.boxplot(
                
                [RT1_basic[0],RT1_undo[0],
                 RTlater_basic[0],RTlater_undo[0],
                 RTsubmit_basic[0],RTsubmit_undo[0]],
    
                 positions =[1,2, 3.5,4.5, 6,7],
                 widths = 0.3,
                 showfliers=False,
                 whis = 1.5,
                 medianprops = dict(color = 'k')) 

axs.set_xticks([1,1.5,2, 3.5,4,4.5, 6,6.5,7])
axs.set_xticklabels(labels = ['\nwithout \nundo','first choice','\nwith \nundo',
                              '\nwithout \nundo','later choices','\nwith \nundo', 
                              '\nwithout \nundo','submit','\nwith \nundo'])#,fontsize=18
axs.set_ylabel('Response time (s)')

# run 2-independent-sample t test
stat1, p1 = ttest_ind(RT1_basic[0], RT1_undo[0], equal_var=False)
x1, x2 = 1,2  
if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
    y, h, col = bx['caps'][1]._y[0] + 2, 0.5, 'k'
else:
    y, h, col = bx['caps'][3]._y[0] + 2, 0.5, 'k'

axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

#--------------------------------------
stat2, p2 = ttest_ind(RTlater_basic[0],RTlater_undo[0],equal_var=False)

x1, x2 = 3.5,4.5  
y, h, col = bx['caps'][5]._y[0] + 2, 0.5, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p2)

#--------------------------------------
stat3, p3 = ttest_ind(RTsubmit_basic[0],RTsubmit_undo[0],equal_var=False)

x1, x2 = 6,7
y, h, col = bx['caps'][11]._y[0] + 2, 0.5, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p3)

#--------------------------------------
stat4, p4 = ttest_ind(RT1_basic[0],RTlater_basic[0],equal_var=False)

x1, x2 = 1,3.5
if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
    y, h, col = bx['caps'][1]._y[0] + 3.5, 0.5, 'k'
else:
    y, h, col = bx['caps'][3]._y[0] + 3.5, 0.5, 'k'

axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p4)

#--------------------------------------
stat5, p5 = ttest_ind(RT1_undo[0],RTlater_undo[0],equal_var=False)

x1, x2 = 2,4.5
if bx['caps'][1]._y[0] > bx['caps'][3]._y[0]:
    y, h, col = bx['caps'][1]._y[0] + 5, 0.5, 'k'
else:
    y, h, col = bx['caps'][3]._y[0] + 5, 0.5, 'k'

axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p5)

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
fig.savefig(out_dir + 'conditional_undo_masError.pdf', dpi=600, bbox_inches='tight')
# -


# ## [discarded] average all data points 

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

# plot with puzzle-level RT
bx = axs.boxplot([puzzleID_order_data[puzzleID_order_data['condition']==0]['RT1'],puzzleID_order_data[puzzleID_order_data['condition']==1]['RT1'],
    puzzleID_order_data[puzzleID_order_data['condition']==0]['RTlater'],puzzleID_order_data[puzzleID_order_data['condition']==1]['RTlater'],
    puzzleID_order_data[puzzleID_order_data['condition']==0]['RTsubmit'],puzzleID_order_data[puzzleID_order_data['condition']==1]['RTsubmit']],
   positions =[1,2,3.5,4.5,6,7],widths = 0.3,showfliers=False,whis = 1.5,
   medianprops = dict(color = 'k')) 
    
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
# fig.savefig(out_dir + 'action_RT.png', dpi=600, bbox_inches='tight')
# -

# # different types of undoing RT

# ## averaged within subject

# +
def get_undoRT(index):
    RT = data_choice_level.loc[index,:]
    RT_sub = RT.groupby(['subjects'])['undoRT'].mean()/1000
    RT_sub_sem = sem(RT_sub)
    return [RT_sub,RT_sub_sem]

index_singleUndo = data_choice_level.index[(data_choice_level['firstUndo'] == 1)&(data_choice_level['lastUndo'] == 1)]
RT_singleUndo = get_undoRT(index_singleUndo)

index_firstUndo = data_choice_level.index[(data_choice_level['firstUndo'] == 1) &(data_choice_level['lastUndo'] != 1)]
RT_firstUndo = get_undoRT(index_firstUndo)

index_laterUndo = data_choice_level.index[(data_choice_level['firstUndo'] != 1) & (data_choice_level['undo'] == 1)]
RT_laterUndo = get_undoRT(index_laterUndo)
# -

# ### bar plot

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

bx = axs.bar([1,2,3.2],
             
             [
              np.mean(RT_firstUndo[0]),
              np.mean(RT_laterUndo[0]),
              np.mean(RT_singleUndo[0])],
             
             color = (.7,.7,.7), 
             
             edgecolor = 'k',
             
             yerr=[RT_firstUndo[1],
                  RT_laterUndo[1],
                  RT_singleUndo[1]])

axs.set_xticks([1,1.5,2,3.2])
axs.set_xticklabels(labels = ['\nfirst undo','sequential','\nlater undo','single undo'])#,fontsize=18
axs.set_facecolor('white')
axs.spines['bottom'].set_color('k')
axs.spines['left'].set_color('k')
axs.tick_params(axis='y', colors='k', direction='in',left = True) #, labelsize = 16
axs.tick_params(axis='x', colors='k')
# axs.set_title('S'+str(i+1), fontsize = 16)
axs.set_ylabel('Response time (s)') #,fontsize=18

#--------------------------------------
# run 2-independent-sample t test
stat1, p1 = ttest_ind(RT_firstUndo[0],RT_laterUndo[0],equal_var=False)
x1, x2 = 1,2  
y, h, col = bx[0].get_height() + 0.5, 0.1, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

#--------------------------------------
stat2, p2 = ttest_ind(RT_laterUndo[0],RT_singleUndo[0],equal_var=False)

x1, x2 = 2,3.2 
y, h, col = bx[2].get_height() + 0.5, 0.1, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p2)

#--------------------------------------
stat3, p3 = ttest_ind(RT_firstUndo[0],RT_singleUndo[0],equal_var=False)

x1, x2 = 1,3.2
y, h, col = bx[0].get_height() + 1, 0.1, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p3)
# fig.set_figwidth(26)
# fig.set_figheight(12)

plt.show()
# fig.savefig(out_dir + 'undo_RT.png', dpi=600, bbox_inches='tight')
# -

# ### box plot

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)


bx = axs.boxplot(
    [
        RT_singleUndo[0],
        RT_firstUndo[0],
        RT_laterUndo[0]
    ],
    positions =[1,2.2,3.2],
    widths = 0.3,
    showfliers=False,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,2.2,2.7,3.2])
axs.set_xticklabels(labels = ['single undo','\nfirst undo','sequential','\nlater undo'])#,fontsize=18
axs.set_facecolor('white')
axs.spines['bottom'].set_color('k')
axs.spines['left'].set_color('k')
axs.tick_params(axis='y', colors='k', direction='in',left = True) #, labelsize = 16
axs.tick_params(axis='x', colors='k')
# axs.set_title('S'+str(i+1), fontsize = 16)
axs.set_ylabel('Response time (s)') #,fontsize=18

#--------------------------------------
# run 2-independent-sample t test
stat1, p1 = ttest_ind(RT_firstUndo[0],RT_laterUndo[0],equal_var=False)
x1, x2 = 2.2,3.2  
y, h, col = bx['caps'][3]._y[0] + 0.5, 0.1, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

#--------------------------------------
stat2, p2 = ttest_ind(RT_laterUndo[0],RT_singleUndo[0],equal_var=False)

x1, x2 = 1,3.2 
y, h, col = bx['caps'][3]._y[0] + 1, 0.1, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p2)

#--------------------------------------
stat3, p3 = ttest_ind(RT_firstUndo[0],RT_singleUndo[0],equal_var=False)

x1, x2 = 1,2.2
y, h, col = bx['caps'][3]._y[0] + 0.5, 0.1, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p3)
# fig.set_figwidth(26)
# fig.set_figheight(12)

plt.show()
# fig.savefig(out_dir + 'undo_RT.png', dpi=600, bbox_inches='tight')
# -

# ## [discarded] average all data points 

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
# -

# # branching node RT

# +
index_start = data_choice_level.index[(data_choice_level['branchingFirst'] == True)&(data_choice_level['currNumCities'] == 1)]
index_start_nobranch = data_choice_level.index[(data_choice_level['branching'] != True)&(data_choice_level['currNumCities'] == 1)&(data_choice_level['condition']==1)]
index_start_laterbranch = data_choice_level.index[(data_choice_level['branchingFirst'] != True)&(data_choice_level['branching'] == True)&(data_choice_level['currNumCities'] == 1)]

# the first visit of a branching node, and it is not a start city
index_notstart = data_choice_level.index[(data_choice_level['branchingFirst'] == True)&(data_choice_level['currNumCities'] != 1)]
index_notstart_nobranch = data_choice_level.index[(data_choice_level['branching'] != True)&(data_choice_level['currNumCities'] != 1)&(data_choice_level['condition']==1)&(data_choice_level['checkEnd'] != 1)&(data_choice_level['submit'] != 1)] 
index_nostart_laterbranch = data_choice_level.index[(data_choice_level['branchingFirst'] != True)&(data_choice_level['branching'] == True)&(data_choice_level['currNumCities'] != 1)&(data_choice_level['checkEnd'] != 1)&(data_choice_level['submit']!=1)]
# -

# ## GLMM

RT_start = data_choice_level.loc[index_start + 1,:]
RT_start_nobranch = data_choice_level.loc[index_start_nobranch + 1,:]
RT_start_laterbranch = data_choice_level.loc[index_start_laterbranch + 1,:]

# +
RT_start_df = RT_start[['subjects', 'puzzleID', 'currNos','RT']].copy()
RT_start_df['type'] = 'start'
RT_start_nobranch_df = RT_start_nobranch[['subjects', 'puzzleID', 'currNos','RT']].copy()
RT_start_nobranch_df['type'] = 'start_nobranch'
RT_start_laterbranch_df = RT_start_laterbranch[['subjects', 'puzzleID', 'currNos','RT']].copy()
RT_start_laterbranch_df['type'] = 'start_laterbranch'

RT_df = pd.concat([RT_start_df, RT_start_nobranch_df,RT_start_laterbranch_df])

# + magic_args="-i RT_df" language="R"
#
# RT_df$subject <- factor(RT_df$subject)
# RT_df$puzzleID <- factor(RT_df$puzzleID)
# RT_df$currNos_z <- scale(RT_df$currNos)
# RT_df$logRT <- log(RT_df$RT/1000)

# + language="R"
# model0 = glmer(logRT ~ type + currNos_z + (1|puzzleID) + (1|subject)
#                , data=RT_df)
# summary(model0)

# + language="R"
# model1 = glmer(logRT ~ type + currNos_z + (type|puzzleID) + (1|subject)
#                , data=RT_df)
# summary(model1)

# + language="R"
# model2 = glmer(logRT ~ type + currNos_z + (type|puzzleID) + (type|subject)
#                , data=RT_df)
# summary(model2)

# + language="R"
# ## Estimating DFs and p-values
#
# # get the coefficients for the best fitting model
# coefs <- data.frame(coef(summary(model2)))
#
# # Use the Kenward-Roger approximation to get approximate degrees of freedom
# df.KR <- get_Lb_ddf(model2, fixef(model2))
# coefs$df.KR <-(rep(df.KR, each=4))
#
# # Calculate confidence intervals from the estimates and the standard errors
# coefs$semax <- coefs$Estimate + (coefs$Std..Error)
# coefs$semin <- coefs$Estimate - (coefs$Std..Error)
#
# # get p-values from the t-distribution using the t-values and approximated
# # degrees of freedom
# coefs$p.KR <- 2 * (1 - pt(abs(coefs$t.value), df.KR))
#
# # use normal distribution to approximate p-value (tends to be anti-conservative with small sample sizes)
# coefs$p.z <- 2 * (1 - pnorm(abs(coefs$t.value)))

# + language="R"
# coefs

# + language="R"
# anova(model0, model1)

# + language="R"
# anova(model1, model2)

# + language="R"
# anova(model0, model2)
# -

# ## average within puzzle and subject


# +
def get_branchRT_2(index,groupby_cat,main):
    RT = data_choice_level.loc[index + 1,:]
    RT_sub = RT.groupby(groupby_cat)['RT'].mean().groupby(main).mean()/1000
    RT_sub_sem = sem(RT_sub)
    return [RT_sub,RT_sub_sem]

RT_start_sub = get_branchRT_2(index_start,['subjects','puzzleID'],['subjects'])

RT_start_nobranch_sub = get_branchRT_2(index_start_nobranch,['subjects','puzzleID'],['subjects'])

RT_start_laterbranch_sub = get_branchRT_2(index_start_laterbranch,['subjects','puzzleID'],['subjects'])

RT_notstart_sub = get_branchRT_2(index_notstart,['subjects','puzzleID'],['subjects'])

RT_notstart_nobranch_sub = get_branchRT_2(index_notstart_nobranch,['subjects','puzzleID'],['subjects'])

RT_nostart_laterbranch_sub = get_branchRT_2(index_nostart_laterbranch,['subjects','puzzleID'],['subjects'])
# -


RT_start_sub

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

bx = axs.boxplot(
    [
        RT_start_nobranch_sub[0],
        RT_start_sub[0],
        RT_start_laterbranch_sub[0]
    ],
    positions =[1,2,3],
    widths = 0.3,
    showfliers=False,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,2,2.5,3])
# axs.set_yticks(np.linspace(0,0.16,5))
axs.set_xticklabels(labels = ['not branching',
                              '\n first visit',
                              'branching',
                              '\n later visit'])#,fontsize=18
axs.set_ylabel('Response time (s)')



# run 2-independent-sample t test
stat1, p1 = ttest_ind(RT_start_nobranch_sub[0], RT_start_sub[0], equal_var=False)
x1, x2 = 1,2  
y, h, col = bx['caps'][3]._y[0] + 1, 0.3, 'k'


axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

#--------------------------------------
stat2, p2 = ttest_ind(RT_start_sub[0],RT_start_laterbranch_sub[0], equal_var=False)

x1, x2 = 2,3 
y, h, col = bx['caps'][3]._y[0] + 1, 0.3, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p2)

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
# fig.savefig(out_dir + 'conditional_undo_masError.pdf', dpi=600, bbox_inches='tight')
# -


# ## average within subject


# +
def get_branchRT(index,groupby_cat):
    RT = data_choice_level.loc[index + 1,:]
    RT_sub = RT.groupby([groupby_cat])['RT'].mean()/1000
    RT_sub_sem = sem(RT_sub)
    return [RT_sub,RT_sub_sem]

RT_start_sub = get_branchRT(index_start,'subjects')

RT_start_nobranch_sub = get_branchRT(index_start_nobranch,'subjects')

RT_start_laterbranch_sub = get_branchRT(index_start_laterbranch,'subjects')

RT_notstart_sub = get_branchRT(index_notstart,'subjects')

RT_notstart_nobranch_sub = get_branchRT(index_notstart_nobranch,'subjects')

RT_nostart_laterbranch_sub = get_branchRT(index_nostart_laterbranch,'subjects')
# -


# ### box plot

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

bx = axs.boxplot(
    [
        RT_start_nobranch_sub[0],
        RT_start_sub[0],
        RT_start_laterbranch_sub[0]
    ],
    positions =[1,2,3],
    widths = 0.3,
    showfliers=False,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,2,2.5,3])
# axs.set_yticks(np.linspace(0,0.16,5))
axs.set_xticklabels(labels = ['not branching',
                              '\n first visit',
                              'branching',
                              '\n later visit'])#,fontsize=18
axs.set_ylabel('Response time (s)')



# run 2-independent-sample t test
stat1, p1 = ttest_ind(RT_start_nobranch_sub[0], RT_start_sub[0], equal_var=False)
x1, x2 = 1,2  
y, h, col = bx['caps'][3]._y[0] + 1, 0.3, 'k'


axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

#--------------------------------------
stat2, p2 = ttest_ind(RT_start_sub[0],RT_start_laterbranch_sub[0], equal_var=False)

x1, x2 = 2,3 
y, h, col = bx['caps'][3]._y[0] + 1, 0.3, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p2)

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
# fig.savefig(out_dir + 'conditional_undo_masError.pdf', dpi=600, bbox_inches='tight')


# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

bx = axs.boxplot(
    [
        RT_notstart_nobranch_sub[0],
        RT_notstart_sub[0],
        RT_nostart_laterbranch_sub[0]
    ],
    positions =[1,2,3],
    widths = 0.3,
    showfliers=False,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,2,2.5,3])
# axs.set_yticks(np.linspace(0,0.16,5))
axs.set_xticklabels(labels = ['not branching',
                              '\n first visit',
                              'branching',
                              '\n later visit'])#,fontsize=18
axs.set_ylabel('Response time (s)')



# run 2-independent-sample t test
stat1, p1 = ttest_ind(RT_notstart_nobranch_sub[0], RT_notstart_sub[0], equal_var=False)
x1, x2 = 1,2  
y, h, col = bx['caps'][3]._y[0] + 0.3, 0.1, 'k'


axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

#--------------------------------------
stat2, p2 = ttest_ind(RT_notstart_sub[0],RT_nostart_laterbranch_sub[0], equal_var=False)

x1, x2 = 2,3 
y, h, col = bx['caps'][3]._y[0] + 0.3, 0.1, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p2)

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
# fig.savefig(out_dir + 'conditional_undo_masError.pdf', dpi=600, bbox_inches='tight')
# -


# ### bar plot

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
bx = axs.bar([1,2,3,4,5,6],
             [np.mean(RT_start_sub),np.mean(RT_start_nobranch_sub),np.mean(RT_start_laterbranch_sub),
              np.mean(RT_notstart_sub),np.mean(RT_notstart_nobranch_sub),np.mean(RT_nostart_laterbranch_sub)],
             color=(.7,.7,.7), edgecolor = 'k', 
             yerr=[RT_start_sub_sem,RT_start_nobranch_sub_sem,RT_start_laterbranch_sub_sem,
                   RT_notstart_sub_sem,RT_notstart_nobranch_sub_sem,RT_nostart_laterbranch_sub_sem])

axs.set_xticks([1,2,3,4,5,6])
# axs.set_yticks(np.linspace(0,0.16,5))
axs.set_xticklabels(labels = ['start',
                              'start_nobranch',
                              'start_laterbranch',
                              
                              'notstart',
                              'notstart_nobranch',
                              'nostart_laterbranch'])#,fontsize=18
axs.set_ylabel('Response time (s)')



# run 2-independent-sample t test
stat1, p1 = ttest_ind(RT_start_sub, RT_start_nobranch_sub,equal_var=False)
x1, x2 = 1,2  
if bx[0].get_height() > bx[1].get_height():
    y, h, col = bx[0].get_height() + 1, 0.3, 'k'
else:
    y, h, col = bx[1].get_height() + 1, 0.3, 'k'

axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

#--------------------------------------
stat2, p2 = ttest_ind(RT_notstart_sub,RT_notstart_nobranch_sub,equal_var=False)

x1, x2 = 4,5 
y, h, col = bx[2].get_height() + 1, 0.3, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p2)

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
# fig.savefig(out_dir + 'conditional_undo_masError.pdf', dpi=600, bbox_inches='tight')
# -


# ## average within puzzle

# +
# the first visit of a branching node, and it is the start city
index_start = data_choice_level.index[(data_choice_level['branchingFirst'] == True)&(data_choice_level['currNumCities'] == 1)]
RT_start_puzzle = get_branchRT(index_start,'puzzleID')

index_start_nobranch = data_choice_level.index[(data_choice_level['branching'] != True)&(data_choice_level['currNumCities'] == 1)&(data_choice_level['condition']==1)]
RT_start_nobranch_puzzle = get_branchRT(index_start_nobranch,'puzzleID')

index_start_laterbranch = data_choice_level.index[(data_choice_level['branchingFirst'] != True)&(data_choice_level['branching'] == True)&(data_choice_level['currNumCities'] == 1)]
RT_start_laterbranch_puzzle = get_branchRT(index_start_laterbranch,'puzzleID')


# +
# the first visit of a branching node, and it is not a start city
index_notstart = data_choice_level.index[(data_choice_level['branchingFirst'] == True)&(data_choice_level['currNumCities'] != 1)]
RT_nostart_puzzle = get_branchRT(index_notstart,'puzzleID')

index_notstart_nobranch = data_choice_level.index[(data_choice_level['branching'] != True)&(data_choice_level['currNumCities'] != 1)&(data_choice_level['condition']==1)&(data_choice_level['checkEnd'] != 1)&(data_choice_level['submit'] != 1)] 
RT_nostart_nobranch_puzzle = get_branchRT(index_notstart_nobranch,'puzzleID')

index_nostart_laterbranch = data_choice_level.index[(data_choice_level['branchingFirst'] != True)&(data_choice_level['branching'] == True)&(data_choice_level['currNumCities'] != 1)&(data_choice_level['checkEnd'] != 1)&(data_choice_level['submit']!=1)]
RT_nostart_laterbranch_puzzle = get_branchRT(index_nostart_laterbranch,'puzzleID')

# -


# ### box plot

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

bx = axs.boxplot(
    [
        RT_start_nobranch_puzzle[0],
        RT_start_puzzle[0],
        RT_start_laterbranch_puzzle[0]
    ],
    positions =[1,2,3],
    widths = 0.3,
    showfliers=False,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,2,2.5,3])
axs.set_ylim([0,13])
axs.set_xticklabels(labels = ['not branching',
                              '\n first visit',
                              'branching',
                              '\n later visit'])#,fontsize=18
axs.set_ylabel('Response time (s)')



# run 2-independent-sample t test
stat1, p1 = ttest_ind(RT_start_nobranch_puzzle[0], RT_start_puzzle[0], equal_var=False)
x1, x2 = 1,2  
y, h, col = bx['caps'][3]._y[0] + 1, 0.3, 'k'


axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

#--------------------------------------
stat2, p2 = ttest_ind(RT_start_puzzle[0],RT_start_laterbranch_puzzle[0], equal_var=False)

x1, x2 = 2,3 
y, h, col = bx['caps'][3]._y[0] + 1, 0.3, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p2)

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
fig.savefig(out_dir + 'branching_RT_puzzle.png', dpi=600, bbox_inches='tight')


# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

bx = axs.boxplot(
    [
        RT_nostart_nobranch_puzzle[0],
        RT_nostart_puzzle[0],
        RT_nostart_laterbranch_puzzle[0]
    ],
    positions =[1,2,3],
    widths = 0.3,
    showfliers=False,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,2,2.5,3])
axs.set_yticks(np.linspace(0,3,7))
axs.set_xticklabels(labels = ['not branching',
                              '\n first visit',
                              'branching',
                              '\n later visit'])#,fontsize=18
axs.set_ylabel('Response time (s)')



# run 2-independent-sample t test
stat1, p1 = ttest_ind(RT_nostart_nobranch_puzzle[0], RT_nostart_puzzle[0], equal_var=False)
x1, x2 = 1,2  
y, h, col = bx['caps'][3]._y[0] + 0.1, 0.1, 'k'


axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p1)

#--------------------------------------
stat2, p2 = ttest_ind(RT_nostart_puzzle[0],RT_nostart_laterbranch_puzzle[0], equal_var=False)

x1, x2 = 2,3 
y, h, col = bx['caps'][3]._y[0] + 0.1, 0.1, 'k'
axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
text(p2)

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
# fig.savefig(out_dir + 'conditional_undo_masError.pdf', dpi=600, bbox_inches='tight')
# -


# ### bar plot

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
bx = axs.bar([1,2,3],
             [
              np.mean(RT_start_puzzle),np.mean(RT_start_nobranch_puzzle),np.mean(RT_start_laterbranch_puzzle),
#               np.mean(RT_notstart_sub),np.mean(RT_notstart_nobranch_sub),np.mean(RT_nostart_laterbranch_sub)
             ],
             color=(.7,.7,.7), edgecolor = 'k', 
             yerr=[RT_start_puzzle_sem,RT_start_nobranch_puzzle_sem,RT_start_laterbranch_puzzle_sem,
#                    RT_notstart_sub_sem,RT_notstart_nobranch_sub_sem,RT_nostart_laterbranch_sub_sem
                  ])

axs.set_xticks([1,2,3])
# axs.set_yticks(np.linspace(0,0.16,5))
axs.set_xticklabels(labels = ['start',
                              'start_nobranch',
                              'start_laterbranch',
                              
#                               'notstart',
#                               'notstart_nobranch',
#                               'nostart_laterbranch'
                             ])#,fontsize=18
axs.set_ylabel('Response time (s)')



# # run 2-independent-sample t test
# stat1, p1 = ttest_ind(RT_start_sub, RT_start_nobranch_sub,equal_var=False)
# x1, x2 = 1,2  
# if bx[0].get_height() > bx[1].get_height():
#     y, h, col = bx[0].get_height() + 1, 0.3, 'k'
# else:
#     y, h, col = bx[1].get_height() + 1, 0.3, 'k'

# axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# text(p1)

# #--------------------------------------
# stat2, p2 = ttest_ind(RT_notstart_sub,RT_notstart_nobranch_sub,equal_var=False)

# x1, x2 = 4,5 
# y, h, col = bx[2].get_height() + 1, 0.3, 'k'
# axs.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# text(p2)

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
# fig.savefig(out_dir + 'conditional_undo_masError.pdf', dpi=600, bbox_inches='tight')
# -
# # first RT and numUndo

# +
RT1_undo = undo_condition_data.groupby(['puzzleID'])['RT1'].mean()
RT1_basic = basic_condition_data.groupby(['puzzleID'])['RT1'].mean()
numFullUndo_sub = undo_condition_data.groupby(['puzzleID'])['numFullUndo'].mean()

# %matplotlib notebook
plt.xlabel("first-move RT")
plt.ylabel("average number of undo")

plt.scatter(RT1_undo,numFullUndo_sub, label='undo condition')
plt.scatter(RT1_basic,numFullUndo_sub, label='basic condition')

plt.legend()

print(stats.spearmanr(numFullUndo_sub,RT1_undo))
print(stats.spearmanr(numFullUndo_sub,RT1_basic))


# +
RT1_basic = basic_condition_data.groupby(['puzzleID'])['RT1'].mean()
numError_sub = basic_condition_data.groupby(['puzzleID'])['numError'].mean()

# %matplotlib notebook
plt.xlabel("first-move RT")
plt.ylabel("average number of error")

plt.scatter(RT1_basic,numError_sub,label='basic condition')

plt.legend()

stats.spearmanr(RT1_basic,numError_sub)


# +
RT1_basic = basic_condition_data.groupby(['puzzleID'])['RT1'].mean()
sumSeverityErrors_sub = basic_condition_data.groupby(['puzzleID'])['sumSeverityErrors'].mean()

# %matplotlib notebook
plt.scatter(RT1_basic,sumSeverityErrors_sub)
stats.spearmanr(RT1_basic,sumSeverityErrors_sub)
# -


