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

# +
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# -

from scipy.stats import shapiro
from scipy.stats import normaltest

home_dir = '/Users/dbao/google_drive_db'+'/road_construction/data/2022_online/'
map_dir = 'active_map/'
data_dir  = 'data/preprocessed'
out_dir = home_dir + 'figures/cogsci_2022/'
R_out_dir = home_dir + 'R_analysis_data/'

# +
data_puzzle_level = pd.read_csv(R_out_dir +  'data.csv')
puzzleID_order_data = data_puzzle_level.sort_values(["subjects","puzzleID"])
data_choice_level = pd.read_csv(R_out_dir +  'choice_level/choicelevel_data.csv')

single_condition_data = puzzleID_order_data[puzzleID_order_data['condition']==1].copy()
single_condition_data = single_condition_data.reset_index()
sc_data_choice_level = data_choice_level[data_choice_level['condition']==1].reset_index()
# -

# # Basic counts

# ## histogram of MAS

# +
n_mas = single_condition_data[single_condition_data['subjects'] == 1].groupby(['mas'])['index'].count() # only basic condition

# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
axs.bar(range(7,13),
        n_mas/sum(n_mas),
        color = (.7,.7,.7), 
        edgecolor = 'k',)
axs.set_ylabel('proportion') 
axs.set_xlabel('mas') 
plt.show()
# -

# ## histogram of number of cities within reach

# +
n_reach = data_choice_level[data_choice_level['condition']==0]['within_reach'] # only basic condition

# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
axs.hist(n_reach,
        color = (.7,.7,.7), 
        edgecolor = 'k',)
axs.set_ylabel('counts') 
axs.set_xlabel('number of cities within reach') 
plt.show()
# -

# ## histogram of number of undos in a seq

# +
# histogram of number of undos in a sequence
num_undos = np.array(data_choice_level.index[data_choice_level['lastUndo']==1]) - np.array(data_choice_level.index[data_choice_level['firstUndo']==1]) + 1

# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
axs.hist(num_undos, bins=10,color=[.7,.7,.7])
plt.ylabel('Frequency')
plt.xlabel('Number of Undos in a sequence')
axs.set_xticks(np.linspace(1,10,10))
# axs.set_xticklabels('')

fig.set_figwidth(4)
fig.set_figheight(4)
# fig.savefig(out_dir + 'undo_num_seq.png', dpi=600, bbox_inches='tight')
# fig.savefig(out_dir + 'undo_num_seq.pdf', dpi=600, bbox_inches='tight')
# -

# # individual number of connected cities

numCity_basic = data_puzzle_level[data_puzzle_level['condition']==0].groupby(['subjects'])['numCities'].mean()
numCity_undo = data_puzzle_level[data_puzzle_level['condition']==1].groupby(['subjects'])['numCities'].mean()
mas =  np.mean(data_puzzle_level['mas'])

# +
# %matplotlib notebook

n_sub = len(numCity_basic)
fig0, ax0 = plt.subplots()
subInd = [x+1 for x in sorted(range(len(numCity_basic)),key=lambda k:list(numCity_basic)[k])]
sorted_numCity_undo = [numCity_undo[x] for x in subInd]
ax0.bar(list(range(1,n_sub+1)),sorted(numCity_basic),color = "#dda15e",alpha=0.5,label='basic')
ax0.bar(list(range(1,n_sub+1)),sorted_numCity_undo,alpha=0.5,label='undo')
ax0.axhline(mas)
ax0.invert_xaxis()
ax0.set_xticks([])
# ax0.set_xticklabels(subInd,{'fontsize': 6})
ax0.set_xlabel("subjects")
ax0.set_ylabel("average number of connected cities")
ax0.legend()
# fig0.suptitle('Relative change of count of defer in group condition compared to self condition')
# print('the mean of relat## Number of undo

### subject-level

undo_puzzle = single_condition_data[single_condition_data['numUNDO']>0].groupby(['subjects']).size()
count = [len(single_condition_data.groupby(['subjects']).size())]
for i in range(1,47):
    count.append(sum(undo_puzzle>=i))

fig, axs = plt.subplots()

plt.bar(list(range(0,47)),count)
axs.set_xlabel("undo in >= number of puzzles")
axs.set_ylabel("number of subjects")
# axs.plot(bins[1][:-1], bins[0], color = '#81b29a', linewidth=3)

### puzzle-level

order = single_condition_data.groupby(['puzzleID'])['numFullUndo'].mean().to_frame()
sort_order = order.sort_values('numFullUndo')

# %matplotlib notebook
fig, axs = plt.subplots(1, 1)

bx = sns.barplot(x='puzzleID', y='numFullUndo', data = single_condition_data, color = '#ccd5ae',order=sort_order.index) 
ive change in group level is:'+ str(statistics.mean(groupDeferRate)))
plt.show()
#fig0.savefig(out_dir + 'connected_individual.pdf', dpi=600, bbox_inches='tight')

# -

# # Number of undo

# ## subject-level

# ### undo in >= number of puzzles

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
# -

# ### undo presses

num_undo = data_puzzle_level[data_puzzle_level['condition']==1].groupby(['subjects'])['numUNDO'].mean()

# +
# %matplotlib notebook

n_sub = len(num_undo)
fig0, ax0 = plt.subplots()
subInd = [x+1 for x in sorted(range(len(num_undo)),key=lambda k:list(num_undo)[k])]
ax0.bar(list(range(1,n_sub+1)),sorted(num_undo),color = "#dda15e")
ax0.set_xlabel("subjects")
ax0.set_ylabel("average number of undo presses")
ax0.invert_xaxis()
ax0.set_xticks([])
# -

# ### undo sequences

# +
sub_mean = single_condition_data.groupby(['subjects'])['numFullUndo'].mean().to_frame()
sub_sem = single_condition_data.groupby(['subjects'])['numFullUndo'].sem().to_frame()
sort_sub_mean = sub_mean.sort_values('numFullUndo')
sort_sub_sem = sub_sem.reindex(sort_sub_mean.index)

# %matplotlib notebook
fig, axs = plt.subplots(1, 1)

n_sub = len(sub_mean)
axs.bar(list(range(1,n_sub+1)),sort_sub_mean['numFullUndo'],
        yerr = sort_sub_sem['numFullUndo'],
        color = "#d4a373")

axs.set_xticks([])
axs.set_xlabel("Subjects")
axs.set_ylabel("Average number of undo sequences")
# axs.invert_xaxis()


fig.savefig(out_dir + 'undo_sequence_subject.png', dpi=600, bbox_inches='tight')
# -
# ### correlation of undoing presses and sequences 

np.unique(np.array(single_condition_data['numFullUndo'][single_condition_data['numUNDO']!=0]))
undo_once_count_sub = single_condition_data.groupby(['subjects'])['numUNDO'].mean()
undo_once_sequences = single_condition_data.groupby(['subjects'])['numFullUndo'].mean()


# +
undo_presses = np.array(single_condition_data.groupby(['subjects'])['numUNDO'].mean())
undo_length  = np.array(single_condition_data.groupby(['subjects'])['numUNDO'].mean())/np.array(single_condition_data.groupby(['subjects'])['numFullUndo'].mean())

undo_length[np.isnan(undo_length)] = 0

# +
fig1, ax1 = plt.subplots()
# ax1.plot(undo_count_sub,undo_benefit_sub,'o',s=2,c='k')
ax1.scatter(undo_presses,undo_length,10,c='k')

ax1.set_xlabel('Average number of pressing undo buttons')
ax1.set_ylabel('Average number of undoing in a sequence')
# fig1.savefig(out_dir + 'undo_num_length.png', dpi=600, bbox_inches='tight')
# fig1.savefig(out_dir + 'undo_num_length.pdf', dpi=600, bbox_inches='tight')

# -


from scipy.stats import spearmanr, pearsonr
# spearmanr(undo_presses,undo_length)
pearsonr(undo_presses,undo_length)

# +
fig1, ax1 = plt.subplots()

ax1.plot([0,19],[0,19],'--',c='k')
ax1.scatter(undo_presses,undo_once_sequences, 40, marker='o', facecolors='none', edgecolors='k')

ax1.set_xlim(0,19)
ax1.set_xticks(np.linspace(0,18,7).astype(np.int16))

ax1.set_ylim(0,19)
ax1.set_yticks(np.linspace(0,18,7).astype(np.int16))

fig1.set_figwidth(4)
fig1.set_figheight(4)

ax1.set_xlabel('Average number of undo presses')
ax1.set_ylabel('Average number of undo sequences')
fig1.savefig(out_dir + 'undo_seq_presses_scatter.png', dpi=600, bbox_inches='tight')
# fig1.savefig(out_dir + 'undo_num_seq.pdf', dpi=600, bbox_inches='tight')

# -

# ## puzzle-level

# +
puzzle_mean = single_condition_data.groupby(['puzzleID'])['numFullUndo'].mean().to_frame()
puzzle_sem = single_condition_data.groupby(['puzzleID'])['numFullUndo'].sem().to_frame()
sort_puzzle_mean = puzzle_mean.sort_values('numFullUndo')
sort_puzzle_sem = puzzle_sem.reindex(sort_puzzle_mean.index)

# %matplotlib notebook
fig, axs = plt.subplots(1, 1)

n_puzzle = len(puzzle_mean)
axs.bar(list(range(1,n_puzzle+1)),sort_puzzle_mean['numFullUndo'],
        yerr = sort_puzzle_sem['numFullUndo'],
        color = "#ccd5ae")

# bx = sns.barplot(x='puzzleID', y='numFullUndo', 
#                  data = single_condition_data, 
#                  color = '#ccd5ae',
#                  order=sort_order.index) 

axs.xaxis.set_ticks([])
axs.set_xlabel('Puzzles')
axs.set_ylabel('Average number of undo sequences')

fig.savefig(out_dir + 'undo_sequence_puzzle.png', dpi=600, bbox_inches='tight')
# -

#
