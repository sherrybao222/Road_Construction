# ---
# jupyter:
#   jupytext:
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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import shapiro
from scipy.stats import normaltest

home_dir = '/Users/dbao/google_drive_db'+'/road_construction/data/2022_online/'
map_dir = 'active_map/'
data_dir  = 'data/preprocessed'
out_dir = 'figures/figures_all/'
R_out_dir = home_dir + 'R_analysis_data/'

data_puzzle_level = pd.read_csv(R_out_dir +  'data.csv')
puzzleID_order_data = data_puzzle_level.sort_values(["subjects","puzzleID"])
data_choice_level = pd.read_csv(R_out_dir +  'choice_level/choicelevel_data.csv')

numCity_basic = data_puzzle_level[data_puzzle_level['condition']==0].groupby(['subjects'])['numCities'].mean()
numCity_undo = data_puzzle_level[data_puzzle_level['condition']==1].groupby(['subjects'])['numCities'].mean()
mas =  np.mean(data_puzzle_level['mas'])

# +
# %matplotlib notebook

n_sub = len(numCity_basic)
fig0, ax0 = plt.subplots()
subInd = [x+1 for x in sorted(range(len(numCity_basic)),key=lambda k:list(numCity_basic)[k])]
sorted_numCity_undo = [numCity_undo[x] for x in subInd]
ax0.bar(list(range(1,n_sub+1)),sorted(numCity_basic),color = "#dda15e",alpha=0.5)
ax0.bar(list(range(1,n_sub+1)),sorted(sorted_numCity_undo),alpha=0.5)
ax0.axhline(mas)
ax0.invert_xaxis()
ax0.set_xticks([])
# ax0.set_xticklabels(subInd,{'fontsize': 6})
ax0.set_xlabel("subjects")
ax0.set_ylabel("average number of connected cities")
# fig0.suptitle('Relative change of count of defer in group condition compared to self condition')
# print('the mean of relative change in group level is:'+ str(statistics.mean(groupDeferRate)))

# -

num_undo = data_puzzle_level[data_puzzle_level['condition']==1].groupby(['subjects'])['numUNDO'].mean()

# +
# %matplotlib notebook

n_sub = len(num_undo)
fig0, ax0 = plt.subplots()
subInd = [x+1 for x in sorted(range(len(num_undo)),key=lambda k:list(num_undo)[k])]
ax0.bar(list(range(1,n_sub+1)),sorted(num_undo),color = "#dda15e")
ax0.set_xlabel("subjects")
ax0.set_ylabel("average counts of undo usage")
ax0.invert_xaxis()
ax0.set_xticks([])
# -

numFullUndo = data_puzzle_level[data_puzzle_level['condition']==1].groupby(['subjects'])['numFullUndo'].mean()

# +
# %matplotlib notebook

n_sub = len(numFullUndo)
fig0, ax0 = plt.subplots()
subInd = [x+1 for x in sorted(range(len(numFullUndo)),key=lambda k:list(numFullUndo)[k])]
ax0.bar(list(range(1,n_sub+1)),sorted(numFullUndo),color = "#dda15e")
ax0.set_xlabel("subjects")
ax0.set_ylabel("average counts of full undoing")
ax0.invert_xaxis()
ax0.set_xticks([])
# -


