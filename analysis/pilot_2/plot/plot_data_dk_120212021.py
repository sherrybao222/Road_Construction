import json
import numpy as np
from statistics import mean, stdev, median
from operator import eq
import seaborn as sns;

sns.set()
import matplotlib.pyplot as plt
import math
import matplotlib.lines as mlines
from scipy.stats import wilcoxon
import pandas as pd

## ==========================================================================
# import data

data_all = []  # prepare for all data
subs = [1, 2, 4]  # subject index
budget = [200, 350, 400]

# directories
home_dir = './'
map_dir = 'active_map/'
data_dir  = 'data/in-lab-pre-pilot/'
# out_dir = 'road_construction/experiments/pilot_0320/figures/'

from glob import glob
import numpy as np
from numpy import genfromtxt

flist = glob(home_dir + data_dir + '/preprocess2_sub_*.csv')
# import experiment data
for fname in flist:
    with open(fname, 'r') as f:
        all_data = pd.read_csv(f)
        data_all.append(all_data)


# load basic map
with open(home_dir + map_dir + 'basicMap.json', 'r') as file:
    basic_map = json.load(file)
# load undo map
with open(home_dir + map_dir + 'undoMap.json', 'r') as file:
    undo_map = json.load(file)


out_dir = 'figures_prepilot/'
## ==========================================================================
# rc scatter all - final
## ==========================================================================
BASIC_ALL = {'basic_list':[],'basic':[]}
UNDO_ALL  = {'undo_list':[],'undo':[]}

fig, axs = plt.subplots(2, len(data_all), sharey=True)
for j in range(0, 2):
    for i in range(0, len(data_all)):
        undo_trials  = data_all[i][data_all[i].map_name == 'undo']
        u_undo_maps = np.unique(np.array(undo_trials['map_id']))
        undo_list = [] # MAS
        undo      = [] # Number of cities connected
        ti = 0
        prev_mapid = -1 # arbitrary number
        # for ti in range(undo_trials.shape[0]):
        while ti < undo_trials.shape[0]:
            if prev_mapid != np.array(undo_trials.map_id)[ti]: # which means if the trial has changed
                single_trial = undo_trials[np.array(undo_trials.map_id) == np.array(undo_trials.map_id)[ti]]
                undo_list.append(np.array(single_trial.mas_all)[0])
                undo.append(np.array(single_trial.n_city_all)[-1])
                prev_mapid = np.array(undo_trials.map_id)[ti]
            ti += 1

        basic_trials = data_all[i][data_all[i].map_name == 'basic']
        u_basic_maps = np.unique(np.array(undo_trials['map_id']))
        basic_list = [] # MAS
        basic      = [] # Number of cities connected
        ti = 0
        prev_mapid = -1 # arbitrary number
        # for ti in range(undo_trials.shape[0]):
        while ti < basic_trials.shape[0]:
            if prev_mapid != np.array(basic_trials.map_id)[ti]: # which means if the trial has changed
                single_trial = basic_trials[np.array(basic_trials.map_id) == np.array(basic_trials.map_id)[ti]]
                basic_list.append(np.array(single_trial.mas_all)[0])
                basic.append(np.array(single_trial.n_city_all)[-1])
                prev_mapid = np.array(basic_trials.map_id)[ti]
            ti += 1

        BASIC_ALL['basic_list'].append(basic_list)
        BASIC_ALL['basic'].append(basic)
        UNDO_ALL['undo_list'].append(undo_list)
        UNDO_ALL['undo'].append(undo)

        u, c = np.unique(np.c_[basic_list, basic], return_counts=True, axis=0)
        u1, c1 = np.unique(np.c_[undo_list, undo], return_counts=True, axis=0)
        if j == 0:
            axs[j, i].scatter(u[:, 0], u[:, 1], s=c * 15, facecolors='none',
                              edgecolors='#0776d8')
        else:
            axs[j, i].scatter(u1[:, 0], u1[:, 1], s=c1 * 15, facecolors='none',
                              edgecolors='#e13f42')
        # ax = sns.heatmap(num_mx,cmap="YlGnBu",linewidths=.3,linecolor = 'k')

        axs[j, i].set_xlim((4, 12))
        axs[j, i].set_ylim((4, 12))
        x0, x1 = axs[j, i].get_xlim()
        y0, y1 = axs[j, i].get_ylim()
        axs[j, i].set_aspect(abs(x1 - x0) / abs(y1 - y0))

        h1, = axs[j, i].plot(axs[j, i].get_xlim(), axs[j, i].get_ylim(), ls="--", color='k', label='optimal')  # diagnal
        h2, = axs[j, i].plot((8, 12), (4, 8), ls="--", color='#968aad', label='greedy')  # diagnal
        axs[j, i].grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)
        axs[j, i].set_facecolor('white')

        axs[j, i].set_xticks(np.arange(x0, x1, 1.0))
        axs[j, i].set_yticks(np.arange(y0, y1, 1.0))
        axs[0, i].title.set_text('S' + str(i + 1))

        axs[1, i].set_xlabel("Number connectable (maximal)")
        axs[j, 0].set_ylabel("Number connected (actual)")

title_1 = mlines.Line2D([], [], color='white', label='without undo')
title_2 = mlines.Line2D([], [], color='white', label='with undo')
rc_led_1 = mlines.Line2D([], [], color='white', marker='o', markeredgecolor='#727bda',
                         markersize=math.sqrt(1 * 15), label='1')
rc_led_2 = mlines.Line2D([], [], color='white', marker='o', markeredgecolor='#727bda',
                         markersize=math.sqrt(5 * 15), label='5')
rc_led_3 = mlines.Line2D([], [], color='white', marker='o', markeredgecolor='#727bda',
                         markersize=math.sqrt(10 * 15), label='10')
rc_led_4 = mlines.Line2D([], [], color='white', marker='o', markeredgecolor='#727bda',
                         markersize=math.sqrt(15 * 15), label='15')
undo_led_1 = mlines.Line2D([], [], color='white', marker='o', markeredgecolor='#e13f42',
                           markersize=math.sqrt(1 * 15), label='1')
undo_led_2 = mlines.Line2D([], [], color='white', marker='o', markeredgecolor='#e13f42',
                           markersize=math.sqrt(5 * 15), label='5')
undo_led_3 = mlines.Line2D([], [], color='white', marker='o', markeredgecolor='#e13f42',
                           markersize=math.sqrt(10 * 15), label='10')
undo_led_4 = mlines.Line2D([], [], color='white', marker='o', markeredgecolor='#e13f42',
                           markersize=math.sqrt(15 * 15), label='15')

lgd = plt.legend(bbox_to_anchor=(2.04, 1), prop={'size': 12}, title="Number of trials", handletextpad=0.01,
                 handlelength=3,
                 handles=[title_1, rc_led_1, rc_led_2, rc_led_3, rc_led_4,
                          title_2, undo_led_1, undo_led_2, undo_led_3, undo_led_4], facecolor='white', ncol=2)
plt.legend(handles=[h1, h2], bbox_to_anchor=(2.04, 0.3), facecolor='white')
plt.gca().add_artist(lgd)

for vpack in lgd._legend_handle_box.get_children():
    vpack.get_children()[0].get_children()[0].set_width(0)

# ax.set_aspect('equal')
fig.set_figwidth(12)
fig.set_figheight(9)

plt.show()
fig.savefig(out_dir + 'rc_scatter_all.png', dpi=600, bbox_inches='tight')
plt.close(fig)
## ==========================================================================






## ==========================================================================
# rc_undo_hist - final
## ==========================================================================
# {'basic_list':[],'basic':[]}
# {'undo_list':[],'undo':[]}

BASIC_DIFF_ALL = {'diff':[]}
UNDO_DIFF_ALL  = {'diff_undo':[]}

fig, axs = plt.subplots(1, len(data_all), sharey=True)
for i in range(0,len(data_all)):
    diff = (np.array(BASIC_ALL['basic'][i]) - np.array(BASIC_ALL['basic_list'][i]) ).tolist()
    diff_undo = (np.array(UNDO_ALL['undo'][i]) - np.array(UNDO_ALL['undo_list'][i])).tolist()

    BASIC_DIFF_ALL['diff'].append(diff)
    UNDO_DIFF_ALL['diff_undo'].append(diff_undo)

    mean1 = mean(diff)
    mean2 = mean(diff_undo)

    temp = [diff, diff_undo]
    temp = np.array(temp)
    new = temp.transpose()
    axs[i].hist(new, range(-6, 2), color=['#0776d8', '#e13f42'], density=1,
                align='left', edgecolor='k')

    axs[i].set_ylim((0, 1))
    axs[i].set_xlim((-6, 1))
    axs[i].set_xticks(range(-5, 1))
    axs[i].set_yticks(np.arange(0, 1.1, 0.1))
    axs[i].set_yticklabels([0, '', 0.2, '', 0.4, '', 0.6, '', 0.8, '', 1.0])

    axs[i].set_facecolor('white')
    axs[i].spines['bottom'].set_color('k')
    axs[i].spines['left'].set_color('k')
    axs[i].tick_params(axis='y', colors='k', direction='in', left=True)
    axs[i].title.set_text('S' + str(i + 1))
    axs[i].text(-5.5, 0.7, 'without undo mean:' + '{:.2f}'.format(mean1), fontsize=6)
    axs[i].text(-5.5, 0.6, 'with undo mean:' + '{:.2f}'.format(mean2), fontsize=6)
axs[1].set_xlabel('Number (actual connected - maximum connectable)')
axs[0].set_ylabel('Proportion of trials')

import matplotlib.patches as mpatches

rc_led = mpatches.Patch(color='#0776d8', label='without undo')
undo_led = mpatches.Patch(color='#e13f42', label='with undo')
lgd = plt.legend(handles=[rc_led, undo_led], facecolor='white')

fig.set_figwidth(10)

plt.show()
fig.savefig(out_dir + 'rc_undo_hist_all.png', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)
## ==========================================================================


## ==========================================================================
# rc_undo_hist - final
## ==========================================================================
UNDO_P  = dict()
BASIC_P = dict()
for a in range(-6, 1):
    UNDO_P[str(a)] = []
    BASIC_P[str(a)] = []

UNDO_P['data']=[]
BASIC_P['data']=[]

for i in range(len(data_all)):
    # save for each
    undo_p  = []
    basic_p = []
    for a in range(-6, 1):
        BASIC_P[str(a)].append(np.sum(np.array(BASIC_DIFF_ALL['diff'][i]) == a) / np.array(BASIC_DIFF_ALL['diff'][i]).shape[0])
        UNDO_P[str(a)].append(np.sum(np.array(UNDO_DIFF_ALL['diff_undo'][i]) == a) / np.array(UNDO_DIFF_ALL['diff_undo'][i]).shape[0])
        basic_p.append(np.sum(np.array(BASIC_DIFF_ALL['diff'][i]) == a) / np.array(BASIC_DIFF_ALL['diff'][i]).shape[0])
        undo_p.append(np.sum(np.array(UNDO_DIFF_ALL['diff_undo'][i]) == a) / np.array(UNDO_DIFF_ALL['diff_undo'][i]).shape[0])

    UNDO_P['data'].append(undo_p)
    BASIC_P['data'].append(basic_p)

undo_p_ = np.array(UNDO_P['data'])
basic_p_ = np.array(BASIC_P['data'])

mean_undo_p_  = np.mean(undo_p_, axis = 0)
mean_basic_p_ = np.mean(basic_p_, axis = 0)

std_undo_p_  = np.std(undo_p_, axis = 0)/np.sqrt(len(data_all))
std_basic_p_ = np.std(basic_p_, axis = 0)/np.sqrt(len(data_all))

fig, ax = plt.subplots()
# basic_
x_pos = np.arange(-6,1)-.1
ax.bar(x_pos, mean_basic_p_, yerr=std_basic_p_, align='center', alpha=1, width=.2,color='#0776d8', ecolor='black', capsize=10)
ax.set_ylabel('Proportion of trials')
# ax.set_xticks(x_pos)
ax.set_title('Number (actual connected - maximum connectable)')

# undo_
x_pos = np.arange(-6,1)+.1
ax.bar(x_pos, mean_undo_p_, yerr=std_undo_p_, align='center', alpha=1, width=.2,color='#e13f42', ecolor='black', capsize=10)
ax.set_ylabel('Proportion of trials')
# ax.set_xticks(x_pos)
ax.set_title('Number (actual connected - maximum connectable)')

x_pos = np.arange(-6,1)
ax.set_xticks(x_pos)

rc_led = mpatches.Patch(color='#0776d8', label='without undo')
undo_led = mpatches.Patch(color='#e13f42', label='with undo')
lgd = plt.legend(handles=[rc_led, undo_led], facecolor='white')
ax.yaxis.grid(True)

fig.set_figwidth(10)

plt.show()
fig.savefig(out_dir + 'rc_undo_hist_all_across_subjects.png', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)
## ==========================================================================


## ==========================================================================
# number undos boxplot - final
## ==========================================================================
N_UNDO_ALL = {'n_undo':[]}

fig, axs = plt.subplots(1, len(data_all), sharey=True)
for i in range(0, len(data_all)):
    undo_trials = data_all[i][data_all[i].map_name == 'undo']
    u_undo_maps = np.unique(np.array(undo_trials['map_id']))
    undo = []  # sum of undos
    ti = 0
    prev_mapid = -1  # arbitrary number
    # for ti in range(undo_trials.shape[0]):
    while ti < undo_trials.shape[0]:
        if prev_mapid != np.array(undo_trials.map_id)[ti]:  # which means if the trial has changed
            single_trial = undo_trials[np.array(undo_trials.map_id) == np.array(undo_trials.map_id)[ti]]
            undo.append(np.sum(np.array(single_trial.undo_all)))
            prev_mapid = np.array(undo_trials.map_id)[ti]
        ti += 1

    N_UNDO_ALL['n_undo'].append(undo)


    axs[i].boxplot(undo, widths=0.6)
    axs[i].plot([1] * len(undo), undo, 'o',
                markerfacecolor='#727bda', markeredgecolor='none', alpha=0.2)

    axs[i].set_ylim((-4, 40))
    axs[i].set_xlim((0, 2))

    axs[i].set_xticklabels([])

    axs[i].set_facecolor('white')
    axs[i].spines['bottom'].set_color('k')
    axs[i].spines['left'].set_color('k')
    axs[i].title.set_text('S' + str(i + 1))
axs[1].set_xlabel('Number of undos per trial')
axs[0].set_ylabel('Number of undos per trial')
plt.show()
fig.savefig(out_dir + 'n_undo_hist_all.png', dpi=600)
plt.close(fig)
## ==========================================================================


## ==========================================================================
# number undos bar - across subjects
## ==========================================================================
N_UNDO_ALL['data'] = []
for i in range(0, len(data_all)):
    N_UNDO_ALL['data'].append(np.mean(N_UNDO_ALL['n_undo'][i]))

mean_n_undo = np.mean(N_UNDO_ALL['data'])
std_n_undo  = np.std(N_UNDO_ALL['data'])/np.sqrt(len(data_all))
fig, ax = plt.subplots()
x_pos = [1]
ax.bar(1, mean_n_undo, yerr=std_n_undo, align='center', alpha=1, width=.2,color='#e13f42', ecolor='black', capsize=10)
ax.set_ylabel('Number of undos per trial')
ax.set_title('')
ax.set_xticks([1])
ax.yaxis.grid(True)
plt.xlim(.5,1.5)

fig.set_figwidth(2)

plt.show()
fig.savefig(out_dir + 'n_undo_hist_all_across_subjects.png', dpi=600, bbox_inches='tight')
plt.close(fig)
## ==========================================================================


## ==========================================================================
# number of undos (y) with MAS (x)
## ==========================================================================
N_UNDO_MAS_ALL = {'n_undo_mas':[], 'p_n_undo_mas':[], 'mas_ind':[]}
mas_ind = np.linspace(1,12,12).astype(np.int16).tolist() # index for the MAS of i-th trial.
N_UNDO_MAS_ALL['mas_ind'] = mas_ind

for i in range(len(data_all)):
    undo_trials = data_all[i][data_all[i].map_name == 'undo']
    u_undo_maps = np.unique(np.array(undo_trials['map_id']))
    ti = 0
    prev_mapid = -1  # arbitrary number

    # empty list to save per subject
    mas_ = [0]*12

    # for ti in range(undo_trials.shape[0]):
    while ti < undo_trials.shape[0]:
        if prev_mapid != np.array(undo_trials.map_id)[ti]:  # which means if the trial has changed
            single_trial = undo_trials[np.array(undo_trials.map_id) == np.array(undo_trials.map_id)[ti]]
            single_undo_trial = single_trial[np.array(single_trial.undo_all) == 1]

            for tti in single_undo_trial.n_city_all:
                mas_[tti] += 1
            prev_mapid = np.array(undo_trials.map_id)[ti]
        ti += 1

    N_UNDO_MAS_ALL['n_undo_mas'].append(mas_)
    N_UNDO_MAS_ALL['p_n_undo_mas'].append((np.array(mas_) / np.sum(mas_)).tolist())

data_ = np.array(N_UNDO_MAS_ALL['p_n_undo_mas'])
data_ = data_[~np.isnan(data_).any(axis=1)] # exclude some participants who never undo.

fig, axs = plt.subplots()
mean_data_ = np.mean(data_, axis = 0)
std_data_  = np.std(data_, axis = 0)/np.sqrt(len(data_))

axs.plot(mas_ind[1:], mean_data_[1:],  marker='o',
                markerfacecolor='#727bda', markeredgecolor='none')
axs.errorbar(mas_ind[1:], mean_data_[1:], yerr=std_data_[1:], capsize=3, ls='None', color='k')

axs.set_title('')
axs.set_xticks(mas_ind[1:])
axs.yaxis.grid(True)
fig.set_figwidth(4)
plt.show()
fig.savefig(out_dir + 'p_num_undo_X_MAS.png', dpi=600, bbox_inches='tight')
plt.close(fig)


data_ = np.array(N_UNDO_MAS_ALL['n_undo_mas'])
data_ = data_[~np.isnan(data_).any(axis=1)] # exclude some participants who never undo.

fig, axs = plt.subplots()
mean_data_ = np.mean(data_, axis = 0)
std_data_  = np.std(data_, axis = 0)/np.sqrt(len(data_))

axs.plot(mas_ind[1:], mean_data_[1:],  marker='o',
                markerfacecolor='#727bda', markeredgecolor='none')
axs.errorbar(mas_ind[1:], mean_data_[1:], yerr=std_data_[1:], capsize=3, ls='None', color='k')

axs.set_title('')
axs.set_xticks(mas_ind[1:])
axs.yaxis.grid(True)
fig.set_figwidth(4)
plt.show()
fig.savefig(out_dir + 'num_undo_X_MAS.png', dpi=600, bbox_inches='tight')
plt.close(fig)
## ==========================================================================


## ==========================================================================
# undo once and series of undo
## ==========================================================================
LEN_UNDO_ERROR = {'undo_once':[], 'undo_series':[], 'error_ind':[]}
error_ind = np.linspace(-12,12,25).astype(np.int16)
LEN_UNDO_ERROR['error_ind'] = error_ind.tolist()

for i in range(len(data_all)):
    undo_trials = data_all[i][data_all[i].map_name == 'undo']
    u_undo_maps = np.unique(np.array(undo_trials['map_id']))

    ti = 0
    prev_mapid = -1  # arbitrary number

    undo_once_errors_ = np.array([0]*25)
    undo_series_errors_ = np.array([0]*25)

    while ti < undo_trials.shape[0]:
        if prev_mapid != np.array(undo_trials.map_id)[ti]:  # which means if the trial has changed
            single_trial = undo_trials[np.array(undo_trials.map_id) == np.array(undo_trials.map_id)[ti]]

            mas_all_trial = np.array(single_trial.mas_all)
            errors_trial = [0, *(mas_all_trial[1:] - mas_all_trial[:-1]).tolist()]

            prev_undo = 0
            len_undo  = 0
            for tti in range(single_trial.shape[0]):
                if np.array(single_trial.undo_all)[tti] == 1:
                    len_undo += 1
                else:
                    if prev_undo ==1:
                        if len_undo == 1:
                            undo_once_errors_[error_ind == errors_trial[tti]] += 1
                        else:
                            undo_series_errors_[error_ind == errors_trial[tti]] += 1

                    len_undo = 0
                prev_undo = np.array(single_trial.undo_all)[tti]




            prev_mapid = np.array(undo_trials.map_id)[ti]
        ti += 1

    LEN_UNDO_ERROR['undo_once'].append(undo_once_errors_.tolist())
    LEN_UNDO_ERROR['undo_series'].append(undo_series_errors_.tolist())

fig, axs = plt.subplots(1,2)
# undo once
mean_undo_once = np.mean(np.array(LEN_UNDO_ERROR['undo_once']),axis=0)
std_undo_once = np.std(np.array(LEN_UNDO_ERROR['undo_once']),axis=0)/np.sqrt(len(data_all))

x_pos = np.arange(-7,1)
start_ind = np.where(error_ind == np.array(x_pos)[0])[0].squeeze()
end_ind   = np.where(error_ind == np.array(x_pos)[-1])[0].squeeze()

axs[0].bar(x_pos, mean_undo_once[start_ind:end_ind+1], yerr=std_undo_once[start_ind:end_ind+1], align='center', alpha=1, width=.8, ecolor='black', capsize=10)
axs[0].set_xlabel('Error')
axs[0].set_ylabel('Count')
axs[0].set_title('Undo once')
axs[0].set_xticks(x_pos)
axs[0].yaxis.grid(True)

# undo once
mean_undo_series = np.mean(np.array(LEN_UNDO_ERROR['undo_series']),axis=0)
std_undo_series = np.std(np.array(LEN_UNDO_ERROR['undo_series']),axis=0)/np.sqrt(len(data_all))

axs[1].bar(x_pos, mean_undo_series[start_ind:end_ind+1], yerr=std_undo_series[start_ind:end_ind+1], align='center', alpha=1, width=.8, ecolor='black', capsize=10)
axs[1].set_xlabel('Error')
axs[1].set_title('Series of undo')
axs[1].set_xticks(x_pos)
axs[1].yaxis.grid(True)

fig.set_figwidth(6)
plt.show()
fig.savefig(out_dir + 'undo_once_series_count.png', dpi=600, bbox_inches='tight')
plt.close(fig)


fig, axs = plt.subplots(1,2)
# undo once
data_undo_once = np.array(LEN_UNDO_ERROR['undo_once'])
undo_once_validsub = []
for i in range(len(data_all)):
    if not np.sum(data_undo_once[i,:]) == 0:
        undo_once_validsub.append(data_undo_once[i,:]/np.sum(data_undo_once[i,:]))
undo_once_validsub = np.array(undo_once_validsub)

mean_undo_once = np.mean(undo_once_validsub,axis=0)
std_undo_once = np.std(undo_once_validsub,axis=0)/np.sqrt(len(undo_once_validsub))

x_pos = np.arange(-7,1)
start_ind = np.where(error_ind == np.array(x_pos)[0])[0].squeeze()
end_ind   = np.where(error_ind == np.array(x_pos)[-1])[0].squeeze()

axs[0].bar(x_pos, mean_undo_once[start_ind:end_ind+1], yerr=std_undo_once[start_ind:end_ind+1], align='center', alpha=1, width=.8, ecolor='black', capsize=10)
axs[0].set_xlabel('Error')
axs[0].set_ylabel('Proportion')
axs[0].set_title('Undo once')
axs[0].set_xticks(x_pos)
axs[0].yaxis.grid(True)
axs[0].set_ylim(0,.8)

# undo once
data_undo_series = np.array(LEN_UNDO_ERROR['undo_series'])
undo_series_validsub = []
for i in range(len(data_all)):
    if not np.sum(data_undo_series[i,:]) == 0:
        undo_series_validsub.append(data_undo_series[i,:]/np.sum(data_undo_series[i,:]))
undo_series_validsub = np.array(undo_series_validsub)

mean_undo_series = np.mean(undo_series_validsub,axis=0)
std_undo_series = np.std(undo_series_validsub,axis=0)/np.sqrt(len(undo_series_validsub))

axs[1].bar(x_pos, mean_undo_series[start_ind:end_ind+1], yerr=std_undo_series[start_ind:end_ind+1], align='center', alpha=1, width=.8, ecolor='black', capsize=10)
axs[1].set_xlabel('Error')
axs[1].set_title('Series of undo')
axs[1].set_xticks(x_pos)
axs[1].yaxis.grid(True)
axs[1].set_ylim(0,.8)

fig.set_figwidth(6)
plt.show()
fig.savefig(out_dir + 'undo_once_series_p.png', dpi=600, bbox_inches='tight')
plt.close(fig)
## ==========================================================================












## ==========================================================================
# DATA SAVE FOR THE FURTHER ANALYSIS USING R
R_out_dir = 'R_analysis/'

# Puzzle level
# number of cities connected
numCities = []
# MAS
mas = []
# number of optimal solutions
nos = []
# undo condition or not : 1 for with undo condition and 0 for without undo condition
undo_c = []
# budget left after the maximum number of cities have been connected.
leftover = []
# number of errors in a puzzle
numError = []
# sum of severity of errors
sumSeverityErrors = []
# number of undos
numUNDO = []
# time taken for a trial
TT = []

for i in range(len(data_all)):
    ti = 0
    prev_mapid = -1  # arbitrary number
    prev_mapname = 'undo'

    # empty list to save per subject
    temp_numCities         = []
    temp_mas               = []
    temp_nos               = []
    temp_undo_c            = []
    temp_leftover          = []
    temp_numError          = []
    temp_sumSeverityErrors = []
    temp_numUNDO = []
    temp_TT = []

    # for ti in range(undo_trials.shape[0]):
    while ti < data_all[i].shape[0]:
        if (prev_mapid != np.array(data_all[i].map_id)[ti]) or (prev_mapname != data_all[i].map_name[ti]):  # which means if the trial has changed
            single_trial = data_all[i][np.array(data_all[i].map_id) == np.array(data_all[i].map_id)[ti]]
            temp_numCities.append(np.array(single_trial.n_city_all)[-1])
            temp_mas.append(np.array(single_trial.mas_all)[0])
            temp_nos.append(np.array(single_trial.n_opt_paths_all)[0])
            temp_undo_c.append(np.double(np.array(single_trial.map_name)[0] == 'undo').astype(np.int16))
            temp_leftover.append(np.array(single_trial.budget_all)[-1])
            mas_all_trial = np.array(single_trial.mas_all)
            errors_trial = (mas_all_trial[1:] - mas_all_trial[:-1])
            temp_numError.append(np.sum(errors_trial<0)) # how many errors?
            temp_sumSeverityErrors.append(np.sum(np.abs(errors_trial[errors_trial<0])))
            temp_numUNDO.append(np.sum(np.array(single_trial.undo_all)))
            temp_TT.append(np.array(single_trial.time_all)[-1]/1000)

            prev_mapid = np.array(data_all[i].map_id)[ti]
            prev_mapname = data_all[i].map_name[ti]
        ti += 1
    numCities.append(temp_numCities)
    mas.append(temp_mas)
    nos.append(temp_nos)
    undo_c.append(temp_undo_c)
    leftover.append(temp_leftover)
    numError.append(temp_numError)
    sumSeverityErrors.append(temp_sumSeverityErrors)
    numUNDO.append(temp_numUNDO)
    TT.append(temp_TT)

np.savetxt(R_out_dir + 'numCities.csv', np.array(numCities).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'mas.csv', np.array(mas).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'nos.csv', np.array(nos).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'undo_c.csv', np.array(undo_c).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'leftover.csv', np.array(leftover).transpose(),fmt='%f',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'numError.csv', np.array(numError).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'sumSeverityErrors.csv', np.array(sumSeverityErrors).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'numUNDO.csv', np.array(numUNDO).astype(np.int16).transpose(),fmt='%d',delimiter=',',encoding=None)
np.savetxt(R_out_dir + 'TT.csv', np.array(TT).transpose(),fmt='%f',delimiter=',',encoding=None)
## ==========================================================================








