# -*- coding: utf-8 -*-
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
import math

import matplotlib.pyplot as plt
import seaborn as sns

import json
from anytree.importer import JsonImporter
from anytree import PreOrderIter

# +
from scipy import stats
from scipy.stats import sem
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import ttest_rel,ttest_ind

import statsmodels.api as sm
import pylab as py

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
# fitdistrplus = importr('fitdistrplus')
boot = importr('boot')
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

single_condition_data = puzzleID_order_data[puzzleID_order_data['condition']==1].copy()
single_condition_data = single_condition_data.reset_index()
sc_data_choice_level = data_choice_level[data_choice_level['condition']==1].reset_index()

with open(home_dir +'tree_data/undo_tree', 'r') as file:
    undo_tree = json.load(file)


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

# # error distribution

# ## puzzle: MAS

# +
undo_mas = puzzleID_order_data[puzzleID_order_data['condition']==0].groupby(['mas'])['sumSeverityErrors'].mean().to_frame()

print(undo_mas)

# +
# %matplotlib notebook
fig, axs = plt.subplots(1, 1)

bx = sns.barplot(x='mas', y='numError', data = puzzleID_order_data[puzzleID_order_data['condition']==0], color = '#ccd5ae') 

# -

# ## puzzle: optimal solutions

# +
error_basic = np.array(puzzleID_order_data[puzzleID_order_data['condition']==0]['numError']) 
# error_undo = np.array(puzzleID_order_data[puzzleID_order_data['condition']==1]['numError']) 

n_optimal = np.array(puzzleID_order_data[puzzleID_order_data['condition']==0]['nos'])

bins1 = stats.binned_statistic(n_optimal, error_basic, 'mean', bins=[1,3,6,9,100])
# bins2 = stats.binned_statistic(n_optimal, error_undo, 'mean', bins=[1,3,6,9,15])
# +
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
# ## choice: connected cities

# +
error_step = data_choice_level[data_choice_level['condition']==0].groupby(['currNumCities'])['error'].mean().to_frame()

print(error_step)


# +
# %matplotlib notebook
fig, axs = plt.subplots(1, 1)

bx = sns.barplot(x='currNumCities', y='severityOfErrors', data = data_choice_level[data_choice_level['condition']==0], color = '#ccd5ae') 


# +
# %matplotlib notebook
fig, axs = plt.subplots(1, 1)

bx = sns.barplot(x='currNumCities', y='error', data = data_choice_level[data_choice_level['condition']==0], color = '#ccd5ae') 

# -

# # numUndo distribution

# ## puzzle: MAS

# +
undo_mas = single_condition_data.groupby(['mas'])['numFullUndo'].mean().to_frame()

print(undo_mas)

# +
# %matplotlib notebook
fig, axs = plt.subplots(1, 1)

bx = sns.barplot(x='mas', y='numFullUndo', data = single_condition_data, color = '#ccd5ae') 
# -

# ## [TODO] puzzle: optimal solutions

# # benefit of undo

# ## benefits of undo - num of full undoing

# ### averaged within puzzle, subject, average within each number of undo category

# +
numFullUndo_bin = np.array(puzzleID_order_data[puzzleID_order_data.condition == 1]['numFullUndo'])
numFullUndo_bin[(numFullUndo_bin >= 4)] = 4

benefit_undo = (np.array(puzzleID_order_data[puzzleID_order_data['condition']==1]['numCities']) 
        - np.array(puzzleID_order_data[puzzleID_order_data['condition']==0]['numCities']))

subjectID = np.array(puzzleID_order_data[puzzleID_order_data.condition == 1]['subjects'])
puzzleID = np.array(puzzleID_order_data.loc[puzzleID_order_data.condition == 1,'puzzleID'])

dat = pd.DataFrame({'subjects':subjectID, 'puzzleID':puzzleID,
                    'numFullUndo_bin':numFullUndo_bin, 
                    'benefit_undo':benefit_undo})
# -

benefit_sub = dat.groupby(['puzzleID','subjects','numFullUndo_bin'])['benefit_undo'].mean().groupby(['subjects','numFullUndo_bin']).mean().groupby(['numFullUndo_bin'])
benefit_sub.count()

# +
# %matplotlib notebook
fig, axs = plt.subplots()

y1 = benefit_sub.mean()
c1 = benefit_sub.std()/np.sqrt(len(np.unique(dat.subjects)))

x=range(0,5)
axs.set_xticks([0,1,2,3,4])
axs.set_xticklabels([0,1,2,3,'4+'])

axs.plot(x,y1 , color = '#81b29a', linewidth=3)
axs.fill_between(x, (y1-c1), (y1+c1), color='#81b29a', alpha=.2)

axs.set_yticks(np.linspace(0,0.35,8))
axs.set_xticks([0,1,2,3,4])
axs.set_xticklabels([0,1,2,3,'4+'])
axs.set_xlabel('Number of undo sequences')
axs.set_ylabel('Benefit of undo (undo condition - basic condition)')

# -

# ### all data point, average within each number of undo category

# +
basic_score = puzzleID_order_data[puzzleID_order_data['condition']==0]['numCities'].reset_index(drop=True)
basic_score_z = basic_score/puzzleID_order_data[puzzleID_order_data['condition']==0]['mas'].reset_index(drop=True)
single_condition_data['numCities_z'] = single_condition_data['numCities']/single_condition_data['mas']

single_condition_data['undo_benefit'] = single_condition_data['numCities'] - basic_score
single_condition_data['undo_benefit_z'] = single_condition_data['numCities_z'] - basic_score_z

undo_benefit_z_sub = single_condition_data.groupby(['subjects'])['undo_benefit_z'].mean()
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


# ### all data point - scatter plot

scatter_data = single_condition_data.groupby(['undo_benefit_z','numFullUndo'])['index'].size().to_frame(name = 'count').reset_index()

# %matplotlib notebook
fig1, ax1 = plt.subplots()
sns.scatterplot(x='numFullUndo', y='undo_benefit_z', size = scatter_data['count'], sizes = (3,100), data=scatter_data) 
ax1.set_xlabel("number of undo")
ax1.set_ylabel("benefit of undo")

# ### average within subject. correlation

fig11, ax1 = plt.subplots()
ax1.plot(undo_count_sub,undo_benefit_z_sub,'o')
ax1.set_xlabel("average number of undo")
ax1.set_ylabel("benefit of undo")
fig11.savefig(out_dir + 'undobenefit_individual.pdf', dpi=600, bbox_inches='tight')

#TODO: with a caption stating that each point is a subject, the Spearman rho, and the p-value
stats.spearmanr(undo_count_sub,undo_benefit_z_sub)

# ### average within puzzle. correlation

undo_benefit_puzzle = single_condition_data.groupby(['puzzleID'])['undo_benefit'].mean()
undo_count_puzzle = single_condition_data.groupby(['puzzleID'])['numFullUndo'].mean()
fig1, ax1 = plt.subplots()
ax1.plot(undo_count_puzzle,undo_benefit_puzzle,'o')
ax1.set_xlabel("average number of undo")
ax1.set_ylabel("benefit of undo")

# ## GLMM

# ### benefit of undo

# +
basic_score = puzzleID_order_data[puzzleID_order_data['condition']==0]['numCities'].reset_index(drop=True)
basic_score_z = basic_score/puzzleID_order_data[puzzleID_order_data['condition']==0]['mas'].reset_index(drop=True)
single_condition_data['numCities_z'] = single_condition_data['numCities']/single_condition_data['mas']

single_condition_data['undo_benefit'] = single_condition_data['numCities'] - basic_score
single_condition_data['undo_benefit_z'] = single_condition_data['numCities_z'] - basic_score_z

undo_benefit_z_sub = single_condition_data.groupby(['subjects'])['undo_benefit_z'].mean()
undo_count_sub = single_condition_data.groupby(['subjects'])['numFullUndo'].mean()

# + magic_args="-i single_condition_data" language="R"
# #https://stats.stackexchange.com/questions/29781/when-conducting-multiple-regression-when-should-you-center-your-predictor-varia
# single_condition_data$subjects <- factor(single_condition_data$subjects)
# single_condition_data$puzzleID <- factor(single_condition_data$puzzleID)
# single_condition_data$mas <- factor(single_condition_data$mas)
# # single_condition_data$numFullUndo[single_condition_data$numFullUndo >4] <- 4
# # single_condition_data$numFullUndo <- factor(single_condition_data$numFullUndo)
#
# str(single_condition_data)

# + language="R"
#
# model = lmer(undo_benefit_z ~ numFullUndo + (numFullUndo|subjects),
#                                   data=single_condition_data , control=lmerControl(optimizer="optimx",
#                                                                    optCtrl=list(method="nlminb")))
#
# # get the coefficients for the best fitting model
# summary(model)

# + language="R"
# # anova(model)
# # plot(model)
#
# ranef(model)
# ## QQ-plots:
# # par(mfrow = c(1, 2))
# # qqnorm(ranef(model)$subjects[, 1], main = "Random effects of subjects")
# # qqnorm(resid(model), main = "Residuals")
#
# qqPlot(resid(model), distribution = "norm")
# -

# ### missed points

puzzleID_order_data['missed_points'] = puzzleID_order_data['mas'] - puzzleID_order_data['numCities']
puzzleID_order_data = puzzleID_order_data[puzzleID_order_data['missed_points'] >= 0]

# + magic_args="-i puzzleID_order_data" language="R"
#
# puzzleID_order_data$subjects <- factor(puzzleID_order_data$subjects)
# puzzleID_order_data$puzzleID <- factor(puzzleID_order_data$puzzleID)
# puzzleID_order_data$condition <- factor(puzzleID_order_data$condition)
# puzzleID_order_data$missed_points_sqrt <- sqrt(puzzleID_order_data$missed_points)
# str(puzzleID_order_data)

# + language="R"
#  #
# model0 = lme4::glmer(missed_points ~  numFullUndo + (1|subjects) + (1|puzzleID) ,#+ (condition + 0 | subjects)
#                                   data = puzzleID_order_data, family = "poisson")
#
# # get the coefficients for the best fitting model
# summary(model0)

# + language="R"
# # plot(model0) # https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/plot.lme.html
# ranef(model0)
# ## QQ-plots:
# # par(mfrow = c(1, 2))
# # qqnorm(ranef(model0)$subjects[, 1], main = "Random effects of subjects")
# # qqnorm(resid(model0), main = "Residuals")
#
# # qqPlot(resid(model0), distribution = "norm")
#
# # https://stats.stackexchange.com/questions/295340/what-to-do-with-glm-gamma-when-residuals-are-not-normally-distributed/302413#302413
# # model0.diag <- glm.diag(model0)
# # glm.diag.plots(model0, model0.diag)

# + language="R"
#
# model1 = lme4::glmer(missed_points ~ condition + numFullUndo + (1|subjects) + (1|puzzleID),
#                                   data = puzzleID_order_data, family = "poisson")
#
# # get the coefficients for the best fitting model
# summary(model1)

# + language="R"
#
# anova(model0,model1)
# -

# # Where people started undo and stopped


# +
# # Fixing random state for reproducibility
# np.random.seed(19680801)

# # # some random data
# # x = np.random.randn(1000)
# # y = np.random.randn(1000)

def scatter_hist(x, y, ax, ax_histx, ax_histy, alpha=1):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, alpha=alpha, color=[0,0,0])
    ax.set_xlabel('Normalized number of cities connected (undo start)')
    ax.set_ylabel('Normalized number of cities connected (undo target)')

    # now determine nice limits by hand:
    binwidth = 0.03
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color=[.7,.7,.7])
    ax_histy.hist(y, bins=bins, orientation='horizontal', color=[.7,.7,.7])


# -

# ## all undo

# +
mas_map = [sc_data_choice_level.allMAS[sc_data_choice_level["puzzleID"]==i].tolist()[0] for i in np.unique(sc_data_choice_level['puzzleID'])]
str_ct = []
end_ct = []
str_ct_mean = []
end_ct_mean = []

for sub in range(100):
    str_ct_sbj = []
    end_ct_sbj = []
    dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index"])
    
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()
        str_ct_pz = []
        end_ct_pz = []
        
        for index in range(len(dat_sbj_pzi)):# or mas_map[pzi
#             print(dat_sbj_pzi.currNumCities[index])
            # -1: so that it can be ranged from 0 to 1
            if dat_sbj_pzi.firstUndo[index] == 1:
#                 if dat_sbj_pzi.currNumCities[index]+1-1 <= 0:
#                     print(dat_sbj_pzi.currNumCities[index])
                str_ct_pz.append((dat_sbj_pzi.currNumCities[index]+1-1)/(mas_map[pzi]-1)) # or datum.currMas
                # because the dataset is counting startcity as 1

            if dat_sbj_pzi.lastUndo[index] == 1:
#                 if dat_sbj_pzi.currNumCities[index]-1 <= 0:
#                     print(dat_sbj_pzi.currNumCities[index])
                end_ct_pz.append((dat_sbj_pzi.currNumCities[index]-1)/(mas_map[pzi]-1))
      
        str_ct_sbj.extend(str_ct_pz)
        end_ct_sbj.extend(end_ct_pz)
        
    str_ct.extend(str_ct_sbj)
    end_ct.extend(end_ct_sbj)
    str_ct_mean.append(np.mean(str_ct_sbj))
    end_ct_mean.append(np.mean(end_ct_sbj))
# -


# ### avarage across puzzle for each subject

# +
end_ct_mean = np.array(end_ct_mean)
str_ct_mean = np.array(str_ct_mean)

str_ct_  = str_ct_mean[~np.isnan(str_ct_mean)]
end_ct_  = end_ct_mean[~np.isnan(end_ct_mean)]

# ind = np.argsort(str_ct_-end_ct_ )
ind = np.argsort(end_ct_ )

# +
# %matplotlib notebook
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(6, 6)) #

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)
offset_ = .05# offset for better visualization
ax.set_xlim(0,1+offset_)
ax.set_ylim(0,1+offset_)
ax_histx.set_xlim(0-offset_,1+offset_)
ax_histy.set_ylim(0-offset_,1+offset_)
# use the previously defined function
scatter_hist(str_ct_, end_ct_, ax, ax_histx, ax_histy)

# +
# only scatter plot
# %matplotlib notebook

fig, ax = plt.subplots(figsize=(6, 6))
offset_ = .05# offset for better visualization
ax.set_xlim(0-offset_,1+offset_)
ax.set_ylim(0-offset_,1+offset_)
plt.scatter(str_ct_, end_ct_,color=[0,0,0])
plt.xlabel('normalized number of cities connected (undo start)')
plt.ylabel('normalized number of cities connected (undo target)')
# -

# ### every data points

# +
end_ct = np.array(end_ct)
str_ct = np.array(str_ct)

str_ct = str_ct[~np.isnan(str_ct)]
end_ct = end_ct[~np.isnan(end_ct)]

# ind = np.argsort(str_ct - end_ct)
# ind = np.argsort(str_ct)
ind = np.argsort(end_ct)
#     plt.plot([str_ct[ind[i]], end_ct[ind[i]]], [i,i], 'k.--', linewidth=.1)

# +
# %matplotlib notebook

left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(6, 6))

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)
offset_ = .05 # offset for better visualization
ax.set_xlim(0,1+offset_)
ax.set_ylim(0,1+offset_)
ax_histx.set_xlim(0-offset_,1+offset_)
ax_histy.set_ylim(0-offset_,1+offset_)
# use the previously defined function
scatter_hist(str_ct, end_ct, ax, ax_histx, ax_histy, alpha=0.05)
# -
# ## only sequential undo

# +
mas_map = [sc_data_choice_level.allMAS[sc_data_choice_level["puzzleID"]==i].tolist()[0] for i in np.unique(sc_data_choice_level['puzzleID'])]
str_ct = []
end_ct = []
str_ct_mean = []
end_ct_mean = []

for sub in range(100):
    str_ct_sbj = []
    end_ct_sbj = []
    dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index"])
    
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()
        str_ct_pz = []
        end_ct_pz = []
        
        for index in range(len(dat_sbj_pzi)):# or mas_map[pzi
#             print(dat_sbj_pzi.currNumCities[index])
            # -1: so that it can be ranged from 0 to 1
            if (dat_sbj_pzi.firstUndo[index] == 1)&(dat_sbj_pzi.lastUndo[index] != 1):
#                 if dat_sbj_pzi.currNumCities[index]+1-1 <= 0:
#                     print(dat_sbj_pzi.currNumCities[index])
                str_ct_pz.append((dat_sbj_pzi.currNumCities[index]+1-1)/(mas_map[pzi]-1)) # or datum.currMas
                # because the dataset is counting startcity as 1

            if (dat_sbj_pzi.firstUndo[index] != 1)&(dat_sbj_pzi.lastUndo[index] == 1):
#                 if dat_sbj_pzi.currNumCities[index]-1 <= 0:
#                     print(dat_sbj_pzi.currNumCities[index])
                end_ct_pz.append((dat_sbj_pzi.currNumCities[index]-1)/(mas_map[pzi]-1))
      
        str_ct_sbj.extend(str_ct_pz)
        end_ct_sbj.extend(end_ct_pz)
        
    str_ct.extend(str_ct_sbj)
    end_ct.extend(end_ct_sbj)
    str_ct_mean.append(np.mean(str_ct_sbj))
    end_ct_mean.append(np.mean(end_ct_sbj))


# +
# %matplotlib notebook
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(6, 6))

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)
offset_ = .05 # offset for better visualization
ax.set_xlim(0,1+offset_)
ax.set_ylim(0,1+offset_)
ax_histx.set_xlim(0-offset_,1+offset_)
ax_histy.set_ylim(0-offset_,1+offset_)
# use the previously defined function
scatter_hist(str_ct, end_ct, ax, ax_histx, ax_histy, alpha=0.05)
fig.savefig(out_dir + 'undo_initial_target_dist.png', dpi=600, bbox_inches='tight')
# -
# ## only undo target == start city

# +
mas_map = [sc_data_choice_level.allMAS[sc_data_choice_level["puzzleID"]==i].tolist()[0] for i in np.unique(sc_data_choice_level['puzzleID'])]
str_ct = []
end_ct = []
str_ct_mean = []
end_ct_mean = []

for sub in range(100):
    str_ct_sbj = []
    end_ct_sbj = []
    dat_sbj  = sc_data_choice_level[(sc_data_choice_level['subjects']==sub)].sort_values(["puzzleID","index"])
    
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()
        str_ct_pz = []
        end_ct_pz = []
        
        for index in range(len(dat_sbj_pzi)):# or mas_map[pzi
#             print(dat_sbj_pzi.currNumCities[index])
            # -1: so that it can be ranged from 0 to 1
            if (dat_sbj_pzi.firstUndo[index] == 1)&(dat_sbj_pzi.lastUndo[index] != 1):
#                 if dat_sbj_pzi.currNumCities[index]+1-1 <= 0:
#                     print(dat_sbj_pzi.currNumCities[index])
                temp_index = index
    
            if (dat_sbj_pzi.firstUndo[index] != 1)&(dat_sbj_pzi.lastUndo[index] == 1)&(dat_sbj_pzi.currNumCities[index] == 1):
#                 if dat_sbj_pzi.currNumCities[index]-1 <= 0:
#                     print(dat_sbj_pzi.currNumCities[index])
                end_ct_pz.append((dat_sbj_pzi.currNumCities[index]-1)/(mas_map[pzi]-1))
                str_ct_pz.append((dat_sbj_pzi.currNumCities[temp_index]+1-1)/(mas_map[pzi]-1)) # or datum.currMas
                # because the dataset is counting startcity as 1
      
        str_ct_sbj.extend(str_ct_pz)
        end_ct_sbj.extend(end_ct_pz)
        
    str_ct.extend(str_ct_sbj)
    end_ct.extend(end_ct_sbj)
    str_ct_mean.append(np.mean(str_ct_sbj))
    end_ct_mean.append(np.mean(end_ct_sbj))


# +
# %matplotlib notebook
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(6, 6))

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)
offset_ = .05 # offset for better visualization
ax.set_xlim(0,1+offset_)
ax.set_ylim(0,1+offset_)
ax_histx.set_xlim(0-offset_,1+offset_)
ax_histy.set_ylim(0-offset_,1+offset_)
# use the previously defined function
scatter_hist(str_ct, end_ct, ax, ax_histx, ax_histy, alpha=0.05)
# fig.savefig(out_dir + 'undo_initial_target_dist.png', dpi=600, bbox_inches='tight')
# -
# ## only undo target != start city

# +
mas_map = [sc_data_choice_level.allMAS[sc_data_choice_level["puzzleID"]==i].tolist()[0] for i in np.unique(sc_data_choice_level['puzzleID'])]
str_ct = []
end_ct = []
str_ct_mean = []
end_ct_mean = []

for sub in range(100):
    str_ct_sbj = []
    end_ct_sbj = []
    dat_sbj  = sc_data_choice_level[(sc_data_choice_level['subjects']==sub)].sort_values(["puzzleID","index"])
    
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()
        str_ct_pz = []
        end_ct_pz = []
        
        for index in range(len(dat_sbj_pzi)):# or mas_map[pzi
#             print(dat_sbj_pzi.currNumCities[index])
            # -1: so that it can be ranged from 0 to 1
            if (dat_sbj_pzi.firstUndo[index] == 1)&(dat_sbj_pzi.lastUndo[index] != 1):
#                 if dat_sbj_pzi.currNumCities[index]+1-1 <= 0:
#                     print(dat_sbj_pzi.currNumCities[index])
                temp_index = index
    
            if (dat_sbj_pzi.firstUndo[index] != 1)&(dat_sbj_pzi.lastUndo[index] == 1)&(dat_sbj_pzi.currNumCities[index] != 1):
#                 if dat_sbj_pzi.currNumCities[index]-1 <= 0:
#                     print(dat_sbj_pzi.currNumCities[index])
                end_ct_pz.append((dat_sbj_pzi.currNumCities[index]-1)/(mas_map[pzi]-1))
                str_ct_pz.append((dat_sbj_pzi.currNumCities[temp_index]+1-1)/(mas_map[pzi]-1)) # or datum.currMas
                # because the dataset is counting startcity as 1
      
        str_ct_sbj.extend(str_ct_pz)
        end_ct_sbj.extend(end_ct_pz)
        
    str_ct.extend(str_ct_sbj)
    end_ct.extend(end_ct_sbj)
    str_ct_mean.append(np.mean(str_ct_sbj))
    end_ct_mean.append(np.mean(end_ct_sbj))


# +
# %matplotlib notebook
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(6, 6))

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)
offset_ = .05 # offset for better visualization
ax.set_xlim(0,1+offset_)
ax.set_ylim(0,1+offset_)
ax_histx.set_xlim(0-offset_,1+offset_)
ax_histy.set_ylim(0-offset_,1+offset_)
# use the previously defined function
scatter_hist(str_ct, end_ct, ax, ax_histx, ax_histy, alpha=0.05)
# fig.savefig(out_dir + 'undo_initial_target_dist.png', dpi=600, bbox_inches='tight')
# -
# # Undo initial state

# ## budget (before submit/undo at the end of trial)

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

# ## error

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
# fig.savefig(out_dir + 'undo_instant_error.pdf', dpi=600, bbox_inches='tight')
# -


# ### different errors lead to different types of undo? - histogram

# +
# groupby_error_single = single_severity_error.value_counts()
# groupby_error_series = series_severity_error.value_counts()

# groupby_error_series.at[4] = 0
# groupby_error_series.at[6] = 0

# +
index_single_undo =  data_choice_level.index[(data_choice_level['firstUndo'] == 1)&(data_choice_level['lastUndo'] == 1)]
df_beforeSingleUndo = data_choice_level.loc[index_single_undo-1,:]

single_severity_error = df_beforeSingleUndo['severityOfErrors']
df_beforeSingleUndo['accumulatedError'] = df_beforeSingleUndo['allMAS'] - df_beforeSingleUndo['currMas']
groupby_error_single = df_beforeSingleUndo.groupby(['severityOfErrors','accumulatedError'])['subjects'].count()
groupby_error_single = groupby_error_single.reset_index()

groupby_error_single.loc[groupby_error_single['accumulatedError']>groupby_error_single['severityOfErrors'],'accumulatedError']= 'noninstant'
groupby_error_single = groupby_error_single.groupby(['severityOfErrors','accumulatedError']).sum()
groupby_error_single = groupby_error_single.reset_index()

groupby_error_single['undoType'] = 'single'



index_series_undo =  data_choice_level.index[(data_choice_level['firstUndo'] == 1)&(data_choice_level['lastUndo'] != 1)]
df_beforeSeriesUndo = data_choice_level.loc[index_series_undo-1,:]

series_severity_error = df_beforeSeriesUndo['severityOfErrors']
df_beforeSeriesUndo['accumulatedError'] = df_beforeSeriesUndo['allMAS'] - df_beforeSeriesUndo['currMas']
groupby_error_series = df_beforeSeriesUndo.groupby(['severityOfErrors','accumulatedError'])['subjects'].count()
groupby_error_series = groupby_error_series.reset_index()

# here the "noinstant" means there is some already exsistent error before the current error
groupby_error_series.loc[groupby_error_series['accumulatedError']>groupby_error_series['severityOfErrors'],'accumulatedError']= 'noninstant'
groupby_error_series = groupby_error_series.groupby(['severityOfErrors','accumulatedError']).sum()
groupby_error_series = groupby_error_series.reset_index()

groupby_error_series['undoType'] = 'serial'

groupby_error = pd.concat([groupby_error_single,groupby_error_series])

# add empty entries
d = {'severityOfErrors': [4,4,4,6,6,6], 'accumulatedError': [4,'noninstant','noninstant', 6,'noninstant','noninstant'],'subjects': [0,0,0,0,0,0],'undoType': ['serial','serial','single', 'serial','serial','single',]}
df = pd.DataFrame(data=d)
groupby_error = pd.concat([groupby_error,df]).reset_index()
print(groupby_error)


# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
single_index1 = (groupby_error['accumulatedError']!='noninstant')&(groupby_error['undoType']=='single')
single_index2 = (groupby_error['accumulatedError']=='noninstant')&(groupby_error['undoType']=='single')
single_proportion = groupby_error.loc[single_index1 ,'subjects']/sum(groupby_error['subjects'])
single_proportion2 = groupby_error.loc[single_index2 ,'subjects']/sum(groupby_error['subjects'])

serial_index1 = (groupby_error['accumulatedError']!='noninstant')&(groupby_error['undoType']=='serial')
serial_index2 = (groupby_error['accumulatedError']=='noninstant')&(groupby_error['undoType']=='serial')

x = groupby_error.loc[single_index1 ,'severityOfErrors']
width = 0.35  # the width of the bars

axs.bar(x - width/2, single_proportion, width, label='single undo-only instant error',hatch='//',color='#d4a373',edgecolor='black')
axs.bar(x + width/2, single_proportion2, width, label='single undo-with accumulated error',hatch='//',color='#ccd5ae',edgecolor='black')
axs.bar(x - width/2, groupby_error.loc[serial_index1 ,'subjects']/sum(groupby_error['subjects']),width, bottom=single_proportion,
       label='serial undo-only instant error',color='#d4a373',edgecolor='black')
axs.bar(x + width/2, groupby_error.loc[serial_index2 ,'subjects']/sum(groupby_error['subjects']),width, bottom=single_proportion2,
       label='serial undo-with accumulated error',color='#ccd5ae',edgecolor='black')

axs.set_ylabel('proportion of first undo')
axs.set_xlabel('instant error before undo action')
axs.legend()
plt.show()
fig.savefig(out_dir + 'undotype_errortype.pdf', dpi=600, bbox_inches='tight')


# -
# ### different errors lead to different types of undo? - barplot

# +
data_choice_level['accumulatedError'] = data_choice_level['allMAS'] - data_choice_level['currMas']

singleundo = (data_choice_level.firstUndo == 1)&(data_choice_level.lastUndo == 1)
df_before = data_choice_level.loc[data_choice_level.loc[singleundo,:].index-1,:]
noinstant_singleundo = (df_before.accumulatedError > df_before.severityOfErrors)
instant_singleundo = (df_before.accumulatedError <= df_before.severityOfErrors)

sequentialundo = (data_choice_level.firstUndo == 1)&(data_choice_level.lastUndo != 1)
df_before = data_choice_level.loc[data_choice_level.loc[sequentialundo,:].index-1,:]
noinstant_sequentialundo = (df_before.accumulatedError > df_before.severityOfErrors)
instant_sequentialundo = (df_before.accumulatedError <= df_before.severityOfErrors)

allundo = (data_choice_level.firstUndo == 1)

instant_singleundo_freq = sum(instant_singleundo)/sum(allundo)
instant_sequentialundo_freq = sum(instant_sequentialundo)/sum(allundo)

noinstant_singleundo_freq = sum(noinstant_singleundo)/sum(allundo)
noinstant_sequentialundo_freq = sum(noinstant_sequentialundo)/sum(allundo)

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

width = 1  # the width of the bars

axs.bar(1 - width/2, instant_singleundo_freq, width, 
        label='instant_singleundo_freq',hatch='//',color='#d4a373',edgecolor='black')
axs.bar(1 - width/2, instant_sequentialundo_freq, width, bottom=instant_singleundo_freq,
       label='instant_sequentialundo_freq',color='#d4a373',edgecolor='black')

axs.bar(1.5 + width/2, noinstant_singleundo_freq, width, hatch='//',
       label='noinstant_singleundo_freq',color='#ccd5ae',edgecolor='black')
axs.bar(1.5 + width/2, noinstant_sequentialundo_freq, width,bottom=noinstant_singleundo_freq,
       label='noinstant_sequentialundo_freq',color='#ccd5ae',edgecolor='black')

axs.set_ylabel('proportion of undo')
axs.legend()
plt.show()
# fig.savefig(out_dir + 'undotype_errortype.pdf', dpi=600, bbox_inches='tight')


# -
# ### probability of undo -- conditional on error

data_choice_level_df = sc_data_choice_level[((sc_data_choice_level['undo'] == 0)|(sc_data_choice_level['firstUndo'] == 1)|(sc_data_choice_level['lastUndo'] == 1))]#
data_choice_level_df = data_choice_level_df.reset_index(drop=True)


# #### accumulated error

# +
# FROM EACH SUBJECT
dat_subjects = []
for i in np.unique(np.array(data_choice_level_df['subjects'])):
    
#     temp_data = []
    undo_list_no = []
    undo_list_yes = []
    
    for j in np.unique(sc_data_choice_level['puzzleID']):
        
#         temp_data_puz =[]
        dat_sbj_pzi =  data_choice_level_df[(data_choice_level_df.subjects == i)&(data_choice_level_df.puzzleID == j)].reset_index()
        puzzle_error = dat_sbj_pzi['allMAS'] - dat_sbj_pzi['currMas']
        # no error --------------------------------------------------
        index_error = puzzle_error.index[puzzle_error == 0]
        index_error = np.array(index_error)
        index_error += 1
        if np.any(index_error>(dat_sbj_pzi.shape[0]-1)):
            index_error = np.delete(index_error, np.where(index_error>(dat_sbj_pzi.shape[0]-1)))
        
#         undo_list_no = []
        for ind in index_error:    
            if (dat_sbj_pzi.firstUndo[ind] == 0)&(dat_sbj_pzi.lastUndo[ind] == 0)&(dat_sbj_pzi.currNumCities[ind-1] > 1):
                undo_list_no.append(0)
            elif (dat_sbj_pzi['firstUndo'][ind] == 1):
                undo_list_no.append(1)

#         temp_data_puz.append(np.mean(undo_list_no))
        
        # YES error --------------------------------------------------
        index_error = puzzle_error.index[puzzle_error != 0]
        index_error = np.array(index_error)
        index_error += 1
        if np.any(index_error>(dat_sbj_pzi.shape[0]-1)):
            index_error = np.delete(index_error, np.where(index_error>(dat_sbj_pzi.shape[0]-1)))
        
#         undo_list_yes = []
        for ind in index_error:    
            if (dat_sbj_pzi.firstUndo[ind] == 0)&(dat_sbj_pzi.lastUndo[ind] == 0)&(dat_sbj_pzi.currNumCities[ind-1] > 1):
                undo_list_yes.append(0)
            elif (dat_sbj_pzi['firstUndo'][ind] == 1):
                undo_list_yes.append(1)

#         temp_data_puz.append(np.mean(undo_list_yes))
    
#         temp_data.append(temp_data_puz)
    
#     dat_subjects.append(np.nanmean(temp_data,axis=0))
    dat_subjects.append([np.nanmean(undo_list_no),np.nanmean(undo_list_yes)])

dat_subjects = np.array(dat_subjects)

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

# axs.bar([1,2],
#         np.mean(dat_subjects,axis = 0),
#         color=[.7,.7,.7], 
#         edgecolor = 'k', 
#         yerr=np.std(dat_subjects,axis = 0)/np.sqrt(dat_subjects.shape[0]))

bx = axs.boxplot(
    [
        dat_subjects[:,0],
        dat_subjects[:,1]
    ],
    positions =[1,2],
    widths = 0.3,
    showfliers=False,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,2])
axs.set_yticks(np.linspace(0,0.4,6))
axs.set_xticklabels(labels = ['No error', 'Error'])#,fontsize=18

axs.set_xlabel('accumulated error')
axs.set_ylabel('probability of undo')

stat1, p1 = ttest_ind(dat_subjects[:,0], dat_subjects[:,1])
axs.set_title(r"$p = {0:s}$".format(as_si(p1,1)))

# fig.set_figheight(4)
# fig.set_figwidth(2)
plt.show()
fig.savefig(out_dir + 'conditional_undo_masError.pdf', dpi=600, bbox_inches='tight')




# +
# The paired t–test assumes that the differences between pairs are normally distributed
# ref: http://www.biostathandbook.com/pairedttest.html

sm.qqplot(stats.zscore(dat_subjects[:,0]-dat_subjects[:,1]), line ='45')
py.show()
# -

# #### instant error

# +
# FROM EACH SUBJECT
dat_subjects = []
for i in np.unique(np.array(data_choice_level_df['subjects'])):
    
#     temp_data = []
    undo_list_no = []
    undo_list_yes = []
    
    for j in np.unique(sc_data_choice_level['puzzleID']):
        
#         temp_data_puz =[]
        dat_sbj_pzi =  data_choice_level_df[(data_choice_level_df.subjects == i)&(data_choice_level_df.puzzleID == j)].reset_index()

        # no error --------------------------------------------------
        index_error = dat_sbj_pzi['severityOfErrors'].index[(dat_sbj_pzi['severityOfErrors'] == 0)&(dat_sbj_pzi['submit'] != 1)]
        index_error = np.array(index_error)
        index_error += 1
        if np.any(index_error>(dat_sbj_pzi.shape[0]-1)):
            index_error = np.delete(index_error, np.where(index_error>(dat_sbj_pzi.shape[0]-1)))
        
#         undo_list_no = []
        for ind in index_error:    
            if (dat_sbj_pzi.firstUndo[ind] == 0)&(dat_sbj_pzi.lastUndo[ind] == 0)&(dat_sbj_pzi.currNumCities[ind-1] > 1):
                undo_list_no.append(0)
            elif (dat_sbj_pzi['firstUndo'][ind] == 1):
                undo_list_no.append(1)

#         temp_data_puz.append(np.mean(undo_list_no))
        
        # YES error --------------------------------------------------
        index_error = dat_sbj_pzi['severityOfErrors'].index[(dat_sbj_pzi['severityOfErrors'] != 0)&(dat_sbj_pzi['submit'] != 1)]
        index_error = np.array(index_error)
        index_error += 1
        if np.any(index_error>(dat_sbj_pzi.shape[0]-1)):
            index_error = np.delete(index_error, np.where(index_error>(dat_sbj_pzi.shape[0]-1)))
        
#         undo_list_yes = []
        for ind in index_error:    
            if (dat_sbj_pzi.firstUndo[ind] == 0)&(dat_sbj_pzi.lastUndo[ind] == 0)&(dat_sbj_pzi.currNumCities[ind-1] > 1):
                undo_list_yes.append(0)
            elif (dat_sbj_pzi['firstUndo'][ind] == 1):
                undo_list_yes.append(1)

#         temp_data_puz.append(np.mean(undo_list_yes))
    
#         temp_data.append(temp_data_puz)
    
#     dat_subjects.append(np.nanmean(temp_data,axis=0))
    dat_subjects.append([np.nanmean(undo_list_no),np.nanmean(undo_list_yes)])

dat_subjects = np.array(dat_subjects)

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
# axs.bar([1,2],
#         np.mean(dat_subjects,axis = 0),
#         color=[.7,.7,.7], 
#         edgecolor = 'k', 
#         yerr=np.std(dat_subjects,axis = 0)/np.sqrt(dat_subjects.shape[0]))

bx = axs.boxplot(
    [
        dat_subjects[:,0],
        dat_subjects[:,1]
    ],
    positions =[1,2],
    widths = 0.3,
    showfliers=False,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,2])
axs.set_yticks(np.linspace(0,0.2,6))
axs.set_xticklabels(labels = ['no error', 'error'])#,fontsize=18

axs.set_xlabel('Instant error')
axs.set_ylabel('Probability of undo')



stat1, p1 = ttest_ind(dat_subjects[:,0], dat_subjects[:,1])
print(stat1)
print(p1)
axs.set_title(r"$p = {0:s}$".format(as_si(p1,1)))

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
fig.savefig(out_dir + 'undo_prob_instant_error.png', dpi=600, bbox_inches='tight')



# +
# The paired t–test assumes that the differences between pairs are normally distributed
# ref: http://www.biostathandbook.com/pairedttest.html

sm.qqplot(stats.zscore(dat_subjects[:,0]-dat_subjects[:,1]), line ='45')
py.show()
# -

# ### probability of undo -- conditional on error - undo target

# +
# FROM EACH SUBJECT
data_choice_level_df_2 = sc_data_choice_level[((sc_data_choice_level['undo'] == 0)|(sc_data_choice_level['firstUndo'] == 1)|(sc_data_choice_level['lastUndo'] == 1))]#
data_choice_level_df_2 = data_choice_level_df_2.reset_index(drop=True)

print(len(sc_data_choice_level))
print(len(data_choice_level_df_2))

# -


# #### accumulated error

# ##### averaged within subject

# +
dat_subjects = []
for i in np.unique(np.array(data_choice_level_df_2['subjects'])):

    no_undo = 0
    undo_start = 0
    undo_nostart = 0
    
    index_subjects =  data_choice_level_df_2.index[data_choice_level_df_2['subjects'] == i]
    
    puzzle_error = data_choice_level_df_2['allMAS'] - data_choice_level_df_2['currMas']
    
    # YES error -----------------------------------------------------------
    index_error = puzzle_error.index[puzzle_error != 0]
    index_error = np.array(index_error)
    index_error = np.intersect1d(index_error, index_subjects)
    index_error += 1
    if np.any(index_error>(data_choice_level_df_2.shape[0]-1)):
        index_error = np.delete(index_error, np.where(index_error>(data_choice_level_df_2.shape[0]-1)))
   
    n_move = len(index_error)
    for ind in index_error:
        run = 1
        if data_choice_level_df_2['firstUndo'][ind] == 0:
            no_undo = no_undo + 1
        elif (data_choice_level_df_2['lastUndo'][ind] == 1)&(data_choice_level_df_2['firstUndo'][ind] == 1):
            n_move = n_move - 1
        elif (data_choice_level_df_2['lastUndo'][ind] != 1)&(data_choice_level_df_2['firstUndo'][ind] == 1):
            while run == 1:
                ind = ind + 1
                if (data_choice_level_df_2['lastUndo'][ind] == 1)&(data_choice_level_df_2['currNumCities'][ind] == 1):
                    undo_start = undo_start + 1
                    run = 0
                elif (data_choice_level_df_2['lastUndo'][ind] == 1)&(data_choice_level_df_2['currNumCities'][ind] > 1):
                    undo_nostart = undo_nostart + 1
                    run = 0
        
    
    if n_move == 0:
        print(i)
    else:
        temp_data = [no_undo/n_move,undo_start/n_move,undo_nostart/n_move]
    
    dat_subjects.append(temp_data)

dat_subjects = np.array(dat_subjects)
print(np.mean(dat_subjects,axis=0))

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
# axs.bar([1,2],
#         np.mean(dat_subjects,axis = 0),
#         color=[.7,.7,.7], 
#         edgecolor = 'k', 
#         yerr=np.std(dat_subjects,axis = 0)/np.sqrt(dat_subjects.shape[0]))

bx = axs.boxplot(
    [
        dat_subjects[:,1],
        dat_subjects[:,2]
    ],
    positions =[1,2],
    widths = 0.3,
    showfliers=False,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,2])
axs.set_yticks(np.linspace(0,0.12,5))
axs.set_xticklabels(labels = ['start city', 'non-start city'])#,fontsize=18

axs.set_xlabel('Accumulated error')
axs.set_ylabel('Probability of undo')



stat1, p1 = ttest_ind(dat_subjects[:,1], dat_subjects[:,2])
print(stat1)
print(p1)
axs.set_title(r"$p = {0:s}$".format(as_si(p1,1)))

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
# fig.savefig(out_dir + 'undo_prob_accumulated_error_target.png', dpi=600, bbox_inches='tight')

# -


# ##### all data point

# +
list_start = []
list_nostart = []

for i in np.unique(np.array(data_choice_level_df_2['subjects'])):

    no_undo = 0
    undo_start = 0
    undo_nostart = 0
    
    index_subjects =  data_choice_level_df_2.index[data_choice_level_df_2['subjects'] == i]
    
    puzzle_error = data_choice_level_df_2['allMAS'] - data_choice_level_df_2['currMas']
    
#     # no error -----------------------------------------------------------
#     index_error = puzzle_error.index[puzzle_error == 0]
#     index_error = np.array(index_error)
#     index_error = np.intersect1d(index_error, index_subjects)

#     index_error += 1 # the next index
    
#     if np.any(index_error>(data_choice_level_df.shape[0]-1)):
#         index_error = np.delete(index_error, np.where(index_error>(data_choice_level_df.shape[0]-1)))

# #     temp_data.append(np.mean(data_choice_level['undo'][index_error]))
#     temp_data.append(np.mean(data_choice_level_df['firstUndo'][index_error]))

    # YES error -----------------------------------------------------------
    index_error = puzzle_error.index[puzzle_error != 0]
    index_error = np.array(index_error)
    index_error = np.intersect1d(index_error, index_subjects)
    index_error += 1
    if np.any(index_error>(data_choice_level_df_2.shape[0]-1)):
        index_error = np.delete(index_error, np.where(index_error>(data_choice_level_df_2.shape[0]-1)))
   

    for ind in index_error:
        run = 1
        if data_choice_level_df_2['firstUndo'][ind] == 0:
            list_start.append(0)
            list_nostart.append(0)
#         elif (data_choice_level_df_2['firstUndo'][ind] == data_choice_level_df_2['lastUndo'][ind])&(data_choice_level_df_2['firstUndo'][ind] == 1):
#             n_move = n_move - 1
        elif (data_choice_level_df_2['firstUndo'][ind] != data_choice_level_df_2['lastUndo'][ind])&(data_choice_level_df_2['firstUndo'][ind] == 1):
            while run == 1:
                ind = ind + 1
                if (data_choice_level_df_2['lastUndo'][ind] == 1)&(data_choice_level_df_2['currNumCities'][ind] == 1):
#                     undo_start = undo_start + 1
                    list_start.append(1)
                    list_nostart.append(0)
                    run = 0
                elif (data_choice_level_df_2['lastUndo'][ind] == 1)&(data_choice_level_df_2['currNumCities'][ind] > 1):
                    list_start.append(0)
                    list_nostart.append(1)
                    run = 0
    
#     if n_move == 0:
#         print(i)
#     else:
#         temp_data = [no_undo/n_move,undo_start/n_move,undo_nostart/n_move]
    
#     dat_subjects.append(temp_data)

count_start = sum(list_start)
count_nostart = sum(list_nostart)
print([count_start/len(list_start),count_nostart/len(list_nostart)])

# -

from statsmodels.stats.proportion import proportions_ztest
stat, pval = proportions_ztest([count_start,count_nostart], [len(list_start),len(list_nostart)])
print('{0:0.3f}'.format(pval))

# #### no accumulated error

# +
dat_subjects = []
for i in np.unique(np.array(data_choice_level_df_2['subjects'])):
    no_undo = 0
    undo_start = 0
    undo_nostart = 0
    
    index_subjects =  data_choice_level_df_2.index[data_choice_level_df_2['subjects'] == i]
    
    puzzle_error = data_choice_level_df_2['allMAS'] - data_choice_level_df_2['currMas']
    
    # no error -----------------------------------------------------------
    index_error = puzzle_error.index[puzzle_error == 0]
    index_error = np.array(index_error)
    index_error = np.intersect1d(index_error, index_subjects)

    index_error += 1 # the next index
    
    if np.any(index_error>(data_choice_level_df_2.shape[0]-1)):
        index_error = np.delete(index_error, np.where(index_error>(data_choice_level_df_2.shape[0]-1)))

    n_move = len(index_error) # number of all moves
    for ind in index_error:
        run = 1
        if data_choice_level_df_2['firstUndo'][ind] == 0:
            no_undo = no_undo + 1
        elif (data_choice_level_df_2['firstUndo'][ind] == data_choice_level_df_2['lastUndo'][ind])&(data_choice_level_df_2['firstUndo'][ind] == 1):
            n_move = n_move - 1
        elif (data_choice_level_df_2['firstUndo'][ind] != data_choice_level_df_2['lastUndo'][ind])&(data_choice_level_df_2['firstUndo'][ind] == 1):
            while run == 1:
                ind = ind + 1
                if (data_choice_level_df_2['lastUndo'][ind] == 1)&(data_choice_level_df_2['currNumCities'][ind] == 1):
                    undo_start = undo_start + 1
                    run = 0
                elif (data_choice_level_df_2['lastUndo'][ind] == 1)&(data_choice_level_df_2['currNumCities'][ind] > 1):
                    undo_nostart = undo_nostart + 1
                    run = 0

#     # YES error -----------------------------------------------------------
#     index_error = puzzle_error.index[puzzle_error != 0]
#     index_error = np.array(index_error)
#     index_error = np.intersect1d(index_error, index_subjects)
#     index_error += 1
#     if np.any(index_error>(data_choice_level_df_2.shape[0]-1)):
#         index_error = np.delete(index_error, np.where(index_error>(data_choice_level_df_2.shape[0]-1)))
   
#     n_move = len(index_error) # number of all moves
#     for ind in index_error:
#         if data_choice_level_df_2['firstUndo'][ind] == 0:
#             no_undo = no_undo + 1
#         elif (data_choice_level_df_2['firstUndo'][ind] == data_choice_level_df_2['lastUndo'][ind])&(data_choice_level_df_2['firstUndo'][ind] == 1):
#             n_move = n_move - 1
#         elif (data_choice_level_df_2['firstUndo'][ind] != data_choice_level_df_2['lastUndo'][ind])&(data_choice_level_df_2['firstUndo'][ind] == 1):
#             while run == 1:
#                 ind = ind + 1
#                 if (data_choice_level_df_2['lastUndo'][ind] == 1)&(data_choice_level_df_2['currNumCities'][ind] == 1):
#                     undo_start = undo_start + 1
#                     run = 0
#                 elif (data_choice_level_df_2['lastUndo'][ind] == 1)&(data_choice_level_df_2['currNumCities'][ind] > 1):
#                     undo_nostart = undo_nostart + 1
#                     run = 0
    
    if n_move == 0:
        print(i)
    else:
        temp_data = [no_undo/n_move,undo_start/n_move,undo_nostart/n_move]
    
    dat_subjects.append(temp_data)

dat_subjects = np.array(dat_subjects)
# print(np.mean(dat_subjects,axis=0))

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
# axs.bar([1,2],
#         np.mean(dat_subjects,axis = 0),
#         color=[.7,.7,.7], 
#         edgecolor = 'k', 
#         yerr=np.std(dat_subjects,axis = 0)/np.sqrt(dat_subjects.shape[0]))

bx = axs.boxplot(
    [
        dat_subjects[:,1],
        dat_subjects[:,2]
    ],
    positions =[1,2],
    widths = 0.3,
    showfliers=False,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,2])
axs.set_yticks(np.linspace(0,0.024,7))
axs.set_xticklabels(labels = ['start city', 'non-start city'])#,fontsize=18

axs.set_xlabel('No accumulated error')
axs.set_ylabel('Probability of undo')



stat1, p1 = ttest_ind(dat_subjects[:,1], dat_subjects[:,2])
print(stat1)
print(p1)
axs.set_title(r"$p = {0:s}$".format(as_si(p1,1)))

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
fig.savefig(out_dir + 'undo_prob_no_accumulated_error_target.png', dpi=600, bbox_inches='tight')

# -


# #### instant error

# ##### averaged within subject

# +
# FROM EACH SUBJECT
dat_subjects = []
for i in np.unique(np.array(data_choice_level_df_2['subjects'])):
    
    temp_data = []
    for j in np.unique(sc_data_choice_level['puzzleID']):
        
        no_undo = 0
        undo_start = 0
        undo_nostart = 0
    
        index_sub_puz =  data_choice_level_df_2.index[(data_choice_level_df_2.subjects == i)&(data_choice_level_df_2.puzzleID == j)]

        # YES error --------------------------------------------------
        index_error = data_choice_level_df_2['severityOfErrors'].index[data_choice_level_df_2['severityOfErrors'] != 0]
        index_error = np.array(index_error)
        index_error = np.intersect1d(index_error, index_sub_puz)
        index_error += 1
        if np.any(index_error>(data_choice_level_df_2.shape[0]-1)):
            index_error = np.delete(index_error, np.where(index_error>(data_choice_level_df_2.shape[0]-1)))

        n_move = len(index_error) # number of all moves
        for ind in index_error:
            run = 1
            if data_choice_level_df_2['firstUndo'][ind] == 0:
                no_undo = no_undo + 1
            elif (data_choice_level_df_2['firstUndo'][ind] == data_choice_level_df_2['lastUndo'][ind])&(data_choice_level_df_2['firstUndo'][ind] == 1):
                n_move = n_move - 1
            elif (data_choice_level_df_2['firstUndo'][ind] != data_choice_level_df_2['lastUndo'][ind])&(data_choice_level_df_2['firstUndo'][ind] == 1):
                while run == 1:
                    ind = ind + 1
                    if (data_choice_level_df_2['lastUndo'][ind] == 1)&(data_choice_level_df_2['currNumCities'][ind] == 1):
                        undo_start = undo_start + 1
                        run = 0
                    elif (data_choice_level_df_2['lastUndo'][ind] == 1)&(data_choice_level_df_2['currNumCities'][ind] > 1):
                        undo_nostart = undo_nostart + 1
                        run = 0

        if n_move == 0:
            continue
        else:
            temp_data_puz = [no_undo/n_move,undo_start/n_move,undo_nostart/n_move]
        temp_data.append(temp_data_puz)
        
    dat_subjects.append(np.nanmean(temp_data,axis=0))

dat_subjects = np.array(dat_subjects)
# -

dat_subjects

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
# axs.bar([1,2],
#         np.mean(dat_subjects,axis = 0),
#         color=[.7,.7,.7], 
#         edgecolor = 'k', 
#         yerr=np.std(dat_subjects,axis = 0)/np.sqrt(dat_subjects.shape[0]))

bx = axs.boxplot(
    [
        dat_subjects[:,1],
        dat_subjects[:,2]
    ],
    positions =[1,2],
    widths = 0.3,
    showfliers=False,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,2])
axs.set_yticks(np.linspace(0,0.08,5))
axs.set_xticklabels(labels = ['start city', 'non-start city'])#,fontsize=18

axs.set_xlabel('Instant error')
axs.set_ylabel('Probability of undo')



stat1, p1 = ttest_ind(dat_subjects[:,1], dat_subjects[:,2])
print(stat1)
print(p1)
axs.set_title(r"$p = {0:s}$".format(as_si(p1,1)))

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
# fig.savefig(out_dir + 'undo_prob_instant_error_target.png', dpi=600, bbox_inches='tight')

# -


# ##### all data

# +
# FROM EACH SUBJECT
list_start = []
list_nostart = []
for i in np.unique(np.array(data_choice_level_df_2['subjects'])):

    for j in np.unique(data_choice_level_df_2['puzzleID']):

        dat_sbj_pzi =  data_choice_level_df_2[(data_choice_level_df_2.subjects == i)&(data_choice_level_df_2.puzzleID == j)].reset_index()

        # NO error --------------------------------------------------
        index_error = dat_sbj_pzi['severityOfErrors'].index[(dat_sbj_pzi['severityOfErrors'] != 0)&(dat_sbj_pzi['submit'] != 1)]
        index_error = np.array(index_error)
        index_error += 1
        if np.any(index_error>(dat_sbj_pzi.shape[0]-1)):
            index_error = np.delete(index_error, np.where(index_error>(dat_sbj_pzi.shape[0]-1)))

        for ind in index_error:    
            if (dat_sbj_pzi.firstUndo[ind] == 0)&(dat_sbj_pzi.lastUndo[ind] == 0)&(dat_sbj_pzi.currNumCities[ind-1] > 1):
                list_start.append(0)
                list_nostart.append(0)
            elif (dat_sbj_pzi['firstUndo'][ind] == 1)&(dat_sbj_pzi['lastUndo'][ind] != 1):
                while dat_sbj_pzi.lastUndo[ind] != 1:
                    ind = ind + 1
                if (dat_sbj_pzi['currNumCities'][ind] == 1):
                    list_start.append(1)
                    list_nostart.append(0)
                elif (dat_sbj_pzi['currNumCities'][ind] > 1):
                    list_start.append(0)
                    list_nostart.append(1)
    
count_start = sum(list_start)
count_nostart = sum(list_nostart)
print([count_start/len(list_start),count_nostart/len(list_nostart)])


# -

stat, pval = proportions_ztest([count_start,count_nostart], [len(list_start),len(list_nostart)])
print('{0:0.3f}'.format(pval))

print(len(list_start))
print(len(list_nostart))

# #### no instant error

# ##### all data - dk

# +
mas_map = [sc_data_choice_level.allMAS[sc_data_choice_level["puzzleID"]==i].tolist()[0] for i in np.unique(sc_data_choice_level['puzzleID'])]

no_error_mat = []
error_mat = []

# no_error_ct_sbj = []
# error_ct_sbj = []

for sub in np.unique(np.array(sc_data_choice_level['subjects'])):
    
#     dat_sbj  = data_choice_level_df_3[data_choice_level_df_3['subjects']==sub].sort_values(["puzzleID","index"])
        
    error_ct_pz = [] # to start 1 # to non-start 2
    no_error_ct_pz = [] # to start 1 # to non-start 2
    
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        
        dat_sbj_pzi = sc_data_choice_level[(sc_data_choice_level['puzzleID'] == pzi)&(sc_data_choice_level['subjects']==sub)].reset_index()
        
        index = 1 
        while index < len(dat_sbj_pzi):
#             print('*'*40)
#             print( len(dat_sbj_pzi))
#             print(index)
            
            if (dat_sbj_pzi.firstUndo[index] == 1) and (dat_sbj_pzi.lastUndo[index] == 1): # for non-single undo only (doesnt chjange the result)
                index += 1
                continue
                
            # ==================== yes undo
            if dat_sbj_pzi.firstUndo[index] == 1: 
                index_temp = index
                
                while dat_sbj_pzi.lastUndo[index] != 1:
                    index +=1 # exclude laterUndo automatically
                    
                pend = np.double((dat_sbj_pzi.currNumCities[index]-1)/(mas_map[pzi]-1)!=0)    
                if dat_sbj_pzi.severityOfErrors[index_temp-1]==0: # if no error
                    no_error_ct_pz.append(pend+1)# to start 1 # to non-start 2
                else: # if error
                    error_ct_pz.append(pend+1)# to start 1 # to non-start 2
                    
            # ==================== no undo
#             else:
            elif (dat_sbj_pzi.firstUndo[index] == 0)&(dat_sbj_pzi.currNumCities[index-1] > 1): 
                pend = 0 
                if dat_sbj_pzi.severityOfErrors[index-1]==0:
                    no_error_ct_pz.append(pend)# did not undo 0 # to start 1 # to non-start 2
                else:
                    error_ct_pz.append(pend)# did not undo 0 # to start 1 # to non-start 2
            index += 1
                
#     no_error_ct_sbj.append([(np.array(no_error_ct_pz)==1).mean(), (np.array(no_error_ct_pz)==2).mean()])
#     error_ct_sbj.append([(np.array(error_ct_pz)==1).mean(), (np.array(error_ct_pz)==2).mean()])
        
    no_error_mat.extend(no_error_ct_pz)
    error_mat.extend(error_ct_pz)
    
#     no_error_mat.extend(no_error_ct_sbj)
#     error_mat.extend(no_error_ct_sbj)

# -

print((np.array(error_mat)==1).mean())
print((np.array(error_mat)==2).mean())
print((np.array(no_error_mat)==1).mean())
print((np.array(no_error_mat)==2).mean())

print(len(np.array(error_mat)==1))
print(len(np.array(error_mat)==2))
print(len(np.array(no_error_mat)==2))
print(len(np.array(no_error_mat)))
print(skip_count)

# +
dat1 = np.double(np.array(no_error_mat)==1)
dat2 = np.double(np.array(no_error_mat)==2)
n1 = dat1.shape[0]
p1 = np.sum(dat1)/n1
n2 = dat2.shape[0]
p2 = np.sum(dat2)/n2

p = (n1*p1 + n2*p2) / (n1+n2)

z = (p1-p2)/np.sqrt(p*(1-p) *(1/n1+1/n2))

stats.norm.sf(abs(z))*2

# +
dat1 = np.double(np.array(error_mat)==1)
dat2 = np.double(np.array(error_mat)==2)
n1 = dat1.shape[0]
p1 = np.sum(dat1)/n1
n2 = dat2.shape[0]
p2 = np.sum(dat2)/n2

p = (n1*p1 + n2*p2) / (n1+n2)

z = (p1-p2)/np.sqrt(p*(1-p) *(1/n1+1/n2))

stats.norm.sf(abs(z))*2
# -

# ##### all data - dq

# +
# FROM EACH SUBJECT
list_start = []
list_nostart = []
for i in np.unique(np.array(data_choice_level_df_2['subjects'])):

    for j in np.unique(data_choice_level_df_2['puzzleID']):

        dat_sbj_pzi =  data_choice_level_df_2[(data_choice_level_df_2.subjects == i)&(data_choice_level_df_2.puzzleID == j)].reset_index()

        # NO error --------------------------------------------------
        index_error = dat_sbj_pzi['severityOfErrors'].index[(dat_sbj_pzi['severityOfErrors'] == 0)&(dat_sbj_pzi['submit'] != 1)]
        index_error = np.array(index_error)
        index_error += 1
        if np.any(index_error>(dat_sbj_pzi.shape[0]-1)):
            index_error = np.delete(index_error, np.where(index_error>(dat_sbj_pzi.shape[0]-1)))

        for ind in index_error:    
            if (dat_sbj_pzi.firstUndo[ind] == 0)&(dat_sbj_pzi.lastUndo[ind] == 0)&(dat_sbj_pzi.currNumCities[ind-1] > 1):
                list_start.append(0)
                list_nostart.append(0)
            elif (dat_sbj_pzi['firstUndo'][ind] == 1)&(dat_sbj_pzi['lastUndo'][ind] != 1):
                while dat_sbj_pzi.lastUndo[ind] != 1:
                    ind = ind + 1
                if (dat_sbj_pzi['currNumCities'][ind] == 1):
                    list_start.append(1)
                    list_nostart.append(0)
                elif (dat_sbj_pzi['currNumCities'][ind] > 1):
                    list_start.append(0)
                    list_nostart.append(1)
    
count_start = sum(list_start)
count_nostart = sum(list_nostart)
print([count_start/len(list_start),count_nostart/len(list_nostart)])


# -

stat, pval = proportions_ztest([count_start,count_nostart], [len(list_start),len(list_nostart)])
print('{0:0.3f}'.format(pval))

# ##### averaged within subject

# +
# FROM EACH SUBJECT
dat_subjects = []
for i in np.unique(np.array(data_choice_level_df_2['subjects'])):
    
    temp_data = []
    for j in np.unique(sc_data_choice_level['puzzleID']):
        
        no_undo = 0
        undo_start = 0
        undo_nostart = 0
    
        index_sub_puz =  data_choice_level_df_2.index[(data_choice_level_df_2.subjects == i)&(data_choice_level_df_2.puzzleID == j)]

        # NO error --------------------------------------------------
        index_error = data_choice_level_df_2['severityOfErrors'].index[data_choice_level_df_2['severityOfErrors'] == 0]
        index_error = np.array(index_error)
        index_error = np.intersect1d(index_error, index_sub_puz)
        index_error += 1
        if np.any(index_error>(data_choice_level_df_2.shape[0]-1)):
            index_error = np.delete(index_error, np.where(index_error>(data_choice_level_df_2.shape[0]-1)))

        n_move = len(index_error) # number of all moves
        for ind in index_error:
            run = 1
            if data_choice_level_df_2['firstUndo'][ind] == 0:
                no_undo = no_undo + 1
            elif (data_choice_level_df_2['firstUndo'][ind] == data_choice_level_df_2['lastUndo'][ind])&(data_choice_level_df_2['firstUndo'][ind] == 1):
                n_move = n_move - 1
            elif (data_choice_level_df_2['firstUndo'][ind] != data_choice_level_df_2['lastUndo'][ind])&(data_choice_level_df_2['firstUndo'][ind] == 1):
                while run == 1:
                    ind = ind + 1
                    if (data_choice_level_df_2['lastUndo'][ind] == 1)&(data_choice_level_df_2['currNumCities'][ind] == 1):
                        undo_start = undo_start + 1
                        run = 0
                    elif (data_choice_level_df_2['lastUndo'][ind] == 1)&(data_choice_level_df_2['currNumCities'][ind] > 1):
                        undo_nostart = undo_nostart + 1
                        run = 0

        if n_move == 0:
            continue
        else:
            temp_data_puz = [no_undo/n_move,undo_start/n_move,undo_nostart/n_move]
        temp_data.append(temp_data_puz)
        
    dat_subjects.append(np.nanmean(temp_data,axis=0))

dat_subjects = np.array(dat_subjects)

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
# axs.bar([1,2],
#         np.mean(dat_subjects,axis = 0),
#         color=[.7,.7,.7], 
#         edgecolor = 'k', 
#         yerr=np.std(dat_subjects,axis = 0)/np.sqrt(dat_subjects.shape[0]))

bx = axs.boxplot(
    [
        dat_subjects[:,1],
        dat_subjects[:,2]
    ],
    positions =[1,2],
    widths = 0.3,
    showfliers=False,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

axs.set_xticks([1,2])
axs.set_yticks(np.linspace(0,0.08,9))
axs.set_xticklabels(labels = ['start city', 'non-start city'])#,fontsize=18

axs.set_xlabel('No instant error')
axs.set_ylabel('Probability of undo')



stat1, p1 = ttest_ind(dat_subjects[:,1], dat_subjects[:,2])
print(stat1)
print(p1)
axs.set_title(r"$p = {0:s}$".format(as_si(p1,1)))

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
# fig.savefig(out_dir + 'undo_prob_no_instant_error_target.png', dpi=600, bbox_inches='tight')

# -


# ## proportion of first undo depending on end/optimal

# ### separate bar plot

# #### all first undo

# +
dat_subjects = []
for i in np.unique(np.array(data_choice_level['subjects'])):
    df_subjects =  data_choice_level[data_choice_level['subjects'] == i]
    
    allundo = (df_subjects.firstUndo == 1)
    undo_index = df_subjects[allundo].index - 1
    # end --------------------------------------------------
    end_notoptimal = (df_subjects.loc[undo_index,'allMAS']!= df_subjects.loc[undo_index,'currMas'])&(df_subjects.loc[undo_index,'checkEnd']==1)
    
    end_optimal = (df_subjects.loc[undo_index,'allMAS']== df_subjects.loc[undo_index,'currMas'])&(df_subjects.loc[undo_index,'checkEnd']==1)

    # not end  --------------------------------------------------
    notend_notoptimal = (df_subjects.loc[undo_index,'allMAS']!= df_subjects.loc[undo_index,'currMas'])&(df_subjects.loc[undo_index,'checkEnd']!=1)
    
    notend_optimal = (df_subjects.loc[undo_index,'allMAS']== df_subjects.loc[undo_index,'currMas'])&(df_subjects.loc[undo_index,'checkEnd']!=1)
    
    # get prop --------------------------------------------------
    if (sum(allundo) != 0):
        end_notoptimal_freq = sum(end_notoptimal)/sum(allundo)
        end_optimal_freq = sum(end_optimal)/sum(allundo)
        notend_notoptimal_freq = sum(notend_notoptimal)/sum(allundo)
        notend_optimal_freq = sum(notend_optimal)/sum(allundo)
    
        dat_subjects.append([end_notoptimal_freq,end_optimal_freq,notend_notoptimal_freq,notend_optimal_freq])
    else: 
        dat_subjects.append([0,0,0,0])
    
dat_subjects = np.array(dat_subjects)
print(np.mean(dat_subjects,axis=0))
# -

# #### only using sequential undo

# +
dat_subjects = []
for i in np.unique(np.array(data_choice_level['subjects'])):
    df_subjects =  data_choice_level[data_choice_level['subjects'] == i]
    
    allundo = (df_subjects.firstUndo == 1)&(df_subjects.lastUndo != 1)
    undo_index = df_subjects[allundo].index - 1
    # end --------------------------------------------------
    end_notoptimal = (df_subjects.loc[undo_index,'allMAS']!= df_subjects.loc[undo_index,'currMas'])&(df_subjects.loc[undo_index,'checkEnd']==1)
    
    end_optimal = (df_subjects.loc[undo_index,'allMAS']== df_subjects.loc[undo_index,'currMas'])&(df_subjects.loc[undo_index,'checkEnd']==1)

    # not end  --------------------------------------------------
    notend_notoptimal = (df_subjects.loc[undo_index,'allMAS']!= df_subjects.loc[undo_index,'currMas'])&(df_subjects.loc[undo_index,'checkEnd']!=1)
    
    notend_optimal = (df_subjects.loc[undo_index,'allMAS']== df_subjects.loc[undo_index,'currMas'])&(df_subjects.loc[undo_index,'checkEnd']!=1)
    
    # get prop --------------------------------------------------
    if (sum(allundo) != 0):
        end_notoptimal_freq = sum(end_notoptimal)/sum(allundo)
        end_optimal_freq = sum(end_optimal)/sum(allundo)
        notend_notoptimal_freq = sum(notend_notoptimal)/sum(allundo)
        notend_optimal_freq = sum(notend_optimal)/sum(allundo)
    
        dat_subjects.append([end_notoptimal_freq,end_optimal_freq,notend_notoptimal_freq,notend_optimal_freq])
    else: 
        dat_subjects.append([0,0,0,0])
    
dat_subjects = np.array(dat_subjects)
print(np.mean(dat_subjects,axis=0))

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
axs.bar([1,2,3,4],np.mean(dat_subjects,axis = 0),
        color=['#ccd5ae','#ccd5ae','#fefae0','#fefae0'],
        hatch=['//','','//',''],edgecolor = 'k', 
        yerr=np.std(dat_subjects,axis = 0)/np.sqrt(dat_subjects.shape[0]))

axs.set_xticks([1,1.5,2,3,3.5,4])
axs.set_xticklabels(labels = ['not optimal',
                              '\n end state',
                              'optimal',
                              'not optimal',
                              '\n non-end state',
                              'optimal'])#,fontsize=18

axs.set_ylabel('proportion of undo')
axs.set_xlabel('state before undo')
# fig.set_figheight(4)
# fig.set_figwidth(3)

plt.show()
# fig.savefig(out_dir + 'conditional_pundo_givenError.pdf', dpi=600, bbox_inches='tight')
# -

# ### stacked bar

# +
notoptimal = (data_choice_level.firstUndo == 1)&(data_choice_level.allMAS != data_choice_level.currMas)
end_notoptimal = (data_choice_level.loc[data_choice_level.loc[notoptimal,:].index-1,:].checkEnd==1)

optimal = (data_choice_level.firstUndo == 1)&(data_choice_level.allMAS == data_choice_level.currMas)
end_optimal = (data_choice_level.loc[data_choice_level.loc[optimal,:].index-1,:].checkEnd==1)

allundo = (data_choice_level.firstUndo == 1)
notend = (data_choice_level.loc[data_choice_level.loc[allundo,:].index-1,:].checkEnd!=1)

end_notoptimal_freq = sum(end_notoptimal)/sum(allundo)
end_optimal_freq = sum(end_optimal)/sum(allundo)
notend_freq = sum(notend)/sum(allundo)

print(notend_freq)

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

width = 1  # the width of the bars

axs.bar(1 - width/2, end_notoptimal_freq, width, label='end_notoptimal',hatch='//',color='#d4a373',edgecolor='black')
axs.bar(1 - width/2, end_optimal_freq, width, bottom=end_notoptimal_freq,
       label='end_optimal',color='#d4a373',edgecolor='black')
axs.bar(1.5 + width/2, notend_freq, width,
       label='notend',color='#ccd5ae',edgecolor='black')

axs.set_ylabel('proportion of first undo')
axs.legend()
plt.show()
# fig.savefig(out_dir + 'undotype_errortype.pdf', dpi=600, bbox_inches='tight')
# -

# ## probability of undo depending on end/optima

# +
dat_subjects = []
for i in np.unique(np.array(data_choice_level_df['subjects'])):
    temp_data = []
    
    df_subjects =  data_choice_level_df[data_choice_level_df['subjects'] == i]
    
    # end notoptimal--------------------------------------------------
    end_notoptimal = (df_subjects.checkEnd == 1)&(df_subjects.submit != 1)&(df_subjects.allMAS != df_subjects.currMas)
    index_undo = df_subjects.loc[end_notoptimal,:].index+1
    
    if np.any(index_undo>(max(df_subjects.index))):
        index_undo = np.delete(index_undo, np.where(index_undo>max(df_subjects.index)))
    temp_data.append(np.nanmean(df_subjects['firstUndo'][index_undo]))
    
    # end optimal--------------------------------------------------
    end_optimal = (df_subjects.checkEnd == 1)&(df_subjects.submit != 1)&(df_subjects.allMAS == df_subjects.currMas)
    index_undo = df_subjects.loc[end_optimal,:].index+1

    if np.any(index_undo>(max(df_subjects.index))):
        index_undo = np.delete(index_undo, np.where(index_undo>max(df_subjects.index)))
    if np.isnan(np.nanmean(df_subjects['firstUndo'][index_undo])):
        temp_data.append(0)
    else:
        temp_data.append(np.nanmean(df_subjects['firstUndo'][index_undo]))

    # notend notoptimal--------------------------------------------------
    notend_notoptimal = (df_subjects.checkEnd != 1)&(df_subjects.submit != 1)&(df_subjects.allMAS != df_subjects.currMas)
    index_undo = df_subjects.loc[notend_notoptimal,:].index+1

    if np.any(index_undo>(max(df_subjects.index))):
        index_undo = np.delete(index_undo, np.where(index_undo>max(df_subjects.index)))
    temp_data.append(np.nanmean(df_subjects['firstUndo'][index_undo]))
    
    # notend optimal--------------------------------------------------
    notend_optimal = (df_subjects.checkEnd != 1)&(df_subjects.submit != 1)&(df_subjects.allMAS == df_subjects.currMas)
    index_undo = df_subjects.loc[notend_optimal,:].index+1

    if np.any(index_undo>(max(df_subjects.index))):
        index_undo = np.delete(index_undo, np.where(index_undo>max(df_subjects.index)))
    temp_data.append(np.nanmean(df_subjects['firstUndo'][index_undo]))
    
    # end/not end ----------------------------------------------------------
    end = (df_subjects.checkEnd == 1)&(df_subjects.submit != 1)
    index_undo = df_subjects.loc[end,:].index+1
    
    if np.any(index_undo>(max(df_subjects.index))):
        index_undo = np.delete(index_undo, np.where(index_undo>max(df_subjects.index)))   
    if np.isnan(np.nanmean(df_subjects['firstUndo'][index_undo])):
        temp_data.append(0)
    else:
        temp_data.append(np.nanmean(df_subjects['firstUndo'][index_undo]))
    
    notend = (df_subjects.checkEnd != 1)&(df_subjects.submit != 1)
    index_undo = df_subjects.loc[notend,:].index+1

    if np.any(index_undo>(max(df_subjects.index))):
        index_undo = np.delete(index_undo, np.where(index_undo>max(df_subjects.index)))
    if np.isnan(np.nanmean(df_subjects['firstUndo'][index_undo])):
        temp_data.append(0)
    else:
        temp_data.append(np.nanmean(df_subjects['firstUndo'][index_undo]))
    
    # optimal/not optimal ----------------------------------------------------------
    optimal = (df_subjects.allMAS == df_subjects.currMas)
    index_undo = df_subjects.loc[optimal,:].index+1
    
    if np.any(index_undo>(max(df_subjects.index))):
        index_undo = np.delete(index_undo, np.where(index_undo>max(df_subjects.index)))        
    if np.isnan(np.nanmean(df_subjects['firstUndo'][index_undo])):
        temp_data.append(0)
    else:
        temp_data.append(np.nanmean(df_subjects['firstUndo'][index_undo]))
    
    notoptimal = (df_subjects.allMAS != df_subjects.currMas)
    index_undo = df_subjects.loc[notoptimal,:].index+1

    if np.any(index_undo>(max(df_subjects.index))):
        index_undo = np.delete(index_undo, np.where(index_undo>max(df_subjects.index)))
    if np.isnan(np.nanmean(df_subjects['firstUndo'][index_undo])):
        temp_data.append(0)
    else:
        temp_data.append(np.nanmean(df_subjects['firstUndo'][index_undo]))
    
    # --------------------------------------------------------------------------
    dat_subjects.append(temp_data)

dat_subjects = np.array(dat_subjects)
# print(dat_subjects)

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
# axs.bar([1,2,3,4],np.nanmean(dat_subjects,axis = 0),
#         color=['#ccd5ae','#ccd5ae','#fefae0','#fefae0'],
#         hatch=['//','','//',''],edgecolor = 'k', 
#         yerr=np.std(dat_subjects,axis = 0)/np.sqrt(dat_subjects.shape[0]))

bx = axs.boxplot(
    [
        dat_subjects[:,4],
        dat_subjects[:,5],
    ],
    positions =[1,2],
    widths = 0.3,
    showfliers=False,
    patch_artist=True,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

colors = ['#ccd5ae','#ccd5ae']
for patch, color in zip(bx['boxes'], colors):
    patch.set_facecolor(color)

axs.set_xticks([1,2])
axs.set_xticklabels(labels = ['end state',
                              'non-end state'])#,fontsize=18

axs.set_ylabel('Probability of undo')
axs.set_xlabel('State before undo')

stat1, p1 = ttest_ind(dat_subjects[:,4], dat_subjects[:,5])
print(stat1)
print(p1)
axs.set_title(r"$p = {0:s}$".format(as_si(p1,1)))

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
fig.savefig(out_dir + 'undo_prob_end_state.png', dpi=600, bbox_inches='tight')

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
# axs.bar([1,2,3,4],np.nanmean(dat_subjects,axis = 0),
#         color=['#ccd5ae','#ccd5ae','#fefae0','#fefae0'],
#         hatch=['//','','//',''],edgecolor = 'k', 
#         yerr=np.std(dat_subjects,axis = 0)/np.sqrt(dat_subjects.shape[0]))

bx = axs.boxplot(
    [
        dat_subjects[:,6],
        dat_subjects[:,7],
    ],
    positions =[1,2],
    widths = 0.3,
    showfliers=False,
    patch_artist=True,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

colors = ['#fefae0','#fefae0']
for patch, color in zip(bx['boxes'], colors):
    patch.set_facecolor(color)

axs.set_xticks([1,2])
axs.set_xticklabels(labels = ['optimal',
                              'not optimal'])#,fontsize=18

axs.set_ylabel('Probability of undo')
axs.set_xlabel('State before undo')

stat1, p1 = ttest_ind(dat_subjects[:,6], dat_subjects[:,7])
print(stat1)
print(p1)
axs.set_title(r"$p = {0:s}$".format(as_si(p1,1)))

# fig.set_figheight(4)
# fig.set_figwidth(3)
plt.show()
fig.savefig(out_dir + 'undo_prob_optimal.png', dpi=600, bbox_inches='tight')

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 2)
# axs.bar([1,2,3,4],np.nanmean(dat_subjects,axis = 0),
#         color=['#ccd5ae','#ccd5ae','#fefae0','#fefae0'],
#         hatch=['//','','//',''],edgecolor = 'k', 
#         yerr=np.std(dat_subjects,axis = 0)/np.sqrt(dat_subjects.shape[0]))

bx1 = axs[0].boxplot(
    [
        dat_subjects[:,0],
        dat_subjects[:,1]
    ],
    positions =[1,2],
    widths = 0.3,
    showfliers=False,
    patch_artist=True,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

bx2 = axs[1].boxplot(
    [
        dat_subjects[:,2],
        dat_subjects[:,3]
    ],
    positions =[1,2],
    widths = 0.3,
    showfliers=False,
    patch_artist=True,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

colors_1 = ['#ccd5ae','#ccd5ae']
colors_2 = ['#fefae0','#fefae0']
for patch, color in zip(bx1['boxes'], colors_1):
    patch.set_facecolor(color)
for patch, color in zip(bx2['boxes'], colors_2):
    patch.set_facecolor(color)
    
hatches = ['//', '']
for bx in [bx1,bx2]:
    for i in range(len(bx['boxes'])):
        bx['boxes'][i].set(hatch = hatches[i])

axs[0].set_xticks([1,1.5,2])
axs[1].set_xticks([1,1.5,2])
axs[1].set_yticks(np.linspace(0,0.05,6))
axs[0].set_xticklabels(labels = ['not optimal',
                              '\n end state',
                              'optimal'])
axs[1].set_xticklabels(labels = ['not optimal',
                              '\n non-end state',
                              'optimal'])#,fontsize=18

axs[0].set_ylabel('probability of undo')
# fig.set_figheight(4)
# fig.set_figwidth(3)

plt.show()
fig.savefig(out_dir + 'undo_prob_end_optimal_state.png', dpi=600, bbox_inches='tight')
# -

# ## Proportion of single/sequential undo depending on end

# +
singleundo = (data_choice_level.firstUndo == 1)&(data_choice_level.lastUndo == 1)
end_singleundo = (data_choice_level.loc[data_choice_level.loc[singleundo,:].index-1,:].checkEnd==1)
noend_singleundo = (data_choice_level.loc[data_choice_level.loc[singleundo,:].index-1,:].checkEnd!=1)

sequentialundo = (data_choice_level.firstUndo == 1)&(data_choice_level.lastUndo != 1)
end_sequentialundo = (data_choice_level.loc[data_choice_level.loc[sequentialundo,:].index-1,:].checkEnd==1)
noend_sequentialundo = (data_choice_level.loc[data_choice_level.loc[sequentialundo,:].index-1,:].checkEnd!=1)

allundo = (data_choice_level.firstUndo == 1)

end_singleundo_freq = sum(end_singleundo)/sum(allundo)
end_sequentialundo_freq = sum(end_sequentialundo)/sum(allundo)

noend_singleundo_freq = sum(noend_singleundo)/sum(allundo)
noend_sequentialundo_freq = sum(noend_sequentialundo)/sum(allundo)

# +
# %matplotlib notebook

fig, axs = plt.subplots(1, 1)

width = 1  # the width of the bars

axs.bar(1 - width/2, end_singleundo_freq, width, 
        label='end_singleundo',hatch='//',color='#d4a373',edgecolor='black')
axs.bar(1 - width/2, end_sequentialundo_freq, width, bottom=end_singleundo_freq,
       label='end_sequentialundo',color='#d4a373',edgecolor='black')
axs.bar(1.5 + width/2, noend_singleundo_freq, width, hatch='//',
       label='noend_singleundo',color='#ccd5ae',edgecolor='black')
axs.bar(1.5 + width/2, noend_sequentialundo_freq, width,bottom=noend_singleundo_freq,
       label='noend_sequentialundo',color='#ccd5ae',edgecolor='black')

axs.set_ylabel('proportion of undo')
axs.legend()
plt.show()
# fig.savefig(out_dir + 'undotype_errortype.pdf', dpi=600, bbox_inches='tight')
# -

# # Undo target state

# ## Whether the undo target is start city may be dependent on the accumulated error

# +
accu_error = []

for sub in range(100):
    dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index"])
    accu_error_puzzle = []
    
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        
     
        lastUndo_idx_start = dat_sbj_pzi[(dat_sbj_pzi["lastUndo"]==1)&(dat_sbj_pzi["choice"]==0)].index
        lastUndo_idx_nostart = dat_sbj_pzi[(dat_sbj_pzi["lastUndo"]==1)&(dat_sbj_pzi["choice"]!=0)].index
        lastUndo_idx = dat_sbj_pzi[(dat_sbj_pzi["lastUndo"]==1)].index
        
        start_idx = np.where(np.isin(lastUndo_idx,lastUndo_idx_start))
        firstUndo_idx_start = dat_sbj_pzi[dat_sbj_pzi["firstUndo"]==1].index[start_idx]
        nostart_idx = np.where(np.isin(lastUndo_idx,lastUndo_idx_nostart))
        firstUndo_idx_nostart = dat_sbj_pzi[dat_sbj_pzi["firstUndo"]==1].index[nostart_idx]
        
        df_beforeUndo_start = dat_sbj_pzi.loc[firstUndo_idx_start-1,:]
        accu_severity_error_start = df_beforeUndo_start['allMAS'] - df_beforeUndo_start['currMas']
        
        df_beforeUndo_nostart = dat_sbj_pzi.loc[firstUndo_idx_nostart-1,:]
        accu_severity_error_nostart = df_beforeUndo_nostart['allMAS'] - df_beforeUndo_nostart['currMas']
        
        accu_error_puzzle.append([np.nanmean(accu_severity_error_start),np.nanmean(accu_severity_error_nostart)])
        
    accu_error_puzzle =  np.array(accu_error_puzzle)
    accu_error_puzzle =  np.nanmean(accu_error_puzzle,axis=0)

    accu_error.append(accu_error_puzzle)
    
accu_error = np.array(accu_error)
# -

# %matplotlib notebook
plt.figure()
plt.bar(range(2), np.nanmean(accu_error,axis=0),
        color=[.7,.7,.7], edgecolor = 'k', 
        yerr=np.nanstd(accu_error,axis = 0)/np.sqrt(accu_error.shape[0]))
plt.xticks([0,1], ['start city','not start city'])
plt.ylabel('accumulated error')

# ## compare 3 states: before-undo target state, undo target state and after-undo target state if undo target is not the start city

# +
accu_error_3 = []

for sub in range(100):
    dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index"])
    accu_error_puzzle = []
    
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        
        accu_severity_error_after = []
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        
     
        lastUndo_idx_nostart = dat_sbj_pzi[(dat_sbj_pzi["lastUndo"]==1)&(dat_sbj_pzi["choice"]!=0)].index
        
        df_undoTarget = dat_sbj_pzi.loc[lastUndo_idx_nostart,:]
        accu_severity_error = df_undoTarget['allMAS'] - df_undoTarget['currMas']
        
        df_undoTarget_before = dat_sbj_pzi.loc[lastUndo_idx_nostart-1,:]
        accu_severity_error_before = df_undoTarget_before['allMAS'] - df_undoTarget_before['currMas']
        
        undoTarget_after_choice = [eval(path)[:-1] for path in list(df_undoTarget['path'])]
        try: 
            if len(undoTarget_after_choice)!=0:
                for item in undoTarget_after_choice:
                    after_path = str(item)
                    df_after = dat_sbj_pzi[dat_sbj_pzi["path"]==after_path].iloc[0]
                    accu_severity_error_after.append((df_after['allMAS'] - df_after['currMas']))
        except: 
            accu_severity_error_after.append(None)
        
        accu_error_puzzle.append([np.nanmean(accu_severity_error_before),np.nanmean(accu_severity_error),np.nanmean(accu_severity_error_after)])

    accu_error_puzzle =  np.array(accu_error_puzzle)
    accu_error_puzzle =  np.nanmean(accu_error_puzzle,axis=0)

    accu_error_3.append(accu_error_puzzle)
    
accu_error_3 = np.array(accu_error_3)

# -

# %matplotlib notebook
plt.figure()
plt.bar(range(3), np.nanmean(accu_error_3,axis=0),
        color=[.7,.7,.7], edgecolor = 'k', 
        yerr=np.nanstd(accu_error_3,axis = 0)/np.sqrt(accu_error.shape[0]))
plt.xticks([0,1,2], ['before target undo one less','at target','after target undo one more'])
plt.ylabel('accumulated error')

# ## Check the position where they "first" made errors, and whether undo target is that
#

# ### undo target is not the start

# +
accu_error_3 = []

for sub in range(101):
    dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index"])
    accu_error_puzzle = []
    
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        
     
        lastUndo_idx_nostart = dat_sbj_pzi[(dat_sbj_pzi["lastUndo"]==1)&(dat_sbj_pzi["choice"]!=0)].index
        
        df_undoTarget = dat_sbj_pzi.loc[lastUndo_idx_nostart,:]
        accu_severity_error = list(df_undoTarget['allMAS'] - df_undoTarget['currMas'])
        
        df_undoTarget_before = dat_sbj_pzi.loc[lastUndo_idx_nostart-1,:]
        accu_severity_error_before = list(df_undoTarget_before['allMAS'] - df_undoTarget_before['currMas'])
        
        category = [np.nan]*len(accu_severity_error)
        
        for i in range(len(accu_severity_error)): 
            if accu_severity_error_before[i]==0:
                category[i] = 0
            elif (accu_severity_error[i]==0)&(accu_severity_error_before[i]>0):
                category[i] = 1
            elif accu_severity_error[i] > 0:
                category[i] = 2
        
        accu_error_puzzle.extend(category)
    accu_error_puzzle =  np.array(accu_error_puzzle)
    accu_error_3.append([np.sum(accu_error_puzzle==0), np.sum(accu_error_puzzle==1) ,np.sum(accu_error_puzzle==2)])
    
accu_error_3 = np.array(accu_error_3)

# -

# exclude some never undoing subjects
accu_error_3 = accu_error_3[np.where(np.sum(np.array(accu_error_3),axis=1)!=0),:]
accu_error_3 = accu_error_3.squeeze()
accu_error_3_p = accu_error_3/ np.sum(accu_error_3,axis = 1)[:,None]

# %matplotlib notebook
plt.figure()
plt.bar(range(3), np.mean(accu_error_3_p,axis=0),
        color=[.7,.7,.7], edgecolor = 'k', 
        yerr=np.std(accu_error_3_p,axis = 0)/np.sqrt(accu_error_3_p.shape[0]))
plt.xticks([0,1,2], ['undo too many','undo right','undo too few'])
plt.ylabel('probability')

# ### When undo target is the start 

# +
accu_error_2 = []

for sub in range(101):
    dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index"])
    accu_error_puzzle = []
    
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        
     
        lastUndo_idx_start = dat_sbj_pzi[(dat_sbj_pzi["lastUndo"]==1)&(dat_sbj_pzi["choice"]==0)].index

        df_undoTarget = dat_sbj_pzi.loc[lastUndo_idx_start,:]
        accu_severity_error = list(df_undoTarget['allMAS'] - df_undoTarget['currMas'])
        
        df_undoTarget_before = dat_sbj_pzi.loc[lastUndo_idx_start-1,:]
        accu_severity_error_before = list(df_undoTarget_before['allMAS'] - df_undoTarget_before['currMas'])
        
        category = [np.nan]*len(accu_severity_error)
        
        for i in range(len(accu_severity_error)): 
            if accu_severity_error_before[i]==0:
                category[i] = 0
            elif (accu_severity_error[i]==0)&(accu_severity_error_before[i]>0):
                category[i] = 1
        
        accu_error_puzzle.extend(category)

        
    accu_error_puzzle =  np.array(accu_error_puzzle)
    accu_error_2.append([np.sum(accu_error_puzzle==0), np.sum(accu_error_puzzle==1)])
    
accu_error_2 = np.array(accu_error_2)
# -

# exclude some never undoing subjects
accu_error_2 = accu_error_2[np.where(np.sum(np.array(accu_error_2),axis=1)!=0),:]
accu_error_2 = accu_error_2.squeeze()
accu_error_2_p = accu_error_2/ np.sum(accu_error_2,axis = 1)[:,None]

# %matplotlib notebook
plt.figure()
plt.bar(range(2), np.mean(accu_error_2_p,axis=0),
        color=[.7,.7,.7], edgecolor = 'k', 
        yerr=np.std(accu_error_2_p,axis = 0)/np.sqrt(accu_error_2_p.shape[0]))
plt.xticks([0,1], ['undo too many','undo right'])
plt.ylabel('probability')

# # Branching node

# ## histogram of the position of branching node (similar to undo target, but not counting every visit

# +
pos_branching = data_choice_level[data_choice_level['branchingFirst']==True]['currNumCities']

# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
axs.hist(pos_branching,
        color = (.7,.7,.7), 
        edgecolor = 'k',)
axs.set_ylabel('counts') 
axs.set_xlabel('the postion of branching node') 
plt.show()
# -


# ##  number of visits to branching node

# +
each_trial = data_choice_level[data_choice_level['condition']==1].groupby(["subjects","puzzleID"])
n_undotarget_visit = each_trial["branching"].count()
n_undotarget_visit

# %matplotlib notebook

fig, axs = plt.subplots(1, 1)
axs.hist(n_undotarget_visit,
        color = (.7,.7,.7), 
        bins = [2,3,4,5,6,7,8,9],
        edgecolor = 'k',)
axs.set_ylabel('counts') 
axs.set_xlabel('number of branching node visit') 
axs.set_xticks([2,3,4,5,6,7,8])
axs.set_xticklabels(["1","2","3","4","5","6","7+"])
plt.show()
# -

# # After undo, do people choose a different path?

# ## Choose the same city?

# ### all cases

# +
undo_same_diff = []

for sub in range(100):
    dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index"])
    undo_same_diff_puzzle = []
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        same_puzzle = 0
        diff_puzzle = 0
        
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        
     
        lastUndo_idx = dat_sbj_pzi[dat_sbj_pzi["lastUndo"]==1].index
        
        idxx = np.array(dat_sbj_pzi["choice"][lastUndo_idx-1]) != np.array(dat_sbj_pzi["choice"][lastUndo_idx+1])
        diff_puzzle += np.sum(idxx)
        idxx = np.array(dat_sbj_pzi["choice"][lastUndo_idx-1]) == np.array(dat_sbj_pzi["choice"][lastUndo_idx+1])
        same_puzzle += np.sum(idxx)
        
        undo_same_diff_puzzle.append([same_puzzle, diff_puzzle])
    undo_same_diff_puzzle =  np.array(undo_same_diff_puzzle)
    undo_same_diff_puzzle =  np.sum(undo_same_diff_puzzle,axis=0)

    undo_same_diff.append(undo_same_diff_puzzle)
    
undo_same_diff = np.array(undo_same_diff)
# -

# exclude some never undoing subjects
undo_same_diff = undo_same_diff[np.where(np.sum(np.array(undo_same_diff),axis=1)!=0),:]
undo_same_diff = undo_same_diff.squeeze()
undo_same_diff_p = undo_same_diff/ np.sum(undo_same_diff,axis = 1)[:,None]

# +
# %matplotlib notebook

plt.figure()
plt.bar(range(2), np.mean(undo_same_diff_p,axis=0),
        color=[.7,.7,.7], edgecolor = 'k', 
        yerr=np.std(undo_same_diff_p,axis = 0)/np.sqrt(undo_same_diff_p.shape[0]))
plt.xticks([0,1], ['same','different'])
plt.xlabel('Next city after undoing')
# -

# ### separate for undo target is start city or not

# +
undo_same_diff_start = [] # whether it is a start city
undo_same_diff_nostart = []

for sub in range(100):
    dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index"])
    
    undo_same_diff_puzzle_start = []
    undo_same_diff_puzzle_nostart = []
    
    for pzi in np.unique(sc_data_choice_level['puzzleID']):        
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        
    
        lastUndo_idx_start = dat_sbj_pzi[(dat_sbj_pzi["lastUndo"]==1)&(dat_sbj_pzi["choice"]==0)].index  
        lastUndo_idx_nostart = dat_sbj_pzi[(dat_sbj_pzi["lastUndo"]==1)&(dat_sbj_pzi["choice"]!=0)].index

        idxx_start = np.array(dat_sbj_pzi["choice"][lastUndo_idx_start-1]) != np.array(dat_sbj_pzi["choice"][lastUndo_idx_start+1])
        diff_puzzle_start = np.sum(idxx_start)
        idxx_start = np.array(dat_sbj_pzi["choice"][lastUndo_idx_start-1]) == np.array(dat_sbj_pzi["choice"][lastUndo_idx_start+1])
        same_puzzle_start = np.sum(idxx_start)
        
        idxx_nostart = np.array(dat_sbj_pzi["choice"][lastUndo_idx_nostart-1]) != np.array(dat_sbj_pzi["choice"][lastUndo_idx_nostart+1])
        diff_puzzle_nostart = np.sum(idxx_nostart)
        idxx_nostart = np.array(dat_sbj_pzi["choice"][lastUndo_idx_nostart-1]) == np.array(dat_sbj_pzi["choice"][lastUndo_idx_nostart+1])
        same_puzzle_nostart = np.sum(idxx_nostart)
        
        undo_same_diff_puzzle_start.append([same_puzzle_start, diff_puzzle_start]) # one number for each puzzle
        undo_same_diff_puzzle_nostart.append([same_puzzle_nostart, diff_puzzle_nostart])
        
    undo_same_diff_puzzle_start =  np.array(undo_same_diff_puzzle_start) # one number for each subject
    undo_same_diff_puzzle_start =  np.sum(undo_same_diff_puzzle_start,axis=0) 
    undo_same_diff_puzzle_nostart =  np.array(undo_same_diff_puzzle_nostart)
    undo_same_diff_puzzle_nostart =  np.sum(undo_same_diff_puzzle_nostart,axis=0)
    
    undo_same_diff_start.append(undo_same_diff_puzzle_start)
    undo_same_diff_nostart.append(undo_same_diff_puzzle_nostart)
    
undo_same_diff_start = np.array(undo_same_diff_start)
undo_same_diff_nostart = np.array(undo_same_diff_nostart)

# +
# exclude some never undoing subjects
undo_same_diff_start = undo_same_diff_start[np.where(np.sum(np.array(undo_same_diff_start),axis=1)!=0),:]
undo_same_diff_start = undo_same_diff_start.squeeze()
undo_same_diff_start_p = undo_same_diff_start/ np.sum(undo_same_diff_start,axis = 1)[:,None]

undo_same_diff_nostart = undo_same_diff_nostart[np.where(np.sum(np.array(undo_same_diff_nostart),axis=1)!=0),:]
undo_same_diff_nostart = undo_same_diff_nostart.squeeze()
undo_same_diff_nostart_p = undo_same_diff_nostart/ np.sum(undo_same_diff_nostart,axis = 1)[:,None]

undo_same_diff_p_mean = np.concatenate((np.mean(undo_same_diff_start_p,axis=0), np.mean(undo_same_diff_nostart_p,axis=0)), axis=None)
undo_same_diff_p_yerr = np.concatenate((np.std(undo_same_diff_start_p,axis = 0)/np.sqrt(undo_same_diff_start_p.shape[0]), 
                                        np.std(undo_same_diff_nostart_p,axis = 0)/np.sqrt(undo_same_diff_nostart_p.shape[0])), axis=None)

# +
# %matplotlib notebook

plt.figure()
plt.bar([0,1,2,3], undo_same_diff_p_mean,
        color=[.7,.7,.7], edgecolor = 'k',
        yerr=undo_same_diff_p_yerr
       )
plt.xticks([0,0.5,1, 2,2.5,3], ['same','\n start city','different','same','\nnot start city','different'])
plt.xlabel('Next city after undoing')
# -

# ## Choose the same path (total overlap in path, the undo is unnecessary)?

# +
undo_for_better = []

for sub in range(100):
    dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index"])
    
    undo_for_puzzle = []
    
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()           

        firstUndo_idx_ = dat_sbj_pzi[dat_sbj_pzi["firstUndo"]==1].index
        firstUndo_idx_ = firstUndo_idx_.to_list()

        lastUndo_idx = dat_sbj_pzi[dat_sbj_pzi["lastUndo"]==1].index
        lastUndo_idx = lastUndo_idx.to_list()

        submit_idx = dat_sbj_pzi[dat_sbj_pzi["submit"]==1].index
        submit_idx = submit_idx.to_list()

        lastUndo_idx_ = [0]
        lastUndo_idx_.extend(lastUndo_idx) # put 0 at the beginning
        firstUndo_idx = submit_idx.copy()
        firstUndo_idx_.extend(firstUndo_idx) # put submit at the end
        
        for tr in range(len(firstUndo_idx_)-1):
            Prev_seq = dat_sbj_pzi['choice'][lastUndo_idx_[tr]:firstUndo_idx_[tr]].to_list()
            Curr_seq = dat_sbj_pzi['choice'][lastUndo_idx_[tr+1]:firstUndo_idx_[tr+1]].to_list()
            
#             print('*'*10)
#             print(tr)
#             print(Prev_seq)
#             print(Curr_seq)
#             print('*'*10)
#             print(Prev_seq)
#             print(Curr_seq)
#             undo_target = Curr_seq[0]
#             print(undo_target)

            # if a participant undid more than the length of the previous undoing sequence
            if dat_sbj_pzi['currNumCities'][lastUndo_idx_[tr]]>dat_sbj_pzi['currNumCities'][lastUndo_idx_[tr+1]]: 
                # then it is simply whether the previous undoing sequence is in the current undoing sequence
                if np.all(np.isin(Prev_seq, Curr_seq)): # main function!
                    undo_for_puzzle.append(1)
                else:
                    undo_for_puzzle.append(0)
            else:
                # first find undo target
                undo_target = Curr_seq[0]
                Prev_seq_temp = Prev_seq[np.argwhere(np.array(Prev_seq)==undo_target).squeeze():]
                len_frag = min(len(Prev_seq_temp), len(Curr_seq))
                Prev_seq_temp = Prev_seq_temp[:len_frag]
                Curr_seq_temp = Curr_seq[:len_frag]

                if np.all(Prev_seq_temp==Curr_seq_temp):
                    undo_for_puzzle.append(1)
                else:
                    undo_for_puzzle.append(0)
#             print('hup')
    undo_for_puzzle =  np.array(undo_for_puzzle)
    undo_for_better.append([np.sum(undo_for_puzzle==0), np.sum(undo_for_puzzle==1)])
undo_for_better = np.array(undo_for_better)
# -

# exclude some never undoing subjects
undo_for_better = undo_for_better[np.where(np.sum(np.array(undo_for_better),axis=1)!=0),:]
undo_for_better = undo_for_better.squeeze()
undo_for_better_p = undo_for_better/ np.sum(undo_for_better,axis = 1)[:,None]

# %matplotlib notebook
plt.figure()
plt.bar(range(2), np.mean(undo_for_better_p,axis=0),
        color=[.7,.7,.7], edgecolor = 'k', 
        yerr=np.std(undo_for_better_p,axis = 0)/np.sqrt(undo_for_better_p.shape[0]))
plt.xticks([0,1], ['different','same'])
plt.xlabel('Next path after undoing')

# ## count number of visits to a state (not undo target)

# +
importer = JsonImporter()
visit = []
for ti in range(len(undo_tree)): # loop through trials
    root = importer.import_(undo_tree[ti])
    
    for node in PreOrderIter(root): # loop through the tree
        n_child = len(node.children)
        visit.append(node.visit)

visit.count(2)
# -

# # After undo, how often do people actually choose better ?! 

# ## better move

# ### TODO: only if they choose different paths

# +
undo_for_better = []

for sub in range(100):
    dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index"])
    undo_for_puzzle = []
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        

        firstUndo_idx = dat_sbj_pzi[dat_sbj_pzi["firstUndo"]==1].index
        path_bf_undo = dat_sbj_pzi["currMas"][firstUndo_idx-1] # the mas of the state before undo
        
        lastUndo_idx = dat_sbj_pzi[dat_sbj_pzi["lastUndo"]==1].index
        path_af_undo = dat_sbj_pzi["currMas"][lastUndo_idx+1] # the mas of the state after undo
        
        # I think it doesn't mean it choose different path, it means at least once the chosen city is different after undo
        if np.any(np.array(dat_sbj_pzi["choice"][lastUndo_idx-1]) != np.array(dat_sbj_pzi["choice"][lastUndo_idx+1])):
            
            idxx = np.array(dat_sbj_pzi["choice"][lastUndo_idx-1]) != np.array(dat_sbj_pzi["choice"][lastUndo_idx+1])
            undo_for_puzzle.extend(np.sign(np.array(path_af_undo[idxx]) - np.array(path_bf_undo[idxx])))
#         else:
#             print('hup')
    undo_for_puzzle =  np.array(undo_for_puzzle)
    undo_for_better.append([np.sum(undo_for_puzzle<0), np.sum(undo_for_puzzle==0) ,np.sum(undo_for_puzzle>0)])
undo_for_better = np.array(undo_for_better)
# -

# exclude some never undoing subjects
undo_for_better = undo_for_better[np.where(np.sum(np.array(undo_for_better),axis=1)!=0),:]
undo_for_better = undo_for_better.squeeze()
undo_for_better_p = undo_for_better/ np.sum(undo_for_better,axis = 1)[:,None]

# %matplotlib notebook
plt.figure()
plt.bar(range(3), np.mean(undo_for_better_p,axis=0),
        color=[.7,.7,.7], edgecolor = 'k', 
        yerr=np.std(undo_for_better_p,axis = 0)/np.sqrt(undo_for_better_p.shape[0]))
plt.xticks([0,1,2], ['worse','no_diff','better'])
plt.xlabel('After undoing')

# ### all cases (including undoing to the same paths)

# +
undo_better = []

for sub in range(101):
    dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index"])
    undo_puzzle = []
    
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        

        firstUndo_idx = dat_sbj_pzi[dat_sbj_pzi["firstUndo"]==1].index
        path_bf_undo = dat_sbj_pzi["currMas"][firstUndo_idx-1]
        
        lastUndo_idx = dat_sbj_pzi[dat_sbj_pzi["lastUndo"]==1].index
        path_af_undo = dat_sbj_pzi["currMas"][lastUndo_idx+1]
        
        change = np.sign(np.array(path_af_undo) - np.array(path_bf_undo))
        category = [np.nan]*len(change) #initialize
        
        for i in range(len(change)): 
            if change[i] < 0: # worse
                category[i] = 0
            elif (change[i] == 0)& (dat_sbj_pzi["allMAS"][firstUndo_idx[i]-1] != dat_sbj_pzi["currMas"][firstUndo_idx[i]-1]):#(dat_sbj_pzi["choice"][firstUndo_idx[i]-1]) == np.array(dat_sbj_pzi["choice"][lastUndo_idx[i]+1]):
                category[i] = 1
            elif (change[i] == 0)& (dat_sbj_pzi["allMAS"][firstUndo_idx[i]-1] == dat_sbj_pzi["currMas"][firstUndo_idx[i]-1]):
                category[i] = 2
            elif change[i] > 0: # better
                category[i] = 3
        # only after undoing to a different path 
#         if np.any(np.array(dat_sbj_pzi["choice"][lastUndo_idx-1]) != np.array(dat_sbj_pzi["choice"][lastUndo_idx+1])):
#             idxx = np.array(dat_sbj_pzi["choice"][lastUndo_idx-1]) != np.array(dat_sbj_pzi["choice"][lastUndo_idx+1])
        undo_puzzle.extend(category)
#         else:
#             print('hup')
    undo_puzzle =  np.array(undo_puzzle)
    undo_better.append([np.sum(undo_puzzle==0), np.sum(undo_puzzle==1) ,np.sum(undo_puzzle==2), np.sum(undo_puzzle==3)])
undo_better = np.array(undo_better)

# -

# exclude some never undoing subjects
undo_better = undo_better[np.where(np.sum(np.array(undo_better),axis=1)!=0),:]
undo_better = undo_better.squeeze()
undo_better_p = undo_better/ np.sum(undo_better,axis = 1)[:,None]

# +
# %matplotlib notebook
plt.figure()
# plt.bar(range(4), 
#         np.mean(undo_for_better_p,axis=0),
#         color=[.7,.7,.7], edgecolor = 'k', 
#         yerr=np.std(undo_for_better_p,axis = 0)/np.sqrt(undo_for_better_p.shape[0]))

bx = plt.boxplot(
    [
        undo_better_p[:,0],
        undo_better_p[:,1],
        undo_better_p[:,2],
        undo_better_p[:,3]
    ],
    positions =[0,1,2,3],
    widths = 0.3,
    showfliers=False,
    whis = 1.5,
   medianprops = dict(color = 'k'))  #

plt.xticks([0,1,2,3], 
           ['worse','no difference \n (not optimal)',
            'no difference \n (optimal)','better'])
# plt.xlabel('Move after undo')
plt.ylabel('Proportion of move after undo')
plt.savefig(out_dir + 'move_after_undo_prob.png', dpi=600, bbox_inches='tight')
# -

# ## better path

# ###  Including going to the same path

# +
undo_for_better = []

for sub in range(101):
    dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index"])
    undo_for_puzzle = []
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        

        firstUndo_idx = dat_sbj_pzi[dat_sbj_pzi["firstUndo"]==1].index
        path_bf_undo = dat_sbj_pzi["currMas"][firstUndo_idx-1]
        firstUndo_idx = firstUndo_idx.to_list()
        
        submit_idx = dat_sbj_pzi[dat_sbj_pzi["submit"]==1].index
        submit_idx = submit_idx.to_list()
        
        firstUndo_idx.extend(submit_idx) 
        
        for i in range(len(firstUndo_idx)-1):
            undo_for_puzzle.append(dat_sbj_pzi["currMas"][firstUndo_idx[i+1]-1] - dat_sbj_pzi["currMas"][firstUndo_idx[i]-1])
    
#             print('hup')
    undo_for_puzzle =  np.array(undo_for_puzzle)
    undo_for_better.append([np.sum(undo_for_puzzle<0), np.sum(undo_for_puzzle==0) ,np.sum(undo_for_puzzle>0)])
undo_for_better = np.array(undo_for_better)
# -

# exclude some never undoing subjects
undo_for_better = undo_for_better[np.where(np.sum(np.array(undo_for_better),axis=1)!=0),:]
undo_for_better = undo_for_better.squeeze()
undo_for_better_p = undo_for_better/ np.sum(undo_for_better,axis = 1)[:,None]

# %matplotlib notebook
plt.figure()
plt.bar(range(3), np.mean(undo_for_better_p,axis=0),
        color=[.7,.7,.7], edgecolor = 'k', 
        yerr=np.std(undo_for_better_p,axis = 0)/np.sqrt(undo_for_better_p.shape[0]))
plt.xticks([0,1,2], ['worse','no_diff','better'])
plt.xlabel('After undoing')

# ## before undo or after undo, are they in a optimal path?

# +
optimal_before_all = []
optimal_after_all = []

for sub in range(101):
    dat_sbj  = sc_data_choice_level[sc_data_choice_level['subjects']==sub].sort_values(["puzzleID","index"])
    optimal_before = []
    optimal_after = []
    
    for pzi in np.unique(sc_data_choice_level['puzzleID']):
        dat_sbj_pzi = dat_sbj[dat_sbj['puzzleID'] == pzi].reset_index()        

        firstUndo_idx = dat_sbj_pzi[dat_sbj_pzi["firstUndo"]==1].index
        path_bf_undo = (dat_sbj_pzi["allMAS"][firstUndo_idx-1] - dat_sbj_pzi["currMas"][firstUndo_idx-1])
        
        lastUndo_idx = dat_sbj_pzi[dat_sbj_pzi["lastUndo"]==1].index
        path_af_undo = (dat_sbj_pzi["allMAS"][lastUndo_idx+1] - dat_sbj_pzi["currMas"][lastUndo_idx+1])
        
        optimal_before.extend(path_bf_undo)
        optimal_after.extend(path_af_undo)

    optimal_before = np.array(optimal_before)
    optimal_after = np.array(optimal_after)
    
    optimal_before_all.append([np.sum(optimal_before>0), np.sum(optimal_before==0)])
    optimal_after_all.append([np.sum(optimal_after>0), np.sum(optimal_after==0)])
    
optimal_before_all = np.array(optimal_before_all)
optimal_after_all = np.array(optimal_after_all)

# +
# exclude some never undoing subjects
optimal_before_all = optimal_before_all[np.where(np.sum(np.array(optimal_before_all),axis=1)!=0),:]
optimal_before_all = optimal_before_all.squeeze()
optimal_before_all_p = optimal_before_all/ np.sum(optimal_before_all,axis = 1)[:,None]

optimal_after_all = optimal_after_all[np.where(np.sum(np.array(optimal_after_all),axis=1)!=0),:]
optimal_after_all = optimal_after_all.squeeze()
optimal_after_all_p = optimal_after_all/ np.sum(optimal_after_all,axis = 1)[:,None]
# -

# %matplotlib notebook
plt.figure()
plt.bar(range(2), [np.mean(optimal_before_all_p,axis=0)[1],np.mean(optimal_after_all_p,axis=0)[1]],
        color=[.7,.7,.7], edgecolor = 'k',
        yerr=[np.std(optimal_before_all_p,axis = 0)[1]/np.sqrt(optimal_before_all_p.shape[0]),
              np.std(optimal_after_all_p,axis = 0)[1]/np.sqrt(optimal_after_all_p.shape[0])])
plt.xticks([0,1], ['before undo','after undo'])
plt.ylabel('probability of being in optimal path')


