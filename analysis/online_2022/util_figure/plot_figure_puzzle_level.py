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
from glob import glob
from scipy.stats import wilcoxon

class figure_plott:
    def __init__(self, home_dir='./', map_dir='active_map/', data_dir='data/in-lab-pre-pilot/', out_dir = 'figures_prepilot/'):
        self.home_dir = home_dir
        self.map_dir = map_dir
        self.data_dir = data_dir
        self.out_dir = out_dir

        data_all = []
        flist = glob(self.home_dir + self.data_dir + '/preprocess4_sub_*.csv')
        # import experiment data
        for fname in flist:
            with open(fname, 'r') as f:
                all_data = pd.read_csv(f)
                data_all.append(all_data)

        self.data_all = data_all

        # load basic map
        with open(self.home_dir + self.map_dir + 'basicMap.json', 'r') as file:
            self.basic_map = json.load(file)
        # load undo map
        with open(self.home_dir + self.map_dir + 'undoMap.json', 'r') as file:
            self.undo_map = json.load(file)

        ####################################
        # DATA loading
        # Puzzle-level data
        R_out_dir = 'R_analysis/'
        self.numCities = np.genfromtxt(R_out_dir + 'numCities.csv', delimiter=',')
        self.mas = np.genfromtxt(R_out_dir + 'mas.csv', delimiter=',')
        self.nos = np.genfromtxt(R_out_dir + 'nos.csv', delimiter=',')
        self.undo_c = np.genfromtxt(R_out_dir + 'undo_c.csv', delimiter=',')
        self.leftover = np.genfromtxt(R_out_dir + 'leftover.csv', delimiter=',')
        self.numError = np.genfromtxt(R_out_dir + 'numError.csv', delimiter=',')
        self.sumSeverityErrors = np.genfromtxt(R_out_dir + 'sumSeverityErrors.csv', delimiter=',')
        self.numUNDO = np.genfromtxt(R_out_dir + 'numUNDO.csv', delimiter=',')
        self.TT = np.genfromtxt(R_out_dir + 'TT.csv', delimiter=',')
        self.puzzleID = np.genfromtxt(R_out_dir + 'puzzleID.csv', delimiter=',')

        self.data_puzzle_level = np.genfromtxt(R_out_dir +  'data.csv', delimiter=',', names=True)

        # Choice-level data
        self.choicelevel_undo_c = np.genfromtxt(R_out_dir + 'choicelevel_undo_c.csv')
        self.choicelevel_undo = np.genfromtxt(R_out_dir + 'choicelevel_undo.csv')
        self.choicelevel_severityOfErrors = np.genfromtxt(R_out_dir + 'choicelevel_severityOfErrors.csv')
        self.choicelevel_error = np.genfromtxt(R_out_dir + 'choicelevel_error.csv')
        self.choicelevel_currNumCities = np.genfromtxt(R_out_dir + 'choicelevel_currNumCities.csv')
        self.choicelevel_currMas = np.genfromtxt(R_out_dir + 'choicelevel_currMas.csv')
        self.choicelevel_currNos = np.genfromtxt(R_out_dir + 'choicelevel_currNos.csv')
        self.choicelevel_RT = np.genfromtxt(R_out_dir + 'choicelevel_RT.csv')
        self.choicelevel_undoRT = np.genfromtxt(R_out_dir + 'choicelevel_undoRT.csv')
        self.choicelevel_subjects = np.genfromtxt(R_out_dir + 'choicelevel_subjects.csv')
        self.choicelevel_puzzleID = np.genfromtxt(R_out_dir + 'choicelevel_puzzleID.csv')
        self.choicelevel_trialID = np.genfromtxt(R_out_dir + 'choicelevel_trialID.csv')
        self.choicelevel_leftover = np.genfromtxt(R_out_dir + 'choicelevel_leftover.csv')

        self.data_choice_level = np.genfromtxt(R_out_dir +  'choicelevel_data.csv', delimiter=',', names=True)
        self.data_undo_choice_level = np.genfromtxt(R_out_dir +  'choicelevel_undo_data.csv', delimiter=',', names=True)

    def filter_subset(self,tag_name):
        def inner_func(ind,data):
            temp = []
            for j in range(data.shape[1]):
                temp_set = ind[np.argwhere(ind[:, 1] == j)].squeeze()
                temp.append(data[temp_set[:, 0], j])

            return np.array(temp).transpose()
        R_out_dir = 'R_analysis/'
        if tag_name == 'undo':

            subset_ind = np.argwhere(self.undo_c == 1)
            self.numCities = inner_func(subset_ind, self.numCities)
            self.mas = inner_func(subset_ind, self.mas)
            self.nos = inner_func(subset_ind, self.nos)
            self.undo_c = inner_func(subset_ind, self.undo_c)
            self.leftover = inner_func(subset_ind, self.leftover)
            self.numError = inner_func(subset_ind, self.numError)
            self.sumSeverityErrors = inner_func(subset_ind, self.sumSeverityErrors)
            self.numUNDO = inner_func(subset_ind, self.numUNDO)
            self.TT = inner_func(subset_ind, self.TT)
            self.puzzleID = inner_func(subset_ind, self.puzzleID)

            # self.data_puzzle_level = np.genfromtxt(R_out_dir +  'data.csv', delimiter=',', names=True)

            # Choice-level data
            subset_ind = np.argwhere(self.choicelevel_undo_c==1)
            self.choicelevel_undo_c = self.choicelevel_undo_c[subset_ind.squeeze()]
            self.choicelevel_undo = self.choicelevel_undo[subset_ind.squeeze()]
            self.choicelevel_severityOfErrors = self.choicelevel_severityOfErrors[subset_ind.squeeze()]
            self.choicelevel_error = self.choicelevel_error[subset_ind.squeeze()]
            self.choicelevel_currNumCities = self.choicelevel_currNumCities[subset_ind.squeeze()]
            self.choicelevel_currMas = self.choicelevel_currMas[subset_ind.squeeze()]
            self.choicelevel_currNos = self.choicelevel_currNos[subset_ind.squeeze()]
            self.choicelevel_RT = self.choicelevel_RT[subset_ind.squeeze()]
            self.choicelevel_undoRT = self.choicelevel_undoRT[subset_ind.squeeze()]
            self.choicelevel_subjects = self.choicelevel_subjects[subset_ind.squeeze()]
            self.choicelevel_puzzleID = self.choicelevel_puzzleID[subset_ind.squeeze()]
            self.choicelevel_trialID = self.choicelevel_trialID[subset_ind.squeeze()]
            self.choicelevel_leftover = self.choicelevel_leftover[subset_ind.squeeze()]

            self.choicelevel_puzzleerror = []
            prev_tr = -1
            prev_mas = 999
            for ti in range(len(self.choicelevel_trialID)):
                if prev_tr != self.choicelevel_trialID[ti]:
                    prev_mas = self.choicelevel_currMas[ti]

                self.choicelevel_puzzleerror.append(prev_mas-self.choicelevel_currNumCities[ti])

            self.choicelevel_puzzleerror = np.array(self.choicelevel_puzzleerror)

            # self.data_choice_level = np.genfromtxt(R_out_dir +  'choicelevel_data.csv', delimiter=',', names=True)
            # self.data_undo_choice_level = np.genfromtxt(R_out_dir +  'choicelevel_undo_data.csv', delimiter=',', names=True)


        elif tag_name == 'basic':

            subset_ind = np.argwhere(self.undo_c == 0)
            self.numCities = inner_func(subset_ind, self.numCities)
            self.mas = inner_func(subset_ind, self.mas)
            self.nos = inner_func(subset_ind, self.nos)
            self.undo_c = inner_func(subset_ind, self.undo_c)
            self.leftover = inner_func(subset_ind, self.leftover)
            self.numError = inner_func(subset_ind, self.numError)
            self.sumSeverityErrors = inner_func(subset_ind, self.sumSeverityErrors)
            self.numUNDO = inner_func(subset_ind, self.numUNDO)
            self.TT = inner_func(subset_ind, self.TT)
            self.puzzleID = inner_func(subset_ind, self.puzzleID)

            # self.data_puzzle_level = np.genfromtxt(R_out_dir +  'data.csv', delimiter=',', names=True)

            # Choice-level data
            subset_ind = np.argwhere(self.choicelevel_undo_c==0)
            self.choicelevel_undo_c = self.choicelevel_undo_c[subset_ind.squeeze()]
            self.choicelevel_undo = self.choicelevel_undo[subset_ind.squeeze()]
            self.choicelevel_severityOfErrors = self.choicelevel_severityOfErrors[subset_ind.squeeze()]
            self.choicelevel_error = self.choicelevel_error[subset_ind.squeeze()]
            self.choicelevel_currNumCities = self.choicelevel_currNumCities[subset_ind.squeeze()]
            self.choicelevel_currMas = self.choicelevel_currMas[subset_ind.squeeze()]
            self.choicelevel_currNos = self.choicelevel_currNos[subset_ind.squeeze()]
            self.choicelevel_RT = self.choicelevel_RT[subset_ind.squeeze()]
            self.choicelevel_undoRT = self.choicelevel_undoRT[subset_ind.squeeze()]
            self.choicelevel_subjects = self.choicelevel_subjects[subset_ind.squeeze()]
            self.choicelevel_puzzleID = self.choicelevel_puzzleID[subset_ind.squeeze()]
            self.choicelevel_trialID = self.choicelevel_trialID[subset_ind.squeeze()]
            self.choicelevel_leftover = self.choicelevel_leftover[subset_ind.squeeze()]

            self.choicelevel_puzzleerror = []
            prev_tr = -1
            prev_mas = 999
            for ti in range(len(self.choicelevel_trialID)):
                if prev_tr != self.choicelevel_trialID[ti]:
                    prev_mas = self.choicelevel_currMas[ti]

                self.choicelevel_puzzleerror.append(prev_mas-self.choicelevel_currNumCities[ti])

            self.choicelevel_puzzleerror = np.array(self.choicelevel_puzzleerror)

            # self.data_choice_level = np.genfromtxt(R_out_dir +  'choicelevel_data.csv', delimiter=',', names=True)
            # self.data_undo_choice_level = np.genfromtxt(R_out_dir +  'choicelevel_undo_data.csv', delimiter=',', names=True)
    
    def rc_severity_of_errors_hist(self, n_bins =5):
        out_dir = self.out_dir

        # undo_ = self.choicelevel_undo
        # need to analyze whether participants undid in the next move after the error.
        undo_ = np.array([*self.choicelevel_undo[1:].tolist(),0])
        # subjects
        ind_subjects = np.unique(self.choicelevel_subjects)
        errors = self.choicelevel_severityOfErrors
        ind_severity_of_errors = np.unique(self.choicelevel_severityOfErrors)
        soe = np.zeros((len(ind_subjects),len(ind_severity_of_errors)))

        bin_ = [*ind_severity_of_errors.tolist(), 999]
        bin_2 = [*ind_severity_of_errors[1:].tolist(), 999]


        temp_hist = []
        temp_hist2 = []
        for i in range(len(ind_subjects)):

            ind_subid = np.where(self.choicelevel_subjects == i)
            temp ,_ = np.histogram(errors[ind_subid[0]], bins= bin_,density=True)
            temp_hist.append(temp)

            temp ,_ = np.histogram(errors[ind_subid[0]], bins= bin_2,density=True)
            temp_hist2.append(temp)

        temp_hist = np.array(temp_hist)
        temp_hist2 = np.array(temp_hist2)

        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(temp_hist), axis=0)
        std_undo_ = np.nanstd(np.array(temp_hist), axis=0) / np.sqrt(len(ind_subjects))

        x_pos = ind_severity_of_errors
        fig, axs = plt.subplots(1, 1)

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', edgecolor='black',facecolor=[1,1,1],  capsize=10)
        axs.set_xlabel('choice-level error')
        axs.set_ylabel('P')
        axs.set_xticks(x_pos)
        bin_name = [*[str(int(i)) for i in bin_[:-2]], '5+']

        axs.set_xticklabels(bin_name)
        axs.yaxis.grid(True)
        plt.ylim(0,1)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'hist_choicelevel_error.png', dpi=600, bbox_inches='tight')
        plt.close(fig)



        fig, axs = plt.subplots(1, 1)

        mean_undo_ = np.nanmean(np.array(temp_hist2), axis=0)
        std_undo_ = np.nanstd(np.array(temp_hist2), axis=0) / np.sqrt(len(ind_subjects))

        x_pos = ind_severity_of_errors[1:]
        fig, axs = plt.subplots(1, 1)


        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', edgecolor='black',facecolor=[1,1,1],  capsize=10)
        axs.set_xlabel('choice-level error')
        axs.set_ylabel('P')
        axs.set_xticks(x_pos)
        bin_name = [*[str(int(i)) for i in bin_2[:-2]], '5+']

        axs.set_xticklabels(bin_name)
        axs.yaxis.grid(True)
        plt.ylim(0,1)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'hist_choicelevel_error1.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def rc_undo_mas_errors_choice(self):
        out_dir = self.out_dir

        # undo_ = self.choicelevel_undo
        # need to analyze whether participants undid in the next move after the error.
        undo_ = np.array([*self.choicelevel_undo[2:].tolist(),0])
        # subjects
        ind_subjects = np.unique(self.choicelevel_subjects)

        errors = self.choicelevel_currMas[:-1] - self.choicelevel_currMas[1:]
        errors[errors<0] = 0

        # ind_severity_of_errors = np.unique(self.choicelevel_severityOfErrors)
        ind_severity_of_errors = np.unique(errors)
        soe = np.zeros((len(ind_subjects),len(ind_severity_of_errors)))

        for i in range(len(ind_subjects)):
            temp_ =[]
            for j in ind_severity_of_errors:
                # temp_.append(np.sum(self.choicelevel_severityOfErrors[np.where(self.choicelevel_subjects == i)] == j))

                ind_subid = np.where(self.choicelevel_subjects == i)
                ind_errorid = np.where(errors == j)

                ind_ind = np.intersect1d(ind_subid[0], ind_errorid[0])

                num_undo = undo_[ind_ind]
                if np.prod(num_undo.shape) == 0:
                    temp_.append(np.nan)
                else:
                    temp_.append(np.mean(num_undo))

            soe[i,:]  = np.array(temp_)

        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(soe), axis=0)
        std_undo_ = np.nanstd(np.array(soe), axis=0) / np.sqrt(len(ind_subjects))

        x_pos = ind_severity_of_errors
        fig, axs = plt.subplots(1, 1)

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', edgecolor='black', facecolor=[1,1,1], capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('p(undo)')
        axs.set_xticks(x_pos)
        axs.yaxis.grid(True)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'severity_mas_error0_undo.png', dpi=600, bbox_inches='tight')
        plt.close(fig)


# ========================================================================
# Proportion of trials for each puzzle-level error in undo/basic conditions
    def rc_undo_hist_all_across_subjects_ag(self):
        data_all = self.data_all

        BASIC_ALL = {'basic_list':[],'basic':[]}
        UNDO_ALL  = {'undo_list':[],'undo':[]}

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

        BASIC_DIFF_ALL = {'diff': []}
        UNDO_DIFF_ALL = {'diff_undo': []}

        for i in range(0, len(data_all)):
            diff = (np.array(BASIC_ALL['basic'][i]) - np.array(BASIC_ALL['basic_list'][i])).tolist()
            diff_undo = (np.array(UNDO_ALL['undo'][i]) - np.array(UNDO_ALL['undo_list'][i])).tolist()

            BASIC_DIFF_ALL['diff'].append(np.abs(diff))
            UNDO_DIFF_ALL['diff_undo'].append(np.abs(diff_undo))

        UNDO_P = dict()
        BASIC_P = dict()
        for a in range(9):
            UNDO_P[str(a)] = []
            BASIC_P[str(a)] = []

        UNDO_P['data'] = []
        BASIC_P['data'] = []

        for i in range(len(self.data_all)):
            # save for each
            undo_p = []
            basic_p = []
            for a in range(9):
                BASIC_P[str(a)].append(
                    np.sum(np.array(BASIC_DIFF_ALL['diff'][i]) == a) / np.array(BASIC_DIFF_ALL['diff'][i]).shape[0])
                UNDO_P[str(a)].append(np.sum(np.array(UNDO_DIFF_ALL['diff_undo'][i]) == a) /
                                      np.array(UNDO_DIFF_ALL['diff_undo'][i]).shape[0])
                basic_p.append(
                    np.sum(np.array(BASIC_DIFF_ALL['diff'][i]) == a) / np.array(BASIC_DIFF_ALL['diff'][i]).shape[0])
                undo_p.append(np.sum(np.array(UNDO_DIFF_ALL['diff_undo'][i]) == a) /
                              np.array(UNDO_DIFF_ALL['diff_undo'][i]).shape[0])

            UNDO_P['data'].append(undo_p)
            BASIC_P['data'].append(basic_p)

        undo_p_ = np.array(UNDO_P['data'])
        basic_p_ = np.array(BASIC_P['data'])

        mean_undo_p_ = np.mean(undo_p_, axis=0)
        mean_basic_p_ = np.mean(basic_p_, axis=0)

        std_undo_p_ = np.std(undo_p_, axis=0) / np.sqrt(len(data_all))
        std_basic_p_ = np.std(basic_p_, axis=0) / np.sqrt(len(data_all))

        fig, ax = plt.subplots()
        # basic_
        x_pos = np.arange(9) - .1
        ax.bar(x_pos, mean_basic_p_, yerr=std_basic_p_, align='center', alpha=1, width=.2, color='#0776d8',
               ecolor='black', capsize=10)
        ax.set_ylabel('Proportion of trials')
        # ax.set_xticks(x_pos)
        ax.set_title('Number (actual connected - maximum connectable)')

        # undo_
        x_pos = np.arange(9) + .1
        ax.bar(x_pos, mean_undo_p_, yerr=std_undo_p_, align='center', alpha=1, width=.2, color='#e13f42',
               ecolor='black', capsize=10)
        ax.set_ylabel('Proportion of trials')
        # ax.set_xticks(x_pos)
        ax.set_title('Number (actual connected - maximum connectable)')

        x_pos = np.arange(9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(np.abs(x_pos))

        import matplotlib.patches as mpatches

        rc_led = mpatches.Patch(color='#0776d8', label='without undo')
        undo_led = mpatches.Patch(color='#e13f42', label='with undo')
        lgd = plt.legend(handles=[rc_led, undo_led], facecolor='white')
        ax.yaxis.grid(True)

        fig.set_figwidth(10)

        plt.show()
        fig.savefig(self.out_dir + 'rc_undo_hist_all_across_subjects_abs.png', dpi=600, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')

        ax.set_xlim(-.5, 5.5)
        plt.show()
        fig.savefig(self.out_dir + 'rc_undo_hist_all_across_subjects_abs_5to0.png', dpi=600, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.close(fig)

        # summed up for 3+
        undo_p_3 = np.zeros((undo_p_.shape[0],4))
        undo_p_3[:,:3] = undo_p_[:,:3]
        undo_p_3[:,3]  = np.sum(undo_p_[:,3:], axis=1)

        basic_p_3 = np.zeros((basic_p_.shape[0],4))
        basic_p_3[:,:3] = basic_p_[:,:3]
        basic_p_3[:,3]  = np.sum(basic_p_[:,3:], axis=1)

        mean_undo_p_ = np.mean(undo_p_3, axis=0)
        mean_basic_p_ = np.mean(basic_p_3, axis=0)

        std_undo_p_ = np.std(undo_p_3, axis=0) / np.sqrt(len(data_all))
        std_basic_p_ = np.std(basic_p_3, axis=0) / np.sqrt(len(data_all))

        fig, ax = plt.subplots()
        # basic_
        x_pos = np.arange(4) - .1
        ax.bar(x_pos, mean_basic_p_, yerr=std_basic_p_, align='center', alpha=1, width=.2, color='#0776d8',
               ecolor='black', capsize=10)
        ax.set_ylabel('Proportion of trials')
        # ax.set_xticks(x_pos)
        ax.set_title('Number (actual connected - maximum connectable)')

        # undo_
        x_pos = np.arange(4) + .1
        ax.bar(x_pos, mean_undo_p_, yerr=std_undo_p_, align='center', alpha=1, width=.2, color='#e13f42',
               ecolor='black', capsize=10)
        ax.set_ylabel('Proportion of trials')
        # ax.set_xticks(x_pos)
        ax.set_title('Number (actual connected - maximum connectable)')

        x_pos = np.arange(4)
        ax.set_xticks(x_pos)

        x_tick_labels = [str(i) for i in x_pos]
        x_tick_labels[-1] = '3+'
        ax.set_xticklabels(x_tick_labels)

        import matplotlib.patches as mpatches

        rc_led = mpatches.Patch(color='#0776d8', label='without undo')
        undo_led = mpatches.Patch(color='#e13f42', label='with undo')
        lgd = plt.legend(handles=[rc_led, undo_led], facecolor='white')
        ax.yaxis.grid(True)

        fig.set_figwidth(10)

        plt.show()
        fig.savefig(self.out_dir + 'rc_undo_hist_all_across_subjects_abs_3plus.png', dpi=600, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        fig.close()
        
    def rc_undo_hist_all_across_subjects(self):
        
        data_all = self.data_all

        BASIC_ALL = {'basic_list':[],'basic':[]}
        UNDO_ALL  = {'undo_list':[],'undo':[]}

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

        BASIC_DIFF_ALL = {'diff': []}
        UNDO_DIFF_ALL = {'diff_undo': []}

        for i in range(0, len(data_all)):
            diff = (np.array(BASIC_ALL['basic'][i]) - np.array(BASIC_ALL['basic_list'][i])).tolist()
            diff_undo = (np.array(UNDO_ALL['undo'][i]) - np.array(UNDO_ALL['undo_list'][i])).tolist()

            BASIC_DIFF_ALL['diff'].append(diff)
            UNDO_DIFF_ALL['diff_undo'].append(diff_undo)

        UNDO_P = dict()
        BASIC_P = dict()
        for a in range(-8, 1):
            UNDO_P[str(a)] = []
            BASIC_P[str(a)] = []

        UNDO_P['data'] = []
        BASIC_P['data'] = []

        for i in range(len(self.data_all)):
            # save for each
            undo_p = []
            basic_p = []
            for a in range(-8, 1):
                BASIC_P[str(a)].append(
                    np.sum(np.array(BASIC_DIFF_ALL['diff'][i]) == a) / np.array(BASIC_DIFF_ALL['diff'][i]).shape[0])
                UNDO_P[str(a)].append(np.sum(np.array(UNDO_DIFF_ALL['diff_undo'][i]) == a) /
                                      np.array(UNDO_DIFF_ALL['diff_undo'][i]).shape[0])
                basic_p.append(
                    np.sum(np.array(BASIC_DIFF_ALL['diff'][i]) == a) / np.array(BASIC_DIFF_ALL['diff'][i]).shape[0])
                undo_p.append(np.sum(np.array(UNDO_DIFF_ALL['diff_undo'][i]) == a) /
                              np.array(UNDO_DIFF_ALL['diff_undo'][i]).shape[0])

            UNDO_P['data'].append(undo_p)
            BASIC_P['data'].append(basic_p)

        undo_p_ = np.array(UNDO_P['data'])
        basic_p_ = np.array(BASIC_P['data'])

        mean_undo_p_ = np.mean(undo_p_, axis=0)
        mean_basic_p_ = np.mean(basic_p_, axis=0)

        std_undo_p_ = np.std(undo_p_, axis=0) / np.sqrt(len(data_all))
        std_basic_p_ = np.std(basic_p_, axis=0) / np.sqrt(len(data_all))

        fig, ax = plt.subplots()
        # basic_
        x_pos = np.arange(-8, 1) - .1
        ax.bar(x_pos, mean_basic_p_, yerr=std_basic_p_, align='center', alpha=1, width=.2, color='#0776d8',
               ecolor='black', capsize=10)
        ax.set_ylabel('Proportion of trials')
        # ax.set_xticks(x_pos)
        ax.set_title('Number (actual connected - maximum connectable)')

        # undo_
        x_pos = np.arange(-8, 1) + .1
        ax.bar(x_pos, mean_undo_p_, yerr=std_undo_p_, align='center', alpha=1, width=.2, color='#e13f42',
               ecolor='black', capsize=10)
        ax.set_ylabel('Proportion of trials')
        # ax.set_xticks(x_pos)
        ax.set_title('Number (actual connected - maximum connectable)')

        x_pos = np.arange(-8, 1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(np.abs(x_pos))

        import matplotlib.patches as mpatches

        rc_led = mpatches.Patch(color='#0776d8', label='without undo')
        undo_led = mpatches.Patch(color='#e13f42', label='with undo')
        lgd = plt.legend(handles=[rc_led, undo_led], facecolor='white')
        ax.yaxis.grid(True)

        fig.set_figwidth(10)

        plt.show()
        fig.savefig(self.out_dir + 'rc_undo_hist_all_across_subjects.png', dpi=600, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')

        ax.set_xlim(-5.5, 0.5)
        plt.show()
        fig.savefig(self.out_dir + 'rc_undo_hist_all_across_subjects_5to0.png', dpi=600, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.close(fig)
    
    def rc_severe_error_undo(self):
        data_all = self.data_all
        out_dir = self.out_dir
        LEN_UNDO_ERROR = {'undo_': [], 'error_ind': [], 'undo_counter': [], 't_counter': []}
        error_ind = np.linspace(-12, 12, 25).astype(np.int16)
        LEN_UNDO_ERROR['error_ind'] = error_ind.tolist()

        for i in range(len(data_all)):
            undo_trials = data_all[i][data_all[i].map_name == 'undo']
            u_undo_maps = np.unique(np.array(undo_trials['map_id']))

            ti = 0
            prev_trialid = -1
            prev_mapid = -1  # arbitrary number

            t_counter = 0
            undo_counter = 0

            undo_errors_ = np.array([0] * 25)

            while ti < undo_trials.shape[0]:
                if prev_trialid != np.array(undo_trials.trial_id)[ti]:
                    # if prev_mapid != np.array(undo_trials.map_id)[ti]:  # which means if the trial has changed
                    single_trial = undo_trials[np.array(undo_trials.map_id) == np.array(undo_trials.map_id)[ti]]

                    mas_all_trial = np.array(single_trial.mas_all)
                    errors_trial = [0, *(mas_all_trial[1:] - mas_all_trial[:-1]).tolist()]
                    prev_undo = 0
                    len_undo = 0
                    for tti in range(single_trial.shape[0]):
                        t_counter += 1
                        if np.array(single_trial.undo_all)[tti] == 1:
                            undo_counter += 1
                        else:
                            if prev_undo == 1:
                                undo_errors_[error_ind == errors_trial[tti]] += 1

                            len_undo = 0
                        prev_undo = np.array(single_trial.undo_all)[tti]

                    prev_mapid = np.array(undo_trials.map_id)[ti]
                    prev_trialid = np.array(undo_trials.trial_id)[ti]
                ti += 1

            LEN_UNDO_ERROR['undo_'].append(undo_errors_.tolist())
            LEN_UNDO_ERROR['t_counter'].append(t_counter)
            LEN_UNDO_ERROR['undo_counter'].append(undo_counter)

        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.mean(np.array(LEN_UNDO_ERROR['undo_']), axis=0)
        std_undo_ = np.std(np.array(LEN_UNDO_ERROR['undo_']), axis=0) / np.sqrt(len(data_all))

        x_pos = np.arange(-7, 1)
        start_ind = np.where(error_ind == np.array(x_pos)[0])[0].squeeze()
        end_ind = np.where(error_ind == np.array(x_pos)[-1])[0].squeeze()

        axs.bar(x_pos, mean_undo_[start_ind:end_ind + 1], yerr=std_undo_[start_ind:end_ind + 1],
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('Count')
        axs.set_title('Undo')
        axs.set_xticks(x_pos)
        axs.yaxis.grid(True)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'severity_error_undo.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

        fig, axs = plt.subplots(1, 1)
        # undo once
        data_undo_ = np.array(LEN_UNDO_ERROR['undo_'])
        undo_validsub = []
        for i in range(len(data_all)):
            if not np.sum(data_undo_[i, :]) == 0:
                undo_validsub.append(data_undo_[i, :] / np.sum(data_undo_[i, :]))
        undo_validsub = np.array(undo_validsub)

        mean_undo_ = np.mean(undo_validsub, axis=0)
        std_undo_ = np.std(undo_validsub, axis=0) / np.sqrt(len(undo_validsub))

        x_pos = np.arange(-7, 1)
        start_ind = np.where(error_ind == np.array(x_pos)[0])[0].squeeze()
        end_ind = np.where(error_ind == np.array(x_pos)[-1])[0].squeeze()

        axs.bar(x_pos, mean_undo_[start_ind:end_ind + 1], yerr=std_undo_[start_ind:end_ind + 1],
                   align='center',
                   alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('Proportion')
        axs.set_title('Undo')
        axs.set_xticks(x_pos)
        axs.yaxis.grid(True)
        axs.set_ylim(0, .8)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'severity_error_undo_p.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
   
    def rc_error_undo_length(self):
        data_all = self.data_all
        out_dir = self.out_dir
        LEN_UNDO_ERROR = {'undo_once': [], 'undo_series': [], 'error_ind': [], 'undo_counter': [], 't_counter': []}
        error_ind = np.linspace(-12, 12, 25).astype(np.int16)
        LEN_UNDO_ERROR['error_ind'] = error_ind.tolist()

        for i in range(len(data_all)):
            undo_trials = data_all[i][data_all[i].map_name == 'undo']
            u_undo_maps = np.unique(np.array(undo_trials['map_id']))

            ti = 0
            prev_trialid = -1
            prev_mapid = -1  # arbitrary number

            t_counter = 0
            undo_counter = 0

            undo_once_errors_ = np.array([0] * 25)
            undo_series_errors_ = np.array([0] * 25)

            while ti < undo_trials.shape[0]:
                if prev_trialid != np.array(undo_trials.trial_id)[ti]:
                    # if prev_mapid != np.array(undo_trials.map_id)[ti]:  # which means if the trial has changed
                    single_trial = undo_trials[np.array(undo_trials.map_id) == np.array(undo_trials.map_id)[ti]]

                    mas_all_trial = np.array(single_trial.mas_all)
                    errors_trial = [0, *(mas_all_trial[1:] - mas_all_trial[:-1]).tolist()]
                    prev_undo = 0
                    len_undo = 0
                    for tti in range(single_trial.shape[0]):
                        t_counter += 1
                        if np.array(single_trial.undo_all)[tti] == 1:
                            len_undo += 1
                            undo_counter += 1
                        else:
                            if prev_undo == 1:
                                if len_undo == 1:
                                    undo_once_errors_[error_ind == errors_trial[tti]] += 1
                                else:
                                    undo_series_errors_[error_ind == errors_trial[tti]] += 1

                            len_undo = 0
                        prev_undo = np.array(single_trial.undo_all)[tti]

                    prev_mapid = np.array(undo_trials.map_id)[ti]
                    prev_trialid = np.array(undo_trials.trial_id)[ti]
                ti += 1

            LEN_UNDO_ERROR['undo_once'].append(undo_once_errors_.tolist())
            LEN_UNDO_ERROR['undo_series'].append(undo_series_errors_.tolist())
            LEN_UNDO_ERROR['t_counter'].append(t_counter)
            LEN_UNDO_ERROR['undo_counter'].append(undo_counter)

        fig, axs = plt.subplots(1, 2)
        # undo once
        mean_undo_once = np.mean(np.array(LEN_UNDO_ERROR['undo_once']), axis=0)
        std_undo_once = np.std(np.array(LEN_UNDO_ERROR['undo_once']), axis=0) / np.sqrt(len(data_all))

        x_pos = np.arange(-7, 1)
        start_ind = np.where(error_ind == np.array(x_pos)[0])[0].squeeze()
        end_ind = np.where(error_ind == np.array(x_pos)[-1])[0].squeeze()

        axs[0].bar(x_pos, mean_undo_once[start_ind:end_ind + 1], yerr=std_undo_once[start_ind:end_ind + 1],
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs[0].set_xlabel('Error')
        axs[0].set_ylabel('Count')
        axs[0].set_title('Undo once')
        axs[0].set_xticks(x_pos)
        axs[0].yaxis.grid(True)

        # undo once
        mean_undo_series = np.mean(np.array(LEN_UNDO_ERROR['undo_series']), axis=0)
        std_undo_series = np.std(np.array(LEN_UNDO_ERROR['undo_series']), axis=0) / np.sqrt(len(data_all))

        axs[1].bar(x_pos, mean_undo_series[start_ind:end_ind + 1], yerr=std_undo_series[start_ind:end_ind + 1],
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs[1].set_xlabel('Error')
        axs[1].set_title('Series of undo')
        axs[1].set_xticks(x_pos)
        axs[1].yaxis.grid(True)

        fig.set_figwidth(6)
        plt.show()
        fig.savefig(out_dir + 'undo_once_series_count.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

        fig, axs = plt.subplots(1, 2)
        # undo once
        data_undo_once = np.array(LEN_UNDO_ERROR['undo_once'])
        undo_once_validsub = []
        for i in range(len(data_all)):
            if not np.sum(data_undo_once[i, :]) == 0:
                undo_once_validsub.append(data_undo_once[i, :] / np.sum(data_undo_once[i, :]))
        undo_once_validsub = np.array(undo_once_validsub)

        mean_undo_once = np.mean(undo_once_validsub, axis=0)
        std_undo_once = np.std(undo_once_validsub, axis=0) / np.sqrt(len(undo_once_validsub))

        x_pos = np.arange(-7, 1)
        start_ind = np.where(error_ind == np.array(x_pos)[0])[0].squeeze()
        end_ind = np.where(error_ind == np.array(x_pos)[-1])[0].squeeze()

        axs[0].bar(x_pos, mean_undo_once[start_ind:end_ind + 1], yerr=std_undo_once[start_ind:end_ind + 1], align='center',
                   alpha=1, width=.8, ecolor='black', capsize=10)
        axs[0].set_xlabel('Error')
        axs[0].set_ylabel('Proportion')
        axs[0].set_title('Undo once')
        axs[0].set_xticks(x_pos)
        axs[0].yaxis.grid(True)
        axs[0].set_ylim(0, .8)

        # undo once
        data_undo_series = np.array(LEN_UNDO_ERROR['undo_series'])
        undo_series_validsub = []
        for i in range(len(data_all)):
            if not np.sum(data_undo_series[i, :]) == 0:
                undo_series_validsub.append(data_undo_series[i, :] / np.sum(data_undo_series[i, :]))
        undo_series_validsub = np.array(undo_series_validsub)

        mean_undo_series = np.mean(undo_series_validsub, axis=0)
        std_undo_series = np.std(undo_series_validsub, axis=0) / np.sqrt(len(undo_series_validsub))

        axs[1].bar(x_pos, mean_undo_series[start_ind:end_ind + 1], yerr=std_undo_series[start_ind:end_ind + 1],
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs[1].set_xlabel('Error')
        axs[1].set_title('Series of undo')
        axs[1].set_xticks(x_pos)
        axs[1].yaxis.grid(True)
        axs[1].set_ylim(0, .8)

        fig.set_figwidth(6)
        plt.show()
        fig.savefig(out_dir + 'undo_once_series_p.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
    
# =========================================================================
# Number of undo as a function of MAS
    def rc_undo_x_mas(self):
        data_all = self.data_all
        N_UNDO_MAS_ALL = {'n_undo_mas': [], 'p_n_undo_mas': [], 'mas_ind': []}
        mas_ind = np.linspace(1, 12, 12).astype(np.int16).tolist()  # index for the MAS of i-th trial.
        N_UNDO_MAS_ALL['mas_ind'] = mas_ind

        for i in range(len(data_all)):
            undo_trials = data_all[i][data_all[i].map_name == 'undo']
            u_undo_maps = np.unique(np.array(undo_trials['map_id']))
            ti = 0
            prev_mapid = -1  # arbitrary number

            # empty list to save per subject
            mas_ = [0] * 12

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
        data_ = data_[~np.isnan(data_).any(axis=1)]  # exclude some participants who never undo.

        fig, axs = plt.subplots()
        mean_data_ = np.mean(data_, axis=0)
        std_data_ = np.std(data_, axis=0) / np.sqrt(len(data_))

        axs.plot(mas_ind[1:], mean_data_[1:], marker='o',
                 markerfacecolor='#727bda', markeredgecolor='none')
        axs.errorbar(mas_ind[1:], mean_data_[1:], yerr=std_data_[1:], capsize=3, ls='None', color='k')

        axs.set_title('')
        axs.set_xticks(mas_ind[1:])
        axs.yaxis.grid(True)
        fig.set_figwidth(4)
        plt.show()
        fig.savefig(self.out_dir + 'p_num_undo_X_MAS.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

        data_ = np.array(N_UNDO_MAS_ALL['n_undo_mas'])
        data_ = data_[~np.isnan(data_).any(axis=1)]  # exclude some participants who never undo.

        fig, axs = plt.subplots()
        mean_data_ = np.mean(data_, axis=0)
        std_data_ = np.std(data_, axis=0) / np.sqrt(len(data_))

        axs.plot(mas_ind[1:], mean_data_[1:], marker='o',
                 markerfacecolor='#727bda', markeredgecolor='none')
        axs.errorbar(mas_ind[1:], mean_data_[1:], yerr=std_data_[1:], capsize=3, ls='None', color='k')

        axs.set_title('')
        axs.set_xticks(mas_ind[1:])
        axs.yaxis.grid(True)
        fig.set_figwidth(4)
        plt.show()
        fig.savefig(self.out_dir + 'num_undo_X_MAS.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
  
    def rc_undo_c_numciti(self):
        out_dir = self.out_dir
        w_undo =[]
        wo_undo = []
        for i in range(self.numCities.shape[1]):
            ind_undo = np.where(self.undo_c[:,i]==1)
            ind_wo_undo = np.where(self.undo_c[:,i]==0)
            w_undo.append(self.numCities[ind_undo,i])
            wo_undo.append(self.numCities[ind_wo_undo,i])
        w_undo = np.array(w_undo).squeeze().transpose()
        wo_undo = np.array(wo_undo).squeeze().transpose()

        stat1, p1 = wilcoxon(np.mean(w_undo,axis=0), np.mean(wo_undo,axis=0))
        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.mean([np.mean(w_undo,axis=0), np.mean(wo_undo,axis=0)],axis=1)
        std_undo_ = np.std([np.mean(w_undo,axis=0), np.mean(wo_undo,axis=0)],axis=1) / np.sqrt(self.numCities.shape[1])


        fig, axs = plt.subplots(1, 1)

        axs.bar([1,2], mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('condition')
        axs.set_ylabel('n_city')
        axs.set_xticks([1,2])
        axs.yaxis.grid(True)

        axs.set_title('p='+str(p1))
        axs.set_xticklabels(['w undo','wo undo'])
        fig.set_figwidth(4)
        plt.show()
        plt.ylim([8, 10])
        fig.savefig(out_dir + 'nct_condition.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
    
    def rc_undo_severity_of_errors_choice(self):
        out_dir = self.out_dir

        # undo_ = self.choicelevel_undo
        # need to analyze whether participants undid in the next move after the error.
        undo_ = np.array([*self.choicelevel_undo[1:].tolist(),0])
        # subjects
        ind_subjects = np.unique(self.choicelevel_subjects)
        ind_severity_of_errors = np.unique(self.choicelevel_severityOfErrors)
        soe = np.zeros((len(ind_subjects),len(ind_severity_of_errors)))

        for i in range(len(ind_subjects)):
            temp_ =[]
            for j in ind_severity_of_errors:
                # temp_.append(np.sum(self.choicelevel_severityOfErrors[np.where(self.choicelevel_subjects == i)] == j))

                ind_subid = np.where(self.choicelevel_subjects == i)
                ind_errorid = np.where(self.choicelevel_severityOfErrors == j)

                ind_ind = np.intersect1d(ind_subid[0], ind_errorid[0])

                num_undo = undo_[ind_ind]
                if np.prod(num_undo.shape) == 0:
                    temp_.append(np.nan)
                else:
                    temp_.append(np.mean(num_undo))

            soe[i,:]  = np.array(temp_)

        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(soe), axis=0)
        std_undo_ = np.nanstd(np.array(soe), axis=0) / np.sqrt(len(ind_subjects))

        x_pos = ind_severity_of_errors
        fig, axs = plt.subplots(1, 1)

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('Count')
        axs.set_title('Undo')
        axs.set_xticks(x_pos)
        axs.yaxis.grid(True)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'severity_error0_undo.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
    
    def rc_undo_severity_of_puzzle_errors_choice(self):
        out_dir = self.out_dir

        # undo_ = self.choicelevel_undo
        undo_ = np.array([*self.choicelevel_undo[1:].tolist(),0])
        # subjects
        ind_subjects = np.unique(self.choicelevel_subjects)
        ind_severity_of_errors = np.unique(self.choicelevel_puzzleerror)
        soe = np.zeros((len(ind_subjects),len(ind_severity_of_errors)))

        for i in range(len(ind_subjects)):
            temp_ =[]
            for j in ind_severity_of_errors:
                # temp_.append(np.sum(self.choicelevel_severityOfErrors[np.where(self.choicelevel_subjects == i)] == j))

                ind_subid = np.where(self.choicelevel_subjects == i)
                ind_errorid = np.where(self.choicelevel_puzzleerror == j)

                ind_ind = np.intersect1d(ind_subid[0], ind_errorid[0])

                num_undo = undo_[ind_ind]
                if np.prod(num_undo.shape) == 0:
                    temp_.append(np.nan)
                else:
                    temp_.append(np.mean(num_undo))

            soe[i,:]  = np.array(temp_)

        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(soe), axis=0)
        std_undo_ = np.nanstd(np.array(soe), axis=0) / np.sqrt(len(ind_subjects))

        x_pos = ind_severity_of_errors

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('Count')
        axs.set_title('Undo')
        axs.set_xticks(x_pos)
        axs.yaxis.grid(True)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'severity_puzzle_error0_undo.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
    
    def rc_undo_severity_of_errors_puzzle(self):
        out_dir = self.out_dir

        numcities = self.numCities
        mas = self.mas

        errors = mas-numcities
        ind_errors = np.unique(errors)
        undo_ = self.numUNDO

        # subjects
        soe = np.zeros((errors.shape[1], len(ind_errors)))

        for i in range(errors.shape[1]):
            temp_ =[]
            for j in ind_errors:
                num_undo = undo_[np.where(errors[:,i] == j),i]
                if num_undo.shape[1] == 0:
                    temp_.append(np.nan)
                else:
                    temp_.append(np.mean(num_undo))

            soe[i,:]  = np.array(temp_)


        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(soe), axis=0)
        std_undo_ = np.nanstd(np.array(soe), axis=0) / np.sqrt(errors.shape[1])

        x_pos = ind_errors

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('Count')
        axs.set_title('Undo')
        axs.set_xticks(x_pos)
        axs.yaxis.grid(True)

        wilcoxon(np.array(soe)[:,6],np.array(soe)[:,7])
        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'severity_error0_undo_puzzle.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
    
    def rc_undo_mas_choice(self):
        out_dir = self.out_dir

        undo_ = self.choicelevel_undo
        # subjects
        ind_subjects = np.unique(self.choicelevel_subjects)
        ind_currMas = np.unique(self.choicelevel_currMas)
        mas = np.zeros((len(ind_subjects),len(ind_currMas)))

        for i in range(len(ind_subjects)):
            temp_ =[]
            for j in ind_currMas:
                ind_subid = np.where(self.choicelevel_subjects == i)
                ind_currMasid = np.where(self.choicelevel_currMas == j)

                ind_ind = np.intersect1d(ind_subid[0], ind_currMasid[0])

                num_undo = undo_[ind_ind]
                if np.prod(num_undo.shape) == 0:
                    temp_.append(np.nan)
                else:
                    temp_.append(np.mean(num_undo))
            mas[i,:]  = np.array(temp_)


        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(mas), axis=0)
        std_undo_ = np.nanstd(np.array(mas), axis=0) / np.sqrt(len(ind_subjects))

        x_pos = ind_currMas

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('Count')
        axs.set_title('Undo')
        axs.set_xticks(x_pos)
        axs.yaxis.grid(True)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'mas_undo.png', dpi=600, bbox_inches='tight')
        plt.close(fig)
    
# =========================================================================
# Everage number of connected cities       
    def rc_undo_mas_puzzle(self):
        out_dir = self.out_dir

        numcities = self.numCities
        MAS = self.mas

        ind_mas = np.unique(MAS)
        undo_ = self.numUNDO

        # subjects
        mas = np.zeros((MAS.shape[1], len(ind_mas)))

        for i in range(MAS.shape[1]):
            temp_ =[]

            for j in ind_mas:
                num_undo = undo_[np.where(MAS[:,i] == j),i]
                if num_undo.shape[1] == 0:
                    temp_.append(np.nan)
                else:
                    temp_.append(np.mean(num_undo))

            # for j in ind_mas:
            #     temp_.append(np.sum(MAS[:,i] == j))

            mas[i,:]  = np.array(temp_)


        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(mas), axis=0)
        std_undo_ = np.nanstd(np.array(mas), axis=0) / np.sqrt(MAS.shape[1])

        x_pos = ind_mas

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', edgecolor='black',facecolor=[1,1,1],  capsize=10)
        axs.set_xlabel('MAS')
        axs.set_ylabel('Number of undo')
        axs.set_xticks(x_pos)
        axs.yaxis.grid(True)

        wilcoxon(np.array(mas)[:,4],np.array(mas)[:,5])
        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'mas_undo_puzzle.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def rc_undo_nct_choice(self):
        out_dir = self.out_dir

        undo_ = self.choicelevel_undo
        # subjects
        ind_subjects = np.unique(self.choicelevel_subjects)
        ind_currnct = np.unique(self.choicelevel_currNumCities)
        nct = np.zeros((len(ind_subjects),len(ind_currnct)))

        for i in range(len(ind_subjects)):
            temp_ =[]
            for j in ind_currnct:
                ind_subid = np.where(self.choicelevel_subjects == i)
                ind_currNumCities = np.where(self.choicelevel_currNumCities == j)

                ind_ind = np.intersect1d(ind_subid[0], ind_currNumCities[0])

                num_undo = undo_[ind_ind]
                if np.prod(num_undo.shape) == 0:
                    temp_.append(np.nan)
                else:
                    temp_.append(np.mean(num_undo))

            nct[i,:]  = np.array(temp_)


        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(nct), axis=0)
        std_undo_ = np.nanstd(np.array(nct), axis=0) / np.sqrt(len(ind_subjects))

        x_pos = ind_currnct

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('Count')
        axs.set_title('Undo')
        axs.set_xticks(x_pos)
        axs.yaxis.grid(True)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'nct_undo.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def rc_nct_undo_puzzle(self, n_bins=5):
        out_dir = self.out_dir

        NCT = self.numCities
        MAS = self.mas
        undo_ = self.numUNDO

        ind_undo = np.unique(undo_)

        ind_nct = np.unique(NCT)


        undo__ = undo_.reshape((-1))
        NCT__ = NCT.reshape((-1))


        # bin_ = [0]
        bin_ = []
        bin_.extend([np.quantile(undo__[np.where(undo__!= 0)], i) for i in np.linspace(0,1,n_bins+1)])

        # n_bins = 9
        # bin_ = np.histogram_bin_edges([1, np.max(undo_)], bins=n_bins)
        # subjects
        # nct = np.zeros((NCT.shape[1], len(ind_nct)))
        # undo = np.zeros((undo_.shape[1], 7))
        temp_hist = []
        for i in range(NCT.shape[1]):
            ind = np.digitize(undo_[:,i], bin_)
            temp_hist.append([np.mean(NCT[np.where(ind==j),i]) for j in range(len(bin_))])

        temp_hist = np.array(temp_hist)


        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(temp_hist), axis=0)
        std_undo_ = np.nanstd(np.array(temp_hist), axis=0) / np.sqrt(NCT.shape[1])

        x_pos = np.arange(n_bins)

        axs.bar(x_pos, mean_undo_[1:], yerr=std_undo_[1:],
                   align='center', alpha=1, width=.4, ecolor='black', capsize=10, color=[1,1,1], edgecolor=[0,0,0])

        axs.set_xticks(np.linspace(0,n_bins,n_bins+1)-.5)
        axs.set_xticklabels(np.array(bin_).astype(np.int16))

        axs.set_xlabel('Number of undo')
        axs.set_ylabel('# of cities connected')
        axs.yaxis.grid(True)
        plt.ylim(7.5,9.5)
        # plt.xlim(0,n_bins)
        axs.set_yticks(np.linspace(7.5,9.5,5))
        plt.xlim(-1,n_bins)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'nct_undo_puzzle' + str(n_bins) + '.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(temp_hist), axis=0)
        std_undo_ = np.nanstd(np.array(temp_hist), axis=0) / np.sqrt(NCT.shape[1])

        x_pos = np.arange(n_bins)

        axs.bar(x_pos, mean_undo_[1:], yerr=std_undo_[1:],
                   align='center', alpha=1, width=.4, ecolor='black', capsize=10, color=[1,1,1], edgecolor=[0,0,0])

        axs.set_xticks(np.linspace(0,n_bins,n_bins+1)-.5)
        axs.set_xticklabels(np.array(bin_).astype(np.int16))

        axs.set_xlabel('Number of undo')
        axs.set_ylabel('# of cities connected')
        axs.yaxis.grid(True)
        plt.ylim(7.5,9.5)
        # plt.xlim(0,n_bins)
        axs.set_yticks(np.linspace(7.5,9.5,5))

        fig.set_figwidth(4)
        y2 = [mean_undo_[0]+std_undo_[0],mean_undo_[0]+std_undo_[0]]
        y1 = [mean_undo_[0]-std_undo_[0], mean_undo_[0]-std_undo_[0]]
        fig.set_figwidth(4)
        axs.plot([-1, n_bins], [mean_undo_[0] , mean_undo_[0]], color='red')
        axs.fill_between([-1, n_bins], y1, y2, facecolor='red', alpha=.4, interpolate=True, edgecolor=[0,0,0])
        plt.xlim(-1,n_bins)
        plt.show()

        fig.savefig(out_dir + 'nct_undo0red_puzzle_' + str(n_bins) + '.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def rc_severity_of_errors_undo_puzzle(self, n_bins=5):
        out_dir = self.out_dir

        NCT = self.numCities
        undo_ = self.numUNDO

        numcities = self.numCities
        mas = self.mas

        errors = mas-numcities

        undo__ = undo_.reshape((-1))

        # bin_ = [0]
        bin_ = []
        bin_.extend([np.quantile(undo__[np.where(undo__!= 0)], i) for i in np.linspace(0,1,n_bins+1)])

        # n_bins = 9
        # bin_ = np.histogram_bin_edges([1, np.max(undo_)], bins=n_bins)
        # subjects
        # nct = np.zeros((NCT.shape[1], len(ind_nct)))
        # undo = np.zeros((undo_.shape[1], 7))
        temp_hist = []
        for i in range(errors.shape[1]):
            ind = np.digitize(undo_[:,i], bin_)
            temp_hist.append([np.mean(errors[np.where(ind==j),i]) for j in range(len(bin_))])

        temp_hist = np.array(temp_hist)


        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(temp_hist), axis=0)
        std_undo_ = np.nanstd(np.array(temp_hist), axis=0) / np.sqrt(NCT.shape[1])

        x_pos = np.arange(n_bins)

        axs.bar(x_pos, mean_undo_[1:], yerr=std_undo_[1:],
                   align='center', alpha=1, width=.4, ecolor='black', capsize=10, color=[1,1,1], edgecolor=[0,0,0])

        axs.set_xticks(np.linspace(0,n_bins,n_bins+1)-.5)
        axs.set_xticklabels(np.array(bin_).astype(np.int16))

        plt.xlim(-1,n_bins)
        axs.set_xlabel('Number of undo')
        axs.set_ylabel('Error')
        axs.yaxis.grid(True)
        plt.ylim(0,1)
        # plt.xlim(0,n_bins)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'undo_severity_of_errors_puzzle' + str(n_bins) + '.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(temp_hist), axis=0)
        std_undo_ = np.nanstd(np.array(temp_hist), axis=0) / np.sqrt(NCT.shape[1])

        x_pos = np.arange(n_bins)

        axs.bar(x_pos, mean_undo_[1:], yerr=std_undo_[1:],
                   align='center', alpha=1, width=.4, ecolor='black', capsize=10, color=[1,1,1], edgecolor=[0,0,0])

        axs.set_xticks(np.linspace(0,n_bins,n_bins+1)-.5)
        axs.set_xticklabels(np.array(bin_).astype(np.int16))

        plt.xlim(-1,n_bins)
        axs.set_xlabel('Number of undo')
        axs.set_ylabel('# of cities connected')
        axs.yaxis.grid(True)
        plt.ylim(0,1)
        # plt.xlim(0,n_bins)

        y2 = [mean_undo_[0]+std_undo_[0],mean_undo_[0]+std_undo_[0]]
        y1 = [mean_undo_[0]-std_undo_[0], mean_undo_[0]-std_undo_[0]]
        fig.set_figwidth(4)
        axs.plot([-1, n_bins], [mean_undo_[0] , mean_undo_[0]], color='red')
        axs.fill_between([-1, n_bins], y1, y2, facecolor='red', alpha=.4, interpolate=True, edgecolor=[0,0,0])
        plt.show()

        fig.savefig(out_dir + 'undo0_severity_of_errors_puzzle' + str(n_bins) + '.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def rc_undo_nos_hist_choice(self,n_bins = 3):
        out_dir = self.out_dir

        undo_ = self.choicelevel_undo
        # subjects
        ind_subjects = np.unique(self.choicelevel_subjects)
        ind_currNos = np.unique(self.choicelevel_currNos)
        ind_currNos_loc = np.linspace(0,len(ind_currNos)-1,len(ind_currNos))
        nos = np.zeros((len(ind_subjects),len(ind_currNos)))

        NOS = self.nos
        NOS__ = NOS.reshape((-1))

        bin_ = []
        bin_.extend([np.quantile(self.choicelevel_currNos[self.choicelevel_currNos!=1], i) for i in np.linspace(0,1,n_bins+1)])


        temp_hist = []
        for i in range(len(ind_subjects)):
            ind_subid = np.where(self.choicelevel_subjects == i)
            ind = np.digitize(self.choicelevel_currNos[ind_subid], bin_)
            undo_sub = undo_[ind_subid]
            temp_hist.append([np.mean(undo_sub[np.where(ind==j)]) for j in range(len(bin_))])

        temp_hist = np.array(temp_hist)

        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(temp_hist), axis=0)
        std_undo_ = np.nanstd(np.array(temp_hist), axis=0) / np.sqrt(NOS.shape[1])

        x_pos = np.arange(n_bins+1)

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.4, ecolor='black', capsize=10, color=[1,1,1], edgecolor=[0,0,0])

        axs.set_xticks(x_pos)
        bin_ = [1,*[int(i) for i in bin_[:-1]]]
        bin_ = [str(i) for i in bin_]
        bin_[-1] = '4+'
        axs.set_xticklabels(bin_)

        # plt.xlim(-1,n_bins)
        axs.set_xlabel('Number of optimal solutions')
        axs.set_ylabel('p(undo)')
        axs.yaxis.grid(True)
        # plt.ylim(0,6)
        # plt.xlim(0,n_bins)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'nos_pundo_choicelevel' + str(n_bins) + '.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

# ========================================================================
# Number of undo as a function of number of optimal solutions  
    def rc_undo_nos_hist_puzzle(self,n_bins = 10):
        out_dir = self.out_dir

        numcities = self.numCities
        NOS = self.nos
        undo_ = self.numUNDO

        ind_nos = np.unique(NOS)
        ind_nos_loc = np.linspace(0,len(ind_nos)-1,len(ind_nos))

        NOS__ = NOS.reshape((-1))

        bin_ = []
        bin_.extend([np.quantile(NOS__[np.where(NOS__!= 0)], i) for i in np.linspace(0,1,n_bins+1)])

        temp_hist = []
        for i in range(NOS.shape[1]):
            ind = np.digitize(NOS[:,i], bin_)
            temp_hist.append([np.mean(undo_[np.where(ind==j),i]) for j in range(len(bin_))])

        temp_hist = np.array(temp_hist)

        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(temp_hist), axis=0)
        std_undo_ = np.nanstd(np.array(temp_hist), axis=0) / np.sqrt(NOS.shape[1])

        x_pos = np.arange(n_bins)

        axs.bar(x_pos, mean_undo_[1:], yerr=std_undo_[1:],
                   align='center', alpha=1, width=.4, ecolor='black', capsize=10, color=[1,1,1], edgecolor=[0,0,0])

        axs.set_xticks(np.linspace(0,n_bins,n_bins+1)-.5)
        axs.set_xticklabels(np.array(bin_).astype(np.int16))

        plt.xlim(-1,n_bins)
        axs.set_xlabel('Number of optimal solutions')
        axs.set_ylabel('Number of undo')
        axs.yaxis.grid(True)
        plt.ylim(0,6)
        # plt.xlim(0,n_bins)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'nos_undo_puzzle' + str(n_bins) + '.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def rc_undo_severity_of_errors_puzzle(self):
        out_dir = self.out_dir

        numcities = self.numCities
        mas = self.mas

        errors = mas-numcities
        ind_errors = np.unique(errors)
        undo_ = self.numUNDO

        # subjects
        soe = np.zeros((errors.shape[1], len(ind_errors)))

        for i in range(errors.shape[1]):
            temp_ =[]
            for j in ind_errors:
                num_undo = undo_[np.where(errors[:,i] == j),i]
                if num_undo.shape[1] == 0:
                    temp_.append(np.nan)
                else:
                    temp_.append(np.mean(num_undo))

            soe[i,:]  = np.array(temp_)


        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(soe), axis=0)
        std_undo_ = np.nanstd(np.array(soe), axis=0) / np.sqrt(errors.shape[1])

        x_pos = ind_errors

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('Count')
        axs.set_title('Undo')
        axs.set_xticks(x_pos)
        axs.yaxis.grid(True)

        wilcoxon(np.array(soe)[:,6],np.array(soe)[:,7])
        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'severity_error0_undo_puzzle.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def rc_undo_nct_puzzle(self):
        out_dir = self.out_dir

        NCT = self.numCities
        MAS = self.mas
        undo_ = self.numUNDO

        ind_nct = np.unique(NCT)

        # subjects
        nct = np.zeros((NCT.shape[1], len(ind_nct)))

        for i in range(NCT.shape[1]):
            temp_ =[]

            for j in ind_nct:
                num_undo = undo_[np.where(NCT[:,i] == j),i]
                if num_undo.shape[1] == 0:
                    temp_.append(np.nan)
                else:
                    temp_.append(np.mean(num_undo))

            nct[i,:]  = np.array(temp_)


        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(nct), axis=0)
        std_undo_ = np.nanstd(np.array(nct), axis=0) / np.sqrt(NCT.shape[1])

        x_pos = ind_nct

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('Count')
        axs.set_title('Undo')
        axs.set_xticks(x_pos)
        axs.yaxis.grid(True)

        wilcoxon(np.array(nct)[:,6],np.array(nct)[:,7])

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'nct_undo_puzzle.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def rc_undo_nos_choice(self):
        out_dir = self.out_dir

        undo_ = self.choicelevel_undo
        # subjects
        ind_subjects = np.unique(self.choicelevel_subjects)
        ind_currNos = np.unique(self.choicelevel_currNos)
        ind_currNos_loc = np.linspace(0,len(ind_currNos)-1,len(ind_currNos))
        nos = np.zeros((len(ind_subjects),len(ind_currNos)))

        for i in range(len(ind_subjects)):
            temp_ =[]
            for j in ind_currNos:
                ind_subid = np.where(self.choicelevel_subjects == i)
                ind_currNosss = np.where(self.choicelevel_currNos == j)

                ind_ind = np.intersect1d(ind_subid[0], ind_currNosss[0])

                num_undo = undo_[ind_ind]
                if np.prod(num_undo.shape) == 0:
                    temp_.append(np.nan)
                else:
                    temp_.append(np.mean(num_undo))

            nos[i,:]  = np.array(temp_)


        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.mean(np.array(nos), axis=0)
        std_undo_ = np.std(np.array(nos), axis=0) / np.sqrt(len(ind_subjects))

        x_pos = ind_currNos_loc

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('Count')
        axs.set_title('Undo')
        axs.set_xticks(x_pos)
        axs.set_xticklabels(ind_currNos.astype(np.int16))
        plt.xticks(rotation=90)
        axs.yaxis.grid(True)

        fig.set_figwidth(15)
        plt.show()
        fig.savefig(out_dir + 'nos_undo.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def rc_undo_nos_puzzle(self):
        out_dir = self.out_dir

        numcities = self.numCities
        NOS = self.nos
        undo_ = self.numUNDO

        ind_nos = np.unique(NOS)
        ind_nos_loc = np.linspace(0,len(ind_nos)-1,len(ind_nos))

        # subjects
        nos = np.zeros((NOS.shape[1], len(ind_nos)))
        # p_nos = np.zeros((NOS.shape[1], len(ind_nos)))

        for i in range(NOS.shape[1]):
            temp_ =[]

            for j in ind_nos:
                num_undo = undo_[np.where(NOS[:,i] == j),i]
                if num_undo.shape[1] == 0:
                    temp_.append(np.nan)
                else:
                    temp_.append(np.mean(num_undo))
            # for j in ind_nos:
            #     temp_.append(np.sum(NOS[:,i] == j))

            nos[i,:]  = np.array(temp_)
            # p_nos[i,:] = np.array(temp_)/np.sum(temp_)


        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.nanmean(np.array(nos), axis=0)
        std_undo_ = np.nanstd(np.array(nos), axis=0) / np.sqrt(NOS.shape[1])

        x_pos = ind_nos_loc

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('Count')
        axs.set_title('Undo')
        axs.set_xticks(x_pos)
        axs.set_xticklabels(ind_nos.astype(np.int16))
        plt.xticks(rotation=90)
        axs.yaxis.grid(True)

        fig.set_figwidth(15)
        plt.show()
        fig.savefig(out_dir + 'nos_undo_puzzle.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def rc_undo_leftover_choice(self):
        out_dir = self.out_dir

        undo_ = self.choicelevel_undo
        # subjects
        ind_ind_sub = self.choicelevel_subjects
        ind_subjectss = np.unique(self.choicelevel_subjects)
        ind_leftover = np.unique(self.choicelevel_leftover)
        
        
        leftOver = self.choicelevel_leftover
        
        n_bins = 10


        bin_ = np.histogram_bin_edges([0, 300], bins=n_bins)
        lod_undid = np.zeros((len(ind_subjectss),n_bins))
        lod_nono  = np.zeros((len(ind_subjectss),n_bins))
        ind_ind_sub = ind_ind_sub[:-1]
        leftOver = leftOver[:-1]
        undo_    = undo_[1:]
        leftOver_ = []
        
        for i in range(len(ind_subjectss)):
            ind_subjects = np.where(ind_ind_sub == i)
            ind_nono     = np.where(undo_==0)
            ind_undid    = np.where(undo_==1)

            temp_hist, _ = np.histogram(leftOver[np.intersect1d(ind_subjects,ind_nono)],density=True,bins =bin_)
            lod_nono[i,:] =temp_hist
            temp_hist, _ = np.histogram(leftOver[np.intersect1d(ind_subjects,ind_undid)],density=True,bins =bin_)
            lod_undid[i,:] =temp_hist
            leftOver_.append([np.mean(leftOver[np.intersect1d(ind_subjects,ind_nono)]), np.mean(leftOver[np.intersect1d(ind_subjects,ind_undid)])])
        # fig, axs = plt.subplots(1, 2,  gridspec_kw={'width_ratios': [5, 1]})
        fig, axs = plt.subplots(1, 1)
        ax01 = axs.bar(np.linspace(0,n_bins-1,n_bins)+.5-.1, np.nanmean(lod_nono,axis=0), yerr=np.nanstd(lod_nono,axis=0)/np.sqrt(lod_nono.shape[0]),
                   align='center', alpha=1, width=.2,edgecolor='black', ecolor='black', capsize=10, color=[.3,.3,.3], label='Did not undo in the puzzle')
        ax02 = axs.bar(np.linspace(0,n_bins-1,n_bins)+.5+.1, np.nanmean(lod_undid,axis=0), yerr=np.nanstd(lod_undid,axis=0)/np.sqrt(lod_nono.shape[0]),
                   align='center', alpha=1, width=.2,edgecolor='black', ecolor='black', capsize=10, color=[.7,.7,.7], label='Did undo in the puzzle')
        axs.set_xticks(np.linspace(0,n_bins,n_bins+1))
        axs.set_xticklabels(bin_.astype(np.int16))
        axs.legend([ax01, ax02],["Did not undo in the puzzle", "Did undo in the puzzle"])
        axs.legend(["Did not undo in the puzzle", "Did undo in the puzzle"])

        # import matplotlib.patches as mpatches
        # rc_led = mpatches.Patch(color=[0,0,0], label='Did not undo in the puzzle')
        # undo_led = mpatches.Patch(color=[.7,.7,.7], label='Did undo in the puzzle')
        # lgd = plt.legend(handles=[rc_led, undo_led], facecolor='white')
        fig.set_figwidth(12)
        plt.show()

        fig.savefig(out_dir + 'leftover_undo.png', dpi=600, bbox_inches='tight')
        plt.close(fig)



        # remaining budget
        fig, axs = plt.subplots(1, 1)

        leftOver_ = np.array(leftOver_)
        ax01 = axs.bar(1, np.nanmean(leftOver_[0]), yerr=np.nanstd(leftOver_[0])/np.sqrt(lod_nono.shape[0]),
                   align='center', alpha=1, width=.4, ecolor='black', capsize=10, color=[0,0,0], label='Did not undo in the puzzle')
        ax02 = axs.bar(2, np.nanmean(leftOver_[1]), yerr=np.nanstd(leftOver_[1])/np.sqrt(lod_nono.shape[0]),
                   align='center', alpha=1, width=.4, ecolor='black', capsize=10, color=[.7,.7,.7], label='Did undo in the puzzle')
        axs.set_xticks([1,2])
        axs.set_xticklabels(['Did not undo', 'Did undo'])
        fig.set_figwidth(4)
        plt.show()
        plt.xlim(.5, 2.5)
        fig.savefig(out_dir + 'mean_leftover_undo.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

    def rc_undo_leftover_puzzle(self):
        out_dir = self.out_dir

        numcities = self.numCities
        MAS = self.mas

        ind_mas = np.unique(MAS)

        # subjects
        mas = np.zeros((MAS.shape[1], len(ind_mas)))
        p_mas = np.zeros((MAS.shape[1], len(ind_mas)))

        for i in range(MAS.shape[1]):
            temp_ =[]
            for j in ind_mas:
                temp_.append(np.sum(MAS[:,i] == j))

            mas[i,:]  = np.array(temp_)
            p_mas[i,:] = np.array(temp_)/np.sum(temp_)


        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.mean(np.array(mas), axis=0)
        std_undo_ = np.std(np.array(mas), axis=0) / np.sqrt(MAS.shape[1])

        x_pos = ind_mas

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('Count')
        axs.set_title('Undo')
        axs.set_xticks(x_pos)
        axs.yaxis.grid(True)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'mas_undo_puzzle.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

        fig, axs = plt.subplots(1, 1)
        # undo once
        mean_undo_ = np.mean(np.array(p_mas), axis=0)
        std_undo_ = np.std(np.array(p_mas), axis=0) / np.sqrt(MAS.shape[1])

        x_pos = ind_mas

        axs.bar(x_pos, mean_undo_, yerr=std_undo_,
                   align='center', alpha=1, width=.8, ecolor='black', capsize=10)
        axs.set_xlabel('Error')
        axs.set_ylabel('Count')
        axs.set_title('Undo')
        axs.set_xticks(x_pos)
        axs.yaxis.grid(True)

        fig.set_figwidth(4)
        plt.show()
        fig.savefig(out_dir + 'mas_undo_p_puzzle.png', dpi=600, bbox_inches='tight')
        plt.close(fig)

