from llh_ibs_basics_dk5 import ibs_early_stopping, ibs_early_stopping_test, ibs_basic_test, ibs_early_stopping_para_trials
import json
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
import time

# TODO: now its turn to make

def run_ibs_early_stopping(inparams, LL_lower, sub_data, basic_map,r):
    nLL = ibs_early_stopping(inparams, LL_lower, sub_data, basic_map)
    return nLL
def run_ibs_early_stopping_test(inparams, LL_lower, sub_data, basic_map, r,value_func):
    nLL = ibs_early_stopping_test(inparams, LL_lower, sub_data, basic_map,value_func=value_func)
    return nLL

def run_ibs_basic_test(inparams, sub_data, basic_map, r,value_func):
    nLL = ibs_basic_test(inparams, sub_data, basic_map, value_func=value_func)
    return nLL

def ibs_grepeats(inparams, LL_lower, sub_data,basic_map,repeats):
    '''
        ibs with early stopping
        sequential
        with non-trial-dependent repeated sampling defined by a general repeat number
        returns the log likelihood of current subject dataset
    '''
    nll_single_r = [] # nll for single repeat of a single run

    for r in range(repeats):
        nLL,time_sequence,count_iteration = ibs_early_stopping(inparams, LL_lower, sub_data,basic_map)
        nll_single_r.append(nLL)
        # print('subject:'+str(sub)+',run:'+str(n)+',repeat:'+str(r)+',ll:'+str(nLL))

    nll_avg = -sum(nll_single_r)/repeats
    return nll_avg#,time_sequence,count_iteration

def ibs_grepeats_para_trials_local(para, LL_lower, sub_data, basic_map, repeats, value_func='legacy'):
    '''
        ibs with early stopping
        sequential
        with non-trial-dependent repeated sampling defined by a general repeat number
        returns the log likelihood of current subject dataset
    '''
    nll_single_r = []  # nll for single repeat of a single run

    start_time = time.time()
    for r in range(repeats):

        nLL, time_sequence, count_iteration = ibs_early_stopping_para_trials(para, LL_lower, sub_data,
                                                                      basic_map,value_func=value_func)
        # nLL, time_sequence, count_iteration = ibs_early_stopping_test(para, LL_lower, sub_data,
        #                                                               basic_map,value_func=value_func)


        nll_single_r.append(nLL)
        # print('subject:'+str(sub)+',run:'+str(n)+',repeat:'+str(r)+',ll:'+str(nLL))

    print('###### IBS grepeats time lapse ' + str(time.time() - start_time))
    nll_avg = -sum(nll_single_r) / repeats
    return nll_avg  # ,time_sequence,count_iteration

def ibs_grepeats_test_local(param, LL_lower, sub_data, basic_map, repeats, value_func='legacy'):
    '''
        ibs with early stopping
        sequential
        with non-trial-dependent repeated sampling defined by a general repeat number
        returns the log likelihood of current subject dataset
    '''
    nll_single_r = []  # nll for single repeat of a single run

    for r in range(repeats):
        nLL, time_sequence, count_iteration = ibs_early_stopping_test(param, LL_lower, sub_data,
                                                                      basic_map,value_func=value_func)
        nll_single_r.append(nLL)
        # print('subject:'+str(sub)+',run:'+str(n)+',repeat:'+str(r)+',ll:'+str(nLL))

    nll_avg = -sum(nll_single_r) / repeats
    return nll_avg  # ,time_sequence,count_iteration

def ibs_grepeats_basic_test_local(inparams, sub_data, basic_map, repeats, value_func='legacy'):
    '''
        ibs with early stopping
        sequential
        with non-trial-dependent repeated sampling defined by a general repeat number
        returns the log likelihood of current subject dataset
    '''
    nll_single_r = []  # nll for single repeat of a single run

    for r in range(repeats):
        LL, L = ibs_basic_test(inparams, sub_data, basic_map,value_func=value_func)
        nll_single_r.append(LL)
        # print('subject:'+str(sub)+',run:'+str(n)+',repeat:'+str(r)+',ll:'+str(nLL))

    nll_avg = -sum(nll_single_r) / repeats
    return nll_avg  # ,time_sequence,count_iteration

def ibs_grepeats_test(inparams, LL_lower, sub_data, basic_map, repeats, value_func='legacy'):
    '''
        ibs with early stopping
        sequential
        with non-trial-dependent repeated sampling defined by a general repeat number
        returns the log likelihood of current subject dataset
    '''
    nll_single_r = []  # nll for single repeat of a single run

    a_pool = multiprocessing.Pool(10)
    func = partial(run_ibs_early_stopping_test,inparams, LL_lower, sub_data, basic_map,value_func=value_func)
    nll_single_r = a_pool.map(func, range(repeats))
    a_pool.close()
    a_pool.join()

    # print(nll_single_r)
    nll_avg = -sum([nll_single_r[i][0] for i in range(len(nll_single_r))]) / repeats
    return nll_avg  # ,time_sequence,count_iteration

def ibs_basic_grepeats_test(inparams, sub_data, basic_map, repeats, value_func='legacy'):
    '''
        ibs with early stopping
        sequential
        with non-trial-dependent repeated sampling defined by a general repeat number
        returns the log likelihood of current subject dataset
    '''
    nll_single_r = []  # nll for single repeat of a single run

    a_pool = multiprocessing.Pool(10)
    func = partial(run_ibs_basic_test,inparams, sub_data, basic_map,value_func=value_func)
    nll_single_r = a_pool.map(func, range(repeats))
    a_pool.close()
    a_pool.join()

    # print(nll_single_r)
    nll_avg = -sum([nll_single_r[i][0] for i in range(len(nll_single_r))]) / repeats
    return nll_avg  # ,time_sequence,count_iteration

def ibs_grepeats_hpc(inparams, LL_lower, sub_data,basic_map,repeats):
    '''
        ibs with early stopping
        sequential
        with non-trial-dependent repeated sampling defined by a general repeat number
        returns the log likelihood of current subject dataset
    '''
    nll_single_r = [] # nll for single repeat of a single run


#    for r in range(repeats):
#        nLL = ibs_early_stopping(inparams, LL_lower, sub_data,basic_map)
#        nll_single_r.append(nLL)
        # print('subject:'+str(sub)+',run:'+str(n)+',repeat:'+str(r)+',ll:'+str(nLL))

    a_pool = multiprocessing.Pool(10)
    func = partial(run_ibs_early_stopping,inparams, LL_lower, sub_data, basic_map)
    nll_single_r = a_pool.map(func, range(repeats))
    a_pool.close()
    a_pool.join()

    nll_avg = -sum(nll_single_r)/repeats
    return nll_avg

if __name__ == "__main2__":

    # hpc directories
    home_dir = './'
    # input_dir = 'data_mod_ibs/'
    input_dir = 'data_online/'
    map_dir = 'active_map/'
    output_dir = 'll/'

    inparams = [0.01, 15, 0.05, 0.1, 15, 1, 1, 1]
    # inparams = [1, 1, 1, 0.01, 15, 0.05, 0.1] #[1, 1, 1, 0, 30, 1, 0] #before switching order


    with open(home_dir + map_dir + 'basicMap.json', 'r') as file:
        basic_map_ = json.load(file)

    with open(home_dir + map_dir + 'basic_map_48_all4', 'r') as file:
        basic_map = json.load(file)

    subs = [1, 2, 4]  # subject index
    # XX = [0, 0, 0, 0, 0.1, 0.01, 0]
    sub = 1
    sub_data_ = pd.read_csv(home_dir + 'data_mod_ibs/' + 'mod_ibs_preprocess_sub_' + str(int(sub)) + '.csv')
    # sub_data = pd.read_csv(home_dir + input_dir + 'preprocess4_sub_5e29dcdf1654ed023e89cc6a_2022-01-20_REWARD618.csv')

    from glob import glob
    flist = glob(home_dir + input_dir + 'preprocess4*.csv')
    sub_data = pd.read_csv(flist[int(1)-1])
    sub_data = sub_data[sub_data['condition'] == 'undo'].reset_index()
    # sub_data = sub_data[sub_data['condition'] == 'basic'].reset_index()

    # LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data_['n_u_all'])])
    LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if n != 0]) # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

    repeats = 2
    # inparams = XX
    print('inparams ' + str(inparams))
    # nll_avg = ibs_grepeats(inparams, LL_lower, sub_data, basic_map, repeats)

    # nll_avg = ibs_grepeats_test_local( inparams, LL_lower, sub_data, [basic_map_], repeats, value_func='legacy')
    nll_avg = ibs_grepeats_test_local( inparams, LL_lower, sub_data, [basic_map_], repeats, value_func='power_law_wp')


    print('nll ' + str(nll_avg))

if __name__ == "__main2__":

    # hpc directories
    home_dir = './'
    # input_dir = 'data_mod_ibs/'
    input_dir = 'data_online/'
    map_dir = 'active_map/'
    output_dir = 'll/'


    # inparams  = [0.2,15,.5, 0.1, 0.01, 0, 0.1,  0.1,  0.1,  0.1] # Lower bounds
    # inparams = [0.2, 15, 0.5, 0.1, 15, 1, 1, 1]
    # params_name = ['stopping_probability','pruning_threshold','lapse_rate',
    #         'feature_dropping_rate','undoing_threshold','w1','w2','w3']

    # inparams = [0.2, 15, 0.5, 0.1, 5, 1, 1, 1, 1, 1]
    # params_name = ['stopping_probability', 'pruning_threshold', 'lapse_rate',
    #               'feature_dropping_rate', 'undoing_threshold', 'undo_inverse_temparature', 'w1', 'w2', 'w3', 'w4']

    # inparams = [0.09765625, 5.644352679400763, 0.2781572722109066, 0.478759765625, 9.6484375,
    #             1.5771484375, 9.70703125, 9.5703125, 1]
    # params_name = ['stopping_probability','pruning_threshold','lapse_rate',
    #         'feature_dropping_rate','undo_inverse_temparature','w1','w2','w3','w4']

    # inparams = [
    #     0.4710693359375, 0.8572512183805009, 0.019081117703627897, 0.46612548828125, 4.94384765625, 8.939208984375, 9.82421875, 4.55810546875, 6.9873046875]
    # params_name = ['stopping_probability', 'pruning_threshold', 'lapse_rate',
    #               'feature_dropping_rate', 'undoing_threshold', 'undo_inverse_temparature', 'w1', 'w2', 'w3']

    inparams = [
        0.4710693359375, 0.8572512183805009, 0.019081117703627897, 0.46612548828125, 4.94384765625, 8.939208984375,
        9.82421875, 4.55810546875, 6.9873046875]
    params_name = ['stopping_probability', 'pruning_threshold', 'lapse_rate',
                  'feature_dropping_rate', 'undoing_threshold', 'w1', 'w2', 'w3'];

    with open(home_dir + map_dir + 'basicMap.json', 'r') as file:
        basic_map_ = json.load(file)

    # with open(home_dir + map_dir + 'basic_map_48_all4', 'r') as file:
    #     basic_map = json.load(file)

    subs = [1, 2, 4]  # subject index
    # XX = [0, 0, 0, 0, 0.1, 0.01, 0]
    sub = 1
    # sub_data_ = pd.read_csv(home_dir + 'data_mod_ibs/' + 'mod_ibs_preprocess_sub_' + str(int(sub)) + '.csv')
    # sub_data = pd.read_csv(home_dir + input_dir + 'preprocess4_sub_5e29dcdf1654ed023e89cc6a_2022-01-20_REWARD618.csv')

    from glob import glob
    flist = glob(home_dir + input_dir + 'preprocess4*.csv')
    sub_data = pd.read_csv(flist[int(6)-1])
    # sub_data = sub_data[sub_data['condition'] == 'undo'].reset_index()
    # sub_data = sub_data[sub_data['condition'] == 'basic'].reset_index()

    # LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data_['n_u_all'])])
    LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if n != 0]) # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

    repeats = 1
    # inparams = XX
    print('inparams ' + str(inparams))
    # nll_avg = ibs_grepeats(inparams, LL_lower, sub_data, basic_map, repeats)
    from bads_ibs import params_by_name

    para = params_by_name(inparams, params_name)



    # nll_avg = ibs_grepeats_para_trials_local( para, LL_lower, sub_data, [basic_map_], repeats, value_func='power_law_free_undo_softmax_undo')
    # nll_avg = ibs_grepeats_para_trials_local( para, LL_lower, sub_data, [basic_map_], repeats, value_func='power_law_free_undo_softmax_undo')

    start_time = time.time()
    nll_avg = ibs_grepeats_para_trials_local( para, LL_lower, sub_data, [basic_map_], repeats, value_func='legacy')

    print('###### IBS grepeats time lapse ' + str( (time.time() - start_time) *10))


    # nll_avg = ibs_grepeats_test_local( para, LL_lower, sub_data, [basic_map_], repeats, value_func='legacy')
    # nll_avg = ibs_grepeats_basic_test_local( inparams, sub_data, [basic_map_], repeats, value_func='legacy')

    # nll_avg = ibs_grepeats_test( inparams, LL_lower, sub_data, [basic_map_], repeats, value_func='power_law_wp')
    print('nll ' + str(nll_avg))


if __name__ == "__main__":

    # RANDOM


    # hpc directories
    home_dir = './'
    # input_dir = 'data_mod_ibs/'
    input_dir = 'data_online/'
    map_dir = 'active_map/'
    output_dir = 'll/'

    # inparams = [0.01, 0, 0,
    #             0, 10,
    #             1, 0, 0, 10, 20] # optimal agent ucb
    #
    # params_name = ['stopping_probability','pruning_threshold','lapse_rate',
    #         'feature_dropping_rate','undoing_threshold',
    #                'w1','w2','w3','w4','ucb_confidence']

    inparams = [0.01, 0, 0,
          0, 10,
          1, 0, 0, 10, 20, 0, 0]

    params_name = ['stopping_probability', 'pruning_threshold', 'lapse_rate',
                  'feature_dropping_rate', 'undoing_threshold',
                  'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'ucb_confidence']


    with open(home_dir + map_dir + 'basicMap.json', 'r') as file:
        basic_map_ = json.load(file)


    sub = 1

    from glob import glob
    flist = glob(home_dir + input_dir + 'preprocess4*.csv')
    sub_data = pd.read_csv(flist[int(6)-1])
    sub_data = pd.read_csv('./data_online/preprocess4_sub_6093db807d15df55e6990a28_2022-01-19_REWARD550.csv')
    LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if n != 0]) # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

    repeats = 1
    print('inparams ' + str(inparams))
    from utils import params_by_name

    para = params_by_name(inparams, params_name)

    start_time = time.time()
    nll_avg = ibs_grepeats_test_local( para, LL_lower, sub_data, [basic_map_], repeats, value_func='power_law_free')
    nll_avg = ibs_grepeats_para_trials_local( para, LL_lower, sub_data, [basic_map_], repeats, value_func='power_law2_free_budgetBias')

    print('###### IBS grepeats time lapse ' + str( (time.time() - start_time) *10))

    print('nll ' + str(nll_avg))