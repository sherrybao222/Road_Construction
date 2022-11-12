from utils import params_by_name
from gen_trials_based_on_trials import gen_trials_based_on_trials
import time
import datetime

import json
import os
import pickle

import pandas as pd

def generate_sim_paramsets(inparams, save_total = False, # save with video or not
                             save_dir_name = '05042022_',
                             name_model = 'power_law_free',
                             save_filename='gen',
                             params_name = ['stopping_probability','pruning_threshold','lapse_rate',
                            'feature_dropping_rate','undoing_threshold',
                            'w1','w2','w3','w4','ucb_confidence']):
    
    start_time = time.time()
    
    ## directories -----------------------------------------------------------------------
    # hpc directories
    home_dir = './'
    # input_dir = 'data_mod_ibs/'
    input_dir = 'data_online/'
    map_dir = 'active_map/'
    output_dir = 'll/'
    save_dir_name = save_dir_name + name_model
    dir_param = save_dir_name + '/' + save_filename + '/'
    if not os.path.exists(dir_param):
        os.makedirs(dir_param)

    ## load data --------------------------------------------------------------------------
    with open(home_dir + map_dir + 'basicMap.json', 'r') as file:
        basic_map_ = json.load(file)
    with open(home_dir + map_dir + 'tree/map_tree', 'r') as file:
        map_tree = json.load(file)

    sub_data = pd.read_csv(home_dir + input_dir + 'preprocess4_sub_60da65c630ff4389c297b03c_RC-phaser6_2022-01-20_21h02.32.215.csv') #sherry: just use it as a data structure 
    # use only basic trials
    sub_data = sub_data.iloc[sub_data.index[sub_data.condition=='basic']]
    sub_data.reset_index()

    # LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if n != 0]) # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

    repeats = 1

    gi = 1 # initialize
    gen_num = len(inparams) # number of all parameters
    while True:
        gdt = datetime.datetime.now().strftime("%Y%m%d%H%M%S") # time tag
        
        print('inparams ' + str(inparams[gi-1]))
        try:
            inparams_ = inparams[gi-1].tolist()
        except:
            inparams_ = inparams[gi-1].copy()
        inparams_copied = inparams_

        para = params_by_name(inparams_copied, params_name, count_par=int(10))

        with open(dir_param + '/params{}.json'.format(gdt), 'w') as file:
            json.dump({'inparams':inparams_copied, 'params_name':params_name},file)

        out_sub, k_hist= gen_trials_based_on_trials(para, LL_lower, sub_data, [basic_map_, map_tree],
                                        value_func='power_law_free_linear', save_total=save_total, dir_param = dir_param, save_count= 0, vis=True)


        mas_s = [map_tree[i]['mas'][0] for i in range(len(map_tree))]

        submitted = out_sub[out_sub.submit == 1]
        undo_submitted = submitted[submitted.condition == 'undo'].sort_values(by=['map_id'])
        basic_submitted = submitted[submitted.condition == 'basic'].sort_values(by=['map_id'])

        out_sub.to_csv(dir_param + '/'+ save_filename +'_{}.csv'.format(gdt))
        pd.DataFrame(k_hist).to_csv(dir_param + '/khist_{}.csv'.format(gdt))
        print('######')
        print(k_hist)
        print("FINAL MEAN:{}".format(np.mean(k_hist)))

        gi+=1
        if gi > gen_num:
            break

    print('###### time lapse ' + str( (time.time() - start_time) *10))
