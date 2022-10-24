from llh_ibs_general_repeats_dk5 import ibs_grepeats, ibs_grepeats_test, ibs_grepeats_test_local, ibs_grepeats_basic_test_local, ibs_basic_grepeats_test, ibs_early_stopping_para_trials, ibs_grepeats_para_trials_local #from bads_prepare import sub_data,repeats,basic_map,LL_lower # import x and y data
import json
import pandas as pd
import numpy as np
from glob import glob
# from model_dk_ucb4 import params


# hpc directories
home_dir = './'
# input_dir = 'data_mod_ibs/'
input_dir = 'data_online/'
map_dir = 'active_map/'

# with open(home_dir + map_dir + 'basic_map_48_all4','r') as file:
#     basic_map = json.load(file)
# with open(home_dir + map_dir + 'basicMap.json','r') as file:
#     basic_map = json.load(file)

with open(home_dir + map_dir + 'basicMap.json', 'r') as file:
   basic_map_ = json.load(file)

basic_map = [basic_map_]

# general repeats
repeats = 1
class params_by_name:
    def __init__(self, values, names, count_par = 10,
                    **kwargs
                    ):
        self.weights = []
        i_weights = 1
        for key, value in zip(names, values):
            if 'w' in key:
                if key.index('w') == 0:
                    self.weights.append(value)
                    i_weights+=1
            if key == "undoing_threshold":
               self.undoing_threshold = value
            elif key == "undo_inverse_temparature":
               self.undo_inverse_temparature = value
            elif key == "feature_dropping_rate":
               self.feature_dropping_rate = value
            elif key == "stopping_probability":
               self.stopping_probability = value
            elif key == "pruning_threshold":
               self.pruning_threshold = value
            elif key == "lapse_rate":
               self.lapse_rate = value
            elif key == "ucb_confidence":
               self.ucb_confidence = value

            elif key == "undoing_threshold_wu":
               self.undoing_threshold_wu = value
            elif key == "undo_inverse_temparature_wu":
                self.undo_inverse_temparature_wu = value
            elif key == "feature_dropping_rate_wu":
                self.feature_dropping_rate_wu = value
            elif key == "stopping_probability_wu":
                self.stopping_probability_wu = value
            elif key == "pruning_threshold_wu":
                self.pruning_threshold_wu = value
            elif key == "lapse_rate_wu":
                self.lapse_rate_wu = value
            elif key == "ucb_confidence_wu":
                self.ucb_confidence_wu = value
        self.count_par = count_par
        self.i_th = None
        self.visits = None




def ibs_interface(w1, w2, w3, w4, w5, w6, w7, sub):
     '''
     interface used to connect BADS and ibs_repeat
     '''
     input_dir = 'data_mod_ibs/'
     sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_'+str(int(sub)) + '.csv')
     LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
    
     inparams = [w1, w2, w3, w4, w5, w6, w7]
     print('inparams '+str(inparams))
     nll_avg =ibs_grepeats(inparams,LL_lower,sub_data,basic_map,repeats)
 	
     return nll_avg#,time_sequence,count_iteration

def ibs_interface_test(w, value_func, f_name):
     '''
     interface used to connect BADS and ibs_repeat
     '''
     sub_data = pd.read_csv(f_name)
     # LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
     LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if
                        n != 0])  # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

     inparams = [i for i in w]
     print('inparams '+str(inparams))
     nll_avg = ibs_grepeats_test(inparams,LL_lower,sub_data,basic_map,int(repeats), value_func)

     return nll_avg#,time_sequence,count_iteration


def ibs_basic_interface_test(w, value_func, f_name):
     '''
     interface used to connect BADS and ibs_repeat
     '''
     sub_data = pd.read_csv(f_name)
     # LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
     LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if
                        n != 0])  # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

     inparams = [i for i in w]
     print('inparams '+str(inparams))
     nll_avg = ibs_grepeats_test(inparams,LL_lower,sub_data,basic_map,int(repeats), value_func)

     return nll_avg#,time_sequence,count_iteration

def ibs_basic_interface_test_undo(w, value_func, f_name):
     '''
     interface used to connect BADS and ibs_repeat
     '''
     sub_data = pd.read_csv(f_name)
     sub_data = sub_data[sub_data['condition'] == 'undo'].reset_index()
     # LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
     LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if
                        n != 0])  # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

     # inparams = [w1, w2, w3, w4, w5, w6, w7]
     inparams = [i for i in w]
     print('inparams '+str(inparams))
     # nll_avg =ibs_grepeats_test(inparams,LL_lower,sub_data,basic_map, int(repeats), value_func)
     nll_avg =ibs_basic_grepeats_test(inparams,sub_data,basic_map, int(repeats), value_func)

     return nll_avg#,time_sequence,count_iteration


def ibs_interface_test_undo(w, value_func, f_name):
     '''
     interface used to connect BADS and ibs_repeat
     '''
     sub_data = pd.read_csv(f_name)
     sub_data = sub_data[sub_data['condition'] == 'undo'].reset_index()
     # LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
     LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if
                        n != 0])  # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

     # inparams = [w1, w2, w3, w4, w5, w6, w7]
     inparams = [i for i in w]
     print('inparams '+str(inparams))
     # nll_avg =ibs_grepeats_test(inparams,LL_lower,sub_data,basic_map, int(repeats), value_func)
     nll_avg =ibs_grepeats_test(inparams,LL_lower,sub_data,basic_map, int(repeats), value_func)

     return nll_avg#,time_sequence,count_iteration

def ibs_interface_test_basic(w, value_func, f_name):
     '''
     interface used to connect BADS and ibs_repeat
     '''
     sub_data = pd.read_csv(f_name)
     sub_data = sub_data[sub_data['condition'] == 'basic'].reset_index()
     # LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
     LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if
                        n != 0])  # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

     # inparams = [w1, w2, w3, w4, w5, w6, w7]
     inparams = [i for i in w]
     print('inparams '+str(inparams))
     nll_avg =ibs_grepeats_test(inparams,LL_lower,sub_data,basic_map,int(repeats), value_func)
     # nll_avg =ibs_grepeats_test_local(inparams,LL_lower,sub_data,basic_map,repeats, value_func)

     return nll_avg#,time_sequence,count_iteration

def ibs_interface_para_trial(w,param_name, value_func, f_name):
     '''
     interface used to connect BADS and ibs_repeat
     '''

     print(param_name)
     sub_data = pd.read_csv(f_name)
     # LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
     LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if
                        n != 0])  # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

     # inparams = [w1, w2, w3, w4, w5, w6, w7]
     inparams = [i for i in w]
     print('inparams ' + str(inparams))

     para = params_by_name(w, param_name)

     nll_avg = ibs_grepeats_para_trials_local(para, LL_lower, sub_data, basic_map, int(repeats), value_func)
     # nll_avg =ibs_grepeats_test_local(inparams,LL_lower,sub_data,basic_map,repeats, value_func)

     return nll_avg#,time_sequence,count_iteration

def ibs_interface_para_trial_dbg(w,param_name, value_func, f_name):
     '''
     interface used to connect BADS and ibs_repeat
     '''
     print(param_name)
     sub_data = pd.read_csv(f_name)
     # LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
     LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if
                        n != 0])  # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

     # inparams = [w1, w2, w3, w4, w5, w6, w7]
     inparams = [i for i in w]
     print('inparams ' + str(inparams))

     para = params_by_name(w, param_name)

     # nll_avg = ibs_grepeats_para_trials_local(para, LL_lower, sub_data, basic_map, int(repeats), value_func)
     nll_avg =ibs_grepeats_test_local(para,LL_lower,sub_data,basic_map, int(repeats), value_func)

     return nll_avg#,time_sequence,count_iteration

def ibs_interface_para_trial_undo(w,param_name, value_func, f_name):
     '''
     interface used to connect BADS and ibs_repeat
     '''
     sub_data = pd.read_csv(f_name)
     sub_data = sub_data[sub_data['condition'] == 'undo'].reset_index()
     # LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
     LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if
                        n != 0])  # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

     # inparams = [w1, w2, w3, w4, w5, w6, w7]
     inparams = [i for i in w]
     print('inparams ' + str(inparams))

     para = params_by_name(w, param_name)

     nll_avg = ibs_grepeats_para_trials_local(para, LL_lower, sub_data, basic_map, int(repeats), value_func)
     # nll_avg =ibs_grepeats_test_local(inparams,LL_lower,sub_data,basic_map,repeats, value_func)

     return nll_avg#,time_sequence,count_iteration


def ibs_interface_para_trial_basic(w, param_name, value_func, f_name):
    '''
    interface used to connect BADS and ibs_repeat
    '''
    sub_data = pd.read_csv(f_name)
    sub_data = sub_data[sub_data['condition'] == 'basic'].reset_index()
    # LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
    LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if
                n != 0])  # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

    # inparams = [w1, w2, w3, w4, w5, w6, w7]
    inparams = [i for i in w]
    print('inparams ' + str(inparams))

    para = params_by_name(w, param_name)

    nll_avg = ibs_grepeats_para_trials_local(para, LL_lower, sub_data, basic_map, int(repeats), value_func)
    # nll_avg =ibs_grepeats_test_local(inparams,LL_lower,sub_data,basic_map,repeats, value_func)

    return nll_avg#,time_sequence,count_iteration

if __name__ == "__main__":   
    nll, time_seq, n_sample = ibs_interface_test_undo([1, 1, 1, 0.01, 15, 0.05, 0.1], 1, 'legacy ')
    
    with open(home_dir + 'check_time_sample','w') as file: 
        json.dump({'time sequence':time_seq,
               'number of samples':n_sample}, file)
    
    

