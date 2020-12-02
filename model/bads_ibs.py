from llh_ibs_general_repeats import ibs_grepeats
#from bads_prepare import sub_data,repeats,basic_map,LL_lower # import x and y data
import json
import pandas as pd
import numpy as np


# hpc directories
home_dir = '/home/db4058/road_construction/data/'
input_dir = 'data_pilot_preprocessed/'
map_dir = 'active_map/'

with open(home_dir + map_dir + 'basic_map_48_all4','r') as file:
    basic_map = json.load(file) 

# general repeats
repeats = 20

def ibs_interface(w1, w2, w3, w4, w5, w6, w7, sub):
     '''
     interface used to connect BADS and ibs_repeat
     '''
     sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_'+str(int(sub)) + '.csv')
     LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
    
     inparams = [w1, w2, w3, w4, w5, w6, w7]
     print('inparams '+str(inparams))
     nll_avg =ibs_grepeats(inparams,LL_lower,sub_data,basic_map,repeats)
 	
     return nll_avg#,time_sequence,count_iteration



if __name__ == "__main__":   
    nll, time_seq, n_sample = ibs_interface(1, 1, 1, 0.01, 15, 0.05, 0.1)
    
    with open(home_dir + 'check_time_sample','w') as file: 
        json.dump({'time sequence':time_seq,
               'number of samples':n_sample}, file)
    
    

