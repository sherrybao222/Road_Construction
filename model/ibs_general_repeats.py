from ibs_basics import ibs_early_stopping
import json
import pandas as pd
import numpy as np

def ibs_grepeats(inparams, LL_lower, sub_data,basic_map,repeats):
    '''
        ibs with early stopping
        sequential
        with non-trial-dependent repeated sampling defined by a general repeat number
        returns the log likelihood of current subject dataset
    '''
    nll_single_r = [] # nll for single repeat of a single run
    
    for r in range(repeats):
        nLL = ibs_early_stopping(inparams, LL_lower, sub_data,basic_map)
        nll_single_r.append(nLL)
        # print('subject:'+str(sub)+',run:'+str(n)+',repeat:'+str(r)+',ll:'+str(nLL))

    nll_avg = sum(nll_single_r)/repeats 
    return -nll_avg
           

if __name__ == "__main__":
    # directories
    home_dir = '/Users/dbao/google_drive/'
    input_dir = 'road_construction/experiments/pilot_0320/data/data_pilot_preprocessed/'
    output_dir = 'road_construction/experiments/pilot_0320/data/data_pilot_preprocessed/ll/'
    map_dir = 'road_construction/experiments/pilot_0320/map/active_map/'  
    
    # set parameters
    inparams = [1, 1, 1, 0.01, 15, 0.05, 0.1]

    with open(home_dir + map_dir + 'basic_map_48_all4','r') as file:
        basic_map = json.load(file) 
        
    subs = [1,2,4] # subject index 
    n_run = 5 # number of runs
    repeats = 30
    
    nll_all = [] # nll for all runs and subjects
    
    for sub in subs:
        nll_all_r = [] # nll for all runs
        
        sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_'+str(sub) + '.csv')
        LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
        
        for n in range(n_run): 
            nll_avg = ibs_grepeats(inparams, LL_lower, sub_data,basic_map,repeats)            
            
            with open(home_dir + output_dir + 'grepeat_ibs_LL_r'+str(repeats)+'_n'+
                      str(n)+'_s' + str(sub),'w') as file: 
                json.dump({'total ll':nll_avg},file,indent=4)
            
            nll_all_r.append(nll_avg)
    
        nll_all.append(nll_all_r)
    
    nll_std = list(np.std(nll_all,axis=1))
    
    with open(home_dir + output_dir + 'std_ibs_r'+str(repeats),'w') as file: 
        json.dump({'all nll':nll_all,
               'std':nll_std},file,indent=4)
