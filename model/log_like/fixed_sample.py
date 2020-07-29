from best_first_search import new_node_current,make_move,params
import time
import ast # to convert string to list
import numpy as np
import json
import pandas as pd

def fixed_basic(inparams,subject_data,basic_map,K = 1000): # K = fixed sampling number
    '''
        fixed_sampling
        sequential
        no repeat
        returns the log likelihood of current subject dataset
    '''
    start_time = time.time()
    # initialize parameters
    para = params(w1=inparams[0],w2=inparams[1],w3=inparams[2],
              stopping_probability=inparams[3],
              pruning_threshold=inparams[4],
              lapse_rate=inparams[5],
              feature_dropping_rate=inparams[6])	
    L = [0]*len(subject_data) # initialize log likelihood for each move in the dataset
    hits = [0]*len(subject_data) # initialize hits for each move
    
    for idx in range(len(subject_data)): # loop over all moves
        hit_check = [0]*K # initialize hit check
        dist = basic_map[0][subject_data.loc[idx,'map_id']]['distance']
        for k in range(K): 
            
            node_now = new_node_current(subject_data.loc[idx,'choice_all'],
                                ast.literal_eval(subject_data.loc[idx,'remain_all']), 
                                dist, subject_data.loc[idx,'budget_all'], subject_data.loc[idx,'n_city_all'], 
                                para.weights, n_u = subject_data.loc[idx,'n_u_all'])
            decision = make_move(node_now,dist,para)
            if (decision.name == subject_data.loc[idx,'choice_next_all']):
                hit_check[k] = 1
        
        hits[idx] = sum(hit_check)
        L[idx] = np.log((hits[idx]+1)/(K+1))
        print('idx='+str(idx)+',hits='+str(hits[idx])+',LL='+str(L[idx]))
        
    LL = sum(L)
    print('Final LL: '+str(LL)+', time lapse: '+str(time.time()-start_time))
    return LL, L, hits

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # directories
    home_dir = '/Users/sherrybao/Downloads/research/'
    input_dir = 'road_construction/rc_all_data/data/data_pilot_preprocessed/'
    output_dir = 'road_construction/rc_all_data/data/data_pilot_preprocessed/'
    map_dir = 'road_construction/map/active_map/'
    
    inparams = [1, 1, 1, 0.01, 15, 0.05, 0.1]

    with open(home_dir + map_dir + 'basic_map_48_all4','r') as file:
        basic_map = json.load(file) 
    
    subs = [1,2,4]#,2,4] # subject index 
    
    for sub in subs:
        sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_'+str(sub) + '.csv')
        
        LL, L, hits = fixed_basic(inparams,sub_data,basic_map)       
            
        with open(home_dir + output_dir + 'fixed_LL_3_' + str(sub),'w') as file: 
            json.dump((LL,L,hits),file)
