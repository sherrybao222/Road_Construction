from best_first_search import new_node_current,make_move,params
import time
import ast # to convert string to list
import json
import pandas as pd
import numpy as np

def harmonic_sum(n):
	''' 
	return sum of harmonic series from 1 to n-1
	when n=1, return 0
	'''
	s = 0.0
	for i in range(1, n):
		s += 1.0/i
	return s

def ibs_repeat(inparams,subject_data,repeats,basic_map):
    '''
        ibs without early stopping
        sequential
        with trial-dependent repeated sampling
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
    L_all = [] # ll for all calculation
    
    for idx in range(len(subject_data)): # loop over all moves
        L_repeat = [0]* repeats[idx]
        
        for r in range(repeats[idx]):
            K = 1
            
            dist = basic_map[0][subject_data.loc[idx,'map_id']]['distance']
            node_now = new_node_current(subject_data.loc[idx,'choice_all'],
                                ast.literal_eval(subject_data.loc[idx,'remain_all']), 
                                dist, subject_data.loc[idx,'budget_all'], subject_data.loc[idx,'n_city_all'], 
                                para.weights, n_u = subject_data.loc[idx,'n_u_all'])
            decision = make_move(node_now,dist,para)
            
            while not (decision.name == subject_data.loc[idx,'choice_next_all']):
                K += 1
                
                node_now = new_node_current(subject_data.loc[idx,'choice_all'],
                        ast.literal_eval(subject_data.loc[idx,'remain_all']), 
                        dist, subject_data.loc[idx,'budget_all'], subject_data.loc[idx,'n_city_all'], 
                        para.weights, n_u = subject_data.loc[idx,'n_u_all'])
                decision = make_move(node_now,dist,para)
            
            print('move_id: '+str(idx)+', iteration: '+str(K)+', repeat: '+str(r))
            L_repeat[r] = -harmonic_sum(K)
            
        L[idx] = sum(L_repeat)/repeats[idx]
        L_all.append(L_repeat)        
        print('LL for '+ str(idx)+': '+str(L[idx]))
        
    LL = sum(L)
    print('Final LL: '+str(LL)+', time lapse: '+str(time.time()-start_time))
    return -LL, L, L_all # return negative ll


if __name__ == "__main__":
    # directories
    home_dir = '/Users/sherrybao/google_drive/'
                #'/Users/Toby/Downloads/bao/'
    input_dir = 'road_construction/experiments/pilot_0320/data/data_pilot_preprocessed/'
    output_dir = 'road_construction/experiments/pilot_0320/data/data_pilot_preprocessed/ll/'
    map_dir = 'road_construction/experiments/pilot_0320/map/active_map/'  
    
    inparams = [1, 1, 1, 0.01, 15, 0.05, 0.1]

    with open(home_dir + map_dir + 'basic_map_48_all4','r') as file:
        basic_map = json.load(file) 
        
    subs = [1,2,4] # subject index 
    n_run = 1 # number of repeated runs
    budget = 100
    
    nll_all = [] # nll for all repeats and subjects
    
    for sub in subs:
        nll_all_r = [] # nll for all repeats

        sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_'+str(sub) + '.csv')

        with open(home_dir + output_dir + 'n_repeat_b' + str(budget) + 
                  '_' + str(sub),'r') as file:
            repeats = json.load(file) 
        
        for r in range(n_run):
            
            nLL, L, L_all = ibs_repeat(inparams,sub_data,repeats,basic_map)
            nll_all_r.append(nLL)	
            
            with open(home_dir + output_dir + 'repeat_ibs_LL_b'+str(budget)+'_r'+
                      str(r)+'_' + str(sub),'w') as file: 
                json.dump({'total ll':-nLL,
                           'move ll':L,
                           'repeat move ll':L_all},file,indent=4)
    
        nll_all.append(nll_all_r)
    
    nll_std = list(np.std(nll_all,axis=1))
    
    with open(home_dir + output_dir + 'std_ibs_b'+str(budget),'w') as file: 
        json.dump({'all nll':nll_all,
               'std':nll_std},file,indent=4)

