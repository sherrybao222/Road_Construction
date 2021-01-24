from model_greedy import new_node_current,make_move,params
import time
import pandas as pd
import numpy as np
import json
import ast # to convert string to list
from scipy import special
import math
from statistics import mean


def harmonic_sum(n):
	''' 
	return sum of harmonic series from 1 to n-1
	when n=1, return 0
	'''
	s = 0.0
	for i in range(1, n):
		s += 1.0/i
	return s

def ibs_basic(inparams,subject_data,basic_map):
    '''
        ibs without early stopping
        sequential
        no repeat
        returns the log likelihood of current subject dataset
    '''
    start_time = time.time()
    # initialize parameters
    para = params(w1=inparams[0],w2=inparams[1],w3=inparams[2],
              pruning_threshold=inparams[4],
              lapse_rate=inparams[5],
              feature_dropping_rate=inparams[6])	
    L = [0]*len(subject_data) # initialize log likelihood for each move in the dataset
    
    for idx in range(len(subject_data)): # loop over all moves
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
     
        print('move_id: '+str(idx)+', iteration: '+str(K))
        L[idx] = -harmonic_sum(K)
    LL = sum(L)
    print('Final LL: '+str(LL)+', time lapse: '+str(time.time()-start_time))
    return LL, L

def ibs_early_stopping(inparams, LL_lower, subject_data,basic_map):
    
    '''
    implement ibs with early stopping
    sequential
    returns the log likelihood of current subject
    '''
    start_time = time.time()
    time_sequence = [] # bfs time sequence
    
    # initialize parameters
    para = params(w1=inparams[0],w2=inparams[1],w3=inparams[2],
                  pruning_threshold=inparams[3],
                  lapse_rate=inparams[4],
                  feature_dropping_rate=inparams[5])	
	
    # initialize iteration data
    hit_target = [False]*len(subject_data) # true if hit for each move
    count_iteration = [1]*len(subject_data) # count of iteration for each move
    k = 0 # iteration number (the whole process / max of all trials)
    LL_k = 0 # total ll

	# iterate until meets early stopping criteria
    while hit_target.count(False) > 0:
#        iter_start_time = time.time()

        if LL_k	<= LL_lower:
            LL_k = LL_lower
            print('*********************** exceeds LL lower bound, break')
            break
                    
        LL_k = 0
        k += 1
#        print('Iteration k='+str(k), flush=True)
		
        for idx in range(len(subject_data)):
            
            if hit_target[idx]: # if current move was already hit by previous iterations
                LL_k += harmonic_sum(count_iteration[idx])
                continue # end the current idx and continue calculation for the next
                        
            dist = basic_map[0][subject_data.loc[idx,'map_id']]['distance']
            name = subject_data.loc[idx,'choice_all']
            remain = ast.literal_eval(subject_data.loc[idx,'remain_all'])
            budget_remain = subject_data.loc[idx,'budget_all']
            n_city = subject_data.loc[idx,'n_city_all']
            n_u = subject_data.loc[idx,'n_u_all']
            
            bfs_start_time = time.time() # record bfs start time
            
            node_now = new_node_current(name,
                                remain, 
                                dist, budget_remain, n_city, 
                                para.weights, n_u = n_u)
            decision = make_move(node_now,dist,para)
            
            move_time = (time.time() - bfs_start_time)
            time_sequence.append(move_time) # record bfs time
            
            if decision.name == subject_data.loc[idx,'choice_next_all']: # hit
                hit_target[idx] = True
                LL_k += harmonic_sum(count_iteration[idx])
            else: # not hit
                count_iteration[idx] += 1
                
        LL_k = -LL_k - (hit_target.count(False))*harmonic_sum(k)
#        print('\tKth LL_k '+str(LL_k), flush=True)
        
        # hit_number = hit_target.count(True)
        # print('\thit_target '+str(hit_number), flush=True)
        # print('\tmax_position '+str(count_iteration.index(max(count_iteration))))

        
        # print('IBS iter time lapse '+str(time.time() - iter_start_time), flush=True)


    print('IBS total time lapse '+str(time.time() - start_time))
#    print('Final LL_k: '+str(LL_k))
    return LL_k, time_sequence,count_iteration

def prep_compute_repeats(inparams, repeat, subject_data, basic_map):
    '''
        repeat ibs_basic
    '''   
#    start_time = time.time()
    L_repeat = [0]* repeat
    
    for r in range(repeat): 
        dumped, L_repeat[r] = ibs_basic(inparams,subject_data,basic_map)
        # cannot use early stopping because not all move has a calculated ll        
    
#    print('time lapse: '+str(time.time()-start_time))
    return L_repeat


def compute_repeats(budget_S, data_size, L_repeat):
    '''
        practical implementation of trial-dependent repeated ibs
        1. Choose a default parameter vector,
            and run IBS with a large numberzof repeats (e.g. R=100)
        2. Compute the optimal repeats R* given the estimated likelihood p^
            ane a total budget of expected samples S per likelihood evaluation,
            and round up.
        3. Return the computed repeat number for each trial.        
    '''               

    L = np.mean(L_repeat, axis=0) # LL average among repeats for each trial
    
    P = np.exp(L) # calculate "likelihood" probability from log likelihood of each trial
    # compute optimal repeat for each trial
    R = [0]*data_size # initialize computed number of repeats for each trial
    for i in range(data_size): # equation 39
        for j in range(data_size):
            R[i] += np.sqrt(special.spence(1-P[j])/P[j])
        R[i] = math.ceil(budget_S * (1/R[i]) * np.sqrt(P[i]*special.spence(1-P[i]))) # equation 39, round up
    print('computed repeats:' + str(R))
    return R
        

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # directories
    home_dir = '/Users/dbao/google_drive_db/'
    input_dir = 'road_construction/experiments/pilot_0320/data/data_pilot_preprocessed/'
    output_dir = 'road_construction/experiments/pilot_0320/data/data_pilot_preprocessed/ll/'
    map_dir = 'road_construction/experiments/pilot_0320/map/active_map/'
    
    inparams = [1, 1, 1, 0.01, 15, 0.05, 0.1] #[1, 1, 1, 0, 30, 1, 0]

    with open(home_dir + map_dir + 'basic_map_48_all4','r') as file:
        basic_map = json.load(file) 
    
    subs = [1,2,4] # subject index 
    
    for sub in subs:
        sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_'+str(sub) + '.csv')
        sub_size = len(sub_data)
    
# ===========================================================================
# calculate trial-dependent repeats and save
        # L_repeat = prep_compute_repeats(inparams, 100, sub_data, basic_map)
            
        # with open(home_dir + output_dir + 'L_repeat_' + str(sub),'w') as file: 
        #     json.dump((L_repeat), file)

        with open(home_dir + output_dir + 'L_repeat_' + str(sub),'r') as file:
            L_repeat = json.load(file) 
            
        budget = 30
        R = compute_repeats(sub_size*budget, sub_size, L_repeat)
            
#         with open(home_dir + output_dir + 'n_repeat_b' + str(budget) + 
#                   '_' + str(sub),'w') as file: 
#             json.dump(R, file)
# # ===========================================================================
# compare random sample number and sample number from ibs with a set of parameters
        # LL_lower = np.sum([np.log(1.0/n) for n in list(sub_data['n_u_all'])])
        # nLL, count_iteration = ibs_early_stopping(inparams, LL_lower, sub_data, basic_map)
        # count_random = list(sub_data['n_u_all'])
        
        # mean_count = mean(count_iteration)
        # sem = np.std(count_iteration)/math.sqrt(len(count_iteration))