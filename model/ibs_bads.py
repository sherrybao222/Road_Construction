from best_first_search import new_node_current,make_move,params
import time
import ast # to convert string to list

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
    for idx in range(len(subject_data)): # loop over all moves
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
                print('move_id: '+str(idx)+', iteration: '+str(K))
                
                node_now = new_node_current(subject_data.loc[idx,'choice_all'],
                        ast.literal_eval(subject_data.loc[idx,'remain_all']), 
                        dist, subject_data.loc[idx,'budget_all'], subject_data.loc[idx,'n_city_all'], 
                        para.weights, n_u = subject_data.loc[idx,'n_u_all'])
                decision = make_move(node_now,dist,para)
            
            L[idx] += -harmonic_sum(K) # sum of repeats
        L[idx] = L[idx]/repeats[idx]
        
        print('LL for '+ str(idx)+': '+str(L[idx]))
        
    LL = sum(L)
    print('Final LL: '+str(LL)+', time lapse: '+str(time.time()-start_time))
    return -LL # return negative ll

