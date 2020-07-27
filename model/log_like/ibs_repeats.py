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

if __name__ == "__main__":
    # directories
    home_dir = '/Users/Toby/Downloads/bao/'
    input_dir = 'road_construction/rc_all_data/data/data_pilot_preprocessed/'
    map_dir = 'road_construction/map/active_map/'
    
    with open(home_dir + map_dir + 'basic_map_48_all4','r') as file:
        basic_map = json.load(file) 
        
    subs = [1]#,2,4] # subject index 
    
    eng = matlab.engine.start_matlab()
    
    for sub in subs:
        sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_'+str(sub) + '.csv')
     
        with open(home_dir + input_dir + 'n_repeat_' + str(sub),'r') as file:
            repeats = json.load(file) 
        
        result = eng.bads("@bads_ll", x0,lb,ub,plb,pub,nargout=2) # BADS will call bads_ll.m to get neg LL
        	
        print('result[0]', result[0])
        print('result[1]', result[1])
        
    eng.quit()