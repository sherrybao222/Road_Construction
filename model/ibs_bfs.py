from best_first_search import new_node,make_move,params
import time

def harmonic_sum(n):
	''' 
	return sum of harmonic series from 1 to n-1
	when n=1, return 0
	'''
	s = 0.0
	for i in range(1, n):
		s += 1.0/i
	return s

def ibs_early_stopping(inparams,  
					puzzle_cache,
					LL_lower,
					subject_data,
					subject_answer, 
					subject_puzzle,
					threshold_num=100): 
    
    '''
	implement ibs with early stopping
	sequential
	returns the log likelihood of current subject
	'''
    
    start_time = time.time()
    
	# initialize parameters
    para = params(w1=inparams[0],w2=inparams[1],w3=inparams[2],
                  stopping_probability=inparams[3],
                  pruning_threshold=inparams[4],
                  lapse_rate=inparams[5],
                  feature_dropping_rate=inparams[6])
	
    # initialize iteration data
	hit_target = [False]*len(subject_data) # True if hit for each move
	count_iteration = [1]*len(subject_data) # count of iteration for each move
	k = 0
	LL_k = 0
	# previous_hit = 0
	# count_repeat = 0

	# iterate until meets early stopping criteria
	while hit_target.count(False) > 0: # while there is no hit
		if LL_k	<= LL_lower:
			LL_k = LL_lower
			print('*********************** exceeds LL lower bound, break')
			break
		# if count_repeat >= threshold_num:
			# LL_k = LL_lower
			# print('*********************** hit number stays same for '+str(threshold_num)+' iterations, break')
			# break
		LL_k = 0
		k += 1
		print('Iteration k='+str(k))
        
		for idx in range(len(subject_data)):
			if hit_target[idx]: # if current move was already hit by previous iterations
				LL_k += harmonic_sum(count_iteration[idx])
				continue
			decision = make_move(now,dist_city,para)
			if make_id(decision.board)==subject_answer[idx]: # hit
				hit_target[idx] = True
				LL_k += harmonic_sum(count_iteration[idx])
			else: # not hit
				count_iteration[idx] += 1
		LL_k = -LL_k - (hit_target.count(False))*harmonic_sum(k)
		hit_number = hit_target.count(True)
		# print('\thit_target '+str(hit_number)+', previous_hit '+str(previous_hit))
		print('\tKth LL_k '+str(LL_k))
		# if hit_number == previous_hit:
			# count_repeat += 1
			# print('count repeat increased to '+str(count_repeat)+' for hit number '+str(hit_number))
		# else:
			# count_repeat = 0
		# previous_hit = hit_number

	print('IBS total time lapse '+str(time.time() - start_time))
	print('Final LL_k: '+str(LL_k))
	return LL_k


