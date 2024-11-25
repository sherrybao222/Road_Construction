from  model_dk_ucb5 import new_node_current, new_node_current_seq, initial_node_saving, make_move, make_move_undo, make_move_weights, make_move_undo_weights, params
import time
import pandas as pd
import numpy as np
import json
import ast # to convert string to list
from scipy import special
import math
from statistics import mean
import multiprocessing
from functools import partial

repeats_para_trial = 1
repeats = 100

def harmonic_sum(n):
    '''
    return sum of harmonic series from 1 to n-1
    when n=1, return 0
    '''
    s = 0.0
    for i in range(1, n):
        s += 1.0/i
    return s

# OUTDATED - DK
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
              stopping_probability=inparams[3],
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

# OUTDATED - DK
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
                  stopping_probability=inparams[3],
                  pruning_threshold=inparams[4],
                  lapse_rate=inparams[5],
                  feature_dropping_rate=inparams[6])

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
            # print(subject_data)
            # print(basic_map)
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

    print('IBS total time lapse '+str(time.time() - start_time))
#    print('Final LL_k: '+str(LL_k))
    return LL_k, time_sequence,count_iteration


def ibs_basic_test(inparams, subject_data, basic_map, value_func='legacy'):
    '''
        ibs without early stopping
        sequential
        no repeat
        returns the log likelihood of current subject dataset
    '''
    start_time = time.time()
    # initialize parameters
    if len(inparams) == 8:
        para = params(w1=inparams[5],w2=inparams[6],w3=inparams[7],
                      stopping_probability=inparams[0],
                      pruning_threshold=inparams[1],
                      lapse_rate=inparams[2],
                      feature_dropping_rate=inparams[3],
                      undoing_threshold=inparams[4])
    elif len(inparams) == 9:
        para = params(w1=inparams[5],w2=inparams[6],w3=inparams[7],w4=inparams[8],
                      stopping_probability=inparams[0],
                      pruning_threshold=inparams[1],
                      lapse_rate=inparams[2],
                      feature_dropping_rate=inparams[3],
                      undoing_threshold=inparams[4])

    L = [0]*len(subject_data) # initialize log likelihood for each move in the dataset
    index1 = subject_data[subject_data['submit'] == 1].index

    for id in index1:
        L[id] = -harmonic_sum(1)
        L[id-1] = -harmonic_sum(1)

    bfs_start_time = time.time()  # record bfs start time

    prev_trial_id = -1
    for idx in range(len(subject_data)): # loop over all moves
        K=1

        curr_trial_id = subject_data.loc[idx, 'trial_id']

        if subject_data.loc[idx, 'submit'] != 1:  # for those are not submit
            exempt_decision = 0  # exemption of making moves while undoing.

            # if hit_target[idx]:  # if current move was already hit by previous iterations
            #     LL_k += harmonic_sum(count_iteration[idx])
            #     continue_later = True
            #     # continue # end the current idx and continue calculation for the next

            dist = basic_map[0][subject_data.loc[idx, 'map_id']]['distance']
            name = subject_data.loc[idx, 'currentChoice']

            # remain = ast.literal_eval(subject_data_.loc[idx,'remain_all'])
            # remain for cities with in reach and
            cities_reach = ast.literal_eval(subject_data.loc[idx, 'cities_reach'])
            cities_taken = ast.literal_eval(subject_data.loc[idx, 'chosen_all'])
            cities_taken = np.setdiff1d(cities_taken, name).tolist()  # exclude current location
            remain = np.union1d(cities_reach, cities_taken).tolist()

            # budget_remain = subject_data_.loc[idx,'budget_all']
            budget_remain = subject_data.loc[idx, 'currentBudget']

            # n_city = subject_data_.loc[idx,'n_city_all']
            # n_u = subject_data_.loc[idx,'n_u_all']
            n_city = subject_data.loc[idx, 'n_city_all']
            n_u = subject_data.loc[idx, 'n_within_reach']

            if curr_trial_id != prev_trial_id:
                node_now = new_node_current(name,
                                            cities_reach, cities_taken,
                                            dist, budget_remain, n_city,
                                            para.weights, n_u=n_u, value_func=value_func)
            else:
                node_now = new_node_current_seq(node_now, name,
                                                cities_reach, cities_taken,
                                                dist, budget_remain, n_city,
                                                para.weights, n_u=n_u, value_func=value_func)
            done = True
            while done and (subject_data.loc[idx+1, 'submit'] != 1):

                if subject_data.loc[idx, 'condition'] == 'undo':
                    decision = make_move_undo(node_now, dist, para, value_func=value_func)
                else:
                    decision = make_move(node_now, dist, para, value_func=value_func)

                # decide whether want to save this current decision because participants are doing series of undos
                if subject_data.loc[idx + 1, 'undoIndicator'] == 1:
                    # If it is the case, then save the decision.
                    # Or even better way is find to when they are undoing.
                    ix = 0
                    while subject_data.loc[idx + 1 + ix, 'undoIndicator']:
                        exempt_decision += 1
                        ix += 1
                    # If it is the case, then save the next decision.
                    decision_undo = decision  # decision that made when people decided to undo

                # set up the next choice
                # undoing condition
                if exempt_decision == 0:
                    # for the submit and the last choice: just ignore
                    if idx + 1 < len(subject_data):  # not the final one
                        # and especially for those are not submitting
                        if subject_data.loc[idx + 1, 'submit'] != 1:
                            if decision.name == subject_data.loc[idx + 1, 'currentChoice']:  # hit
                                done = False
                else:  # while they are undoing
                    exempt_decision -= 1
                    # for the submit and the last choice: just ignore
                    if idx + 1 < len(subject_data):  # not the final one
                        # and especially for those are not submitting
                        if subject_data.loc[idx + 1, 'submit'] != 1:
                            # if the decision is in the current node's undo-able list (means it is one of the visited cities)
                            if decision_undo.name in node_now.city_undo:  # hit (at least on the way)
                                done = False
                            else:  # not hit
                                # if it is not hit then replan
                                exempt_decision = 0
                                del decision_undo

                K += 1
        prev_trial_id = curr_trial_id

        print('move_id: '+str(idx)+', iteration: '+str(K))
        L[idx] = -harmonic_sum(K)

    LL = sum(L)
    print('Final LL: '+str(LL)+', time lapse: '+str(time.time()-start_time))
    return LL, L


def ibs_early_stopping_test(para, LL_lower, subject_data, basic_map, value_func = 'legacy'):

    '''
    implement ibs with early stopping
    sequential
    returns the log likelihood of current subject
    '''


    '''
    Process subject data to only take account the undo target not every undo presses.
    '''

    undo_idxx = subject_data.index[subject_data.undoIndicator == 1]


    start_time = time.time()
    time_sequence = [] # bfs time sequence


    # initialize iteration data
    debug_target = [False]*len(subject_data) # true if hit for each move
    hit_target = [False]*len(subject_data) # true if hit for each move
    index1 = subject_data[subject_data['submit'] == 1].index
    for id in index1:
        debug_target[id] = True
        debug_target[id-1] = True

        hit_target[id] = True
        hit_target[id-1] = True


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

        prev_trial_id = -1
        prev_condition = ''
        from tqdm import tqdm
        ab = 0
        for idx in range(len(subject_data)):
            continue_later = False
            curr_trial_id = subject_data.loc[idx, 'trial_id']
            curr_condition =  subject_data.loc[idx, 'condition']

            if subject_data.loc[idx,'submit'] != 1: # for those are not submit
                exempt_decision = 0 # exemption of making moves while undoing.

                visits_ = np.zeros(30)
                idff = 0
                idxs_trial_id = subject_data.index[subject_data['trial_id'] == curr_trial_id].to_list()
                cts_trial_id = subject_data['currentChoice'].loc[idxs_trial_id].to_list()

                for iii in range(idxs_trial_id.index(idx) + 1):
                    visits_[cts_trial_id[iii]] += 1
                    idff += 1

                if hit_target[idx]: # if current move was already hit by previous iterations
                    LL_k += harmonic_sum(count_iteration[idx])
                    continue_later = True
                    # continue # end the current idx and continue calculation for the next
                # print(subject_data)
                # print(basic_map)
                # if subject_data.loc[idx,'trial_id'] != trial_id:
                # init_dat = initial_node_saving(idx, basic_map, subject_data)
                dist = basic_map[0][subject_data.loc[idx,'map_id']]['distance']
                name = subject_data.loc[idx,'currentChoice']

                # remain = ast.literal_eval(subject_data_.loc[idx,'remain_all'])
                # remain for cities with in reach and
                cities_reach = ast.literal_eval(subject_data.loc[idx,'cities_reach'])
                cities_taken = ast.literal_eval(subject_data.loc[idx,'chosen_all'])
                cities_taken = np.setdiff1d(cities_taken, name).tolist() # exclude current location
                remain = np.union1d(cities_reach, cities_taken).tolist()

                # budget_remain = subject_data_.loc[idx,'budget_all']
                budget_remain = subject_data.loc[idx,'currentBudget']

                # n_city = subject_data_.loc[idx,'n_city_all']
                # n_u = subject_data_.loc[idx,'n_u_all']
                n_city = subject_data.loc[idx,'n_city_all']
                n_u    = subject_data.loc[idx,'n_within_reach']

                # print(curr_trial_id)
                # print(curr_condition)
                # print(prev_condition)
                # print(idx)
                if (curr_trial_id != prev_trial_id)  or (curr_condition != prev_condition):
                    # if name != 0:
                    #     print('')
                    node_now = new_node_current(name,
                                        cities_reach, cities_taken,
                                        dist, budget_remain, n_city,
                                                para, n_u=n_u, value_func=value_func, i_th = idff, visits = visits_, condition=subject_data.loc[idx, 'condition'] )
                    # print(node_now)
                else:
                    node_now = new_node_current_seq(node_now,name,
                                        cities_reach, cities_taken,
                                        dist, budget_remain, n_city,
                                                para, n_u=n_u, value_func=value_func, i_th = idff, visits = visits_, condition=subject_data.loc[idx, 'condition'] )
                    # print(node_now)

                # print(continue_later)
                if not continue_later:
                    bfs_start_time = time.time() # record bfs start time
                    # np.random.seed(1)
                    if exempt_decision == 0:
                        if subject_data.loc[idx, 'condition'] == 'undo':
                            decision, ws = make_move_undo_weights(node_now, dist, para, value_func=value_func, i_th = idff+1, visits = visits_)
                        else:
                            decision, ws = make_move_weights(node_now, dist, para, value_func=value_func, i_th = idff+1, visits = visits_)
                    idff += 1

                    # decide whether want to save this current decision because participants are doing series of undos
                    # [07/05/22] Now changed to mark every undo presses as true hit iff the undo targets are matched.
                    if subject_data.loc[idx+1, 'undoIndicator'] == 1:
                        # [07/05/22] check whether the target is the same.
                        cities_along = []
                        ix = 0
                        while subject_data.loc[idx+1+ix, 'undoIndicator'] :
                            cities_along.append(subject_data['currentChoice'][idx+1+ix])
                            exempt_decision += 1
                            ix += 1
                        # preregistered hit
                        if cities_along[-1] == decision.name:
                            pre_hit = [True]*len(cities_along)
                        else:
                            pre_hit = [False]*len(cities_along)



                        # [previous version] the version takes account every undo button press
                        # If it is the case, then save the decision.
                        # Or even better way is find to when they are undoing.
                        # ix = 0
                        # while subject_data.loc[idx+1+ix, 'undoIndicator'] :
                        #     exempt_decision += 1
                        #     ix+=1
                        # If it is the case, then save the next decision.
                        # decision_undo = decision # decision that made when people decided to undo


                    move_time = (time.time() - bfs_start_time)
                    time_sequence.append(move_time) # record bfs time

                    # set up the next choice
                    # not undoing (connecting cities)
                    if exempt_decision == 0:
                        # for the submit and the last choice: just ignore
                        if idx+1 < len(subject_data): # not the final one
                            # and especially for those are not submitting
                            if subject_data.loc[idx+1, 'submit'] != 1:
                                ab += 1
                                debug_target[idx] = True
                                if decision.name == subject_data.loc[idx+1,'currentChoice']: # hit
                                    hit_target[idx] = True
                                    LL_k += harmonic_sum(count_iteration[idx])
                                else: # not hit
                                    count_iteration[idx] += 1
                    else: # while they are undoing
                        exempt_decision -= 1
                        # for the submit and the last choice: just ignore
                        if idx+1 < len(subject_data): # not the final one
                            # and especially for those are not submitting
                            if subject_data.loc[idx+1, 'submit'] != 1:
                                # if the decision is in the current node's undo-able list (means it is one of the visited cities)
                                ab += 1
                                # [07/05/2022]
                                debug_target[idx] = True # check whether every trial is visited
                                if pre_hit.pop(0): # hit
                                    hit_target[idx] = True
                                    LL_k += harmonic_sum(count_iteration[idx])
                                else: # not a hit
                                    # hit_target[idx] = False # its already false.
                                    count_iteration[idx] += 1

                                    # check
                                    if hit_target[idx] == True:
                                        print('what?')

                                # [previous version]
                                # debug_target[idx] = True
                                # if decision_undo.name in node_now.city_undo: # hit (at least on the way)
                                #     hit_target[idx] = True
                                #     LL_k += harmonic_sum(count_iteration[idx])
                                # else: # not hit
                                #     count_iteration[idx] += 1
                                #     # if it is not hit then replan
                                #     exempt_decision = 0
                                #     del decision_undo

            prev_trial_id = curr_trial_id
            prev_condition = curr_condition

        LL_k = -LL_k - (hit_target.count(False))*harmonic_sum(k)
        # print('\tKth LL_k '+str(LL_k), flush=True)

        hit_number = hit_target.count(True)
        # print('\thit_target '+str(hit_number) + ' / ' + str(len(hit_target)), flush=True)
        # print('\tmax_position '+str(count_iteration.index(max(count_iteration))))

        # print('IBS iter time lapse '+str(time.time() - iter_start_time), flush=True)

    print('IBS total time lapse '+str(time.time() - start_time))
    print('Final LL_k: '+str(LL_k))
    return LL_k, time_sequence,count_iteration

def run_para_trials(LL_k, para, hit_target, subject_data, basic_map, r, value_func = 'legacy', repeats_para_trial = 1):

    count_iteration = [0]*len(subject_data) # count of iteration for each move
    time_sequence = [] # bfs time sequence

    # initialize iteration data
    index1 = subject_data[subject_data['submit'] == 1].index
    k  = 0
    repeat_i = 0
    # iterate until meets early stopping criteria
    while repeat_i < repeats_para_trial:
        repeat_i += 1
        k += 1

        prev_trial_id = -1
        prev_condition = ''
        # print(len(subject_data))
        for idx in range(len(subject_data)):
            continue_later = False
            curr_trial_id = subject_data.loc[idx, 'trial_id']
            curr_condition =  subject_data.loc[idx, 'condition']

            if subject_data.loc[idx, 'submit'] != 1:  # for those are not submit
                exempt_decision = 0  # exemption of making moves while undoing.


                visits_ = np.zeros(30)
                idff = 0
                idxs_trial_id = subject_data.index[subject_data['trial_id'] == curr_trial_id].to_list()
                cts_trial_id = subject_data['currentChoice'].loc[idxs_trial_id].to_list()

                for iii in range(idxs_trial_id.index(idx) + 1):
                    visits_[cts_trial_id[iii]] += 1
                    idff += 1


                if hit_target[idx]:  # if current move was already hit by previous iterations
                    LL_k += harmonic_sum(count_iteration[idx])
                    continue_later = True
                    # continue # end the current idx and continue calculation for the next
                # if subject_data.loc[idx,'trial_id'] != trial_id:
                # init_dat = initial_node_saving(idx, basic_map, subject_data)
                dist = basic_map[0][subject_data.loc[idx, 'map_id']]['distance']
                name = subject_data.loc[idx, 'currentChoice']

                # remain = ast.literal_eval(subject_data_.loc[idx,'remain_all'])
                # remain for cities with in reach and
                cities_reach = ast.literal_eval(subject_data.loc[idx, 'cities_reach'])
                cities_taken = ast.literal_eval(subject_data.loc[idx, 'chosen_all'])
                cities_taken = np.setdiff1d(cities_taken, name).tolist()  # exclude current location
                remain = np.union1d(cities_reach, cities_taken).tolist()

                budget_remain = subject_data.loc[idx, 'currentBudget']

                n_city = subject_data.loc[idx, 'n_city_all']
                n_u = subject_data.loc[idx, 'n_within_reach']

                if (curr_trial_id != prev_trial_id) or (curr_condition != prev_condition):
                    node_now = new_node_current(name,
                                                cities_reach, cities_taken,
                                                dist, budget_remain, n_city,
                                                para, n_u=n_u, value_func=value_func, i_th = idff, visits = visits_, condition= subject_data.loc[idx, 'condition'] )
                else:
                    node_now = new_node_current_seq(node_now, name,
                                                    cities_reach, cities_taken,
                                                    dist, budget_remain, n_city,
                                                    para, n_u=n_u, value_func=value_func, i_th = idff, visits = visits_, condition= subject_data.loc[idx, 'condition'] )

                # print(continue_later)
                if not continue_later:
                    bfs_start_time = time.time()  # record bfs start time
                    # np.random.seed(1)
                    if exempt_decision == 0:
                        if subject_data.loc[idx, 'condition'] == 'undo':
                            decision, ws = make_move_undo_weights(node_now, dist, para, value_func=value_func, i_th = idff+1, visits = visits_)
                        else:
                            decision, ws = make_move_weights(node_now, dist, para, value_func=value_func, i_th = idff+1, visits = visits_)

                    idff += 1
                    # decide whether want to save this current decision because participants are doing series of undos
                    # [07/05/22] Now changed to mark every undo presses as true hit iff the undo targets are matched.
                    if subject_data.loc[idx + 1, 'undoIndicator'] == 1:
                        # [07/05/22] check whether the target is the same.
                        cities_along = []
                        ix = 0
                        while subject_data.loc[idx+1+ix, 'undoIndicator'] :
                            cities_along.append(subject_data['currentChoice'][idx+1+ix])
                            exempt_decision += 1
                            ix += 1
                        # preregistered hit
                        if cities_along[-1] == decision.name:
                            pre_hit = [True]*len(cities_along)
                        else:
                            pre_hit = [False]*len(cities_along)

                        # # [previous version] the version takes account every undo button press
                        # # If it is the case, then save the decision.
                        # # Or even better way is find to when they are undoing.
                        # ix = 0
                        # while subject_data.loc[idx + 1 + ix, 'undoIndicator']:
                        #     exempt_decision += 1
                        #     ix += 1
                        # # If it is the case, then save the next decision.
                        # decision_undo = decision  # decision that made when people decided to undo

                    move_time = (time.time() - bfs_start_time)
                    time_sequence.append(move_time)  # record bfs time

                    # set up the next choice
                    # undoing condition
                    if exempt_decision == 0:
                        # for the submit and the last choice: just ignore
                        if idx + 1 < len(subject_data):  # not the final one
                            # and especially for those are not submitting
                            if subject_data.loc[idx + 1, 'submit'] != 1:
                                if decision.name == subject_data.loc[idx + 1, 'currentChoice']:  # hit
                                    hit_target[idx] = True
                                    LL_k += harmonic_sum(count_iteration[idx])
                                else:  # not hit
                                    count_iteration[idx] += 1
                    else:  # while they are undoing
                        exempt_decision -= 1
                        # for the submit and the last choice: just ignore
                        if idx + 1 < len(subject_data):  # not the final one
                            # and especially for those are not submitting
                            if subject_data.loc[idx + 1, 'submit'] != 1:
                                # if the decision is in the current node's undo-able list (means it is one of the visited cities)

                                # [07/05/2022]
                                if pre_hit.pop(0): # hit
                                    hit_target[idx] = True
                                    LL_k += harmonic_sum(count_iteration[idx])
                                else: # not a hit
                                    # hit_target[idx] = False # its already false.
                                    count_iteration[idx] += 1

                                    # check
                                    if hit_target[idx] == True:
                                        print('what?')

                                # if decision_undo.name in node_now.city_undo:  # hit (at least on the way)
                                #     hit_target[idx] = True
                                #     LL_k += harmonic_sum(count_iteration[idx])
                                # else:  # not hit
                                #     count_iteration[idx] += 1
                                #     # if it is not hit then replan
                                #     exempt_decision = 0
                                #     del decision_undo

            prev_trial_id = curr_trial_id
            prev_condition = curr_condition
            # if hit_target.count(False) == 0:
            #     break
        LL_k = -LL_k - (hit_target.count(False)) * harmonic_sum(k)
        # if hit_target.count(False) == 0:
        #     break

    return hit_target, count_iteration

def ibs_early_stopping_para_trials(para, LL_lower, subject_data, basic_map, value_func = 'legacy'):

    '''
    implement ibs with early stopping
    sequential
    returns the log likelihood of current subject
    '''

    count_iteration = [1]*len(subject_data)
    start_time = time.time()
    time_sequence = [] # bfs time sequence


    # initialize iteration data
    hit_target = [False]*len(subject_data) # true if hit for each move
    index1 = subject_data[subject_data['submit'] == 1].index
    for id in index1:
        hit_target[id] = True
        hit_target[id-1] = True
    # hit_target_gb = hit_target.copy()
    # global hit_target_gb
    LL_k = 0
    k_ = 0
    # iterate until meets early stopping criteria
    while hit_target.count(False) > 0:
#        iter_start_time = time.time()

        if LL_k	<= LL_lower:
            LL_k = LL_lower
            print('*********************** exceeds LL lower bound, break')
            break


        tots = []
        # run_para_trials(LL_k, para, hit_target, subject_data, basic_map, value_func = 'legacy', repeats_para_trial = repeats_para_trial)
        # (LL_k, para, hit_target, subject_data, basic_map, value_func = 'legacy', repeats_para_trial = repeats_para_trial)
        a_pool = multiprocessing.Pool(20)
        # a_pool = multiprocessing.Pool(5)
        func = partial(run_para_trials, LL_k, para, hit_target, subject_data, basic_map, value_func=value_func, repeats_para_trial = repeats_para_trial)
        tots = a_pool.map(func, range(repeats))
        a_pool.close()
        a_pool.join()


        for i in range(len(tots)):
            if  hit_target.count(False) == 0:
                break
            llk = 0
            k_ += 1

            hit_target_ = tots[i][0]
            count_iteration_ = tots[i][1]
            # hit_target = np.any(np.array([hit_target,tots[i]]),axis=0).tolist()

            t_indx = np.argwhere(np.array(hit_target) == True).squeeze().tolist()
            f_indx = np.argwhere(np.array(hit_target) == False).squeeze().tolist()

            t_indx_ = np.argwhere(np.array(hit_target_) == True).squeeze().tolist()
            f_indx_ = np.argwhere(np.array(hit_target_) == False).squeeze().tolist()

            if type(f_indx) != list:
                f_indx = [f_indx]
            if type(t_indx_) != list:
                t_indx_ = [t_indx_]

            for ix in f_indx:
                if ix in t_indx_:
                    hit_target[ix] = True
                    count_iteration[ix] += count_iteration_[ix]


            # previously false but true now.
            for idx in range(len(subject_data)):
                if hit_target[idx]:
                    llk += harmonic_sum(count_iteration[idx])
                else:
                    count_iteration[idx] += count_iteration_[idx]


        LL_k = -llk - (hit_target.count(False)) * harmonic_sum(k_)
        print('\tKth LL_k '+str(LL_k), flush=True)

        hit_number = hit_target.count(True)
        print('\thit_target '+str(hit_number) + ' / ' + str(len(hit_target)), flush=True)
        print('\tk_iteration '+str(k_), flush=True)
        # print('\tmax_position '+str(count_iteration.index(max(count_iteration))))
    if LL_k == 0:
        print(llk)
        print(count_iteration)

    print('IBS total time lapse '+str(time.time() - start_time))
    print('Final LL_k: '+str(LL_k))
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
    # hpc directories
    home_dir = './'
    input_dir = 'data_mod_ibs/'
    map_dir = 'active_map/'
    output_dir = 'll/'

    inparams = [0.1, 15, 0.05, 0.1, 1, 1, 1]
    # inparams = [1, 1, 1, 0.01, 15, 0.05, 0.1] #[1, 1, 1, 0, 30, 1, 0] #before switching order

    with open(home_dir + map_dir + 'basicMap.json', 'r') as file:
        basic_map_ = json.load(file)

    with open(home_dir + map_dir + 'basic_map_48_all4', 'r') as file:
        basic_map = json.load(file)

    subs = [1, 2, 4]  # subject index
    # XX = [0, 0, 0, 0, 0.1, 0.01, 0]
    sub = 1

    sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_' + str(int(sub)) + '.csv')
    LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_u_all'])])

    print('inparams ' + str(inparams))
    nll_avg = ibs_grepeats_test(inparams, LL_lower, sub_data, basic_map, repeats)

    for sub in subs:
        sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_' + str(sub) + '.csv')
        sub_size = len(sub_data)

        with open(home_dir + output_dir + 'L_repeat_' + str(sub),'r') as file:
            L_repeat = json.load(file)

        budget = 30
        R = compute_repeats(sub_size*budget, sub_size, L_repeat)


if __name__ == "__main_db__":
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