# gen_sim_discount3.py uses different value function other than just simple linear function for every kind of features.
# it might be possible there is power law for overestimating/underestimating number of cities already connected.

from  model_dk_ucb5 import new_node_current, new_node_current_seq, initial_node_saving, make_move_weights, make_move_undo_weights, make_move, make_move_undo, params, new_node_previous
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
import os

import datetime
repeats_para_trial = 1
repeats = 100

max_trial = 2000

def harmonic_sum(n):
    '''
    return sum of harmonic series from 1 to n-1
    when n=1, return 0
    '''
    s = 0.0
    for i in range(1, n):
        s += 1.0/i
    return s


def gen_trials_based_on_trials(para, LL_lower, subject_data, basic_map, value_func = 'legacy', save_total=False, dir_param = './',save_count= 0, vis = False, max_stopping=True):
    '''
    implement ibs with early stopping
    sequential
    returns the log likelihood of current subject
    '''


    conditions      = []
    trial_ids       = []
    map_id          = []
    mas             = []
    cities_reaches  = []
    n_within_reaches= []

    n_opt_paths_all = [] # need to be updated

    # for i in range(len(np.unique(subject_data['trial_id']))):
    for i in np.unique(subject_data['trial_id']):
        conditions.append(subject_data['condition'][subject_data['trial_id']==i].to_numpy()[0])
        trial_ids.append(subject_data['trial_id'][subject_data['trial_id']==i].to_numpy()[0])
        map_id.append(subject_data['map_id'][subject_data['trial_id']==i].to_numpy()[0])
        mas.append(subject_data['mas_all'][subject_data['trial_id']==i].to_numpy()[0])
        cities_reaches.append( ast.literal_eval(subject_data['cities_reach'][subject_data['trial_id']==i].to_numpy()[0]))
        n_within_reaches.append(subject_data['n_within_reach'][subject_data['trial_id']==i].to_numpy()[0])

    sub_info = {'condition':conditions,'trial_id':trial_ids,
                'map_id':map_id,'mas_all':mas,'cities_reach':cities_reaches,
                'n_within_reach':n_within_reaches}

    k_max = 200 # max iteration
    k_hist = [] # history of K

    map_tree = basic_map[1]


    # for random model it only generates
    # currently it is just for random
    # TODO
    # it terminates when there is no better option.
    done = False
    idx = 0
    keys_to_append = subject_data.keys().to_list()
    keys_to_append.append('value')
    try:
        keys_to_append.pop(keys_to_append.index('index'))
    except:
        ''
    out_sub = pd.DataFrame(columns = keys_to_append)
    # ['condition', 'trial_id', 'map_id',
    #  'undoIndicator', 'submit', 'checkEnd', 'currentChoice',
    #  'chosen_all', 'n_city_all','n_within_reach',
    #  'cities_reach', 'currentBudget', 'time_all', 'rt_all',
    #  'n_opt_paths_all', 'n_subopt_paths_all', 'mas_all', 'tortuosity_all']



    from contextlib import nullcontext
    if save_total:
        vidir = dir_param + '/vid_{0:02d}'.format(save_count)
        if not os.path.exists(vidir):
            os.makedirs(vidir)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as manimation

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        frame_rate = 3
        writer_total = FFMpegWriter(fps=frame_rate, metadata=metadata)
        fig = plt.figure(figsize=(6, 6))
        plt.xlim(-205, 205)
        plt.ylim(-205, 205)

        tots = writer_total.saving(fig, vidir + "/gen_tot.mp4", 100)
    else:
        tots = nullcontext()

    with tots:
        while not done:
            if save_total:
                writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
                tot = writer.saving(fig, vidir + "/gen_trial{0:02d}.mp4".format(idx), 100)
            else :
                tot = nullcontext()

            with tot:

                if max_stopping:
                    if out_sub.shape[0] > max_trial:
                        break
                if save_total:
                    plt.cla()
                    plt.xlim(-205, 205)
                    plt.ylim(-205, 205)

                    plt.text(140, 170, 'Trial: {}\n'.format(idx + 1) + sub_info['condition'][idx])
                    for _ in range(frame_rate):
                        writer.grab_frame()
                    for _ in range(frame_rate):
                        writer_total.grab_frame()
                try:
                    TS = map_tree[sub_info['map_id'][idx]]
                except:
                    TS= [ ]

                currPath = [0]
                ind_currpath = np.where([currPath == TS['paths'][j] for j in range(len(TS['paths']))])[0].squeeze().tolist()



                dist = basic_map[0][sub_info['map_id'][idx]]['distance']

                name = 0

                cities_reach = sub_info['cities_reach'][idx].copy()
                cities_reach_ = sub_info['cities_reach'][idx].copy()
                cities_taken = []

                n_city = 1
                budget_remain = 300.0
                n_u = sub_info['n_within_reach'][idx]

                value = -999

                # print(curr_trial_id

                done_trial = False
                idff = 0

                # visits_ = np.ones(30)*1e-3
                # visits_ = np.zeros(30)
                visits_ = np.zeros(30)
                # visits_[name] += 1
                idff += 1
                node_now = new_node_current(name,
                                            cities_reach, cities_taken,
                                            dist, budget_remain, n_city,
                                            para, value_func=value_func, i_th = idff, visits = visits_, condition = sub_info['condition'][idx] )
                                            # para, n_u=n_u, value_func=value_func, i_th = idff+1, visits = visits_)
                visits_[name] += 1
                if vis:
                    print('*'*10 + ' INITIAL node_now')
                    print(node_now)
                    print('NodeName:{}, Value:{}, iternum:{}, Visits:{}'.format(node_now.name,node_now.value.round(2),idff, visits_[name]))

                sg = pd.DataFrame(data=[[sub_info['condition'][idx], idx, sub_info['map_id'][idx],
                                         0, 0, 0, 0,
                                         str([0]), 1, sub_info['n_within_reach'][idx],
                                         sub_info['cities_reach'][idx], 300.0, 999, -1,
                                         TS['optpath'][ind_currpath], TS['suboptpath'][ind_currpath], TS['mas'][ind_currpath], 999.0, node_now.value ]],
                                  columns=keys_to_append)

                # ['condition', 'trial_id', 'map_id',
                #  'undoIndicator', 'submit', 'checkEnd', 'currentChoice',
                #  'chosen_all', 'n_city_all','n_within_reach',
                #  'cities_reach', 'currentBudget', 'time_all', 'rt_all',
                #  'n_opt_paths_all', 'n_subopt_paths_all', 'mas_all', 'tortuosity_all']
                out_sub = out_sub.append(sg)

                if save_total:
                    ctnames = ['{0:2d}'.format(int(i)) for i in np.linspace(0, 29, 30).tolist()]
                    xys = np.array(
                        [basic_map[0][sub_info['map_id'][idx]]['x'][:], basic_map[0][sub_info['map_id'][idx]]['y'][:]])
                    plt.scatter(xys[0, :], xys[1, :], s=10, c='k')
                    for count in range(30):
                        plt.annotate(ctnames[count], xys[:, count])
                    # selected = plt.scatter(basic_map[0][sub_info['map_id'][idx]]['x'][0], basic_map[0][sub_info['map_id'][idx]]['y'][0], s=40, c='r')
                    selected = plt.scatter(xys[0, 0], xys[1, 0], s=40, c='r')
                    selected_line = plt.plot([xys[0, 0], xys[0, 0]], [xys[1, 0], xys[1, 0]], c='r')
                    plt.xlim(-205, 205)
                    plt.ylim(-205, 205)
                    plt.title('Starting from 0')
                    prev_val = plt.text(-180, 190, 'Value: {}'.format(np.round(node_now.value,2)))

                    plt.draw()
                    for _ in range(frame_rate):
                        writer.grab_frame()
                    for _ in range(frame_rate):
                        writer_total.grab_frame()

                nos = node_now
                ii = 1
                while nos.parent:
                    ii += 1
                    nos = nos.parent
                if node_now.n_c != ii:
                    print('d')


                plotlines = []
                while not done_trial:
                    if vis:
                        print('*'*10 + '{} node_now'.format(idx))
                        print(node_now)
                    name_bf = node_now.name

                    if sub_info['condition'][idx] == 'undo':
                        decision, ws = make_move_undo_weights(node_now, dist, para, value_func=value_func, i_th = idff+1, visits = visits_)
                        # decision = make_move_undo(node_now, dist, para, value_func=value_func)
                    else:
                        decision, ws = make_move_weights(node_now, dist, para, value_func=value_func, i_th = idff+1, visits = visits_)
                        # decision = make_move(node_now, dist, para, value_func=value_func)

                    idff += 1
                    if vis:
                        print('*'*10 + '{} DECISION'.format(idx))
                        print(decision)
                        print('NodeName:{}, Value:{}, iternum:{}, Visits:{}'.format(decision.name,decision.value.round(2),idff, visits_[decision.name]))
                    # if name == decision.name:
                    #     done_trial = False
                    #     break

                    name = decision.name
                    budget_remain = decision.budget
                    cities_taken = decision.city_undo
                    n_city = decision.n_c
                    value = decision.value

                    nos = decision
                    ii = 1
                    while nos.parent:
                        ii += 1
                        nos = nos.parent


                    if (node_now == decision) or (idff>k_max):
                        currPath = [i.name for i in decision.path]
                        ind_currpath = np.where([currPath == TS['paths'][j] for j in range(len(TS['paths']))])[0].squeeze().tolist()
                        city_list = decision.city_undo.copy()
                        city_list.append(name)

                        cities_remain = decision.city.copy()
                        cities_wr = []
                        for c in cities_remain:
                            if dist[decision.name][c] <= decision.budget:
                                cities_wr.append(c)
                        sg = pd.DataFrame(data=[[sub_info['condition'][idx], idx, sub_info['map_id'][idx],
                                                 0, 1, 1, name,
                                                 str(city_list), decision.n_c, decision.n_u,
                                                 str(cities_wr), decision.budget, 999, -1,
                                                 TS['optpath'][ind_currpath], TS['suboptpath'][ind_currpath], TS['mas'][ind_currpath], 999.0, decision.value]],
                                          columns=keys_to_append)
                        # ['condition','trial_id','map_id',
                        #  'undoIndicator','submit','checkEnd','currentChoice',
                        #  'chosen_all','n_city_all','n_within_reach',
                        #  'cities_reach','currentBudget','time_all','rt_all',
                        #  'n_opt_paths_all','n_subopt_paths_all','mas_all','tortuosity_all','value']

                        out_sub = out_sub.append(sg)

                        if save_total:
                            # rendering - connected next city
                            selected.remove()
                            selected_line = selected_line.pop(0)
                            selected_line.remove()
                            prev_val.remove()
                            plt.draw()
                            plt.xlim(-205, 205)
                            plt.ylim(-205, 205)
                            plt.title('episode length exceeded its max: {}'.format(idff))
                            prev_val = plt.text(-180, 190, 'Value: {}'.format(np.round(decision.value,2)))
                            plt.draw()
                            for _ in range(frame_rate):
                                writer.grab_frame()
                            for _ in range(frame_rate):
                                writer_total.grab_frame()

                        break

                    # max(decision.children, key=lambda node: node.value)
                    if n_city < node_now.n_c: # undid

                        nos = node_now
                        ii = 1
                        while nos.parent:
                            ii += 1
                            nos = nos.parent
                        if node_now.n_c != ii:
                            print('d')

                        cities_reach = decision.city
                        temp = node_now
                        while True:

                            if decision.name == temp.name:
                                break
                            temp = temp.parent

                            if temp:
                                currPath = [i.name for i in temp.path]
                                ind_currpath = np.where([currPath == TS['paths'][j] for j in range(len(TS['paths']))])[0].squeeze().tolist()
                                city_list = temp.city_undo.copy()
                                city_list.append(temp.name)

                                cities_remain = temp.city.copy()
                                cities_wr = []
                                for c in cities_remain:
                                    if dist[temp.name][c] <= temp.budget:
                                        cities_wr.append(c)
                                try:
                                    sg = pd.DataFrame(data=[[sub_info['condition'][idx], idx, sub_info['map_id'][idx],
                                                             1, 0, int(temp.n_u == 0), temp.name,
                                                             str(city_list), temp.n_c, temp.n_u,
                                                             str(cities_wr), temp.budget, 999, -1,
                                                             TS['optpath'][ind_currpath], TS['suboptpath'][ind_currpath], TS['mas'][ind_currpath], 999.0, temp.value]],
                                              columns=keys_to_append)
                                except:
                                    print('')
                                # ['condition','trial_id','map_id',
                                #  'undoIndicator','submit','checkEnd','currentChoice',
                                #  'chosen_all','n_city_all','n_within_reach',
                                #  'cities_reach','currentBudget','time_all','rt_all',
                                #  'n_opt_paths_all','n_subopt_paths_all','mas_all','tortuosity_all','value']
                                out_sub = out_sub.append(sg)

                                if save_total:
                                    # rendering - undoing
                                    selected.remove()
                                    selected_line = selected_line.pop(0)
                                    selected_line.remove()
                                    prev_val.remove()
                                    plt.draw()

                                    toberemoved = plotlines[-1].pop(0)
                                    toberemoved.remove()
                                    plotlines.pop()

                                    selected = plt.scatter(xys[0, temp.name], xys[1, temp.name], s=40, c='r')
                                    selected_line = plt.plot([xys[0, temp.name], xys[0, temp.name]],
                                                             [xys[1, temp.name], xys[1, temp.name]],
                                                             c='r')
                                    plt.xlim(-205, 205)
                                    plt.ylim(-205, 205)
                                    plt.title('Undo to {}, episode length: {}'.format(temp.name, idff))
                                    prev_val = plt.text(-180, 190, 'Value: {}'.format(np.round(temp.value,2)))
                                    plt.draw()
                                    writer.grab_frame()
                                    writer_total.grab_frame()

                        node_now = temp

                    else:
                        currPath = [i.name for i in decision.path]
                        ind_currpath = np.where([currPath == TS['paths'][j] for j in range(len(TS['paths']))])[0].squeeze().tolist()
                        city_list = decision.city_undo.copy()
                        city_list.append(decision.name)

                        cities_remain = decision.city.copy()
                        cities_wr = []
                        for c in cities_remain:
                            if dist[decision.name][c] <= decision.budget:
                                cities_wr.append(c)

                        sg = pd.DataFrame(data=[[sub_info['condition'][idx], idx, sub_info['map_id'][idx],
                                                 0, 0, 0, name,
                                                 str(city_list), decision.n_c, decision.n_u,
                                                 str(cities_wr), decision.budget, 999, -1,
                                                 TS['optpath'][ind_currpath], TS['suboptpath'][ind_currpath], TS['mas'][ind_currpath], 999.0, decision.value]],
                                          columns=keys_to_append)
                        # ['condition','trial_id','map_id',
                        #  'undoIndicator','submit','checkEnd','currentChoice',
                        #  'chosen_all','n_city_all','n_within_reach',
                        #  'cities_reach','currentBudget','time_all','rt_all',
                        #  'n_opt_paths_all','n_subopt_paths_all','mas_all','tortuosity_all','value']
                        out_sub = out_sub.append(sg)

                        cities_reach = decision.city
                        node_now = new_node_current_seq(node_now, name,
                                                        cities_reach, cities_taken,
                                                        dist, budget_remain, n_city,
                                                        para, value_func=value_func, i_th = idff+1, visits = visits_, condition = sub_info['condition'][idx])
                                                        # para, n_u=n_u, value_func=value_func, i_th = idff+1, visits = visits_)

                        if save_total:
                            # rendering - connected next city
                            selected.remove()
                            selected_line = selected_line.pop(0)
                            selected_line.remove()
                            prev_val.remove()
                            plt.draw()

                            selected = plt.scatter(xys[0, name], xys[1, name], s=40, c='r')
                            line = plt.plot([xys[0, name_bf], xys[0, name]], [xys[1, name_bf], xys[1, name]], c='k')
                            plotlines.append(line)
                            selected_line = plt.plot([xys[0, name_bf], xys[0, name]], [xys[1, name_bf], xys[1, name]], c='r')
                            plt.xlim(-205, 205)
                            plt.ylim(-205, 205)
                            plt.title('City {} connected, episode length: {}'.format(name, idff))
                            prev_val = plt.text(-180, 190, 'Value: {}'.format(np.round(decision.value,2)))

                            plt.draw()
                            writer.grab_frame()
                            writer_total.grab_frame()

                    visits_[node_now.name] +=1


                idx += 1
                k_hist.append(idff)

                if idx>len(np.unique(sub_info['map_id']))-1:
                    break

            del node_now

    out_sub = out_sub.reset_index()
    idxx = out_sub.index[out_sub.submit==1]
    # Change the check End that are not marked because the model dosent know when to terminate.
    for idx in idxx:
        out_sub.checkEnd[idx-1]=1

    # out_sub.checkEnd[len(out_sub) - 1] = 1


    return out_sub, k_hist

def generate_sim(t_param, save_total = False):
    t_param = int(t_param)

    # hpc directories
    home_dir = './'
    # input_dir = 'data_mod_ibs/'
    input_dir = 'data_online/'
    map_dir = 'active_map/'
    output_dir = 'll/'

    with open(home_dir + map_dir + 'basicMap.json', 'r') as file:
        basic_map_ = json.load(file)

    with open(home_dir + map_dir + 'tree/map_tree', 'r') as file:
        map_tree = json.load(file)

    sub_data = pd.read_csv(home_dir + input_dir + 'preprocess4_sub_unidentifiedID2_RC-Phaser_2022-01-04_11h42.26.779_REWARD426.csv')
    LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if n != 0]) # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

    inparams = [0.427352070808411,10.1158168075583,0.0100133791818332,0.202838897705078,0,9.33591604232788,7.16987967491150,3.59587907791138,0.280662775039673, 1]
    LB  = [0.1, 1, 0.01, 0.01, 0, 0,  0,  0,  0, 0]  # Lower bounds
    UB  = [0.9, 30,  1,    0.99, 0, 10, 10, 10, 1, 10]   # Upper bounds
    params_name = ['stopping_probability','pruning_threshold','lapse_rate',
            'feature_dropping_rate','undoing_threshold','w1','w2','w3','w4','ucb_confidence']

    # inparams = [0.327148437500000,0.216181159146692,0.0473450097917813,0.140136718750000,8.83789062500000,7.93945312500000,9.40429687500000,2.56835937500000,0.213867187500000]
    # LB = [0.1, 1,  0.01, 0.01, 0, 0, 0, 0, 0]
    # UB = [0.9, 30, 1,    0.99, 10, 10, 10, 10, 1]
    # params_name = ['stopping_probability','pruning_threshold','lapse_rate',
    #     'feature_dropping_rate','undo_inverse_temparature','w1','w2','w3','w4']


    repeats = 1
    # inparams = XX
    print('inparams ' + str(inparams))
    print('target parameter ' + params_name[t_param])
    from utils import params_by_name

    start_time = time.time()
    save_dir_name = '05042022_power_law_free'
    # save_dir_name = '/scratch/dk4315/05012022_power_law_free_undo_softmax'
    import os
    import pickle
    inparams_copied = inparams.copy()
    if LB[t_param] != UB[t_param]:
        iter_t_params = np.linspace(LB[t_param], UB[t_param], 10)
        dir_param = save_dir_name + '/' + params_name[t_param]
        if not os.path.exists(dir_param):
            os.makedirs(dir_param)

        for count, iter_t_param in enumerate(iter_t_params):
            # if True:
            if not os.path.exists(dir_param + '/khist_{0:02d}.csv'.format(count)):
                inparams_copied[t_param] = iter_t_param
                para = params_by_name(inparams_copied, params_name)
                # out_sub, k_hist= gen_trials_based_on_trials(para, LL_lower, sub_data, [basic_map_, map_tree], value_func='power_law_free', save_total=save_total)
                out_sub, k_hist= gen_trials_based_on_trials(para, LL_lower, sub_data, [basic_map_, map_tree], value_func='power_law_free', save_total=save_total, dir_param = dir_param, save_count= count)

                out_sub.to_csv(dir_param + '/gen_{0:02d}.csv'.format(count))
                pd.DataFrame(k_hist).to_csv(dir_param + '/khist_{0:02d}.csv'.format(count))


    print('###### IBS grepeats time lapse ' + str( (time.time() - start_time) *10))

def generate_sim_paramsets(inparams, save_total = False,
                             save_dir_name = '05042022_',
                             name_model = 'power_law_free',
                             save_filename='gen',
                             params_name = ['stopping_probability','pruning_threshold','lapse_rate',
                            'feature_dropping_rate','undoing_threshold',
                            'w1','w2','w3','w4','ucb_confidence']):
    from utils import params_by_name
    # hpc directories
    home_dir = './'
    # input_dir = 'data_mod_ibs/'
    input_dir = 'data_online/'
    map_dir = 'active_map/'
    output_dir = 'll/'

    import json
    with open(home_dir + map_dir + 'basicMap.json', 'r') as file:
        basic_map_ = json.load(file)

    with open(home_dir + map_dir + 'tree/map_tree', 'r') as file:
        map_tree = json.load(file)

    # sub_data = pd.read_csv(home_dir + input_dir + 'preprocess4_sub_unidentifiedID2_RC-Phaser_2022-01-04_11h42.26.779_REWARD426.csv')
    sub_data = pd.read_csv(home_dir + input_dir + 'preprocess4_sub_60da65c630ff4389c297b03c_RC-phaser6_2022-01-20_21h02.32.215.csv')

    # use only basic trials
    sub_data = sub_data.iloc[sub_data.index[sub_data.condition=='basic']]
    sub_data.reset_index()

    LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if n != 0]) # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

    params_name = params_name

    repeats = 1
    # inparams = XX

    start_time = time.time()
    # save_dir_name = '/scratch/dk4315/05012022_power_law_free_undo_softmax'
    save_dir_name = save_dir_name + name_model
    import os
    import pickle
    dir_param = save_dir_name + '/' + save_filename + '/'
    if not os.path.exists(dir_param):
        os.makedirs(dir_param)
    gi = 1
    gen_num = len(inparams)
    while True:
        gdt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        print('inparams ' + str(inparams[gi-1]))
        try:
            inparams_ = inparams[gi-1].tolist()
        except:
            inparams_ = inparams[gi-1].copy()
        inparams_copied = inparams_

        para = params_by_name(inparams_copied, params_name, count_par=int(10))

        with open(dir_param + '/params{}.json'.format(gdt), 'w') as file:
            json.dump({'inparams':inparams_copied, 'params_name':params_name},file)

        # out_sub, k_hist= gen_trials_based_on_trials(para, LL_lower, sub_data, [basic_map_, map_tree], value_func='power_law_free', save_total=save_total)
        out_sub, k_hist= gen_trials_based_on_trials(para, LL_lower, sub_data, [basic_map_, map_tree],
                                        value_func='power_law_free_linear', save_total=save_total, dir_param = dir_param, save_count= 0, vis=True)


        mas_s = [map_tree[i]['mas'][0] for i in range(len(map_tree))]

        submitted = out_sub[out_sub.submit == 1]
        undo_submitted = submitted[submitted.condition == 'undo'].sort_values(by=['map_id'])
        basic_submitted = submitted[submitted.condition == 'basic'].sort_values(by=['map_id'])

        out_sub.to_csv(dir_param + '/'+ save_filename +'_{}.csv'.format(gdt))
        pd.DataFrame(k_hist).to_csv(dir_param + '/khist_{}.csv'.format(gdt))
        print('######')
        print(k_hist)
        print("FINAL MEAN:{}".format(np.mean(k_hist)))
        gi+=1
        if gi > gen_num:
            break

    print('###### IBS grepeats time lapse ' + str( (time.time() - start_time) *10))


def generate_sim_indiv_param(param, sub_filename, save_total = False,
                             save_dir_name = '05042022_',
                             name_model = 'power_law_free',
                             save_filename='gen',
                             params_name = ['stopping_probability','pruning_threshold','lapse_rate',
                            'feature_dropping_rate','undoing_threshold',
                            'w1','w2','w3','w4','ucb_confidence']):

    # hpc directories
    home_dir = './'
    # input_dir = 'data_mod_ibs/'
    input_dir = 'data_online/'
    map_dir = 'active_map/'
    output_dir = 'll/'

    import json
    with open(home_dir + map_dir + 'basicMap.json', 'r') as file:
        basic_map_ = json.load(file)

    with open(home_dir + map_dir + 'tree/map_tree', 'r') as file:
        map_tree = json.load(file)

    # sub_data = pd.read_csv(home_dir + input_dir + 'preprocess4_sub_unidentifiedID2_RC-Phaser_2022-01-04_11h42.26.779_REWARD426.csv')
    sub_data = pd.read_csv(home_dir + input_dir + sub_filename)
    LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if n != 0]) # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.

    # 'power_law_free'
    inparams = param
    params_name = params_name

    repeats = 1
    # inparams = XX
    print('inparams ' + str(inparams))
    from utils import params_by_name

    start_time = time.time()
    # save_dir_name = '/scratch/dk4315/05012022_power_law_free_undo_softmax'
    save_dir_name = save_dir_name + name_model
    import os
    import pickle
    print(type(inparams))
    try:
        inparams = inparams.tolist()
    except:
        inparams= inparams.copy()
    inparams_copied = inparams
    dir_param = save_dir_name + '/'
    if not os.path.exists(dir_param):
        os.makedirs(dir_param)

    if True:
    # if not os.path.exists(dir_param + '/khist.csv'):
    #     para = params_by_name(inparams_copied, params_name)
        para = params_by_name(inparams_copied, params_name, count_par=int(10))

        with open(dir_param + '/params.json', 'w') as file:
            json.dump({'inparams':inparams, 'params_name':params_name},file)

        # out_sub, k_hist= gen_trials_based_on_trials(para, LL_lower, sub_data, [basic_map_, map_tree], value_func='power_law_free', save_total=save_total)
        out_sub, k_hist= gen_trials_based_on_trials(para, LL_lower, sub_data, [basic_map_, map_tree], value_func='power_law_free', save_total=save_total, dir_param = dir_param, save_count= 0, vis=True)


        mas_s = [map_tree[i]['mas'][0] for i in range(len(map_tree))]

        submitted = out_sub[out_sub.submit == 1]
        undo_submitted = submitted[submitted.condition == 'undo'].sort_values(by=['map_id'])
        basic_submitted = submitted[submitted.condition == 'basic'].sort_values(by=['map_id'])

        out_sub.to_csv(dir_param + '/'+ save_filename +'.csv')
        pd.DataFrame(k_hist).to_csv(dir_param + '/khist.csv')
        pd.DataFrame(inparams).to_csv(dir_param + '/params.csv')

    print('###### IBS grepeats time lapse ' + str( (time.time() - start_time) *10))

def maximize_reward_sim_fixed_param(inparams, save_total = False, id = 0, save_dir_name = '05042022_power_law_free'):

    # hpc directories
    home_dir = './'
    # input_dir = 'data_mod_ibs/'
    input_dir = 'data_online/'
    map_dir = 'active_map/'
    output_dir = 'll/'

    import json
    with open(home_dir + map_dir + 'basicMap.json', 'r') as file:
        basic_map_ = json.load(file)

    with open(home_dir + map_dir + 'tree/map_tree', 'r') as file:
        map_tree = json.load(file)

    sub_data = pd.read_csv(home_dir + input_dir + 'preprocess4_sub_unidentifiedID2_RC-Phaser_2022-01-04_11h42.26.779_REWARD426.csv')
    LL_lower = np.sum([np.log(1.0 / n) for n in list(sub_data['n_within_reach']) if n != 0]) # n is 0 when it is terminal. then it would be just undo or submit. But we only count undo and clicking on the cities so submit is not our interest. it would be 1, and log would be 0.
    # sub_data = sub_data[sub_data['condition'] == 'undo'].reset_index()


    # 'power_law_free'
    LB  = [0.1, 1, 0.01, 0.01, 0, 0,  0,  0,  0, 0]  # Lower bounds
    UB  = [0.9, 30,  1,    0.99, 10, 10, 10, 10, 1, 0]   # Upper bounds
    params_name = ['stopping_probability','pruning_threshold','lapse_rate',
            'feature_dropping_rate','undoing_threshold',
                   'w1','w2','w3','w4','ucb_confidence']


    # 'power_law_free'
    # inparams = [0.327148437500000,0.216181159146692,0.0473450097917813,
    # 0.140136718750000,8.83789062500000,7.93945312500000,9.40429687500000,2.56835937500000,0.213867187500000]
    # LB = [0.1, 1,  0.01, 0.01, 0, 0, 0, 0, 0]
    # UB = [0.9, 30, 1,    0.99, 10, 10, 10, 10, 1]
    # params_name = ['stopping_probability','pruning_threshold','lapse_rate',
    #     'feature_dropping_rate','undo_inverse_temparature','w1','w2','w3','w4']


    # 'power_law_free_undo_softmax'
    # inparams = [0.1,12,0.02,
    #             0,  10,  9, 9.7, 2.4, 0.16, 0]
    # LB = [0.1, 1, 0.01, 0.01, 0, 0, 0, 0, 0]
    # UB = [0.9, 30, 1,   0.99, 10, 10, 10, 10, 1]
    # params_name = ['stopping_probability', 'pruning_threshold', 'lapse_rate',
    #               'feature_dropping_rate', 'undo_inverse_temparature', 'w1', 'w2', 'w3', 'w4', 'ucb_confidence']


    # 'power_law_free_undo_softmax'
    # inparams = [0.1,0.1,0.02,
    #             0.17,0,9,9.7,2.4,0.16, 0]
    # LB = [0.1, 1, 0.01, 0.01, 0, 0, 0, 0, 0]
    # UB = [0.9, 30, 1,   0.99, 10, 10, 10, 10, 1]
    # params_name = ['stopping_probability', 'pruning_threshold', 'lapse_rate',
    #               'feature_dropping_rate', 'undo_inverse_temparature', 'w1', 'w2', 'w3', 'w4', 'ucb_confidence']

    repeats = 1
    # inparams = XX
    print('inparams ' + str(inparams))
    from bads_ibs import params_by_name

    start_time = time.time()
    # save_dir_name = '/scratch/dk4315/05012022_power_law_free_undo_softmax'
    import os
    import pickle
    inparams_copied = inparams.copy()
    # dir_param = save_dir_name + '/' + 'fixed_param'
    # if not os.path.exists(dir_param):
    #     os.makedirs(dir_param)

    if True:
    # if not os.path.exists(dir_param + '/khist.csv'):
    #     para = params_by_name(inparams_copied, params_name)
        para = params_by_name(inparams_copied, params_name, count_par=int(10))

        # with open(dir_param + '/params.json', 'w') as file:
        #     json.dump({'inparams':inparams, 'params_name':params_name},file)

        # out_sub, k_hist= gen_trials_based_on_trials(para, LL_lower, sub_data, [basic_map_, map_tree], value_func='power_law_free', save_total=save_total)
        out_sub, k_hist= gen_trials_based_on_trials(para, LL_lower, sub_data, [basic_map_, map_tree], value_func='power_law_free', save_total=save_total, save_count= 0)


        mas_s = [map_tree[i]['mas'][0] for i in range(len(map_tree))]

        submitted_nct = out_sub[out_sub.submit == 1].sort_values(by=['map_id'])['n_city_all']

        # out_sub.to_csv(dir_param + '/gen.csv')
        # pd.DataFrame(k_hist).to_csv(dir_param + '/khist.csv')
    mas_s = np.array(mas_s[:46])
    nct_s = np.array(submitted_nct)
    rr = np.corrcoef(np.array([mas_s[:46], submitted_nct.to_list()]))

    output = [-rr[0,1], (mas_s-nct_s).mean(), -nct_s.mean()]

    print('TARGET: {}'.format(output[id]))
    return output[id]

def render_trials(para, LL_lower, subject_data, basic_map, value_func = 'legacy', save_vid = False):
    # render trials
    '''
    implement ibs with early stopping
    sequential
    returns the log likelihood of current subject
    '''


    conditions      = []
    trial_ids       = []
    map_id          = []
    mas             = []
    cities_reaches  = []
    n_within_reaches= []

    n_opt_paths_all = [] # need to be updated

    for i in range(len(np.unique(subject_data['trial_id']))):
        conditions.append(subject_data['condition'][subject_data['trial_id']==i].to_numpy()[0])
        trial_ids.append(subject_data['trial_id'][subject_data['trial_id']==i].to_numpy()[0])
        map_id.append(subject_data['map_id'][subject_data['trial_id']==i].to_numpy()[0])
        mas.append(subject_data['mas_all'][subject_data['trial_id']==i].to_numpy()[0])
        cities_reaches.append( ast.literal_eval(subject_data['cities_reach'][subject_data['trial_id']==i].to_numpy()[0]))
        n_within_reaches.append(subject_data['n_within_reach'][subject_data['trial_id']==i].to_numpy()[0])

    sub_info = {'condition':conditions,'trial_id':trial_ids,
                'map_id':map_id,'mas_all':mas,'cities_reach':cities_reaches,
                'n_within_reach':n_within_reaches}

    k_max = 200 # max iteration
    k_hist = [] # history of K




    # for random model it only generates
    # currently it is just for random
    # TODO
    # it terminates when there is no better option.
    done = False
    idx = 0

    import matplotlib.pyplot as plt
    plt.figure()


    out_sub = pd.DataFrame(columns = subject_data.keys().to_list())

    while not done:
        sg = pd.DataFrame(data=[[sub_info['condition'][idx], idx, sub_info['map_id'][idx],
                                 0, 0, 0, 0,
                                 str([0]), 1, sub_info['n_within_reach'][idx],
                                 sub_info['cities_reach'][idx], 300.0, 999, -1,
                                 999, 999, sub_info['mas_all'][idx], 999.0]],
                          columns=subject_data.keys())
        out_sub = out_sub.append(sg)


        dist = basic_map[0][sub_info['map_id'][idx]]['distance']

        name = 0

        cities_reach = sub_info['cities_reach'][idx].copy()
        cities_reach_ = sub_info['cities_reach'][idx].copy()
        cities_taken = []

        n_city = 1
        budget_remain = 300.0
        n_u = sub_info['n_within_reach'][idx]

        value = -999

        # print(curr_trial_id
        node_now = new_node_current(name,
                                    cities_reach, cities_taken,
                                    dist, budget_remain, n_city,
                                    para.weights, n_u=n_u, value_func=value_func)
        print('*'*10 + ' INITIAL node_now')
        print(node_now)

        done_trial = False
        idff = 0
        while not done_trial:


            print('*'*10 + '{} node_now'.format(idx))
            print(node_now)

            if sub_info['condition'][idx] == 'undo':
                decision, ws = make_move_undo_weights(node_now, dist, para, value_func=value_func)
                # decision = make_move_undo(node_now, dist, para, value_func=value_func)
            else:
                decision, ws = make_move_weights(node_now, dist, para, value_func=value_func)
                # decision = make_move(node_now, dist, para, value_func=value_func)

            print('*'*10 + '{} DECISION'.format(idx))
            print(decision)

            # if name == decision.name:
            #     done_trial = False
            #     break

            name = decision.name
            budget_remain = decision.budget
            cities_taken = decision.city_undo
            n_city = decision.n_c
            value = decision.value


            if (node_now == decision) or (idff>k_max):
                sg = pd.DataFrame(data=[[sub_info['condition'][idx], idx, sub_info['map_id'][idx],
                                         0, 1, 1, name,
                                         str(decision.city_undo), decision.n_c, decision.n_u,
                                         str([999]), decision.budget, 999, -1,
                                         999, 999, 999, 999.0]],
                                  columns=subject_data.keys())
                out_sub = out_sub.append(sg)
                break

            if n_city < node_now.n_c: # undid
                cities_reach = decision.city
                temp = node_now
                while True:

                    if decision.name == temp.name:
                        break
                    temp = temp.parent

                    if temp:
                        sg = pd.DataFrame(data=[[sub_info['condition'][idx], idx, sub_info['map_id'][idx],
                                                 1, 0, int(temp.n_u == 0), temp.name,
                                                 str(temp.city_undo), temp.n_c, temp.n_u,
                                                 str([999]), temp.budget, 999, -1,
                                                 999, 999, 999, 999.0]],
                                          columns=subject_data.keys())
                        out_sub = out_sub.append(sg)

                node_now = temp

            else:
                # ['condition', 'trial_id', 'map_id',
                #  'undoIndicator', 'submit', 'checkEnd', 'currentChoice',
                #  'chosen_all', 'n_city_all','n_within_reach',
                #  'cities_reach', 'currentBudget', 'time_all', 'rt_all',
                #  'n_opt_paths_all', 'n_subopt_paths_all', 'mas_all', 'tortuosity_all']
                sg = pd.DataFrame(data=[[sub_info['condition'][idx], idx, sub_info['map_id'][idx],
                                         0, 0, int(decision.n_u == 0), name,
                                         str(decision.city_undo), decision.n_c, decision.n_u,
                                         str([999]), decision.budget, 999, -1,
                                         999, 999, 999, 999.0]],
                                  columns=subject_data.keys())
                out_sub = out_sub.append(sg)

                cities_reach = decision.city
                node_now = new_node_current_seq(node_now, name,
                                                cities_reach, cities_taken,
                                                dist, budget_remain, n_city,
                                                para.weights, n_u=n_u, value_func=value_func)

            idff += 1


        idx += 1
        k_hist.append(idff)

        if idx>91:
            break

        del node_now

    plt.show()







import sys
if __name__ == "__main__":
    # arg = sys.argv[1]
    arg = 0

    print(arg)
    # generate_sim(int(arg), True)

    inparams = [0,0.01,15,0.01,
                0.15,99,
                2, 1.14, 0.58, -0.5]

    inparams_all = []
    for i in range(int(1e2)):
        inparams_ = inparams.copy()
        inparams_[6] = np.random.choice(np.linspace(1,4 , 10000))
        inparams_[7] = np.random.choice(np.linspace(0.1, 4, 10000))
        inparams_[8] = np.random.choice(np.linspace(0.1, 1, 10000))
        inparams_[9] = np.random.choice(np.linspace(-5, 0, 10000))
        inparams_all.append(inparams_)

    params_name = ['stopping_probability', 'pruning_threshold', 'lapse_rate',
                   'feature_dropping_rate', 'undoing_threshold',
                  'w1', 'w2', 'w3', 'w4','ucb_confidence', ]

    # bounds
    generate_sim_paramsets(inparams_all, False, save_dir_name='1M', name_model= 'power_law_free_linear')

    print('')

    # generate_sim_fixed_param(inparams,True, save_dir_name='06172022_power_law_free_LINEAR')

# if __name__ == "__main__":
'''
if __name__ == "__optimize__":


    inparams = [0.2, 10, 0,
                0, 1,
                1, 1.478, 0.535, -1, 10] # optimal agent ucb
    # 'power_law_free'
    LB  = [0.1, 1, 0.01, 0.01, 0, 0,  0,  0,  -10, -20]  # Lower bounds
    UB  = [0.9, 30,  1,    0.99, 10, 10, 10, 10, 10, 20]   # Upper bounds
    # maximize_reward_sim_fixed_param(inparams, False, id=0)
    # params_name = ['stopping_probability','pruning_threshold','lapse_rate',
    #         'feature_dropping_rate','undoing_threshold',
    #                'w1','w2','w3','w4','ucb_confidence']
    f = lambda x: maximize_reward_sim_fixed_param(np.array(x), False, id=1)
    from scipy import optimize
    xopts = optimize.minimize(f, inparams, method='Nelder-Mead', bounds = np.array([LB,UB]).transpose() )
    with open('id1_fit.json','w') as f:
        json.dump(xopts,f)


    f = lambda x: maximize_reward_sim_fixed_param(np.array(x), False, id=2)
    from scipy import optimize
    xopts = optimize.minimize(f, inparams, method='Nelder-Mead', bounds = np.array([LB,UB]).transpose() )
    with open('id2_fit.json','w') as f:
        json.dump(xopts,f)


    f = lambda x: maximize_reward_sim_fixed_param(np.array(x), False, id=0)
    from scipy import optimize
    xopts = optimize.minimize(f, inparams, method='Nelder-Mead', bounds = np.array([LB,UB]).transpose() )
    with open('id0_fit.json','w') as f:
        json.dump(xopts,f)

    generate_sim_fixed_param(True,   save_dir_name = '0506022_undo_large_invundo_-1_budget_10confi')


    # arg = 0
    #
    # print(arg)
    # generate_sim(int(arg), False)
'''