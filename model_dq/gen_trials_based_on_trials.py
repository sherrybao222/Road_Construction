from  model_dk_ucb5 import new_node_current, new_node_current_seq, make_move_weights, make_move_undo_weights

import pandas as pd
import numpy as np
import ast # to convert string to list

from contextlib import nullcontext
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

import os

max_trial = 2000

def gen_trials_based_on_trials(para, LL_lower, 
                               subject_data, basic_map, value_func = 'legacy', 
                               save_total=False, dir_param = './', save_count= 0, vis = False, max_stopping=True):
    '''
    '''

    k_max = 200 # max iteration
    k_hist = [] # history of K

    done = False
    idx = 0

    ## initialize data structure ----------------------------------------------------------------------
    conditions      = []
    trial_ids       = []
    map_id          = []
    mas             = []
    cities_reaches  = []
    n_within_reaches= []

    n_opt_paths_all = [] # need to be updated

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
    map_tree = basic_map[1]


    # for random model it only generates
    # currently it is just for random
    # TODO
    # it terminates when there is no better option.

    keys_to_append = subject_data.keys().to_list()
    keys_to_append.append('value') # current node value
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


    ## make videos ------------------------------------------------------------
    if save_total: 
        vidir = dir_param + '/vid_{0:02d}'.format(save_count)
        if not os.path.exists(vidir):
            os.makedirs(vidir)

        matplotlib.use("Agg")

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        
        frame_rate = 3
        writer_total = FFMpegWriter(fps=frame_rate, metadata=metadata)        
        fig = plt.figure(figsize=(6, 6))
        plt.xlim(-205, 205)
        plt.ylim(-205, 205)

        tots = writer_total.saving(fig, vidir + "/gen_tot.mp4", 100)

    while not done: # done for all maps

        if max_stopping: # for looping
            if out_sub.shape[0] > max_trial:
                break

        if save_total:
            writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
            tot = writer.saving(fig, vidir + "/gen_trial{0:02d}.mp4".format(idx), 100)

            plt.cla()
            plt.xlim(-205, 205)
            plt.ylim(-205, 205)
            plt.text(140, 170, 'Trial: {}\n'.format(idx + 1) + sub_info['condition'][idx])

            for _ in range(frame_rate):
                writer.grab_frame()
            for _ in range(frame_rate):
                writer_total.grab_frame()

        ## load map info and initialize -----------------------------------------------------------
        try:
            TS = map_tree[sub_info['map_id'][idx]]
        except:
            TS= [ ]
        currPath = [0]
        ind_currpath = np.where([currPath == TS['paths'][j] for j in range(len(TS['paths']))])[0].squeeze().tolist()
        
        dist = basic_map[0][sub_info['map_id'][idx]]['distance']
        cities_reach = sub_info['cities_reach'][idx].copy()
        cities_taken = []

        done_trial = False
        n_city = 1
        budget_remain = 300.0

        idff = 0 #sherry: time step
        visits_ = np.zeros(30)
        name = 0

        ## starting city ---------------------------------------------------------------------------------
        idff += 1      
        node_now = new_node_current(name,
                                    cities_reach, cities_taken,
                                    dist, budget_remain, n_city,
                                    para, value_func=value_func, i_th = idff, visits = visits_, condition = sub_info['condition'][idx] )
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
    
        ###################################################
        nos = node_now
        ii = 1
        while nos.parent:
            ii += 1
            nos = nos.parent
        if node_now.n_c != ii:
            print('d')
        ###################################################

        plotlines = []

        while not done_trial:

            idff += 1

            if vis:
                print('*'*10 + '{} node_now'.format(idx))
                print(node_now)

            name_bf = node_now.name

            if sub_info['condition'][idx] == 'undo':
                decision, ws = make_move_undo_weights(node_now, dist, para, value_func=value_func, i_th = idff+1, visits = visits_)
            else:
                decision, ws = make_move_weights(node_now, dist, para, value_func=value_func, i_th = idff+1, visits = visits_)

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

            #########################
            nos = decision
            ii = 1
            while nos.parent:
                ii += 1
                nos = nos.parent
            #########################

            if (node_now == decision) or (idff > k_max): # submit

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
            else: # not undid

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


        k_hist.append(idff)
        idx += 1
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

