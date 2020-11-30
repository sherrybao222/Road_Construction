from map_class import Map
from model_bfs import new_node_current,make_move,params
import time
from statistics import mean,stdev,sqrt

def single_trial(map_content, map_id):
    '''
    simulation of one map
    '''
    
    # load map from map file
    trial = Map(map_content, map_id)
    dict_city = dict(zip(list(range(0,trial.N)), trial.xy)) 
    dict_city_remain = dict_city.copy()
    dist_city = trial.distance.copy()
    
    # simulate
    choice_sequence = [0]
    time_sequence = []
    
    start_time = time.time()
    
    start = new_node_current(0, dict_city_remain, dist_city, trial.budget_remain, 0, para.weights)
    now = start
    while True:
        choice = make_move(now,dist_city,para)
#        print('choice: '+ str(choice.name))
        choice_sequence.append(choice.name)
        move_time = (time.time() - start_time)
        time_sequence.append(move_time)

        if choice.determined == 1:
            break
        
        start_time = time.time()
        
        # use previously chosen node as new start
        new_start = new_node_current(choice.name, choice.city, dist_city, 
                                     choice.budget, choice.n_c, para.weights)        
        now = new_start
        
    return choice_sequence,time_sequence

def all_trial(map_content,n_maps):
    '''
    simulation of all maps
    '''

    all_done = False
    choices_all = []
    times_all = []
    
    while not all_done:
        for map_id in range(0, n_maps):
            choices,times = single_trial(map_content, map_id)
            choices_all.append(choices)
            times_all.append(times)
        all_done = True
        
    return choices_all,times_all


if __name__ == "__main__":
    
    # directories
    home_dir = '/Users/dbao/google_drive/'
    map_dir = 'road_construction/experiments/pilot_0320/map/active_map/'
    input_dir = 'road_construction/experiments/pilot_0320/data/'

    # =============================================================================
    # setting up parameters
    
    # set parameters
    inparams = [1, 1, 1, 0.01, 15, 0.05, 0.1]
    para = params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
    					stopping_probability=inparams[3],
    					pruning_threshold=inparams[4],
    					lapse_rate=inparams[5], feature_dropping_rate=inparams[6])

          
    # load maps
    import json
    with open(home_dir + map_dir + 'basic_map_48_all4','r') as file: 
        map_content = json.load(file)[0] 

    n_maps = 48
    choices_all,times_all = all_trial(map_content,n_maps)

    flat_list = [item for sublist in times_all for item in sublist]
    avg_move = mean(flat_list)
    
    # =============================================================================
    # use hpc ibs bfs time output
    import json
    with open(home_dir + input_dir + 'check_time_sample','r') as file: 
        time_sample = json.load(file)

    
    time_avg = mean(time_sample['time sequence'])
    time_err = stdev(time_sample['time sequence'])/sqrt(len(time_sample['time sequence']))
    
    sample = mean(time_sample['number of samples'])
    sample_err = stdev(time_sample['number of samples'])/sqrt(len(time_sample['number of samples']))
