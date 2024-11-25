from ibs_general_repeats import ibs_grepeats
# import matlab.engine
# import json
# import pandas as pd

'''
starting script to test BADS interface
start to fit with BADS!
this script call BADS from python matlab engine,
BADS will call a matlab ll.m script
'''
# # w1, w2, w3, stopping_probability, pruning_threshold, 
# # lapse_rate, feature_dropping_rate
# x0 = matlab.double([1, 1, 1, 0.01, 15, 0.05, 0.1])   # Starting point
# lb = matlab.double([0.01, 0.01, 0.01, 0.00001, 2, 0.00001, 0.00001])  # Lower bounds
# ub = matlab.double([10, 10, 10, 0.5, 30, 0.5, 0.5])  # Upper bounds
# plb = matlab.double([0.1, 0.1, 0.1, 0.001, 5, 0.01, 0.01])   # Plausible lower bounds
# pub = matlab.double([5, 5, 5, 0.3, 20, 0.3, 0.3])  # Plausible upper bounds
# # nonbcon = matlab.double([]);
# # options = matlab.double([]);

# # directories
# home_dir = '/Users/Toby/Downloads/bao/'
# input_dir = 'road_construction/rc_all_data/data/data_pilot_preprocessed/'
# map_dir = 'road_construction/map/active_map/'

# with open(home_dir + map_dir + 'basic_map_48_all4','r') as file:
#     basic_map = json.load(file) 
    
# subs = [1]#,2,4] # subject index 

# eng = matlab.engine.start_matlab()

# for sub in subs:
#     sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_'+str(sub) + '.csv')
 
#     with open(home_dir + input_dir + 'n_repeat_' + str(sub),'r') as file:
#         repeats = json.load(file) 
    
#     result = eng.bads("@bads_ll", x0,lb,ub,plb,pub,nargout=2) # BADS will call bads_ll.m to get neg LL
    	
#     print('result[0]', result[0])
#     print('result[1]', result[1])
    
# eng.quit()


    