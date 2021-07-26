import pandas as pd
import numpy as np

def comp_value(weights,n_c,n_u,budget):
                             
    value = weights[0] * n_c + weights[1] * n_u + \
            weights[2] * (budget/30) 
    
    value_noise = value + np.random.normal()
    
    return value, value_noise

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # directories
    home_dir = '/Users/dbao/google_drive/'
    input_dir = 'road_construction/experiments/pilot_0320/data/data_pilot_preprocessed/'
    output_dir = 'road_construction/rc_all_data/data/data_pilot_preprocessed/ll/'
    map_dir = 'road_construction/experiments/pilot_0320/map/active_map/'
    
    weights = [1, 0.1, 0.1]
    
    subs = [1]#,2,4] # subject index 
    
    for sub in subs:
        sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_'+str(sub) + '.csv')
        sub_size = len(sub_data)
        
        value_all = []
        value_noise_all = []
        
        for i in range(sub_size):
            value, value_noise = comp_value(weights, sub_data.loc[i,'n_city_all'],
                       sub_data.loc[i,'n_u_all'],sub_data.loc[i,'budget_all'])
            
            value_all.append(value)
            value_noise_all.append(value_noise)


