import pandas as pd

def comp_value(weights):
                             
    value = weights[0] * node.n_c + weights[1] * node.n_u + \
            weights[2] * (node.budget/30) + np.random.normal()

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # directories
    home_dir = '/Users/dbao/google_drive/'
    input_dir = 'road_construction/experiments/pilot_0320/data/data_pilot_preprocessed/'
    output_dir = 'road_construction/rc_all_data/data/data_pilot_preprocessed/ll/'
    map_dir = 'road_construction/experiments/pilot_0320/map/active_map/'
    
    weights = [0.1, 0.1, 0.1]
    
    subs = [1]#,2,4] # subject index 
    
    for sub in subs:
        sub_data = pd.read_csv(home_dir + input_dir + 'mod_ibs_preprocess_sub_'+str(sub) + '.csv')
        sub_size = len(sub_data)
    


