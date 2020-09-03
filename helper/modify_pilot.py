import pandas as pd
import ast # to convert string to list

# directories
home_dir = '/Users/sherrybao/Downloads/research/'
inoutput_dir = 'road_construction/rc_all_data/data/data_pilot_preprocessed/'

subs = [1,2,4] # subject index 

for num in subs:
    
    cp_map_id = []
    cp_n_city = []
    index = []

    data = pd.read_csv(home_dir + inoutput_dir + 'ibs_preprocess_sub_'+str(num) + '.csv')
    
    for i in range(len(data)):
        if (data.loc[i,'choice_next_all'] not in ast.literal_eval(data.loc[i,'u_trial_all'])):
            cp_map_id.append(data.loc[i,'map_id'])
            cp_n_city.append(data.loc[i,'n_city_all'])
            index.append(i)
    
    try:       
        for i in index:
            data.loc[i,'choice_next_all'] = ast.literal_eval(data.loc[i,'u_trial_all'])
    except:
        pass
    
    data.to_csv(home_dir + inoutput_dir + 'mod_ibs_preprocess_sub_'+str(num) + '.csv', index=False)

        
            
        

    