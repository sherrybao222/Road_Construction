import json
import pandas as pd

# directories
home_dir = '/Users/sherrybao/Downloads/research/'
input_dir = 'road_construction/rc_all_data/data/data_pilot/'
output_dir = 'road_construction/rc_all_data/data/data_pilot_preprocessed/'

subs = [1,2,4] # subject index 
orders_1 = [[2,3,3,2],
          [3,2,2,3]] # condition orders
all_city = list(range(1,30)) # all city names (excluding start)

for num in subs:
    
    basic_index = [] # basic version trial index
    choice_all = [] # all single choice
    chosen_all = [] # all chosen cities at the single choice
    remain_all = [] # remaining cities at the single choice
    budget_all = [] # budget at the choice
    n_city_all =[] # number of connected cities at the choice
    map_id = [] 

    with open(home_dir + input_dir+'/sub_'+str(num)+'/test_all_'+str(num),'r') as file: 
        all_data = json.load(file)
        
    order_ind = int(num)%2
    for i,cond in enumerate(orders_1[order_ind]):
        if cond == 2:
            basic_index.extend(range(i*24,(i+1)*24))
            
    for i,idx in enumerate(basic_index):
        choice_trial = all_data[0][idx]['choice_dyn']
        choice_trial.remove(0)
        choice_all.extend(choice_trial)
        
        budget_trial = all_data[0][idx]['budget_dyn']
        budget_trial.remove(300)
        budget_all.extend(budget_trial)
        
        n_city_all.extend(range(1,len(choice_trial)+1))
        map_id_single = [i]*len(choice_trial)
        map_id.extend(map_id_single)
        
    move_id = list(range(len(choice_all)))
        
    for i in move_id:
        chosen_trial = all_data[0][basic_index[map_id[i]]]['choice_dyn'][:n_city_all[i]]
        chosen_all.append(chosen_trial)
        remain_trial = [x for x in all_city if x not in chosen_trial]
        remain_all.append(remain_trial)
    
    data = {'move_id':move_id, 'map_id':map_id, 'choice_all':choice_all, 
            'budget_all':budget_all, 'n_city_all':n_city_all,
            'chosen_all':chosen_all, 'remain_all':remain_all}
    # Create DataFrame 
    df = pd.DataFrame(data) 
    df.to_csv(home_dir + output_dir + 'preprocess_sub_'+str(num) + '.csv', index=False)
