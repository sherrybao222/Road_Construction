import json

# directories
home_dir = '/Users/sherrybao/'
input_dir = 'Downloads/research/road_construction/rc_all_data/data/data_pilot/'

subs = [1] # subject index ,2,4
orders_1 = [[2,3,3,2],
          [3,2,2,3]]
basic_index = []
for num in subs:
    with open(home_dir + input_dir+'/sub_'+str(num)+'/test_all_'+str(num),'r') as file: 
        all_data = json.load(file)
        order_ind = int(num)%2
        for i,cond in enumerate(orders_1[order_ind]):
            if cond == 2:
                basic_index.extend(range(i*24,(i+1)*24))

