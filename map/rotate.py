import scipy.io as sio
import numpy as np
import json

class Map:    
    def __init__(self, map_content, trl_id):
        
        self.loadmap = map_content[trl_id]
        self.order = np.nan
        
        self.N = self.loadmap['N']
        self.total = self.loadmap['total']   # total budget
        self.budget_remain = self.loadmap['total'] # remaining budget()
        
        self.R = self.loadmap['R']
        self.r = self.loadmap['r']
        self.phi = [every + 180 for every in self.loadmap['phi']]
        self.x = np.sqrt(self.r) * np.cos(self.phi)
        self.x = self.x.astype(int).tolist()
        self.y = np.sqrt(self.r) * np.sin(self.phi)
        self.y = self.y.astype(int).tolist()
        self.xy = [(self.x[i], self.y[i]) for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city
        self.distance = self.loadmap['distance']

#with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map_24','r') as file: 
#    basic_map = json.load(file) 
with open('/Users/sherrybao/Downloads/Research/Road_Construction/map/basic_map/basic_map_training','r') as file: 
    basic_map = json.load(file) 

new_list = []
    
for i in range(0,len(basic_map)): #[0]
    map_ = Map(basic_map, i)
    map_ = map_.__dict__
    new_list.append(map_)
    
## saving mat file
#sio.savemat('undo_map_24.mat', {'map_list':new_list})
## saving json
#with open('undo_map_24','w') as file: 
#    json.dump(new_list,file)

# saving mat file
sio.savemat('undo_map_training.mat', {'map_list':new_list})
# saving json
with open('undo_map_training','w') as file: 
    json.dump(new_list,file)
    

