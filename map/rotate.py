import scipy.io as sio
import numpy as np
from scipy.spatial import distance_matrix

class Map:    
    def __init__(self, map_content, trl_id):
        
        self.loadmap = map_content['map_list'][0,trl_id][0,0]
        self.order = np.nan

        self.N = self.loadmap.N.tolist()[0][0]
        self.radius = 10     # radius of city
        self.total = self.loadmap.total.tolist()[0][0]   # total budget
        self.budget_remain = self.loadmap.total.copy().tolist()[0][0]  # remaining budget()
        
        self.R = self.loadmap.R.tolist()[0]
        self.r = self.loadmap.r.tolist()[0]
        self.phi = [every + 180 for every in self.loadmap.phi.tolist()[0]]
        
        self.x = np.sqrt(self.r) * np.cos(self.phi) + 1000
        self.x = self.x.astype(int)
        self.y = np.sqrt(self.r) * np.sin(self.phi) + 800
        self.y = self.y.astype(int)
        self.xy = [(self.x[i], self.y[i]) for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000)     # city distance matrix


new_list = []
basic_map = sio.loadmat('/Users/sherrybao/Downloads/Research/Road_Construction/map/test_basic.mat',  struct_as_record=False)
for i in range(0,len(basic_map['map_list'][0])):
    map_ = Map(basic_map, i)
    new_list.append(map_)
    
# saving
sio.savemat('test_undo.mat', {'map_list':new_list})

    

