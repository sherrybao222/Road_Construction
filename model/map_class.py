import numpy as np
import math
from scipy.spatial import distance_matrix


class Map:
    def __init__(self, map_content, map_id): 
        
#        self.circle_map()
        self.load_map(map_content, map_id)

    def circle_map(self):
        # map parameters
        self.N = 11    # total city number, including start
        self.radius = 5     # radius of city
        self.total = 300    # total budget
        self.budget_remain = 300    # remaining budget
        
        self.R = 200*200 #circle radius' sqaure
        self.r = np.random.uniform(0, self.R, self.N).tolist() 
        self.phi = np.random.uniform(0,2 * math.pi, self.N).tolist()  
        self.x = np.sqrt(self.r) * np.cos(self.phi)
        self.x = self.x.astype(int).tolist() 
        self.y = np.sqrt(self.r) * np.sin(self.phi)
        self.y = self.y.astype(int).tolist() 
        self.xy = [(self.x[i], self.y[i]) for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000)     # city distance matrix
        self.dist_city = self.distance.copy()
                
        self.dict_city = dict(zip(list(range(0,self.N)), self.xy)) 
        self.dict_city_remain = self.dict_city.copy()
      
    def load_map(self, map_content, map_id):
        
        self.loadmap = map_content[map_id]
        self.order = np.nan
        
        self.N = self.loadmap['N']
        self.radius = 5     # radius of city
        self.total = self.loadmap['total']   # total budget
        self.budget_remain = self.loadmap['total'] # remaining budget()
        
        self.R = self.loadmap['R']
        self.r = self.loadmap['r']
        self.phi = self.loadmap['phi']
        self.x = self.loadmap['x']
        self.y = self.loadmap['y']
        self.xy = [[self.x[i], self.y[i]] for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city
        self.distance = self.loadmap['distance']      
        self.dist_city = self.distance.copy()
                
        self.dict_city = dict(zip(list(range(0,self.N)), self.xy)) 
        self.dict_city_remain = self.dict_city.copy()


