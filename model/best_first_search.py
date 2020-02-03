import numpy as np
from anytree import Node
import math
import random
from scipy.spatial import distance_matrix


# define class for returning multiple values
class ReturnValue:
  def __init__(self, value, determined, cities_remain):
     self.value = value
     self.determined = determined
     self.cities_remain = cities_remain
#----------------------------------------------------------------------

def calculate_value(node, cities, dist, budget, n_c):
    # weight
    w_c = 1
    w_u = 1
    w_b = 1
    
    n_u = 0
    
    cities_remain = cities.copy()
    del cities_remain[node.name] # delete the current chosen node

    for c in cities_remain:
        if dist[node.name][c] <= budget:
            n_u = n_u + 1
            
    if n_u == 0:
        node.determined = 1
    else:
        node.determined = 0
    
    value = w_c * n_c + w_u * n_u + w_b * budget
    
    node.value = value
    node.budget = budget
    node.n_c = n_c
    node.city = cities_remain
    
#------------------------------------------------------------------------
def determined(root):
    try:
        n =  max(root.children,key=lambda node:node.value)
        if bool(n.children):
            determined = 1
        else:
            if n.determined == 1:
                determined = 1  
            else:
                determined = 0
                
    except:
        if root.determined == 1:
            determined = 1  
        else:
            determined = 0


    return determined
    
#------------------------------------------------------------------------

def select_node(root):
    
    n = root
    
    while bool(n.children):
        n =  max(n.children,key=lambda node:node.value)
    
    return n

#------------------------------------------------------------------------

def expand_node(n, dist):
    
    s = n.name
        
    for child in n.city:
        if dist[s][child] <= n.budget:
            c = Node(child, parent = n)
            budget_remain = n.budget - dist[s][child]
            n_cc = n.n_c + 1
            calculate_value(c, n.city, dist, budget_remain, n_cc)
        
#    V_max = max(c.value)
    
#    for c in children:
#        if abs(c.value - V_max) > theta:
#            remove_child(c)
 
#------------------------------------------------------------------------
           
def backpropagate(n,root):
    try:
        max_child = max(n.children, key=lambda node:node.value)
        n.value = max_child.value
    except:
        n.value = n.value
    if n != root:
        backpropagate(n.parent,root)
    
#------------------------------------------------------------------------

def stop(gamma):
    return random.random() < gamma

#------------------------------------------------------------------------   
def make_move(root): # cities,budget,n_c,
    
#    if lapse(lambda_):
#        return ramdom_move(s)
#    else:
                
    gamma = 0     
    
    while (not stop(gamma)) and (not determined(root)):
        n = select_node(root)
        expand_node(n,dist_city)
        backpropagate(n,root) 
            
    n =  max(root.children,key=lambda node:node.value)
     
    return n
#--------------------------------------------------------------------------
class Map:
    def __init__(self):#, map_content, map_id): 
        
        self.circle_map()
#        self.load_map(map_content, map_id)

    def circle_map(self):
        # map parameters
        self.N = 30     # total city number, including start
        self.radius = 10     # radius of city
        self.total = 400    # total budget
        self.budget_remain = 400    # remaining budget

        self.R = 400*400 #circle radius' sqaure
        self.r = np.random.uniform(0, self.R, self.N) 
        self.phi = np.random.uniform(0,2 * math.pi, self.N) 
        self.x = np.sqrt(self.r) * np.cos(self.phi) + 1000
        self.x = self.x.astype(int)
        self.y = np.sqrt(self.r) * np.sin(self.phi) + 800
        self.y = self.y.astype(int)
        self.xy = [[self.x[i], self.y[i]] for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000)     # city distance matrix
      
    def load_map(self, map_content, map_id):
        
        self.loadmap = map_content['map_list'][0,map_id][0,0]
        self.order = np.nan
        
        self.N = self.loadmap.N.tolist()[0][0]
        self.radius = 5     # radius of city
        self.total = self.loadmap.total.tolist()[0][0]   # total budget
        self.budget_remain = self.loadmap.total.copy().tolist()[0][0]  # remaining budget()
        
        self.R = self.loadmap.R.tolist()[0]
        self.r = self.loadmap.r.tolist()[0]
        self.phi = self.loadmap.phi.tolist()[0]
        self.x = self.loadmap.x.tolist()[0]
        self.y = self.loadmap.y.tolist()[0]
        self.xy = self.loadmap.xy.tolist()
        
        self.city_start = self.loadmap.city_start.tolist()[0]
        self.distance = self.loadmap.distance.tolist()


trial = Map()
dict_city = dict(zip(list(range(0,trial.N)), trial.xy)) 
dict_city_remain = dict_city.copy()
dist_city = trial.distance.copy()
# -------------------------------------------------------------------------
# main 
start = Node(0)
n_c = 0 # number of connected cities
# calculate start node value
calculate_value(start, dict_city_remain, dist_city, trial.budget_remain, n_c) 
now = start

while not determined(now):
    
    choice = make_move(now)
    now = choice
    print(now.name)
    