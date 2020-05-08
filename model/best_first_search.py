import numpy as np
from anytree import Node
import math
import random
from scipy.spatial import distance_matrix

#----------------------------------------------------------------------
def calculate_value(node, cities, dist, budget, n_c):
    # weight
    w_c = 1
    w_u = 1
    w_b = 1
    
    #------------------------------------------------------------------ 
    # calculate n of cities within reach
    n_u = 0
    
    cities_remain = cities.copy()
    del cities_remain[node.name] # delete the current chosen node

    for c in cities_remain:
        if dist[node.name][c] <= budget:
            n_u = n_u + 1
            
    #------------------------------------------------------------------    
    if n_u == 0:
        node.determined = 1
    else:
        node.determined = 0
    
    value = w_c * n_c + w_u * n_u + w_b * (budget/100) + np.random.normal()
    
    node.value = value
    node.budget = budget
    node.n_c = n_c
    node.city = cities_remain
    
#------------------------------------------------------------------------
def determined(root):
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
def expand_node(n, dist, theta):
    
    s = n.name
        
    for child in n.city:
        if dist[s][child] <= n.budget:
            c = Node(child, parent = n) # child node
            budget_remain = n.budget - dist[s][child]
            n_cc = n.n_c + 1
            calculate_value(c, n.city, dist, budget_remain, n_cc)
            
    #------------------------------------------------------------------ 
    # pruning    
    try:
        V_max = max(n.children, key=lambda node:node.value)
        
        for c in n.children:
            if abs(c.value - V_max.value) > theta:
                remove_child(c)
    except:
        pass
 
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
def lapse(lambda_):
    return random.random() < lambda_
#------------------------------------------------------------------------
def remove_child(c):
    c.parent = None
    del c
#------------------------------------------------------------------------   
def make_move(s): 
    
    if lapse(lambda_):
        return ramdom_move(s,dist_city)
    else:    
    #------------------------------------------------------------------                 
        root = s
        
        while (not stop(gamma)) and (not determined(root)):
            n = select_node(root)
            expand_node(n,dist_city,theta)
            backpropagate(n,root) 
                
        n =  max(root.children,key=lambda node:node.value)
     
    return n
#------------------------------------------------------------------------
def ramdom_move(s,dist):
    candidates = []
    for c in s.city:
        if dist[s.name][c] <= s.budget:
            candidates.append(c)   
    n = Node(random.choice(candidates), parent = s)
    budget_remain = s.budget - dist[s.name][n.name]
    n_cc = s.n_c + 1
    calculate_value(n, s.city, dist, budget_remain, n_cc)

    return n

#--------------------------------------------------------------------------
class Map:
    def __init__(self):#, map_content, map_id): 
        
        self.circle_map()
#        self.load_map(map_content, map_id)

    def circle_map(self):
        # map parameters
        self.N = 15     # total city number, including start
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

# setting parameters
gamma = 0.1
theta = 5
lambda_ = 0.1

# generate map
trial = Map()
dict_city = dict(zip(list(range(0,trial.N)), trial.xy)) 
dict_city_remain = dict_city.copy()
dist_city = trial.distance.copy()

# -------------------------------------------------------------------------
# main 
start = Node(0)
n_c = 0 # number of connected cities
calculate_value(start, dict_city_remain, 
                dist_city, trial.budget_remain, n_c) # calculate start node value
now = start

while True:    
    choice = make_move(now)
    now = choice
    try:  print(now.name)
    except: print(now)
   
    if now.determined == 1:
        break

from anytree import RenderTree
for pre, _, node in RenderTree(start):
     print("%s%s:%s" % (pre, node.name,node.value))
     
#from anytree.exporter import DotExporter
#for line in DotExporter(start):
#    print(line)
#
#from anytree.exporter import UniqueDotExporter
#UniqueDotExporter(start).to_dotfile("tree.dot")