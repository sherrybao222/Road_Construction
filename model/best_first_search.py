from anytree import Node
from anytree import RenderTree
from anytree.exporter import JsonExporter, DictExporter
import numpy as np
import math
import random
from scipy.spatial import distance_matrix

import sys

orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f


#----------------------------------------------------------------------
def new_node(name, parent, cities, dist, budget, n_c, weights):
    
    node = Node(name,parent) 
    
    if parent is not None:
        budget_remain = budget - dist[parent.name][node.name]
    else:
        budget_remain = budget
        
    n_cc = n_c + 1
    
    #------------------------------------------------------------------ 
    # calculate n of cities within reach
    n_u = 0
    
    cities_remain = cities.copy()
    del cities_remain[node.name] # delete the current chosen node

    for c in cities_remain:
        if dist[node.name][c] <= budget_remain:
            n_u = n_u + 1

    #------------------------------------------------------------------                            
    value = weights[0] * n_cc + weights[1] * n_u + weights[2] * (budget_remain/100) #+ np.random.normal()

    #------------------------------------------------------------------                            
    node.value = value
    node.budget = budget_remain
    node.n_c = n_cc
    node.city = cities_remain

    #------------------------------------------------------------------    
    if n_u == 0:
        node.determined = 1
    else:
        node.determined = 0
    
    return node

#------------------------------------------------------------------------
def dropfeature(sigma):
    # weight
    w_c = 1
    w_u = 1
    w_b = 1

    if random.random() < sigma:
        w_c = 0
    if random.random() < sigma:
        w_u = 0
    if random.random() < sigma:
        w_b = 0
    return [w_c,w_u,w_b]
       
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
def expand_node(n, dist, theta, weights):
    
    s = n.name
        
    for child in n.city:
        if dist[s][child] <= n.budget:
            new_node(child, n, n.city, dist, n.budget, n.n_c, weights)            
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
def stop(gamma,count):
    return ((random.random() < gamma) or count > 7)
#------------------------------------------------------------------------
def lapse(lambda_):
    return random.random() < lambda_
#------------------------------------------------------------------------
def remove_child(c):
    c.parent = None
    del c
#------------------------------------------------------------------------   
def make_move(s,dist_city): 
    
    if lapse(lambda_):
        return ramdom_move(s,dist_city)
    else:    
    #------------------------------------------------------------------  
        weights = dropfeature(sigma)               
        root = s
        
        count = 0 # count of same selected node
        
        # 1st iteration
        n = select_node(root)
        print('select node: '+ str(n.name))
        
        expand_node(n,dist_city,theta,weights)
        print('expand_node:')
        for pre, _, node in RenderTree(start):
            print("%s%s:%s" % (pre, node.name,node.value))
            
        print('backpropagate:')
        backpropagate(n,root)         
        for pre, _, node in RenderTree(start):
            print("%s%s:%s" % (pre, node.name,node.value))

            
        # from 2nd iteration
        while (not stop(gamma,count)) and (not determined(root)):
            selectnode = n
                
            n = select_node(root)
            
            if n.name == selectnode.name:
                count = count+1
            else:
                count = 0
                
            print('select node: '+ str(n.name))
            
            expand_node(n,dist_city,theta,weights)
            expand_node(n,dist_city,theta,weights)
            print('expand_node:')
            for pre, _, node in RenderTree(start):
                print("%s%s:%s" % (pre, node.name,node.value))
                
            print('backpropagate:')
            backpropagate(n,root) 
            for pre, _, node in RenderTree(start):
                print("%s%s:%s" % (pre, node.name,node.value))
                
        n =  max(root.children,key=lambda node:node.value)
     
    return n
#------------------------------------------------------------------------
def ramdom_move(s,dist):
    candidates = []
    for c in s.city:
        if dist[s.name][c] <= s.budget:
            candidates.append(c)       
    n = new_node(random.choice(candidates), s, s.city, dist, s.budget, s.n_c, [1,1,1])
    return n

#--------------------------------------------------------------------------
class Map:
    def __init__(self):#, map_content, map_id): 
        
        self.circle_map()
#        self.load_map(map_content, map_id)

    def circle_map(self):
        # map parameters
        self.N = 10     # total city number, including start
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

# setting parameters
gamma = 0.01
theta = 15
lambda_ = 0
sigma = 0.01

# -------------------------------------------------------------------------
# main 
if __name__ == "__main__":

    # generate map
    trial = Map()
    dict_city = dict(zip(list(range(0,trial.N)), trial.xy)) 
    dict_city_remain = dict_city.copy()
    dist_city = trial.distance.copy()
    
    start = new_node(0, None, dict_city_remain, dist_city, trial.budget_remain, -1, [1,1,1])
    now = start
    while True:    
        choice = make_move(now,dist_city)
        now = choice
        print('choice: '+ str(now.name))
        for pre, _, node in RenderTree(start):
            print("%s%s:%s" % (pre, node.name,node.value))
        
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1, sharey=True)

        axs.plot(trial.x,trial.y,'o',
       markerfacecolor = '#727bda',markeredgecolor = 'none')
        #ax = sns.heatmap(num_mx,cmap="YlGnBu",linewidths=.3,linecolor = 'k')
        
        axs.set_xlim((-300,300))
        axs.set_ylim((-200,200))
        x0,x1 = axs.get_xlim()
        y0,y1 = axs.get_ylim()
        axs.set_aspect(abs(x1-x0)/abs(y1-y0))
        
        axs.set_facecolor('white')
        
        if now.determined == 1:
            break
        
    
    for pre, _, node in RenderTree(start):
         print("%s%s:%s" % (pre, node.name,node.value))

sys.stdout = orig_stdout
f.close()

#    exporter = DictExporter(attriter=lambda attrs: [(k, v) for k, v in attrs if k == "value"])
#    import dicttoxml  
#    xml = dicttoxml.dicttoxml(exporter.export(start),attr_type=False )   
#    from xml.dom.minidom import parseString
#    dom = parseString(xml)
#    with open("test.xml", "w") as f:
#        f.write(dom.toprettyxml())
         
         
    #from anytree.exporter import DotExporter
    #for line in DotExporter(start):
    #    print(line)
    #
    #from anytree.exporter import UniqueDotExporter
    #UniqueDotExporter(start).to_dotfile("tree.dot")