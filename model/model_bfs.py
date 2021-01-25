from anytree import Node
from anytree import RenderTree
#from anytree.exporter import JsonExporter, DictExporter
import random
import numpy as np
from map_class import Map
import math

#----------------------------------------------------------------------
class params:
	def __init__(self, w1, w2, w3,
					stopping_probability,
					pruning_threshold,
					lapse_rate,
					feature_dropping_rate,
					count_par = 10):
		self.weights = [w1, w2, w3]
		self.feature_dropping_rate = feature_dropping_rate
		self.stopping_probability = stopping_probability
		self.pruning_threshold = pruning_threshold
		self.lapse_rate = lapse_rate
		self.count_par = count_par
        
#----------------------------------------------------------------------
def new_node_current(name, cities, dist, budget, n_c, weights, **kwargs):
    
    '''
    creat new tree node with current node's information
    '''
    
    node = Node(name,None) 
       
    #------------------------------------------------------------------ 
    # calculate n of cities within reach    
    if len(kwargs) != 0:
        for key, value in kwargs.items():
            if key == "n_u":
                n_u = value
    else:
        n_u = 0
        cities_remain = cities.copy()
    
        try: del cities_remain[node.name] # delete the current chosen node
        except: pass

        for c in cities_remain:
            if dist[node.name][c] <= budget:
                n_u = n_u + 1
    node.n_u = n_u
    
    #------------------------------------------------------------------                            
    try: node.city = cities_remain # all cities, not only cities within reach
    except: node.city = cities
        
    node.budget = budget
    node.n_c = n_c

    #------------------------------------------------------------------                            
    # value = weights[0] * node.n_c + weights[1] * node.n_u + \
    #         weights[2] * (node.budget/30) + np.random.normal()
            
    # if node.budget == 300:
    #     value = weights[1] * node.n_u/pow(node.budget,2) * 10000 + \
    #         np.random.normal()
    # else:
    #     value = weights[0] * node.n_c/(300 - node.budget) * 100 + \
    #         weights[1] * node.n_u/pow(node.budget,2) * 10000 + \
    #         np.random.normal()

    # value = weights[0] * node.n_c + weights[1] * node.n_u + \
    #         np.random.normal()
            
    value = weights[0] * node.n_c - weights[2] * ((300 - node.budget)/30) + \
            np.random.normal()

            
    node.value = value

    #------------------------------------------------------------------    
    if node.n_u == 0:
        node.determined = 1
    else:
        node.determined = 0
    
    return node

def new_node_previous(name, parent, dist, weights):
    
    '''
    creat new tree node with parent node's information
    '''
    node = Node(name,parent) 
    
    node.budget = parent.budget - dist[parent.name][node.name]
    node.n_c = parent.n_c + 1

    #------------------------------------------------------------------ 
    # calculate n of cities within reach    
    n_u = 0
    cities_remain = parent.city.copy()
    
    try: del cities_remain[node.name] # delete the current chosen node
    except: pass

    for c in cities_remain:
        if dist[node.name][c] <= node.budget:
            n_u = n_u + 1
            
    node.n_u = n_u
    node.city = cities_remain # all cities, not only cities within reach
       
    #------------------------------------------------------------------                            
    # value = weights[0] * node.n_c + weights[1] * node.n_u + \
    #         weights[2] * (node.budget/100) + np.random.normal()
            
    # if node.budget == 300:
    #     value = weights[1] * node.n_u/pow(node.budget,2) * 10000 + \
    #         np.random.normal()
    # else:
    #     value = weights[0] * node.n_c/(300 - node.budget) * 100 + \
    #         weights[1] * node.n_u/pow(node.budget,2) * 10000 + \
    #         np.random.normal()
    
    # value = weights[0] * node.n_c + weights[1] * node.n_u + \
    #         np.random.normal()
    
    value = weights[0] * node.n_c - weights[2] * ((300 - node.budget)/30) + \
            np.random.normal()

    node.value = value

    #------------------------------------------------------------------    
    if node.n_u == 0:
        node.determined = 1
    else:
        node.determined = 0
    
    return node

#------------------------------------------------------------------------
def dropfeature(set_weights,sigma):
    new_weights = set_weights.copy()
    
    if random.random() < sigma:
        new_weights[0] = 0
    if random.random() < sigma:
        new_weights[1] = 0
    if random.random() < sigma:
        new_weights[2] = 0
    return new_weights
       
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
            new_node_previous(child, n, dist, weights)            
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
def stop(gamma,count,count_par):
    return ((random.random() < gamma) or count > count_par)
#------------------------------------------------------------------------
def lapse(lambda_):
    return random.random() < lambda_
#------------------------------------------------------------------------
def remove_child(c):
    c.parent = None
    del c
#------------------------------------------------------------------------   
def make_move(s,dist_city,para): 
    
    '''
    main function of best-first tree search
    '''
    
    if lapse(para.lapse_rate):
        return ramdom_move(s,dist_city,para.weights)
    else:    
    #------------------------------------------------------------------  
        new_weights = dropfeature(para.weights,para.feature_dropping_rate)              
        root = s
        
        count = 0 # count of same selected node
        
        # 1st iteration
        if (not determined(root)):
            n = select_node(root)
#            print('select node: '+ str(n.name))
            
            expand_node(n,dist_city,para.pruning_threshold,new_weights)
#            print('expand_node:')
#            for pre, _, node in RenderTree(start):
#                print("%s%s:%s" % (pre, node.name,node.value))
                
            backpropagate(n,root)   
#            print('backpropagate:')
#            for pre, _, node in RenderTree(start):
#                print("%s%s:%s" % (pre, node.name,node.value))

            selectnode = max(root.children,key=lambda node:node.value)
            
        # from 2nd iteration
        while (not stop(para.stopping_probability,count,para.count_par)) and (not determined(root)):                
            n = select_node(root)
            
                
#            print('select node: '+ str(n.name))
            
            expand_node(n,dist_city,para.pruning_threshold,new_weights)
#            print('expand_node:')
#            for pre, _, node in RenderTree(start):
#                print("%s%s:%s" % (pre, node.name,node.value))
                
            backpropagate(n,root) 
#            print('backpropagate:')
#            for pre, _, node in RenderTree(start):
#                print("%s%s:%s" % (pre, node.name,node.value))

            new_selectnode = max(root.children,key=lambda node:node.value)
            
            if new_selectnode == selectnode:
                count = count+1
            else:
                count = 0
            
            selectnode = new_selectnode

                
        max_ =  max(root.children,key=lambda node:node.value)
     
    return max_
#------------------------------------------------------------------------
def ramdom_move(s,dist,set_weights):
    candidates = []
    for c in s.city:
        if dist[s.name][c] <= s.budget:
            candidates.append(c)       
    n = new_node_previous(random.choice(candidates), s, dist, set_weights)
    return n

# -------------------------------------------------------------------------
if __name__ == "__main__":
    
    '''
    model simulation for a map
    '''
    #--------------------------------------------------------------------------
    # set parameters
    inparams = [1, 1, 1, 0.01, 15, 0.01, 0.01]
    para = params(w1=inparams[0], w2=inparams[1], w3=inparams[2], 
    					stopping_probability=inparams[3],
    					pruning_threshold=inparams[4],
    					lapse_rate=inparams[5], feature_dropping_rate=inparams[6])

    #--------------------------------------------------------------------------
    # generate map
    trial = Map()
    dict_city = dict(zip(list(range(0,trial.N)), trial.xy)) 
    dict_city_remain = dict_city.copy()
    dist_city = trial.distance.copy()
    
    # save console output 2/1   
    # import sys
    # orig_stdout = sys.stdout
    # f = open('output.txt', 'w')
    # sys.stdout = f

    # simulate
    choice_sequence = [0]
    start = new_node_current(0, dict_city_remain, dist_city, trial.budget_remain, 0, para.weights)
    now = start
    while True:    
        choice = make_move(now,dist_city,para)
        print('choice: '+ str(choice.name))
        choice_sequence.append(choice.name)
        
        # visualize tree                 
        for pre, _, node in RenderTree(now):
            print("%s%s:%s" % (pre, node.name,node.value))
        
        if choice.determined == 1:
            break
        
        new_start = new_node_current(choice.name, choice.city, dist_city, 
                                     choice.budget, choice.n_c, para.weights)        
        now = new_start

    # save console output 2/2
    # sys.stdout = orig_stdout
    # f.close()

# -------------------------------------------------------------------------
    # visualize map     
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 1, sharey=True)

    axs.plot(trial.x,trial.y,'o', markerfacecolor = 'k',markeredgecolor = 'none')
    axs.plot(trial.x[0],trial.y[0],'o', markerfacecolor = '#FF6666',markeredgecolor = 'none')
    for i in range(0,trial.N):
        plt.text(trial.x[i]+10,trial.y[i]+10, i, fontsize=7)
        
    import operator         
    plt.plot(operator.itemgetter(*choice_sequence)(trial.x), 
         operator.itemgetter(*choice_sequence)(trial.y), 'b-')

    axs.set_xlim((-300,300))
    axs.set_ylim((-200,200))
    x0,x1 = axs.get_xlim()
    y0,y1 = axs.get_ylim()
    axs.set_aspect(abs(x1-x0)/abs(y1-y0))
    
    axs.set_facecolor('white')
    plt.axis('off')
    
    fig.set_figwidth(10)
    plt.show()
    
    

# -------------------------------------------------------------------------
#  convert to xml
#    import dict2xml
#    exporter = DictExporter(attriter=lambda attrs: [(k, v) for k, v in attrs if k == "name"])
#    xml = dict2xml.dict2xml(exporter.export(start))   
#    with open("test.xml", "w") as f:
#        f.write(xml)
         
# -------------------------------------------------------------------------
#  convert to dot
#    from anytree.exporter import DotExporter
#    for line in DotExporter(start):
#        print(line)
#    
#    from anytree.exporter import UniqueDotExporter
#    UniqueDotExporter(start).to_dotfile("tree.dot")