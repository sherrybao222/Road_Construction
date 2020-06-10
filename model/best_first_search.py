from anytree import Node
from anytree import RenderTree
#from anytree.exporter import JsonExporter, DictExporter
import random
import numpy as np
from map_class import Map

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
def make_move(s,dist_city): 
    
    if lapse(lambda_):
        return ramdom_move(s,dist_city)
    else:    
    #------------------------------------------------------------------  
        weights = dropfeature(set_weights,sigma)              
        root = s
        
        count = 0 # count of same selected node
        
        # 1st iteration
        if (not determined(root)):
            n = select_node(root)
            print('select node: '+ str(n.name))
            
            expand_node(n,dist_city,theta,weights)
#            print('expand_node:')
#            for pre, _, node in RenderTree(start):
#                print("%s%s:%s" % (pre, node.name,node.value))
                
            backpropagate(n,root)   
#            print('backpropagate:')
#            for pre, _, node in RenderTree(start):
#                print("%s%s:%s" % (pre, node.name,node.value))

            selectnode = max(root.children,key=lambda node:node.value)
            
        # from 2nd iteration
        while (not stop(gamma,count,count_par)) and (not determined(root)):                
            n = select_node(root)
            
                
            print('select node: '+ str(n.name))
            
            expand_node(n,dist_city,theta,weights)
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
def ramdom_move(s,dist):
    candidates = []
    for c in s.city:
        if dist[s.name][c] <= s.budget:
            candidates.append(c)       
    n = new_node(random.choice(candidates), s, s.city, dist, s.budget, s.n_c, set_weights)
    return n

#--------------------------------------------------------------------------
# setting parameters
gamma = 0.01 # stopping probability
theta = 15 # prunning criteria
lambda_ = 0 # lapse rate
sigma = 0.01 # drop feature probability
count_par = 5
set_weights = [1,1,1]

# -------------------------------------------------------------------------
# main 
if __name__ == "__main__":

    # save console output 2/2    
    import sys
    orig_stdout = sys.stdout
    f = open('output.txt', 'w')
    sys.stdout = f

    # generate map
    trial = Map()
    dict_city = dict(zip(list(range(0,trial.N)), trial.xy)) 
    dict_city_remain = dict_city.copy()
    dist_city = trial.distance.copy()
    
    # simulate
    choice_sequence = [0]
    start = new_node(0, None, dict_city_remain, dist_city, trial.budget_remain, -1, set_weights)
    now = start
    while True:    
        choice = make_move(now,dist_city)
        print('choice: '+ str(choice.name))
        choice_sequence.append(choice.name)
        
        # visualize tree                 
        for pre, _, node in RenderTree(now):
            print("%s%s:%s" % (pre, node.name,node.value))
        
        if choice.determined == 1:
            break
        
        new_start = new_node(choice.name, None, now.city, dist_city, now.budget, now.n_c, set_weights)
        now = new_start

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
    
    # save console output 2/2
    sys.stdout = orig_stdout
    f.close()
    

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