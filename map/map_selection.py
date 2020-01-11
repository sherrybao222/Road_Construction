import random
from scipy.spatial import distance_matrix
from itertools import combinations, permutations
import numpy as np
import operator 
import math
import matplotlib.pyplot as plt
import scipy.io as sio
from anytree import Node
from anytree.exporter import DictExporter
# helper functions
# =============================================================================
# remove list nesting
def remove_nest(l,output): 
    for i in l: 
        if type(i) == list: 
            remove_nest(i,output) 
        else: 
            output.append(i) 
            
# different maps
# =============================================================================
class uniform_map:
    def __init__(self): 
        # map parameters
        self.N = 11 # total city number, including start
        self.total = 700 # total budget
        
        self.x = random.sample(range(51, 649), self.N) # x axis of all cities
        self.y = random.sample(range(51, 649), self.N) # y axis of all cities
        self.xy = [[self.x[i], self.y[i]] for i in range(0, len(self.x))] # combine x and y
   
        self.city_start = self.xy[0] # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000) # city distance matrix
#------------------------------------------------------------------------------
class gaussian_map:
    def __init__(self):
        # map parameters
        self.N = 11 # total city number, including start
        self.total = 100 # total budget
        
        mean = [0, 0]
        cov = [[33333, 0], [0, 33333]]  # diagonal covariance
        
        self.xy = np.random.multivariate_normal(mean, cov, self.N)
        self.x, self.y = self.xy.T #transpose
   
        self.city_start = self.xy[0] # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000) # city distance matrix

#------------------------------------------------------------------------------
class circle_map:    
    def __init__(self):
        # map parameters
        self.N = 30     # total city number, including start
        self.total = 400    # total budget

        self.R = 400*400 #circle radius' sqaure
        self.r = np.random.uniform(0, self.R, self.N) 
        self.phi = np.random.uniform(0,2 * math.pi, self.N) 
        self.x = np.sqrt(self.r) * np.cos(self.phi) + 1000
        self.x = self.x.astype(int)
        self.y = np.sqrt(self.r) * np.sin(self.phi) + 800
        self.y = self.y.astype(int)
        self.xy = [(self.x[i], self.y[i]) for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000)     # city distance matrix

# different paths
# =============================================================================         
# optimal within budget
def optimal_(mmap):
    # calculate all possible paths within budget
    for i in reversed(range(1,mmap.N)):
        pool = combinations(range(1,mmap.N), i)
        paths = [list(permutations(x)) for x in pool]
        paths_list = []
        remove_nest(paths,paths_list)
    
        dist = np.zeros(len(paths_list))
        index = 0
    
        for path in paths_list: 
            now = 0 # start city
            for j in range(0,i): # loop all possible cities
                dist[index] = dist[index] + mmap.distance[now][path[j]]
                now = path[j]
                
            index = index + 1
        
        if any(x<= mmap.total for x in dist):
            n_optimal = i
            break
    
    dict_path = dict(zip(paths_list, dist)) 
    sorted_path = sorted(dict_path.items(), key=operator.itemgetter(1))
    optimal_index_wozero = sorted_path[0][0]
    optimal_index = (0,) + optimal_index_wozero   
    
    return n_optimal, optimal_index, sorted_path
#------------------------------------------------------------------------------

def optimal(mmap,now):
    # breath first search (only budget)
    all_ = list(range(0, mmap.N))
    for j in now.path:
        all_.remove(j.name)
        
    for i in all_:
        budget_remain = now.budget - mmap.distance[now.name][i]
        if budget_remain >= 0:
            node = Node(i, parent = now, budget = budget_remain)
            optimal(mmap,node)     
        
#------------------------------------------------------------------------------
# greedy within budget
def greedy(mmap):
    dist_greedy = 0 # the current distance sum of greedy path
    greedy_index = [0] # greedy path starting index
    i = 0 # current connected city number
    matrix_copy = mmap.distance.copy()
    while i <= mmap.N-2:
        dist_list = matrix_copy[greedy_index[i]] # choose the related column/row
        dist = np.amin(dist_list[dist_list != 0]) # the smallest non-zero distance
    
        if (dist_greedy + dist > mmap.total): 
            break
        else:
            dist_greedy = dist_greedy + dist # update current distance sum of greedy path
            index_np = np.where(dist_list == dist) # find the chosen city index
            matrix_copy[:,greedy_index[i]] = 0 # cannot choose one city twice
            matrix_copy[greedy_index[i],:] = 0
            i = i + 1
            n_greedy = i
            greedy_index = np.append(greedy_index,index_np[0])     
            
    return n_greedy, greedy_index

# main
# =============================================================================   
n_map = 5 # number of maps needed
map_list = []
optimal_list = []
greedy_list = []
optimal_number = []
greedy_number = []
diff_list = []
exporter = DictExporter()

while True:
    mmap = circle_map()
    root_ = Node(0, budget = mmap.total)
    optimal(mmap,root_) 
    n_greedy, greedy_index = greedy(mmap)
    diff =  abs(root_.height - n_greedy)
    
    if diff >= 1:
        map_list.append(mmap)
        optimal_list.append(exporter.export(root_))
        greedy_list.append(greedy_index)
        optimal_number.append(root_.height)
        greedy_number.append(n_greedy)
        diff_list.append(diff)
        
    if len(map_list) == n_map:
        break
    
   
# saving
sio.savemat('test_basic.mat', {'map_list':map_list, 'diff_list':diff_list,
                                    'optimal_list':optimal_list,'greedy_list':greedy_list,
                                    'optimal_number':optimal_number,'greedy_number':greedy_number})

# draw 
#plt.plot(operator.itemgetter(*optimal_index)(mmap.x), 
#         operator.itemgetter(*optimal_index)(mmap.y), 'ro-')
#plt.plot(operator.itemgetter(*greedy_index)(mmap.x), 
#         operator.itemgetter(*greedy_index)(mmap.y), 'bo-')
#
#plt.plot(mmap.x[1:],mmap.y[1:],'go')
#plt.plot(mmap.x[0],mmap.y[0],'cv')
#
#plt.gca().set_aspect('equal', adjustable='box')
#plt.axis('scaled')
#
#plt.show()
#print(diff)


