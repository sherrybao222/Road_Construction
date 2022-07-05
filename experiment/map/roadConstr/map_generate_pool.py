'''
generate a large number of maps as a pool to choose later. 
(we only choose maps with a diffence == 4 of greedy solution and optimal solution. See map_filterDiff4.py)
'''
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
from itertools import chain
from anytree import Walker
import json

save_path = '/Users/dbao/google_drive_db/road_construction/data/test_2021/experiment/map/map_pool/'

# helper functions
# =============================================================================
# remove list nesting
def remove_nest(l,output): 
    for i in l: 
        if type(i) == list: 
            remove_nest(i,output) 
        else: 
            output.append(i) 
            
# different kinds of maps (we use circle_maps in the end)
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
        self.N = 30 # total city number, including start
        self.total = 300 # total budget
        
        mean = [0, 0]
        cov = [[200, 0], [0, 200]]  # diagonal covariance
        
        self.xy = np.random.multivariate_normal(mean, cov, self.N)
        self.x, self.y = self.xy.T #transpose
   
        self.city_start = self.xy[0] # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000) # city distance matrix

#------------------------------------------------------------------------------
class circle_map:    
    def __init__(self):
        # map parameters
        self.N = 30     # total city number, including start
        self.total = 300    # total budget

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

# algorithms for calculating different path solutions
# =============================================================================         
# optimal algorithm 1: calculate all possible paths
def optimal_(mmap):
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
# optimal algorithm 2: use tree search (we use this method in the end)
def optimal(mmap,now):
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
            if len(index_np[0]) > 1:
                index_np = random.choice(index_np)
            matrix_copy[:,greedy_index[i]] = 0 # cannot choose one city twice
            matrix_copy[greedy_index[i],:] = 0
            i = i + 1
            n_greedy = i
            greedy_index = np.append(greedy_index,index_np[0])     
            
    return n_greedy, greedy_index.tolist()

# main function to run
# =============================================================================   
n_map = 500 # number of maps needed
map_list = []
breath_first_tree = []
optimal_list = []
greedy_list = []
optimal_number = []
greedy_number = []
diff_list = []
exporter = DictExporter()
index = 2500 # index for labeling maps

while True:
    mmap = circle_map()
    
    distance_copy = mmap.distance.copy()
    distance_copy = chain.from_iterable(zip(*distance_copy))
    
    if any(i < 10 and i != 0 for i in list(distance_copy)): # exclude the map if cities are too close. 
        continue

    # get optimal solution
    root_ = Node(0, budget = mmap.total)
    optimal(mmap,root_) 
    # get greedy solution
    n_greedy, greedy_index = greedy(mmap)
    # calculate difference
    diff =  abs(root_.height - n_greedy)
    
    if diff >= 1: # also exclude maps which have no difference at all

        # make mmap json serializable
        mmap.distance = mmap.distance.tolist()
        mmap = mmap.__dict__ 

        map_list.append(mmap)
        
        depth = []
        leaves = root_.leaves
        for leave in leaves:
            depth.append(leave.depth)
#        plt.hist(depth,range(len(set(depth))+2),align='left') # plot path length summary for the current map
#        plt.savefig('path_length/path_length_' + str(index) + '.png')
#        plt.clf()
        index = index + 1

        # get optimal path
        w = Walker()
        path = w.walk(root_, leaves[depth.index(max(depth))])
        optimal_index = [0]
        for item in path[2]:
            optimal_index.append(item.name)

        # save
        breath_first_tree.append(exporter.export(root_))

        optimal_list.append(optimal_index)
        greedy_list.append(greedy_index)

        optimal_number.append(root_.height)
        greedy_number.append(n_greedy)

        diff_list.append(diff)
        
        print(index)
    if len(map_list) == n_map:
        break

with open(save_path+'basic_map_2500','w') as file: 
    json.dump(map_list,file)
with open(save_path+'basic_tree_2500','w') as file: 
    json.dump(breath_first_tree,file)
with open(save_path+'basic_summary_2500','w') as file: 
    json.dump((diff_list,optimal_list,greedy_list,optimal_number,greedy_number),file) 

# saving yaml
#import yaml
#with open('basic_map', 'w') as file:
#    yaml.dump(map_list, file)
#with open('basic_tree','w') as file: 
#    yaml.dump(breath_first_tree,file)
#with open('basic_summary','w') as file: 
#    yaml.dump((diff_list,optimal_list,greedy_list,optimal_number,greedy_number),file) 
   
# saving mat file
#sio.savemat('basic_map_test.mat', {'map_list':map_list})
#sio.savemat('basic_tree_test.mat', {'breath_first_tree':breath_first_tree})
#sio.savemat('basic_summary_test.mat', {'diff_list':diff_list,
#                                    'optimal_list':optimal_list,'greedy_list':greedy_list,
#                                    'optimal_number':optimal_number,'greedy_number':greedy_number})

## draw 
#plt.plot(operator.itemgetter(*optimal_index)(map_list[0].x), 
#         operator.itemgetter(*optimal_index)(map_list[0].y), 'ro-')
#plt.plot(operator.itemgetter(*greedy_list[0])(map_list[0].x), 
#         operator.itemgetter(*greedy_list[0])(map_list[0].y), 'bo-')
#
#plt.plot(map_list[0].x[1:],map_list[0].y[1:],'go')
#plt.plot(map_list[0].x[0],map_list[0].y[0],'cv')
#
#plt.xlim(0, 1500)
#plt.ylim(0, 1500)
#plt.gca().set_aspect('equal', adjustable='box')
##plt.axis('scaled')
#
#plt.show()
##print(diff)
    
#with open('basic_map.yaml') as file:
#    # The FullLoader parameter handles the conversion from YAML
#    # scalar values to Python the dictionary format
#    fruits_list = yaml.load(file,Loader=yaml.Loader)