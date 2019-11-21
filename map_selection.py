import random
from scipy.spatial import distance_matrix
from itertools import combinations, permutations
import numpy as np
import operator 
import matplotlib.pyplot as plt

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
def remove_nest(l,output): 
    for i in l: 
        if type(i) == list: 
            remove_nest(i,output) 
        else: 
            output.append(i) 

#------------------------------------------------------------------------------            
# calculate all path lengths within budget
def calculate_all(mmap):
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
        
    return dist, paths_list, n_optimal

#------------------------------------------------------------------------------        
# optimal path       
def optimal(dist, paths_list):
    dict_path = dict(zip(paths_list, dist)) 
    sorted_path = sorted(dict_path.items(), key=operator.itemgetter(1))
    optimal_index_wozero = sorted_path[0][0]
    optimal_index = (0,) + optimal_index_wozero
    
    return optimal_index

#------------------------------------------------------------------------------
# greedy with budget
def greedy(mmap):
    dist_greedy = 0
    greedy_index = [0]
    i = 0
    matrix_copy = mmap.distance.copy()
    while i <= mmap.N-2:
        dist_list = matrix_copy[greedy_index[i]]
        dist = np.amin(dist_list[dist_list != 0])
    
        if (dist_greedy + dist > mmap.total):
            break
        else:
            dist_greedy = dist_greedy + dist
            index_np = np.where(dist_list == dist)
            matrix_copy[:,greedy_index[i]] = 0
            matrix_copy[greedy_index[i],:] = 0
            i = i + 1
            n_greedy = i
            greedy_index = np.append(greedy_index,index_np[0])     
            
    return n_greedy, greedy_index

#------------------------------------------------------------------------------
mmap = gaussian_map()
dist, paths_list, n_optimal = calculate_all(mmap)   
optimal_index = optimal(dist, paths_list) 
n_greedy, greedy_index = greedy(mmap)

diff =  abs(n_optimal - n_greedy)

# draw 
plt.plot(operator.itemgetter(*optimal_index)(mmap.x), 
         operator.itemgetter(*optimal_index)(mmap.y), 'ro-')
plt.plot(operator.itemgetter(*greedy_index)(mmap.x), 
         operator.itemgetter(*greedy_index)(mmap.y), 'bo-')

plt.plot(mmap.x[1:],mmap.y[1:],'go')
plt.plot(mmap.x[0],mmap.y[0],'cv')

plt.axis('off')

plt.show()   


