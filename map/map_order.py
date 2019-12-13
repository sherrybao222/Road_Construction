from scipy.spatial import distance_matrix
import numpy as np
import math
import scipy.io as sio
import operator 
import matplotlib.pyplot as plt

# maps
# =============================================================================
class circle_map:    
    def __init__(self):
        # map parameters
        self.N = 11     # total city number, including start
        self.radius = 10     # radius of city
        self.total = 700    # total budget
        self.budget_remain = 700    # remaining budget


        self.R = 450*450 #circle radius' sqaure
        self.r = np.random.uniform(0, self.R, self.N) 
        self.phi = np.random.uniform(0,2 * math.pi, self.N) 
        self.x = np.sqrt(self.r) * np.cos(self.phi) + 1000
        self.x = self.x.astype(int)
        self.y = np.sqrt(self.r) * np.sin(self.phi) + 950
        self.y = self.y.astype(int)
        self.xy = [(self.x[i], self.y[i]) for i in range(0, len(self.x))]   # combine x and y
        
        self.city_start = self.xy[0]    # start city
        self.distance = distance_matrix(self.xy, self.xy, p=2, threshold=10000)     # city distance matrix

# paths
# =============================================================================      
# greedy all cities
def greedy(mmap):
    dist_greedy = 0
    index_greedy = np.zeros(mmap.N,dtype = int)
    i = 0
    matrix_copy = mmap.distance.copy()
    while i <= mmap.N-2:
        dist_list = matrix_copy[int(index_greedy[i])]
        dist = np.amin(dist_list[dist_list != 0])
        dist_greedy = dist_greedy + dist
        index_np = np.where(dist_list == dist)
        matrix_copy[:,int(index_greedy[i])] = 0
        matrix_copy[int(index_greedy[i]),:] = 0
        i = i + 1
        index_greedy[i] = index_np[0]
    
    return index_greedy

# main
# =============================================================================  
n_map = 2 # number of maps needed
map_list = []
order_list = []

for i in range(0,n_map):
    mmap = circle_map()
    map_list.append(mmap)
    greedy_index = greedy(mmap) 
    order_list.append(greedy_index)
    
    # draw
#    plt.plot(operator.itemgetter(*order_list[i])(mmap.x), 
#             operator.itemgetter(*order_list[i])(mmap.y), 'bo-')
#    for i,txt in enumerate(order_list[i]):
#        plt.text(mmap.x[txt],mmap.y[txt],i, fontsize=11)
#    plt.plot(mmap.x,mmap.y,'go')
#    plt.show() 
   
# saving
sio.savemat('test.mat', {'map_list':map_list,'order_list':order_list})
