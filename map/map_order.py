from scipy.spatial import distance_matrix
import numpy as np
import math
import scipy.io as sio
import operator 
from itertools import permutations 
import random
import matplotlib.pyplot as plt
import scipy
import scipy.optimize as optimize
from itertools import chain



# maps
# =============================================================================
class circle_map:    
    def __init__(self, budget):
        # map parameters
        self.N = 11    # total city number, including start
        self.radius = 10     # radius of city
        self.total = budget    # total budget
        self.budget_remain = budget    # remaining budget


        self.R = 400*400 #circle radius' sqaure
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
    
    return index_greedy, 'greedy'

# =============================================================================
# optimal all cities
# calculate path length
def optimal(mmap):
    paths = permutations(range(1,mmap.N)) 
    paths_list = list(paths)
    dist = np.zeros(len(paths_list))
    index = 0
    for path in paths_list: 
        now = 0 
        dist[index] = 0
        for i in range(0,mmap.N-1):
            dist[index] = dist[index] + mmap.distance[now][path[i]]
            now = path[i]
        index = index + 1

    dict_path = dict(zip(paths_list, dist)) 
    sorted_path = sorted(dict_path.items(), key=operator.itemgetter(1))
    optimal_index_wozero = sorted_path[0][0]
    optimal_index = (0,) + optimal_index_wozero
    
    return optimal_index, 'optimal'

# =============================================================================
# repulsive force field
def field(pos, city_x,city_y,sigma):
    x,y = pos 
    return sum(scipy.exp(-((city_x - x)**2+(city_y - y)**2)/(2*sigma**2))/(2*math.pi*sigma**2))

def field_pos(mmap):
    position = []
    for i in range(0,mmap.N):
#        initial_guess = [1000, 750]
#        cons = {'type': 'eq', 
#            'fun': lambda pos: (pos[0] - mmap.x[i])**2 + (pos[1] - mmap.y[i])**2 - 20**2}
#        result = optimize.minimize(field, initial_guess, 
#                                   args=(mmap.x,mmap.y,50),constraints=cons)
        bounds = [(mmap.x[i]-20, mmap.x[i]+20), (mmap.y[i]-20, mmap.y[i]+20)]
        result = optimize.shgo(field, bounds, args=(mmap.x,mmap.y,20))
        position.append(result.x)
    return position


# main
# =============================================================================  
n_map = 4 # number of maps needed
map_ind = int(n_map/2)*[0]
map_ind.extend(int(n_map/2)*[0])
random.shuffle(map_ind)
map_list = []
order_list = []
name_list = []
pos_list = []
budget_list = [400,700]
i = 0
while True:
    budget_index = i%len(budget_list)

    mmap = circle_map(budget_list[budget_index])
    
    distance_copy = mmap.distance.copy()
    distance_copy = chain.from_iterable(zip(*distance_copy))
    
    if any(i < 10 and i != 0 for i in list(distance_copy)):
        continue    
    pos = field_pos(mmap)
    if map_ind[i] == 0:
        [index,name] = greedy(mmap)
    else:
        [index,name] = optimal(mmap)
    map_list.append(mmap)
    order_list.append(index)
    name_list.append(name)
    pos_list.append(pos)
    i = i + 1
    if len(map_list) == n_map:
        break

    # draw
#    plt.plot(operator.itemgetter(*order_list[i])(mmap.x), 
#             operator.itemgetter(*order_list[i])(mmap.y), 'bo-')
#    for i,txt in enumerate(order_list[i]):
#        plt.text(mmap.x[txt],mmap.y[txt],i, fontsize=11)
#    plt.plot(mmap.x,mmap.y,'go')
#    plt.show() 
   
# saving
sio.savemat('test.mat', {'map_list':map_list,'order_list':order_list,'name_list':name_list,'pos_list':pos_list})
