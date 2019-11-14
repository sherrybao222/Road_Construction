#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from itertools import combinations, permutations
import operator 
import matplotlib.pyplot as plt

def remove_nest(l,output): 
    for i in l: 
        if type(i) == list: 
            remove_nest(i,output) 
        else: 
            output.append(i) 
            
# generate random dot
N = 7 # first one is starting point
x = np.random.rand(N)
y = np.random.rand(N)
budget = 1
city = [(x[i], y[i]) for i in range(0, len(x))] 
name = list(map(str, list(range(0,N))))
dict_city = dict(zip(name, city)) 

# generate distance matrix
matrix = np.zeros((N,N))
for i in name:
    for j in name:
        matrix[int(i)][int(j)] = math.sqrt((dict_city[i][0] - dict_city[j][0])**2 + \
                                 (dict_city[i][1] - dict_city[j][1])**2)

# calculate all path lengths with budget
for i in reversed(range(1,N)):
    pool = combinations(range(1,N), i)
    paths = [list(permutations(x)) for x in pool]
    paths_list = []
    remove_nest(paths,paths_list)
    
    dist = np.zeros(len(paths_list))
    index = 0
    
    for path in paths_list: 
        now = 0 # start city
        for j in range(0,i): # loop all possible cities
            dist[index] = dist[index] + matrix[now][path[j]]
            now = path[j]
        index = index + 1
        
    if any(x<= budget for x in dist):
        n_optimal = i
        break
 
# optimal path       
dict_path = dict(zip(paths_list, dist)) 
sorted_path = sorted(dict_path.items(), key=operator.itemgetter(1))
optimal_index_wozero = sorted_path[0][0]
optimal_index = (0,) + optimal_index_wozero

# greedy with budget
dist_greedy = 0
greedy_index = [0]
i = 0
matrix_copy = matrix.copy()
while i <= N-2:
    dist_list = matrix_copy[greedy_index[i]]
    dist = np.amin(dist_list[dist_list != 0])
    
    if (dist_greedy + dist > budget):
        n_greedy = i
        break
    else:
        dist_greedy = dist_greedy + dist
        index_np = np.where(dist_list == dist)
        matrix_copy[:,greedy_index[i]] = 0
        matrix_copy[greedy_index[i],:] = 0
        i = i + 1
        greedy_index = np.append(greedy_index,index_np[0])
    
diff =  abs(n_optimal - n_greedy)

# greedy all cities
#dist_greedy = 0
#index_greedy = np.zeros(N,dtype = int)
#i = 0
#matrix_copy = matrix.copy()
#while i <= N-2:
#    dist_list = matrix_copy[int(index_greedy[i])]
#    dist = np.amin(dist_list[dist_list != 0])
#    dist_greedy = dist_greedy + dist
#    index_np = np.where(dist_list == dist)
#    matrix_copy[:,int(index_greedy[i])] = 0
#    matrix_copy[int(index_greedy[i]),:] = 0
#    i = i + 1
#    index_greedy[i] = index_np[0]

# draw optimal path
plt.plot(operator.itemgetter(*optimal_index)(x), 
         operator.itemgetter(*optimal_index)(y), 'ro-')
for i,txt in enumerate(optimal_index):
    plt.text(x[txt],y[txt],i, fontsize=11)
plt.plot(x,y,'go')
plt.show()   

# draw greedy path
plt.plot(operator.itemgetter(*greedy_index)(x), 
         operator.itemgetter(*greedy_index)(y), 'bo-')
for i,txt in enumerate(greedy_index):
    plt.text(x[txt],y[txt],i, fontsize=11)
plt.plot(x,y,'go')
plt.show()   