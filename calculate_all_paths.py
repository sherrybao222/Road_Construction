#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from itertools import permutations 
import operator 



# generate random dot
N = 7 # first one is starting point
x = np.random.rand(N)
y = np.random.rand(N)
city = [(x[i], y[i]) for i in range(0, len(x))] 
name = list(map(str, list(range(0,N))))
dict_city = dict(zip(name, city)) 

# generate distance matrix
matrix = np.zeros((N,N))
for i in name:
    for j in name:
        matrix[int(i)][int(j)] = math.sqrt((dict_city[i][0] - dict_city[j][0])**2 + \
                                 (dict_city[i][1] - dict_city[j][1])**2)

# calculate path length
paths = permutations(range(1,N)) 
paths_list = list(paths)
dist = np.zeros(len(paths_list))
index = 0
for path in paths_list: 
    now = 0 
    dist[index] = 0
    for i in range(0,N-1):
        dist[index] = dist[index] + matrix[now][path[i]]
        now = path[i]
    index = index + 1

dict_path = dict(zip(paths_list, dist)) 
sorted_path = sorted(dict_path.items(), key=operator.itemgetter(1))

# greedy
dist_greedy = 0
index_greedy = np.zeros(N)
i = 0
matrix_copy = matrix.copy()
while i <= N-2:
    dist_list = matrix_copy[int(index_greedy[i])]
    dist = np.amin(dist_list[dist_list != 0])
    dist_greedy = dist_greedy + dist
    index_np = np.where(dist_list == dist)
    matrix_copy[:,int(index_greedy[i])] = 0
    matrix_copy[int(index_greedy[i]),:] = 0
    i = i + 1
    index_greedy[i] = index_np[0]
    