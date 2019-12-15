#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from anytree import Node
import math
import random

whole_budget = 1
budget_remain = whole_budget

# generate random dot
N = 10
x = np.random.rand(N)
y = np.random.rand(N)
city = [(x[i], y[i]) for i in range(0, len(x))] 
name = list(map(str, list(range(0,N))))
dict_city = dict(zip(name, city)) 

dict_city_remain = dict_city.copy()


# define class for returning multiple values
class ReturnValue:
  def __init__(self, value, determined, cities_remain):
     self.value = value
     self.determined = determined
     self.cities_remain = cities_remain
#----------------------------------------------------------------------

def calculate_value(node,cities,budget,n_c):
    # weight
    w_c = 1
    w_u = 1
    w_b = 1
    
    n_u = 0
    
    cities_remain = cities.copy()
    del cities_remain[node]
    
    for c in cities_remain:
        if ((cities[c][0] - cities[node][0])**2 + \
           (cities[c][1] - cities[node][1])**2 )<= budget**2:
            n_u = n_u + 1
            
    if n_u == 0:
        determined = 1
    else:
        determined = 0
    
    value = w_c * n_c + w_u * n_u + w_b * budget
    
    return ReturnValue(value,determined,cities_remain)
    
#------------------------------------------------------------------------

def select_node(root):
    
    n = root
    
    while bool(n.children):
        n =  max(n.children,key=lambda node:node.value)
        
    return n

#------------------------------------------------------------------------

def expand_node(n,cities,budget,n_c):
    
    s = n.name
    
    cities_remain = cities.copy()
    del cities_remain[s]
    
    for child in cities_remain:

        dist = math.sqrt((cities[child][0] - cities[s][0])**2 + \
                         (cities[child][1] - cities[s][1])**2)
        if dist <= budget:
            budget_remain = budget - dist
            n_cc = n_c + 1
            child_result = calculate_value(child,cities_remain,
                                           budget_remain,n_cc)
            Node(child, value = child_result.value, parent = n,
                 budget = budget_remain,n_c = n_cc,
                 city = cities_remain)
        
#    V_max = max(c.value)
    
#    for c in children:
#        if abs(c.value - V_max) > theta:
#            remove_child(c)
 
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

def stop(gamma):
    return random.random() < gamma

#------------------------------------------------------------------------   
def make_move(root,cities,budget,n_c,result):
    
#    if lapse(lambda_):
#        return ramdom_move(s)
#    else:
    
    
    for child in result.cities_remain:
        
        dist = math.sqrt((cities[child][0] - cities[root.name][0])**2 + \
                         (cities[child][1] - cities[root.name][1])**2)
        
        if dist <= budget:
            
            budget_remain = budget - dist
            n_cc = n_c + 1
            child_result = calculate_value(child,result.cities_remain,
                                           budget_remain,n_cc)
            Node(child, value = child_result.value, parent = root,
                 budget = budget_remain, n_c = n_cc,
                 city = result.cities_remain)
#        else:
#            budget_remain = budget
#            n_cc = n_c
            
    gamma = 0.5     
    
    while (not stop(gamma)) and (not result.determined):
        n = select_node(root)
        expand_node(n,n.city,n.budget,n.n_c)
        backpropagate(n,root) 
            
    n =  max(root.children,key=lambda node:node.value)
     
    return n
#--------------------------------------------------------------------------
    
n_c = 0 # number of connected cities
now = Node('0') # start point
result = calculate_value(now.name,dict_city_remain,budget_remain,n_c)

while result.determined == 0:
    
    choice = make_move(now,dict_city,budget_remain,n_c,result)
    print(choice.name)
    del dict_city_remain[now.name]
    
    dist = math.sqrt((dict_city[choice.name][0] - dict_city[now.name][0])**2 + \
                     (dict_city[choice.name][1] - dict_city[now.name][1])**2)
    budget_remain = budget_remain - dist
    now = choice
    n_c = n_c + 1
    result = calculate_value(now.name,dict_city_remain,budget_remain,n_c)
    #del dict_city_remain[choice.name]
    