from anytree import Node
from anytree import RenderTree
#from anytree.exporter import JsonExporter, DictExporter
import random
import numpy as np
from map_class import Map
import math
import ast # to convert string to list
import pickle as pkl
from utils import params_by_name

with open('v_param.pkl','rb') as f:
    v_param = pkl.load(f)
v_param = v_param['v_param']
with open('v_param_offset.pkl','rb') as f:
    v_param_offset = pkl.load(f)
v_param_offset = v_param_offset['v_param']

#----------------------------------------------------------------------
class params:
    def __init__(self, stopping_probability,
                    pruning_threshold,
                    lapse_rate,
                    feature_dropping_rate,
                    count_par = 10,
                    **kwargs
                    ):
        self.weights = []
        if len(kwargs) != 0:
            for key, value in kwargs.items():
                if key == "w1":
                    w1 = value
                    self.weights.append(value)
                elif key == "w2":
                    w2 = value
                    self.weights.append(value)
                elif key == "w3":
                    w3 = value
                    self.weights.append(value)
                elif key == "w4":
                    w4 = value
                    self.weights.append(value)
                elif key == "undoing_threshold":
                    self.undoing_threshold = value
                elif key == "undo_inverse_temparature":
                    self.undo_inverse_temparature = value
                elif key == "ucb_confidence":
                    self.ucb_confidence = value


        self.feature_dropping_rate = feature_dropping_rate
        self.stopping_probability = stopping_probability
        self.pruning_threshold = pruning_threshold
        self.lapse_rate = lapse_rate
        self.count_par = count_par

def value_function_out(value_func, para, node, **kwargs):
    if len(kwargs) != 0:
        for key, value in kwargs.items():
            if key == "i_bfs":
                i_bfs = value
            if key == "condition":
                condition = value
    if not 'i_bfs' in locals():
        i_bfs = 0
    weights = para.weights


    # elif value_func == 'power_law':
    #     # TODO:
    #     #  value 1 - Standard way: power law reward function
    #     weights[1] = 1
    #     value = weights[0] * node.n_c + v_param[0] * np.power(node.n_u, v_param[1])  + \
    #         weights[2] * (node.budget/30)
    #
    # elif value_func == 'power_law_wp': # power law but with parameter
    #     value = weights[0] * node.n_c + weights[1] * v_param[0] * np.power(node.n_u, v_param[1])  + \
    #         weights[2] * (node.budget/30)

    if value_func == 'legacy':
        value = weights[0] * node.n_c + weights[1] * node.n_u + \
            weights[2] * (node.budget/30)



    elif value_func == 'power_law_free': # power law with full degree of freedom
        value = np.power(node.n_c,weights[0]) + weights[1] * np.power(node.n_u, weights[2])  + \
            weights[3] * (node.budget/30)

    elif value_func == 'power_law_free_budgetBias':  # power law with full degree of freedom
        value = np.power(node.n_c, weights[0]) + weights[1] * np.power(node.n_u, weights[2]) + \
            weights[3] * ((node.budget / 30) - weights[4])

    elif value_func == 'power_law2_free':  # power law with full degree of freedom
        value = weights[0]*np.power(node.n_c, weights[1]) + weights[2] * np.power(node.n_u, weights[3]) + \
            weights[4] * (node.budget / 30)

    elif value_func == 'power_law2_free_budgetBias':  # power law with full degree of freedom
        value = weights[0]*np.power(node.n_c, weights[1]) + weights[2] * np.power(node.n_u, weights[3]) + \
            weights[4] * ((node.budget / 30) - weights[5])

    elif value_func == 'power_law3_free': # power law with full degree of freedom
        value = np.power(node.n_c, weights[0]) + np.power(node.n_u, weights[1]) + \
            weights[2] * (node.budget/30)

    elif value_func == 'power_law3_free_budgetBias':  # power law with full degree of freedom
        value = np.power(node.n_c, weights[0]) + np.power(node.n_u, weights[1]) + \
            weights[2] * ((node.budget / 30) - weights[3])






    elif value_func == 'power_law_free_wu': # power law with full degree of freedom
        value = np.power(node.n_c,weights[0]) + weights[1] * np.power(node.n_u, weights[2])  + \
            weights[3] * (node.budget/30)
        if condition=='undo':
            value = np.power(node.n_c,weights[4]) + weights[5] * np.power(node.n_u, weights[6])  + \
                weights[7] * (node.budget/30)

    elif value_func == 'power_law_free_wu_budgetBias':  # power law with full degree of freedom
        value = np.power(node.n_c, weights[0]) + weights[1] * np.power(node.n_u, weights[2]) + \
            weights[3] * ((node.budget / 30) - weights[4])
        if condition=='undo':
            value = np.power(node.n_c, weights[5]) + weights[6] * np.power(node.n_u, weights[7]) + \
                weights[8] * ((node.budget / 30) - weights[9])

    elif value_func == 'power_law2_free_wu':  # power law with full degree of freedom
        value = weights[0]*np.power(node.n_c, weights[1]) + weights[2] * np.power(node.n_u, weights[3]) + \
            weights[4] * (node.budget / 30)
        if condition=='undo':
            value = weights[5]*np.power(node.n_c, weights[6]) + weights[7] * np.power(node.n_u, weights[8]) + \
                weights[9] * (node.budget / 30)

    elif value_func == 'power_law2_free_wu_budgetBias':  # power law with full degree of freedom
        value = weights[0]*np.power(node.n_c, weights[1]) + weights[2] * np.power(node.n_u, weights[3]) + \
            weights[4] * ((node.budget / 30) - weights[5])
        if condition=='undo':
            value = weights[6]*np.power(node.n_c, weights[7]) + weights[8] * np.power(node.n_u, weights[9]) + \
                weights[10] * ((node.budget / 30) - weights[11])

    elif value_func == 'power_law3_free_wu': # power law with full degree of freedom
        value = np.power(node.n_c, weights[0]) + np.power(node.n_u, weights[1]) + \
            weights[2] * (node.budget/30)
        if condition=='undo':
            value = np.power(node.n_c, weights[3]) + np.power(node.n_u, weights[4]) + \
                weights[5] * (node.budget/30)

    elif value_func == 'power_law3_free_wu_budgetBias':  # power law with full degree of freedom
        value = np.power(node.n_c, weights[0]) + np.power(node.n_u, weights[1]) + \
            weights[2] * ((node.budget / 30) - weights[3])
        if condition=='undo':
            value = np.power(node.n_c, weights[4]) + np.power(node.n_u, weights[5]) + \
                weights[6] * ((node.budget / 30) - weights[7])





    elif value_func == 'power_law_free_linear': # power law with full degree of freedom
        value = node.n_c*weights[0] + weights[1] * np.power(node.n_u, weights[2])  + \
            weights[3] * (node.budget/30)

    elif value_func == 'power_law_free_budgetBias_linear':  # power law with full degree of freedom
        value = node.n_c*weights[0] + weights[1] * np.power(node.n_u, weights[2]) + \
            weights[3] * ((node.budget / 30) - weights[4])

    elif value_func == 'power_law3_free_linear': # power law with full degree of freedom
        value = node.n_c*weights[0] + np.power(node.n_u, weights[1]) + \
            weights[2] * (node.budget/30)

    elif value_func == 'power_law3_free_budgetBias_linear':  # power law with full degree of freedom
        value = node.n_c*weights[0] + np.power(node.n_u, weights[1]) + \
            weights[2] * ((node.budget / 30) - weights[3])






    elif value_func == 'power_law_free_wu_linear': # power law with full degree of freedom
        value = node.n_c*weights[0] + weights[1] * np.power(node.n_u, weights[2])  + \
            weights[3] * (node.budget/30)
        if condition=='undo':
            value = node.n_c*weights[4] + weights[5]* np.power(node.n_u, weights[6])  + \
                weights[7] * (node.budget/30)

    elif value_func == 'power_law_free_wu_budgetBias_linear':  # power law with full degree of freedom
        value = node.n_c*weights[0] + weights[1] * np.power(node.n_u, weights[2]) + \
            weights[3] * ((node.budget / 30) - weights[4])
        if condition=='undo':
            value = node.n_c*weights[5] + weights[6] * np.power(node.n_u, weights[7]) + \
                weights[8] * ((node.budget / 30) - weights[9])
    elif value_func == 'power_law3_free_wu_linear': # power law with full degree of freedom
        value = node.n_c*weights[0] + np.power(node.n_u, weights[1]) + \
            weights[2] * (node.budget/30)
        if condition=='undo':
            value = node.n_c*weights[3] + np.power(node.n_u, weights[4]) + \
                weights[5] * (node.budget/30)

    elif value_func == 'power_law3_free_wu_budgetBias_linear':  # power law with full degree of freedom
        value = node.n_c*weights[0] + np.power(node.n_u, weights[1]) + \
            weights[2] * ((node.budget / 30) - weights[3])
        if condition=='undo':
            value = node.n_c*weights[4] + np.power(node.n_u, weights[5]) + \
                weights[6] * ((node.budget / 30) - weights[7])



    elif value_func == 'power_law_free_wue_linear': # power law with full degree of freedom
        value = node.n_c*weights[0] + weights[1] * np.power(node.n_u, weights[2])  + \
            weights[3] * (node.budget/30)

    elif value_func == 'power_law_free_wue_budgetBias_linear':  # power law with full degree of freedom
        value = node.n_c*weights[0] + weights[1] * np.power(node.n_u, weights[2]) + \
            weights[3] * ((node.budget / 30) - weights[4])

    elif value_func == 'power_law3_free_wue_linear': # power law with full degree of freedom
        value = node.n_c*weights[0] + np.power(node.n_u, weights[1]) + \
            weights[2] * (node.budget/30)

    elif value_func == 'power_law3_free_wue_budgetBias_linear':  # power law with full degree of freedom
        value = node.n_c*weights[0] + np.power(node.n_u, weights[1]) + \
            weights[2] * ((node.budget / 30) - weights[3])
    #

    value += np.random.normal()
    denom = para.visits[node.name]
    numer = np.log(para.i_th + i_bfs)

    if denom == 0:
        denom = 1e-3

    if condition=='undo' and '_wu' in value_func:
        value += para.ucb_confidence_wu*np.sqrt(numer/denom)
    else:
        value += para.ucb_confidence*np.sqrt(numer/denom)
    # print(para.ucb_confidence*np.sqrt(np.log(para.i_th)/para.visits[node.name]))

    # debug
    if np.abs(value)>1e5:
        print(value)
    # elif value_func == 'power_law_free_linear': # power law with full degree of freedom
    #     value = node.n_c*weights[0] + weights[1] * np.power(node.n_u, weights[2])  +   weights[3] * (node.budget/30)

    return value


class initial_node_saving():
    def __init__(self, basic_map, subject_data):
        self.basic_map = basic_map
        self.subject_data = subject_data
    def _append(self, idx):
        self.dist = self.basic_map[0][self.subject_data.loc[idx, 'map_id']]['distance']
        self.name = self.subject_data.loc[idx, 'currentChoice']

        # remain = ast.literal_eval(subject_data_.loc[idx,'remain_all'])
        # remain for cities with in reach and
        self.cities_reach = ast.literal_eval(self.subject_data.loc[idx, 'cities_reach'])
        self.cities_taken = ast.literal_eval(self.subject_data.loc[idx, 'chosen_all'])
        self.cities_taken = np.setdiff1d(self.cities_taken, self.name).tolist()  # exclude current location
        self.remain = np.union1d(self.cities_reach, self.cities_taken).tolist()

        # budget_remain = subject_data_.loc[idx,'budget_all']
        self.budget_remain = self.subject_data.loc[idx, 'currentBudget']

        # n_city = subject_data_.loc[idx,'n_city_all']
        # n_u = subject_data_.loc[idx,'n_u_all']
        self.n_city = self.subject_data.loc[idx, 'n_city_all']
        self.n_u = self.subject_data.loc[idx, 'n_within_reach']

def new_node_current_seq(n, name, cities, cities2, dist, budget, n_c, para, **kwargs):

    '''
    n: current node
    name: first decision's name

    '''
    if len(kwargs) != 0:
        for key, value in kwargs.items():
            if key == "value_func":
                value_func = value
            if key == "i_th":
                i_th = value
            if key == "visits":
                visits = value
            if key == "condition":
                condition = value
    s = n.name

    # print(n)
    # n_next = n.children[0]

    # for child in n.city:
    #     if dist[s][child] <= n.budget:
    #         new_node_previous(child, n, dist, weights, value_func = value_func)
    children_name = [i.name for i in n.children]
    # if not np.any(children_name == name): #len(n.children) == 1: # means that random choice in the previous one.
    #     node = Node(name, None)
    #     node.parent = n
    # else:
    #     for child in n.children:
    #         if child.name != name:
    #             remove_child(child)
    #     if len(n.children) != 1:
    #         print('why')
    #     node = n.children[0]

    if np.any(children_name == name):
        for child in n.children:
            if child.name != name:
                remove_child(child)
        # if len(n.children) != 1:
        #     print('why')
        node = n.children[0]
    else:
        node = Node(name, None)
        # if it is undo
        if np.any(n.city_undo==name):
            while n.name != name:
                # print(n)
                n_ = n
                n = n.parent
                n_.parent = None
            node = n
            # print(n)
        else:
            node.parent = n



    #------------------------------------------------------------------
    # calculate n of cities within reach
    if len(kwargs) != 0:
        for key, value in kwargs.items():
            if key == "n_u":
                n_u = value
            if key == "value_func":
                value_func = value
    # if not 'n_u' in kwargs.keys():
    #     n_u = 0
    #     cities_remain = cities.copy()
    #     cities_undo   = cities2.copy()

    #     try:
    #         cities_remain.pop(cities_remain.index(node.name))
    #     except:
    #         pass
    #     # try: del cities_remain[node.name] # delete the current chosen node
    #     # except: pass

    #     for c in cities_remain:
    #         if dist[node.name][c] <= budget:
    #             n_u = n_u + 1
    node.n_u = n_u

    #------------------------------------------------------------------
    node.city = cities
    #------------------------------------------------------------------
    node.city_undo = cities2
    node.budget = budget
    node.n_c = n_c

    #------------------------------------------------------------------

    # if node.budget == 300:
    #     value = weights[1] * node.n_u/pow(node.budget,2) * 10000 + \
    #         np.random.normal()
    # else:
    #     value = weights[0] * node.n_c/(300 - node.budget) * 100 + \
    #         weights[1] * node.n_u/pow(node.budget,2) * 10000 + \
    #         np.random.normal()

    # value = weights[0] * node.n_c + weights[1] * node.n_u + \
    #         np.random.normal()

    node.i_th = para.i_th = i_th
    node.visits = para.visits = visits
    value = value_function_out(value_func, para, node, condition=condition)

    #  value 2 - Standard way: but with weight
    #  value 3 - Uncertainty-driven: using tortuosity and


    # TODO:
    #  value 1 - Planning RL: value 3 - Planning only in the history (forseable undo).


    node.value = value

    #------------------------------------------------------------------
    if node.n_u == 0:
        node.determined = 1
    else:
        node.determined = 0

    # flush children nodes
    for c in node.children:
        remove_child(c)

    return node

#----------------------------------------------------------------------
def new_node_current(name, cities, cities2, dist, budget, n_c, para, **kwargs):

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
            if key == "value_func":
                value_func = value
            if key == "i_th":
                i_th = value
            if key == "visits":
                visits = value
            if key == "condition":
                condition = value

    if not 'n_u' in kwargs.keys():
        n_u = 0
        cities_remain = cities.copy()
        cities_undo   = cities2.copy()

        try:
            cities_remain.pop(cities_remain.index(node.name))
        except:
            pass
        # try: del cities_remain[node.name] # delete the current chosen node
        # except: pass

        for c in cities_remain:
            if dist[node.name][c] <= budget:
                n_u = n_u + 1
    node.n_u = n_u

    #------------------------------------------------------------------
    try: node.city = cities_remain 
    except: node.city = cities

    #------------------------------------------------------------------
    try: node.city_undo = cities_undo # all cities, not only cities within reach
    except: node.city_undo = cities2

    node.budget = budget
    node.n_c = n_c

    #------------------------------------------------------------------

    # if node.budget == 300:
    #     value = weights[1] * node.n_u/pow(node.budget,2) * 10000 + \
    #         np.random.normal()
    # else:
    #     value = weights[0] * node.n_c/(300 - node.budget) * 100 + \
    #         weights[1] * node.n_u/pow(node.budget,2) * 10000 + \
    #         np.random.normal()

    # value = weights[0] * node.n_c + weights[1] * node.n_u + \
    #         np.random.normal()

    node.i_th = para.i_th = i_th
    node.visits = para.visits = visits
    value = value_function_out(value_func, para, node, condition=condition)

    #  value 2 - Standard way: but with weight
    #  value 3 - Uncertainty-driven: using tortuosity and


    # TODO:
    #  value 1 - Planning RL: value 3 - Planning only in the history (forseable undo).


    node.value = value

    #------------------------------------------------------------------
    if node.n_u == 0:
        node.determined = 1
    else:
        node.determined = 0

    return node

def new_node_previous(name, parent, dist, para, **kwargs):

    '''
    creat new tree node with parent node's information
    '''

    #------------------------------------------------------------------
    # calculate n of cities within reach
    if len(kwargs) != 0:
        for key, value in kwargs.items():
            if key == "value_func":
                value_func = value
            if key == "i_bfs":
                i_bfs = value
            if key == "condition":
                condition = value

    if not 'i_bfs' in locals():
        i_bfs = 0
    if not 'condition' in locals():
        condition = 0
    # remove the overlapping ones
    # for c in parent.children:
    #   if c.name == name:
    #       remove_child(c)
    # formultiples = [i.name for i in parent.children]

    node = Node(name,parent)

    node.budget = parent.budget - dist[parent.name][node.name]
    node.n_c = parent.n_c + 1

    #------------------------------------------------------------------
    # calculate n of cities within reach
    n_u = 0
    cities_remain = parent.city.copy()
    cities_undo   = parent.city_undo.copy()


    try: cities_remain.pop(cities_remain.index(node.name))
    except: pass
    # try: del cities_remain[node.name] # delete the current chosen node
    # except: pass

    for c in cities_remain:
        if dist[node.name][c] <= node.budget:
            n_u = n_u + 1

    node.n_u = n_u
    node.city = cities_remain # all cities, not only cities within reach
    cities_undo.append(parent.name)
    node.city_undo = cities_undo

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

    node.i_th = para.i_th
    node.visits = para.visits
    value = value_function_out(value_func, para, node, i_bfs=i_bfs, condition=condition)

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
    for i in range(len(new_weights)):
        if random.random() < sigma:
            new_weights[i] = 0
    # if random.random() < sigma:
    #     new_weights[0] = 0
    # if random.random() < sigma:
    #     new_weights[1] = 0
    # if random.random() < sigma:
    #     new_weights[2] = 0
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
    # print(root)
    n = root

    while bool(n.children):
        n =  max(n.children,key=lambda node:node.value)

    return n

#------------------------------------------------------------------------
def select_node_undo(root, para):

    n = root

    # nos = n
    # ii = 1
    # while nos.parent:
    #     ii += 1
    #     nos = nos.parent
    # if n.n_c != ii:
    #     print('d')
    # 
    # nos = root
    # ii=1
    # while nos.parent:
    #     ii+=1
    #     nos = nos.parent
    # if root.n_c != ii:
    #     print('d')


    while bool(n.children):
        n = max(n.children,key=lambda node:node.value)

    ns_undo = []
    # undoing
    n_parent = root.parent
    while bool(n_parent):
        ns_undo.append(n_parent)
        n_parent = n_parent.parent

    # nos = n
    # ii=1
    # while nos.parent:
    #     ii+=1
    #     nos = nos.parent
    # if n.n_c != ii:
    #     print('d')
    #
    # nos = root
    # ii=1
    # while nos.parent:
    #     ii+=1
    #     nos = nos.parent
    # if root.n_c != ii:
    #     print('d')

    if ns_undo: # if there is something to undo
        n_undo = max(ns_undo,key=lambda node:node.value)
        # select the best

        n = max([n, n_undo],key=lambda node:node.value)

        # n = max([n, n_undo],key=lambda node:node.value)
        # print('n_undo value: {} // n value: {}'.format(n_undo.value, n.value))
        # if n_undo.value > n.value:
        #     print('what?')

    # nos = n
    # ii=1
    # while nos.parent:
    #     ii+=1
    #     nos = nos.parent
    # if n.n_c != ii:
    #     print('d')

    # return undo target as well when it went backward
    if root.name != n.name: # first if it is different
        if root.n_c > n.n_c:
            return n, True
    return n, False

#------------------------------------------------------------------------
def expand_node(n, dist, theta, para, **kwargs):

    if len(kwargs) != 0:
        for key, value in kwargs.items():
            if key == "value_func":
                value_func = value
            if key == "root":
                root = value
            if key == "condition":
                condition = value

    if not 'condition' in locals():
        condition = ''

    s = n.name

    for child in n.city:
        if dist[s][child] <= n.budget:
            new_node_previous(child, n, dist, para, value_func = value_func, condition=condition)
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
def expand_node_undo(n, dist, theta, para, **kwargs):

    if len(kwargs) != 0:
        for key, value in kwargs.items():
            if key == "value_func":
                value_func = value
            if key == "root":
                root = value
            if key == "i_bfs":
                i_bfs = value

    if not 'i_bfs' in locals():
        i_bfs = 0

    s = n.name

    # nos = root
    # ii=1
    # while nos.parent:
    #     ii+=1
    #     nos = nos.parent
    # if root.n_c != ii:
    #     print('d')

    # empty for the undoing one
    # tobedeleted = 0
    if root.n_c > n.n_c:
        tobedeleted = len(n.children)
        # for c in n.children:
            # remove_child(c)
    # if para.visits[n.name]>6:
    #     print(n)
    # print('node_search')
    # print(n)
    # print('node_children')
    for child in n.city:
        if dist[s][child] <= n.budget:
            # print(child)
            new_node_previous(child, n, dist, para, value_func = value_func, i_bfs=i_bfs+1)

    # if root.n_c > n.n_c:
    #     for ci in range(tobedeleted):
    #         remove_child(n.children[ci])
    #     root.parent = n

    # nos = root
    # ii=1
    # while nos.parent:
    #     ii+=1
    #     nos = nos.parent
    # if root.n_c != ii:
    #     print('d')

    #------------------------------------------------------------------
    # pruning
    try:
        V_max = max(n.children, key=lambda node:node.value)

        #
        # nos = n
        # ii=1
        # while nos.parent:
        #     ii+=1
        #     nos = nos.parent
        # if n.n_c != ii:
        #     print('d')
        #
        #
        # nos = root
        # ii=1
        # while nos.parent:
        #     ii+=1
        #     nos = nos.parent
        # if root.n_c != ii:
        #     print('d')


        for c in n.children:
            # dont disconnect for undoing
            if  root.n_c < n.n_c:
                if abs(c.value - V_max.value) > theta:
                    remove_child(c)


        # nos = root
        # ii=1
        # while nos.parent:
        #     ii+=1
        #     nos = nos.parent
        # if root.n_c != ii:
        #     print('d')
        #
        # #
        # nos = n
        # ii=1
        # while nos.parent:
        #     ii+=1
        #     nos = nos.parent
        # if n.n_c != ii:
        #     print('d')

        # if i_th is updated then remove.
        name_list = [i.name for i in n.children]
        ith_list = [i.i_th for i in n.children]
        nodes = []
        for c in np.unique(np.array(name_list)):
            i_c, = np.where(np.array(name_list) == c)
            # nodes.append(n.children[i_c[np.argmax(np.array(ith_list)[i_c])]])
            nodes.append(n.children[i_c[np.argmax(np.array(ith_list)[i_c])]])
            # for ii_c in i_c:
            #     nodes.append(n.children[i_c[np.argmax(np.array(ith_list)[i_c])]])
        n.children = tuple(nodes)

        # # when undid, disconnected with parent
        # nos = root
        # ii=1
        # while nos.parent:
        #     ii+=1
        #     nos = nos.parent
        # if root.n_c != ii:
        #     print('d')
        # nos = n
        # ii=1
        # while nos.parent:
        #     ii+=1
        #     nos = nos.parent
        # if n.n_c != ii:
        #     print('d')
        # if len(nodes)==0:
        #     print('d')

        if root.n_c > n.n_c:
            nos = root
            ii=1
            while nos.parent:
                ii+=1
                nos = nos.parent
            name_list = [i.name for i in n.children]
            if root.n_c != ii:
                if np.any(np.array(name_list) == nos.name):
                    nos.parent = n

    except:
        pass

#------------------------------------------------------------------------
def backpropagate(n,root):
    # dont update for undoing because it is 'back' propagation?
    # if n.n_c <= root.n_c:
    # print(n)
    # if (n.budget <= root.budget) or np.allclose(n.budget,root.budget):
    # # if (n.budget <= root.budget):
    try:
        max_child = max(n.children, key=lambda node:node.value)
        n.value = max_child.value
    except:
        n.value = n.value
    if n != root:
        backpropagate(n.parent,root)

def backpropagate_undo(n,n_name, root):
    # t_undo: undo target.
    # dont update for undoing because it is 'back' propagation?
    # if n.n_c <= root.n_c:
    # print(n)
    try:
        max_child = max(n.children, key=lambda node:node.value)
        n.value = max_child.value
    except:
        n.value = n.value
    if n.name != n_name:
        backpropagate_undo(n.parent,t_undo)

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
def remove_node(c):
    del c


#------------------------------------------------------------------------
def make_move(s,dist_city,para, **kwargs):

    if len(kwargs) != 0:
        for key, value in kwargs.items():
            if key == "value_func":
                value_func = value
    '''
    main function of best-first tree search
    '''
    # print(s)
    if lapse(para.lapse_rate):
        # print('random_choice')
        return ramdom_move(s,dist_city,para.weights,value_func=value_func)
    else:
    #------------------------------------------------------------------

        # n_parent = s.parent
        # for i in range(s.n_c - 1):
        #     n_parent = n_parent.parent
        # if n_parent:
        #     if n_parent.name != 0:
        #         []

        new_weights = dropfeature(para.weights,para.feature_dropping_rate)
        root = s

        count = 0 # count of same selected node

        # 1st iteration
        if (not determined(root)):
        # it should do BFS because it is possible to undo.

            n = select_node(root)

            expand_node(n,dist_city,para.pruning_threshold,new_weights,value_func=value_func, root=root)
    #            for pre, _, node in RenderTree(start):
    #                print("%s%s:%s" % (pre, node.name,node.value))

            backpropagate(n, root)
    #            for pre, _, node in RenderTree(start):
    #                print("%s%s:%s" % (pre, node.name,node.value))

            selectnode = max(root.children,key=lambda node:node.value)

        # print(root)
        # from 2nd iteration
        nvr = True
        while (not stop(para.stopping_probability,count,para.count_par)) and (not determined(root)):
            nvr = False
            n = select_node(root)
            # print(n)

            #            print('select node: '+ str(n.name))

            expand_node(n, dist_city, para.pruning_threshold, new_weights, value_func=value_func, root=root)


            # new_selectnode = max(n.children, key=lambda node: node.value)
            backpropagate(n, root)
            # print(n)
            # print(n.value)
            #            print('backpropagate:')
            #            for pre, _, node in RenderTree(start):
            #                print("%s%s:%s" % (pre, node.name,node.value))

            new_selectnode = max(root.children,key=lambda node:node.value)

            if new_selectnode == selectnode:
                count = count+1
            else:
                count = 0

            selectnode = new_selectnode

        if determined(root):
            max_ = root
        else:
            max_ =  max(root.children,key=lambda node:node.value)

    return max_

#------------------------------------------------------------------------
def make_move_undo(s,dist_city,para, **kwargs):

    if len(kwargs) != 0:
        for key, value in kwargs.items():
            if key == "value_func":
                value_func = value
    '''
    main function of best-first tree search
    '''

    if lapse(para.lapse_rate):
        # print('random_choice')
        return ramdom_move_undo(s,dist_city,para.weights,value_func=value_func)
    else:
    #------------------------------------------------------------------

        # n_parent = s.parent
        # for i in range(s.n_c - 1):
        #     n_parent = n_parent.parent
        # if n_parent:
        #     if n_parent.name != 0:
        #         []

        new_weights = dropfeature(para.weights,para.feature_dropping_rate)
        root = s

        count = 0 # count of same selected node

        # 1st iteration
        # if (not determined(root)):
        # it should do BFS because it is possible to undo.

        n, t_undo = select_node_undo(root, para)

        expand_node_undo(n,dist_city,para.pruning_threshold,new_weights,value_func=value_func, root=root)
#            for pre, _, node in RenderTree(start):
#                print("%s%s:%s" % (pre, node.name,node.value))

        if t_undo:
            # backpropagate(n, root)
            backpropagate_undo(n,n.name, root)
        else:
            backpropagate(n, root)
#            for pre, _, node in RenderTree(start):
#                print("%s%s:%s" % (pre, node.name,node.value))

        # if they undid
        if (n.budget > root.budget):
            selectnode = max(n.children,key=lambda node:node.value)
        else:
            if determined(root):
                selectnode = root
            else:
                selectnode = max(root.children,key=lambda node:node.value)

        # print(root)
        # from 2nd iteration
        nvr = True
        while (not stop(para.stopping_probability,count,para.count_par)):

            # print(n)
            # print(n.value)

            nvr = False
            n, t_undo = select_node_undo(root, para)

            #            print('select node: '+ str(n.name))

            # if t_undo:
            #     print(n)
            #     print(n.value)
            # else:
            #     print(n)
            #     print(n.value)

            expand_node_undo(n, dist_city, para.pruning_threshold, new_weights, value_func=value_func, root=root)

            nos = n
            ii = 1
            while nos.parent:
                ii += 1
                nos = nos.parent
            if n.n_c != ii:
                print('d')
            # new_selectnode = max(n.children, key=lambda node: node.value)
            if t_undo:
                backpropagate_undo(n,n.name, root)
                # print(n)
                # print(n.value)
            else:
                backpropagate(n, root)
                # print(n)
                # print(n.value)

            nos = n
            ii = 1
            while nos.parent:
                ii += 1
                nos = nos.parent
            if n.n_c != ii:
                print('d')

            if (n.n_c < root.n_c):
                new_selectnode = max(n.children, key=lambda node: node.value)
            else:
                if determined(root):
                    new_selectnode = root
                else:
                    new_selectnode = max(root.children,key=lambda node:node.value)

            if new_selectnode == selectnode:
                count = count+1
            else:
                count = 0

            selectnode = new_selectnode

        if root.children:
            temp = list(root.children)
            if (n.n_c < root.n_c): # in case you searched for the undoing option
                # if it exceed the  threshold
                max_in_children = max(root.children, key=lambda node: node.value)

                # if the model uses softmax
                if hasattr(para, 'undo_inverse_temparature'):
                    if hasattr(para, 'undoing_threshold'):
                        if (n.value - max_in_children.value) > para.undoing_threshold:
                            backtracked_path = []
                            rooTemp = root
                            while rooTemp.parent:
                                # print(rooTemp)
                                backtracked_path.append(rooTemp.parent)
                                if (rooTemp.parent.name == n.name):
                                    break
                                rooTemp = rooTemp.parent
                            backtracked_path.reverse()

                            updatedtrack_path = []
                            nTemp = n
                            for nb in backtracked_path:
                                if not nTemp.children:
                                    break
                                nTemp_c = np.array(nTemp.children)[[c.name == nb.name for c in nTemp.children]].tolist()
                                if nTemp_c:
                                    maxtemp = max(nTemp_c, key=lambda node: node.value)
                                    updatedtrack_path.append(maxtemp)
                                    nTemp = maxtemp
                                else:
                                    break

                            if updatedtrack_path: # if it is empty it means there is no need of update
                                vv = np.array([n_n.value for n_n in updatedtrack_path])
                                selection = np.random.choice(range(0, len(updatedtrack_path)),
                                                             p=np.exp(para.undo_inverse_temparature * vv) / np.sum(
                                                                 np.exp(para.undo_inverse_temparature * vv)))
                                max_ = updatedtrack_path[selection]
                            else:
                                max_ = n
                        else:
                            max_ =  max_in_children

                    else:
                        vv = np.array([n.value, max_in_children.value])/30
                        selection = np.random.choice([0, 1], p=np.exp(para.undo_inverse_temparature * vv) / np.sum(
                            np.exp(para.undo_inverse_temparature * vv)))
                        if selection == 1:
                            max_ = max_in_children
                        else:
                            max_ = n
                else:
                    if (n.value - max_in_children.value) > para.undoing_threshold:
                        temp.append(n)
                        max_ = max(temp, key=lambda node: node.value)
                    else:
                        max_ =  max_in_children
            else:
                if determined(root):
                    max_ = root
                else:
                    max_ = max(root.children, key=lambda node: node.value)
        else:
            max_ = n

    return max_

# temporary
def copy_para(para):
    keys = para.__dict__.keys()
    params_name = []
    inparams = []
    for key in keys:
        if key == 'weights':
            for i, val in enumerate(para.weights):
                inparams.append(val)
                params_name.append('w'+str(i+1))
        else:
            inparams.append(para.__dict__[key])
            params_name.append(key)

    para_copied = params_by_name(inparams, params_name,count_par=inparams[params_name.index('count_par')])

    return para_copied

#------------------------------------------------------------------------
def make_move_weights(s, dist_city, para, **kwargs):

    '''
    main function of best-first tree search
    s: current node
    '''
    if len(kwargs) != 0:
        for key, value in kwargs.items():
            if key == "value_func":
                value_func = value
            if key == "i_th":
                i_th = value
            if key == "visits":
                visits = value

    para_temp = copy_para(para)
    para_temp.i_th = i_th
    para_temp.visits = visits

    if lapse(para.lapse_rate):
        # print('random_choice')
        return ramdom_move(s,dist_city,para ,value_func=value_func), para.weights

    else:
    #------------------------------------------------------------------

        # n_parent = s.parent
        # for i in range(s.n_c - 1):
        #     n_parent = n_parent.parent
        # if n_parent:
        #     if n_parent.name != 0:
        #         []

        new_weights = dropfeature(para_temp.weights, para_temp.feature_dropping_rate)

        para_temp.weights = new_weights

        root = s

        count = 0 # count of same selected node

        # 1st iteration
        if (not determined(root)):
        # it should do BFS because it is possible to undo.

            n = select_node(root)

            # para_temp.visits[n.name] += 1
            expand_node(n,dist_city, para_temp.pruning_threshold, para_temp, value_func=value_func, root=root)
    #            for pre, _, node in RenderTree(start):
    #                print("%s%s:%s" % (pre, node.name,node.value))

            backpropagate(n, root)
    #            for pre, _, node in RenderTree(start):
    #                print("%s%s:%s" % (pre, node.name,node.value))
            selectnode = max(root.children,key=lambda node:node.value)

        # print(root)

        # from 2nd iteration
        while (not stop(para_temp.stopping_probability, count, para_temp.count_par)) and (not determined(root)):
            
            n = select_node(root)
            # print(n)

            #            print('select node: '+ str(n.name))

            # para_temp.visits[n.name] += 1
            expand_node(n, dist_city, para_temp.pruning_threshold, para_temp, value_func=value_func, root=root)


            # new_selectnode = max(n.children, key=lambda node: node.value)
            backpropagate(n, root)
            # print(n)
            # print(n.value)
            #            print('backpropagate:')
            #            for pre, _, node in RenderTree(start):
            #                print("%s%s:%s" % (pre, node.name,node.value))

            new_selectnode = max(root.children,key=lambda node:node.value)

            if new_selectnode == selectnode:
                count = count+1
            else:
                count = 0

            # if count > para_temp.count_par:
            #     print('count_exceeded')

            selectnode = new_selectnode

        if determined(root):
            max_ = root
        else:
            try:
                max_ =  max(root.children,key=lambda node:node.value)
            except:
                max_ = root # means nothing has passed the threshold.

    return max_, new_weights

#------------------------------------------------------------------------
def make_move_undo_weights(s,dist_city,para, **kwargs):

    if len(kwargs) != 0:
        for key, value in kwargs.items():
            if key == "value_func":
                value_func = value
            if key == "i_th":
                i_th = value
            if key == "visits":
                visits = value

    condition = 'undo'
    para_temp = copy_para(para)
    para_temp.i_th = i_th
    para_temp.visits = visits

    if '_wu' in value_func:
        para_temp.ucb_confidence = para.ucb_confidence_wu
        para_temp.stopping_probability = para.stopping_probability_wu
        para_temp.pruning_threshold = para.pruning_threshold_wu
        para_temp.lapse_rate = para.lapse_rate_wu
        para_temp.feature_dropping_rate = para.feature_dropping_rate_wu



    '''
    main function of best-first tree search
    '''
    # print(s)
    i_bfs_ref = s.n_c
    i_bfs     = 0

    if lapse(para_temp.lapse_rate):
        # print('random_choice')
        return ramdom_move_undo(s,dist_city,para_temp,value_func=value_func, condition=condition), para_temp.weights
    else:
    #------------------------------------------------------------------

        # one more step for getting children nodes if undo action's value does not exceed undo threshold

        new_weights = dropfeature(para_temp.weights,para_temp.feature_dropping_rate)

        para_temp.weights = new_weights

        root = s

        count = 0 # count of same selected node

        # 1st iteration
        if (not determined(root)):
        # it should do BFS because it is possible to undo.

            n = select_node(root)

            expand_node(n,dist_city,para_temp.pruning_threshold,para_temp,value_func=value_func, root=root, condition=condition)

            backpropagate(n, root)
            selectnode = max(root.children,key=lambda node:node.value)

        # from 2nd iteration
        nvr = True
        while (not stop(para_temp.stopping_probability,count,para_temp.count_par)) and (not determined(root)):
            nvr = False
            n = select_node(root)

            expand_node(n, dist_city, para_temp.pruning_threshold, para_temp, value_func=value_func, root=root, condition=condition)
            backpropagate(n, root)
            new_selectnode = max(root.children,key=lambda node:node.value)
            if new_selectnode == selectnode:
                count = count+1
            else:
                count = 0

            # if count > para_temp.count_par:
            #     print('count_exceeded')

            selectnode = new_selectnode

            # nos = n
            # ii = 1
            # while nos.parent:
            #     ii += 1
            #     nos = nos.parent
            # if n.n_c != ii:
            #     print('d')

        # nos = root
        # ii = 1
        # while nos.parent:
        #     ii += 1
        #     nos = nos.parent
        # if root.n_c != ii:
        #     print('d')


    #------------------------------------------------------------------
        # n_parent = s.parent
        # for i in range(s.n_c - 1):
        #     n_parent = n_parent.parent
        # if n_parent:
        #     if n_parent.name != 0:
        #         []

        new_weights = dropfeature(para_temp.weights,para_temp.feature_dropping_rate)
        # root = s

        count = 0 # count of same selected node

        # 1st iteration
        # if (not determined(root)):
        # it should do BFS because it is possible to undo.

        # nos = root
        # ii = 1
        # while nos.parent:
        #     ii += 1
        #     nos = nos.parent
        # if root.n_c != ii:
        #     print('d')
        n, t_undo = select_node_undo(root, para_temp)

        # nos = n
        # ii = 1
        # while nos.parent:
        #     ii += 1
        #     nos = nos.parent
        # if n.n_c != ii:
        #     print('d')
        # para_temp.visits[n.name] += 1
        expand_node_undo(n,dist_city,para_temp.pruning_threshold,para_temp,value_func=value_func, root=root, condition=condition)
#            for pre, _, node in RenderTree(start):
#                print("%s%s:%s" % (pre, node.name,node.value))

        # nos = n
        # ii = 1
        # while nos.parent:
        #     ii += 1
        #     nos = nos.parent
        # if n.n_c != ii:
        #     print('d')


        # nos = root
        # ii = 1
        # while nos.parent:
        #     ii += 1
        #     nos = nos.parent
        # if root.n_c != ii:
        #     print('d')


        # if t_undo:
            # backpropagate(n, root)
            # backpropagate_undo(n,n.name, root)
        # else:
        #     backpropagate(n, root)
#            for pre, _, node in RenderTree(start):
#                print("%s%s:%s" % (pre, node.name,node.value))

        # if they undid
        if (n.budget > root.budget):
            selectnode = max(n.children,key=lambda node:node.value)
        else:
            if determined(root):
                selectnode = root
            else:
                selectnode = max(root.children,key=lambda node:node.value)

        # nos = root
        # ii = 1
        # while nos.parent:
        #     ii += 1
        #     nos = nos.parent
        # if root.n_c != ii:
        #     print('d')


        # print(root)
        # from 2nd iteration
        nvr = True
        while (not stop(para_temp.stopping_probability,count,para_temp.count_par)):

            # print(n)
            # print(n.value)

            # nos = root
            # ii = 1
            # while nos.parent:
            #     ii += 1
            #     nos = nos.parent
            # if root.n_c != ii:
            #     print('d')

            # nos = n
            # ii = 1
            # while nos.parent:
            #     ii += 1
            #     nos = nos.parent
            # if n.n_c != ii:
            #     print('d')

            nvr = False
            n, t_undo = select_node_undo(root, para_temp)
            i_bfs = np.abs(i_bfs_ref - n.n_c)
            #            print('select node: '+ str(n.name))

            # if t_undo:
            #     print('undid')
            #     print(n)
            #     print(n.value)
            # else:
            #     print('connect')
            #     print(n)
            #     print(n.value)

            # nos = root
            # ii = 1
            # while nos.parent:
            #     ii += 1
            #     nos = nos.parent
            # if root.n_c != ii:
            #     print('d')
            # nos = n
            # ii = 1
            # while nos.parent:
            #     ii += 1
            #     nos = nos.parent
            # if n.n_c != ii:
            #     print('d')

            # para_temp.visits[n.name] += 1
            expand_node_undo(n, dist_city, para_temp.pruning_threshold, para_temp, value_func=value_func, root=root, i_bfs=i_bfs,condition=condition)

            # nos = root
            # ii = 1
            # while nos.parent:
            #     ii += 1
            #     nos = nos.parent
            # if root.n_c != ii:
            #     print('d')
            # nos = n
            # ii = 1
            # while nos.parent:
            #     ii += 1
            #     nos = nos.parent
            # if n.n_c != ii:
            #     print('d')


            # new_selectnode = max(n.children, key=lambda node: node.value)
            if t_undo:
                backpropagate_undo(n,n.name, root)
                # print(n)
                # print(n.value)
            else:
                backpropagate(n, root)
                # print(n)
                # print(n.value)

            # nos = n
            # ii = 1
            # while nos.parent:
            #     ii += 1
            #     nos = nos.parent
            # if n.n_c != ii:
            #     print('d')
            # nos = root
            # ii = 1
            # while nos.parent:
            #     ii += 1
            #     nos = nos.parent
            # if root.n_c != ii:
            #     print('d')


            if (n.n_c < root.n_c):
                new_selectnode = max(n.children, key=lambda node: node.value)
            else:
                if determined(root):
                    new_selectnode = root
                else:
                    new_selectnode = max(root.children,key=lambda node:node.value)

            if new_selectnode == selectnode:
                count = count+1
            else:
                count = 0

            selectnode = new_selectnode

        if root.children: # if it has expanded itself to deeper tree
            temp = list(root.children)
            if (n.n_c < root.n_c): # in case you searched for the undoing option
                # if it exceed the  threshold
                max_in_children = max(root.children, key=lambda node: node.value)

                # if the model uses softmax
                if hasattr(para_temp, 'undo_inverse_temparature'):
                    if hasattr(para_temp, 'undoing_threshold'):
                        if (n.value - max_in_children.value) > para_temp.undoing_threshold:
                            backtracked_path = []
                            rooTemp = root
                            while rooTemp.parent:
                                # print(rooTemp)
                                backtracked_path.append(rooTemp.parent)
                                if (rooTemp.parent.name == n.name):
                                    break
                                rooTemp = rooTemp.parent
                            backtracked_path.reverse()

                            updatedtrack_path = []
                            nTemp = n
                            for nb in backtracked_path:
                                if not nTemp.children:
                                    break
                                nTemp_c = np.array(nTemp.children)[[c.name == nb.name for c in nTemp.children]].tolist()
                                if nTemp_c:
                                    maxtemp = max(nTemp_c, key=lambda node: node.value)
                                    updatedtrack_path.append(maxtemp)
                                    nTemp = maxtemp
                                else:
                                    break

                            if updatedtrack_path: # if it is empty it means there is no need of update
                                vv = np.array([n_n.value for n_n in updatedtrack_path])
                                selection = np.random.choice(range(0, len(updatedtrack_path)),
                                                             p=np.exp(para_temp.undo_inverse_temparature * vv) / np.sum(
                                                                 np.exp(para_temp.undo_inverse_temparature * vv)))
                                max_ = updatedtrack_path[selection]
                            else:
                                max_ = n
                        else:
                            max_ =  max_in_children

                    else:
                        vv = np.array([n.value, max_in_children.value])
                        selection = np.random.choice([0, 1], p=np.exp(para_temp.undo_inverse_temparature * vv) / np.sum(
                            np.exp(para_temp.undo_inverse_temparature * vv)))
                        if selection == 1:
                            max_ = max_in_children
                        else:
                            max_ = n
                else:
                    if (n.value - max_in_children.value) > para_temp.undoing_threshold:
                        temp.append(n)
                        max_ = max(temp, key=lambda node: node.value)
                    else:
                        max_ =  max_in_children
            else:
                if determined(root):
                    max_ = root
                else:
                    max_ = max(root.children, key=lambda node: node.value)
        else: # if it chose to undo as its best, never expanded to deeper branches
            # or determined
            if hasattr(para_temp, 'undoing_threshold'):
                if (n.value - root.value) > para_temp.undoing_threshold:
                    max_ = n
                else:
                    max_ = root
            else:
                max_ = n

    # #debuging part.
    # max__ = max([max_, s], key=lambda node: node.value)
    # if s.name == max__.name:
    #     if s.name != max_.name:
    #         print('hehey')
    return max_, new_weights


#------------------------------------------------------------------------
def ramdom_move(s,dist,para,value_func='legacy'):
    candidates = []
    for c in s.city:
        if dist[s.name][c] <= s.budget:
            candidates.append(c)
    if len(candidates)==0:
        s.determined = 1
    if not determined(s):
        n = new_node_previous(random.choice(candidates), s, dist, para,value_func=value_func)
    else:
        n = s
    return n

#------------------------------------------------------------------------
def ramdom_move_undo(s,dist,para,value_func='legacy',**kwargs): # random move including undo moves.

    if len(kwargs) != 0:
        for key, value in kwargs.items():
            if key == "condition":
                condition = value

    candidates = []
    for c in s.city:
        if dist[s.name][c] <= s.budget:
            candidates.append(c)
    candidates.extend(s.city_undo)

    candidate = random.choice(candidates)
    if candidate in s.city_undo: # means you want to random undo move
        s_ = s
        while candidate != s_.name:
            n = s_.parent
            # s_.parent = None
            s_ = n
    else:
        n = new_node_previous(candidate, s, dist, para,value_func=value_func, condition=condition)
    # if not determined(s):
    #     n = new_node_previous(candidate, s, dist, set_weights,value_func=value_func)
    # else:
    #     n = s
    return n
# -------------------------------------------------------------------------
if __name__ == "__main__":

    home_dir = './'
    input_dir = 'data_mod_ibs/'
    map_dir = 'active_map/'
    output_dir = 'll/'

    import json
    with open(home_dir + map_dir + 'basicMap.json', 'r') as file:
        basic_map_ = json.load(file)



    '''
    model simulation for a map
    '''
    #--------------------------------------------------------------------------
    # set parameters
    param_name = ['stopping_probability','pruning_threshold','lapse_rate',
        'feature_dropping_rate','undoing_threshold','w1','w2','w3','w4']
    inparams = [0.886420939117670,0.444449546742360,0.0445324866572417,0.323783950880170,0,9.96650915592909,9.65530082583427,2.13627340272069,0.237018225714564]
    para = params(w1=inparams[5], w2=inparams[6], w3=inparams[7],
                        stopping_probability=inparams[0],
                        pruning_threshold=inparams[1],undoing_threshold=inparams[4],
                        lapse_rate=inparams[2], feature_dropping_rate=inparams[3])
    'preprocess4_sub_unidentifiedID2_RC-Phaser_2022-01-04_11h42.26.779_REWARD426.csv'
    #--------------------------------------------------------------------------
    # generate map
    trial = Map(basic_map_, 0)
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
    start = new_node_current(0, dict_city_remain, dist_city, trial.budget_remain, 0, para)
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