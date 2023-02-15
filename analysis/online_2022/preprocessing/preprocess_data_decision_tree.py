#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 13:20:26 2022

@author: dbao
"""
import pandas as pd
from anytree import Node
from anytree.exporter import JsonExporter
from glob import glob
import json


# import data
data_all = []

# directories
# home_dir = 'G:\My Drive\\researches\\nyu\\road-construction-local-dk\data_online_2022/'
# map_dir = 'active_map/'
# data_dir  = 'data/preprocessed'
home_dir = '/Users/dbao/My_Drive'+'/road_construction/data/2022_online/'
map_dir = 'active_map/'
data_dir  = 'data/preprocessed'
R_out_dir = home_dir+'R_analysis_data/choice_level/'


flist = glob(home_dir + data_dir + '/preprocess4_sub_*.csv')
# import experiment data
for fname in flist:
    with open(fname, 'r') as f:
        all_data = pd.read_csv(f)
        data_all.append(all_data)


basic_tree = []
undo_tree = []
for i in range(len(data_all)): # per subjects
    subID = i
    current_trialID = int(data_all[i]['trial_id'][0])
    current_mapID = int(data_all[i]['map_id'][0])
    current_condition = data_all[i]['condition'][0]
    
    for ti in range(data_all[i].shape[0]): # per choice
        trialID = int(data_all[i]['trial_id'][ti])
        mapID = int(data_all[i]['map_id'][ti])
        condition = data_all[i]['condition'][ti]

        if ti == 0: # start of subject data

            start_node = Node(0, budget = 300, RT= - 1, subID=subID, trialID=current_trialID, mapID = current_mapID,visit=1)
            current_node = start_node
        
        elif (trialID!=current_trialID)|(mapID!=current_mapID)|(condition!=current_condition): # start of trial
            
            exporter = JsonExporter(indent=2, sort_keys=True)
            json_trial = exporter.export(start_node)

            if current_condition == "basic":
                basic_tree.append(json_trial)
            elif current_condition == "undo":
                undo_tree.append(json_trial)
                
            current_trialID = trialID
            current_mapID = mapID
            current_condition = condition
            
            start_node = Node(0, budget = 300, RT= - 1, subID=subID, trialID=current_trialID, mapID = current_mapID,visit=1)
            current_node = start_node
            
        elif not data_all[i]['undoIndicator'][ti]:
            budget = float(data_all[i]['currentBudget'][ti])
            RT = int(data_all[i]['rt_all'][ti])
            city = int(data_all[i]['currentChoice'][ti])  
            
            indicator = 0

            for child in current_node.children:
                if (child.name == city)&(child.budget == budget):
                    child.visit = child.visit+1
                    current_node = child
                    indicator = 1      
            if not indicator:     
                new_node = Node(city, budget = budget, RT= RT, subID=subID, trialID=current_trialID, mapID = current_mapID, parent = current_node,visit=1)
                current_node = new_node
            
        elif data_all[i]['undoIndicator'][ti]:
            current_node = current_node.parent

save_path = '/Users/dbao/My_Drive/road_construction/data/2022_online/tree_data/'    
with open(save_path + 'basic_tree','w') as file: 
    json.dump(basic_tree,file)
with open(save_path + 'undo_tree','w') as file: 
    json.dump(undo_tree,file)
