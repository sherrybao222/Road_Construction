#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 13:41:32 2021

@author: dbao
"""
import json

# directories
home_dir = '/Users/dbao/google_drive_db/'
map_dir = 'road_construction/data/test_2021/experiment/map/active_map/'

# load maps from json
with open(home_dir + map_dir + 'basic_map_training_20','r') as file:
    train_basic_map = json.load(file) 
with open(home_dir + map_dir + 'undo_map_20_training','r') as file:
    train_undo_map = json.load(file) 

with open(home_dir + map_dir + 'basic_map_48_all4','r') as file:
    basic_map = json.load(file) 
with open(home_dir + map_dir + 'undo_map_48_all4','r') as file:
    undo_map = json.load(file) 

new_basic_map = basic_map[0]
with open('basic_map_48_all4_online','w') as file: 
    json.dump(new_basic_map,file)

new_undo_map = []
for i in range(0,len(undo_map)): #[0]
    new_undo_map.append(undo_map[i]['loadmap'])
with open('undo_map_48_all4_online','w') as file: 
    json.dump(new_undo_map,file)

new_train_map = train_basic_map[0]
with open(home_dir + map_dir +'map_training_online','w') as file: 
    json.dump(new_train_map,file)

new_undo_trainmap = []
for i in range(0,len(train_undo_map)): #[0]
    new_undo_trainmap.append(train_undo_map[i]['loadmap'])
with open(home_dir + map_dir +'undo_map_training_online','w') as file: 
    json.dump(new_undo_trainmap,file)
