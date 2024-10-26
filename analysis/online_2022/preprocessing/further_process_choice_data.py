import json
from anytree.importer import JsonImporter
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter

import pandas as pd


home_dir = '/Users/dbao/My_Drive'+'/road_construction/data/2022_online/'
map_dir = 'active_map/'
data_dir  = 'data/preprocessed'
out_dir = home_dir + 'figures/figures_all/'
R_out_dir = home_dir + 'R_analysis_data/'

# with open(home_dir +'tree_data/basic_tree', 'r') as file:
#     basic_tree = json.load(file)
with open(home_dir +'tree_data/undo_tree', 'r') as file:
    undo_tree = json.load(file)

data_choice_level = pd.read_csv(R_out_dir +  'choice_level/choicelevel_data_without_tree.csv') #, index_col=0

importer = JsonImporter()
visit = []

for ti in range(len(undo_tree)): # loop through trials 
    root = importer.import_(undo_tree[ti])
    print(RenderTree(root, style=AsciiStyle()))
    for node in PreOrderIter(root): # loop through the tree
        n_child = len(node.children)
        visit.append(node.visit)

        if (n_child > 1): # if it is a branching node
            city = node.name
            mapID = node.mapID
            subID = node.subID
            trialID = node.trialID

            print('subID:',subID, 'mapID:',mapID, 'trialID:',trialID, 'city:',city)
        
            get_ind = data_choice_level.index[(data_choice_level['subjects'] == subID)&(data_choice_level['puzzleID'] == mapID)&
                                              (data_choice_level['trialID'] == trialID)&(data_choice_level['choice'] == city)].tolist()
            
            data_choice_level.loc[get_ind,'branching'] = True
            data_choice_level.loc[get_ind[0],'branchingFirst'] = True # the first visit of a branching node
            # fill the na values as false
            data_choice_level['branching'] = data_choice_level['branching'].fillna(False)
            data_choice_level['branchingFirst'] = data_choice_level['branchingFirst'].fillna(False)

data_choice_level.to_csv(R_out_dir +  'choice_level/choicelevel_data.csv')  