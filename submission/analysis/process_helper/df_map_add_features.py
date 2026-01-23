import pandas as pd
import numpy as np
from tools.calculate_rolling_mean import calculate_rolling_mean

def df_map_add_features(data_puzzle_level, data_choice_level):
    data_puzzle_level = data_puzzle_level.sort_values(["subjects","trialID"])
    data_puzzle_level['avg_numCities_before'] = data_puzzle_level.groupby('subjects').apply(calculate_rolling_mean).reset_index(level=0, drop=True)
    data_puzzle_level = data_puzzle_level.sort_values(["subjects","puzzleID","condition"])
    data_puzzle_level['mas'] = data_puzzle_level.mas - 1

    # add branching-node RT to puzzle data
    index_start = data_choice_level.index[(data_choice_level['branchingFirst'] == True)]
    RT_branching = list(data_choice_level.loc[index_start+1, 'RT'])
    subjects_chosen = list(data_choice_level.loc[index_start+1, 'subjects'])
    puzzle_chosen = list(data_choice_level.loc[index_start+1, 'puzzleID'])
    for i in range(len(subjects_chosen)): #
        index_chosen = data_puzzle_level.index[(data_puzzle_level['condition']==1)&(data_puzzle_level['subjects']==subjects_chosen[i])&(data_puzzle_level['puzzleID']==puzzle_chosen[i])]
        data_puzzle_level.loc[index_chosen,'RT_branching'] = RT_branching[i]
    data_puzzle_level['RT_branching'] = data_puzzle_level['RT_branching']/1000

    return data_puzzle_level