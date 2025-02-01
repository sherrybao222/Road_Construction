import pandas as pd
import numpy as np

def df_subset_undo(df):
    '''
    select the data where first undo is possible
    '''
    # Use query method for cleaner condition checking
    condition = f"(submit != 1) & (currNumCities != 0) & (undo != 1)" 
    index = df.query(condition).index
    undo_data = df.loc[index + 1, ["subjects","puzzleID", "firstUndo",'lastUndo', "undo", "allMAS", "currMas","RT"]] 
    # Direct assignment from the same dataframe using .loc to avoid chaining
    columns_to_copy = ["currNumCities", "N_more", "severityOfErrors", "move_missed_reward", 
                       "error", "cumulative_error", "state_missed_reward", "checkEnd", "leftover", "budget_change", "within_reach_change", "efficiency", "within_reach"]
    for column in columns_to_copy:
        undo_data[column] = df.loc[index, column].values
    undo_data['sequential_undo'] = (undo_data.firstUndo != undo_data.lastUndo)&(undo_data.firstUndo == 1)
    undo_data['single_undo'] = (undo_data.firstUndo == undo_data.lastUndo)&(undo_data.firstUndo == 1)

    undo_data['terminal_undo'] = (undo_data.firstUndo == 1)&(undo_data.checkEnd == 1)
    undo_data['nonterminal_undo'] = (undo_data.firstUndo == 1)&(undo_data.checkEnd == 0)
    undo_data["suboptimal_state"] = undo_data["cumulative_error"]>0

    temp, cutoff_budget = pd.cut(undo_data['leftover'], 10 , labels = False, retbins=True)
    undo_data["leftover_bin"] = temp
    temp, cutoff_budgetchange = pd.cut(undo_data['budget_change'], 10 , labels = False, retbins=True)
    undo_data["budget_change_bin"] = temp
    temp, cutoff_withinreachchange = pd.cut(undo_data['within_reach_change'], 10 , labels = False, retbins=True, duplicates='drop')
    undo_data["within_reach_change_bin"] = temp
    temp, cutoff_efficiency = pd.cut(undo_data['efficiency'], 10 , labels = False, retbins=True, duplicates='drop')
    undo_data["efficiency_bin"] = temp

    return undo_data, np.around(cutoff_budget, 1), np.around(cutoff_budgetchange, 1), np.around(cutoff_withinreachchange, 1), np.around(cutoff_efficiency, 1)
