import numpy as np

def df_subset_1state(df, data_puzzle_level, single_condition_data, condition, firstUndo):

    """
    select the submit/undo data [at] terminal state
    """

    index = df.query(condition).index
    df_beforeUndo = df.loc[index-1,:]
    index_end_undo = df_beforeUndo.index[(df_beforeUndo.checkEnd == 1)] 
    state_undo_1undo = df_beforeUndo.loc[index_end_undo, ['subjects','puzzleID',"cumulative_error", 'allMAS', 'currNumCities', "scaled_N_more_bin"]]

    state_undo_1undo = state_undo_1undo.merge(data_puzzle_level.loc[data_puzzle_level.condition==0, ["subjects","puzzleID", "RT1"]], on=["subjects","puzzleID"], how = "left")
    state_undo_1undo = state_undo_1undo.rename(columns = {"RT1":"RT1_basic"})
    state_undo_1undo = state_undo_1undo.merge(single_condition_data[["subjects","puzzleID", "RT1", "action_gap", "avg_numCities_before"]], on=["subjects","puzzleID"], how = "left") # , "RT_branching"
    state_undo_1undo["RT"] = list(df.loc[index_end_undo+1,'RT'])
    state_undo_1undo["RT1_log"] = np.log(state_undo_1undo["RT1"])#+1
    state_undo_1undo["RT1_basic_log"] = np.log(state_undo_1undo["RT1_basic"]+1)
    state_undo_1undo['firstUndo'] = firstUndo
    state_undo_1undo['error'] = state_undo_1undo['cumulative_error'] > 0
    state_undo_1undo['RPE'] = state_undo_1undo['currNumCities'] - state_undo_1undo['avg_numCities_before']
    return state_undo_1undo