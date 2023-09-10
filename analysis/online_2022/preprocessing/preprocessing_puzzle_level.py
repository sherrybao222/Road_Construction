import pandas as pd
import numpy as np
from glob import glob

# directories
home_dir = '/Users/dbao/My_Drive'+'/road_construction/data/2022_online/'
map_dir = 'active_map/'
data_dir  = 'data/preprocessed'
# home_dir = 'G:\My Drive\\researches\\nyu\\road-construction-local-dk\data_online_2022/'
# map_dir = 'active_map/'
# data_dir  = 'data/preprocessed'
R_out_dir = home_dir+'R_analysis_data/'

flist = glob(home_dir + data_dir + '/preprocess4_sub_*.csv')

# Lists to hold final data
all_data = []
subjectID = 0
# Read files and preprocess
for fname in flist:
    df = pd.read_csv(fname)

    # Replace 'undo' with 1 and 'basic' with 0
    df.replace({'undo': 1, 'basic': 0}, inplace=True)

    grouped = df.groupby('trial_id')

    for _, group in grouped:
        single_data = {
            'subjects': subjectID,
            'puzzleID': group.iloc[0]['map_id'],
            'undo_c': int(group.iloc[0]['condition']),
            'trialID': group.iloc[0]['trial_id'],
            'block': int(group.iloc[0]['trial_id'] / 23),

            'reward': pow(group.iloc[-1]['n_city_all'], 2),
            'numCities': group.iloc[-1]['n_city_all'],
            'mas': group.iloc[0]['mas_all'],
            'nos': group.iloc[0]['n_opt_paths_all'],
            'leftover': group.iloc[-1]['currentBudget'],
            'numUNDO': group['undoIndicator'].sum(),
            'numEnd': group['checkEnd'].sum() - 1,
            'TT': (group.iloc[-1]['time_all'] - group.iloc[0]['time_all']) / 1000,
            'RT1': group.iloc[1]['rt_all'] / 1000,
            'RTsubmit': group.iloc[-1]['rt_all'] / 1000,
            'tortuosity': group.iloc[-1]['tortuosity_all'],
        }

        mas_all_trial = group['mas_all'].values
        errors_trial = (mas_all_trial[1:] - mas_all_trial[:-1])
        single_data['numError'] = (errors_trial < 0).sum()
        single_data['sumSeverityErrors'] = np.abs(errors_trial[errors_trial < 0]).sum()
        single_data['final_sumSeverityErrors'] = mas_all_trial[0] - mas_all_trial[-1]
        single_data['SeverityError1'] = np.abs(errors_trial[0])

        if group.iloc[0]['condition'] == 1:
            undos = group['undoIndicator'].values
            single_data['numFullUndo'] = ((undos[1:] == 1) & (undos[:-1] == 0)).sum()

        RT_later_mask = (group['rt_all'] != -1) & (group['undoIndicator'] != 1) & (group['submit'] != 1)
        RT_later = group[RT_later_mask]['rt_all']
        single_data['RTlater'] = RT_later.mean() / 1000

        all_data.append(single_data)
    print("============")
    print(subjectID)
    subjectID += 1

df_final = pd.DataFrame(all_data)
df_final.to_csv(R_out_dir + 'data.csv', index=False)

